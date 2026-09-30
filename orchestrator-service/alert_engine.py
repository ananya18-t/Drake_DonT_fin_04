"""
alert_engine.py

Fuses the three signal sources into reviewer-facing alerts:

1. Graph detectors (Neo4j): insider action -> transfer links, structuring clusters, money cycles.
2. ML triage (Isolation Forest service): per-transaction anomaly percentiles for every
   transaction that a detector surfaced.
3. A composite score = 50% graph-pattern severity + 50% ML evidence, bucketed to a risk level.
   ML evidence blends the highest percentile with the share of transactions the model actually
   flagged, so a lone transfer that is merely unusual cannot carry an alert on its own.

It also builds the investigation payload (Cytoscape graph, timeline, LLM evidence brief) in the
exact shape the frontend's ``src/api/client.js`` parses.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

import httpx

from config import Settings
from graph_queries import ThreatDetectionEngine

logger = logging.getLogger(__name__)

# Severity contributed by each graph pattern (summed, capped at 1.0). A single insider action
# followed by a transfer is common in normal operations; corroborating patterns carry the weight.
PATTERN_WEIGHTS: Dict[str, float] = {
    "INSIDER_OVERRIDE_ALERT": 0.35,
    "INSIDER_MANUAL_UNFREEZE": 0.30,
    "INSIDER_MODIFY_PHONE": 0.25,
    "CIRCULAR_FLOW": 0.40,
    "STRUCTURING": 0.30,
    "DORMANT_ACCOUNT_ACTIVITY": 0.25,
    "MULTI_ACCOUNT_INSIDER": 0.20,
    "PRIVILEGED_EMPLOYEE": 0.05,
}
STANDALONE_BASE = {"STRUCTURING": 0.10, "CIRCULAR_FLOW": 0.10}

BREACH_TAGS: Dict[str, str] = {
    "INSIDER_OVERRIDE_ALERT": "Segregation of Duties",
    "INSIDER_MANUAL_UNFREEZE": "Account Control Bypass",
    "INSIDER_MODIFY_PHONE": "KYC Record Tampering",
    "CIRCULAR_FLOW": "AML Layering",
    "STRUCTURING": "BSA/AML Structuring",
    "DORMANT_ACCOUNT_ACTIVITY": "Dormant Account Misuse",
}

ACTION_CATEGORY = {"OVERRIDE_ALERT": "POLICY_BYPASS", "MANUAL_UNFREEZE": "POLICY_BYPASS"}
ML_BATCH_SENDERS = 40


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _money(amount: float) -> str:
    return f"${amount:,.0f}"


def risk_level(composite: float) -> str:
    if composite >= 80:
        return "CRITICAL"
    if composite >= 65:
        return "HIGH"
    if composite >= 45:
        return "MEDIUM"
    return "LOW"


@dataclass
class MLResult:
    percentile: float
    anomalous: bool
    features: List[str]


@dataclass
class AlertRecord:
    """Internal alert with its full evidence sets; ``to_api`` renders the public shape."""

    alert_id: str
    kind: str
    title: str = ""
    description: str = ""
    emp_id: Optional[str] = None
    employee_name: Optional[str] = None
    employee_privileged: bool = False
    patterns: Set[str] = field(default_factory=set)
    account_ids: Set[str] = field(default_factory=set)
    dormant_account_ids: Set[str] = field(default_factory=set)
    tx_ids: Set[str] = field(default_factory=set)
    log_ids: Set[str] = field(default_factory=set)
    actions: Counter = field(default_factory=Counter)
    total_amount: float = 0.0
    cycles: List[Dict[str, Any]] = field(default_factory=list)
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    graph_score: float = 0.0
    ml_score: Optional[float] = None
    ml_available: bool = True
    composite_score: float = 0.0
    risk_level: str = "LOW"
    status: str = "OPEN"
    assigned_to: Optional[str] = None
    assigned_at: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def touch(self, timestamp: str) -> None:
        if self.first_seen is None or timestamp < self.first_seen:
            self.first_seen = timestamp
        if self.last_seen is None or timestamp > self.last_seen:
            self.last_seen = timestamp

    def to_api(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "case_id": self.alert_id,
            "title": self.title,
            "description": self.description,
            "risk_level": self.risk_level,
            "composite_score": round(self.composite_score, 1),
            "graph_score": round(self.graph_score, 4),
            "ml_score": round(self.ml_score, 4) if self.ml_score is not None else 0.0,
            "status": self.status,
            "emp_id": self.emp_id,
            "employee_name": self.employee_name,
            "created_at": self.last_seen,
            "assigned_to": self.assigned_to,
            "patterns": sorted(self.patterns),
        }


class AlertEngine:
    """Builds and holds the alert book. Neo4j calls are blocking; run them off the event loop."""

    def __init__(self, db: ThreatDetectionEngine, settings: Settings):
        self.db = db
        self.settings = settings
        self.alerts: Dict[str, AlertRecord] = {}
        self.ml_results: Dict[str, MLResult] = {}
        self.transactions_scanned = 0
        self.built_at: Optional[str] = None

    # ================================================================== detection

    def detect(self) -> Dict[str, AlertRecord]:
        """Run all graph detectors and group their hits into alerts (no ML yet)."""
        s = self.settings
        alerts: Dict[str, AlertRecord] = {}

        # 1. Insider action -> transfer, one alert per employee.
        for link in self.db.find_suspicious_employee_account_link(s.INSIDER_WINDOW_HOURS):
            alert_id = f"ALT-INS-{link['emp_id']}"
            alert = alerts.get(alert_id)
            if alert is None:
                alert = alerts[alert_id] = AlertRecord(
                    alert_id=alert_id, kind="INSIDER", emp_id=link["emp_id"],
                    employee_name=link["employee_name"], employee_privileged=bool(link["is_privileged"]),
                )
            alert.patterns.add(f"INSIDER_{link['action_type']}")
            alert.account_ids.add(link["account_id"])
            if link["account_status"] == "DORMANT":
                alert.dormant_account_ids.add(link["account_id"])
            if link["log_id"] not in alert.log_ids:
                alert.log_ids.add(link["log_id"])
                alert.actions[link["action_type"]] += 1
                alert.touch(link["access_time"])
            if link["tx_id"] not in alert.tx_ids:
                alert.tx_ids.add(link["tx_id"])
                alert.total_amount += float(link["amount"])
                alert.touch(link["tx_time"])

        insider_by_account: Dict[str, AlertRecord] = {}
        for alert in alerts.values():
            for account_id in alert.account_ids:
                insider_by_account.setdefault(account_id, alert)

        # 2. Structuring: >= N near-threshold transfers from one sender inside a sliding window.
        for sender, rows in self._structuring_clusters().items():
            target = insider_by_account.get(sender)
            if target is None:
                alert_id = f"ALT-STR-{sender}"
                target = alerts[alert_id] = AlertRecord(alert_id=alert_id, kind="STRUCTURING")
            target.patterns.add("STRUCTURING")
            target.account_ids.add(sender)
            if rows[0]["sender_status"] == "DORMANT":
                target.dormant_account_ids.add(sender)
            for row in rows:
                if row["tx_id"] not in target.tx_ids:
                    target.tx_ids.add(row["tx_id"])
                    target.total_amount += float(row["amount"])
                target.touch(row["tx_time"])

        # 3. Circular flows.
        for index, cycle in enumerate(self.db.find_circular_transfers(s.CYCLE_MAX_HOPS, s.CYCLE_MAX_DAYS)):
            accounts = cycle["account_path"][:-1]
            target = next((insider_by_account[a] for a in accounts if a in insider_by_account), None)
            if target is None:
                alert_id = f"ALT-CYC-{min(accounts)}"
                target = alerts.setdefault(alert_id, AlertRecord(alert_id=alert_id, kind="CYCLE"))
            target.patterns.add("CIRCULAR_FLOW")
            target.cycles.append(cycle)
            target.account_ids.update(accounts)
            for tx_id, amount, timestamp in zip(cycle["tx_path"], cycle["amounts"], cycle["timestamps"]):
                if tx_id not in target.tx_ids:
                    target.tx_ids.add(tx_id)
                    target.total_amount += float(amount)
                target.touch(timestamp)

        for alert in alerts.values():
            if alert.dormant_account_ids:
                alert.patterns.add("DORMANT_ACCOUNT_ACTIVITY")
            if alert.kind == "INSIDER" and len(alert.account_ids) >= 3:
                alert.patterns.add("MULTI_ACCOUNT_INSIDER")
            if alert.employee_privileged:
                alert.patterns.add("PRIVILEGED_EMPLOYEE")
            base = STANDALONE_BASE.get(alert.kind, 0.0) if alert.kind != "INSIDER" else 0.0
            alert.graph_score = min(1.0, base + sum(PATTERN_WEIGHTS.get(p, 0.0) for p in alert.patterns))
            self._describe(alert)
        return alerts

    def _structuring_clusters(self) -> Dict[str, List[Dict[str, Any]]]:
        s = self.settings
        window = timedelta(hours=s.STRUCTURING_WINDOW_HOURS)
        by_sender: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in self.db.find_near_threshold_transfers(s.REPORTING_THRESHOLD, s.STRUCTURING_MARGIN):
            by_sender[row["sender_id"]].append(row)
        clusters: Dict[str, List[Dict[str, Any]]] = {}
        for sender, rows in by_sender.items():
            times = [_ts(r["tx_time"]) for r in rows]
            flagged: Set[int] = set()
            start = 0
            for end in range(len(rows)):
                while times[end] - times[start] > window:
                    start += 1
                if end - start + 1 >= s.STRUCTURING_MIN_COUNT:
                    flagged.update(range(start, end + 1))
            if flagged:
                clusters[sender] = [rows[i] for i in sorted(flagged)]
        return clusters

    @staticmethod
    def _describe(alert: AlertRecord) -> None:
        n_acc, n_tx = len(alert.account_ids), len(alert.tx_ids)
        dormant = f" ({len(alert.dormant_account_ids)} dormant)" if alert.dormant_account_ids else ""
        if alert.kind == "INSIDER":
            action = ", ".join(f"{count}x {name}" for name, count in alert.actions.most_common())
            extras = [label for p, label in (("STRUCTURING", "near-threshold structuring"),
                                             ("CIRCULAR_FLOW", "circular fund flow")) if p in alert.patterns]
            alert.title = (
                f"Insider {alert.employee_name} ({alert.emp_id}) actions preceded "
                f"{n_tx} transfer{'s' if n_tx != 1 else ''} from {n_acc} account{'s' if n_acc != 1 else ''}{dormant}"
            )
            alert.description = (
                f"{action} on customer accounts, followed within the insider window by "
                f"{n_tx} outbound transfers totalling {_money(alert.total_amount)}"
                + (f"; pattern also shows {' and '.join(extras)}" if extras else "") + "."
            )
        elif alert.kind == "STRUCTURING":
            account = next(iter(alert.account_ids))
            alert.title = f"Structuring: {n_tx} near-threshold transfers from {account}"
            alert.description = (
                f"{n_tx} transfers just under the reporting threshold totalling {_money(alert.total_amount)}."
            )
        else:
            path = alert.cycles[0]["account_path"] if alert.cycles else sorted(alert.account_ids)
            alert.title = f"Circular money flow across {n_acc} accounts"
            alert.description = (
                f"Funds moved {' -> '.join(path)} in {n_tx} chronological transfers "
                f"totalling {_money(alert.total_amount)}."
            )

    # ================================================================== ML fusion

    async def score_with_ml(self, alerts: Dict[str, AlertRecord], ml_url: str, context_fetch) -> bool:
        """Score every surfaced transaction with the ML engine. Returns False if ML is down."""
        pattern_txs = set().union(*(a.tx_ids for a in alerts.values())) if alerts else set()
        rows = await context_fetch("get_transactions", sorted(pattern_txs))
        by_sender: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_sender[row["sender_account"]].append(row)
        senders = sorted(by_sender)
        results: Dict[str, MLResult] = {}
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                for start in range(0, len(senders), ML_BATCH_SENDERS):
                    chunk = senders[start:start + ML_BATCH_SENDERS]
                    context = await context_fetch("get_ml_context", chunk)
                    to_score = [tx for sender in chunk for tx in by_sender[sender]]
                    response = await client.post(f"{ml_url}/score-batch", json={
                        "transactions": to_score,
                        "historical_transactions": context["transactions"],
                        "access_logs": context["access_logs"],
                        "accounts": context["accounts"],
                    })
                    response.raise_for_status()
                    for tx, scored in zip(to_score, response.json()["results"]):
                        results[tx["tx_id"]] = MLResult(
                            scored["anomaly_score"], scored["is_anomalous"], scored["top_contributing_features"]
                        )
        except (httpx.HTTPError, KeyError) as exc:
            logger.warning(f"ML engine unavailable ({exc}); alerts use graph score only.")
            for alert in alerts.values():
                alert.ml_available = False
                alert.composite_score = alert.graph_score * 100
                alert.risk_level = risk_level(alert.composite_score)
            return False

        self.ml_results = results
        for alert in alerts.values():
            scored = [results[t] for t in alert.tx_ids if t in results]
            if scored:
                flagged_share = sum(r.anomalous for r in scored) / len(scored)
                alert.ml_score = 0.5 * max(r.percentile for r in scored) + 0.5 * flagged_share
            else:
                alert.ml_score = 0.0
            alert.composite_score = 100 * (0.5 * alert.graph_score + 0.5 * alert.ml_score)
            alert.risk_level = risk_level(alert.composite_score)
        return True

    # ================================================================== investigation

    def _display_accounts(self, alert: AlertRecord, tx_rows: List[Dict[str, Any]]) -> List[str]:
        """Pick the highest-risk accounts so the graph stays readable."""
        if len(alert.account_ids) <= self.settings.MAX_GRAPH_ACCOUNTS:
            return sorted(alert.account_ids)
        best: Dict[str, float] = defaultdict(float)
        for row in tx_rows:
            result = self.ml_results.get(row["tx_id"])
            best[row["sender_account"]] = max(best[row["sender_account"]], result.percentile if result else 0.0)
        ranked = sorted(alert.account_ids, key=lambda a: (-best.get(a, 0.0), a))
        return sorted(ranked[: self.settings.MAX_GRAPH_ACCOUNTS])

    def build_investigation(self, alert: AlertRecord) -> Dict[str, Any]:
        """Graph + timeline + evidence brief for one alert (blocking; run in a thread)."""
        tx_rows = self.db.get_transactions(sorted(alert.tx_ids))
        shown_accounts = set(self._display_accounts(alert, tx_rows))
        cycle_accounts = {a for c in alert.cycles for a in c["account_path"]}
        shown_txs = [
            t for t in tx_rows
            if t["sender_account"] in shown_accounts or t["sender_account"] in cycle_accounts
        ]
        logs = (
            self.db.get_access_logs([alert.emp_id], sorted(shown_accounts)) if alert.emp_id else []
        )
        risky_logs = [l for l in logs if l["log_id"] in alert.log_ids]
        counterparties = {t["receiver_account"] for t in shown_txs}
        all_accounts = sorted(shown_accounts | counterparties | {t["sender_account"] for t in shown_txs})
        account_data = {row["a"]["account_id"]: row["a"] for row in self.db.get_accounts(all_accounts)}

        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        anomalous_senders = {
            t["sender_account"] for t in shown_txs
            if (r := self.ml_results.get(t["tx_id"])) and r.anomalous
        }

        employee: Dict[str, Any] = {}
        if alert.emp_id:
            employee = self.db.get_employees([alert.emp_id])[0]["e"]
            reasons = [f"{count}x {action}" for action, count in alert.actions.most_common()]
            if alert.employee_privileged:
                reasons.append("Privileged access tier")
            nodes.append({
                "id": alert.emp_id, "type": "Employee", "label": employee.get("name", alert.emp_id),
                "anomalous": True, "risk_reasons": reasons, "data": employee,
            })

        for account_id in all_accounts:
            data = account_data.get(account_id, {"account_id": account_id})
            reasons = []
            if account_id in alert.dormant_account_ids:
                reasons.append("Dormant account sending funds")
            if account_id in anomalous_senders:
                reasons.append("Sender of ML-flagged transfers")
            if account_id in cycle_accounts:
                reasons.append("Part of circular flow")
            nodes.append({
                "id": account_id, "type": "Account",
                "label": data.get("customer_name") or account_id,
                "anomalous": account_id in alert.account_ids,
                "risk_reasons": reasons, "data": data,
            })

        for tx in shown_txs:
            result = self.ml_results.get(tx["tx_id"])
            anomalous = bool(result and result.anomalous)
            nodes.append({
                "id": tx["tx_id"], "type": "Transaction", "label": _money(float(tx["amount"])),
                "anomalous": anomalous,
                "risk_reasons": (
                    [f"ML percentile {result.percentile:.2f}"] + [f"driver: {f}" for f in result.features[:3]]
                    if result else []
                ),
                "data": {**tx, "ml_anomaly_score": result.percentile if result else None},
            })
            edges.append({"id": f"{tx['tx_id']}-SENT", "source": tx["sender_account"], "target": tx["tx_id"],
                          "type": "SENT", "amount": tx["amount"], "timestamp": tx["timestamp"],
                          "anomalous": anomalous})
            edges.append({"id": f"{tx['tx_id']}-TO", "source": tx["tx_id"], "target": tx["receiver_account"],
                          "type": "TO", "amount": tx["amount"], "timestamp": tx["timestamp"],
                          "anomalous": anomalous})

        for log in logs:
            edges.append({"id": log["log_id"], "source": log["emp_id"], "target": log["account_id"],
                          "type": "ACCESSED", "action_type": log["action_type"], "timestamp": log["timestamp"],
                          "anomalous": log["log_id"] in alert.log_ids})

        timeline = [
            {
                "event_id": log["log_id"], "timestamp": log["timestamp"],
                "category": ACTION_CATEGORY.get(log["action_type"], "ACCESS"),
                "title": log["action_type"].replace("_", " ").title(),
                "description": f"{alert.employee_name or log['emp_id']} performed {log['action_type']} "
                               f"on {log['account_id']}",
                "emp_id": log["emp_id"], "account_id": log["account_id"], "action_type": log["action_type"],
            }
            for log in logs
        ] + [
            {
                "event_id": tx["tx_id"], "timestamp": tx["timestamp"], "category": "TRANSFER",
                "title": f"{tx['type'].replace('_', ' ').title()} {_money(float(tx['amount']))}",
                "description": f"{tx['sender_account']} -> {tx['receiver_account']}"
                               + (f" (ML percentile {r.percentile:.2f})" if (r := self.ml_results.get(tx["tx_id"])) else ""),
                "account_id": tx["sender_account"], "counterparty_account_id": tx["receiver_account"],
                "amount": tx["amount"], "channel": tx["type"],
            }
            for tx in shown_txs
        ]
        timeline.sort(key=lambda e: e["timestamp"])

        fraud_window = None
        stamps = [e["timestamp"] for e in timeline if e["event_id"] in alert.tx_ids or e["event_id"] in alert.log_ids]
        if stamps:
            fraud_window = {"start": min(stamps), "end": max(stamps)}

        evidence = self._evidence_brief(alert, employee, shown_accounts, risky_logs, shown_txs, timeline, account_data)
        return {
            "graph": {"nodes": nodes, "edges": edges},
            "timeline": timeline,
            "associated_account_ids": sorted(alert.account_ids),
            "fraud_window": fraud_window,
            "evidence": evidence,
            "truncated": len(shown_accounts) < len(alert.account_ids),
        }

    def _evidence_brief(self, alert, employee, shown_accounts, risky_logs, shown_txs, timeline, account_data):
        """Compact, factual input for the LLM and the rule-based fallback."""
        ml_rows = [(t, self.ml_results[t]) for t in alert.tx_ids if t in self.ml_results]
        feature_counts = Counter(f for _, r in ml_rows if r.anomalous for f in r.features[:3])
        findings = []
        if alert.actions:
            findings.append({
                "title": "Insider privileged action",
                "detail": f"{alert.employee_name} ({alert.emp_id}, {employee.get('role', 'employee')}) performed "
                          + ", ".join(f"{c}x {a}" for a, c in alert.actions.most_common())
                          + f" on {len(alert.account_ids)} customer accounts.",
                "evidence_refs": [l["log_id"] for l in risky_logs],
            })
            findings.append({
                "title": "Transfers followed the insider action",
                "detail": f"{len(alert.tx_ids)} transfers totalling {_money(alert.total_amount)} left these "
                          f"accounts within {self.settings.INSIDER_WINDOW_HOURS}h of the action.",
                "evidence_refs": [t["tx_id"] for t in shown_txs],
            })
        if alert.dormant_account_ids:
            findings.append({
                "title": "Dormant accounts reactivated",
                "detail": f"{len(alert.dormant_account_ids)} of the accounts were DORMANT before sending funds.",
                "evidence_refs": sorted(alert.dormant_account_ids),
            })
        if "STRUCTURING" in alert.patterns:
            findings.append({
                "title": "Structuring below reporting threshold",
                "detail": f"Clusters of >= {self.settings.STRUCTURING_MIN_COUNT} transfers between "
                          f"{_money(self.settings.REPORTING_THRESHOLD - self.settings.STRUCTURING_MARGIN)} and "
                          f"{_money(self.settings.REPORTING_THRESHOLD)} within "
                          f"{self.settings.STRUCTURING_WINDOW_HOURS}h.",
                "evidence_refs": [t["tx_id"] for t in shown_txs],
            })
        for cycle in alert.cycles:
            findings.append({
                "title": "Circular fund flow",
                "detail": f"Funds returned to origin via {' -> '.join(cycle['account_path'])}.",
                "evidence_refs": cycle["tx_path"],
            })
        tags = sorted({BREACH_TAGS[p] for p in alert.patterns if p in BREACH_TAGS})
        action = (
            "Escalate to the MLRO, freeze the linked accounts and suspend the employee's privileged access "
            "pending review." if alert.risk_level in ("CRITICAL", "HIGH")
            else "Assign to an L1 analyst for manual review of the linked activity."
        )
        return {
            "alert": {k: v for k, v in alert.to_api().items() if k not in ("assigned_to", "status")},
            "employee": {k: employee.get(k) for k in ("emp_id", "name", "department", "role", "access_tier",
                                                       "is_privileged")} if employee else None,
            "accounts": [
                {k: account_data.get(a, {}).get(k) for k in ("account_id", "customer_name", "status",
                                                             "risk_category", "balance")}
                for a in sorted(shown_accounts)
            ],
            "accounts_note": (
                f"showing {len(shown_accounts)} of {len(alert.account_ids)} linked accounts"
                if len(shown_accounts) < len(alert.account_ids) else None
            ),
            "findings": findings,
            "ml": {
                "scored_count": len(ml_rows),
                "anomalous_count": sum(r.anomalous for _, r in ml_rows),
                "max_percentile": max((r.percentile for _, r in ml_rows), default=0.0),
                "top_features": [f for f, _ in feature_counts.most_common(5)],
                "top_tx_ids": [t for t, r in sorted(ml_rows, key=lambda x: -x[1].percentile)[:5]],
            } if alert.ml_available else {"note": "ML engine unavailable"},
            "breach_tags": tags,
            "recommended_action": action,
            "timeline": timeline,
        }
