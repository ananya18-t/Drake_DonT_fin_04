"""
Integration tests against the running stack (Neo4j loaded, ML engine on :8001, gateway on :8000).
llama-server is optional: without it, explanations must fall back to the rule-based narrative.

Run from ``orchestrator-service/``:  ``python -m pytest -q test_gateway.py``
"""

from datetime import datetime

import httpx
import pytest

BASE = "http://localhost:8000/api/v1"
RISK_LEVELS = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
CASE_STATUSES = {"OPEN", "ASSIGNED", "ESCALATED", "FROZEN", "CLOSED"}
NODE_TYPES = {"Employee", "Account", "Transaction"}
EDGE_TYPES = {"ACCESSED", "SENT", "TO"}
CATEGORIES = {"ACCESS", "TRANSFER", "POLICY_BYPASS"}


@pytest.fixture(scope="module")
def client():
    try:
        httpx.get("http://localhost:8000/health", timeout=2).raise_for_status()
    except httpx.HTTPError:
        pytest.skip("gateway not running on :8000")
    with httpx.Client(base_url=BASE, timeout=120) as c:
        yield c


@pytest.fixture(scope="module")
def alerts(client):
    response = client.get("/alerts")
    assert response.status_code == 200
    return response.json()


def _iso(value):
    datetime.fromisoformat(value.replace("Z", "+00:00"))
    return True


def test_alert_contract(alerts):
    assert alerts, "no alerts produced"
    for a in alerts:
        assert a["alert_id"] and a["case_id"] and a["title"]
        assert a["risk_level"] in RISK_LEVELS and a["status"] in CASE_STATUSES
        assert 0 <= a["graph_score"] <= 1 and 0 <= a["ml_score"] <= 1 and 0 <= a["composite_score"] <= 100
        assert isinstance(a["patterns"], list) and _iso(a["created_at"])
    scores = [a["composite_score"] for a in alerts]
    assert scores == sorted(scores, reverse=True)


def test_planted_scenarios_are_critical(alerts):
    by_id = {a["alert_id"]: a for a in alerts}
    scenario_a = by_id["ALT-INS-EMP-A-001"]
    assert scenario_a["risk_level"] == "CRITICAL"
    assert {"STRUCTURING", "DORMANT_ACCOUNT_ACTIVITY", "INSIDER_OVERRIDE_ALERT"} <= set(scenario_a["patterns"])
    scenario_b = by_id["ALT-INS-EMP-B-001"]
    assert scenario_b["risk_level"] == "CRITICAL" and "CIRCULAR_FLOW" in scenario_b["patterns"]
    # The two insiders are the top of the queue; background noise stays below CRITICAL.
    assert {a["alert_id"] for a in alerts[:2]} == {"ALT-INS-EMP-A-001", "ALT-INS-EMP-B-001"}
    assert sum(a["risk_level"] == "CRITICAL" for a in alerts) == 2


def test_filters(client):
    critical = client.get("/alerts", params={"risk_level": "critical"}).json()
    assert critical and all(a["risk_level"] == "CRITICAL" for a in critical)
    assert len(client.get("/alerts", params={"limit": 3}).json()) == 3


@pytest.mark.parametrize("alert_id", ["ALT-INS-EMP-A-001", "ALT-INS-EMP-B-001"])
def test_investigation_contract(client, alert_id):
    body = client.get(f"/alerts/{alert_id}/investigation").json()
    assert body["alert"]["alert_id"] == alert_id
    nodes, edges = body["graph"]["nodes"], body["graph"]["edges"]
    ids = {n["id"] for n in nodes}
    assert len(ids) == len(nodes), "duplicate node ids"
    assert {n["type"] for n in nodes} <= NODE_TYPES and "Employee" in {n["type"] for n in nodes}
    for e in edges:
        assert e["type"] in EDGE_TYPES and e["source"] in ids and e["target"] in ids, e
    assert any(e["type"] == "ACCESSED" and e["anomalous"] for e in edges)
    timeline = body["timeline"]
    assert timeline and all(ev["category"] in CATEGORIES and _iso(ev["timestamp"]) for ev in timeline)
    assert [ev["timestamp"] for ev in timeline] == sorted(ev["timestamp"] for ev in timeline)
    explanation = body["explanation"]
    assert explanation["summary"] and explanation["rationale"] and explanation["model"]
    assert body["associated_account_ids"] and body["fraud_window"]["start"] <= body["fraud_window"]["end"]


def test_large_case_graph_is_capped(client):
    body = client.get("/alerts/ALT-INS-EMP-A-001/investigation").json()
    senders = {e["source"] for e in body["graph"]["edges"] if e["type"] == "SENT"}
    assert len(senders) <= 15 < len(body["associated_account_ids"])


def test_unknown_alert_404(client):
    assert client.get("/alerts/ALT-NOPE/investigation").status_code == 404


def test_reviewers(client):
    reviewers = client.get("/reviewers").json()
    assert reviewers and all(r["reviewer_id"] and r["name"] for r in reviewers)
    assert "EMP-A-001" not in {r["reviewer_id"] for r in reviewers}, "flagged insider offered as reviewer"


def test_case_workflow(client, alerts):
    case_id = next(a["alert_id"] for a in reversed(alerts) if a["status"] == "OPEN")
    reviewer = client.get("/reviewers").json()[0]["reviewer_id"]
    assigned = client.post(f"/cases/{case_id}/assign", json={"reviewer_id": reviewer}).json()
    assert assigned["status"] == "ASSIGNED" and assigned["assigned_to"] == reviewer
    escalated = client.post(f"/cases/{case_id}/escalate", json={"note": "pytest"}).json()
    assert escalated["status"] == "ESCALATED"
    listed = {a["alert_id"]: a for a in client.get("/alerts").json()}[case_id]
    assert listed["status"] == "ESCALATED" and listed["assigned_to"] == reviewer
    bad = client.post(f"/cases/{case_id}/freeze", json={"account_ids": ["NOT-LINKED"]})
    assert bad.status_code == 400
    assert client.post(f"/cases/{case_id}/assign", json={}).status_code == 422


def test_dossier_exports(client):
    pdf = client.get("/cases/ALT-INS-EMP-B-001/dossier", params={"format": "pdf"})
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    assert "evidence-dossier-ALT-INS-EMP-B-001.pdf" in pdf.headers["content-disposition"]
    js = client.get("/cases/ALT-INS-EMP-B-001/dossier", params={"format": "json"}).json()
    assert js["case_id"] == "ALT-INS-EMP-B-001" and js["timeline"] and js["explanation"]["summary"]


def test_system_status(client):
    status = client.get("/system/status").json()
    assert status["nodes"]["ml"]["online"] and status["nodes"]["graph"]["online"]
    assert status["metrics"]["transactions_scanned"] == 10_000
    assert status["metrics"]["critical_alerts"] == 2
