"""
graph_queries.py

Parameter-driven Cypher queries over the fraud graph:
(:Employee)-[:ACCESSED {action_type, timestamp}]->(:Account)-[:SENT]->(:Transaction)-[:TO]->(:Account)

Every method returns plain Python data with timestamps converted to ISO-8601 strings, so
callers never see neo4j driver types.
"""

import logging
from typing import Any, Dict, Iterable, List

from neo4j import Driver, GraphDatabase
from neo4j.graph import Node
from neo4j.time import DateTime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RISKY_ACTIONS = ("OVERRIDE_ALERT", "MODIFY_PHONE", "MANUAL_UNFREEZE")


def _plain(value: Any) -> Any:
    """Recursively convert neo4j temporal values to ISO strings."""
    if isinstance(value, DateTime):
        return value.to_native().isoformat()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


class ThreatDetectionEngine:
    """
    Parameter-driven Cypher query engine for detecting financial crime patterns.
    Wraps Neo4j queries into executable, structured Python methods.
    """

    def __init__(self, uri: str, user: str, password: str):
        self.driver: Driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        """Closes the Neo4j driver connection."""
        self.driver.close()
        logger.info("Database connection closed.")

    def _run(self, query: str, **params: Any) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            return [_plain(record.data()) for record in session.run(query, **params)]

    def ping(self) -> None:
        """Raise if the database is unreachable or credentials are wrong."""
        self.driver.verify_connectivity()

    # ------------------------------------------------------------------ detectors

    def find_suspicious_employee_account_link(
        self,
        window_hours: int = 48,
        actions: Iterable[str] = RISKY_ACTIONS,
    ) -> List[Dict[str, Any]]:
        """
        Detects employees who performed high-risk actions (overrides, contact-detail changes,
        manual unfreezes) on an account shortly before that account sent money.
        One row per (access event, transaction).
        """
        query = """
        MATCH (e:Employee)-[acc:ACCESSED]->(a:Account)-[:SENT]->(tx:Transaction)-[:TO]->(r:Account)
        WHERE acc.action_type IN $actions
          AND tx.timestamp >= acc.timestamp
          AND tx.timestamp <= acc.timestamp + duration({hours: $window_hours})
        RETURN e.emp_id AS emp_id,
               e.name AS employee_name,
               e.role AS employee_role,
               e.is_privileged AS is_privileged,
               a.account_id AS account_id,
               a.status AS account_status,
               acc.log_id AS log_id,
               acc.action_type AS action_type,
               acc.timestamp AS access_time,
               tx.tx_id AS tx_id,
               tx.amount AS amount,
               tx.timestamp AS tx_time,
               r.account_id AS receiver_id
        ORDER BY tx.timestamp
        """
        records = self._run(query, window_hours=window_hours, actions=list(actions))
        logger.info(f"Found {len(records)} insider action -> transfer links.")
        return records

    def find_circular_transfers(self, max_hops: int = 4, max_days: int = 7) -> List[Dict[str, Any]]:
        """
        Identifies directed cycles where funds return to the originating account through
        2..max_hops transfers, in chronological order, within max_days. Each cycle is
        returned once (rotations are de-duplicated on the smallest account id).
        """
        query = f"""
        MATCH path = (start:Account)
              ((:Account)-[:SENT]->(t:Transaction)-[:TO]->(:Account)){{2,{int(max_hops)}}}
              (start)
        WHERE all(i IN range(0, size(t) - 2) WHERE t[i].timestamp <= t[i + 1].timestamp)
          AND t[-1].timestamp <= t[0].timestamp + duration({{days: $max_days}})
        WITH t, [n IN nodes(path) WHERE n:Account | n.account_id] AS accounts
        // Simple cycles only: every account appears once (the start repeats at the end).
        WHERE size(reduce(seen = [], x IN accounts | CASE WHEN x IN seen THEN seen ELSE seen + x END))
              = size(accounts) - 1
        RETURN accounts AS account_path,
               [x IN t | x.tx_id] AS tx_path,
               [x IN t | x.amount] AS amounts,
               [x IN t | x.timestamp] AS timestamps
        """
        records = self._run(query, max_days=max_days)
        unique: Dict[frozenset, Dict[str, Any]] = {}
        for record in records:
            unique.setdefault(frozenset(record["tx_path"]), record)
        cycles = list(unique.values())
        logger.info(f"Found {len(cycles)} circular transfer patterns.")
        return cycles

    def find_near_threshold_transfers(self, threshold: float = 10_000.0, margin: float = 1_000.0) -> List[Dict[str, Any]]:
        """
        Returns every transfer in the band (threshold - margin, threshold], ordered per sender,
        so the caller can apply a sliding time window (structuring / smurfing).
        """
        query = """
        MATCH (sender:Account)-[:SENT]->(tx:Transaction)-[:TO]->(receiver:Account)
        WHERE tx.amount > $threshold - $margin AND tx.amount <= $threshold
        RETURN sender.account_id AS sender_id,
               sender.status AS sender_status,
               tx.tx_id AS tx_id,
               tx.amount AS amount,
               tx.timestamp AS tx_time,
               receiver.account_id AS receiver_id
        ORDER BY sender_id, tx_time
        """
        return self._run(query, threshold=threshold, margin=margin)

    # ------------------------------------------------------------------ entity fetches

    def get_ml_context(self, account_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Everything the ML engine needs to score transactions sent by these accounts:
        their full send history, access logs against them, and their status.
        """
        transactions = self._run(
            """
            UNWIND $ids AS aid
            MATCH (a:Account {account_id: aid})-[:SENT]->(tx:Transaction)-[:TO]->(r:Account)
            RETURN tx.tx_id AS tx_id, a.account_id AS sender_account, r.account_id AS receiver_account,
                   tx.amount AS amount, tx.timestamp AS timestamp, tx.type AS type
            """,
            ids=account_ids,
        )
        access_logs = self._run(
            """
            UNWIND $ids AS aid
            MATCH (e:Employee)-[acc:ACCESSED]->(a:Account {account_id: aid})
            RETURN acc.log_id AS log_id, e.emp_id AS emp_id, a.account_id AS account_id,
                   acc.action_type AS action, acc.timestamp AS timestamp
            """,
            ids=account_ids,
        )
        accounts = self._run(
            "UNWIND $ids AS aid MATCH (a:Account {account_id: aid}) "
            "RETURN a.account_id AS account_id, a.status AS status",
            ids=account_ids,
        )
        return {"transactions": transactions, "access_logs": access_logs, "accounts": accounts}

    def get_employees(self, emp_ids: List[str]) -> List[Dict[str, Any]]:
        return self._run(
            "UNWIND $ids AS eid MATCH (e:Employee {emp_id: eid}) RETURN properties(e) AS e",
            ids=emp_ids,
        )

    def get_accounts(self, account_ids: List[str]) -> List[Dict[str, Any]]:
        return self._run(
            "UNWIND $ids AS aid MATCH (a:Account {account_id: aid}) RETURN properties(a) AS a",
            ids=account_ids,
        )

    def get_transactions(self, tx_ids: List[str]) -> List[Dict[str, Any]]:
        return self._run(
            """
            UNWIND $ids AS tid
            MATCH (s:Account)-[:SENT]->(tx:Transaction {tx_id: tid})-[:TO]->(r:Account)
            RETURN tx.tx_id AS tx_id, s.account_id AS sender_account, r.account_id AS receiver_account,
                   tx.amount AS amount, tx.timestamp AS timestamp, tx.type AS type
            """,
            ids=tx_ids,
        )

    def get_access_logs(self, emp_ids: List[str], account_ids: List[str]) -> List[Dict[str, Any]]:
        """Access events by the given employees on the given accounts."""
        return self._run(
            """
            MATCH (e:Employee)-[acc:ACCESSED]->(a:Account)
            WHERE e.emp_id IN $emp_ids AND a.account_id IN $account_ids
            RETURN acc.log_id AS log_id, e.emp_id AS emp_id, a.account_id AS account_id,
                   acc.action_type AS action_type, acc.timestamp AS timestamp
            ORDER BY acc.timestamp
            """,
            emp_ids=emp_ids,
            account_ids=account_ids,
        )

    def list_reviewers(self, exclude_emp_ids: Iterable[str] = ()) -> List[Dict[str, Any]]:
        """Staff in compliance-type departments who can own a case (flagged employees excluded)."""
        return self._run(
            """
            MATCH (e:Employee)
            WHERE (e.department CONTAINS 'Compliance' OR e.department CONTAINS 'Financial Crime'
                   OR e.department CONTAINS 'Risk' OR e.department CONTAINS 'Audit')
              AND NOT e.emp_id IN $exclude
            RETURN e.emp_id AS reviewer_id, e.name AS name, e.role AS role
            ORDER BY e.emp_id
            """,
            exclude=list(exclude_emp_ids),
        )

    def count_transactions(self) -> int:
        return self._run("MATCH (t:Transaction) RETURN count(t) AS n")[0]["n"]

    def freeze_accounts(self, account_ids: List[str]) -> List[str]:
        """Set status FROZEN; returns the ids that exist."""
        rows = self._run(
            "UNWIND $ids AS aid MATCH (a:Account {account_id: aid}) "
            "SET a.status = 'FROZEN' RETURN a.account_id AS account_id",
            ids=account_ids,
        )
        return [row["account_id"] for row in rows]

    def get_subgraph_for_incident(
        self,
        anchor_id: str
    ) -> Dict[str, Any]:
        """
        Extracts the full local neighborhood (up to 2 hops) around an anchor ID
        (emp_id, account_id, or tx_id) and formats it natively for D3/Cytoscape.
        """
        query = """
        MATCH (n)
        WHERE n.emp_id = $anchor_id OR n.account_id = $anchor_id OR n.tx_id = $anchor_id
        MATCH path = (n)-[*0..2]-(m)
        RETURN path
        LIMIT 500
        """

        nodes_dict: Dict[str, Any] = {}
        links_dict: Dict[str, Any] = {}

        def _get_canonical_id(node: Node) -> str:
            """Extracts the domain-specific primary key from a Node."""
            if "Account" in node.labels: return node["account_id"]
            if "Employee" in node.labels: return node["emp_id"]
            if "Transaction" in node.labels: return node["tx_id"]
            return node.element_id

        with self.driver.session() as session:
            result = session.run(query, anchor_id=anchor_id)

            for record in result:
                path = record["path"]
                for node in path.nodes:
                    cid = _get_canonical_id(node)
                    if cid not in nodes_dict:
                        nodes_dict[cid] = {
                            "id": cid,
                            "labels": list(node.labels),
                            "properties": _plain(dict(node)),
                        }
                for rel in path.relationships:
                    rel_id = rel.element_id
                    if rel_id not in links_dict:
                        links_dict[rel_id] = {
                            "id": rel_id,
                            "source": _get_canonical_id(rel.nodes[0]),
                            "target": _get_canonical_id(rel.nodes[1]),
                            "type": rel.type,
                            "properties": _plain(dict(rel)),
                        }

        subgraph = {
            "nodes": list(nodes_dict.values()),
            "links": list(links_dict.values())
        }
        logger.info(f"Extracted subgraph for '{anchor_id}': {len(subgraph['nodes'])} nodes, {len(subgraph['links'])} links.")
        return subgraph


if __name__ == "__main__":
    from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER, settings

    engine = ThreatDetectionEngine(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    try:
        print("\n--- Suspicious Employee Overrides ---")
        insider_links = engine.find_suspicious_employee_account_link(window_hours=settings.INSIDER_WINDOW_HOURS)
        by_emp: Dict[str, int] = {}
        for link in insider_links:
            by_emp[link["emp_id"]] = by_emp.get(link["emp_id"], 0) + 1
        for emp_id, count in sorted(by_emp.items(), key=lambda kv: -kv[1])[:10]:
            print(f"{emp_id}: {count} linked transfers")

        print("\n--- Circular Money Flow Patterns ---")
        for circle in engine.find_circular_transfers(settings.CYCLE_MAX_HOPS, settings.CYCLE_MAX_DAYS):
            print(f"Cycle: {' -> '.join(circle['account_path'])}")

        print("\n--- Near-threshold transfers ---")
        near = engine.find_near_threshold_transfers(settings.REPORTING_THRESHOLD, settings.STRUCTURING_MARGIN)
        print(f"{len(near)} transfers in the structuring band")
    finally:
        engine.close()
