"""
db_loader.py

Loads the canonical data-engine CSVs into Neo4j:
(:Employee)-[:ACCESSED]->(:Account)-[:SENT]->(:Transaction)-[:TO]->(:Account).
Timestamps are stored as Neo4j ``datetime`` values so time-window queries work.
Re-running is idempotent (MERGE on primary keys).
"""

import csv
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List

from neo4j import GraphDatabase, Driver
from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER, settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class Neo4jDataLoader:
    """
    Production-grade ETL loader for Neo4j.
    Handles schema constraints and batch ingestion of canonical CSV data.
    """
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        self.batch_size = 2000
        
    def close(self) -> None:
        """Closes the Neo4j driver connection."""
        self.driver.close()
        logger.info("Neo4j connection closed.")

    def setup_schema(self) -> None:
        """Creates unique constraints which also serve as indexes for fast lookups."""
        queries = [
            "CREATE CONSTRAINT emp_id_unique IF NOT EXISTS FOR (e:Employee) REQUIRE e.emp_id IS UNIQUE",
            "CREATE CONSTRAINT acc_id_unique IF NOT EXISTS FOR (a:Account) REQUIRE a.account_id IS UNIQUE",
            "CREATE CONSTRAINT tx_id_unique IF NOT EXISTS FOR (t:Transaction) REQUIRE t.tx_id IS UNIQUE",
            "CREATE INDEX tx_timestamp IF NOT EXISTS FOR (t:Transaction) ON (t.timestamp)",
            "CREATE INDEX tx_amount IF NOT EXISTS FOR (t:Transaction) ON (t.amount)",
        ]
        
        with self.driver.session() as session:
            for query in queries:
                session.run(query)
        logger.info("Database constraints and indexes verified.")

    def _read_csv_in_batches(self, file_path: Path) -> Iterator[List[Dict[str, Any]]]:
        """Reads a CSV file and yields lists of dictionaries in defined batch sizes."""
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}. Skipping.")
            return
            
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            batch = []
            for row in reader:
                batch.append(row)
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

    def load_employees(self, csv_path: Path) -> None:
        """Ingests Employee nodes."""
        query = """
        UNWIND $batch AS row
        MERGE (e:Employee {emp_id: row.emp_id})
        SET e.name = row.name,
            e.department = row.department,
            e.role = row.role,
            e.access_tier = row.access_tier,
            e.is_privileged = toLower(row.is_privileged) = 'true'
        """
        self._execute_batch(csv_path, query, "Employees")

    def load_accounts(self, csv_path: Path) -> None:
        """Ingests Account nodes."""
        query = """
        UNWIND $batch AS row
        MERGE (a:Account {account_id: row.account_id})
        SET a.customer_name = row.customer_name,
            a.status = row.status,
            a.balance = toFloat(row.balance),
            a.risk_category = row.risk_category
        """
        self._execute_batch(csv_path, query, "Accounts")

    def load_transactions(self, csv_path: Path) -> None:
        """Ingests Transaction nodes and wire up [:SENT] and [:TO] edges to Accounts."""
        query = """
        UNWIND $batch AS row
        MATCH (sender:Account {account_id: row.sender_account})
        MATCH (receiver:Account {account_id: row.receiver_account})
        MERGE (t:Transaction {tx_id: row.tx_id})
        SET t.amount = toFloat(row.amount),
            t.timestamp = datetime(row.timestamp),
            t.type = row.type
        MERGE (sender)-[:SENT]->(t)
        MERGE (t)-[:TO]->(receiver)
        """
        self._execute_batch(csv_path, query, "Transactions and Transfer Edges")

    def load_access_logs(self, csv_path: Path) -> None:
        """Ingests AccessLogs as [:ACCESSED] edges between Employees and Accounts."""
        query = """
        UNWIND $batch AS row
        MATCH (e:Employee {emp_id: row.emp_id})
        MATCH (a:Account {account_id: row.account_id})
        MERGE (e)-[r:ACCESSED {log_id: row.log_id}]->(a)
        SET r.timestamp = datetime(row.timestamp),
            r.action_type = row.action
        """
        self._execute_batch(csv_path, query, "Access Log Edges")

    def _execute_batch(self, csv_path: Path, query: str, entity_name: str) -> None:
        """Helper method to execute UNWIND queries in batches."""
        total_processed = 0
        with self.driver.session() as session:
            for batch in self._read_csv_in_batches(csv_path):
                session.run(query, batch=batch)
                total_processed += len(batch)
        logger.info(f"Ingested {total_processed} records for {entity_name}.")

    def print_ingestion_stats(self) -> None:
        """Queries the database to print final entity counts."""
        queries = {
            "Employee Nodes": "MATCH (n:Employee) RETURN count(n) AS count",
            "Account Nodes": "MATCH (n:Account) RETURN count(n) AS count",
            "Transaction Nodes": "MATCH (n:Transaction) RETURN count(n) AS count",
            "ACCESSED Edges": "MATCH ()-[r:ACCESSED]->() RETURN count(r) AS count",
            "SENT/TO Edges": "MATCH ()-[r:SENT|TO]->() RETURN count(r) AS count"
        }
        
        print("\n--- Final Ingestion Statistics ---")
        with self.driver.session() as session:
            for label, query in queries.items():
                result = session.run(query).single()
                print(f"{label}: {result['count']}")
        print("----------------------------------\n")


if __name__ == "__main__":
    DATA_DIR = Path(settings.DATA_DIR)

    loader = Neo4jDataLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        logger.info("Starting Graph Data Ingestion...")
        
        # 1. Initialize constraints (critical before loading data)
        loader.setup_schema()
        
        # 2. Load standalone nodes
        loader.load_employees(DATA_DIR / "employees.csv")
        loader.load_accounts(DATA_DIR / "accounts.csv")
        
        # 3. Load connected components (requires nodes to exist)
        loader.load_transactions(DATA_DIR / "transactions.csv")
        loader.load_access_logs(DATA_DIR / "access_logs.csv")
        
        # 4. Verify output
        loader.print_ingestion_stats()
        
    except Exception as e:
        logger.error(f"ETL Pipeline Failed: {e}", exc_info=True)
        raise SystemExit(1)
    finally:
        loader.close()