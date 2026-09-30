"""Generate a reproducible, schema-validated financial crime dataset.

The output contains exactly 100 employees, 1,000 accounts, 10,000 transactions,
and 5,000 access logs. Transaction ground truth is balanced to an exact 3%
suspicious rate (300 of 10,000 transactions); insider-assisted smurfing and
circular-flow scenarios are embedded in otherwise benign background data.

Run from any directory with ``python data-engine/generator.py``. Faker, NumPy,
and Pydantic v2 must be installed. CSV files are written under
``data-engine/data/`` by default.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from faker import Faker
from pydantic import BaseModel

from scenarios import generate_scenario_a, generate_scenario_b, generate_scenario_c
from schemas import (
    AccessLogSchema,
    AccountSchema,
    EmployeeSchema,
    GroundTruthLabelSchema,
    TransactionSchema,
)


Record = dict[str, Any]
EMPLOYEE_COUNT = 100
ACCOUNT_COUNT = 1_000
TRANSACTION_COUNT = 10_000
ACCESS_LOG_COUNT = 5_000
ANOMALY_RATE = 0.03
ANOMALOUS_TRANSACTION_COUNT = int(TRANSACTION_COUNT * ANOMALY_RATE)
SCENARIO_A_BATCHES = 72
SCENARIO_B_CYCLES = 4


def _as_record(model: BaseModel) -> Record:
    """Return a model-validated record as a Python dictionary."""
    return model.model_dump(mode="python")


def _label(
    target_id: str,
    target_type: Literal["TRANSACTION", "EMPLOYEE", "ACCOUNT"],
    value: Literal["SUSPICIOUS", "BENIGN"],
    scenario_tag: str,
) -> Record:
    """Build a validated ground-truth row."""
    return _as_record(
        GroundTruthLabelSchema(
            target_id=target_id,
            target_type=target_type,
            label=value,
            scenario_tag=scenario_tag,
        )
    )


def _random_time(rng: np.random.Generator, start: datetime, days: int = 365) -> datetime:
    """Sample a UTC timestamp within the configured synthetic year."""
    seconds = int(rng.integers(0, days * 24 * 60 * 60))
    return start + timedelta(seconds=seconds)


def _scenario_tag(
    scenario: dict[str, Any], target_type: str, target_id: str, default: str
) -> str:
    """Look up a scenario label tag for an entity, or return its default tag."""
    for row in scenario["ground_truth"]:
        if row["target_type"] == target_type and row["target_id"] == target_id:
            return str(row["scenario_tag"])
    return default


def _make_scenario_data(
    faker: Faker, rng: np.random.Generator, start: datetime
) -> tuple[list[Record], list[Record], list[Record], list[Record], set[str], set[str], dict[str, str]]:
    """Create injected scenario records and track their ground-truth metadata."""
    scenario_a = generate_scenario_a(seed=104, start_time=start)
    scenario_b = generate_scenario_b(seed=205, start_time=start)
    scenario_c = generate_scenario_c(seed=306, start_time=start)

    employees: list[Record] = []
    accounts: list[Record] = []
    access_logs: list[Record] = []
    transactions: list[Record] = []
    suspicious_employees: set[str] = set()
    suspicious_accounts: set[str] = set()
    entity_tags: dict[str, str] = {}

    for scenario in (scenario_a, scenario_b, scenario_c):
        default_tag = (
            "SCENARIO_A_SMURFING_INSIDER_HELP"
            if scenario is scenario_a
            else "SCENARIO_B_CIRCULAR_MONEY_FLOW"
            if scenario is scenario_b
            else "SCENARIO_C_LEGITIMATE_HIGH_VOLUME_BASELINE"
        )
        employees.extend(scenario["employees"])
        accounts.extend(scenario["accounts"])
        access_logs.extend(scenario["access_logs"])
        transactions.extend(scenario["transactions"])
        for employee in scenario["employees"]:
            employee_id = str(employee["emp_id"])
            entity_tags[f"EMPLOYEE:{employee_id}"] = _scenario_tag(
                scenario, "EMPLOYEE", employee_id, default_tag
            )
            if scenario is not scenario_c:
                suspicious_employees.add(employee_id)
        for account in scenario["accounts"]:
            account_id = str(account["account_id"])
            entity_tags[f"ACCOUNT:{account_id}"] = _scenario_tag(
                scenario, "ACCOUNT", account_id, default_tag
            )
            if scenario is not scenario_c:
                suspicious_accounts.add(account_id)

    # Reuse the concrete A and B patterns from scenarios.py for the first
    # episode/cycle. Additional distinct episodes retain their same shapes.
    a_employee_id = str(scenario_a["employees"][0]["emp_id"])
    b_employee_id = str(scenario_b["employees"][0]["emp_id"])
    for batch_index in range(2, SCENARIO_A_BATCHES + 1):
        event_time = start + timedelta(days=(batch_index - 1) * 3)
        source_id = f"ACC-A-DORMANT-{batch_index:03d}"
        accounts.append(
            _as_record(
                AccountSchema(
                    account_id=source_id,
                    customer_name=faker.company(),
                    balance=72_000.0,
                    status="DORMANT",
                    risk_category="ELEVATED",
                )
            )
        )
        access_logs.append(
            _as_record(
                AccessLogSchema(
                    log_id=f"LOG-A-{batch_index:03d}",
                    emp_id=a_employee_id,
                    account_id=source_id,
                    action="OVERRIDE_ALERT",
                    timestamp=event_time,
                )
            )
        )
        for transfer_index in range(1, 5):
            receiver_id = f"EXT-A-{batch_index:03d}-{transfer_index}"
            accounts.append(
                _as_record(
                    AccountSchema(
                        account_id=receiver_id,
                        customer_name=f"External Beneficiary {batch_index}-{transfer_index}",
                        balance=2_000.0,
                        status="ACTIVE",
                        risk_category="STANDARD",
                    )
                )
            )
            transactions.append(
                _as_record(
                    TransactionSchema(
                        tx_id=f"TX-A-{batch_index:03d}-{transfer_index}",
                        sender_account=source_id,
                        receiver_account=receiver_id,
                        amount=9_500.0,
                        timestamp=event_time + timedelta(minutes=20 * transfer_index),
                        type="WIRE",
                    )
                )
            )
            entity_tags[f"TRANSACTION:TX-A-{batch_index:03d}-{transfer_index}"] = (
                "SCENARIO_A_SMURFING_INSIDER_HELP"
            )
        suspicious_accounts.add(source_id)
        suspicious_accounts.update(
            f"EXT-A-{batch_index:03d}-{transfer_index}" for transfer_index in range(1, 5)
        )
        entity_tags[f"ACCOUNT:{source_id}"] = "SCENARIO_A_SMURFING_INSIDER_HELP"
        for transfer_index in range(1, 5):
            entity_tags[f"ACCOUNT:EXT-A-{batch_index:03d}-{transfer_index}"] = (
                "SCENARIO_A_SMURFING_INSIDER_HELP"
            )

    for cycle_index in range(2, SCENARIO_B_CYCLES + 1):
        event_time = start + timedelta(days=(cycle_index - 1) * 21)
        cycle_accounts = [f"ACC-B-{cycle_index:02d}-{letter}" for letter in "ABC"]
        for letter, account_id in zip("ABC", cycle_accounts, strict=True):
            accounts.append(
                _as_record(
                    AccountSchema(
                        account_id=account_id,
                        customer_name=f"Meridian Holdings {cycle_index}-{letter}",
                        balance=500_000.0,
                        status="ACTIVE",
                        risk_category="ELEVATED",
                    )
                )
            )
            suspicious_accounts.add(account_id)
            entity_tags[f"ACCOUNT:{account_id}"] = "SCENARIO_B_CIRCULAR_MONEY_FLOW"
        access_logs.append(
            _as_record(
                AccessLogSchema(
                    log_id=f"LOG-B-{cycle_index:02d}",
                    emp_id=b_employee_id,
                    account_id=cycle_accounts[0],
                    action="MODIFY_PHONE",
                    timestamp=event_time,
                )
            )
        )
        for edge_index, (sender, receiver) in enumerate(
            zip(cycle_accounts, cycle_accounts[1:] + cycle_accounts[:1], strict=True),
            start=1,
        ):
            tx_id = f"TX-B-{cycle_index:02d}-{edge_index}"
            transactions.append(
                _as_record(
                    TransactionSchema(
                        tx_id=tx_id,
                        sender_account=sender,
                        receiver_account=receiver,
                        amount=float(rng.integers(180_000, 260_001) // 5_000 * 5_000),
                        timestamp=event_time + timedelta(hours=16 * edge_index),
                        type="INTERNAL_TRANSFER",
                    )
                )
            )
            entity_tags[f"TRANSACTION:{tx_id}"] = "SCENARIO_B_CIRCULAR_MONEY_FLOW"
        suspicious_accounts.update(cycle_accounts)

    # Ensure the first A/B exemplars are marked with their scenario labels.
    return (
        employees,
        accounts,
        access_logs,
        transactions,
        suspicious_employees,
        suspicious_accounts,
        entity_tags,
    )


def _write_csv(path: Path, rows: Sequence[Record]) -> None:
    """Write a homogeneous list of records as a UTF-8 CSV file."""
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            normalized = {
                key: value.isoformat() if isinstance(value, datetime) else value
                for key, value in row.items()
            }
            writer.writerow(normalized)


def generate_dataset(
    output_dir: Path | str | None = None,
    *,
    seed: int = 42,
    start_time: datetime | None = None,
) -> dict[str, Any]:
    """Generate, validate, export, and summarize the complete dataset.

    Args:
        output_dir: Destination directory. Defaults to ``data-engine/data``.
        seed: Seed used for Faker and NumPy random number generation.
        start_time: Start of the generated timeline; must be timezone-aware.

    Returns:
        A summary dictionary with row counts, transaction class balance, and
        suspicious transaction counts by scenario.

    Raises:
        ValueError: If ``start_time`` is naive or generated counts violate the
            requested data contract.
    """
    seed = int(seed)
    faker = Faker("en_US")
    faker.seed_instance(seed)
    rng = np.random.default_rng(seed)
    start = start_time or datetime(2026, 1, 1, tzinfo=timezone.utc)
    if start.tzinfo is None or start.utcoffset() is None:
        raise ValueError("start_time must be timezone-aware")
    start = start.astimezone(timezone.utc)

    (
        employees,
        accounts,
        access_logs,
        transactions,
        suspicious_employees,
        suspicious_accounts,
        entity_tags,
    ) = _make_scenario_data(faker, rng, start)

    # Random benign background fills the exact entity and event totals.
    existing_employee_ids = {str(row["emp_id"]) for row in employees}
    while len(employees) < EMPLOYEE_COUNT:
        employee_id = f"EMP-N-{len(employees) + 1:03d}"
        if employee_id in existing_employee_ids:
            continue
        employees.append(
            _as_record(
                EmployeeSchema(
                    emp_id=employee_id,
                    name=faker.name(),
                    department=str(rng.choice(["Retail Banking", "Operations", "Finance", "Technology", "Compliance"])),
                    role=str(rng.choice(["Analyst", "Associate", "Manager", "Specialist"])),
                    access_tier=str(rng.choice(["TIER_1", "TIER_2", "TIER_3"], p=[0.60, 0.32, 0.08])),
                    is_privileged=bool(rng.random() < 0.12),
                )
            )
        )
        existing_employee_ids.add(employee_id)

    existing_account_ids = {str(row["account_id"]) for row in accounts}
    while len(accounts) < ACCOUNT_COUNT:
        account_id = f"ACC-N-{len(accounts) + 1:04d}"
        if account_id in existing_account_ids:
            continue
        status = str(rng.choice(["ACTIVE", "DORMANT", "FROZEN"], p=[0.88, 0.10, 0.02]))
        accounts.append(
            _as_record(
                AccountSchema(
                    account_id=account_id,
                    customer_name=faker.company(),
                    balance=round(float(rng.lognormal(mean=9.2, sigma=1.0)), 2),
                    status=status,
                    risk_category=str(rng.choice(["LOW", "STANDARD", "ELEVATED", "HIGH"], p=[0.48, 0.34, 0.15, 0.03])),
                )
            )
        )
        existing_account_ids.add(account_id)

    employee_ids = [str(row["emp_id"]) for row in employees]
    account_by_id = {str(row["account_id"]): row for row in accounts}
    active_account_ids = [
        account_id for account_id, row in account_by_id.items() if row["status"] == "ACTIVE"
    ]
    if len(active_account_ids) < 2:
        raise ValueError("At least two ACTIVE accounts are required for background transactions")

    # The scenario templates contribute 7 suspicious transactions and 18
    # benign Scenario C transactions. Additional A/B episodes bring suspicious
    # transactions to exactly 300, leaving the rest as background benign rows.
    suspicious_tx_ids: set[str] = set()
    for row in transactions:
        transaction_id = str(row["tx_id"])
        scenario_name = transaction_id.split("-")[1]
        if scenario_name in {"A", "B"}:
            suspicious_tx_ids.add(transaction_id)
            entity_tags[f"TRANSACTION:{transaction_id}"] = (
                "SCENARIO_A_SMURFING_INSIDER_HELP"
                if scenario_name == "A"
                else "SCENARIO_B_CIRCULAR_MONEY_FLOW"
            )
        else:
            entity_tags[f"TRANSACTION:{transaction_id}"] = (
                "SCENARIO_C_LEGITIMATE_HIGH_VOLUME_BASELINE"
            )

    normal_transaction_count = TRANSACTION_COUNT - ANOMALOUS_TRANSACTION_COUNT - 18
    for normal_start_number in range(1, normal_transaction_count + 1):
        sender_index, receiver_index = rng.choice(len(active_account_ids), size=2, replace=False)
        tx_id = f"TX-N-{normal_start_number:05d}"
        normal_start_number += 1
        amount = float(np.clip(rng.lognormal(mean=8.4, sigma=1.05), 25.0, 250_000.0))
        transactions.append(
            _as_record(
                TransactionSchema(
                    tx_id=tx_id,
                    sender_account=active_account_ids[int(sender_index)],
                    receiver_account=active_account_ids[int(receiver_index)],
                    amount=round(amount, 2),
                    timestamp=_random_time(rng, start),
                    type=str(rng.choice(["WIRE", "INTERNAL_TRANSFER", "ATM"], p=[0.22, 0.70, 0.08])),
                )
            )
        )
        entity_tags[f"TRANSACTION:{tx_id}"] = "BACKGROUND_BENIGN"

    # Use scenario access events as the insider signal; synthesize the balance
    # as normal profile views and routine account maintenance.
    next_log_number = 1
    actions = np.array(["VIEW_PROFILE", "MODIFY_PHONE", "OVERRIDE_ALERT", "MANUAL_UNFREEZE"])
    action_weights = np.array([0.86, 0.11, 0.02, 0.01])
    while len(access_logs) < ACCESS_LOG_COUNT:
        log_id = f"LOG-N-{next_log_number:05d}"
        next_log_number += 1
        access_logs.append(
            _as_record(
                AccessLogSchema(
                    log_id=log_id,
                    emp_id=str(rng.choice(employee_ids)),
                    account_id=str(rng.choice(list(account_by_id))),
                    action=str(rng.choice(actions, p=action_weights)),
                    timestamp=_random_time(rng, start),
                )
            )
        )

    # Enforce the fixed totals and referential integrity before writing.
    if (len(employees), len(accounts), len(transactions), len(access_logs)) != (
        EMPLOYEE_COUNT,
        ACCOUNT_COUNT,
        TRANSACTION_COUNT,
        ACCESS_LOG_COUNT,
    ):
        raise ValueError("Generated row counts do not match the configured contract")
    if len({row["emp_id"] for row in employees}) != EMPLOYEE_COUNT:
        raise ValueError("Employee identifiers are not unique")
    if len({row["account_id"] for row in accounts}) != ACCOUNT_COUNT:
        raise ValueError("Account identifiers are not unique")
    if len({row["tx_id"] for row in transactions}) != TRANSACTION_COUNT:
        raise ValueError("Transaction identifiers are not unique")
    if len({row["log_id"] for row in access_logs}) != ACCESS_LOG_COUNT:
        raise ValueError("Access-log identifiers are not unique")
    known_account_ids = {str(row["account_id"]) for row in accounts}
    known_employee_ids = {str(row["emp_id"]) for row in employees}
    for row in transactions:
        if row["sender_account"] not in known_account_ids or row["receiver_account"] not in known_account_ids:
            raise ValueError(f"Transaction {row['tx_id']} references an unknown account")
    for row in access_logs:
        if row["emp_id"] not in known_employee_ids or row["account_id"] not in known_account_ids:
            raise ValueError(f"Access log {row['log_id']} references an unknown entity")

    # Validate the full exported data again through the canonical contracts.
    employees = [EmployeeSchema.model_validate(row).model_dump(mode="python") for row in employees]
    accounts = [AccountSchema.model_validate(row).model_dump(mode="python") for row in accounts]
    transactions = [TransactionSchema.model_validate(row).model_dump(mode="python") for row in transactions]
    access_logs = [AccessLogSchema.model_validate(row).model_dump(mode="python") for row in access_logs]

    ground_truth: list[Record] = []
    for employee in employees:
        employee_id = str(employee["emp_id"])
        ground_truth.append(
            _label(
                employee_id,
                "EMPLOYEE",
                "SUSPICIOUS" if employee_id in suspicious_employees else "BENIGN",
                entity_tags.get(f"EMPLOYEE:{employee_id}", "BACKGROUND_BENIGN"),
            )
        )
    for account in accounts:
        account_id = str(account["account_id"])
        ground_truth.append(
            _label(
                account_id,
                "ACCOUNT",
                "SUSPICIOUS" if account_id in suspicious_accounts else "BENIGN",
                entity_tags.get(f"ACCOUNT:{account_id}", "BACKGROUND_BENIGN"),
            )
        )
    for transaction in transactions:
        transaction_id = str(transaction["tx_id"])
        suspicious = transaction_id in suspicious_tx_ids or transaction_id.startswith(("TX-A-", "TX-B-"))
        ground_truth.append(
            _label(
                transaction_id,
                "TRANSACTION",
                "SUSPICIOUS" if suspicious else "BENIGN",
                entity_tags.get(
                    f"TRANSACTION:{transaction_id}",
                    "BACKGROUND_BENIGN",
                ),
            )
        )

    tx_labels = [row for row in ground_truth if row["target_type"] == "TRANSACTION"]
    suspicious_labels = [row for row in tx_labels if row["label"] == "SUSPICIOUS"]
    if len(suspicious_labels) != ANOMALOUS_TRANSACTION_COUNT:
        raise ValueError(
            f"Expected {ANOMALOUS_TRANSACTION_COUNT} suspicious transactions; "
            f"generated {len(suspicious_labels)}"
        )

    destination = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent / "data"
    _write_csv(destination / "employees.csv", employees)
    _write_csv(destination / "accounts.csv", accounts)
    _write_csv(destination / "transactions.csv", transactions)
    _write_csv(destination / "access_logs.csv", access_logs)
    _write_csv(destination / "ground_truth_labels.csv", ground_truth)

    anomaly_distribution = Counter(
        str(row["scenario_tag"])
        for row in suspicious_labels
    )
    benign_count = len(tx_labels) - len(suspicious_labels)
    summary: dict[str, Any] = {
        "employees": len(employees),
        "accounts": len(accounts),
        "transactions": len(transactions),
        "access_logs": len(access_logs),
        "ground_truth_labels": len(ground_truth),
        "suspicious_transactions": len(suspicious_labels),
        "benign_transactions": benign_count,
        "transaction_anomaly_rate": len(suspicious_labels) / len(tx_labels),
        "anomaly_distribution": dict(sorted(anomaly_distribution.items())),
        "class_balance": {
            "SUSPICIOUS": len(suspicious_labels),
            "BENIGN": benign_count,
        },
        "output_dir": str(destination.resolve()),
    }
    _print_summary(summary)
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    """Print concise row-count, anomaly, and transaction class metrics."""
    print("Dataset generation complete")
    print("Total counts:")
    for name in ("employees", "accounts", "transactions", "access_logs", "ground_truth_labels"):
        print(f"  {name.replace('_', ' ').title():<22} {summary[name]:>6,}")
    print(f"Transaction anomaly rate: {summary['transaction_anomaly_rate']:.2%}")
    print("Transaction class balance:")
    for label, count in summary["class_balance"].items():
        fraction = count / summary["transactions"]
        print(f"  {label:<12} {count:>6,} ({fraction:.2%})")
    print("Suspicious transaction distribution:")
    for scenario, count in summary["anomaly_distribution"].items():
        print(f"  {scenario}: {count:,}")
    print(f"CSV output: {summary['output_dir']}")


def main() -> None:
    """Parse command-line arguments and run dataset generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None, help="CSV output directory")
    parser.add_argument("--seed", type=int, default=42, help="Faker and NumPy seed")
    args = parser.parse_args()
    generate_dataset(args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
