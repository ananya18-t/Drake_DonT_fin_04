"""Reproducible synthetic insider-risk and transaction scenarios.

Each generator returns plain Python dictionaries whose record fields match
the models in :mod:`schemas`. Datetime values remain ``datetime`` objects so
callers can validate records directly; use Pydantic's JSON-mode dump when
emitting JSON.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, TypedDict

from schemas import (
    AccessLogSchema,
    AccountSchema,
    EmployeeSchema,
    GroundTruthLabelSchema,
    TransactionSchema,
)


Record = dict[str, Any]


class ScenarioPayload(TypedDict):
    """Top-level structure shared by all generated scenarios."""

    scenario_id: str
    description: str
    employees: list[Record]
    accounts: list[Record]
    access_logs: list[Record]
    transactions: list[Record]
    ground_truth: list[Record]


TargetType = Literal["TRANSACTION", "EMPLOYEE", "ACCOUNT"]
Label = Literal["SUSPICIOUS", "BENIGN"]

_DEFAULT_START = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)


def _start_time(value: datetime | None) -> datetime:
    """Return a timezone-aware UTC start time, rejecting ambiguous naive times."""
    start = value if value is not None else _DEFAULT_START
    if start.tzinfo is None or start.utcoffset() is None:
        raise ValueError("start_time must be timezone-aware")
    return start.astimezone(timezone.utc)


def _record(model: EmployeeSchema | AccountSchema | AccessLogSchema | TransactionSchema | GroundTruthLabelSchema) -> Record:
    """Validate a record against its canonical model and return a Python dict."""
    return model.model_dump(mode="python")


def _label(target_id: str, target_type: TargetType, label: Label, scenario: str) -> Record:
    """Build and validate one ground-truth record."""
    return _record(
        GroundTruthLabelSchema(
            target_id=target_id,
            target_type=target_type,
            label=label,
            scenario_tag=scenario,
        )
    )


def generate_scenario_a(
    *, seed: int = 7, start_time: datetime | None = None
) -> ScenarioPayload:
    """Generate dormant-account smurfing with an insider override.

    The dormant source account sends four exact ``$9,500`` wires to external
    beneficiaries. All transfers occur within two hours of the insider's
    compliance override. The seed controls beneficiary identifiers.
    """
    start = _start_time(start_time)
    rng = random.Random(seed)
    scenario = "SCENARIO_A_SMURFING_INSIDER_HELP"
    employee_id = "EMP-A-001"
    source_id = "ACC-A-DORMANT-001"
    employee = _record(
        EmployeeSchema(
            emp_id=employee_id,
            name="Jordan Lee",
            department="Financial Crime Operations",
            role="Senior Compliance Analyst",
            access_tier="TIER_3",
            is_privileged=True,
        )
    )
    accounts = [
        _record(
            AccountSchema(
                account_id=source_id,
                customer_name="Northstar Trading LLC",
                balance=72_000.0,
                status="DORMANT",
                risk_category="ELEVATED",
            )
        )
    ]
    beneficiary_ids = [f"EXT-A-{rng.randrange(100_000, 999_999)}" for _ in range(4)]
    for index, beneficiary_id in enumerate(beneficiary_ids, start=1):
        accounts.append(
            _record(
                AccountSchema(
                    account_id=beneficiary_id,
                    customer_name=f"External Beneficiary {index}",
                    balance=2_000.0,
                    status="ACTIVE",
                    risk_category="STANDARD",
                )
            )
        )

    override_time = start
    access_logs = [
        _record(
            AccessLogSchema(
                log_id="LOG-A-001",
                emp_id=employee_id,
                account_id=source_id,
                action="OVERRIDE_ALERT",
                timestamp=override_time,
            )
        )
    ]
    transactions = [
        _record(
            TransactionSchema(
                tx_id=f"TX-A-{index:03d}",
                sender_account=source_id,
                receiver_account=beneficiary_id,
                amount=9_500.0,
                timestamp=start + timedelta(minutes=20 * index),
                type="WIRE",
            )
        )
        for index, beneficiary_id in enumerate(beneficiary_ids, start=1)
    ]
    labels = [
        _label(employee_id, "EMPLOYEE", "SUSPICIOUS", scenario),
        _label(source_id, "ACCOUNT", "SUSPICIOUS", scenario),
        *(
            _label(tx["tx_id"], "TRANSACTION", "SUSPICIOUS", scenario)
            for tx in transactions
        ),
    ]
    return {
        "scenario_id": "A",
        "description": (
            "A privileged employee overrides a compliance alert on a dormant "
            "account, which sends four $9,500 wires to external beneficiaries "
            "within 80 minutes."
        ),
        "employees": [employee],
        "accounts": accounts,
        "access_logs": access_logs,
        "transactions": transactions,
        "ground_truth": labels,
    }


def generate_scenario_b(
    *, seed: int = 11, start_time: datetime | None = None
) -> ScenarioPayload:
    """Generate employee-assisted KYC tampering followed by circular flow.

    The current canonical access-log action enum has no ``KYC_UPDATE`` value,
    so the profile mutation is represented as ``MODIFY_PHONE`` (the supported
    profile-change action). Three transfers then complete A -> B -> C -> A in
    under 48 hours. The seed varies transfer amounts reproducibly.
    """
    start = _start_time(start_time)
    rng = random.Random(seed)
    scenario = "SCENARIO_B_CIRCULAR_MONEY_FLOW"
    employee_id = "EMP-B-001"
    account_ids = ["ACC-B-A", "ACC-B-B", "ACC-B-C"]
    employee = _record(
        EmployeeSchema(
            emp_id=employee_id,
            name="Morgan Patel",
            department="Client Due Diligence",
            role="KYC Operations Specialist",
            access_tier="TIER_2",
            is_privileged=False,
        )
    )
    accounts = [
        _record(
            AccountSchema(
                account_id=account_id,
                customer_name=f"Meridian Holdings {letter}",
                balance=500_000.0,
                status="ACTIVE",
                risk_category="ELEVATED",
            )
        )
        for account_id, letter in zip(account_ids, ("A", "B", "C"), strict=True)
    ]
    access_logs = [
        _record(
            AccessLogSchema(
                log_id="LOG-B-001",
                emp_id=employee_id,
                account_id=account_ids[0],
                action="MODIFY_PHONE",
                timestamp=start,
            )
        )
    ]
    edges = list(zip(account_ids, account_ids[1:] + account_ids[:1], strict=True))
    transactions = [
        _record(
            TransactionSchema(
                tx_id=f"TX-B-{index:03d}",
                sender_account=sender,
                receiver_account=receiver,
                amount=float(rng.randrange(180_000, 260_001, 5_000)),
                timestamp=start + timedelta(hours=16 * index),
                type="INTERNAL_TRANSFER",
            )
        )
        for index, (sender, receiver) in enumerate(edges, start=1)
    ]
    labels = [
        _label(employee_id, "EMPLOYEE", "SUSPICIOUS", scenario),
        *(_label(account_id, "ACCOUNT", "SUSPICIOUS", scenario) for account_id in account_ids),
        *(
            _label(tx["tx_id"], "TRANSACTION", "SUSPICIOUS", scenario)
            for tx in transactions
        ),
    ]
    return {
        "scenario_id": "B",
        "description": (
            "An employee changes account profile data before three transfers "
            "complete a circular A -> B -> C -> A route in 48 hours."
        ),
        "employees": [employee],
        "accounts": accounts,
        "access_logs": access_logs,
        "transactions": transactions,
        "ground_truth": labels,
    }


def generate_scenario_c(
    *, seed: int = 23, start_time: datetime | None = None
) -> ScenarioPayload:
    """Generate a benign high-volume corporate and high-net-worth baseline.

    The baseline has one payroll batch of twelve routine payments and six
    ordinary high-value wires. Amounts and minute offsets vary deterministically
    with ``seed`` so repeated runs with the same inputs are identical.
    """
    start = _start_time(start_time)
    rng = random.Random(seed)
    scenario = "SCENARIO_C_LEGITIMATE_HIGH_VOLUME_BASELINE"
    payroll_employee_id = "EMP-C-PAYROLL"
    wealth_employee_id = "EMP-C-WEALTH"
    employees = [
        _record(
            EmployeeSchema(
                emp_id=payroll_employee_id,
                name="Taylor Kim",
                department="Payroll",
                role="Payroll Manager",
                access_tier="TIER_2",
                is_privileged=True,
            )
        ),
        _record(
            EmployeeSchema(
                emp_id=wealth_employee_id,
                name="Casey Shah",
                department="Private Banking",
                role="Relationship Manager",
                access_tier="TIER_2",
                is_privileged=False,
            )
        ),
    ]

    accounts: list[Record] = [
        _record(
            AccountSchema(
                account_id="ACC-C-CORP",
                customer_name="Cedar Systems Inc.",
                balance=2_500_000.0,
                status="ACTIVE",
                risk_category="LOW",
            )
        )
    ]
    payroll_account_ids = [f"ACC-C-PAY-{index:02d}" for index in range(1, 13)]
    for index, account_id in enumerate(payroll_account_ids, start=1):
        accounts.append(
            _record(
                AccountSchema(
                    account_id=account_id,
                    customer_name=f"Cedar Systems Employee {index:02d}",
                    balance=4_000.0 + float(index * 125),
                    status="ACTIVE",
                    risk_category="STANDARD",
                )
            )
        )
    wealth_account_ids = [f"ACC-C-HNW-{index:02d}" for index in range(1, 7)]
    for index, account_id in enumerate(wealth_account_ids, start=1):
        accounts.append(
            _record(
                AccountSchema(
                    account_id=account_id,
                    customer_name=f"Private Banking Client {index:02d}",
                    balance=1_000_000.0 + float(index * 250_000),
                    status="ACTIVE",
                    risk_category="STANDARD",
                )
            )
        )

    transactions: list[Record] = []
    payroll_start = start + timedelta(hours=8)
    for index, receiver in enumerate(payroll_account_ids, start=1):
        transactions.append(
            _record(
                TransactionSchema(
                    tx_id=f"TX-C-PAY-{index:03d}",
                    sender_account="ACC-C-CORP",
                    receiver_account=receiver,
                    amount=float(rng.randrange(2_500, 7_501, 50)),
                    timestamp=payroll_start + timedelta(minutes=2 * index),
                    type="INTERNAL_TRANSFER",
                )
            )
        )
    wealth_start = start + timedelta(days=1)
    for index in range(6):
        sender = wealth_account_ids[index]
        receiver = wealth_account_ids[(index + 1) % len(wealth_account_ids)]
        transactions.append(
            _record(
                TransactionSchema(
                    tx_id=f"TX-C-HNW-{index + 1:03d}",
                    sender_account=sender,
                    receiver_account=receiver,
                    amount=float(rng.randrange(125_000, 1_250_001, 25_000)),
                    timestamp=wealth_start + timedelta(hours=3 * index),
                    type="WIRE",
                )
            )
        )

    ground_truth = [
        *(
            _label(employee_id, "EMPLOYEE", "BENIGN", scenario)
            for employee_id in (payroll_employee_id, wealth_employee_id)
        ),
        *(
            _label(account["account_id"], "ACCOUNT", "BENIGN", scenario)
            for account in accounts
        ),
        *(
            _label(tx["tx_id"], "TRANSACTION", "BENIGN", scenario)
            for tx in transactions
        ),
    ]
    return {
        "scenario_id": "C",
        "description": (
            "A high-volume but legitimate baseline with a twelve-payment "
            "corporate payroll batch and six routine high-net-worth wires."
        ),
        "employees": employees,
        "accounts": accounts,
        "access_logs": [],
        "transactions": transactions,
        "ground_truth": ground_truth,
    }


def generate_all_scenarios(
    *, seed: int = 0, start_time: datetime | None = None
) -> dict[str, ScenarioPayload]:
    """Generate all scenarios with stable, distinct seeds and a shared start."""
    start = _start_time(start_time)
    return {
        "A": generate_scenario_a(seed=seed + 7, start_time=start),
        "B": generate_scenario_b(seed=seed + 11, start_time=start),
        "C": generate_scenario_c(seed=seed + 23, start_time=start),
    }
