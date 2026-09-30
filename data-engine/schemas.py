"""Canonical, strictly validated data contracts for the data engine.

These models are intended to be shared by ingestion, analytics, and API
services. They reject unknown fields and implicit Python-side type coercion;
JSON input can be parsed with :meth:`from_json` and serialized with
:meth:`to_json`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field


class StrictSchema(BaseModel):
    """Base model with strict types and a small JSON serialization API."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        validate_assignment=True,
        frozen=False,
    )

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize this model to a JSON string using Pydantic's JSON encoder."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> Self:
        """Validate a JSON string or byte payload and return a model instance."""
        return cls.model_validate_json(payload)


class EmployeeSchema(StrictSchema):
    """Employee identity and privilege metadata."""

    emp_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    department: str = Field(min_length=1)
    role: str = Field(min_length=1)
    access_tier: str = Field(min_length=1)
    is_privileged: bool


class AccountSchema(StrictSchema):
    """Customer account state and risk metadata."""

    account_id: str = Field(min_length=1)
    customer_name: str = Field(min_length=1)
    balance: float = Field(strict=True, allow_inf_nan=False)
    status: Literal["ACTIVE", "DORMANT", "FROZEN"]
    risk_category: str = Field(min_length=1)


class AccessLogSchema(StrictSchema):
    """A recorded employee action against an account."""

    log_id: str = Field(min_length=1)
    emp_id: str = Field(min_length=1)
    account_id: str = Field(min_length=1)
    action: Literal[
        "VIEW_PROFILE",
        "MODIFY_PHONE",
        "OVERRIDE_ALERT",
        "MANUAL_UNFREEZE",
    ]
    timestamp: datetime


class TransactionSchema(StrictSchema):
    """A money movement between two accounts."""

    tx_id: str = Field(min_length=1)
    sender_account: str = Field(min_length=1)
    receiver_account: str = Field(min_length=1)
    amount: float = Field(strict=True, gt=0, allow_inf_nan=False)
    timestamp: datetime
    type: Literal["WIRE", "INTERNAL_TRANSFER", "ATM"]


class GroundTruthLabelSchema(StrictSchema):
    """A labeled transaction, employee, or account for evaluation/training."""

    target_id: str = Field(min_length=1)
    target_type: Literal["TRANSACTION", "EMPLOYEE", "ACCOUNT"]
    label: Literal["SUSPICIOUS", "BENIGN"]
    scenario_tag: str = Field(min_length=1)
