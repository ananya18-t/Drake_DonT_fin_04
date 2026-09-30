"""FastAPI inference service for transaction anomaly scoring.

Run with ``python service.py`` on port 8001. Set ``ML_BIND_HOST`` to the
machine's Tailscale address to restrict the listener to that interface.
``anomaly_score`` is an empirical training-score percentile, not a fraud
probability. Feature contributions are local counterfactual score changes.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, Request
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from feature_extraction import FEATURE_NAMES, extract_features


DEFAULT_ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "model.pkl"
MAX_CONTEXT_ROWS = 10_000


class TransactionRecord(BaseModel):
    """Canonical transaction fields needed to score an event.

    Accepts both the data-engine names (``sender_account``) and the internal
    names (``sender_account_id``).
    """

    model_config = ConfigDict(populate_by_name=True)

    tx_id: str = Field(min_length=1)
    sender_account_id: str = Field(
        min_length=1, validation_alias=AliasChoices("sender_account_id", "sender_account")
    )
    amount: float
    timestamp: datetime
    receiver_account_id: str | None = Field(
        default=None, validation_alias=AliasChoices("receiver_account_id", "receiver_account")
    )
    type: str | None = None
    channel: str | None = None


class AccessRecord(BaseModel):
    """Employee action used for sender-account correlation."""

    model_config = ConfigDict(populate_by_name=True)

    account_id: str
    action_type: str = Field(validation_alias=AliasChoices("action_type", "action"))
    timestamp: datetime
    log_id: str | None = None
    emp_id: str | None = None


class AccountRecord(BaseModel):
    """Account status used for dormant-account features."""

    account_id: str
    status: str


class ScoreRequest(BaseModel):
    """Current transaction and as-of historical context from the orchestrator."""

    transaction: TransactionRecord
    historical_transactions: list[TransactionRecord] = Field(default_factory=list, max_length=MAX_CONTEXT_ROWS)
    access_logs: list[AccessRecord] = Field(default_factory=list, max_length=MAX_CONTEXT_ROWS)
    accounts: list[AccountRecord] = Field(default_factory=list, max_length=MAX_CONTEXT_ROWS)


class ScoreResponse(BaseModel):
    """Investigator-facing anomaly signals; the score is a percentile."""

    tx_id: str
    anomaly_score: float = Field(ge=0.0, le=1.0)
    is_anomalous: bool
    top_contributing_features: list[str]


class BatchScoreRequest(BaseModel):
    """Several transactions scored against one shared as-of context.

    ``transactions`` are the rows to score; ``historical_transactions`` may
    include them or not. Each row only sees context at or before its own
    timestamp, so results match individual ``/score-transaction`` calls.
    """

    transactions: list[TransactionRecord] = Field(min_length=1, max_length=MAX_CONTEXT_ROWS)
    historical_transactions: list[TransactionRecord] = Field(default_factory=list, max_length=MAX_CONTEXT_ROWS)
    access_logs: list[AccessRecord] = Field(default_factory=list, max_length=MAX_CONTEXT_ROWS)
    accounts: list[AccountRecord] = Field(default_factory=list, max_length=MAX_CONTEXT_ROWS)


class BatchScoreResponse(BaseModel):
    """Scores in the same order as the request's ``transactions``."""

    results: list[ScoreResponse]


def _as_utc(value: datetime) -> datetime:
    """Treat timezone-free client timestamps as UTC."""
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _gpu_metadata() -> dict[str, Any]:
    """Probe NVIDIA hardware without making GPU availability a service dependency."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"cuda_gpu_detected": False, "gpu_devices": [], "gpu_probe": "nvidia-smi unavailable"}
    try:
        output = subprocess.run(
            [executable, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, check=True,
        )
        devices = [line.strip() for line in output.stdout.splitlines() if line.strip()]
        return {"cuda_gpu_detected": bool(devices), "gpu_devices": devices, "gpu_probe": "nvidia-smi"}
    except (OSError, subprocess.SubprocessError):
        return {"cuda_gpu_detected": False, "gpu_devices": [], "gpu_probe": "nvidia-smi failed"}


def _load_artifact(path: Path) -> dict[str, Any]:
    """Validate the trusted local model bundle once during application startup."""
    if not path.is_file():
        raise FileNotFoundError(f"model artifact not found: {path}; run train.py first")
    bundle = joblib.load(path)
    required = {"model", "scaler", "feature_names", "training_score_reference"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError("model artifact is incomplete; retrain with the current train.py")
    if tuple(bundle["feature_names"]) != FEATURE_NAMES:
        raise ValueError("model feature registry differs from feature_extraction.py; retrain")
    reference = np.asarray(bundle["training_score_reference"], dtype=np.float64)
    if reference.ndim != 1 or reference.size == 0 or not np.isfinite(reference).all():
        raise ValueError("model artifact has invalid training_score_reference")
    if np.any(np.diff(reference) < 0):
        raise ValueError("training_score_reference must be sorted ascending")
    bundle["training_score_reference"] = reference
    return bundle


def _score(payload: ScoreRequest, bundle: dict[str, Any]) -> ScoreResponse:
    """Build as-of features, score once, and rank local score sensitivities."""
    current = payload.transaction
    current_time = _as_utc(current.timestamp)
    earliest = current_time.timestamp() - 24 * 3600
    history = [
        row.model_dump(mode="json")
        for row in payload.historical_transactions
        if row.tx_id != current.tx_id
        and row.sender_account_id == current.sender_account_id
        and earliest <= _as_utc(row.timestamp).timestamp() <= current_time.timestamp()
    ]
    logs = [
        row.model_dump(mode="json")
        for row in payload.access_logs
        if row.account_id == current.sender_account_id
        and _as_utc(row.timestamp) <= current_time
    ]
    accounts = [
        row.model_dump(mode="json") for row in payload.accounts
        if row.account_id == current.sender_account_id
    ]
    X, names = extract_features(
        [*history, current.model_dump(mode="json")], access_logs=logs, accounts=accounts
    )
    if tuple(names) != tuple(bundle["feature_names"]):
        raise RuntimeError("inference features do not match the trained model")
    scaled = bundle["scaler"].transform(X[-1:])
    model = bundle["model"]
    raw_score = float(-model.decision_function(scaled)[0])
    reference = bundle["training_score_reference"]
    percentile = float(np.searchsorted(reference, raw_score, side="right") / len(reference))
    is_anomalous = bool(model.predict(scaled)[0] == -1)

    # Replace one scaled feature at a time with its training mean (zero).
    # Positive score change indicates a local driver, not a causal attribution.
    counterfactuals = np.repeat(scaled, len(names), axis=0)
    counterfactuals[np.arange(len(names)), np.arange(len(names))] = 0.0
    counterfactual_scores = -model.decision_function(counterfactuals)
    score_changes = raw_score - counterfactual_scores
    ranked = np.argsort(-score_changes)
    top_features = [names[index] for index in ranked if score_changes[index] > 1e-12][:5]
    return ScoreResponse(
        tx_id=current.tx_id, anomaly_score=percentile,
        is_anomalous=is_anomalous, top_contributing_features=top_features,
    )


def create_app(artifact_path: str | Path = DEFAULT_ARTIFACT) -> FastAPI:
    """Construct the API with a model loaded once per worker at startup."""
    path = Path(artifact_path)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.model_bundle = _load_artifact(path)
        application.state.gpu_metadata = _gpu_metadata()
        yield

    application = FastAPI(title="ML Anomaly Engine", version="1.0.0", lifespan=lifespan)

    @application.get("/health")
    def health(request: Request) -> dict[str, Any]:
        """Report model readiness and actual CPU inference device metadata."""
        return {
            "status": "ok",
            "model_loaded": hasattr(request.app.state, "model_bundle"),
            "inference_device": "cpu",
            "cpu_architecture": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "gpu_used_for_inference": False,
            **request.app.state.gpu_metadata,
        }

    @application.post("/score-transaction", response_model=ScoreResponse)
    def score_transaction(payload: ScoreRequest, request: Request) -> ScoreResponse:
        """Score one transaction using supplied as-of context."""
        return _score(payload, request.app.state.model_bundle)

    @application.post("/score-batch", response_model=BatchScoreResponse)
    def score_batch(payload: BatchScoreRequest, request: Request) -> BatchScoreResponse:
        """Score several transactions against shared context, preserving order."""
        bundle = request.app.state.model_bundle
        # De-duplicate by tx_id so a row sent in both lists is counted once.
        history = list({
            row.tx_id: row for row in [*payload.historical_transactions, *payload.transactions]
        }.values())
        return BatchScoreResponse(results=[
            _score(
                ScoreRequest.model_construct(
                    transaction=row, historical_transactions=history,
                    access_logs=payload.access_logs, accounts=payload.accounts,
                ),
                bundle,
            )
            for row in payload.transactions
        ])

    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("ML_BIND_HOST", "0.0.0.0"), port=8001)
