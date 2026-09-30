"""End-to-end tests for the ML inference API against the generated dataset.

Run from ``ml-engine/`` after ``python train.py``:  ``python -m pytest -q``
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from feature_extraction import FEATURE_NAMES, extract_features
from service import DEFAULT_ARTIFACT, create_app


DATA_DIR = Path(__file__).resolve().parent.parent / "data-engine" / "data"


@pytest.fixture(scope="module")
def data() -> dict[str, pd.DataFrame]:
    """Load the canonical CSVs exactly as the data engine writes them."""
    frames = {
        name: pd.read_csv(DATA_DIR / f"{name}.csv", dtype=str)
        for name in ("transactions", "access_logs", "accounts", "ground_truth_labels")
    }
    labels = frames["ground_truth_labels"]
    frames["tx_labels"] = labels[labels["target_type"] == "TRANSACTION"].set_index("target_id")
    return frames


@pytest.fixture(scope="module")
def client() -> TestClient:
    if not DEFAULT_ARTIFACT.is_file():
        pytest.skip("model artifact missing; run train.py first")
    with TestClient(create_app()) as test_client:
        yield test_client


def _context_for(data: dict[str, pd.DataFrame], tx: pd.Series) -> dict:
    """Build the as-of context the orchestrator would send for one transaction."""
    sender = tx["sender_account"]
    history = data["transactions"][data["transactions"]["sender_account"] == sender]
    logs = data["access_logs"][data["access_logs"]["account_id"] == sender]
    accounts = data["accounts"][data["accounts"]["account_id"] == sender]
    return {
        "transaction": tx.to_dict(),
        "historical_transactions": history.to_dict("records"),
        "access_logs": logs.to_dict("records"),
        "accounts": accounts[["account_id", "status"]].to_dict("records"),
    }


def _sample(data: dict[str, pd.DataFrame], scenario: str, n: int) -> pd.DataFrame:
    tx = data["transactions"]
    ids = data["tx_labels"].index[data["tx_labels"]["scenario_tag"] == scenario]
    return tx[tx["tx_id"].isin(ids)].head(n)


def test_features_accept_canonical_column_names(data):
    X, names = extract_features(
        data["transactions"].head(50), data["access_logs"], data["accounts"]
    )
    assert names == FEATURE_NAMES
    assert X.shape == (50, len(FEATURE_NAMES))
    # Canonical `sender_account` must be recognised, not treated as missing.
    assert X[:, FEATURE_NAMES.index("sender_missing")].sum() == 0


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True


@pytest.mark.parametrize(
    "scenario", ["SCENARIO_A_SMURFING_INSIDER_HELP", "SCENARIO_B_CIRCULAR_MONEY_FLOW"]
)
def test_fraud_scenarios_are_flagged(client, data, scenario):
    rows = _sample(data, scenario, 10)
    assert not rows.empty
    for _, tx in rows.iterrows():
        body = client.post("/score-transaction", json=_context_for(data, tx)).json()
        assert body["is_anomalous"], (tx["tx_id"], body)
        assert body["anomaly_score"] >= 0.9
        assert body["top_contributing_features"]


def test_background_traffic_is_mostly_clean(client, data):
    rows = _sample(data, "BACKGROUND_BENIGN", 200)
    flagged = sum(
        client.post("/score-transaction", json=_context_for(data, tx)).json()["is_anomalous"]
        for _, tx in rows.iterrows()
    )
    assert flagged / len(rows) <= 0.05


def test_batch_matches_single_scoring(client, data):
    rows = _sample(data, "SCENARIO_A_SMURFING_INSIDER_HELP", 4)
    sender = rows.iloc[0]["sender_account"]
    rows = rows[rows["sender_account"] == sender]
    context = _context_for(data, rows.iloc[0])
    batch = client.post("/score-batch", json={
        "transactions": rows.to_dict("records"),
        "historical_transactions": context["historical_transactions"],
        "access_logs": context["access_logs"],
        "accounts": context["accounts"],
    }).json()["results"]
    singles = [
        client.post("/score-transaction", json=_context_for(data, tx)).json()
        for _, tx in rows.iterrows()
    ]
    assert batch == singles


def test_rejects_malformed_request(client):
    response = client.post("/score-transaction", json={"transaction": {"tx_id": "X"}})
    assert response.status_code == 422


def test_single_score_latency(client, data):
    tx = _sample(data, "BACKGROUND_BENIGN", 1).iloc[0]
    payload = _context_for(data, tx)
    client.post("/score-transaction", json=payload)  # warm-up
    start = time.perf_counter()
    for _ in range(20):
        client.post("/score-transaction", json=payload)
    per_call_ms = (time.perf_counter() - start) / 20 * 1000
    print(f"\nmean /score-transaction latency: {per_call_ms:.1f} ms")
    assert per_call_ms < 250
