"""Train and evaluate a transaction-level Isolation Forest.

The model is fitted without ground-truth labels. Labels are used only to
stratify the evaluation split and calculate held-out diagnostics. Feature
windows are computed from transactions known up to each event timestamp.

Requires pandas, NumPy, SciPy, joblib, and scikit-learn.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from feature_extraction import extract_features


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data-engine" / "data"
DEFAULT_ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "model.pkl"


@dataclass(frozen=True)
class TrainingResult:
    """Held-out evaluation and score-correlation diagnostics."""

    metrics: dict[str, float]
    test_count: int
    test_positive_count: int
    feature_correlations: tuple[tuple[str, float], ...]
    artifact_path: Path


def _load_labeled_data(data_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load the canonical files and align binary labels by transaction ID."""
    transactions = pd.read_csv(data_dir / "transactions.csv", dtype={"tx_id": "string"})
    labels = pd.read_csv(data_dir / "ground_truth_labels.csv", dtype="string")
    if "target_id" in labels:
        # Canonical data-engine format: one row per entity, keep transaction rows.
        if "target_type" in labels:
            labels = labels[labels["target_type"].str.upper().eq("TRANSACTION")]
        labels = labels.rename(columns={"target_id": "tx_id"})
    if "tx_id" not in transactions or "tx_id" not in labels:
        raise ValueError("transactions and ground_truth_labels must both contain tx_id")
    if transactions["tx_id"].isna().any() or transactions["tx_id"].duplicated().any():
        raise ValueError("transactions contains missing or duplicate tx_id values")
    if labels["tx_id"].isna().any() or labels["tx_id"].duplicated().any():
        raise ValueError("ground_truth_labels contains missing or duplicate tx_id values")

    label_column = next((name for name in ("is_suspicious", "label") if name in labels), None)
    if label_column is None:
        raise ValueError("ground_truth_labels must contain is_suspicious or label")
    raw_labels = labels.set_index("tx_id")[label_column].reindex(transactions["tx_id"])
    if raw_labels.isna().any():
        raise ValueError("ground_truth_labels is missing labels for one or more transactions")
    normalized = raw_labels.astype("string").str.strip().str.lower().map(
        {"0": 0, "1": 1, "false": 0, "true": 1, "benign": 0, "suspicious": 1}
    )
    if normalized.isna().any():
        raise ValueError("ground_truth_labels must contain 0/1, true/false, or SUSPICIOUS/BENIGN values")
    return transactions, normalized.to_numpy(dtype=np.int8)


def _feature_correlations(
    matrix: np.ndarray, scores: np.ndarray, names: tuple[str, ...]
) -> tuple[tuple[str, float], ...]:
    """Rank signed Pearson correlations as a descriptive importance proxy."""
    ranked: list[tuple[str, float]] = []
    score_std = float(np.std(scores))
    for column, name in enumerate(names):
        values = matrix[:, column]
        if len(values) < 2 or score_std == 0 or float(np.std(values)) == 0:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(values, scores)[0, 1])
            if not np.isfinite(correlation):
                correlation = 0.0
        ranked.append((name, correlation))
    return tuple(sorted(ranked, key=lambda item: abs(item[1]), reverse=True))


def train_model(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    *,
    test_size: float = 0.25,
    contamination: float = 0.05,
    random_state: int = 42,
) -> TrainingResult:
    """Fit an unlabeled model, evaluate on held-out rows, and save artifacts.

    The saved joblib dictionary contains ``model``, ``scaler``, and ordered
    ``feature_names``. At inference, call ``extract_features`` with historical
    context, transform with the saved scaler, then use the model. Larger
    values of ``-model.decision_function(X_scaled)`` mean more anomalous.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not 0 < contamination <= 0.5:
        raise ValueError("contamination must be in (0, 0.5]")

    source_dir = Path(data_dir)
    target = Path(artifact_path)
    transactions, labels = _load_labeled_data(source_dir)
    if len(transactions) < 8:
        raise ValueError("at least 8 transactions are required for a hold-out evaluation")

    accounts = pd.read_csv(source_dir / "accounts.csv", dtype={"account_id": "string"})
    access_logs = pd.read_csv(source_dir / "access_logs.csv", dtype={"account_id": "string"})
    X, feature_names = extract_features(transactions, access_logs=access_logs, accounts=accounts)
    if not np.isfinite(X).all():
        raise ValueError("feature extraction produced non-finite values")

    indices = np.arange(len(transactions))
    classes, counts = np.unique(labels, return_counts=True)
    can_stratify = (
        len(classes) == 2
        and counts.min() >= 2
        and round(len(indices) * test_size) >= 2
        and len(indices) - round(len(indices) * test_size) >= 2
    )
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state,
        stratify=labels if can_stratify else None,
    )
    scaler = StandardScaler().fit(X[train_idx])
    X_train = scaler.transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])
    model = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=random_state, n_jobs=-1
    ).fit(X_train)
    training_score_reference = np.sort(-model.decision_function(X_train))

    y_test = labels[test_idx]
    anomaly_scores = -model.decision_function(X_test)
    predictions = (model.predict(X_test) == -1).astype(np.int8)
    tn, fp, _, _ = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    metrics = {
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        # Average precision summarizes the precision-recall curve.
        "pr_auc": float(average_precision_score(y_test, anomaly_scores)) if y_test.any() else float("nan"),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else float("nan"),
    }
    correlations = _feature_correlations(X[test_idx], anomaly_scores, feature_names)

    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "score_direction": "higher_is_more_anomalous",
            "decision_threshold": 0.0,
            "contamination": contamination,
            "training_score_reference": training_score_reference,
        },
        target,
    )
    return TrainingResult(metrics, len(test_idx), int(y_test.sum()), correlations, target)


def main() -> None:
    """Train from the standard data directory or CLI overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    result = train_model(
        data_dir=args.data_dir, artifact_path=args.artifact, test_size=args.test_size,
        contamination=args.contamination, random_state=args.random_state,
    )
    print(f"Held-out transactions: {result.test_count} ({result.test_positive_count} suspicious)")
    for name, value in result.metrics.items():
        print(f"{name}: {value:.4f}" if np.isfinite(value) else f"{name}: N/A (class absent in hold-out)")
    print("Top anomaly-score correlations (descriptive, not causal model importances):")
    for name, value in result.feature_correlations[:10]:
        print(f"  {name}: {value:+.4f}")
    print(f"Saved model and scaler: {result.artifact_path}")


if __name__ == "__main__":
    main()
