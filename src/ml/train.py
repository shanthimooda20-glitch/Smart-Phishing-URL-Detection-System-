"""End-to-end training pipeline for the phishing URL classifier.

Run it with::

    python src/ml/train.py                    # train on data/phishing_urls.csv
    python src/ml/train.py --sample 20000     # quick run on a stratified sample
    python src/ml/train.py --dataset path.csv --output models/phishing_model.pkl

The script loads the dataset, cleans it, extracts the lexical feature matrix,
trains four candidate classifiers, compares them on a held-out test split and
persists the winner (together with everything the prediction service needs to
reproduce the exact same preprocessing).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

import joblib
import numpy as np
import pandas as pd

# Allow `python src/ml/train.py` to resolve the `src` package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import BaseConfig
from src.features.url_features import FEATURE_NAMES, extract_features

LOGGER = logging.getLogger("train")

#: Accepted column names for the URL and the label, in priority order.
_URL_COLUMNS = ("url", "urls", "link", "domain", "website")
_LABEL_COLUMNS = (
    "label", "ismalicious", "is_malicious", "type", "class", "status",
    "result", "target", "phishing", "verdict",
)

#: Textual labels mapped onto the binary target (1 = phishing/malicious).
_POSITIVE_LABELS = frozenset({
    "1", "phishing", "phish", "malicious", "malware", "bad", "spam",
    "defacement", "attack", "fraud", "yes", "true",
})
_NEGATIVE_LABELS = frozenset({
    "0", "-1", "legitimate", "legit", "benign", "good", "safe", "clean",
    "no", "false", "normal", "ham",
})

CLASS_NAMES = ("Legitimate", "Phishing")


# --------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ModelResult:
    """Evaluation summary for one candidate model."""

    name: str
    estimator: Any
    requires_scaling: bool
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    brier: float
    train_seconds: float
    confusion: list[list[int]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary (no estimator object)."""
        return {
            "name": self.name,
            "accuracy": round(self.accuracy, 6),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "roc_auc": round(self.roc_auc, 6),
            "brier_score": round(self.brier, 6),
            "train_seconds": round(self.train_seconds, 3),
            "confusion_matrix": self.confusion,
        }


# --------------------------------------------------------------------------
# Dataset loading and cleaning
# --------------------------------------------------------------------------


def _resolve_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    """Return the first column in ``columns`` matching ``candidates``."""
    lowered = {str(column).strip().lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _coerce_label(value: Any) -> int | None:
    """Map a raw label onto ``0`` (legitimate), ``1`` (phishing) or ``None``."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    token = str(value).strip().lower()
    if token in _POSITIVE_LABELS:
        return 1
    if token in _NEGATIVE_LABELS:
        return 0
    return None


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and clean the URL dataset.

    The CSV must contain a URL column and a label column; the names are
    detected case-insensitively (see ``_URL_COLUMNS`` / ``_LABEL_COLUMNS``).

    Cleaning steps
    --------------
    1. Drop rows with a missing URL or a label that cannot be interpreted.
    2. Strip whitespace and drop empty / absurdly short URLs.
    3. Drop exact duplicate URLs (they would leak between train and test).

    Raises
    ------
    FileNotFoundError
        When the dataset file does not exist.
    ValueError
        When the required columns or usable rows are missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. See data/README.md for how to obtain one."
        )

    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], on_bad_lines="skip")
    url_column = _resolve_column(frame.columns, _URL_COLUMNS)
    label_column = _resolve_column(frame.columns, _LABEL_COLUMNS)
    if url_column is None or label_column is None:
        raise ValueError(
            "Dataset must contain a URL column (url/link/domain) and a label "
            f"column (label/type/isMalicious). Found: {list(frame.columns)}"
        )

    raw_rows = len(frame)
    frame = frame[[url_column, label_column]].rename(
        columns={url_column: "url", label_column: "label"}
    )

    frame["url"] = frame["url"].astype(str).str.strip()
    frame["label"] = frame["label"].map(_coerce_label)

    frame = frame.dropna(subset=["url", "label"])
    frame = frame[frame["url"].str.len() >= 4]
    frame = frame.drop_duplicates(subset=["url"], keep="first")
    frame["label"] = frame["label"].astype(int)

    if frame.empty:
        raise ValueError(f"No usable rows found in {path}.")
    if frame["label"].nunique() < 2:
        raise ValueError("Dataset contains a single class; both classes are required.")

    LOGGER.info(
        "Loaded %s rows from %s (%s dropped during cleaning)",
        f"{len(frame):,}", path.name, f"{raw_rows - len(frame):,}",
    )
    return frame.reset_index(drop=True)


def build_feature_matrix(urls: Sequence[str]) -> pd.DataFrame:
    """Extract the feature matrix for ``urls`` as a :class:`~pandas.DataFrame`."""
    records = [extract_features(url) for url in urls]
    matrix = pd.DataFrame.from_records(records, columns=list(FEATURE_NAMES))
    # Feature extraction is total, but guard against inf/NaN from odd input.
    return matrix.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def drop_constant_features(matrix: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove zero-variance columns.

    A feature that never changes in the training corpus carries no signal and
    — worse — would let the model rely on a value the corpus can never show.
    The bundled dataset stores URLs without a scheme, so ``has_https`` is
    constant there and is dropped automatically. The feature is still
    extracted and surfaced in the UI as an observed indicator.
    """
    constant = [column for column in matrix.columns if matrix[column].nunique() <= 1]
    return matrix.drop(columns=constant), constant


# --------------------------------------------------------------------------
# Model training
# --------------------------------------------------------------------------


def build_candidates(seed: int) -> Dict[str, tuple[Any, bool]]:
    """Return ``{name: (estimator, requires_scaling)}`` for every candidate.

    Only the linear model needs standardised inputs; tree ensembles are
    invariant to monotonic feature scaling, so they are trained on the raw
    matrix to keep the persisted pipeline as simple as possible.
    """
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, C=1.0, random_state=seed),
            True,
        ),
        "Decision Tree": (
            DecisionTreeClassifier(
                max_depth=18, min_samples_leaf=5, class_weight="balanced",
                random_state=seed,
            ),
            False,
        ),
        "Random Forest": (
            # min_samples_leaf=4 trims the fully grown trees: it costs ~0.5
            # accuracy points but roughly halves the serialised model, which
            # keeps start-up and per-request latency reasonable.
            RandomForestClassifier(
                n_estimators=150, max_depth=None, min_samples_leaf=4,
                n_jobs=-1, random_state=seed,
            ),
            False,
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=3,
                subsample=0.9, random_state=seed,
            ),
            False,
        ),
    }


def score_model(
    name: str,
    estimator: Any,
    requires_scaling: bool,
    x_test: np.ndarray,
    y_test: np.ndarray,
    train_seconds: float = 0.0,
) -> ModelResult:
    """Score an already-fitted ``estimator`` on the held-out test split."""
    predictions = estimator.predict(x_test)
    probabilities = estimator.predict_proba(x_test)[:, 1]

    return ModelResult(
        name=name,
        estimator=estimator,
        requires_scaling=requires_scaling,
        accuracy=accuracy_score(y_test, predictions),
        precision=precision_score(y_test, predictions, zero_division=0),
        recall=recall_score(y_test, predictions, zero_division=0),
        f1=f1_score(y_test, predictions, zero_division=0),
        roc_auc=roc_auc_score(y_test, probabilities),
        brier=brier_score_loss(y_test, probabilities),
        train_seconds=train_seconds,
        confusion=confusion_matrix(y_test, predictions).tolist(),
    )


def evaluate_model(
    name: str,
    estimator: Any,
    requires_scaling: bool,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> ModelResult:
    """Fit ``estimator`` and score it on the held-out test split."""
    started = time.perf_counter()
    estimator.fit(x_train, y_train)
    elapsed = time.perf_counter() - started
    return score_model(name, estimator, requires_scaling, x_test, y_test, elapsed)


def calibrate(
    result: ModelResult,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> tuple[ModelResult, Dict[str, Any]]:
    """Isotonically calibrate the winning model, keeping it only if it helps.

    Tree ensembles produce well-*ranked* but poorly *calibrated* scores: a
    forest that outputs 0.9 is not right 90% of the time. Since the interface
    reports a confidence percentage and derives a 0-100 risk score directly
    from the probability, calibration is what makes those numbers mean what
    they say. The calibrated model replaces the raw one only when it lowers
    the Brier score (mean squared probability error) on the test split.
    """
    calibrated = CalibratedClassifierCV(
        estimator=clone(result.estimator), method="isotonic", cv=3
    )
    started = time.perf_counter()
    calibrated.fit(x_train, y_train)
    elapsed = time.perf_counter() - started

    candidate = score_model(
        result.name, calibrated, result.requires_scaling, x_test, y_test, elapsed
    )
    applied = candidate.brier < result.brier
    report = {
        "method": "isotonic (3-fold)",
        "brier_before": round(result.brier, 6),
        "brier_after": round(candidate.brier, 6),
        "roc_auc_before": round(result.roc_auc, 6),
        "roc_auc_after": round(candidate.roc_auc, 6),
        "applied": applied,
    }
    return (candidate if applied else result), report


def compute_importances(
    result: ModelResult,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Sequence[str],
    seed: int,
    max_samples: int = 5000,
) -> Dict[str, float]:
    """Return normalised global feature importances for the winning model.

    Tree ensembles expose ``feature_importances_`` directly; for any other
    estimator (and as a sanity check) permutation importance is computed on a
    capped subsample of the test split so the run stays fast.
    """
    estimator = result.estimator
    if hasattr(estimator, "feature_importances_"):
        raw = np.asarray(estimator.feature_importances_, dtype=float)
        source = "impurity"
    else:
        size = min(max_samples, len(x_test))
        rng = np.random.default_rng(seed)
        index = rng.choice(len(x_test), size=size, replace=False)
        scores = permutation_importance(
            estimator, x_test[index], y_test[index],
            n_repeats=5, random_state=seed, scoring="roc_auc", n_jobs=-1,
        )
        raw = np.clip(scores.importances_mean, 0.0, None)
        source = "permutation"

    total = float(raw.sum()) or 1.0
    LOGGER.info("Feature importance computed via %s", source)
    return {
        name: round(float(value) / total, 6)
        for name, value in zip(feature_names, raw)
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _rule(char: str = "=", width: int = 64) -> str:
    return char * width


def print_header(dataset_path: Path, rows: int, positives: int) -> None:
    """Print the banner shown at the start of a training run."""
    print(_rule())
    print("SMART PHISHING URL DETECTION - MODEL TRAINING")
    print(_rule())
    print()
    print(f"Dataset file : {dataset_path}")
    print(f"Dataset size : {rows:,} URLs")
    print(f"  Phishing   : {positives:,} ({positives / rows:.1%})")
    print(f"  Legitimate : {rows - positives:,} ({1 - positives / rows:.1%})")
    print()


def print_model_report(result: ModelResult) -> None:
    """Print the metric block for a single model."""
    print(f"{result.name}")
    print(f"  Accuracy  : {result.accuracy * 100:6.2f}%")
    print(f"  Precision : {result.precision * 100:6.2f}%")
    print(f"  Recall    : {result.recall * 100:6.2f}%")
    print(f"  F1 Score  : {result.f1 * 100:6.2f}%")
    print(f"  ROC-AUC   : {result.roc_auc * 100:6.2f}%")
    print(f"  Brier     : {result.brier:6.4f}  (lower is better)")
    print(f"  Fit time  : {result.train_seconds:6.2f}s")
    print()


def print_confusion_matrix(matrix: Sequence[Sequence[int]]) -> None:
    """Pretty-print a 2x2 confusion matrix with labelled axes."""
    (true_negative, false_positive), (false_negative, true_positive) = matrix
    print("Confusion matrix (rows = actual, columns = predicted)")
    print(f"{'':>14}{'Legitimate':>14}{'Phishing':>12}")
    print(f"{'Legitimate':>14}{true_negative:>14,}{false_positive:>12,}")
    print(f"{'Phishing':>14}{false_negative:>14,}{true_positive:>12,}")
    print()
    print(f"  False positives (safe URL flagged) : {false_positive:,}")
    print(f"  False negatives (phishing missed)  : {false_negative:,}")
    print()


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def train(
    dataset_path: Path,
    model_path: Path,
    scaler_path: Path,
    metrics_path: Path,
    test_size: float = 0.2,
    seed: int = 42,
    sample: int | None = None,
) -> Dict[str, Any]:
    """Run the full pipeline and persist the best model.

    Returns
    -------
    dict
        The metrics document that is also written to ``metrics_path``.
    """
    frame = load_dataset(dataset_path)

    if sample is not None and sample < len(frame):
        frame, _ = train_test_split(
            frame, train_size=sample, stratify=frame["label"], random_state=seed
        )
        frame = frame.reset_index(drop=True)
        LOGGER.info("Using a stratified sample of %s rows", f"{len(frame):,}")

    print_header(dataset_path, len(frame), int(frame["label"].sum()))

    print("Extracting features ...")
    matrix = build_feature_matrix(frame["url"].tolist())
    matrix, dropped = drop_constant_features(matrix)
    feature_names = list(matrix.columns)
    print(f"  {len(feature_names)} model features retained "
          f"(of {len(FEATURE_NAMES)} extracted)")
    if dropped:
        print(f"  Dropped as constant in this corpus: {', '.join(dropped)}")
    print()

    features = matrix.to_numpy(dtype=float)
    labels = frame["label"].to_numpy(dtype=int)

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, stratify=labels, random_state=seed
    )
    print(f"Train / test split: {len(x_train):,} / {len(x_test):,} "
          f"(test_size={test_size})")
    print()

    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    print(_rule("-"))
    print("MODEL COMPARISON")
    print(_rule("-"))
    print()

    results: list[ModelResult] = []
    for name, (estimator, requires_scaling) in build_candidates(seed).items():
        LOGGER.info("Training %s", name)
        result = evaluate_model(
            name,
            estimator,
            requires_scaling,
            x_train_scaled if requires_scaling else x_train,
            y_train,
            x_test_scaled if requires_scaling else x_test,
            y_test,
        )
        results.append(result)
        print_model_report(result)

    # Selection: ROC-AUC is threshold independent and therefore the fairest
    # comparison; F1 breaks ties between models with equivalent ranking power.
    best = max(results, key=lambda item: (round(item.roc_auc, 4), item.f1))

    print(_rule("-"))
    print(f"BEST MODEL: {best.name}")
    print(_rule("-"))
    print(f"  Selected on ROC-AUC ({best.roc_auc * 100:.2f}%), "
          f"F1 tie-break ({best.f1 * 100:.2f}%)")
    print()

    x_fit = x_train_scaled if best.requires_scaling else x_train
    x_eval = x_test_scaled if best.requires_scaling else x_test

    # Global importances come from the uncalibrated estimator, which exposes
    # them directly; calibration only rescales its output probabilities.
    importances = compute_importances(best, x_eval, y_test, feature_names, seed)

    print("Calibrating probabilities ...")
    deployed, calibration_report = calibrate(best, x_fit, y_train, x_eval, y_test, seed)
    print(f"  Brier score {calibration_report['brier_before']:.4f} -> "
          f"{calibration_report['brier_after']:.4f} "
          f"({'calibrated model deployed' if calibration_report['applied'] else 'kept uncalibrated model'})")
    print()
    print_model_report(deployed)

    predictions = deployed.estimator.predict(x_eval)
    print_confusion_matrix(deployed.confusion)
    print("Classification report")
    print(classification_report(
        y_test, predictions, target_names=list(CLASS_NAMES), digits=4
    ))

    print("Top 10 features by importance")
    for name, value in sorted(importances.items(), key=lambda kv: -kv[1])[:10]:
        bar = "#" * int(round(value * 60))
        print(f"  {name:<26}{value * 100:6.2f}%  {bar}")
    print()

    # Medians power the local sensitivity explanations in the predictor.
    medians = {
        name: float(value)
        for name, value in zip(feature_names, np.median(x_train, axis=0))
    }

    bundle: Dict[str, Any] = {
        "model": deployed.estimator,
        "model_name": deployed.name,
        "calibration": calibration_report,
        "scaler": scaler,
        "requires_scaling": deployed.requires_scaling,
        "feature_names": feature_names,
        "dropped_features": dropped,
        "feature_medians": medians,
        "feature_importances": importances,
        "class_names": list(CLASS_NAMES),
        "metrics": deployed.as_dict(),
        "dataset": {
            "path": str(dataset_path),
            "rows": int(len(frame)),
            "phishing_rows": int(frame["label"].sum()),
            "test_size": test_size,
            "random_state": seed,
        },
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": __import__("sklearn").__version__,
        "schema_version": 1,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path, compress=3)
    joblib.dump(scaler, scaler_path, compress=3)

    metrics_document = {
        "trained_at": bundle["trained_at"],
        "best_model": deployed.name,
        "selection_criterion": "roc_auc (f1 tie-break)",
        "calibration": calibration_report,
        "deployed_metrics": deployed.as_dict(),
        "dataset": bundle["dataset"],
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "dropped_features": dropped,
        "models": [result.as_dict() for result in results],
        "feature_importances": importances,
    }
    metrics_path.write_text(json.dumps(metrics_document, indent=2), encoding="utf-8")

    print(f"Model saved to   : {model_path}")
    print(f"Scaler saved to  : {scaler_path}")
    print(f"Metrics saved to : {metrics_path}")
    print(_rule())
    return metrics_document


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the training script."""
    parser = argparse.ArgumentParser(
        description="Train the phishing URL classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, default=BaseConfig.DATASET_PATH,
                        help="CSV file with `url` and `label` columns")
    parser.add_argument("--output", type=Path, default=BaseConfig.MODEL_PATH,
                        help="Destination for the serialised model bundle")
    parser.add_argument("--scaler-output", type=Path, default=BaseConfig.SCALER_PATH,
                        help="Destination for the standalone scaler")
    parser.add_argument("--metrics-output", type=Path, default=BaseConfig.METRICS_PATH,
                        help="Destination for the JSON metrics report")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of the data held out for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sample", type=int, default=None,
                        help="Train on a stratified sample of N rows")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s"
    )
    args = parse_args(argv)
    try:
        train(
            dataset_path=args.dataset,
            model_path=args.output,
            scaler_path=args.scaler_output,
            metrics_path=args.metrics_output,
            test_size=args.test_size,
            seed=args.seed,
            sample=args.sample,
        )
    except FileNotFoundError as error:
        print(f"\n[ERROR] {error}\n", file=sys.stderr)
        return 2
    except ValueError as error:
        print(f"\n[ERROR] Invalid dataset: {error}\n", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
