"""Prediction service: URL string in, structured risk assessment out.

The service is intentionally the only place that knows how a probability is
turned into a product-level verdict, so the web UI, the REST API and the test
suite all see identical results.

Safety
------
Nothing here touches the network. The submitted URL is parsed as a string and
never requested, resolved, rendered or executed.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import joblib
import numpy as np

from src.features.url_features import (
    FEATURE_NAMES,
    SHORTENER_DOMAINS,
    extract_features,
    features_to_vector,
    normalise_url,
)

LOGGER = logging.getLogger(__name__)

__all__ = [
    "ModelNotAvailableError",
    "PredictionError",
    "PhishingPredictor",
    "PredictionResult",
    "VERDICT_SAFE",
    "VERDICT_SUSPICIOUS",
    "VERDICT_PHISHING",
]

VERDICT_SAFE = "Safe"
VERDICT_SUSPICIOUS = "Suspicious"
VERDICT_PHISHING = "Phishing"

#: Risk-score bands (upper bound, label), evaluated in order.
_RISK_BANDS: tuple[tuple[int, str], ...] = (
    (25, "Low"),
    (50, "Medium"),
    (75, "High"),
    (101, "Critical"),
)

#: Minimum probability shift (in points) for a feature to be reported.
_MIN_CONTRIBUTION = 0.5

#: Human-readable descriptions used in the explanation panel.
_FEATURE_LABELS: Dict[str, str] = {
    "url_length": "Overall URL length",
    "domain_length": "Hostname length",
    "path_length": "URL path length",
    "num_dots": "Number of dots",
    "num_hyphens": "Number of hyphens",
    "num_underscores": "Number of underscores",
    "num_slashes": "Number of path segments",
    "num_special_chars": "Special-character count",
    "num_digits": "Digit count",
    "num_subdomains": "Number of subdomains",
    "has_https": "HTTPS in use",
    "has_ip_address": "IP address used as hostname",
    "has_at_symbol": "'@' symbol in URL",
    "has_double_slash_redirect": "Double slash inside the path",
    "has_port": "Explicit port number",
    "num_suspicious_keywords": "Suspicious keyword count",
    "num_query_params": "Query-parameter count",
    "num_fragments": "URL fragment count",
    "num_percent_encodings": "Percent-encoded characters",
    "domain_entropy": "Domain randomness (entropy)",
    "url_entropy": "URL randomness (entropy)",
    "digit_ratio": "Digit-to-character ratio",
    "letter_ratio": "Letter-to-character ratio",
    "longest_token_length": "Longest unbroken token",
    "tld_length": "Top-level-domain length",
    "is_suspicious_tld": "High-abuse top-level domain",
    "is_shortened": "URL shortener",
    "has_hyphen_in_domain": "Hyphen inside the registrable domain",
}


class PredictionError(RuntimeError):
    """Raised when an analysis cannot be completed."""


class ModelNotAvailableError(PredictionError):
    """Raised when the serialised model bundle is missing or unreadable."""


@dataclass(slots=True)
class PredictionResult:
    """Structured outcome of a single URL analysis."""

    url: str
    prediction: str
    confidence: float
    risk_score: int
    risk_level: str
    phishing_probability: float
    features: Dict[str, float]
    contributions: List[Dict[str, Any]] = field(default_factory=list)
    indicators: List[Dict[str, str]] = field(default_factory=list)
    model_name: str = ""
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)

    def to_api_dict(self) -> Dict[str, Any]:
        """Return the compact payload used by ``POST /api/analyze``."""
        return {
            "url": self.url,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "phishing_probability": self.phishing_probability,
            "model": self.model_name,
            "timestamp": self.timestamp,
        }


class PhishingPredictor:
    """Loads the trained bundle once and scores URLs against it.

    Parameters
    ----------
    model_path:
        Path to the joblib bundle produced by ``src/ml/train.py``.
    suspicious_threshold, phishing_threshold:
        Probability cut-offs that map the binary classifier's output onto the
        three product verdicts.
    """

    def __init__(
        self,
        model_path: Path,
        suspicious_threshold: float = 0.35,
        phishing_threshold: float = 0.70,
    ) -> None:
        self.model_path = Path(model_path)
        self.suspicious_threshold = suspicious_threshold
        self.phishing_threshold = phishing_threshold
        self._bundle: Dict[str, Any] | None = None
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """``True`` when a model file is present on disk."""
        return self.model_path.exists()

    def load(self) -> Dict[str, Any]:
        """Load (and cache) the model bundle.

        Raises
        ------
        ModelNotAvailableError
            When the file is missing or cannot be deserialised.
        """
        if self._bundle is not None:
            return self._bundle
        with self._lock:
            if self._bundle is not None:  # another thread won the race
                return self._bundle
            if not self.model_path.exists():
                raise ModelNotAvailableError(
                    f"No trained model at {self.model_path}. "
                    "Run `python src/ml/train.py` first."
                )
            try:
                bundle = joblib.load(self.model_path)
            except Exception as error:  # noqa: BLE001 - surface as a clean error
                LOGGER.exception("Failed to load model bundle")
                raise ModelNotAvailableError(
                    "The trained model could not be loaded. Please retrain it."
                ) from error
            if not isinstance(bundle, dict) or "model" not in bundle:
                raise ModelNotAvailableError(
                    "The model file is not a valid training bundle. Please retrain."
                )
            self._bundle = bundle
            LOGGER.info(
                "Loaded %s model (%d features, trained %s)",
                bundle.get("model_name", "unknown"),
                len(bundle.get("feature_names", ())),
                bundle.get("trained_at", "unknown"),
            )
            return bundle

    def model_info(self) -> Dict[str, Any]:
        """Return metadata about the loaded model for dashboards and the API."""
        bundle = self.load()
        return {
            "model_name": bundle.get("model_name", "unknown"),
            "trained_at": bundle.get("trained_at"),
            "n_features": len(bundle.get("feature_names", ())),
            "feature_names": list(bundle.get("feature_names", ())),
            "dropped_features": list(bundle.get("dropped_features", ())),
            "metrics": bundle.get("metrics", {}),
            "dataset": bundle.get("dataset", {}),
            "feature_importances": bundle.get("feature_importances", {}),
            "thresholds": {
                "suspicious": self.suspicious_threshold,
                "phishing": self.phishing_threshold,
            },
        }

    # -- scoring ------------------------------------------------------------

    def predict(self, url: str) -> PredictionResult:
        """Analyse ``url`` and return a :class:`PredictionResult`.

        The URL is expected to have been validated already (see
        :func:`src.utils.validators.validate_url`).
        """
        bundle = self.load()
        model = bundle["model"]
        feature_names: Sequence[str] = bundle["feature_names"]

        features = extract_features(url)
        vector = np.asarray(
            [features_to_vector(features, feature_names)], dtype=float
        )

        try:
            matrix = self._build_counterfactual_matrix(vector, bundle)
            probabilities = model.predict_proba(self._prepare(matrix, bundle))[:, 1]
        except Exception as error:  # noqa: BLE001 - never leak sklearn internals
            LOGGER.exception("Scoring failed for a submitted URL")
            raise PredictionError("The URL could not be analysed.") from error

        phishing_probability = float(probabilities[0])
        verdict = self._verdict(phishing_probability)
        risk_score = int(round(phishing_probability * 100))

        return PredictionResult(
            url=url,
            prediction=verdict,
            confidence=round(max(phishing_probability, 1 - phishing_probability) * 100, 2),
            risk_score=risk_score,
            risk_level=self._risk_level(risk_score),
            phishing_probability=round(phishing_probability, 6),
            features=features,
            contributions=self._contributions(
                feature_names, features, probabilities, bundle
            ),
            indicators=build_indicators(url, features),
            model_name=str(bundle.get("model_name", "unknown")),
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    # -- internals ----------------------------------------------------------

    def _prepare(self, matrix: np.ndarray, bundle: Dict[str, Any]) -> np.ndarray:
        """Apply the persisted scaler when the winning model needs it."""
        if bundle.get("requires_scaling") and bundle.get("scaler") is not None:
            return bundle["scaler"].transform(matrix)
        return matrix

    @staticmethod
    def _build_counterfactual_matrix(
        vector: np.ndarray, bundle: Dict[str, Any]
    ) -> np.ndarray:
        """Stack the real row with one median-substituted row per feature.

        Row 0 is the URL itself. Row ``i + 1`` is the same URL with feature
        ``i`` replaced by its training-set median. Scoring the whole stack in a
        single ``predict_proba`` call makes the explanation essentially free.
        """
        feature_names: Sequence[str] = bundle["feature_names"]
        medians = bundle.get("feature_medians", {})
        rows = np.repeat(vector, len(feature_names) + 1, axis=0)
        for index, name in enumerate(feature_names):
            rows[index + 1, index] = float(medians.get(name, 0.0))
        return rows

    def _contributions(
        self,
        feature_names: Sequence[str],
        features: Dict[str, float],
        probabilities: np.ndarray,
        bundle: Dict[str, Any],
        limit: int = 6,
    ) -> List[Dict[str, Any]]:
        """Rank features by their measured effect on *this* prediction.

        For each feature the phishing probability is recomputed with that
        feature reset to its training median. The difference is a genuine
        counterfactual measurement of how much the observed value moved this
        specific prediction — not a global importance score reused as if it
        were a local explanation.
        """
        baseline = float(probabilities[0])
        importances = bundle.get("feature_importances", {})
        contributions: List[Dict[str, Any]] = []

        for index, name in enumerate(feature_names):
            delta = (baseline - float(probabilities[index + 1])) * 100
            if abs(delta) < _MIN_CONTRIBUTION:
                continue
            contributions.append({
                "feature": name,
                "label": _FEATURE_LABELS.get(name, name.replace("_", " ").title()),
                "value": features.get(name, 0.0),
                "impact": round(delta, 2),
                "direction": "increases" if delta > 0 else "reduces",
                "global_importance": round(float(importances.get(name, 0.0)) * 100, 2),
            })

        contributions.sort(key=lambda item: abs(item["impact"]), reverse=True)
        return contributions[:limit]

    def _verdict(self, probability: float) -> str:
        """Map a phishing probability onto the three product verdicts."""
        if probability >= self.phishing_threshold:
            return VERDICT_PHISHING
        if probability >= self.suspicious_threshold:
            return VERDICT_SUSPICIOUS
        return VERDICT_SAFE

    @staticmethod
    def _risk_level(risk_score: int) -> str:
        """Map a 0-100 risk score onto a severity label."""
        for upper_bound, label in _RISK_BANDS:
            if risk_score < upper_bound:
                return label
        return _RISK_BANDS[-1][1]


def build_indicators(url: str, features: Dict[str, float]) -> List[Dict[str, str]]:
    """Describe what was *observed* in the URL, independently of the model.

    These are factual statements about the extracted features. They are shown
    next to — never instead of — the model contributions, so the interface
    never claims that an observation caused the classification.
    """
    scheme, remainder = normalise_url(url)
    host = remainder.split("/")[0].split("@")[-1].split(":")[0].lower()
    host = host[4:] if host.startswith("www.") else host

    indicators: List[Dict[str, str]] = []

    def add(level: str, text: str) -> None:
        indicators.append({"level": level, "text": text})

    if scheme == "https":
        add("good", "Transport encryption (HTTPS) is requested")
    elif scheme == "http":
        add("warning", "Plain HTTP - traffic to this site is not encrypted")

    if features.get("has_ip_address"):
        add("danger", "The hostname is a raw IP address instead of a domain name")
    if features.get("has_at_symbol"):
        add("danger", "Contains an '@' symbol, which can hide the real destination")
    if features.get("has_double_slash_redirect"):
        add("warning", "Contains a double slash inside the path (redirect pattern)")
    if features.get("is_shortened") or host in SHORTENER_DOMAINS:
        add("warning", "Uses a URL shortener, so the destination is hidden")
    if features.get("is_suspicious_tld"):
        add("warning", "Registered under a top-level domain frequently abused for phishing")
    if features.get("has_port"):
        add("warning", "Specifies a non-standard port")

    url_length = features.get("url_length", 0)
    if url_length >= 100:
        add("warning", f"Unusually long URL ({int(url_length)} characters)")
    keyword_count = int(features.get("num_suspicious_keywords", 0))
    if keyword_count >= 3:
        add("danger", f"Contains {keyword_count} sensitive keywords (login, verify, account ...)")
    elif keyword_count > 0:
        add("warning", f"Contains {keyword_count} sensitive keyword(s)")
    if features.get("num_subdomains", 0) >= 3:
        add("warning", f"Deep subdomain chain ({int(features['num_subdomains'])} levels)")
    if features.get("domain_entropy", 0) >= 4.0:
        add("warning", "Hostname characters look randomly generated (high entropy)")
    if features.get("num_percent_encodings", 0) >= 3:
        add("warning", "Heavy use of percent-encoding, which can obfuscate the URL")
    if features.get("has_hyphen_in_domain"):
        add("warning", "Hyphenated domain name, common in brand-impersonation URLs")

    if not any(item["level"] in {"warning", "danger"} for item in indicators):
        add("good", "No structural red flags were found in the URL string")
    return indicators


def describe_feature(name: str) -> str:
    """Return the human-readable label for a feature name."""
    return _FEATURE_LABELS.get(name, name.replace("_", " ").title())


#: Every feature the extractor produces, for UI rendering.
ALL_FEATURE_LABELS: Dict[str, str] = {
    name: _FEATURE_LABELS.get(name, name) for name in FEATURE_NAMES
}
