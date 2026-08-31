"""The single analysis workflow shared by the web UI and the REST API.

Both entry points call :func:`analyse_url`, so a URL is validated, scored and
logged exactly the same way regardless of how it arrived.
"""

from __future__ import annotations

import logging
from typing import Tuple

from flask import current_app

from src.database.repository import record_analysis
from src.ml.predictor import (
    ModelNotAvailableError,
    PhishingPredictor,
    PredictionError,
    PredictionResult,
)
from src.utils.validators import URLValidationError, validate_url

LOGGER = logging.getLogger(__name__)

__all__ = ["analyse_url", "AnalysisUnavailable", "InvalidSubmission"]


class InvalidSubmission(ValueError):
    """The submitted URL is not analysable (user error, HTTP 400)."""


class AnalysisUnavailable(RuntimeError):
    """The service cannot analyse right now (server side, HTTP 503)."""


def _predictor() -> PhishingPredictor:
    return current_app.extensions["predictor"]


def analyse_url(raw_url: object, source: str = "web") -> Tuple[PredictionResult, bool]:
    """Validate, score and persist one URL.

    Parameters
    ----------
    raw_url:
        Untrusted value straight from a form field or JSON body.
    source:
        ``"web"`` or ``"api"``; stored with the record for reporting.

    Returns
    -------
    tuple
        ``(result, stored)`` where ``stored`` reports whether the analysis was
        successfully written to the database.

    Raises
    ------
    InvalidSubmission
        The URL is empty, malformed, too long or uses an unsupported scheme.
    AnalysisUnavailable
        The model is missing or scoring failed.
    """
    max_length = current_app.config["MAX_URL_LENGTH"]
    try:
        validated = validate_url(raw_url, max_length=max_length)
    except URLValidationError as error:
        raise InvalidSubmission(str(error)) from error

    try:
        result = _predictor().predict(validated.normalised)
    except ModelNotAvailableError as error:
        LOGGER.error("Model unavailable: %s", error)
        raise AnalysisUnavailable(
            "The detection model is not available yet. "
            "Run `python src/ml/train.py` to train it."
        ) from error
    except PredictionError as error:
        raise AnalysisUnavailable("The URL could not be analysed.") from error

    stored = record_analysis(result, source=source) is not None
    LOGGER.info(
        "Analysed %s -> %s (risk %s, stored=%s)",
        validated.host, result.prediction, result.risk_score, stored,
    )
    return result, stored
