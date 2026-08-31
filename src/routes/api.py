"""JSON REST API.

All endpoints return JSON, use conventional status codes and never expose
internal error details.
"""

from __future__ import annotations

import logging

from flask import Blueprint, current_app, jsonify, request

from src.database.models import URLAnalysis
from src.database.repository import DatabaseError, query_history
from src.ml.predictor import ModelNotAvailableError
from src.routes.service import AnalysisUnavailable, InvalidSubmission, analyse_url

LOGGER = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)


def _error(message: str, status: int, code: str | None = None):
    """Build a consistent JSON error envelope."""
    return jsonify({"error": code or _STATUS_NAMES.get(status, "Error"),
                    "message": message}), status


_STATUS_NAMES = {
    400: "Bad Request",
    404: "Not Found",
    415: "Unsupported Media Type",
    500: "Internal Server Error",
    503: "Service Unavailable",
}


@api_bp.post("/analyze")
def analyze():
    """Analyse a single URL.

    Request body::

        {"url": "https://example.com"}

    Responses
    ---------
    200 - analysis result
    400 - missing/invalid URL or malformed JSON
    503 - the model has not been trained yet
    """
    if not request.is_json:
        return _error(
            "Send a JSON body with Content-Type: application/json.", 415
        )
    try:
        payload = request.get_json(silent=False)
    except Exception:  # noqa: BLE001 - werkzeug raises BadRequest subclasses
        return _error("The request body is not valid JSON.", 400)

    if not isinstance(payload, dict):
        return _error("The request body must be a JSON object.", 400)

    try:
        result, stored = analyse_url(payload.get("url"), source="api")
    except InvalidSubmission as error:
        return _error(str(error), 400, code="Invalid URL")
    except AnalysisUnavailable as error:
        return _error(str(error), 503, code="Model Unavailable")

    body = result.to_api_dict()
    body["stored"] = stored
    if _wants_details():
        body["features"] = result.features
        body["contributions"] = result.contributions
        body["indicators"] = result.indicators
    return jsonify(body), 200


def _wants_details() -> bool:
    """``True`` when the caller asked for the full feature breakdown."""
    flag = request.args.get("details", "").strip().lower()
    return flag in {"1", "true", "yes"}


@api_bp.get("/history")
def history():
    """Return a paginated slice of the analysis history.

    Query parameters: ``search``, ``prediction``, ``sort``
    (``date``/``risk``/``confidence``/``url``), ``order`` (``asc``/``desc``),
    ``page``, ``per_page``.
    """
    try:
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 20))
    except (TypeError, ValueError):
        return _error("`page` and `per_page` must be integers.", 400)

    try:
        result = query_history(
            search=request.args.get("search", ""),
            prediction=request.args.get("prediction", ""),
            sort=request.args.get("sort", "date"),
            order=request.args.get("order", "desc"),
            page=page,
            per_page=min(per_page, current_app.config["API_MAX_PAGE_SIZE"]),
        )
    except DatabaseError as error:
        return _error(str(error), 503)

    return jsonify({
        "items": [item.to_dict() for item in result["items"]],
        "page": result["page"],
        "pages": result["pages"],
        "per_page": result["per_page"],
        "total": result["total"],
    }), 200


@api_bp.get("/statistics")
def statistics():
    """Return dashboard counters, the daily trend and model metadata."""
    try:
        payload = {
            "statistics": URLAnalysis.statistics(),
            "trend": URLAnalysis.daily_trend(days=14),
        }
    except Exception:  # noqa: BLE001 - database problems must not 500 silently
        LOGGER.exception("Statistics query failed")
        return _error("Statistics are temporarily unavailable.", 503)

    predictor = current_app.extensions["predictor"]
    try:
        payload["model"] = predictor.model_info()
    except ModelNotAvailableError:
        payload["model"] = None
    return jsonify(payload), 200


@api_bp.get("/health")
def health():
    """Liveness probe: reports whether a trained model is loadable."""
    predictor = current_app.extensions["predictor"]
    return jsonify({
        "status": "ok",
        "model_available": predictor.is_ready,
    }), 200
