"""Server-rendered pages: scanner, result, history and dashboard."""

from __future__ import annotations

import json
import logging

from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)

from src.database.models import URLAnalysis
from src.database.repository import DatabaseError, query_history
from src.ml.predictor import ModelNotAvailableError
from src.routes.service import AnalysisUnavailable, InvalidSubmission, analyse_url

LOGGER = logging.getLogger(__name__)

web_bp = Blueprint("web", __name__)

#: Shown under the scanner input as a one-click example.
EXAMPLE_URL = "https://example.com"


@web_bp.get("/")
def index():
    """Landing page with the URL scanner."""
    return render_template(
        "index.html",
        example_url=EXAMPLE_URL,
        recent=_safe_recent(5),
    )


@web_bp.post("/analyze")
def analyze():
    """Handle a scanner submission and render the result page."""
    try:
        result, stored = analyse_url(request.form.get("url"), source="web")
    except InvalidSubmission as error:
        flash(str(error), "warning")
        return redirect(url_for("web.index"))
    except AnalysisUnavailable as error:
        flash(str(error), "danger")
        return redirect(url_for("web.index"))

    return render_template(
        "result.html",
        result=result,
        stored=stored,
        contributions_json=json.dumps(result.contributions),
        features_json=json.dumps(result.features),
    )


@web_bp.get("/history")
def history():
    """Paginated, searchable and sortable list of past analyses."""
    try:
        page = int(request.args.get("page", 1))
    except (TypeError, ValueError):
        page = 1

    try:
        data = query_history(
            search=request.args.get("search", ""),
            prediction=request.args.get("prediction", ""),
            sort=request.args.get("sort", "date"),
            order=request.args.get("order", "desc"),
            page=page,
            per_page=current_app.config["HISTORY_PAGE_SIZE"],
        )
    except DatabaseError as error:
        flash(str(error), "danger")
        data = {
            "items": [], "page": 1, "pages": 1, "total": 0,
            "has_prev": False, "has_next": False,
            "filters": {"search": "", "prediction": "", "sort": "date", "order": "desc"},
        }

    return render_template("history.html", **data)


@web_bp.get("/dashboard")
def dashboard():
    """Security analytics dashboard.

    The page renders the counters server-side and lets ``static/js/dashboard.js``
    pull the same numbers from ``/api/statistics`` for the charts.
    """
    statistics = URLAnalysis.statistics()
    predictor = current_app.extensions["predictor"]
    try:
        model = predictor.model_info()
    except ModelNotAvailableError:
        model = None

    return render_template(
        "dashboard.html",
        statistics=statistics,
        model=model,
        model_metrics=_model_comparison(),
        recent=_safe_recent(8),
    )


@web_bp.get("/about")
def about():
    """Explain how the detector works and where its limits are."""
    predictor = current_app.extensions["predictor"]
    try:
        model = predictor.model_info()
    except ModelNotAvailableError:
        model = None
    return render_template("about.html", model=model, metrics=_model_comparison())


def _safe_recent(limit: int):
    """Recent analyses, tolerating an unavailable database."""
    try:
        return URLAnalysis.recent(limit)
    except Exception:  # noqa: BLE001 - the page must still render
        LOGGER.exception("Could not load recent analyses")
        return []


def _model_comparison() -> dict | None:
    """Load ``models/metrics.json`` (written by the training script)."""
    path = current_app.config["METRICS_PATH"]
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        LOGGER.info("No training metrics available at %s", path)
        return None
