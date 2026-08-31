"""Persistence helpers.

Keeping the queries here means the route handlers never build SQL themselves
and never touch the session directly, so transaction handling and error
translation live in exactly one place.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from sqlalchemy import or_
from sqlalchemy.exc import SQLAlchemyError

from src.database.models import URLAnalysis, db
from src.ml.predictor import PredictionResult

LOGGER = logging.getLogger(__name__)

__all__ = ["DatabaseError", "record_analysis", "query_history"]

_SORTABLE = {
    "date": URLAnalysis.analysis_timestamp,
    "risk": URLAnalysis.risk_score,
    "confidence": URLAnalysis.confidence,
    "url": URLAnalysis.url,
}


class DatabaseError(RuntimeError):
    """Raised when a database operation fails.

    Callers surface a generic message; the underlying exception is logged.
    """


def record_analysis(result: PredictionResult, source: str = "web") -> Optional[URLAnalysis]:
    """Persist ``result`` and return the stored row.

    A logging failure must never cost the user their analysis, so the caller
    can treat ``None`` as "not stored" and still render the verdict.
    """
    record = URLAnalysis(
        url=result.url[:2048],
        prediction=result.prediction,
        confidence=float(result.confidence),
        risk_score=int(result.risk_score),
        risk_level=result.risk_level,
        model_name=result.model_name[:64],
        source=source[:16],
    )
    try:
        db.session.add(record)
        db.session.commit()
        return record
    except SQLAlchemyError:
        db.session.rollback()
        LOGGER.exception("Failed to persist analysis")
        return None


def query_history(
    search: str = "",
    prediction: str = "",
    sort: str = "date",
    order: str = "desc",
    page: int = 1,
    per_page: int = 15,
) -> Dict[str, Any]:
    """Return a filtered, sorted and paginated slice of the analysis history.

    Parameters are treated as untrusted input: ``search`` is bound as a query
    parameter (never string-formatted into SQL), ``sort``/``order`` are matched
    against a whitelist, and the page size is clamped.

    Raises
    ------
    DatabaseError
        When the query cannot be executed.
    """
    page = max(1, int(page or 1))
    per_page = max(1, min(int(per_page or 15), 100))

    query = URLAnalysis.query
    if search:
        pattern = f"%{search.strip()[:200]}%"
        query = query.filter(
            or_(URLAnalysis.url.ilike(pattern), URLAnalysis.risk_level.ilike(pattern))
        )
    if prediction and prediction.lower() != "all":
        query = query.filter(URLAnalysis.prediction == prediction.title())

    column = _SORTABLE.get(sort, URLAnalysis.analysis_timestamp)
    direction = column.asc() if order == "asc" else column.desc()

    try:
        pagination = query.order_by(direction, URLAnalysis.id.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
    except SQLAlchemyError as error:
        LOGGER.exception("History query failed")
        raise DatabaseError("The analysis history could not be loaded.") from error

    return {
        "items": pagination.items,
        "page": pagination.page,
        "pages": pagination.pages or 1,
        "per_page": per_page,
        "total": pagination.total,
        "has_prev": pagination.has_prev,
        "has_next": pagination.has_next,
        "filters": {
            "search": search,
            "prediction": prediction,
            "sort": sort,
            "order": order,
        },
    }
