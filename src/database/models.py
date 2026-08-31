"""SQLAlchemy models and query helpers for stored analyses."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Index, func

db = SQLAlchemy()

__all__ = ["db", "URLAnalysis"]


def _utcnow() -> datetime:
    """Timezone-aware UTC timestamp (``datetime.utcnow`` is deprecated)."""
    return datetime.now(timezone.utc)


class URLAnalysis(db.Model):
    """One stored URL analysis.

    Every scan — from the web UI or the REST API — is persisted here, which is
    what makes the dashboard statistics real rather than decorative.
    """

    __tablename__ = "url_analysis"

    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(2048), nullable=False)
    prediction = db.Column(db.String(16), nullable=False, index=True)
    confidence = db.Column(db.Float, nullable=False)
    risk_score = db.Column(db.Integer, nullable=False, index=True)
    risk_level = db.Column(db.String(16), nullable=False)
    model_name = db.Column(db.String(64), nullable=False, default="")
    source = db.Column(db.String(16), nullable=False, default="web")
    analysis_timestamp = db.Column(
        db.DateTime(timezone=True), nullable=False, default=_utcnow, index=True
    )

    __table_args__ = (
        Index("ix_analysis_prediction_time", "prediction", "analysis_timestamp"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<URLAnalysis {self.id} {self.prediction} {self.risk_score}>"

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the row."""
        timestamp = self.analysis_timestamp
        return {
            "id": self.id,
            "url": self.url,
            "prediction": self.prediction,
            "confidence": round(float(self.confidence), 2),
            "risk_score": int(self.risk_score),
            "risk_level": self.risk_level,
            "model": self.model_name,
            "source": self.source,
            "analysis_timestamp": timestamp.isoformat() if timestamp else None,
        }

    # -- queries ----------------------------------------------------------

    @classmethod
    def statistics(cls) -> Dict[str, Any]:
        """Aggregate counters used by the dashboard and ``GET /api/statistics``.

        Every number is computed from the stored rows; nothing is hard-coded.
        """
        totals = dict(
            db.session.query(cls.prediction, func.count(cls.id))
            .group_by(cls.prediction)
            .all()
        )
        total = int(sum(totals.values()))
        average_confidence = db.session.query(func.avg(cls.confidence)).scalar()
        average_risk = db.session.query(func.avg(cls.risk_score)).scalar()

        risk_levels = dict(
            db.session.query(cls.risk_level, func.count(cls.id))
            .group_by(cls.risk_level)
            .all()
        )

        return {
            "total_scanned": total,
            "safe": int(totals.get("Safe", 0)),
            "suspicious": int(totals.get("Suspicious", 0)),
            "phishing": int(totals.get("Phishing", 0)),
            "average_confidence": round(float(average_confidence or 0.0), 2),
            "average_risk_score": round(float(average_risk or 0.0), 2),
            "risk_levels": {
                level: int(risk_levels.get(level, 0))
                for level in ("Low", "Medium", "High", "Critical")
            },
        }

    @classmethod
    def daily_trend(cls, days: int = 14) -> List[Dict[str, Any]]:
        """Return per-day counts per verdict for the detection-trend chart.

        Dates are grouped in SQL (``DATE(...)``) and gaps are filled in Python
        so the chart always shows a continuous axis.
        """
        from datetime import timedelta

        cutoff = _utcnow() - timedelta(days=days - 1)
        rows = (
            db.session.query(
                func.date(cls.analysis_timestamp).label("day"),
                cls.prediction,
                func.count(cls.id),
            )
            .filter(cls.analysis_timestamp >= cutoff)
            .group_by("day", cls.prediction)
            .all()
        )

        buckets: Dict[str, Dict[str, int]] = {}
        for day, prediction, count in rows:
            key = str(day)[:10]
            buckets.setdefault(key, {})[prediction] = int(count)

        today = _utcnow().date()
        series: List[Dict[str, Any]] = []
        for offset in range(days - 1, -1, -1):
            day = (today - timedelta(days=offset)).isoformat()
            counts = buckets.get(day, {})
            series.append({
                "date": day,
                "safe": counts.get("Safe", 0),
                "suspicious": counts.get("Suspicious", 0),
                "phishing": counts.get("Phishing", 0),
            })
        return series

    @classmethod
    def recent(cls, limit: int = 10) -> List["URLAnalysis"]:
        """Return the most recent analyses, newest first."""
        return (
            cls.query.order_by(cls.analysis_timestamp.desc(), cls.id.desc())
            .limit(limit)
            .all()
        )
