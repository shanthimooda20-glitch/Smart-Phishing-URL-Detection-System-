"""Tests for persistence, filtering and the aggregate statistics."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.database.models import URLAnalysis, db
from src.database.repository import query_history, record_analysis
from src.ml.predictor import PredictionResult


def _result(url: str, prediction: str, risk: int, level: str) -> PredictionResult:
    """Build a prediction result without needing a trained model."""
    return PredictionResult(
        url=url,
        prediction=prediction,
        confidence=90.0,
        risk_score=risk,
        risk_level=level,
        phishing_probability=risk / 100,
        features={"url_length": float(len(url))},
        model_name="Test Model",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@pytest.fixture()
def seeded(app):
    """Insert a small, deterministic set of analyses."""
    rows = [
        _result("https://safe-one.com", "Safe", 10, "Low"),
        _result("https://safe-two.com", "Safe", 20, "Low"),
        _result("http://maybe.com/login", "Suspicious", 55, "High"),
        _result("http://bad-one.tk/verify", "Phishing", 91, "Critical"),
        _result("http://bad-two.tk/login", "Phishing", 78, "Critical"),
    ]
    for row in rows:
        record_analysis(row, source="api")
    return rows


def test_record_analysis_persists_every_field(app):
    """A scan is stored with the exact values that were shown to the user."""
    stored = record_analysis(_result("https://example.com", "Safe", 12, "Low"))
    assert stored is not None

    row = db.session.get(URLAnalysis, stored.id)
    assert row.url == "https://example.com"
    assert row.prediction == "Safe"
    assert row.risk_score == 12
    assert row.risk_level == "Low"
    assert row.model_name == "Test Model"
    assert row.analysis_timestamp is not None


def test_long_urls_are_truncated_to_the_column_width(app):
    """Storage must not blow up on a maximum-length URL."""
    long_url = "http://example.com/" + "a" * 4000
    stored = record_analysis(_result(long_url, "Phishing", 80, "Critical"))
    assert stored is not None
    assert len(stored.url) <= 2048


def test_statistics_are_computed_from_stored_rows(app, seeded):
    """Dashboard counters aggregate the database, never a constant."""
    stats = URLAnalysis.statistics()
    assert stats["total_scanned"] == 5
    assert stats["safe"] == 2
    assert stats["suspicious"] == 1
    assert stats["phishing"] == 2
    assert stats["average_confidence"] == 90.0
    assert stats["risk_levels"]["Critical"] == 2


def test_statistics_on_an_empty_database(app):
    """No rows must yield zeroes rather than a division error."""
    stats = URLAnalysis.statistics()
    assert stats["total_scanned"] == 0
    assert stats["average_confidence"] == 0.0


def test_history_filters_by_prediction(app, seeded):
    """The verdict filter returns only the matching rows."""
    page = query_history(prediction="Phishing")
    assert page["total"] == 2
    assert {row.prediction for row in page["items"]} == {"Phishing"}


def test_history_search_is_parameterised(app, seeded):
    """Search matches substrings and is safe against SQL injection."""
    assert query_history(search="bad-one")["total"] == 1

    # A classic injection payload must match nothing, not drop the table.
    injected = query_history(search="'; DROP TABLE url_analysis; --")
    assert injected["total"] == 0
    assert URLAnalysis.query.count() == 5


def test_history_sorting_uses_a_whitelist(app, seeded):
    """Known sort keys work; an unknown key falls back to the default."""
    ascending = query_history(sort="risk", order="asc")["items"]
    assert [row.risk_score for row in ascending] == sorted(
        row.risk_score for row in ascending
    )

    # An attempt to inject a column name is ignored rather than executed.
    fallback = query_history(sort="risk_score; DROP TABLE url_analysis")
    assert fallback["total"] == 5


def test_history_paginates(app, seeded):
    """Pagination reports the right page count and navigation flags."""
    first = query_history(per_page=2, page=1)
    assert len(first["items"]) == 2
    assert first["pages"] == 3
    assert first["has_next"] is True
    assert first["has_prev"] is False

    last = query_history(per_page=2, page=3)
    assert last["has_next"] is False


def test_daily_trend_covers_a_continuous_date_range(app, seeded):
    """Days without scans are filled with zeroes so the chart has no gaps."""
    trend = URLAnalysis.daily_trend(days=7)
    assert len(trend) == 7
    assert trend[-1]["safe"] == 2  # today's rows
    assert all(day["phishing"] == 0 for day in trend[:-1])

    dates = [day["date"] for day in trend]
    assert dates == sorted(dates)


def test_recent_returns_newest_first(app):
    """The activity feed is ordered by timestamp, newest first."""
    now = datetime.now(timezone.utc)
    for offset, name in enumerate(["oldest", "middle", "newest"]):
        row = URLAnalysis(
            url=f"https://{name}.com", prediction="Safe", confidence=80.0,
            risk_score=10, risk_level="Low", model_name="Test",
            analysis_timestamp=now - timedelta(minutes=10 - offset * 5),
        )
        db.session.add(row)
    db.session.commit()

    assert [row.url for row in URLAnalysis.recent(3)] == [
        "https://newest.com", "https://middle.com", "https://oldest.com"
    ]


def test_to_dict_is_json_serialisable(app):
    """Rows serialise cleanly for the API."""
    stored = record_analysis(_result("https://example.com", "Safe", 5, "Low"))
    payload = stored.to_dict()
    assert payload["url"] == "https://example.com"
    assert payload["analysis_timestamp"].startswith("20")
