"""Tests for the JSON API and the server-rendered pages."""

from __future__ import annotations

from tests.conftest import requires_model


def test_health_endpoint_reports_model_availability(client):
    """The probe answers even when no model has been trained."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json["status"] == "ok"
    assert isinstance(response.json["model_available"], bool)


@requires_model
def test_analyze_returns_the_documented_contract(client):
    """POST /api/analyze answers with the published response shape."""
    response = client.post("/api/analyze", json={"url": "https://example.com"})
    assert response.status_code == 200

    body = response.json
    for field in ("prediction", "confidence", "risk_score", "risk_level"):
        assert field in body
    assert body["prediction"] in {"Safe", "Suspicious", "Phishing"}
    assert 0 <= body["risk_score"] <= 100
    assert body["stored"] is True


@requires_model
def test_analyze_can_return_the_feature_breakdown(client):
    """`?details=true` adds the features, drivers and indicators."""
    response = client.post(
        "/api/analyze?details=true", json={"url": "http://bit.ly/2xY3z"}
    )
    assert response.status_code == 200
    assert len(response.json["features"]) >= 15
    assert "indicators" in response.json


def test_analyze_rejects_an_empty_url(client):
    """A missing URL is a 400 with a friendly message, never a 500."""
    response = client.post("/api/analyze", json={})
    assert response.status_code == 400
    assert "enter a URL" in response.json["message"]


def test_analyze_rejects_a_dangerous_scheme(client):
    """Only http/https can be analysed."""
    response = client.post("/api/analyze", json={"url": "javascript:alert(1)"})
    assert response.status_code == 400
    assert response.json["error"] == "Invalid URL"


def test_analyze_rejects_an_oversized_url(client, app):
    """Oversized input is refused before any model work happens."""
    huge = "http://example.com/" + "a" * (app.config["MAX_URL_LENGTH"] + 10)
    response = client.post("/api/analyze", json={"url": huge})
    assert response.status_code == 400
    assert "too long" in response.json["message"]


def test_analyze_requires_json(client):
    """A form-encoded body gets 415 rather than an unhandled exception."""
    response = client.post("/api/analyze", data="url=https://example.com")
    assert response.status_code == 415


def test_analyze_rejects_a_non_object_body(client):
    """A JSON array is not a valid request body."""
    response = client.post("/api/analyze", json=["https://example.com"])
    assert response.status_code == 400


def test_history_endpoint_paginates(client):
    """History returns the pagination envelope even when empty."""
    response = client.get("/api/history?per_page=5")
    assert response.status_code == 200
    for field in ("items", "page", "pages", "total"):
        assert field in response.json


def test_history_rejects_non_integer_pagination(client):
    """Bad query parameters produce a 400, not a crash."""
    response = client.get("/api/history?page=abc")
    assert response.status_code == 400


def test_statistics_endpoint_shape(client):
    """Statistics always expose the counters and the 14-day trend."""
    response = client.get("/api/statistics")
    assert response.status_code == 200
    assert "statistics" in response.json
    assert len(response.json["trend"]) == 14


def test_unknown_api_route_returns_json_not_html(client):
    """404s under /api are JSON so API clients can parse them."""
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert response.is_json


def test_web_pages_render(client):
    """Every page renders without a trained model present."""
    for path in ("/", "/dashboard", "/history", "/about"):
        assert client.get(path).status_code == 200


def test_unknown_page_renders_the_error_template(client):
    """Web 404s return the styled error page, not a stack trace."""
    response = client.get("/definitely-not-a-page")
    assert response.status_code == 404
    assert b"404" in response.data
    assert b"Traceback" not in response.data


@requires_model
def test_web_analysis_flow_renders_the_result(client):
    """The form flow produces a result page containing the verdict."""
    response = client.post(
        "/analyze",
        data={"url": "http://secure-login.verify-account.tk/webscr?cmd=1"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Risk score" in response.data


def test_web_analysis_flow_reports_invalid_input(client):
    """An invalid submission flashes a message instead of erroring."""
    response = client.post("/analyze", data={"url": "javascript:alert(1)"},
                           follow_redirects=True)
    assert response.status_code == 200
    assert b"http:// and https://" in response.data


@requires_model
def test_hostile_url_is_escaped_in_the_rendered_page(client):
    """A URL containing markup must never be reflected as live HTML."""
    payload = "http://evil.com/<script>alert(1)</script>"
    response = client.post("/analyze", data={"url": payload}, follow_redirects=True)
    assert response.status_code == 200
    assert b"<script>alert(1)</script>" not in response.data
