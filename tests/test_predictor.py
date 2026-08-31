"""Tests for the prediction service."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ml.predictor import (
    ModelNotAvailableError,
    PhishingPredictor,
    build_indicators,
)
from src.features.url_features import extract_features
from tests.conftest import requires_model


def test_missing_model_raises_a_clean_error(tmp_path: Path):
    """A missing artefact must surface as a domain error, not a stack trace."""
    predictor = PhishingPredictor(tmp_path / "nope.pkl")
    assert predictor.is_ready is False
    with pytest.raises(ModelNotAvailableError):
        predictor.predict("https://example.com")


def test_corrupt_model_file_raises_a_clean_error(tmp_path: Path):
    """A damaged pickle must not leak joblib internals to the caller."""
    broken = tmp_path / "broken.pkl"
    broken.write_bytes(b"definitely not a joblib bundle")
    with pytest.raises(ModelNotAvailableError):
        PhishingPredictor(broken).predict("https://example.com")


@requires_model
def test_prediction_result_is_well_formed(predictor):
    """Every field of the public contract is present and in range."""
    result = predictor.predict("https://example.com")

    assert result.prediction in {"Safe", "Suspicious", "Phishing"}
    assert 50.0 <= result.confidence <= 100.0
    assert 0 <= result.risk_score <= 100
    assert result.risk_level in {"Low", "Medium", "High", "Critical"}
    assert 0.0 <= result.phishing_probability <= 1.0
    assert result.features and result.timestamp and result.model_name


@requires_model
def test_risk_score_tracks_the_probability(predictor):
    """The risk score is the phishing probability, not an invented number."""
    result = predictor.predict("http://secure-login.verify-account.tk/webscr?cmd=1")
    assert result.risk_score == round(result.phishing_probability * 100)


@requires_model
def test_obvious_phishing_scores_above_an_obvious_legitimate_url(predictor):
    """Ranking sanity check on two clearly different URLs."""
    phishing = predictor.predict(
        "http://appleid.apple.com-verify-account.serv-login.ml/update/login.php?cmd=verify"
    )
    legitimate = predictor.predict("https://www.google.com")
    assert phishing.risk_score > legitimate.risk_score
    assert phishing.prediction == "Phishing"


@requires_model
def test_verdict_thresholds_are_applied_consistently(predictor):
    """Verdict and risk level must agree with the configured thresholds."""
    for url in [
        "https://www.google.com",
        "http://bit.ly/2xY3z",
        "https://en.wikipedia.org/wiki/Phishing",
    ]:
        result = predictor.predict(url)
        probability = result.phishing_probability
        if probability >= predictor.phishing_threshold:
            assert result.prediction == "Phishing"
        elif probability >= predictor.suspicious_threshold:
            assert result.prediction == "Suspicious"
        else:
            assert result.prediction == "Safe"


@requires_model
def test_contributions_are_measured_counterfactuals(predictor):
    """Each reported contribution must describe a real feature of the URL."""
    result = predictor.predict(
        "http://secure-paypal.verify-account-login.tk/webscr/update.php?cmd=login"
    )
    assert result.contributions, "a phishing-looking URL should have drivers"
    for item in result.contributions:
        assert item["feature"] in result.features
        assert item["value"] == result.features[item["feature"]]
        assert item["direction"] in {"increases", "reduces"}
        assert abs(item["impact"]) >= 0.5


@requires_model
def test_identical_urls_produce_identical_results(predictor):
    """Scoring is deterministic: no randomness leaks into the verdict."""
    first = predictor.predict("https://example.com/login")
    second = predictor.predict("https://example.com/login")
    assert first.phishing_probability == second.phishing_probability
    assert first.features == second.features


@requires_model
def test_model_info_exposes_training_metadata(predictor):
    """The dashboard reads its model facts from the artefact itself."""
    info = predictor.model_info()
    assert info["model_name"]
    assert info["n_features"] >= 15
    assert 0.0 <= info["metrics"]["roc_auc"] <= 1.0
    assert info["thresholds"]["suspicious"] < info["thresholds"]["phishing"]


def test_indicators_describe_only_what_is_present():
    """Indicators are observations, so they must match the extracted features."""
    url = "http://192.168.0.9:8080/wp-admin/verify/login.php"
    indicators = build_indicators(url, extract_features(url))
    text = " ".join(item["text"] for item in indicators)

    assert "IP address" in text
    assert "non-standard port" in text
    assert "not encrypted" in text  # plain HTTP
    assert "No structural red flags" not in text


def test_indicators_follow_the_real_host_not_the_userinfo():
    """`user@host` must be reported as an '@' trick, not as an IP hostname.

    In `http://192.168.0.9@evil.tk/`, the host is `evil.tk`; the IP is only
    userinfo. Claiming "the hostname is an IP address" would be a false
    explanation, so only the '@' indicator may fire.
    """
    url = "http://192.168.0.9@evil.tk/login"
    indicators = build_indicators(url, extract_features(url))
    text = " ".join(item["text"] for item in indicators)

    assert "'@' symbol" in text
    assert "IP address" not in text


def test_clean_url_reports_no_red_flags():
    """A plain HTTPS URL yields a positive, non-alarming summary."""
    url = "https://example.com/"
    indicators = build_indicators(url, extract_features(url))
    assert any("No structural red flags" in item["text"] for item in indicators)
    assert all(item["level"] == "good" for item in indicators)
