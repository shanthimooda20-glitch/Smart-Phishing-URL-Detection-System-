"""Tests for URL feature extraction."""

from __future__ import annotations

import pytest

from src.features.url_features import (
    FEATURE_NAMES,
    extract_features,
    features_to_vector,
    normalise_url,
    shannon_entropy,
    split_host,
)


def test_extract_features_returns_every_declared_feature():
    """The dictionary must be complete and ordered like FEATURE_NAMES."""
    features = extract_features("https://example.com/path?a=1")
    assert list(features.keys()) == list(FEATURE_NAMES)
    assert len(features) >= 15
    assert all(isinstance(value, float) for value in features.values())


def test_scheme_is_recorded_but_not_counted():
    """https must set the flag without inflating the length counters."""
    https = extract_features("https://example.com/login")
    http = extract_features("http://example.com/login")

    assert https["has_https"] == 1.0
    assert http["has_https"] == 0.0
    # Only the flag differs: the scheme is excluded from every other feature.
    assert {k: v for k, v in https.items() if k != "has_https"} == {
        k: v for k, v in http.items() if k != "has_https"
    }


def test_www_prefix_is_canonicalised_away():
    """`www.` is a fixed prefix and must not change the feature vector."""
    with_www = extract_features("https://www.example.com/a")
    without_www = extract_features("https://example.com/a")
    assert with_www == without_www


@pytest.mark.parametrize(
    "url,feature,expected",
    [
        ("http://192.168.1.1/login", "has_ip_address", 1.0),
        ("http://example.com/login", "has_ip_address", 0.0),
        ("http://example.com@evil.com/", "has_at_symbol", 1.0),
        ("http://bit.ly/abc", "is_shortened", 1.0),
        ("http://example.tk/a", "is_suspicious_tld", 1.0),
        ("http://example.com/a", "is_suspicious_tld", 0.0),
        ("http://example.com:8080/a", "has_port", 1.0),
        ("http://my-shop.example.com/a", "has_hyphen_in_domain", 0.0),
        ("http://my-shop.com/a", "has_hyphen_in_domain", 1.0),
        ("http://a.b.c.example.com/x", "num_subdomains", 3.0),
    ],
)
def test_individual_binary_features(url: str, feature: str, expected: float):
    """Spot-check the host-based indicators."""
    assert extract_features(url)[feature] == expected


def test_counting_features_are_accurate():
    """Character counters must reflect the canonicalised URL exactly."""
    features = extract_features("https://secure-login.example.com/a_b/c?x=1&y=2#frag")
    assert features["num_hyphens"] == 1.0
    assert features["num_underscores"] == 1.0
    assert features["num_query_params"] == 2.0
    assert features["num_fragments"] == 1.0
    assert features["num_dots"] == 2.0
    assert features["num_suspicious_keywords"] >= 2.0  # "secure", "login"


def test_percent_encoding_and_digit_ratio():
    """Encoded characters and digit density are measured, not guessed."""
    features = extract_features("http://example.com/a%20b%2Fc?id=123456")
    assert features["num_percent_encodings"] == 2.0
    assert 0 < features["digit_ratio"] < 1


def test_entropy_is_higher_for_random_hostnames():
    """A random-looking domain must score above a pronounceable one."""
    random_domain = extract_features("http://x7q2z9kd3v.com/")["domain_entropy"]
    plain_domain = extract_features("http://aaaaaaaaaa.com/")["domain_entropy"]
    assert random_domain > plain_domain
    assert shannon_entropy("") == 0.0


def test_malformed_input_never_raises():
    """Feature extraction must be total: bad input still yields a vector."""
    for candidate in ["", "   ", "http://", "://x", "http://[bad", "not a url", "a" * 3000]:
        features = extract_features(candidate)
        assert list(features.keys()) == list(FEATURE_NAMES)


def test_features_to_vector_respects_pinned_order():
    """A persisted model's column order wins over FEATURE_NAMES."""
    features = extract_features("https://example.com")
    pinned = ["num_dots", "url_length"]
    assert features_to_vector(features, pinned) == [
        features["num_dots"], features["url_length"]
    ]
    # Unknown columns default to 0.0 rather than raising.
    assert features_to_vector(features, ["does_not_exist"]) == [0.0]


@pytest.mark.parametrize(
    "host,expected",
    [
        ("www.bbc.co.uk", ("www", "bbc.co.uk", "co.uk")),
        ("en.wikipedia.org", ("en", "wikipedia.org", "org")),
        ("a.b.example.com", ("a.b", "example.com", "com")),
        ("localhost", ("", "localhost", "")),
    ],
)
def test_split_host_handles_compound_suffixes(host, expected):
    """Compound public suffixes must not be mistaken for subdomains."""
    assert split_host(host) == expected


def test_normalise_url_splits_the_scheme():
    """The scheme is separated from the analysed remainder."""
    assert normalise_url("HTTPS://example.com/a") == ("https", "example.com/a")
    assert normalise_url("example.com/a") == ("", "example.com/a")
