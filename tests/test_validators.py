"""Tests for URL validation and input hardening."""

from __future__ import annotations

import pytest

from src.utils.validators import URLValidationError, validate_url


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "http://example.com/login?next=/a#top",
        "example.com",                       # scheme is optional
        "http://192.168.0.1:8080/admin",
        "https://sub.domain.co.uk/a/b/c",
    ],
)
def test_accepts_analysable_urls(url: str):
    """Well-formed http(s) URLs and bare domains are accepted."""
    assert validate_url(url).host


@pytest.mark.parametrize(
    "url,fragment",
    [
        ("", "enter a URL"),
        ("   ", "enter a URL"),
        (None, "enter a URL"),
        ("javascript:alert(1)", "http:// and https://"),
        ("data:text/html;base64,PHNjcmlwdD4=", "http:// and https://"),
        ("file:///etc/passwd", "http:// and https://"),
        ("http://", "missing a domain"),
        ("http://exa mple.com", "cannot contain spaces"),
        ("localhostonly", "top-level domain"),
        ("http://exa..mple.com", "invalid domain"),
        (12345, "provided as text"),
    ],
)
def test_rejects_unusable_input(url, fragment: str):
    """Every rejection carries a message written for a human."""
    with pytest.raises(URLValidationError) as error:
        validate_url(url)
    assert fragment in str(error.value)


def test_rejects_oversized_input():
    """Oversized URLs are rejected rather than silently truncated."""
    with pytest.raises(URLValidationError) as error:
        validate_url("http://example.com/" + "a" * 5000, max_length=2048)
    assert "too long" in str(error.value)


def test_control_characters_are_stripped():
    """Embedded control characters cannot smuggle content through."""
    validated = validate_url("https://example.com/\x00\x1flogin")
    assert "\x00" not in validated.normalised
    assert validated.normalised == "https://example.com/login"


def test_missing_scheme_defaults_to_http_without_losing_the_path():
    """A bare domain is normalised, keeping everything the user typed."""
    validated = validate_url("example.com/login?a=1")
    assert validated.normalised == "http://example.com/login?a=1"
    assert validated.scheme == "http"
    assert validated.host == "example.com"


def test_scheme_is_lowercased_but_the_rest_is_preserved():
    """Case-sensitive path components survive normalisation."""
    validated = validate_url("HTTPS://Example.com/CaseSensitive")
    assert validated.normalised == "https://Example.com/CaseSensitive"
