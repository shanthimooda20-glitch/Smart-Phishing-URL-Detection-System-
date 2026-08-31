"""Defensive validation and sanitisation for user supplied URLs.

The application is a *classifier*, not a crawler: submitted URLs are never
requested, resolved or rendered. Validation therefore focuses on rejecting
input that cannot be meaningfully analysed and on bounding the work a single
request can trigger.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlsplit

__all__ = ["URLValidationError", "ValidatedURL", "validate_url"]

#: Only web schemes are meaningful for phishing analysis. Anything else
#: (javascript:, data:, file: ...) is rejected outright.
_ALLOWED_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https"})

_SCHEME_RE: Final[re.Pattern[str]] = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.\-]*):")
_HOST_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:\[[0-9a-fA-F:.]+\]|[A-Za-z0-9¡-￿._~%\-]+)$"
)
#: C0/C1 control characters are stripped before analysis.
_CONTROL_CHARS: Final[re.Pattern[str]] = re.compile(r"[\x00-\x1f\x7f-\x9f]")

MIN_URL_LENGTH: Final[int] = 4


class URLValidationError(ValueError):
    """Raised when a submitted URL cannot be analysed.

    The message is written for end users: it explains the problem without
    echoing back unsanitised input or exposing internals.
    """


@dataclass(frozen=True, slots=True)
class ValidatedURL:
    """A URL that passed validation.

    Attributes
    ----------
    raw:
        The cleaned string exactly as it will be analysed and stored.
    normalised:
        The analysed form: identical to ``raw`` but with a lower-cased scheme,
        and with ``http://`` prepended when the user omitted the scheme.
    host:
        The extracted hostname.
    scheme:
        ``http`` or ``https``.
    """

    raw: str
    normalised: str
    host: str
    scheme: str


def validate_url(url: object, max_length: int = 2048) -> ValidatedURL:
    """Validate and normalise ``url``.

    Parameters
    ----------
    url:
        Candidate value taken straight from a form field or JSON body.
    max_length:
        Upper bound on the accepted length; longer input is rejected instead
        of being truncated so the user knows the URL was not analysed.

    Raises
    ------
    URLValidationError
        With a user-facing message describing the first problem found.
    """
    if url is None:
        raise URLValidationError("Please enter a URL to analyse.")
    if not isinstance(url, str):
        raise URLValidationError("The URL must be provided as text.")

    cleaned = _CONTROL_CHARS.sub("", url).strip()
    if not cleaned:
        raise URLValidationError("Please enter a URL to analyse.")
    if len(cleaned) > max_length:
        raise URLValidationError(
            f"That URL is too long to analyse (limit {max_length} characters)."
        )
    if any(character.isspace() for character in cleaned):
        raise URLValidationError("A URL cannot contain spaces.")

    scheme_match = _SCHEME_RE.match(cleaned)
    if scheme_match:
        scheme = scheme_match.group(1).lower()
        if scheme not in _ALLOWED_SCHEMES:
            raise URLValidationError(
                "Only http:// and https:// URLs can be analysed."
            )
        candidate = cleaned
    else:
        # Users routinely paste bare domains; assume http:// for parsing only.
        scheme = "http"
        candidate = f"http://{cleaned}"

    try:
        parts = urlsplit(candidate)
        host = parts.hostname or ""
    except ValueError as exc:  # malformed IPv6 literal, bad port, ...
        raise URLValidationError("That URL is malformed and cannot be analysed.") from exc

    if not host:
        raise URLValidationError("That URL is missing a domain name.")
    if len(host) > 253:
        raise URLValidationError("That URL's domain name is too long to be valid.")
    if not _HOST_RE.match(host):
        raise URLValidationError("That URL contains an invalid domain name.")
    if host != host.strip("."):
        raise URLValidationError("That URL contains an invalid domain name.")
    if ".." in host:
        raise URLValidationError("That URL contains an invalid domain name.")

    is_ip_literal = host.startswith("[") or _looks_numeric(host)
    if "." not in host and not is_ip_literal and host != "localhost":
        raise URLValidationError(
            "That URL is missing a top-level domain (for example .com)."
        )

    # The analysed string keeps every component of the original input — the
    # port, userinfo and encoded characters are themselves phishing signals,
    # so nothing is dropped during normalisation.
    normalised = candidate if not scheme_match else scheme + cleaned[len(scheme_match.group(1)):]

    return ValidatedURL(raw=cleaned, normalised=normalised, host=host, scheme=scheme)


def _looks_numeric(host: str) -> bool:
    """Return ``True`` for dotted-quad or plain numeric hosts."""
    return all(part.isdigit() for part in host.split(".") if part) and any(
        character.isdigit() for character in host
    )
