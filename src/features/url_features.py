"""Lexical and host-based feature extraction for URLs.

The module is deliberately *offline*: it never resolves DNS, never opens a
socket and never fetches the submitted URL. Every feature is derived from the
URL string itself, which makes analysing a hostile URL completely safe.

Design notes
------------
**Canonicalisation.** Public URL corpora are usually stored in a normalised
form — no ``http://`` prefix and no ``www.`` label — while users always type
both. Feeding the raw string straight into the model would therefore shift
every length and character count between training and serving. Two
normalisations are applied identically in both phases:

* the scheme is stripped and recorded as its own boolean feature
  (``has_https``) rather than being counted as characters;
* a leading ``www.`` label is removed, because it is a fixed prefix that
  carries no information about whether a site is malicious.

**Stable ordering.** :data:`FEATURE_NAMES` is the single source of truth for
the column order. Both the training pipeline and the prediction service build
their matrices through :func:`features_to_vector`, so a model can never be fed
columns in the wrong order.
"""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence
from urllib.parse import urlsplit

__all__ = [
    "FEATURE_NAMES",
    "SUSPICIOUS_KEYWORDS",
    "SUSPICIOUS_TLDS",
    "SHORTENER_DOMAINS",
    "extract_features",
    "extract_features_batch",
    "features_to_vector",
    "normalise_url",
    "split_host",
]


# --------------------------------------------------------------------------
# Reference data
# --------------------------------------------------------------------------

#: Tokens that repeatedly show up in credential-harvesting URLs.
SUSPICIOUS_KEYWORDS: tuple[str, ...] = (
    "account", "admin", "alert", "authenticate", "backup", "banking", "bonus",
    "confirm", "credential", "customer", "ebayisapi", "expire", "free", "gift",
    "invoice", "limited", "live", "locked", "login", "logon", "mail", "office",
    "password", "pay", "paypal", "recover", "refund", "safe", "secure",
    "security", "server", "signin", "submit", "support", "suspend", "unlock",
    "update", "urgent", "user", "validate", "verify", "wallet", "webscr",
    "wp-admin", "wp-content", "wp-includes",
)

#: TLDs that are cheap/free to register and are heavily abused in phishing.
SUSPICIOUS_TLDS: frozenset[str] = frozenset({
    "bid", "buzz", "cam", "cf", "click", "country", "cricket", "date", "download",
    "faith", "fit", "ga", "gdn", "gq", "host", "icu", "kim", "link", "loan", "men",
    "ml", "mom", "monster", "party", "pw", "quest", "racing", "ren", "review",
    "rest", "science", "shop", "space", "stream", "surf", "tk", "top", "trade",
    "webcam", "win", "work", "xin", "xyz", "zip",
})

#: URL shorteners hide the real destination behind an opaque token.
SHORTENER_DOMAINS: frozenset[str] = frozenset({
    "adf.ly", "bit.do", "bit.ly", "bitly.com", "buff.ly", "cutt.ly", "db.tt",
    "goo.gl", "is.gd", "j.mp", "lnkd.in", "ow.ly", "po.st", "q.gs", "rb.gy",
    "rebrand.ly", "s.id", "shorte.st", "t.co", "tiny.cc", "tinyurl.com",
    "tr.im", "u.to", "v.gd", "x.co",
})

#: Second-level suffixes that must not be mistaken for a subdomain.
_COMPOUND_SUFFIXES: frozenset[str] = frozenset({
    "ac.uk", "co.uk", "gov.uk", "org.uk", "me.uk", "net.uk", "sch.uk",
    "co.jp", "or.jp", "ne.jp", "ac.jp", "go.jp",
    "com.au", "net.au", "org.au", "edu.au", "gov.au",
    "com.br", "com.cn", "net.cn", "org.cn", "gov.cn", "edu.cn",
    "co.in", "net.in", "org.in", "gov.in", "ac.in", "edu.in",
    "co.za", "com.mx", "com.ar", "com.tr", "com.sg", "com.hk", "com.tw",
    "co.kr", "co.nz", "co.id", "co.il", "com.pk", "com.my", "com.ph",
})

#: Characters treated as "special" for the special-character density feature.
_SPECIAL_CHARS: frozenset[str] = frozenset("-_.@?&=%+~#$!*,:;()[]{}|^'\"<>\\ ")

_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_PERCENT_RE = re.compile(r"%[0-9a-fA-F]{2}")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_HEX_IP_RE = re.compile(r"^0x[0-9a-fA-F]{8}$")
_TOKEN_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")

#: Canonical column order. Never reorder — retrain instead.
FEATURE_NAMES: tuple[str, ...] = (
    "url_length",
    "domain_length",
    "path_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_special_chars",
    "num_digits",
    "num_subdomains",
    "has_https",
    "has_ip_address",
    "has_at_symbol",
    "has_double_slash_redirect",
    "has_port",
    "num_suspicious_keywords",
    "num_query_params",
    "num_fragments",
    "num_percent_encodings",
    "domain_entropy",
    "url_entropy",
    "digit_ratio",
    "letter_ratio",
    "longest_token_length",
    "tld_length",
    "is_suspicious_tld",
    "is_shortened",
    "has_hyphen_in_domain",
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def shannon_entropy(text: str) -> float:
    """Return the Shannon entropy (bits per character) of ``text``.

    Randomly generated hostnames such as ``xk3l9zq2p.example`` have a much
    higher entropy than pronounceable brand names, which makes this a useful
    signal for algorithmically generated domains.
    """
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum(
        (count / length) * math.log2(count / length) for count in counts.values()
    )


def normalise_url(url: str) -> tuple[str, str]:
    """Split ``url`` into ``(scheme, remainder)``.

    The scheme is removed from the remainder so that character counts are
    comparable between corpora that store schemes and corpora that do not.

    >>> normalise_url("https://example.com/a")
    ('https', 'example.com/a')
    >>> normalise_url("example.com/a")
    ('', 'example.com/a')
    """
    stripped = (url or "").strip()
    match = _SCHEME_RE.match(stripped)
    if match:
        scheme = match.group(0)[:-3].lower()
        return scheme, stripped[match.end():]
    return "", stripped


def _strip_www(host: str) -> str:
    """Remove a leading ``www.`` label from ``host``."""
    return host[4:] if host.lower().startswith("www.") else host


def split_host(host: str) -> tuple[str, str, str]:
    """Split a hostname into ``(subdomain, registrable_domain, tld)``.

    A small compound-suffix table keeps ``bbc.co.uk`` from being read as the
    subdomain ``bbc`` of the domain ``co.uk``. A full public-suffix list would
    be more precise, but it adds a heavyweight dependency for a marginal gain
    on the lexical features used here.
    """
    host = host.strip(".").lower()
    if not host:
        return "", "", ""

    labels = host.split(".")
    if len(labels) == 1:
        return "", labels[0], ""

    suffix_len = 2 if ".".join(labels[-2:]) in _COMPOUND_SUFFIXES else 1
    if len(labels) <= suffix_len:
        return "", host, labels[-1]

    tld = ".".join(labels[-suffix_len:])
    domain = ".".join(labels[-(suffix_len + 1):])
    subdomain = ".".join(labels[: -(suffix_len + 1)])
    return subdomain, domain, tld


def _looks_like_ip(host: str) -> bool:
    """Return ``True`` when ``host`` is an IP literal rather than a name."""
    candidate = host.strip("[]")
    try:
        ipaddress.ip_address(candidate)
        return True
    except ValueError:
        pass
    if _HEX_IP_RE.match(candidate):  # hexadecimal obfuscation, e.g. 0xC0A80001
        return True
    return bool(_IPV4_RE.fullmatch(candidate))


def _longest_token(text: str) -> int:
    """Length of the longest alphanumeric run in ``text``."""
    tokens = [token for token in _TOKEN_SPLIT_RE.split(text) if token]
    return max((len(token) for token in tokens), default=0)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def extract_features(url: str) -> Dict[str, float]:
    """Extract the full feature dictionary for a single ``url``.

    Parameters
    ----------
    url:
        The raw URL string. The scheme is optional.

    Returns
    -------
    dict
        Mapping of feature name to numeric value, ordered exactly like
        :data:`FEATURE_NAMES`.

    Notes
    -----
    The function is total: any string — including malformed input — yields a
    complete feature dictionary rather than raising, so a single bad row can
    never abort a training run.
    """
    scheme, remainder = normalise_url(url)
    # Canonicalise away the ``www.`` prefix (see module docstring).
    if remainder[:4].lower() == "www.":
        remainder = remainder[4:]

    # ``urlsplit`` needs a scheme to populate ``netloc``; a placeholder is used
    # so that user input and dataset rows are parsed by the same code path.
    try:
        parts = urlsplit("//" + remainder, scheme="http")
        host = parts.hostname or ""
        port = parts.port
        path, query, fragment = parts.path, parts.query, parts.fragment
    except ValueError:
        # Malformed netloc (e.g. an invalid port); fall back to a manual split.
        host_part, _, tail = remainder.partition("/")
        host = host_part.split("@")[-1].split(":")[0].lower()
        port = None
        path, _, rest = ("/" + tail).partition("?")
        query, _, fragment = rest.partition("#")

    host_no_www = _strip_www(host)
    subdomain, domain, tld = split_host(host)
    lowered = remainder.lower()

    digits = sum(character.isdigit() for character in remainder)
    letters = sum(character.isalpha() for character in remainder)
    total = len(remainder) or 1

    features: Dict[str, float] = {
        "url_length": float(len(remainder)),
        "domain_length": float(len(host)),
        "path_length": float(len(path)),
        "num_dots": float(remainder.count(".")),
        "num_hyphens": float(remainder.count("-")),
        "num_underscores": float(remainder.count("_")),
        "num_slashes": float(remainder.count("/")),
        "num_special_chars": float(
            sum(character in _SPECIAL_CHARS for character in remainder)
        ),
        "num_digits": float(digits),
        "num_subdomains": float(
            len([label for label in subdomain.split(".") if label and label != "www"])
        ),
        "has_https": float(scheme == "https"),
        "has_ip_address": float(_looks_like_ip(host)),
        "has_at_symbol": float("@" in remainder),
        "has_double_slash_redirect": float("//" in path),
        "has_port": float(port is not None),
        "num_suspicious_keywords": float(
            sum(keyword in lowered for keyword in SUSPICIOUS_KEYWORDS)
        ),
        "num_query_params": float(len([p for p in query.split("&") if p])),
        "num_fragments": float(len([f for f in fragment.split("#") if f])),
        "num_percent_encodings": float(len(_PERCENT_RE.findall(remainder))),
        "domain_entropy": round(shannon_entropy(host_no_www), 4),
        "url_entropy": round(shannon_entropy(remainder), 4),
        "digit_ratio": round(digits / total, 4),
        "letter_ratio": round(letters / total, 4),
        "longest_token_length": float(_longest_token(remainder)),
        "tld_length": float(len(tld)),
        "is_suspicious_tld": float(tld.rsplit(".", 1)[-1] in SUSPICIOUS_TLDS),
        "is_shortened": float(host_no_www in SHORTENER_DOMAINS),
        "has_hyphen_in_domain": float("-" in domain),
    }
    return {name: features[name] for name in FEATURE_NAMES}


def extract_features_batch(urls: Iterable[str]) -> List[Dict[str, float]]:
    """Extract features for many URLs, preserving input order."""
    return [extract_features(url) for url in urls]


def features_to_vector(
    features: Dict[str, Any], feature_names: Sequence[str] | None = None
) -> List[float]:
    """Flatten a feature dictionary into the canonical model input order.

    ``feature_names`` allows a persisted model to pin the exact subset and
    order of columns it was trained on, which protects predictions from any
    later change to :data:`FEATURE_NAMES`.
    """
    names = tuple(feature_names) if feature_names is not None else FEATURE_NAMES
    return [float(features.get(name, 0.0)) for name in names]
