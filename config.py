"""Application configuration.

Configuration is environment driven: every deployment-specific value can be
overridden through environment variables (optionally loaded from a local
``.env`` file), which keeps secrets out of the source tree.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Type

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - exercised only without python-dotenv
    pass


BASE_DIR: Path = Path(__file__).resolve().parent

DATA_DIR: Path = BASE_DIR / "data"
MODEL_DIR: Path = BASE_DIR / "models"
LOG_DIR: Path = BASE_DIR / "logs"


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean flag from the environment."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """Read an integer from the environment, falling back to ``default``."""
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


class BaseConfig:
    """Settings shared by every environment."""

    # --- Flask core ---------------------------------------------------
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-me")
    JSON_SORT_KEYS: bool = False

    # --- Database -----------------------------------------------------
    SQLALCHEMY_DATABASE_URI: str = os.getenv(
        "DATABASE_URL", f"sqlite:///{BASE_DIR / 'phishing.db'}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False

    # --- Paths --------------------------------------------------------
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = DATA_DIR
    MODEL_DIR: Path = MODEL_DIR
    LOG_DIR: Path = LOG_DIR
    DATASET_PATH: Path = Path(os.getenv("DATASET_PATH", DATA_DIR / "phishing_urls.csv"))
    MODEL_PATH: Path = Path(os.getenv("MODEL_PATH", MODEL_DIR / "phishing_model.pkl"))
    SCALER_PATH: Path = Path(os.getenv("SCALER_PATH", MODEL_DIR / "scaler.pkl"))
    METRICS_PATH: Path = Path(os.getenv("METRICS_PATH", MODEL_DIR / "metrics.json"))

    # --- Input security ------------------------------------------------
    MAX_URL_LENGTH: int = _env_int("MAX_URL_LENGTH", 2048)
    MAX_CONTENT_LENGTH: int = _env_int("MAX_CONTENT_LENGTH", 64 * 1024)

    # --- Pagination ----------------------------------------------------
    HISTORY_PAGE_SIZE: int = _env_int("HISTORY_PAGE_SIZE", 15)
    API_MAX_PAGE_SIZE: int = _env_int("API_MAX_PAGE_SIZE", 100)

    # --- Decision thresholds -------------------------------------------
    # The classifier is binary (legitimate vs. phishing). These thresholds map
    # the predicted phishing probability onto the three product-level verdicts.
    # See README -> "From probability to verdict" for the rationale.
    SUSPICIOUS_THRESHOLD: float = float(os.getenv("SUSPICIOUS_THRESHOLD", "0.35"))
    PHISHING_THRESHOLD: float = float(os.getenv("PHISHING_THRESHOLD", "0.70"))

    # --- Logging --------------------------------------------------------
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    DEBUG: bool = False
    TESTING: bool = False


class DevelopmentConfig(BaseConfig):
    """Local development: verbose errors and auto-reload friendly."""

    DEBUG: bool = True
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG").upper()


class TestingConfig(BaseConfig):
    """Used by the pytest suite: in-memory database, no side effects."""

    TESTING: bool = True
    DEBUG: bool = False
    SQLALCHEMY_DATABASE_URI: str = "sqlite:///:memory:"
    LOG_LEVEL: str = "WARNING"


class ProductionConfig(BaseConfig):
    """Production: never leak stack traces, require a real secret key."""

    DEBUG: bool = False
    TESTING: bool = False
    PROPAGATE_EXCEPTIONS: bool = False


_CONFIG_BY_NAME: dict[str, Type[BaseConfig]] = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}


def get_config(name: str | None = None) -> Type[BaseConfig]:
    """Return the configuration class for ``name`` (default: ``FLASK_ENV``)."""
    key = (name or os.getenv("FLASK_ENV", "development")).strip().lower()
    return _CONFIG_BY_NAME.get(key, DevelopmentConfig)
