"""Shared pytest fixtures.

Tests that need a trained model are skipped automatically when
``models/phishing_model.pkl`` is absent, so a fresh clone can run the suite
before the training script has been executed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import BaseConfig, TestingConfig  # noqa: E402
from src import create_app  # noqa: E402
from src.database.models import db  # noqa: E402

MODEL_PATH = BaseConfig.MODEL_PATH

requires_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="No trained model; run `python src/ml/train.py` first.",
)


@pytest.fixture()
def app() -> Iterator:
    """A Flask app bound to a throw-away in-memory database."""
    application = create_app(TestingConfig)
    with application.app_context():
        db.drop_all()
        db.create_all()
        yield application
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    """Flask test client for the API and web routes."""
    return app.test_client()


@pytest.fixture()
def predictor(app):
    """The application's prediction service."""
    return app.extensions["predictor"]
