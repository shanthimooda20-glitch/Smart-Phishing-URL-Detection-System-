"""Application factory for the Smart Phishing URL Detection System."""

from __future__ import annotations

import logging
from typing import Any, Type

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from config import BaseConfig, get_config
from src.database.models import db
from src.ml.predictor import PhishingPredictor
from src.utils.logging_config import configure_logging

LOGGER = logging.getLogger(__name__)

__all__ = ["create_app", "get_predictor"]


def create_app(config_object: Type[BaseConfig] | str | None = None) -> Flask:
    """Build and configure the Flask application.

    Parameters
    ----------
    config_object:
        A configuration class, an environment name (``"testing"``) or ``None``
        to select the environment from ``FLASK_ENV``.
    """
    config: Type[BaseConfig] = (
        config_object if isinstance(config_object, type) else get_config(config_object)
    )

    app = Flask(
        __name__,
        template_folder=str(config.BASE_DIR / "templates"),
        static_folder=str(config.BASE_DIR / "static"),
    )
    app.config.from_object(config)

    configure_logging(app.config["LOG_LEVEL"], app.config["LOG_DIR"])

    db.init_app(app)
    with app.app_context():
        db.create_all()

    app.extensions["predictor"] = PhishingPredictor(
        model_path=app.config["MODEL_PATH"],
        suspicious_threshold=app.config["SUSPICIOUS_THRESHOLD"],
        phishing_threshold=app.config["PHISHING_THRESHOLD"],
    )

    from src.routes.api import api_bp
    from src.routes.web import web_bp

    app.register_blueprint(web_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    _register_error_handlers(app)
    _register_template_helpers(app)

    LOGGER.info(
        "Application ready (env=%s, model=%s)",
        config.__name__,
        "loaded" if app.extensions["predictor"].is_ready else "MISSING - run training",
    )
    return app


def get_predictor() -> PhishingPredictor:
    """Return the predictor bound to the current application."""
    from flask import current_app

    return current_app.extensions["predictor"]


def _wants_json() -> bool:
    """``True`` when the client asked for JSON (API route or Accept header)."""
    if request.path.startswith("/api/"):
        return True
    return request.accept_mimetypes.best == "application/json"


def _register_error_handlers(app: Flask) -> None:
    """Install handlers that never leak stack traces to the client."""

    @app.errorhandler(400)
    @app.errorhandler(404)
    @app.errorhandler(405)
    @app.errorhandler(413)
    def handle_client_error(error: HTTPException):  # type: ignore[no-untyped-def]
        message = getattr(error, "description", "Request could not be processed.")
        if _wants_json():
            return jsonify({"error": error.name, "message": message}), error.code
        return render_template(
            "errors/error.html", code=error.code, title=error.name, message=message
        ), error.code

    @app.errorhandler(Exception)
    def handle_unexpected_error(error: Exception):  # type: ignore[no-untyped-def]
        if isinstance(error, HTTPException):
            return handle_client_error(error)
        # The full traceback goes to the log; the user gets a neutral message.
        LOGGER.exception("Unhandled application error")
        message = "Something went wrong while processing your request."
        if _wants_json():
            return jsonify({"error": "Internal Server Error", "message": message}), 500
        return render_template(
            "errors/error.html", code=500, title="Internal Server Error",
            message=message,
        ), 500


def _register_template_helpers(app: Flask) -> None:
    """Expose small formatting helpers to the Jinja templates."""

    @app.template_filter("verdict_class")
    def verdict_class(verdict: str) -> str:
        """Map a verdict onto its CSS modifier."""
        return {
            "Safe": "safe", "Suspicious": "suspicious", "Phishing": "phishing"
        }.get(str(verdict), "unknown")

    @app.template_filter("risk_class")
    def risk_class(level: str) -> str:
        """Map a risk level onto its CSS modifier."""
        return str(level).lower()

    @app.template_filter("pretty_time")
    def pretty_time(value: Any) -> str:
        """Render an ISO-8601 timestamp as ``YYYY-MM-DD HH:MM UTC``."""
        from datetime import datetime, timezone

        if not value:
            return "unknown"
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value)
            except ValueError:
                return value
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc)
        return value.strftime("%Y-%m-%d %H:%M UTC")

    @app.context_processor
    def inject_globals() -> dict[str, Any]:
        """Values every template can rely on."""
        return {
            "app_name": "Smart Phishing URL Detection",
            "model_ready": app.extensions["predictor"].is_ready,
        }
