"""Centralised logging setup.

Logs go to both stdout (useful in containers) and a rotating file so that a
long-running deployment cannot fill the disk.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_MAX_BYTES = 2 * 1024 * 1024
_BACKUP_COUNT = 3


def configure_logging(level: str = "INFO", log_dir: Path | None = None) -> None:
    """Configure the root logger exactly once.

    Parameters
    ----------
    level:
        Logging level name, e.g. ``"DEBUG"`` or ``"INFO"``.
    log_dir:
        Optional directory for ``app.log``. File logging is skipped silently
        when the directory is not writable (read-only containers).
    """
    root = logging.getLogger()
    if getattr(root, "_phishing_logging_configured", False):
        return

    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(_LOG_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_dir is not None:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_dir / "app.log", maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT
            )
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError:  # pragma: no cover - depends on the filesystem
            root.warning("File logging disabled: %s is not writable", log_dir)

    root._phishing_logging_configured = True  # type: ignore[attr-defined]
