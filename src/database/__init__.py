"""Database package: SQLAlchemy models and the analysis repository."""

from src.database.models import URLAnalysis, db
from src.database.repository import (
    DatabaseError,
    query_history,
    record_analysis,
)

__all__ = [
    "db",
    "URLAnalysis",
    "DatabaseError",
    "record_analysis",
    "query_history",
]
