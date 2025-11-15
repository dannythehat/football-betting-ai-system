"""
Data Ingestion Module
Handles incoming match data from main application
"""

from .models import Team, Match, MatchOdds, MatchResult, Prediction
from .schemas import (
    TeamStatsSchema,
    OddsSchema,
    MatchResultSchema,
    MatchSchema,
    BatchIngestRequest,
    IngestResponse
)
from .database import get_db, get_db_session, init_db, drop_db
from .ingestion import DataIngestionService

__all__ = [
    'Team',
    'Match',
    'MatchOdds',
    'MatchResult',
    'Prediction',
    'TeamStatsSchema',
    'OddsSchema',
    'MatchResultSchema',
    'MatchSchema',
    'BatchIngestRequest',
    'IngestResponse',
    'get_db',
    'get_db_session',
    'init_db',
    'drop_db',
    'DataIngestionService'
]
