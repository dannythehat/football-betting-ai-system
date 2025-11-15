"""
Database models for Football Betting AI System
SQLAlchemy ORM models matching the PostgreSQL schema
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, DateTime, Decimal, Boolean, 
    ForeignKey, Text, Index
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Team(Base):
    __tablename__ = 'teams'
    
    team_id = Column(Integer, primary_key=True, autoincrement=True)
    team_name = Column(String(100), nullable=False, unique=True)
    league = Column(String(50))
    tier = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    statistics = relationship("TeamStatistic", back_populates="team", cascade="all, delete-orphan")
    home_matches = relationship("Match", foreign_keys="Match.home_team_id", back_populates="home_team")
    away_matches = relationship("Match", foreign_keys="Match.away_team_id", back_populates="away_team")


class TeamStatistic(Base):
    __tablename__ = 'team_statistics'
    
    stat_id = Column(Integer, primary_key=True, autoincrement=True)
    team_id = Column(Integer, ForeignKey('teams.team_id', ondelete='CASCADE'))
    season = Column(String(10))
    
    # Overall stats
    matches_played = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    draws = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    goals_scored = Column(Integer, default=0)
    goals_conceded = Column(Integer, default=0)
    
    # Home stats
    home_matches = Column(Integer, default=0)
    home_wins = Column(Integer, default=0)
    home_draws = Column(Integer, default=0)
    home_losses = Column(Integer, default=0)
    home_goals_scored = Column(Integer, default=0)
    home_goals_conceded = Column(Integer, default=0)
    
    # Away stats
    away_matches = Column(Integer, default=0)
    away_wins = Column(Integer, default=0)
    away_draws = Column(Integer, default=0)
    away_losses = Column(Integer, default=0)
    away_goals_scored = Column(Integer, default=0)
    away_goals_conceded = Column(Integer, default=0)
    
    # Averages
    goals_avg = Column(Decimal(3, 2))
    goals_conceded_avg = Column(Decimal(3, 2))
    home_goals_avg = Column(Decimal(3, 2))
    away_goals_avg = Column(Decimal(3, 2))
    corners_avg = Column(Decimal(4, 2))
    cards_avg = Column(Decimal(3, 2))
    
    # Rates
    btts_rate = Column(Decimal(3, 2))
    clean_sheet_rate = Column(Decimal(3, 2))
    
    # Form
    current_form = Column(String(5))
    
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    team = relationship("Team", back_populates="statistics")
    
    __table_args__ = (
        Index('idx_team_stats_team_season', 'team_id', 'season', unique=True),
    )


class Match(Base):
    __tablename__ = 'matches'
    
    match_id = Column(String(50), primary_key=True)
    home_team_id = Column(Integer, ForeignKey('teams.team_id'))
    away_team_id = Column(Integer, ForeignKey('teams.team_id'))
    match_datetime = Column(DateTime, nullable=False)
    league = Column(String(50))
    season = Column(String(10))
    status = Column(String(20), default='scheduled')
    
    # Team stats snapshot at match time
    home_goals_avg = Column(Decimal(3, 2))
    away_goals_avg = Column(Decimal(3, 2))
    home_goals_conceded_avg = Column(Decimal(3, 2))
    away_goals_conceded_avg = Column(Decimal(3, 2))
    home_corners_avg = Column(Decimal(4, 2))
    away_corners_avg = Column(Decimal(4, 2))
    home_cards_avg = Column(Decimal(3, 2))
    away_cards_avg = Column(Decimal(3, 2))
    home_btts_rate = Column(Decimal(3, 2))
    away_btts_rate = Column(Decimal(3, 2))
    home_form = Column(String(5))
    away_form = Column(String(5))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_matches")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_matches")
    result = relationship("MatchResult", back_populates="match", uselist=False, cascade="all, delete-orphan")
    odds = relationship("MatchOdds", back_populates="match", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="match", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_matches_datetime', 'match_datetime'),
        Index('idx_matches_status', 'status'),
        Index('idx_matches_home_team', 'home_team_id'),
        Index('idx_matches_away_team', 'away_team_id'),
    )


class MatchResult(Base):
    __tablename__ = 'match_results'
    
    result_id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(50), ForeignKey('matches.match_id', ondelete='CASCADE'))
    
    # Final score
    home_goals = Column(Integer, nullable=False)
    away_goals = Column(Integer, nullable=False)
    result = Column(String(10))
    
    # Additional stats
    total_goals = Column(Integer)
    home_corners = Column(Integer)
    away_corners = Column(Integer)
    total_corners = Column(Integer)
    home_cards = Column(Integer)
    away_cards = Column(Integer)
    total_cards = Column(Integer)
    
    # Market outcomes
    btts = Column(Boolean)
    over_0_5 = Column(Boolean)
    over_1_5 = Column(Boolean)
    over_2_5 = Column(Boolean)
    over_3_5 = Column(Boolean)
    over_4_5 = Column(Boolean)
    corners_over_8_5 = Column(Boolean)
    corners_over_9_5 = Column(Boolean)
    corners_over_10_5 = Column(Boolean)
    cards_over_3_5 = Column(Boolean)
    cards_over_4_5 = Column(Boolean)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    match = relationship("Match", back_populates="result")
    
    __table_args__ = (
        Index('idx_match_results_match_id', 'match_id'),
    )


class MatchOdds(Base):
    __tablename__ = 'match_odds'
    
    odds_id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(50), ForeignKey('matches.match_id', ondelete='CASCADE'))
    odds_timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Match Result (1X2)
    home_win_odds = Column(Decimal(5, 2))
    draw_odds = Column(Decimal(5, 2))
    away_win_odds = Column(Decimal(5, 2))
    
    # Total Goals Over/Under
    over_0_5_odds = Column(Decimal(5, 2))
    under_0_5_odds = Column(Decimal(5, 2))
    over_1_5_odds = Column(Decimal(5, 2))
    under_1_5_odds = Column(Decimal(5, 2))
    over_2_5_odds = Column(Decimal(5, 2))
    under_2_5_odds = Column(Decimal(5, 2))
    over_3_5_odds = Column(Decimal(5, 2))
    under_3_5_odds = Column(Decimal(5, 2))
    over_4_5_odds = Column(Decimal(5, 2))
    under_4_5_odds = Column(Decimal(5, 2))
    
    # Both Teams To Score
    btts_yes_odds = Column(Decimal(5, 2))
    btts_no_odds = Column(Decimal(5, 2))
    
    # Double Chance
    home_or_draw_odds = Column(Decimal(5, 2))
    away_or_draw_odds = Column(Decimal(5, 2))
    home_or_away_odds = Column(Decimal(5, 2))
    
    # Corners
    corners_over_8_5_odds = Column(Decimal(5, 2))
    corners_under_8_5_odds = Column(Decimal(5, 2))
    corners_over_9_5_odds = Column(Decimal(5, 2))
    corners_under_9_5_odds = Column(Decimal(5, 2))
    corners_over_10_5_odds = Column(Decimal(5, 2))
    corners_under_10_5_odds = Column(Decimal(5, 2))
    
    # Cards
    cards_over_3_5_odds = Column(Decimal(5, 2))
    cards_under_3_5_odds = Column(Decimal(5, 2))
    cards_over_4_5_odds = Column(Decimal(5, 2))
    cards_under_4_5_odds = Column(Decimal(5, 2))
    
    # Metadata
    bookmaker = Column(String(50), default='test_bookmaker')
    is_latest = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    match = relationship("Match", back_populates="odds")
    
    __table_args__ = (
        Index('idx_match_odds_match_id', 'match_id'),
        Index('idx_match_odds_latest', 'match_id', 'is_latest'),
    )


class Prediction(Base):
    __tablename__ = 'predictions'
    
    prediction_id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(50), ForeignKey('matches.match_id', ondelete='CASCADE'))
    
    # Prediction metadata
    model_version = Column(String(20))
    prediction_timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Golden Bets
    is_golden_bet = Column(Boolean, default=False)
    golden_bet_market = Column(String(50))
    golden_bet_selection = Column(String(50))
    golden_bet_probability = Column(Decimal(4, 3))
    golden_bet_confidence = Column(String(20))
    
    # Value Bets
    is_value_bet = Column(Boolean, default=False)
    value_bet_market = Column(String(50))
    value_bet_selection = Column(String(50))
    value_bet_ai_probability = Column(Decimal(4, 3))
    value_bet_implied_probability = Column(Decimal(4, 3))
    value_bet_value = Column(Decimal(4, 3))
    
    # Smart Bet
    smart_bet_market = Column(String(50))
    smart_bet_selection = Column(String(50))
    smart_bet_probability = Column(Decimal(4, 3))
    
    # All probabilities (JSON)
    all_probabilities = Column(JSONB)
    
    # Explanations
    explanation = Column(Text)
    key_factors = Column(JSONB)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    match = relationship("Match", back_populates="predictions")
    
    __table_args__ = (
        Index('idx_predictions_match_id', 'match_id'),
    )
