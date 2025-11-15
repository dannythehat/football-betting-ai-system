"""
Pydantic schemas for data validation
Validates incoming JSON data from main app
"""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class TeamStatsSchema(BaseModel):
    """Team statistics at match time"""
    home_goals_avg: float = Field(..., ge=0, le=10)
    away_goals_avg: float = Field(..., ge=0, le=10)
    home_goals_conceded_avg: float = Field(..., ge=0, le=10)
    away_goals_conceded_avg: float = Field(..., ge=0, le=10)
    home_corners_avg: float = Field(..., ge=0, le=20)
    away_corners_avg: float = Field(..., ge=0, le=20)
    home_cards_avg: float = Field(..., ge=0, le=10)
    away_cards_avg: float = Field(..., ge=0, le=10)
    home_btts_rate: float = Field(..., ge=0, le=1)
    away_btts_rate: float = Field(..., ge=0, le=1)
    home_form: str = Field(..., min_length=5, max_length=5, pattern=r'^[WDL]{5}$')
    away_form: str = Field(..., min_length=5, max_length=5, pattern=r'^[WDL]{5}$')


class OddsSchema(BaseModel):
    """Betting odds for all markets"""
    # Match Result (1X2)
    home_win: float = Field(..., alias='home_win', ge=1.01)
    draw: float = Field(..., ge=1.01)
    away_win: float = Field(..., ge=1.01)
    
    # Total Goals
    over_0_5: float = Field(..., ge=1.01)
    under_0_5: float = Field(..., ge=1.01)
    over_1_5: float = Field(..., ge=1.01)
    under_1_5: float = Field(..., ge=1.01)
    over_2_5: float = Field(..., ge=1.01)
    under_2_5: float = Field(..., ge=1.01)
    over_3_5: float = Field(..., ge=1.01)
    under_3_5: float = Field(..., ge=1.01)
    over_4_5: float = Field(..., ge=1.01)
    under_4_5: float = Field(..., ge=1.01)
    
    # Both Teams To Score
    btts_yes: float = Field(..., ge=1.01)
    btts_no: float = Field(..., ge=1.01)
    
    # Double Chance
    home_or_draw: float = Field(..., ge=1.01)
    away_or_draw: float = Field(..., ge=1.01)
    home_or_away: float = Field(..., ge=1.01)
    
    # Corners
    corners_over_8_5: float = Field(..., ge=1.01)
    corners_under_8_5: float = Field(..., ge=1.01)
    corners_over_9_5: float = Field(..., ge=1.01)
    corners_under_9_5: float = Field(..., ge=1.01)
    corners_over_10_5: float = Field(..., ge=1.01)
    corners_under_10_5: float = Field(..., ge=1.01)
    
    # Cards
    cards_over_3_5: float = Field(..., ge=1.01)
    cards_under_3_5: float = Field(..., ge=1.01)
    cards_over_4_5: float = Field(..., ge=1.01)
    cards_under_4_5: float = Field(..., ge=1.01)

    class Config:
        populate_by_name = True


class MatchResultSchema(BaseModel):
    """Match result data (for historical matches)"""
    home_goals: int = Field(..., ge=0)
    away_goals: int = Field(..., ge=0)
    result: str = Field(..., pattern=r'^(home_win|draw|away_win)$')
    total_goals: int = Field(..., ge=0)
    home_corners: int = Field(..., ge=0)
    away_corners: int = Field(..., ge=0)
    total_corners: int = Field(..., ge=0)
    home_cards: int = Field(..., ge=0)
    away_cards: int = Field(..., ge=0)
    total_cards: int = Field(..., ge=0)
    btts: bool
    over_0_5: bool
    over_1_5: bool
    over_2_5: bool
    over_3_5: bool
    over_4_5: bool
    corners_over_8_5: bool
    corners_over_9_5: bool
    corners_over_10_5: bool
    cards_over_3_5: bool
    cards_over_4_5: bool


class MatchSchema(BaseModel):
    """Single match/fixture data"""
    match_id: str = Field(..., min_length=1, max_length=50)
    match_datetime: datetime
    league: str = Field(..., min_length=1, max_length=50)
    season: Optional[str] = Field(None, max_length=10)
    status: Optional[str] = Field('scheduled', pattern=r'^(scheduled|completed|cancelled)$')
    
    # Teams
    home_team_id: str = Field(..., min_length=1)
    home_team: str = Field(..., min_length=1, max_length=100)
    away_team_id: str = Field(..., min_length=1)
    away_team: str = Field(..., min_length=1, max_length=100)
    
    # Stats at match time
    team_stats_at_match_time: TeamStatsSchema
    
    # Odds
    odds: OddsSchema
    
    # Result (only for completed matches)
    result: Optional[MatchResultSchema] = None

    @validator('season', always=True)
    def set_season(cls, v, values):
        """Auto-generate season from match_datetime if not provided"""
        if v is None and 'match_datetime' in values:
            dt = values['match_datetime']
            year = dt.year
            # Season runs Aug-May, so if month >= 8, season is year/year+1
            if dt.month >= 8:
                return f"{year}-{str(year + 1)[-2:]}"
            else:
                return f"{year - 1}-{str(year)[-2:]}"
        return v


class BatchIngestRequest(BaseModel):
    """Batch data ingestion request"""
    matches: list[MatchSchema] = Field(..., min_items=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "matches": [{
                    "match_id": "12345",
                    "match_datetime": "2025-11-15T14:00:00Z",
                    "league": "Premier League",
                    "home_team_id": "team_001",
                    "home_team": "Team A",
                    "away_team_id": "team_002",
                    "away_team": "Team B",
                    "team_stats_at_match_time": {
                        "home_goals_avg": 1.4,
                        "away_goals_avg": 1.1,
                        "home_goals_conceded_avg": 0.8,
                        "away_goals_conceded_avg": 1.2,
                        "home_corners_avg": 5.2,
                        "away_corners_avg": 4.8,
                        "home_cards_avg": 2.1,
                        "away_cards_avg": 2.3,
                        "home_btts_rate": 0.6,
                        "away_btts_rate": 0.5,
                        "home_form": "WWDWL",
                        "away_form": "LDWDL"
                    },
                    "odds": {
                        "home_win": 1.95,
                        "draw": 3.5,
                        "away_win": 4.0
                    }
                }]
            }
        }


class IngestResponse(BaseModel):
    """Response from data ingestion"""
    success: bool
    message: str
    matches_processed: int
    matches_created: int
    matches_updated: int
    errors: list[str] = []
