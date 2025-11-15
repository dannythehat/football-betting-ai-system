"""
Core data ingestion logic
Processes incoming match data and stores in database
"""

from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import List, Tuple

from .models import Team, Match, MatchOdds, MatchResult
from .schemas import MatchSchema, BatchIngestRequest, IngestResponse


class DataIngestionService:
    """Service for ingesting match data into database"""
    
    def __init__(self, db: Session):
        self.db = db
        self.errors = []
    
    def ingest_batch(self, request: BatchIngestRequest) -> IngestResponse:
        """
        Ingest a batch of matches
        
        Args:
            request: BatchIngestRequest with list of matches
            
        Returns:
            IngestResponse with processing statistics
        """
        matches_created = 0
        matches_updated = 0
        
        for match_data in request.matches:
            try:
                created = self._process_match(match_data)
                if created:
                    matches_created += 1
                else:
                    matches_updated += 1
            except Exception as e:
                self.errors.append(f"Match {match_data.match_id}: {str(e)}")
        
        # Commit all changes
        try:
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            return IngestResponse(
                success=False,
                message=f"Database commit failed: {str(e)}",
                matches_processed=0,
                matches_created=0,
                matches_updated=0,
                errors=self.errors
            )
        
        return IngestResponse(
            success=len(self.errors) == 0,
            message="Batch ingestion completed" if len(self.errors) == 0 else "Batch ingestion completed with errors",
            matches_processed=len(request.matches),
            matches_created=matches_created,
            matches_updated=matches_updated,
            errors=self.errors
        )
    
    def _process_match(self, match_data: MatchSchema) -> bool:
        """
        Process a single match
        
        Returns:
            True if created, False if updated
        """
        # 1. Ensure teams exist
        home_team = self._get_or_create_team(
            match_data.home_team_id,
            match_data.home_team,
            match_data.league
        )
        away_team = self._get_or_create_team(
            match_data.away_team_id,
            match_data.away_team,
            match_data.league
        )
        
        # 2. Create or update match
        match = self.db.query(Match).filter(Match.match_id == match_data.match_id).first()
        
        if match is None:
            # Create new match
            match = Match(
                match_id=match_data.match_id,
                home_team_id=home_team.team_id,
                away_team_id=away_team.team_id,
                match_datetime=match_data.match_datetime,
                league=match_data.league,
                season=match_data.season,
                status=match_data.status,
                # Team stats snapshot
                home_goals_avg=match_data.team_stats_at_match_time.home_goals_avg,
                away_goals_avg=match_data.team_stats_at_match_time.away_goals_avg,
                home_goals_conceded_avg=match_data.team_stats_at_match_time.home_goals_conceded_avg,
                away_goals_conceded_avg=match_data.team_stats_at_match_time.away_goals_conceded_avg,
                home_corners_avg=match_data.team_stats_at_match_time.home_corners_avg,
                away_corners_avg=match_data.team_stats_at_match_time.away_corners_avg,
                home_cards_avg=match_data.team_stats_at_match_time.home_cards_avg,
                away_cards_avg=match_data.team_stats_at_match_time.away_cards_avg,
                home_btts_rate=match_data.team_stats_at_match_time.home_btts_rate,
                away_btts_rate=match_data.team_stats_at_match_time.away_btts_rate,
                home_form=match_data.team_stats_at_match_time.home_form,
                away_form=match_data.team_stats_at_match_time.away_form
            )
            self.db.add(match)
            self.db.flush()  # Get match_id
            created = True
        else:
            # Update existing match
            match.match_datetime = match_data.match_datetime
            match.status = match_data.status
            match.updated_at = datetime.utcnow()
            created = False
        
        # 3. Add/update odds
        self._process_odds(match.match_id, match_data.odds)
        
        # 4. Add result if match is completed
        if match_data.result is not None:
            self._process_result(match.match_id, match_data.result)
        
        return created
    
    def _get_or_create_team(self, team_id: str, team_name: str, league: str) -> Team:
        """Get existing team or create new one"""
        # Try to find by team_name (unique constraint)
        team = self.db.query(Team).filter(Team.team_name == team_name).first()
        
        if team is None:
            team = Team(
                team_name=team_name,
                league=league,
                tier='mid'  # Default tier
            )
            self.db.add(team)
            self.db.flush()  # Get team_id
        
        return team
    
    def _process_odds(self, match_id: str, odds_data):
        """Process and store odds data"""
        # Mark existing odds as not latest
        self.db.query(MatchOdds).filter(
            MatchOdds.match_id == match_id,
            MatchOdds.is_latest == True
        ).update({'is_latest': False})
        
        # Create new odds record
        odds = MatchOdds(
            match_id=match_id,
            odds_timestamp=datetime.utcnow(),
            # Match Result
            home_win_odds=odds_data.home_win,
            draw_odds=odds_data.draw,
            away_win_odds=odds_data.away_win,
            # Total Goals
            over_0_5_odds=odds_data.over_0_5,
            under_0_5_odds=odds_data.under_0_5,
            over_1_5_odds=odds_data.over_1_5,
            under_1_5_odds=odds_data.under_1_5,
            over_2_5_odds=odds_data.over_2_5,
            under_2_5_odds=odds_data.under_2_5,
            over_3_5_odds=odds_data.over_3_5,
            under_3_5_odds=odds_data.under_3_5,
            over_4_5_odds=odds_data.over_4_5,
            under_4_5_odds=odds_data.under_4_5,
            # BTTS
            btts_yes_odds=odds_data.btts_yes,
            btts_no_odds=odds_data.btts_no,
            # Double Chance
            home_or_draw_odds=odds_data.home_or_draw,
            away_or_draw_odds=odds_data.away_or_draw,
            home_or_away_odds=odds_data.home_or_away,
            # Corners
            corners_over_8_5_odds=odds_data.corners_over_8_5,
            corners_under_8_5_odds=odds_data.corners_under_8_5,
            corners_over_9_5_odds=odds_data.corners_over_9_5,
            corners_under_9_5_odds=odds_data.corners_under_9_5,
            corners_over_10_5_odds=odds_data.corners_over_10_5,
            corners_under_10_5_odds=odds_data.corners_under_10_5,
            # Cards
            cards_over_3_5_odds=odds_data.cards_over_3_5,
            cards_under_3_5_odds=odds_data.cards_under_3_5,
            cards_over_4_5_odds=odds_data.cards_over_4_5,
            cards_under_4_5_odds=odds_data.cards_under_4_5,
            is_latest=True
        )
        self.db.add(odds)
    
    def _process_result(self, match_id: str, result_data):
        """Process and store match result"""
        # Check if result already exists
        existing = self.db.query(MatchResult).filter(
            MatchResult.match_id == match_id
        ).first()
        
        if existing:
            # Update existing result
            existing.home_goals = result_data.home_goals
            existing.away_goals = result_data.away_goals
            existing.result = result_data.result
            existing.total_goals = result_data.total_goals
            existing.home_corners = result_data.home_corners
            existing.away_corners = result_data.away_corners
            existing.total_corners = result_data.total_corners
            existing.home_cards = result_data.home_cards
            existing.away_cards = result_data.away_cards
            existing.total_cards = result_data.total_cards
            existing.btts = result_data.btts
            existing.over_0_5 = result_data.over_0_5
            existing.over_1_5 = result_data.over_1_5
            existing.over_2_5 = result_data.over_2_5
            existing.over_3_5 = result_data.over_3_5
            existing.over_4_5 = result_data.over_4_5
            existing.corners_over_8_5 = result_data.corners_over_8_5
            existing.corners_over_9_5 = result_data.corners_over_9_5
            existing.corners_over_10_5 = result_data.corners_over_10_5
            existing.cards_over_3_5 = result_data.cards_over_3_5
            existing.cards_over_4_5 = result_data.cards_over_4_5
        else:
            # Create new result
            result = MatchResult(
                match_id=match_id,
                home_goals=result_data.home_goals,
                away_goals=result_data.away_goals,
                result=result_data.result,
                total_goals=result_data.total_goals,
                home_corners=result_data.home_corners,
                away_corners=result_data.away_corners,
                total_corners=result_data.total_corners,
                home_cards=result_data.home_cards,
                away_cards=result_data.away_cards,
                total_cards=result_data.total_cards,
                btts=result_data.btts,
                over_0_5=result_data.over_0_5,
                over_1_5=result_data.over_1_5,
                over_2_5=result_data.over_2_5,
                over_3_5=result_data.over_3_5,
                over_4_5=result_data.over_4_5,
                corners_over_8_5=result_data.corners_over_8_5,
                corners_over_9_5=result_data.corners_over_9_5,
                corners_over_10_5=result_data.corners_over_10_5,
                cards_over_3_5=result_data.cards_over_3_5,
                cards_over_4_5=result_data.cards_over_4_5
            )
            self.db.add(result)
        
        # Update match status
        match = self.db.query(Match).filter(Match.match_id == match_id).first()
        if match:
            match.status = 'completed'
