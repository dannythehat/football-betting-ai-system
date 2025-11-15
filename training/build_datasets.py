"""
Dataset Builder for Training
Prepares clean training datasets for each market from historical match data
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from sqlalchemy.orm import Session
from sqlalchemy import and_, func

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_ingestion.database import get_db
from data_ingestion.models import Match, MatchResult, MatchOdds, Team, TeamStatistic
from training.config import (
    TRAINING_DATA_PATHS, LOOKBACK_WINDOWS, MIN_MATCHES_FOR_STATS,
    MARKETS, DATA_PROCESSED_DIR
)


class DatasetBuilder:
    """Builds training datasets from database or raw files"""
    
    def __init__(self, session: Optional[Session] = None):
        self.session = session
        self.lookback = LOOKBACK_WINDOWS
    
    def _calculate_rolling_stats(
        self, 
        team_id: int, 
        match_date: datetime,
        is_home: bool,
        window: int = 5
    ) -> Dict:
        """
        Calculate rolling statistics for a team up to a specific date
        
        Args:
            team_id: Team ID
            match_date: Date to calculate stats up to (exclusive)
            is_home: Whether calculating for home or away matches
            window: Number of recent matches to consider
            
        Returns:
            Dictionary with rolling statistics
        """
        if not self.session:
            return {}
        
        # Query recent matches before this date
        if is_home:
            matches = self.session.query(Match, MatchResult).join(
                MatchResult, Match.match_id == MatchResult.match_id
            ).filter(
                and_(
                    Match.home_team_id == team_id,
                    Match.match_datetime < match_date,
                    Match.status == 'completed'
                )
            ).order_by(Match.match_datetime.desc()).limit(window).all()
        else:
            matches = self.session.query(Match, MatchResult).join(
                MatchResult, Match.match_id == MatchResult.match_id
            ).filter(
                and_(
                    Match.away_team_id == team_id,
                    Match.match_datetime < match_date,
                    Match.status == 'completed'
                )
            ).order_by(Match.match_datetime.desc()).limit(window).all()
        
        if len(matches) < MIN_MATCHES_FOR_STATS:
            return {}
        
        # Calculate statistics
        goals_scored = []
        goals_conceded = []
        corners = []
        cards = []
        btts_count = 0
        
        for match, result in matches:
            if is_home:
                goals_scored.append(result.home_goals)
                goals_conceded.append(result.away_goals)
                corners.append(result.home_corners or 0)
                cards.append(result.home_cards or 0)
            else:
                goals_scored.append(result.away_goals)
                goals_conceded.append(result.home_goals)
                corners.append(result.away_corners or 0)
                cards.append(result.away_cards or 0)
            
            if result.btts:
                btts_count += 1
        
        return {
            'goals_avg': np.mean(goals_scored) if goals_scored else 0,
            'goals_conceded_avg': np.mean(goals_conceded) if goals_conceded else 0,
            'corners_avg': np.mean(corners) if corners else 0,
            'cards_avg': np.mean(cards) if cards else 0,
            'btts_rate': btts_count / len(matches) if matches else 0,
            'matches_count': len(matches)
        }
    
    def _get_match_features(self, match: Match, result: MatchResult) -> Dict:
        """
        Extract features for a single match
        
        Args:
            match: Match object
            result: MatchResult object
            
        Returns:
            Dictionary with match features
        """
        # Get rolling stats for both teams
        home_stats_5 = self._calculate_rolling_stats(
            match.home_team_id, match.match_datetime, is_home=True, window=5
        )
        away_stats_5 = self._calculate_rolling_stats(
            match.away_team_id, match.match_datetime, is_home=False, window=5
        )
        
        home_stats_10 = self._calculate_rolling_stats(
            match.home_team_id, match.match_datetime, is_home=True, window=10
        )
        away_stats_10 = self._calculate_rolling_stats(
            match.away_team_id, match.match_datetime, is_home=False, window=10
        )
        
        # Skip if insufficient data
        if not home_stats_5 or not away_stats_5:
            return None
        
        features = {
            # Match info
            'match_id': match.match_id,
            'date': match.match_datetime,
            'league': match.league,
            'home_team_id': match.home_team_id,
            'away_team_id': match.away_team_id,
            
            # Rolling averages (5 matches)
            'home_goals_avg_5': home_stats_5['goals_avg'],
            'away_goals_avg_5': away_stats_5['goals_avg'],
            'home_goals_conceded_avg_5': home_stats_5['goals_conceded_avg'],
            'away_goals_conceded_avg_5': away_stats_5['goals_conceded_avg'],
            'home_corners_avg_5': home_stats_5['corners_avg'],
            'away_corners_avg_5': away_stats_5['corners_avg'],
            'home_cards_avg_5': home_stats_5['cards_avg'],
            'away_cards_avg_5': away_stats_5['cards_avg'],
            'home_btts_rate_5': home_stats_5['btts_rate'],
            'away_btts_rate_5': away_stats_5['btts_rate'],
            
            # Rolling averages (10 matches) if available
            'home_goals_avg_10': home_stats_10.get('goals_avg', home_stats_5['goals_avg']),
            'away_goals_avg_10': away_stats_10.get('goals_avg', away_stats_5['goals_avg']),
            'home_goals_conceded_avg_10': home_stats_10.get('goals_conceded_avg', home_stats_5['goals_conceded_avg']),
            'away_goals_conceded_avg_10': away_stats_10.get('goals_conceded_avg', away_stats_5['goals_conceded_avg']),
            
            # Derived features
            'combined_goals_avg': home_stats_5['goals_avg'] + away_stats_5['goals_avg'],
            'combined_corners_avg': home_stats_5['corners_avg'] + away_stats_5['corners_avg'],
            'combined_cards_avg': home_stats_5['cards_avg'] + away_stats_5['cards_avg'],
            'combined_btts_rate': (home_stats_5['btts_rate'] + away_stats_5['btts_rate']) / 2,
            
            # Attack vs Defense
            'home_attack_vs_away_defense': home_stats_5['goals_avg'] - away_stats_5['goals_conceded_avg'],
            'away_attack_vs_home_defense': away_stats_5['goals_avg'] - home_stats_5['goals_conceded_avg'],
        }
        
        return features
    
    def build_training_table_for_goals(
        self, 
        out_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Build training dataset for Goals Over 2.5 market
        
        Args:
            out_path: Path to save CSV (optional)
            
        Returns:
            DataFrame with training data
        """
        if not self.session:
            raise ValueError("Database session required")
        
        print("🔄 Building Goals Over 2.5 training dataset...")
        
        # Query completed matches with results and odds
        matches = self.session.query(Match, MatchResult, MatchOdds).join(
            MatchResult, Match.match_id == MatchResult.match_id
        ).join(
            MatchOdds, and_(
                Match.match_id == MatchOdds.match_id,
                MatchOdds.is_latest == True
            )
        ).filter(
            Match.status == 'completed'
        ).order_by(Match.match_datetime).all()
        
        rows = []
        for match, result, odds in matches:
            features = self._get_match_features(match, result)
            if not features:
                continue
            
            # Add target and odds
            features['y'] = 1 if result.over_2_5 else 0
            features['odds_over25'] = float(odds.over_2_5_odds) if odds.over_2_5_odds else None
            
            rows.append(features)
        
        df = pd.DataFrame(rows)
        
        # Save if path provided
        if out_path:
            df.to_csv(out_path, index=False)
            print(f"✅ Saved {len(df)} matches to {out_path}")
        
        print(f"✅ Built Goals dataset: {len(df)} matches")
        return df
    
    def build_training_table_for_btts(
        self, 
        out_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Build training dataset for BTTS market"""
        if not self.session:
            raise ValueError("Database session required")
        
        print("🔄 Building BTTS training dataset...")
        
        matches = self.session.query(Match, MatchResult, MatchOdds).join(
            MatchResult, Match.match_id == MatchResult.match_id
        ).join(
            MatchOdds, and_(
                Match.match_id == MatchOdds.match_id,
                MatchOdds.is_latest == True
            )
        ).filter(
            Match.status == 'completed'
        ).order_by(Match.match_datetime).all()
        
        rows = []
        for match, result, odds in matches:
            features = self._get_match_features(match, result)
            if not features:
                continue
            
            features['y'] = 1 if result.btts else 0
            features['odds_btts_yes'] = float(odds.btts_yes_odds) if odds.btts_yes_odds else None
            
            rows.append(features)
        
        df = pd.DataFrame(rows)
        
        if out_path:
            df.to_csv(out_path, index=False)
            print(f"✅ Saved {len(df)} matches to {out_path}")
        
        print(f"✅ Built BTTS dataset: {len(df)} matches")
        return df
    
    def build_training_table_for_cards(
        self, 
        out_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Build training dataset for Cards Over 3.5 market"""
        if not self.session:
            raise ValueError("Database session required")
        
        print("🔄 Building Cards Over 3.5 training dataset...")
        
        matches = self.session.query(Match, MatchResult, MatchOdds).join(
            MatchResult, Match.match_id == MatchResult.match_id
        ).join(
            MatchOdds, and_(
                Match.match_id == MatchOdds.match_id,
                MatchOdds.is_latest == True
            )
        ).filter(
            Match.status == 'completed'
        ).order_by(Match.match_datetime).all()
        
        rows = []
        for match, result, odds in matches:
            features = self._get_match_features(match, result)
            if not features:
                continue
            
            features['y'] = 1 if result.cards_over_3_5 else 0
            features['odds_cards_over35'] = float(odds.cards_over_3_5_odds) if odds.cards_over_3_5_odds else None
            
            rows.append(features)
        
        df = pd.DataFrame(rows)
        
        if out_path:
            df.to_csv(out_path, index=False)
            print(f"✅ Saved {len(df)} matches to {out_path}")
        
        print(f"✅ Built Cards dataset: {len(df)} matches")
        return df
    
    def build_training_table_for_corners(
        self, 
        out_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Build training dataset for Corners Over 9.5 market"""
        if not self.session:
            raise ValueError("Database session required")
        
        print("🔄 Building Corners Over 9.5 training dataset...")
        
        matches = self.session.query(Match, MatchResult, MatchOdds).join(
            MatchResult, Match.match_id == MatchResult.match_id
        ).join(
            MatchOdds, and_(
                Match.match_id == MatchOdds.match_id,
                MatchOdds.is_latest == True
            )
        ).filter(
            Match.status == 'completed'
        ).order_by(Match.match_datetime).all()
        
        rows = []
        for match, result, odds in matches:
            features = self._get_match_features(match, result)
            if not features:
                continue
            
            features['y'] = 1 if result.corners_over_9_5 else 0
            features['odds_corners_over95'] = float(odds.corners_over_9_5_odds) if odds.corners_over_9_5_odds else None
            
            rows.append(features)
        
        df = pd.DataFrame(rows)
        
        if out_path:
            df.to_csv(out_path, index=False)
            print(f"✅ Saved {len(df)} matches to {out_path}")
        
        print(f"✅ Built Corners dataset: {len(df)} matches")
        return df


def build_all_training_datasets():
    """Build all training datasets from database"""
    print("=" * 60)
    print("BUILDING ALL TRAINING DATASETS")
    print("=" * 60)
    
    with get_db() as session:
        builder = DatasetBuilder(session)
        
        # Build each market dataset
        builder.build_training_table_for_goals(
            str(TRAINING_DATA_PATHS['goals'])
        )
        builder.build_training_table_for_btts(
            str(TRAINING_DATA_PATHS['btts'])
        )
        builder.build_training_table_for_cards(
            str(TRAINING_DATA_PATHS['cards'])
        )
        builder.build_training_table_for_corners(
            str(TRAINING_DATA_PATHS['corners'])
        )
    
    print("\n" + "=" * 60)
    print("✅ ALL DATASETS BUILT SUCCESSFULLY")
    print("=" * 60)


# Standalone functions for compatibility
def build_training_table_for_goals(session_or_data_source, out_path: str) -> None:
    """Standalone function to build goals training dataset"""
    if isinstance(session_or_data_source, Session):
        builder = DatasetBuilder(session_or_data_source)
        builder.build_training_table_for_goals(out_path)
    else:
        with get_db() as session:
            builder = DatasetBuilder(session)
            builder.build_training_table_for_goals(out_path)


def build_training_table_for_btts(session_or_data_source, out_path: str) -> None:
    """Standalone function to build BTTS training dataset"""
    if isinstance(session_or_data_source, Session):
        builder = DatasetBuilder(session_or_data_source)
        builder.build_training_table_for_btts(out_path)
    else:
        with get_db() as session:
            builder = DatasetBuilder(session)
            builder.build_training_table_for_btts(out_path)


def build_training_table_for_cards(session_or_data_source, out_path: str) -> None:
    """Standalone function to build cards training dataset"""
    if isinstance(session_or_data_source, Session):
        builder = DatasetBuilder(session_or_data_source)
        builder.build_training_table_for_cards(out_path)
    else:
        with get_db() as session:
            builder = DatasetBuilder(session)
            builder.build_training_table_for_cards(out_path)


def build_training_table_for_corners(session_or_data_source, out_path: str) -> None:
    """Standalone function to build corners training dataset"""
    if isinstance(session_or_data_source, Session):
        builder = DatasetBuilder(session_or_data_source)
        builder.build_training_table_for_corners(out_path)
    else:
        with get_db() as session:
            builder = DatasetBuilder(session)
            builder.build_training_table_for_corners(out_path)


if __name__ == "__main__":
    build_all_training_datasets()
