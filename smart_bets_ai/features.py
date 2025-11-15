"""
Feature Engineering for Smart Bets AI
Transforms raw match data into ML-ready features for the 4 target markets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class FeatureEngineer:
    """
    Feature engineering for football betting predictions
    Focuses on the 4 target markets: Goals O/U 2.5, Cards O/U 3.5, Corners O/U 9.5, BTTS Y/N
    """
    
    def __init__(self):
        self.feature_columns = []
    
    def create_features(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from raw match data
        
        Args:
            matches_df: DataFrame with match data including team stats
            
        Returns:
            DataFrame with engineered features
        """
        df = matches_df.copy()
        
        # Basic features
        df = self._add_basic_features(df)
        
        # Goals-specific features
        df = self._add_goals_features(df)
        
        # Cards-specific features
        df = self._add_cards_features(df)
        
        # Corners-specific features
        df = self._add_corners_features(df)
        
        # BTTS-specific features
        df = self._add_btts_features(df)
        
        # Form features
        df = self._add_form_features(df)
        
        # Store feature columns
        self.feature_columns = [col for col in df.columns if col not in [
            'match_id', 'match_datetime', 'home_team', 'away_team', 
            'home_team_id', 'away_team_id', 'league', 'season', 'status',
            'home_goals', 'away_goals', 'result', 'total_goals', 
            'home_corners', 'away_corners', 'total_corners',
            'home_cards', 'away_cards', 'total_cards', 'btts',
            'over_2_5', 'cards_over_3_5', 'corners_over_9_5', 'btts_yes'
        ]]
        
        return df
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic statistical features"""
        # Combined averages
        df['combined_goals_avg'] = df['home_goals_avg'] + df['away_goals_avg']
        df['combined_goals_conceded_avg'] = df['home_goals_conceded_avg'] + df['away_goals_conceded_avg']
        df['combined_corners_avg'] = df['home_corners_avg'] + df['away_corners_avg']
        df['combined_cards_avg'] = df['home_cards_avg'] + df['away_cards_avg']
        
        # Attack vs Defense matchups
        df['home_attack_vs_away_defense'] = df['home_goals_avg'] - df['away_goals_conceded_avg']
        df['away_attack_vs_home_defense'] = df['away_goals_avg'] - df['home_goals_conceded_avg']
        df['attack_defense_differential'] = df['home_attack_vs_away_defense'] + df['away_attack_vs_home_defense']
        
        return df
    
    def _add_goals_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features specific to goals prediction (O/U 2.5)"""
        # Expected goals
        df['expected_total_goals'] = df['combined_goals_avg']
        df['expected_goals_ratio'] = df['home_goals_avg'] / (df['away_goals_avg'] + 0.1)
        
        # Defensive strength
        df['defensive_strength'] = (df['home_goals_conceded_avg'] + df['away_goals_conceded_avg']) / 2
        
        # Offensive power
        df['offensive_power'] = (df['home_goals_avg'] + df['away_goals_avg']) / 2
        
        # Goals variance indicator
        df['goals_variance'] = abs(df['home_goals_avg'] - df['away_goals_avg'])
        
        return df
    
    def _add_cards_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features specific to cards prediction (O/U 3.5)"""
        # Combined card averages
        df['expected_total_cards'] = df['combined_cards_avg']
        
        # Card differential
        df['cards_differential'] = abs(df['home_cards_avg'] - df['away_cards_avg'])
        
        # High card rate indicator
        df['high_card_rate'] = ((df['home_cards_avg'] > 2.5) & (df['away_cards_avg'] > 2.5)).astype(int)
        
        return df
    
    def _add_corners_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features specific to corners prediction (O/U 9.5)"""
        # Combined corner averages
        df['expected_total_corners'] = df['combined_corners_avg']
        
        # Corner differential
        df['corners_differential'] = abs(df['home_corners_avg'] - df['away_corners_avg'])
        
        # High corner rate indicator
        df['high_corner_rate'] = ((df['home_corners_avg'] > 5.0) & (df['away_corners_avg'] > 4.5)).astype(int)
        
        # Corner dominance
        df['corner_dominance'] = df['home_corners_avg'] / (df['away_corners_avg'] + 0.1)
        
        return df
    
    def _add_btts_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features specific to BTTS prediction (Yes/No)"""
        # Combined BTTS rates
        df['combined_btts_rate'] = (df['home_btts_rate'] + df['away_btts_rate']) / 2
        
        # BTTS probability indicator
        df['btts_likelihood'] = df['combined_btts_rate']
        
        # Both teams scoring capability
        df['both_teams_score_capability'] = (
            (df['home_goals_avg'] > 0.8) & (df['away_goals_avg'] > 0.8)
        ).astype(int)
        
        # Both teams concede indicator
        df['both_teams_concede'] = (
            (df['home_goals_conceded_avg'] > 0.8) & (df['away_goals_conceded_avg'] > 0.8)
        ).astype(int)
        
        return df
    
    def _add_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add form-based features"""
        # Convert form strings to numeric scores
        def form_to_score(form_str):
            if pd.isna(form_str) or not isinstance(form_str, str):
                return 0
            score = 0
            for char in form_str:
                if char == 'W':
                    score += 3
                elif char == 'D':
                    score += 1
            return score
        
        df['home_form_score'] = df['home_form'].apply(form_to_score)
        df['away_form_score'] = df['away_form'].apply(form_to_score)
        df['form_differential'] = df['home_form_score'] - df['away_form_score']
        df['combined_form_score'] = df['home_form_score'] + df['away_form_score']
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names"""
        return self.feature_columns
    
    def prepare_training_data(
        self, 
        matches_df: pd.DataFrame,
        target_market: str
    ) -> tuple:
        """
        Prepare features and target for model training
        
        Args:
            matches_df: DataFrame with match data
            target_market: One of 'goals', 'cards', 'corners', 'btts'
            
        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        # Create features
        df = self.create_features(matches_df)
        
        # Define target based on market
        target_map = {
            'goals': 'over_2_5',
            'cards': 'cards_over_3_5',
            'corners': 'corners_over_9_5',
            'btts': 'btts_yes'
        }
        
        if target_market not in target_map:
            raise ValueError(f"Invalid target_market: {target_market}. Must be one of {list(target_map.keys())}")
        
        target_col = target_map[target_market]
        
        # Check if target exists
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Get features and target
        X = df[self.feature_columns]
        y = df[target_col]
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        return X, y
