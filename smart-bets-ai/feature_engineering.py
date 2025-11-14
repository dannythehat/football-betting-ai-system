"""
Feature Engineering for Smart Bets AI
Transforms raw match data into ML-ready features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime


class FeatureEngineer:
    """
    Transforms raw match and team statistics into features for ML models
    """
    
    def __init__(self):
        self.feature_columns = []
        
    def create_features(self, match_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features from match data
        
        Args:
            match_data: DataFrame with match information and team stats
            
        Returns:
            DataFrame with engineered features
        """
        df = match_data.copy()
        
        # Basic features
        df = self._add_goal_features(df)
        df = self._add_defensive_features(df)
        df = self._add_form_features(df)
        df = self._add_btts_features(df)
        df = self._add_corners_cards_features(df)
        df = self._add_derived_features(df)
        
        # Store feature column names
        self.feature_columns = [col for col in df.columns if col not in [
            'match_id', 'home_team_id', 'away_team_id', 'match_datetime',
            'league', 'season', 'status', 'created_at', 'updated_at'
        ]]
        
        return df
    
    def _add_goal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add goal-related features"""
        # Goal scoring strength
        df['home_attack_strength'] = df['home_goals_avg']
        df['away_attack_strength'] = df['away_goals_avg']
        
        # Combined goals expectation
        df['expected_total_goals'] = df['home_goals_avg'] + df['away_goals_avg']
        
        # Attack vs defense matchup
        df['home_attack_vs_away_defense'] = df['home_goals_avg'] - df['away_goals_conceded_avg']
        df['away_attack_vs_home_defense'] = df['away_goals_avg'] - df['home_goals_conceded_avg']
        
        # Goal difference indicator
        df['home_goal_advantage'] = df['home_goals_avg'] - df['away_goals_avg']
        
        return df
    
    def _add_defensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add defensive strength features"""
        # Defensive strength
        df['home_defense_strength'] = df['home_goals_conceded_avg']
        df['away_defense_strength'] = df['away_goals_conceded_avg']
        
        # Combined defensive weakness
        df['combined_defensive_weakness'] = df['home_goals_conceded_avg'] + df['away_goals_conceded_avg']
        
        # Clean sheet probability indicators
        df['home_clean_sheet_indicator'] = (df['home_goals_conceded_avg'] < 0.8).astype(int)
        df['away_clean_sheet_indicator'] = (df['away_goals_conceded_avg'] < 0.8).astype(int)
        
        return df
    
    def _add_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add form-based features"""
        # Convert form strings to numeric scores
        df['home_form_score'] = df['home_form'].apply(self._form_to_score)
        df['away_form_score'] = df['away_form'].apply(self._form_to_score)
        
        # Form difference
        df['form_difference'] = df['home_form_score'] - df['away_form_score']
        
        # Recent form strength
        df['home_form_strength'] = df['home_form_score'] / 15.0  # Normalize to 0-1
        df['away_form_strength'] = df['away_form_score'] / 15.0
        
        return df
    
    def _add_btts_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Both Teams To Score features"""
        # BTTS probability indicators
        df['btts_combined_rate'] = (df['home_btts_rate'] + df['away_btts_rate']) / 2
        
        # High BTTS probability indicator
        df['btts_likely'] = ((df['home_btts_rate'] > 0.5) & (df['away_btts_rate'] > 0.5)).astype(int)
        
        # BTTS with strong attacks
        df['btts_with_strong_attacks'] = (
            (df['home_goals_avg'] > 1.2) & 
            (df['away_goals_avg'] > 1.0)
        ).astype(int)
        
        return df
    
    def _add_corners_cards_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add corners and cards features"""
        # Total corners expectation
        df['expected_total_corners'] = df['home_corners_avg'] + df['away_corners_avg']
        
        # High corners indicator
        df['high_corners_expected'] = (df['expected_total_corners'] > 10.0).astype(int)
        
        # Total cards expectation
        df['expected_total_cards'] = df['home_cards_avg'] + df['away_cards_avg']
        
        # High cards indicator
        df['high_cards_expected'] = (df['expected_total_cards'] > 4.0).astype(int)
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived and interaction features"""
        # Goal ratio features
        df['home_goal_ratio'] = df['home_goals_avg'] / (df['home_goals_conceded_avg'] + 0.1)
        df['away_goal_ratio'] = df['away_goals_avg'] / (df['away_goals_conceded_avg'] + 0.1)
        
        # Strength differential
        df['overall_strength_diff'] = (
            (df['home_goals_avg'] - df['home_goals_conceded_avg']) -
            (df['away_goals_avg'] - df['away_goals_conceded_avg'])
        )
        
        # High scoring match indicator
        df['high_scoring_expected'] = (df['expected_total_goals'] > 2.5).astype(int)
        
        # Low scoring match indicator
        df['low_scoring_expected'] = (df['expected_total_goals'] < 2.0).astype(int)
        
        # Balanced match indicator
        df['balanced_match'] = (abs(df['home_goal_advantage']) < 0.3).astype(int)
        
        # Dominant home team
        df['home_dominant'] = (df['home_goal_advantage'] > 0.5).astype(int)
        
        # Dominant away team
        df['away_dominant'] = (df['home_goal_advantage'] < -0.5).astype(int)
        
        return df
    
    @staticmethod
    def _form_to_score(form_string: Optional[str]) -> int:
        """
        Convert form string (e.g., 'WWDLW') to numeric score
        W=3, D=1, L=0
        """
        if pd.isna(form_string) or not form_string:
            return 7  # Neutral score
        
        score_map = {'W': 3, 'D': 1, 'L': 0}
        return sum(score_map.get(char, 0) for char in form_string)
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature column names"""
        return self.feature_columns
    
    def create_target_variables(self, results_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Create target variables from match results
        
        Args:
            results_df: DataFrame with match results
            
        Returns:
            Dictionary of target variable Series
        """
        targets = {}
        
        # Match result targets
        targets['home_win'] = (results_df['result'] == 'home_win').astype(int)
        targets['draw'] = (results_df['result'] == 'draw').astype(int)
        targets['away_win'] = (results_df['result'] == 'away_win').astype(int)
        
        # Goals targets
        targets['over_0_5'] = results_df['over_0_5'].astype(int)
        targets['over_1_5'] = results_df['over_1_5'].astype(int)
        targets['over_2_5'] = results_df['over_2_5'].astype(int)
        targets['over_3_5'] = results_df['over_3_5'].astype(int)
        targets['over_4_5'] = results_df['over_4_5'].astype(int)
        
        # BTTS target
        targets['btts'] = results_df['btts'].astype(int)
        
        # Corners targets
        targets['corners_over_8_5'] = results_df['corners_over_8_5'].astype(int)
        targets['corners_over_9_5'] = results_df['corners_over_9_5'].astype(int)
        targets['corners_over_10_5'] = results_df['corners_over_10_5'].astype(int)
        
        # Cards targets
        targets['cards_over_3_5'] = results_df['cards_over_3_5'].astype(int)
        targets['cards_over_4_5'] = results_df['cards_over_4_5'].astype(int)
        
        return targets


def prepare_training_data(matches_df: pd.DataFrame, results_df: pd.DataFrame) -> tuple:
    """
    Prepare complete training dataset with features and targets
    
    Args:
        matches_df: DataFrame with match information
        results_df: DataFrame with match results
        
    Returns:
        Tuple of (features_df, targets_dict, feature_engineer)
    """
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Create features
    features_df = fe.create_features(matches_df)
    
    # Merge with results
    merged_df = features_df.merge(
        results_df,
        on='match_id',
        how='inner'
    )
    
    # Create targets
    targets_dict = fe.create_target_variables(merged_df)
    
    # Get only feature columns
    feature_cols = fe.get_feature_names()
    X = merged_df[feature_cols]
    
    return X, targets_dict, fe
