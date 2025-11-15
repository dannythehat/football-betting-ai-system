"""
Market-Specific Features
Advanced features tailored to each of the 4 betting markets
Goals O/U 2.5, Cards O/U 3.5, Corners O/U 9.5, BTTS Y/N
"""

import numpy as np
from typing import Dict, List, Any


class MarketSpecificFeatures:
    """
    Market-specific feature engineering for the 4 target markets
    Each market gets specialized features based on its unique characteristics
    """
    
    def __init__(self):
        pass
    
    def create_all_market_features(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Generate features for all 4 markets
        
        Args:
            match_data: Dictionary containing match and team data
            
        Returns:
            Dictionary of all market-specific features
        """
        features = {}
        
        features.update(self.create_goals_features(match_data))
        features.update(self.create_corners_features(match_data))
        features.update(self.create_cards_features(match_data))
        features.update(self.create_btts_features(match_data))
        
        return features
    
    def create_goals_features(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Goals O/U 2.5 specific features
        Focus: xG, shots, conversion rates, attacking patterns
        """
        features = {}
        
        # Expected Goals (xG) analysis
        features['home_xg_last_5'] = match_data.get('home_xg_last_5', match_data.get('home_goals_avg', 1.2))
        features['away_xg_last_5'] = match_data.get('away_xg_last_5', match_data.get('away_goals_avg', 1.0))
        features['combined_xg'] = features['home_xg_last_5'] + features['away_xg_last_5']
        
        # xG overperformance/underperformance
        home_actual_goals = match_data.get('home_goals_avg', 1.2)
        away_actual_goals = match_data.get('away_goals_avg', 1.0)
        features['home_xg_diff'] = home_actual_goals - features['home_xg_last_5']
        features['away_xg_diff'] = away_actual_goals - features['away_xg_last_5']
        
        # Shot statistics
        features['home_shots_per_game'] = match_data.get('home_shots_per_game', 12.0)
        features['away_shots_per_game'] = match_data.get('away_shots_per_game', 10.0)
        features['combined_shots_per_game'] = features['home_shots_per_game'] + features['away_shots_per_game']
        
        # Shots on target
        features['home_shots_on_target_pct'] = match_data.get('home_shots_on_target_pct', 0.35)
        features['away_shots_on_target_pct'] = match_data.get('away_shots_on_target_pct', 0.33)
        
        # Conversion efficiency
        features['home_conversion_rate'] = (
            home_actual_goals / features['home_shots_per_game'] 
            if features['home_shots_per_game'] > 0 else 0.1
        )
        features['away_conversion_rate'] = (
            away_actual_goals / features['away_shots_per_game']
            if features['away_shots_per_game'] > 0 else 0.1
        )
        
        # Big chances created/missed
        features['home_big_chances_per_game'] = match_data.get('home_big_chances', 2.5)
        features['away_big_chances_per_game'] = match_data.get('away_big_chances', 2.0)
        
        # Attacking intensity
        features['home_attacking_intensity'] = (
            features['home_shots_per_game'] * features['home_shots_on_target_pct']
        )
        features['away_attacking_intensity'] = (
            features['away_shots_per_game'] * features['away_shots_on_target_pct']
        )
        
        return features
    
    def create_corners_features(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Corners O/U 9.5 specific features
        Focus: possession, attacking style, corner patterns
        """
        features = {}
        
        # Corner patterns by half
        features['home_corners_first_half_avg'] = match_data.get('home_corners_1h_avg', 
                                                                  match_data.get('home_corners_avg', 5.0) * 0.45)
        features['home_corners_second_half_avg'] = match_data.get('home_corners_2h_avg',
                                                                   match_data.get('home_corners_avg', 5.0) * 0.55)
        
        features['away_corners_first_half_avg'] = match_data.get('away_corners_1h_avg',
                                                                  match_data.get('away_corners_avg', 4.5) * 0.45)
        features['away_corners_second_half_avg'] = match_data.get('away_corners_2h_avg',
                                                                   match_data.get('away_corners_avg', 4.5) * 0.55)
        
        # Possession-based corner prediction
        features['home_possession_avg'] = match_data.get('home_possession_pct', 50.0)
        features['away_possession_avg'] = match_data.get('away_possession_pct', 50.0)
        
        # Attacking style score (possession × corners)
        features['home_attacking_style_score'] = (
            features['home_possession_avg'] / 100 * match_data.get('home_corners_avg', 5.0)
        )
        features['away_attacking_style_score'] = (
            features['away_possession_avg'] / 100 * match_data.get('away_corners_avg', 4.5)
        )
        
        # Defensive corner concession
        features['home_corners_conceded_avg'] = match_data.get('home_corners_against_avg', 4.5)
        features['away_corners_conceded_avg'] = match_data.get('away_corners_against_avg', 5.0)
        
        # Expected corners (attack + defense conceded)
        features['expected_home_corners'] = (
            match_data.get('home_corners_avg', 5.0) + features['away_corners_conceded_avg']
        ) / 2
        features['expected_away_corners'] = (
            match_data.get('away_corners_avg', 4.5) + features['home_corners_conceded_avg']
        ) / 2
        features['expected_total_corners'] = features['expected_home_corners'] + features['expected_away_corners']
        
        # Corner consistency
        home_corners_history = match_data.get('home_corners_history', [])
        if len(home_corners_history) >= 5:
            features['home_corners_variance'] = np.var(home_corners_history[-10:])
        else:
            features['home_corners_variance'] = 2.0
        
        return features
    
    def create_cards_features(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Cards O/U 3.5 specific features
        Focus: discipline, fouls, referee, rivalry
        """
        features = {}
        
        # Disciplinary records
        features['home_yellow_cards_avg'] = match_data.get('home_yellows_avg', 1.8)
        features['away_yellow_cards_avg'] = match_data.get('away_yellows_avg', 1.7)
        features['home_red_cards_total'] = match_data.get('home_reds_season', 2)
        features['away_red_cards_total'] = match_data.get('away_reds_season', 1)
        
        # Total cards expectation
        features['expected_total_cards'] = (
            features['home_yellow_cards_avg'] + features['away_yellow_cards_avg'] +
            (features['home_red_cards_total'] + features['away_red_cards_total']) / 10
        )
        
        # Foul statistics
        features['home_fouls_per_game'] = match_data.get('home_fouls_avg', 11.0)
        features['away_fouls_per_game'] = match_data.get('away_fouls_avg', 10.5)
        features['combined_fouls_avg'] = features['home_fouls_per_game'] + features['away_fouls_per_game']
        
        # Fouls to cards ratio
        features['home_fouls_to_cards_ratio'] = (
            features['home_fouls_per_game'] / (features['home_yellow_cards_avg'] + 0.1)
        )
        features['away_fouls_to_cards_ratio'] = (
            features['away_fouls_per_game'] / (features['away_yellow_cards_avg'] + 0.1)
        )
        
        # Aggression indicators
        features['match_rivalry_score'] = match_data.get('rivalry_intensity', 0)  # 0-10 scale
        features['match_importance_score'] = match_data.get('match_importance', 5)  # 0-10 scale
        
        # Referee strictness
        features['referee_cards_per_game'] = match_data.get('referee_cards_avg', 3.5)
        features['referee_strictness'] = match_data.get('referee_strictness_rating', 5.0)  # 0-10 scale
        
        # Combined aggression score
        features['combined_aggression_score'] = (
            (features['combined_fouls_avg'] / 20) * 3 +
            features['match_rivalry_score'] / 10 * 3 +
            features['referee_strictness'] / 10 * 4
        )
        
        return features
    
    def create_btts_features(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """
        BTTS Y/N specific features
        Focus: clean sheets, scoring consistency, defensive vulnerability
        """
        features = {}
        
        # Clean sheet rates
        features['home_clean_sheets_rate'] = match_data.get('home_clean_sheets_rate', 0.3)
        features['away_clean_sheets_rate'] = match_data.get('away_clean_sheets_rate', 0.25)
        
        # Failed to score rates
        features['home_failed_to_score_rate'] = match_data.get('home_blanks_rate', 0.2)
        features['away_failed_to_score_rate'] = match_data.get('away_blanks_rate', 0.25)
        
        # BTTS probability calculation
        features['btts_probability_estimate'] = (
            (1 - features['home_failed_to_score_rate']) *
            (1 - features['away_failed_to_score_rate'])
        )
        
        # Scoring consistency (scored in last N games)
        features['home_scored_in_last_5'] = match_data.get('home_scored_last_5_count', 4)
        features['away_scored_in_last_5'] = match_data.get('away_scored_last_5_count', 3)
        features['home_scoring_consistency'] = features['home_scored_in_last_5'] / 5
        features['away_scoring_consistency'] = features['away_scored_in_last_5'] / 5
        
        # Defensive vulnerability (conceded in last N games)
        features['home_conceded_in_last_5'] = match_data.get('home_conceded_last_5_count', 3)
        features['away_conceded_in_last_5'] = match_data.get('away_conceded_last_5_count', 4)
        features['home_defensive_vulnerability'] = features['home_conceded_in_last_5'] / 5
        features['away_defensive_vulnerability'] = features['away_conceded_in_last_5'] / 5
        
        # Combined BTTS indicators
        features['both_teams_score_capability'] = (
            features['home_scoring_consistency'] * features['away_scoring_consistency']
        )
        features['both_teams_concede_likelihood'] = (
            features['home_defensive_vulnerability'] * features['away_defensive_vulnerability']
        )
        
        # BTTS composite score
        features['btts_composite_score'] = (
            features['btts_probability_estimate'] * 0.4 +
            features['both_teams_score_capability'] * 0.3 +
            features['both_teams_concede_likelihood'] * 0.3
        )
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Return list of all market-specific feature names"""
        goals_features = [
            'home_xg_last_5', 'away_xg_last_5', 'combined_xg',
            'home_xg_diff', 'away_xg_diff',
            'home_shots_per_game', 'away_shots_per_game', 'combined_shots_per_game',
            'home_shots_on_target_pct', 'away_shots_on_target_pct',
            'home_conversion_rate', 'away_conversion_rate',
            'home_big_chances_per_game', 'away_big_chances_per_game',
            'home_attacking_intensity', 'away_attacking_intensity'
        ]
        
        corners_features = [
            'home_corners_first_half_avg', 'home_corners_second_half_avg',
            'away_corners_first_half_avg', 'away_corners_second_half_avg',
            'home_possession_avg', 'away_possession_avg',
            'home_attacking_style_score', 'away_attacking_style_score',
            'home_corners_conceded_avg', 'away_corners_conceded_avg',
            'expected_home_corners', 'expected_away_corners', 'expected_total_corners',
            'home_corners_variance'
        ]
        
        cards_features = [
            'home_yellow_cards_avg', 'away_yellow_cards_avg',
            'home_red_cards_total', 'away_red_cards_total',
            'expected_total_cards',
            'home_fouls_per_game', 'away_fouls_per_game', 'combined_fouls_avg',
            'home_fouls_to_cards_ratio', 'away_fouls_to_cards_ratio',
            'match_rivalry_score', 'match_importance_score',
            'referee_cards_per_game', 'referee_strictness',
            'combined_aggression_score'
        ]
        
        btts_features = [
            'home_clean_sheets_rate', 'away_clean_sheets_rate',
            'home_failed_to_score_rate', 'away_failed_to_score_rate',
            'btts_probability_estimate',
            'home_scored_in_last_5', 'away_scored_in_last_5',
            'home_scoring_consistency', 'away_scoring_consistency',
            'home_conceded_in_last_5', 'away_conceded_in_last_5',
            'home_defensive_vulnerability', 'away_defensive_vulnerability',
            'both_teams_score_capability', 'both_teams_concede_likelihood',
            'btts_composite_score'
        ]
        
        return goals_features + corners_features + cards_features + btts_features
