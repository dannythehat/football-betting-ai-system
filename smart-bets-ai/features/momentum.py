"""
Momentum Analyzer
Momentum, confidence, and psychological factors
Generates 15+ features capturing team psychology and recent performance trends
"""

import numpy as np
from typing import Dict, List, Any
from datetime import datetime, timedelta


class MomentumAnalyzer:
    """
    Analyzes team momentum, confidence, and psychological factors
    Captures intangible elements that affect performance
    """
    
    def __init__(self):
        pass
    
    def analyze_momentum(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Generate momentum and psychological features
        
        Args:
            match_data: Dictionary containing team performance data
            
        Returns:
            Dictionary of momentum features
        """
        features = {}
        
        # Recent form momentum
        features.update(self._form_momentum(match_data))
        
        # Scoring momentum
        features.update(self._scoring_momentum(match_data))
        
        # Defensive momentum
        features.update(self._defensive_momentum(match_data))
        
        # Confidence indicators
        features.update(self._confidence_features(match_data))
        
        # Pressure situations
        features.update(self._pressure_features(match_data))
        
        # Fatigue indicators
        features.update(self._fatigue_features(match_data))
        
        return features
    
    def _form_momentum(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate recent form momentum with weighted scoring"""
        features = {}
        
        # Exponential weights (most recent = highest)
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        
        # Home team momentum
        home_results = match_data.get('home_results_last_4', [])
        if len(home_results) >= 4:
            points = [3 if r == 'W' else 1 if r == 'D' else 0 for r in home_results]
            features['home_momentum_score'] = np.average(points, weights=weights)
        else:
            features['home_momentum_score'] = 1.5
        
        # Away team momentum
        away_results = match_data.get('away_results_last_4', [])
        if len(away_results) >= 4:
            points = [3 if r == 'W' else 1 if r == 'D' else 0 for r in away_results]
            features['away_momentum_score'] = np.average(points, weights=weights)
        else:
            features['away_momentum_score'] = 1.5
        
        # Momentum differential
        features['momentum_differential'] = features['home_momentum_score'] - features['away_momentum_score']
        
        return features
    
    def _scoring_momentum(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze goal scoring momentum trends"""
        features = {}
        
        # Home scoring momentum (recent vs previous)
        home_goals_last_3 = match_data.get('home_goals_last_3', [])
        home_goals_prev_3 = match_data.get('home_goals_prev_3', [])
        
        if home_goals_last_3 and home_goals_prev_3:
            features['home_scoring_momentum'] = np.mean(home_goals_last_3) - np.mean(home_goals_prev_3)
        else:
            features['home_scoring_momentum'] = 0
        
        # Away scoring momentum
        away_goals_last_3 = match_data.get('away_goals_last_3', [])
        away_goals_prev_3 = match_data.get('away_goals_prev_3', [])
        
        if away_goals_last_3 and away_goals_prev_3:
            features['away_scoring_momentum'] = np.mean(away_goals_last_3) - np.mean(away_goals_prev_3)
        else:
            features['away_scoring_momentum'] = 0
        
        # Hot streak indicator (3+ goals in last 2 games)
        features['home_hot_streak'] = 1 if sum(home_goals_last_3[:2]) >= 3 else 0
        features['away_hot_streak'] = 1 if sum(away_goals_last_3[:2]) >= 3 else 0
        
        return features
    
    def _defensive_momentum(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze defensive momentum (improving/declining)"""
        features = {}
        
        # Home defensive momentum (lower conceded = positive momentum)
        home_conceded_last_3 = match_data.get('home_conceded_last_3', [])
        home_conceded_prev_3 = match_data.get('home_conceded_prev_3', [])
        
        if home_conceded_last_3 and home_conceded_prev_3:
            # Positive value = improving defense
            features['home_defensive_momentum'] = np.mean(home_conceded_prev_3) - np.mean(home_conceded_last_3)
        else:
            features['home_defensive_momentum'] = 0
        
        # Away defensive momentum
        away_conceded_last_3 = match_data.get('away_conceded_last_3', [])
        away_conceded_prev_3 = match_data.get('away_conceded_prev_3', [])
        
        if away_conceded_last_3 and away_conceded_prev_3:
            features['away_defensive_momentum'] = np.mean(away_conceded_prev_3) - np.mean(away_conceded_last_3)
        else:
            features['away_defensive_momentum'] = 0
        
        # Clean sheet momentum
        features['home_clean_sheets_last_3'] = sum(1 for g in home_conceded_last_3 if g == 0)
        features['away_clean_sheets_last_3'] = sum(1 for g in away_conceded_last_3 if g == 0)
        
        return features
    
    def _confidence_features(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate team confidence indicators"""
        features = {}
        
        # Home confidence (based on results, goals, position)
        home_results = match_data.get('home_results_last_5', [])
        home_goal_diff = match_data.get('home_recent_goal_diff', 0)
        home_position_change = match_data.get('home_position_change_last_5', 0)
        
        # Confidence score (0-10 scale)
        home_wins = sum(1 for r in home_results if r == 'W')
        features['home_confidence_score'] = (
            (home_wins / len(home_results) * 4) +  # 0-4 from win rate
            (min(home_goal_diff / 5, 3)) +  # 0-3 from goal diff
            (min(home_position_change, 3))  # 0-3 from position improvement
        ) if home_results else 5.0
        
        # Away confidence
        away_results = match_data.get('away_results_last_5', [])
        away_goal_diff = match_data.get('away_recent_goal_diff', 0)
        away_position_change = match_data.get('away_position_change_last_5', 0)
        
        away_wins = sum(1 for r in away_results if r == 'W')
        features['away_confidence_score'] = (
            (away_wins / len(away_results) * 4) +
            (min(away_goal_diff / 5, 3)) +
            (min(away_position_change, 3))
        ) if away_results else 5.0
        
        # Confidence differential
        features['confidence_differential'] = features['home_confidence_score'] - features['away_confidence_score']
        
        return features
    
    def _pressure_features(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """Identify teams under pressure"""
        features = {}
        
        # Winless streak pressure
        home_winless = match_data.get('home_winless_streak', 0)
        away_winless = match_data.get('away_winless_streak', 0)
        
        features['home_under_pressure'] = 1 if home_winless >= 3 else 0
        features['away_under_pressure'] = 1 if away_winless >= 3 else 0
        
        # Bounce-back potential (after heavy loss)
        home_last_loss_margin = match_data.get('home_last_loss_margin', 0)
        away_last_loss_margin = match_data.get('away_last_loss_margin', 0)
        
        features['home_bounce_back'] = 1 if home_last_loss_margin >= 3 else 0
        features['away_bounce_back'] = 1 if away_last_loss_margin >= 3 else 0
        
        # Overperformance risk (regression to mean)
        home_xg_diff = match_data.get('home_actual_vs_expected_points', 0)
        away_xg_diff = match_data.get('away_actual_vs_expected_points', 0)
        
        features['home_overperforming'] = max(0, home_xg_diff)
        features['away_overperforming'] = max(0, away_xg_diff)
        
        return features
    
    def _fatigue_features(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze fatigue and fixture congestion"""
        features = {}
        
        # Matches in last 7 days
        features['home_matches_last_7_days'] = match_data.get('home_matches_last_7_days', 1)
        features['away_matches_last_7_days'] = match_data.get('away_matches_last_7_days', 1)
        
        # Days since last match
        features['home_days_rest'] = match_data.get('home_days_since_last_match', 7)
        features['away_days_rest'] = match_data.get('away_days_since_last_match', 7)
        
        # Travel distance (for away team)
        features['away_travel_distance_km'] = match_data.get('away_travel_distance', 0)
        
        # Fatigue risk indicator
        features['home_fatigue_risk'] = 1 if features['home_matches_last_7_days'] >= 2 and features['home_days_rest'] < 4 else 0
        features['away_fatigue_risk'] = 1 if features['away_matches_last_7_days'] >= 2 and features['away_days_rest'] < 4 else 0
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Return list of all feature names"""
        return [
            'home_momentum_score', 'away_momentum_score', 'momentum_differential',
            'home_scoring_momentum', 'away_scoring_momentum',
            'home_hot_streak', 'away_hot_streak',
            'home_defensive_momentum', 'away_defensive_momentum',
            'home_clean_sheets_last_3', 'away_clean_sheets_last_3',
            'home_confidence_score', 'away_confidence_score', 'confidence_differential',
            'home_under_pressure', 'away_under_pressure',
            'home_bounce_back', 'away_bounce_back',
            'home_overperforming', 'away_overperforming',
            'home_matches_last_7_days', 'away_matches_last_7_days',
            'home_days_rest', 'away_days_rest',
            'away_travel_distance_km',
            'home_fatigue_risk', 'away_fatigue_risk'
        ]
