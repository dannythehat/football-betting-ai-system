"""
Core Statistics Engine
Advanced statistical features beyond basic averages
Generates 20+ features including rolling averages, variance, streaks, and consistency metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta


class CoreStatisticsEngine:
    """
    Advanced statistical feature engineering
    Creates rolling averages, weighted metrics, variance analysis, and streak detection
    """
    
    def __init__(self):
        self.feature_names = []
    
    def create_features(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Generate core statistical features for a match
        
        Args:
            match_data: Dictionary containing match and team statistics
            
        Returns:
            Dictionary of engineered features
        """
        features = {}
        
        # Rolling averages (last 5, 10 games)
        features.update(self._rolling_averages(match_data))
        
        # Weighted recent form (exponential decay)
        features.update(self._weighted_form(match_data))
        
        # Variance and consistency metrics
        features.update(self._variance_metrics(match_data))
        
        # Streak detection
        features.update(self._streak_features(match_data))
        
        # Home/Away splits
        features.update(self._venue_splits(match_data))
        
        # Time-based patterns
        features.update(self._time_patterns(match_data))
        
        return features
    
    def _rolling_averages(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate rolling averages for different windows"""
        features = {}
        
        # Goals rolling averages
        home_goals_history = match_data.get('home_goals_history', [])
        away_goals_history = match_data.get('away_goals_history', [])
        
        if len(home_goals_history) >= 5:
            features['home_goals_last_5'] = np.mean(home_goals_history[-5:])
            features['home_goals_last_10'] = np.mean(home_goals_history[-10:]) if len(home_goals_history) >= 10 else features['home_goals_last_5']
        else:
            features['home_goals_last_5'] = match_data.get('home_goals_avg', 0)
            features['home_goals_last_10'] = match_data.get('home_goals_avg', 0)
        
        if len(away_goals_history) >= 5:
            features['away_goals_last_5'] = np.mean(away_goals_history[-5:])
            features['away_goals_last_10'] = np.mean(away_goals_history[-10:]) if len(away_goals_history) >= 10 else features['away_goals_last_5']
        else:
            features['away_goals_last_5'] = match_data.get('away_goals_avg', 0)
            features['away_goals_last_10'] = match_data.get('away_goals_avg', 0)
        
        # Corners rolling averages
        home_corners_history = match_data.get('home_corners_history', [])
        away_corners_history = match_data.get('away_corners_history', [])
        
        features['home_corners_last_5'] = np.mean(home_corners_history[-5:]) if len(home_corners_history) >= 5 else match_data.get('home_corners_avg', 0)
        features['away_corners_last_5'] = np.mean(away_corners_history[-5:]) if len(away_corners_history) >= 5 else match_data.get('away_corners_avg', 0)
        
        # Cards rolling averages
        home_cards_history = match_data.get('home_cards_history', [])
        away_cards_history = match_data.get('away_cards_history', [])
        
        features['home_cards_last_5'] = np.mean(home_cards_history[-5:]) if len(home_cards_history) >= 5 else match_data.get('home_cards_avg', 0)
        features['away_cards_last_5'] = np.mean(away_cards_history[-5:]) if len(away_cards_history) >= 5 else match_data.get('away_cards_avg', 0)
        
        return features
    
    def _weighted_form(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate exponentially weighted form (recent games matter more)"""
        features = {}
        
        # Exponential decay weights (most recent = highest weight)
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        
        # Home team weighted form
        home_results = match_data.get('home_results_last_4', [])
        if len(home_results) >= 4:
            # Convert W/D/L to points (3/1/0)
            points = [3 if r == 'W' else 1 if r == 'D' else 0 for r in home_results[-4:]]
            features['home_weighted_form'] = np.average(points, weights=weights)
        else:
            features['home_weighted_form'] = self._form_to_points(match_data.get('home_form', ''))
        
        # Away team weighted form
        away_results = match_data.get('away_results_last_4', [])
        if len(away_results) >= 4:
            points = [3 if r == 'W' else 1 if r == 'D' else 0 for r in away_results[-4:]]
            features['away_weighted_form'] = np.average(points, weights=weights)
        else:
            features['away_weighted_form'] = self._form_to_points(match_data.get('away_form', ''))
        
        return features
    
    def _variance_metrics(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate variance and consistency metrics"""
        features = {}
        
        # Goals variance (consistency indicator)
        home_goals_history = match_data.get('home_goals_history', [])
        if len(home_goals_history) >= 5:
            features['home_goals_variance'] = np.var(home_goals_history[-10:])
            features['home_consistency_score'] = 1 / (1 + features['home_goals_variance'])
        else:
            features['home_goals_variance'] = 0.5
            features['home_consistency_score'] = 0.67
        
        away_goals_history = match_data.get('away_goals_history', [])
        if len(away_goals_history) >= 5:
            features['away_goals_variance'] = np.var(away_goals_history[-10:])
            features['away_consistency_score'] = 1 / (1 + features['away_goals_variance'])
        else:
            features['away_goals_variance'] = 0.5
            features['away_consistency_score'] = 0.67
        
        return features
    
    def _streak_features(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """Detect winning/scoring streaks"""
        features = {}
        
        # Win streaks
        home_results = match_data.get('home_results_last_10', [])
        features['home_win_streak'] = self._calculate_streak(home_results, 'W')
        features['home_unbeaten_streak'] = self._calculate_streak(home_results, ['W', 'D'])
        
        away_results = match_data.get('away_results_last_10', [])
        features['away_win_streak'] = self._calculate_streak(away_results, 'W')
        features['away_unbeaten_streak'] = self._calculate_streak(away_results, ['W', 'D'])
        
        # Scoring streaks
        home_goals_history = match_data.get('home_goals_history', [])
        features['home_scoring_streak'] = self._calculate_scoring_streak(home_goals_history)
        
        away_goals_history = match_data.get('away_goals_history', [])
        features['away_scoring_streak'] = self._calculate_scoring_streak(away_goals_history)
        
        return features
    
    def _venue_splits(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """Home/Away performance splits"""
        features = {}
        
        # Home team at home
        features['home_home_goals_avg'] = match_data.get('home_home_goals_avg', match_data.get('home_goals_avg', 0))
        features['home_home_conceded_avg'] = match_data.get('home_home_conceded_avg', match_data.get('home_goals_conceded_avg', 0))
        
        # Away team away
        features['away_away_goals_avg'] = match_data.get('away_away_goals_avg', match_data.get('away_goals_avg', 0))
        features['away_away_conceded_avg'] = match_data.get('away_away_conceded_avg', match_data.get('away_goals_conceded_avg', 0))
        
        # Venue advantage
        features['home_venue_advantage'] = features['home_home_goals_avg'] - match_data.get('home_goals_avg', 0)
        features['away_venue_disadvantage'] = match_data.get('away_goals_avg', 0) - features['away_away_goals_avg']
        
        return features
    
    def _time_patterns(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """Time-based scoring patterns (first/second half)"""
        features = {}
        
        # First half patterns
        features['home_first_half_goals_avg'] = match_data.get('home_first_half_goals_avg', match_data.get('home_goals_avg', 0) * 0.45)
        features['away_first_half_goals_avg'] = match_data.get('away_first_half_goals_avg', match_data.get('away_goals_avg', 0) * 0.45)
        
        # Second half patterns
        features['home_second_half_goals_avg'] = match_data.get('home_second_half_goals_avg', match_data.get('home_goals_avg', 0) * 0.55)
        features['away_second_half_goals_avg'] = match_data.get('away_second_half_goals_avg', match_data.get('away_goals_avg', 0) * 0.55)
        
        # Late goal tendency
        features['home_late_goals_rate'] = match_data.get('home_goals_after_75min_rate', 0.25)
        features['away_late_goals_rate'] = match_data.get('away_goals_after_75min_rate', 0.25)
        
        return features
    
    def _form_to_points(self, form_string: str) -> float:
        """Convert form string (WWDLW) to average points"""
        if not form_string:
            return 1.5
        points = sum(3 if c == 'W' else 1 if c == 'D' else 0 for c in form_string)
        return points / len(form_string) if form_string else 1.5
    
    def _calculate_streak(self, results: List[str], target: Any) -> int:
        """Calculate current streak of specific result(s)"""
        if not results:
            return 0
        
        streak = 0
        targets = [target] if isinstance(target, str) else target
        
        for result in reversed(results):
            if result in targets:
                streak += 1
            else:
                break
        
        return streak
    
    def _calculate_scoring_streak(self, goals_history: List[int]) -> int:
        """Calculate current scoring streak (games with goals)"""
        if not goals_history:
            return 0
        
        streak = 0
        for goals in reversed(goals_history):
            if goals > 0:
                streak += 1
            else:
                break
        
        return streak
    
    def get_feature_names(self) -> List[str]:
        """Return list of all feature names generated by this engine"""
        return [
            'home_goals_last_5', 'home_goals_last_10', 'away_goals_last_5', 'away_goals_last_10',
            'home_corners_last_5', 'away_corners_last_5', 'home_cards_last_5', 'away_cards_last_5',
            'home_weighted_form', 'away_weighted_form',
            'home_goals_variance', 'home_consistency_score', 'away_goals_variance', 'away_consistency_score',
            'home_win_streak', 'home_unbeaten_streak', 'away_win_streak', 'away_unbeaten_streak',
            'home_scoring_streak', 'away_scoring_streak',
            'home_home_goals_avg', 'home_home_conceded_avg', 'away_away_goals_avg', 'away_away_conceded_avg',
            'home_venue_advantage', 'away_venue_disadvantage',
            'home_first_half_goals_avg', 'away_first_half_goals_avg',
            'home_second_half_goals_avg', 'away_second_half_goals_avg',
            'home_late_goals_rate', 'away_late_goals_rate'
        ]
