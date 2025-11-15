"""
Head-to-Head Analyzer
Deep historical analysis between specific teams
Generates 15+ features from historical matchups
"""

import numpy as np
from typing import Dict, List, Any, Optional


class HeadToHeadAnalyzer:
    """
    Analyzes historical head-to-head matchups between teams
    Extracts patterns, trends, and venue-specific insights
    """
    
    def __init__(self, history_depth: int = 10):
        self.history_depth = history_depth
    
    def analyze_h2h(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Generate head-to-head features
        
        Args:
            match_data: Dictionary containing h2h history and team info
            
        Returns:
            Dictionary of h2h features
        """
        features = {}
        
        h2h_history = match_data.get('h2h_history', [])
        
        if not h2h_history:
            # Return default features if no h2h data
            return self._default_h2h_features()
        
        # Historical outcomes
        features.update(self._outcome_features(h2h_history, match_data))
        
        # Scoring patterns
        features.update(self._scoring_patterns(h2h_history))
        
        # Recent trends
        features.update(self._trend_analysis(h2h_history))
        
        # Venue-specific analysis
        features.update(self._venue_analysis(h2h_history, match_data))
        
        # Dominance indicators
        features.update(self._dominance_features(h2h_history, match_data))
        
        return features
    
    def _outcome_features(self, h2h_history: List[Dict], match_data: Dict) -> Dict[str, float]:
        """Historical win/draw/loss patterns"""
        features = {}
        
        home_team = match_data.get('home_team')
        total_matches = len(h2h_history)
        
        if total_matches == 0:
            return {'h2h_home_wins': 0, 'h2h_away_wins': 0, 'h2h_draws': 0}
        
        home_wins = sum(1 for match in h2h_history if match.get('winner') == home_team)
        draws = sum(1 for match in h2h_history if match.get('result') == 'D')
        away_wins = total_matches - home_wins - draws
        
        features['h2h_home_wins'] = home_wins
        features['h2h_away_wins'] = away_wins
        features['h2h_draws'] = draws
        features['h2h_home_win_rate'] = home_wins / total_matches
        features['h2h_draw_rate'] = draws / total_matches
        
        return features
    
    def _scoring_patterns(self, h2h_history: List[Dict]) -> Dict[str, float]:
        """Historical scoring statistics"""
        features = {}
        
        if not h2h_history:
            return self._default_scoring_features()
        
        total_goals = [match.get('total_goals', 0) for match in h2h_history]
        total_corners = [match.get('total_corners', 0) for match in h2h_history]
        total_cards = [match.get('total_cards', 0) for match in h2h_history]
        
        features['h2h_avg_total_goals'] = np.mean(total_goals)
        features['h2h_avg_total_corners'] = np.mean(total_corners)
        features['h2h_avg_total_cards'] = np.mean(total_cards)
        
        # BTTS rate
        btts_count = sum(1 for match in h2h_history 
                        if match.get('home_goals', 0) > 0 and match.get('away_goals', 0) > 0)
        features['h2h_btts_rate'] = btts_count / len(h2h_history)
        
        # Over 2.5 goals rate
        over_2_5_count = sum(1 for goals in total_goals if goals > 2.5)
        features['h2h_over_2_5_rate'] = over_2_5_count / len(h2h_history)
        
        return features
    
    def _trend_analysis(self, h2h_history: List[Dict]) -> Dict[str, float]:
        """Recent vs historical trend comparison"""
        features = {}
        
        if len(h2h_history) < 3:
            return {'h2h_recent_trend': 0, 'h2h_goals_trend': 0}
        
        # Compare last 3 matches vs overall average
        recent_matches = h2h_history[:3]
        recent_goals = np.mean([m.get('total_goals', 0) for m in recent_matches])
        overall_goals = np.mean([m.get('total_goals', 0) for m in h2h_history])
        
        features['h2h_recent_trend'] = recent_goals - overall_goals
        features['h2h_goals_trend'] = 1 if recent_goals > overall_goals else 0
        
        # Recent corners trend
        recent_corners = np.mean([m.get('total_corners', 0) for m in recent_matches])
        overall_corners = np.mean([m.get('total_corners', 0) for m in h2h_history])
        features['h2h_corners_trend'] = recent_corners - overall_corners
        
        return features
    
    def _venue_analysis(self, h2h_history: List[Dict], match_data: Dict) -> Dict[str, float]:
        """Venue-specific h2h analysis"""
        features = {}
        
        home_team = match_data.get('home_team')
        
        # Filter matches at current home venue
        home_venue_matches = [
            match for match in h2h_history 
            if match.get('home_team') == home_team
        ]
        
        if not home_venue_matches:
            return {'h2h_home_venue_goals_avg': 0, 'h2h_home_venue_advantage': 0}
        
        home_venue_goals = np.mean([m.get('total_goals', 0) for m in home_venue_matches])
        overall_goals = np.mean([m.get('total_goals', 0) for m in h2h_history])
        
        features['h2h_home_venue_goals_avg'] = home_venue_goals
        features['h2h_home_venue_advantage'] = home_venue_goals - overall_goals
        
        return features
    
    def _dominance_features(self, h2h_history: List[Dict], match_data: Dict) -> Dict[str, float]:
        """Calculate dominance indicators"""
        features = {}
        
        if not h2h_history:
            return {'h2h_avg_goal_difference': 0, 'h2h_dominance_score': 0}
        
        home_team = match_data.get('home_team')
        
        # Average goal difference
        goal_diffs = []
        for match in h2h_history:
            home_goals = match.get('home_goals', 0)
            away_goals = match.get('away_goals', 0)
            
            # Adjust based on which team was home
            if match.get('home_team') == home_team:
                goal_diffs.append(home_goals - away_goals)
            else:
                goal_diffs.append(away_goals - home_goals)
        
        features['h2h_avg_goal_difference'] = np.mean(goal_diffs)
        features['h2h_goal_diff_variance'] = np.var(goal_diffs)
        
        # Dominance score (weighted by margin of victory)
        dominance = sum(1 if gd > 1 else 0.5 if gd > 0 else 0 for gd in goal_diffs)
        features['h2h_dominance_score'] = dominance / len(goal_diffs)
        
        return features
    
    def _default_h2h_features(self) -> Dict[str, float]:
        """Default features when no h2h data available"""
        return {
            'h2h_home_wins': 0,
            'h2h_away_wins': 0,
            'h2h_draws': 0,
            'h2h_home_win_rate': 0.33,
            'h2h_draw_rate': 0.33,
            'h2h_avg_total_goals': 2.5,
            'h2h_avg_total_corners': 10.0,
            'h2h_avg_total_cards': 3.5,
            'h2h_btts_rate': 0.5,
            'h2h_over_2_5_rate': 0.5,
            'h2h_recent_trend': 0,
            'h2h_goals_trend': 0,
            'h2h_corners_trend': 0,
            'h2h_home_venue_goals_avg': 2.5,
            'h2h_home_venue_advantage': 0,
            'h2h_avg_goal_difference': 0,
            'h2h_goal_diff_variance': 1.0,
            'h2h_dominance_score': 0.5
        }
    
    def _default_scoring_features(self) -> Dict[str, float]:
        """Default scoring features"""
        return {
            'h2h_avg_total_goals': 2.5,
            'h2h_avg_total_corners': 10.0,
            'h2h_avg_total_cards': 3.5,
            'h2h_btts_rate': 0.5,
            'h2h_over_2_5_rate': 0.5
        }
    
    def get_feature_names(self) -> List[str]:
        """Return list of all feature names"""
        return [
            'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
            'h2h_home_win_rate', 'h2h_draw_rate',
            'h2h_avg_total_goals', 'h2h_avg_total_corners', 'h2h_avg_total_cards',
            'h2h_btts_rate', 'h2h_over_2_5_rate',
            'h2h_recent_trend', 'h2h_goals_trend', 'h2h_corners_trend',
            'h2h_home_venue_goals_avg', 'h2h_home_venue_advantage',
            'h2h_avg_goal_difference', 'h2h_goal_diff_variance', 'h2h_dominance_score'
        ]
