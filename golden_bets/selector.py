"""
Golden Bets Selector
Refactored to use integrated predictor with systematic selection
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from predictor.integrated_predictor import IntegratedPredictor


class GoldenBetsSelector:
    """
    Systematic Golden Bets selection using trained models
    """
    
    def __init__(
        self, 
        min_prob: float = 0.70,
        max_picks: int = 3,
        min_league_tier: int = 2
    ):
        """
        Initialize Golden Bets selector
        
        Args:
            min_prob: Minimum probability threshold
            max_picks: Maximum number of picks to return
            min_league_tier: Minimum league tier (1 = top tier)
        """
        self.min_prob = min_prob
        self.max_picks = max_picks
        self.min_league_tier = min_league_tier
        self.predictor = IntegratedPredictor()
    
    def select(self, predictions: List[Dict]) -> List[Dict]:
        """
        Select Golden Bets from predictions
        
        Args:
            predictions: List of dicts with match_id, market, match_data, league, etc.
            
        Returns:
            List of Golden Bet selections
        """
        golden_bets = []
        
        for pred in predictions:
            match_data = pred.get('match_data', {})
            league = pred.get('league', '')
            
            # Get predictions for all markets
            market_probs = self.predictor.predict_all_markets(match_data)
            
            # Find best market
            best_market = max(market_probs.items(), key=lambda x: x[1])
            market_name, probability = best_market
            
            # Check if meets Golden Bet criteria
            if probability >= self.min_prob:
                golden_bets.append({
                    'match_id': pred.get('match_id'),
                    'league': league,
                    'market': market_name,
                    'probability': probability,
                    'match_data': match_data,
                    'confidence': 'high' if probability >= 0.80 else 'medium'
                })
        
        # Sort by probability and take top N
        golden_bets.sort(key=lambda x: x['probability'], reverse=True)
        return golden_bets[:self.max_picks]
    
    def format_golden_bet(self, bet: Dict) -> Dict:
        """
        Format Golden Bet for API response
        
        Args:
            bet: Golden Bet dictionary
            
        Returns:
            Formatted response
        """
        market_names = {
            'goals': 'Over 2.5 Goals',
            'btts': 'Both Teams To Score - Yes',
            'cards': 'Over 3.5 Cards',
            'corners': 'Over 9.5 Corners'
        }
        
        return {
            'match_id': bet['match_id'],
            'league': bet['league'],
            'market_name': market_names.get(bet['market'], bet['market']),
            'selection': market_names.get(bet['market'], bet['market']),
            'probability': bet['probability'],
            'percentage': f"{bet['probability'] * 100:.1f}%",
            'confidence': bet['confidence']
        }
