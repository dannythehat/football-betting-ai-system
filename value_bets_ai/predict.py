"""
Value Bets Prediction Engine
Identifies value betting opportunities from Smart Bets predictions and odds
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from smart_bets_ai.predict import SmartBetsPredictor
from calculator import ValueCalculator
from config import MAX_DAILY_PICKS


class ValueBetsPredictor:
    """Generates Value Bets from Smart Bets predictions and odds"""
    
    def __init__(self):
        self.smart_bets_predictor = SmartBetsPredictor()
        self.calculator = ValueCalculator()
    
    def predict(self, matches_with_odds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate Value Bets predictions
        
        Args:
            matches_with_odds: List of match data with odds for each market
                Example structure:
                {
                    'match_id': '12345',
                    'home_team': 'Team A',
                    'away_team': 'Team B',
                    ... (match features),
                    'odds': {
                        'goals_over_2_5': 2.10,
                        'goals_under_2_5': 1.75,
                        'cards_over_3_5': 2.50,
                        'cards_under_3_5': 1.55,
                        'corners_over_9_5': 1.90,
                        'corners_under_9_5': 1.95,
                        'btts_yes': 1.85,
                        'btts_no': 2.00
                    }
                }
        
        Returns:
            List of Value Bets (top 3 daily picks with positive EV)
        """
        # Get Smart Bets predictions (probabilities for all markets)
        smart_bets = self.smart_bets_predictor.predict_batch(matches_with_odds)
        
        # Calculate value for all predictions
        value_bets = []
        
        for match_data, smart_bet in zip(matches_with_odds, smart_bets):
            # Extract odds from match data
            odds = match_data.get('odds', {})
            
            if not odds:
                continue
            
            # Get all market probabilities from Smart Bets
            all_markets = smart_bet.get('all_markets', {})
            
            # Check each market for value
            for market_key, probability in all_markets.items():
                # Map market key to odds key
                odds_key = self._map_market_to_odds_key(market_key)
                
                if odds_key not in odds:
                    continue
                
                decimal_odds = odds[odds_key]
                
                # Calculate value metrics
                metrics = self.calculator.calculate_all_metrics(probability, decimal_odds)
                
                # Only include if it's a value bet
                if metrics['is_value_bet']:
                    value_bet = {
                        'match_id': match_data['match_id'],
                        'home_team': match_data['home_team'],
                        'away_team': match_data['away_team'],
                        'market_name': self._format_market_name(market_key),
                        'selection_name': self._format_selection_name(market_key),
                        'ai_probability': probability,
                        'decimal_odds': decimal_odds,
                        'implied_probability': metrics['implied_probability'],
                        'value_percentage': metrics['value_percentage'],
                        'expected_value': metrics['expected_value'],
                        'value_score': metrics['value_score'],
                        'bet_category': 'value'
                    }
                    
                    # Add reasoning
                    value_bet['reasoning'] = self._generate_reasoning(value_bet)
                    
                    value_bets.append(value_bet)
        
        # Sort by value score (highest first)
        value_bets.sort(key=lambda x: x['value_score'], reverse=True)
        
        # Return top 3 value bets
        return value_bets[:MAX_DAILY_PICKS]
    
    def _map_market_to_odds_key(self, market_key: str) -> str:
        """Map Smart Bets market key to odds dictionary key"""
        market_mapping = {
            'goals_over': 'goals_over_2_5',
            'goals_under': 'goals_under_2_5',
            'cards_over': 'cards_over_3_5',
            'cards_under': 'cards_under_3_5',
            'corners_over': 'corners_over_9_5',
            'corners_under': 'corners_under_9_5',
            'btts_yes': 'btts_yes',
            'btts_no': 'btts_no'
        }
        return market_mapping.get(market_key, market_key)
    
    def _format_market_name(self, market_key: str) -> str:
        """Format market key into readable market name"""
        market_names = {
            'goals_over': 'Total Goals',
            'goals_under': 'Total Goals',
            'cards_over': 'Total Cards',
            'cards_under': 'Total Cards',
            'corners_over': 'Total Corners',
            'corners_under': 'Total Corners',
            'btts_yes': 'Both Teams To Score',
            'btts_no': 'Both Teams To Score'
        }
        return market_names.get(market_key, market_key)
    
    def _format_selection_name(self, market_key: str) -> str:
        """Format market key into readable selection name"""
        selection_names = {
            'goals_over': 'Over 2.5',
            'goals_under': 'Under 2.5',
            'cards_over': 'Over 3.5',
            'cards_under': 'Under 3.5',
            'corners_over': 'Over 9.5',
            'corners_under': 'Under 9.5',
            'btts_yes': 'Yes',
            'btts_no': 'No'
        }
        return selection_names.get(market_key, market_key)
    
    def _generate_reasoning(self, value_bet: Dict[str, Any]) -> str:
        """Generate detailed reasoning for value bet"""
        ai_prob = value_bet['ai_probability']
        implied_prob = value_bet['implied_probability']
        value_pct = value_bet['value_percentage']
        ev = value_bet['expected_value']
        odds = value_bet['decimal_odds']
        
        reasoning = (
            f"AI probability ({ai_prob:.1%}) significantly exceeds "
            f"bookmaker's implied probability ({implied_prob:.1%}). "
            f"Value: +{value_pct:.1%}. "
            f"Expected Value: +{ev:.1%}. "
            f"At odds of {odds:.2f}, this bet offers strong long-term profit potential. "
            f"Value bets focus on long-term profitability rather than individual bet success rate."
        )
        
        return reasoning


if __name__ == '__main__':
    # Test with sample data
    import json
    
    test_data_path = Path('../test-data/upcoming_matches_with_odds_sample.json')
    
    if test_data_path.exists():
        with open(test_data_path) as f:
            matches = json.load(f)
        
        predictor = ValueBetsPredictor()
        value_bets = predictor.predict(matches)
        
        print(f"\n{'='*60}")
        print(f"VALUE BETS PREDICTIONS")
        print(f"{'='*60}\n")
        print(f"Found {len(value_bets)} Value Bets:\n")
        
        for i, bet in enumerate(value_bets, 1):
            print(f"{i}. {bet['home_team']} vs {bet['away_team']}")
            print(f"   Market: {bet['market_name']}")
            print(f"   Selection: {bet['selection_name']}")
            print(f"   AI Probability: {bet['ai_probability']:.1%}")
            print(f"   Odds: {bet['decimal_odds']:.2f}")
            print(f"   Value: +{bet['value_percentage']:.1%}")
            print(f"   Expected Value: +{bet['expected_value']:.1%}")
            print(f"   {bet['reasoning']}\n")
    else:
        print(f"Test data not found at {test_data_path}")
        print("Please create test data with odds or run with actual match data")
