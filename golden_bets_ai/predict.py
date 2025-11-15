"""Golden Bets Prediction Pipeline"""
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
from filter import GoldenBetsFilter
import sys
sys.path.append('..')
from smart_bets_ai.predict import SmartBetsPredictor

class GoldenBetsPredictor:
    """Generates Golden Bets from Smart Bets predictions"""
    
    def __init__(self):
        self.smart_bets_predictor = SmartBetsPredictor()
        self.filter = GoldenBetsFilter()
    
    def predict(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate Golden Bets predictions
        
        Args:
            matches: List of match data dictionaries
            
        Returns:
            List of Golden Bets (1-3 daily picks)
        """
        # Get Smart Bets predictions first
        smart_bets = self.smart_bets_predictor.predict(matches)
        
        # Get model probabilities for ensemble agreement
        model_probs = self.smart_bets_predictor.get_model_probabilities(matches)
        
        # Filter for Golden Bets
        golden_bets = self.filter.filter_golden_bets(
            smart_bets_predictions=smart_bets,
            model_probabilities=model_probs
        )
        
        # Add Golden Bets specific reasoning
        for bet in golden_bets:
            bet['reasoning'] = self.filter.generate_reasoning(bet)
            bet['bet_category'] = 'golden'
        
        return golden_bets

if __name__ == '__main__':
    # Test with sample data
    test_data_path = Path('../test-data/upcoming_matches_sample.json')
    
    if test_data_path.exists():
        with open(test_data_path) as f:
            matches = json.load(f)
        
        predictor = GoldenBetsPredictor()
        golden_bets = predictor.predict(matches)
        
        print(f"\n{'='*60}")
        print(f"GOLDEN BETS PREDICTIONS")
        print(f"{'='*60}\n")
        print(f"Found {len(golden_bets)} Golden Bets:\n")
        
        for i, bet in enumerate(golden_bets, 1):
            print(f"{i}. {bet['home_team']} vs {bet['away_team']}")
            print(f"   Market: {bet['market_name']}")
            print(f"   Prediction: {bet['selection_name']}")
            print(f"   Confidence: {bet['confidence_score']:.1%}")
            print(f"   Agreement: {bet['ensemble_agreement']:.1%}")
            print(f"   {bet['reasoning']}\n")
    else:
        print(f"Test data not found at {test_data_path}")
        print("Please create test data or run with actual match data")
