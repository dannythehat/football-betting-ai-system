"""
Prediction Service for Smart Bets AI
Generates predictions for the 4 target markets and selects best bet per fixture
"""

import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from .features import FeatureEngineer


class SmartBetsPredictor:
    """
    Generates Smart Bets predictions
    Returns the highest probability bet across all 4 markets for each fixture
    """
    
    def __init__(self, models_dir: str = "smart-bets-ai/models"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.feature_engineer = None
        self.metadata = {}
        
        # Market definitions
        self.markets = {
            'goals': {
                'market_id': 'total_goals',
                'market_name': 'Total Goals',
                'selection_id': 'over_2.5',
                'selection_name': 'Over 2.5 Goals'
            },
            'cards': {
                'market_id': 'total_cards',
                'market_name': 'Total Cards',
                'selection_id': 'over_3.5',
                'selection_name': 'Over 3.5 Cards'
            },
            'corners': {
                'market_id': 'total_corners',
                'market_name': 'Total Corners',
                'selection_id': 'over_9.5',
                'selection_name': 'Over 9.5 Corners'
            },
            'btts': {
                'market_id': 'btts',
                'market_name': 'Both Teams To Score',
                'selection_id': 'yes',
                'selection_name': 'Yes'
            }
        }
        
        self.load_models()
    
    def load_models(self):
        """Load trained models and feature engineer"""
        try:
            # Load models
            for market in ['goals', 'cards', 'corners', 'btts']:
                model_path = self.models_dir / f"{market}_model.pkl"
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        self.models[market] = pickle.load(f)
            
            # Load feature engineer
            fe_path = self.models_dir / "feature_engineer.pkl"
            if fe_path.exists():
                with open(fe_path, 'rb') as f:
                    self.feature_engineer = pickle.load(f)
            else:
                # Fallback to new instance
                self.feature_engineer = FeatureEngineer()
            
            # Load metadata
            metadata_path = self.models_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            
            print(f"✅ Loaded {len(self.models)} models")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not load models: {e}")
            print("Models need to be trained first. Run train.py")
    
    def predict_match(self, match_data: Dict) -> Dict:
        """
        Generate predictions for a single match across all 4 markets
        
        Args:
            match_data: Dictionary with match information and team stats
            
        Returns:
            Dictionary with predictions for all markets
        """
        if not self.models:
            raise ValueError("Models not loaded. Please train models first.")
        
        # Convert to DataFrame
        df = pd.DataFrame([match_data])
        
        # Create features
        df_features = self.feature_engineer.create_features(df)
        X = df_features[self.feature_engineer.get_feature_columns()]
        X = X.fillna(X.mean())
        
        # Get predictions for each market
        predictions = {}
        
        for market, model in self.models.items():
            # Get probability
            proba = model.predict_proba(X)[0, 1]
            
            # Get market info
            market_info = self.markets[market]
            
            predictions[market] = {
                'market_id': market_info['market_id'],
                'market_name': market_info['market_name'],
                'selection_id': market_info['selection_id'],
                'selection_name': market_info['selection_name'],
                'probability': float(proba),
                'percentage': f"{proba * 100:.1f}%"
            }
        
        return predictions
    
    def get_smart_bet(self, match_data: Dict) -> Dict:
        """
        Get the Smart Bet for a match (highest probability across all 4 markets)
        
        Args:
            match_data: Dictionary with match information and team stats
            
        Returns:
            Dictionary with Smart Bet recommendation
        """
        # Get all predictions
        predictions = self.predict_match(match_data)
        
        # Find highest probability
        best_market = max(predictions.items(), key=lambda x: x[1]['probability'])
        market_name = best_market[0]
        best_pred = best_market[1]
        
        # Get alternative markets (sorted by probability)
        alternatives = []
        for market, pred in predictions.items():
            if market != market_name:
                alternatives.append({
                    'market_name': pred['selection_name'],
                    'probability': pred['probability']
                })
        
        alternatives.sort(key=lambda x: x['probability'], reverse=True)
        
        # Build Smart Bet response
        smart_bet = {
            'market_id': best_pred['market_id'],
            'market_name': best_pred['market_name'],
            'selection_id': best_pred['selection_id'],
            'selection_name': best_pred['selection_name'],
            'probability': best_pred['probability'],
            'percentage': best_pred['percentage'],
            'explanation': self._generate_explanation(match_data, market_name, best_pred),
            'alternative_markets': alternatives
        }
        
        return smart_bet
    
    def predict_batch(self, matches: List[Dict]) -> List[Dict]:
        """
        Generate Smart Bets for multiple matches
        
        Args:
            matches: List of match dictionaries
            
        Returns:
            List of Smart Bet predictions
        """
        results = []
        
        for match in matches:
            try:
                smart_bet = self.get_smart_bet(match)
                results.append({
                    'match_id': match.get('match_id'),
                    'smart_bet': smart_bet
                })
            except Exception as e:
                print(f"❌ Error predicting match {match.get('match_id')}: {e}")
                continue
        
        return results
    
    def _generate_explanation(
        self, 
        match_data: Dict, 
        market: str, 
        prediction: Dict
    ) -> str:
        """
        Generate explanation for Smart Bet recommendation
        
        Args:
            match_data: Match data
            market: Market name (goals, cards, corners, btts)
            prediction: Prediction dictionary
            
        Returns:
            Explanation string
        """
        prob = prediction['probability']
        
        # Market-specific explanations
        if market == 'goals':
            combined_avg = match_data.get('home_goals_avg', 0) + match_data.get('away_goals_avg', 0)
            return (
                f"Highest probability outcome across all 4 analyzed markets for this fixture. "
                f"Combined goals average of {combined_avg:.1f} per match with consistent "
                f"historical data supporting this prediction."
            )
        
        elif market == 'cards':
            combined_cards = match_data.get('home_cards_avg', 0) + match_data.get('away_cards_avg', 0)
            return (
                f"Highest probability outcome across all 4 analyzed markets for this fixture. "
                f"Combined cards average of {combined_cards:.1f} per match indicates high "
                f"disciplinary activity expected."
            )
        
        elif market == 'corners':
            combined_corners = match_data.get('home_corners_avg', 0) + match_data.get('away_corners_avg', 0)
            return (
                f"Highest probability outcome across all 4 analyzed markets for this fixture. "
                f"Combined corners average of {combined_corners:.1f} per match with strong "
                f"attacking patterns from both teams."
            )
        
        elif market == 'btts':
            btts_rate = (match_data.get('home_btts_rate', 0) + match_data.get('away_btts_rate', 0)) / 2
            return (
                f"Highest probability outcome across all 4 analyzed markets for this fixture. "
                f"Both teams have strong scoring records with combined BTTS rate of {btts_rate:.1%}."
            )
        
        return f"Highest probability outcome across all 4 analyzed markets for this fixture."
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        return {
            'models_loaded': list(self.models.keys()),
            'metadata': self.metadata,
            'markets': self.markets
        }


def main():
    """Test prediction service"""
    predictor = SmartBetsPredictor()
    
    # Test with sample match
    test_match = {
        'match_id': 'TEST_001',
        'home_team': 'Manchester United',
        'away_team': 'Liverpool',
        'home_goals_avg': 1.8,
        'away_goals_avg': 2.1,
        'home_goals_conceded_avg': 1.0,
        'away_goals_conceded_avg': 0.8,
        'home_corners_avg': 6.2,
        'away_corners_avg': 5.8,
        'home_cards_avg': 2.1,
        'away_cards_avg': 1.9,
        'home_btts_rate': 0.65,
        'away_btts_rate': 0.70,
        'home_form': 'WWDWL',
        'away_form': 'WWWDW'
    }
    
    print("\n" + "=" * 60)
    print("SMART BETS PREDICTION TEST")
    print("=" * 60)
    
    try:
        smart_bet = predictor.get_smart_bet(test_match)
        print(f"\n✅ Smart Bet for {test_match['home_team']} vs {test_match['away_team']}:")
        print(f"   Market: {smart_bet['selection_name']}")
        print(f"   Probability: {smart_bet['percentage']}")
        print(f"   Explanation: {smart_bet['explanation']}")
        print(f"\n   Alternative Markets:")
        for alt in smart_bet['alternative_markets']:
            print(f"   - {alt['market_name']}: {alt['probability']:.1%}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please train models first by running: python smart-bets-ai/train.py")


if __name__ == "__main__":
    main()
