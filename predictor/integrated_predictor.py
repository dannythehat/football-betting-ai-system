"""
Integrated Predictor
Loads trained ensemble models with calibration for all markets
"""

import sys
from pathlib import Path
import pickle
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from features.feature_builder import FeatureBuilder
from training.config import MODELS_DIR, ENSEMBLE_WEIGHTS
from training.utils import ensemble_predictions, apply_calibration


class IntegratedPredictor:
    """
    Integrated predictor using trained ensemble models with calibration
    Replaces placeholder logic with real ML predictions
    """
    
    def __init__(self, models_dir: str = None):
        """
        Initialize predictor with trained models
        
        Args:
            models_dir: Path to models directory (optional)
        """
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.feature_builder = FeatureBuilder()
        
        # Storage for loaded models
        self.models = {
            'goals': {},
            'btts': {},
            'cards': {},
            'corners': {}
        }
        self.calibration_models = {}
        self.metadata = {}
        
        # Load all models
        self._load_all_models()
    
    def _load_all_models(self):
        """Load all trained models and calibration"""
        markets = ['goals', 'btts', 'cards', 'corners']
        
        for market in markets:
            try:
                self._load_market_models(market)
                print(f"✅ Loaded {market} models")
            except Exception as e:
                print(f"⚠️  Warning: Could not load {market} models: {e}")
    
    def _load_market_models(self, market: str):
        """Load models for a specific market"""
        market_dir = self.models_dir / market
        
        if not market_dir.exists():
            raise FileNotFoundError(f"Market directory not found: {market_dir}")
        
        # Load ensemble metadata
        ensemble_meta_path = market_dir / 'ensemble_metadata.json'
        if ensemble_meta_path.exists():
            with open(ensemble_meta_path, 'r') as f:
                self.metadata[market] = json.load(f)
        
        # Load base models
        base_models = self.metadata.get(market, {}).get('base_models', ['xgboost', 'lightgbm', 'logistic'])
        
        for model_type in base_models:
            model_path = market_dir / f"{model_type}_model.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.models[market][model_type] = pickle.load(f)
        
        # Load calibration model
        calib_path = market_dir / 'ensemble_calibration.pkl'
        if calib_path.exists():
            with open(calib_path, 'rb') as f:
                self.calibration_models[market] = pickle.load(f)
    
    def predict_for_match(self, market: str, match_data: Dict) -> float:
        """
        Predict probability for a specific market and match
        
        Args:
            market: Market name ('goals', 'btts', 'cards', 'corners')
            match_data: Dictionary with match information and team stats
            
        Returns:
            Calibrated probability
        """
        if market not in self.models or not self.models[market]:
            raise ValueError(f"No models loaded for market: {market}")
        
        # Build features
        features = self.feature_builder.build_features(match_data)
        feature_names = self.feature_builder.get_feature_names()
        
        # Create DataFrame with correct feature order
        X = pd.DataFrame([features])[feature_names].fillna(0)
        
        # Get predictions from all base models
        predictions = {}
        for model_type, model in self.models[market].items():
            try:
                proba = model.predict_proba(X)[0, 1]
                predictions[model_type] = proba
            except Exception as e:
                print(f"⚠️  Warning: Error predicting with {model_type}: {e}")
                continue
        
        if not predictions:
            raise ValueError(f"No successful predictions for {market}")
        
        # Ensemble predictions
        weights = self.metadata.get(market, {}).get('weights', ENSEMBLE_WEIGHTS)
        ensemble_proba = ensemble_predictions(
            {k: np.array([v]) for k, v in predictions.items()},
            weights
        )[0]
        
        # Apply calibration if available
        if market in self.calibration_models:
            calibration_method = self.metadata.get(market, {}).get('calibration_method', 'isotonic')
            calibrated_proba = apply_calibration(
                self.calibration_models[market],
                np.array([ensemble_proba]),
                calibration_method
            )[0]
            return float(calibrated_proba)
        
        return float(ensemble_proba)
    
    def predict_all_markets(self, match_data: Dict) -> Dict[str, float]:
        """
        Predict probabilities for all markets
        
        Args:
            match_data: Dictionary with match information
            
        Returns:
            Dictionary mapping market names to probabilities
        """
        results = {}
        
        for market in ['goals', 'btts', 'cards', 'corners']:
            try:
                prob = self.predict_for_match(market, match_data)
                results[market] = prob
            except Exception as e:
                print(f"⚠️  Warning: Could not predict {market}: {e}")
                results[market] = 0.5  # Default probability
        
        return results
    
    def get_smart_bet(self, match_data: Dict) -> Dict:
        """
        Get Smart Bet recommendation (highest probability market)
        
        Args:
            match_data: Dictionary with match information
            
        Returns:
            Dictionary with Smart Bet details
        """
        # Get all predictions
        predictions = self.predict_all_markets(match_data)
        
        # Find highest probability
        best_market = max(predictions.items(), key=lambda x: x[1])
        market_name = best_market[0]
        probability = best_market[1]
        
        # Market display names
        market_names = {
            'goals': 'Over 2.5 Goals',
            'btts': 'Both Teams To Score - Yes',
            'cards': 'Over 3.5 Cards',
            'corners': 'Over 9.5 Corners'
        }
        
        return {
            'market': market_name,
            'selection': market_names[market_name],
            'probability': probability,
            'percentage': f"{probability * 100:.1f}%",
            'all_probabilities': predictions
        }
    
    def calculate_value_bet(
        self, 
        market: str, 
        match_data: Dict, 
        odds: float,
        min_value_pct: float = 5.0
    ) -> Optional[Dict]:
        """
        Calculate if a bet has value
        
        Args:
            market: Market name
            match_data: Match information
            odds: Decimal odds
            min_value_pct: Minimum value percentage
            
        Returns:
            Dictionary with value bet info or None if no value
        """
        ai_prob = self.predict_for_match(market, match_data)
        implied_prob = 1 / odds
        
        # Calculate value
        value_pct = ((ai_prob / implied_prob) - 1) * 100
        
        if value_pct >= min_value_pct:
            fair_odds = 1 / ai_prob
            ev = (ai_prob * odds) - 1
            
            return {
                'market': market,
                'ai_probability': ai_prob,
                'implied_probability': implied_prob,
                'odds': odds,
                'fair_odds': fair_odds,
                'value_pct': value_pct,
                'expected_value': ev
            }
        
        return None
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        info = {
            'markets': list(self.models.keys()),
            'models_per_market': {
                market: list(models.keys()) 
                for market, models in self.models.items()
            },
            'has_calibration': {
                market: market in self.calibration_models
                for market in self.models.keys()
            },
            'metadata': self.metadata
        }
        return info


# Convenience function for backward compatibility
def create_predictor() -> IntegratedPredictor:
    """Create and return an integrated predictor instance"""
    return IntegratedPredictor()


if __name__ == "__main__":
    # Test predictor
    predictor = IntegratedPredictor()
    
    # Test match
    test_match = {
        'home_goals_avg_5': 1.8,
        'away_goals_avg_5': 2.1,
        'home_goals_conceded_avg_5': 1.0,
        'away_goals_conceded_avg_5': 0.8,
        'home_corners_avg_5': 6.2,
        'away_corners_avg_5': 5.8,
        'home_cards_avg_5': 2.1,
        'away_cards_avg_5': 1.9,
        'home_btts_rate_5': 0.65,
        'away_btts_rate_5': 0.70
    }
    
    print("\n" + "=" * 60)
    print("INTEGRATED PREDICTOR TEST")
    print("=" * 60)
    
    try:
        smart_bet = predictor.get_smart_bet(test_match)
        print(f"\n✅ Smart Bet: {smart_bet['selection']}")
        print(f"   Probability: {smart_bet['percentage']}")
        print(f"\n   All Probabilities:")
        for market, prob in smart_bet['all_probabilities'].items():
            print(f"   - {market}: {prob:.1%}")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please train models first")
