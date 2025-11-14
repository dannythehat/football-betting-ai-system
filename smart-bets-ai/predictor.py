"""
Prediction Engine for Smart Bets AI
Generates predictions for upcoming matches
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from .feature_engineering import FeatureEngineer
from .model_trainer import SmartBetsModelTrainer


class SmartBetsPredictor:
    """
    Generates Smart Bets predictions for football matches
    """
    
    # Market display names
    MARKET_NAMES = {
        'home_win': 'Home Win',
        'draw': 'Draw',
        'away_win': 'Away Win',
        'over_0_5': 'Over 0.5 Goals',
        'over_1_5': 'Over 1.5 Goals',
        'over_2_5': 'Over 2.5 Goals',
        'over_3_5': 'Over 3.5 Goals',
        'over_4_5': 'Over 4.5 Goals',
        'btts': 'Both Teams To Score',
        'corners_over_8_5': 'Over 8.5 Corners',
        'corners_over_9_5': 'Over 9.5 Corners',
        'corners_over_10_5': 'Over 10.5 Corners',
        'cards_over_3_5': 'Over 3.5 Cards',
        'cards_over_4_5': 'Over 4.5 Cards'
    }
    
    # Market categories
    MARKET_CATEGORIES = {
        'match_result': ['home_win', 'draw', 'away_win'],
        'total_goals': ['over_0_5', 'over_1_5', 'over_2_5', 'over_3_5', 'over_4_5'],
        'btts': ['btts'],
        'corners': ['corners_over_8_5', 'corners_over_9_5', 'corners_over_10_5'],
        'cards': ['cards_over_3_5', 'cards_over_4_5']
    }
    
    def __init__(self, model_trainer: SmartBetsModelTrainer):
        """
        Initialize predictor with trained models
        
        Args:
            model_trainer: Trained SmartBetsModelTrainer instance
        """
        self.trainer = model_trainer
        self.feature_engineer = FeatureEngineer()
        
        if not self.trainer.models:
            raise ValueError("Model trainer has no trained models. Train or load models first.")
    
    def predict_match(self, match_data: pd.DataFrame) -> Dict:
        """
        Generate predictions for a single match
        
        Args:
            match_data: DataFrame with single match information
            
        Returns:
            Dictionary with predictions and Smart Bet recommendation
        """
        # Create features
        features = self.feature_engineer.create_features(match_data)
        
        # Get feature columns in correct order
        feature_cols = self.trainer.feature_names
        X = features[feature_cols]
        
        # Generate predictions for all markets
        all_predictions = {}
        
        for market_name, model in self.trainer.models.items():
            # Get probability prediction
            proba = model.predict_proba(X)[0, 1]
            
            all_predictions[market_name] = {
                'market_id': market_name,
                'market_name': self.MARKET_NAMES.get(market_name, market_name),
                'probability': float(proba),
                'percentage': f"{proba * 100:.1f}%"
            }
        
        # Find Smart Bet (highest probability across all markets)
        smart_bet = self._select_smart_bet(all_predictions)
        
        # Build response
        result = {
            'match_id': match_data['match_id'].iloc[0],
            'all_probabilities': all_predictions,
            'smart_bet': smart_bet,
            'prediction_timestamp': datetime.utcnow().isoformat(),
            'model_version': 'v1.0'
        }
        
        return result
    
    def predict_batch(self, matches_data: pd.DataFrame) -> List[Dict]:
        """
        Generate predictions for multiple matches
        
        Args:
            matches_data: DataFrame with multiple matches
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        for idx, row in matches_data.iterrows():
            match_df = pd.DataFrame([row])
            prediction = self.predict_match(match_df)
            predictions.append(prediction)
        
        return predictions
    
    def _select_smart_bet(self, all_predictions: Dict) -> Dict:
        """
        Select the Smart Bet (highest probability) from all predictions
        
        Args:
            all_predictions: Dictionary of all market predictions
            
        Returns:
            Dictionary with Smart Bet details
        """
        # Find market with highest probability
        best_market = max(
            all_predictions.items(),
            key=lambda x: x[1]['probability']
        )
        
        market_id = best_market[0]
        prediction = best_market[1]
        
        # Get alternative markets for context
        alternatives = self._get_alternative_markets(all_predictions, market_id)
        
        smart_bet = {
            'market_id': market_id,
            'market_name': prediction['market_name'],
            'probability': prediction['probability'],
            'percentage': prediction['percentage'],
            'confidence': self._get_confidence_level(prediction['probability']),
            'alternative_markets': alternatives
        }
        
        return smart_bet
    
    def _get_alternative_markets(
        self, 
        all_predictions: Dict, 
        selected_market: str,
        top_n: int = 3
    ) -> List[Dict]:
        """
        Get top alternative markets (excluding selected one)
        
        Args:
            all_predictions: All market predictions
            selected_market: The selected Smart Bet market
            top_n: Number of alternatives to return
            
        Returns:
            List of alternative market dictionaries
        """
        # Filter out selected market
        alternatives = {
            k: v for k, v in all_predictions.items() 
            if k != selected_market
        }
        
        # Sort by probability
        sorted_alternatives = sorted(
            alternatives.items(),
            key=lambda x: x[1]['probability'],
            reverse=True
        )
        
        # Return top N
        return [
            {
                'market_name': pred['market_name'],
                'probability': pred['probability'],
                'percentage': pred['percentage']
            }
            for _, pred in sorted_alternatives[:top_n]
        ]
    
    @staticmethod
    def _get_confidence_level(probability: float) -> str:
        """
        Convert probability to confidence level
        
        Args:
            probability: Probability value (0-1)
            
        Returns:
            Confidence level string
        """
        if probability >= 0.85:
            return 'very_high'
        elif probability >= 0.75:
            return 'high'
        elif probability >= 0.65:
            return 'medium'
        elif probability >= 0.55:
            return 'moderate'
        else:
            return 'low'
    
    def get_market_predictions_by_category(
        self, 
        all_predictions: Dict
    ) -> Dict[str, List[Dict]]:
        """
        Group predictions by market category
        
        Args:
            all_predictions: All market predictions
            
        Returns:
            Dictionary of predictions grouped by category
        """
        categorized = {}
        
        for category, markets in self.MARKET_CATEGORIES.items():
            categorized[category] = [
                all_predictions[market] 
                for market in markets 
                if market in all_predictions
            ]
        
        return categorized
    
    def analyze_custom_bet(
        self, 
        match_data: pd.DataFrame,
        bet_type: str
    ) -> Dict:
        """
        Analyze a user-selected custom bet
        
        Args:
            match_data: DataFrame with match information
            bet_type: The bet type to analyze (e.g., 'over_2_5')
            
        Returns:
            Dictionary with custom bet analysis
        """
        # Get full prediction
        full_prediction = self.predict_match(match_data)
        
        # Get the requested bet prediction
        if bet_type not in full_prediction['all_probabilities']:
            raise ValueError(f"Invalid bet type: {bet_type}")
        
        custom_bet = full_prediction['all_probabilities'][bet_type]
        smart_bet = full_prediction['smart_bet']
        
        # Determine verdict
        probability = custom_bet['probability']
        verdict = 'good' if probability >= 0.60 else 'risky' if probability >= 0.50 else 'poor'
        
        # Build analysis
        analysis = {
            'match_id': full_prediction['match_id'],
            'bet_type': bet_type,
            'bet_name': custom_bet['market_name'],
            'probability': probability,
            'percentage': custom_bet['percentage'],
            'verdict': verdict,
            'confidence': self._get_confidence_level(probability),
            'smart_bet_comparison': {
                'smart_bet_market': smart_bet['market_name'],
                'smart_bet_probability': smart_bet['probability'],
                'smart_bet_percentage': smart_bet['percentage'],
                'probability_difference': smart_bet['probability'] - probability,
                'is_smart_bet': bet_type == smart_bet['market_id']
            },
            'prediction_timestamp': full_prediction['prediction_timestamp']
        }
        
        # Add note if custom bet is significantly worse than Smart Bet
        if not analysis['smart_bet_comparison']['is_smart_bet']:
            prob_diff = analysis['smart_bet_comparison']['probability_difference']
            if prob_diff > 0.10:
                analysis['note'] = (
                    f"This bet has {prob_diff*100:.1f}% lower probability than our "
                    f"Smart Bet recommendation ({smart_bet['market_name']} at "
                    f"{smart_bet['percentage']}). Consider the Smart Bet for higher "
                    f"success probability."
                )
        
        return analysis


def format_prediction_for_api(prediction: Dict, include_all_markets: bool = False) -> Dict:
    """
    Format prediction for API response
    
    Args:
        prediction: Raw prediction dictionary
        include_all_markets: Whether to include all market probabilities
        
    Returns:
        Formatted prediction dictionary
    """
    formatted = {
        'match_id': prediction['match_id'],
        'smart_bet': {
            'market_id': prediction['smart_bet']['market_id'],
            'market_name': prediction['smart_bet']['market_name'],
            'probability': prediction['smart_bet']['probability'],
            'percentage': prediction['smart_bet']['percentage'],
            'confidence': prediction['smart_bet']['confidence'],
            'alternative_markets': prediction['smart_bet']['alternative_markets']
        },
        'prediction_timestamp': prediction['prediction_timestamp'],
        'model_version': prediction['model_version']
    }
    
    if include_all_markets:
        formatted['all_probabilities'] = prediction['all_probabilities']
    
    return formatted
