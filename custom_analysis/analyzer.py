"""
Custom Bet Analyzer
Analyzes user-selected fixtures and bet types on-demand
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Any
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from smart_bets_ai.predict import SmartBetsPredictor
from config import (
    CONFIDENCE_LEVELS,
    VERDICT_MESSAGES,
    SUPPORTED_MARKETS,
    EDUCATIONAL_NOTES
)

logger = logging.getLogger(__name__)


class CustomBetAnalyzer:
    """
    Analyzes user-selected bets and provides educational feedback
    """
    
    def __init__(self):
        """Initialize analyzer with Smart Bets predictor"""
        try:
            self.smart_predictor = SmartBetsPredictor()
            logger.info("✅ Custom Bet Analyzer initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Smart Bets predictor: {e}")
            raise
    
    def analyze_custom_bet(
        self,
        match_data: Dict[str, Any],
        market_id: str,
        selection_id: str
    ) -> Dict[str, Any]:
        """
        Analyze a user-selected bet
        
        Args:
            match_data: Match information and team stats
            market_id: Market identifier (e.g., 'total_goals')
            selection_id: Selection identifier (e.g., 'over_2.5')
            
        Returns:
            Analysis result with probability, verdict, and educational context
        """
        # Validate market
        if market_id not in SUPPORTED_MARKETS:
            raise ValueError(
                f"Unsupported market: {market_id}. "
                f"Supported markets: {list(SUPPORTED_MARKETS.keys())}"
            )
        
        market_info = SUPPORTED_MARKETS[market_id]
        
        # Validate selection
        if selection_id not in market_info['options']:
            raise ValueError(
                f"Invalid selection '{selection_id}' for market '{market_id}'. "
                f"Valid options: {market_info['options']}"
            )
        
        # Get Smart Bet prediction for comparison
        smart_bet = self._get_smart_bet(match_data)
        
        # Get probability for user's selection
        user_probability = self._get_selection_probability(
            match_data,
            market_id,
            selection_id
        )
        
        # Determine confidence level and verdict
        confidence_level = self._get_confidence_level(user_probability)
        verdict = VERDICT_MESSAGES[confidence_level]
        
        # Generate explanation
        explanation = self._generate_explanation(
            match_data,
            market_id,
            selection_id,
            user_probability
        )
        
        # Generate comparison note
        comparison_note = self._generate_comparison(
            user_probability,
            market_id,
            selection_id,
            smart_bet
        )
        
        # Build response
        result = {
            'match_id': match_data.get('match_id'),
            'home_team': match_data.get('home_team'),
            'away_team': match_data.get('away_team'),
            'user_selection': {
                'market_id': market_id,
                'market_name': market_info['name'],
                'selection_id': selection_id,
                'selection_name': self._format_selection_name(market_id, selection_id)
            },
            'analysis': {
                'probability': user_probability,
                'percentage': f"{user_probability:.1%}",
                'confidence_level': confidence_level,
                'verdict': verdict,
                'explanation': explanation,
                'comparison': comparison_note
            }
        }
        
        # Add Smart Bet alternative if different
        if smart_bet and not self._is_same_selection(
            market_id, selection_id, smart_bet
        ):
            result['smart_bet_alternative'] = {
                'market_id': smart_bet['market_id'],
                'market_name': smart_bet['market_name'],
                'selection_id': smart_bet['selection_id'],
                'selection_name': smart_bet['selection_name'],
                'probability': smart_bet['probability'],
                'percentage': f"{smart_bet['probability']:.1%}"
            }
        
        return result
    
    def _get_smart_bet(self, match_data: Dict) -> Optional[Dict]:
        """Get Smart Bet prediction for the match"""
        try:
            prediction = self.smart_predictor.predict_match(match_data)
            return prediction.get('smart_bet')
        except Exception as e:
            logger.warning(f"Could not get Smart Bet: {e}")
            return None
    
    def _get_selection_probability(
        self,
        match_data: Dict,
        market_id: str,
        selection_id: str
    ) -> float:
        """Get probability for specific selection"""
        try:
            # Get all market predictions
            prediction = self.smart_predictor.predict_match(match_data)
            market_predictions = prediction.get('market_predictions', {})
            
            # Find the specific market
            for market_pred in market_predictions:
                if market_pred['market_id'] == market_id:
                    # Check if this is the positive prediction
                    if market_pred['selection_id'] == selection_id:
                        return market_pred['probability']
                    else:
                        # Return complement probability for opposite selection
                        return 1.0 - market_pred['probability']
            
            # Fallback
            logger.warning(f"Could not find probability for {market_id}/{selection_id}")
            return 0.5
            
        except Exception as e:
            logger.error(f"Error getting selection probability: {e}")
            return 0.5
    
    def _get_confidence_level(self, probability: float) -> str:
        """Determine confidence level from probability"""
        for level, threshold in sorted(
            CONFIDENCE_LEVELS.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if probability >= threshold:
                return level
        return 'very_low'
    
    def _generate_explanation(
        self,
        match_data: Dict,
        market_id: str,
        selection_id: str,
        probability: float
    ) -> str:
        """Generate explanation for the bet"""
        home_team = match_data.get('home_team', 'Home Team')
        away_team = match_data.get('away_team', 'Away Team')
        
        # Market-specific explanations
        if market_id == 'total_goals':
            home_goals = match_data.get('home_goals_avg', 0)
            away_goals = match_data.get('away_goals_avg', 0)
            total_avg = home_goals + away_goals
            
            if 'over' in selection_id:
                return (
                    f"Both teams average {total_avg:.1f} goals combined per match. "
                    f"{home_team} averages {home_goals:.1f} goals at home, "
                    f"{away_team} averages {away_goals:.1f} away. "
                    f"AI probability for Over 2.5 Goals: {probability:.1%}."
                )
            else:
                return (
                    f"Combined goals average is {total_avg:.1f} per match. "
                    f"Both teams show moderate scoring rates. "
                    f"AI probability for Under 2.5 Goals: {probability:.1%}."
                )
        
        elif market_id == 'total_cards':
            home_cards = match_data.get('home_cards_avg', 0)
            away_cards = match_data.get('away_cards_avg', 0)
            total_avg = home_cards + away_cards
            
            if 'over' in selection_id:
                return (
                    f"Both teams average {total_avg:.1f} cards combined per match. "
                    f"{home_team} averages {home_cards:.1f} cards at home, "
                    f"{away_team} averages {away_cards:.1f} away. "
                    f"AI probability for Over 3.5 Cards: {probability:.1%}."
                )
            else:
                return (
                    f"Combined cards average is {total_avg:.1f} per match. "
                    f"Both teams show disciplined play. "
                    f"AI probability for Under 3.5 Cards: {probability:.1%}."
                )
        
        elif market_id == 'total_corners':
            home_corners = match_data.get('home_corners_avg', 0)
            away_corners = match_data.get('away_corners_avg', 0)
            total_avg = home_corners + away_corners
            
            if 'over' in selection_id:
                return (
                    f"Both teams average {total_avg:.1f} corners combined per match. "
                    f"{home_team} averages {home_corners:.1f} corners at home, "
                    f"{away_team} averages {away_corners:.1f} away. "
                    f"AI probability for Over 9.5 Corners: {probability:.1%}."
                )
            else:
                return (
                    f"Combined corners average is {total_avg:.1f} per match. "
                    f"Both teams show moderate attacking patterns. "
                    f"AI probability for Under 9.5 Corners: {probability:.1%}."
                )
        
        elif market_id == 'btts':
            home_btts = match_data.get('home_btts_rate', 0)
            away_btts = match_data.get('away_btts_rate', 0)
            
            if selection_id == 'yes':
                return (
                    f"{home_team} has BTTS in {home_btts:.1%} of home matches, "
                    f"{away_team} has BTTS in {away_btts:.1%} of away matches. "
                    f"AI probability for Both Teams To Score: {probability:.1%}."
                )
            else:
                return (
                    f"Historical BTTS rates suggest one or both teams may not score. "
                    f"AI probability for BTTS No: {probability:.1%}."
                )
        
        # Fallback
        return f"AI analysis indicates {probability:.1%} probability for this outcome."
    
    def _generate_comparison(
        self,
        user_probability: float,
        user_market_id: str,
        user_selection_id: str,
        smart_bet: Optional[Dict]
    ) -> str:
        """Generate comparison with Smart Bet"""
        if not smart_bet:
            return EDUCATIONAL_NOTES['no_smart_bet']
        
        # Check if user's selection matches Smart Bet
        if self._is_same_selection(user_market_id, user_selection_id, smart_bet):
            return EDUCATIONAL_NOTES['matches_smart_bet']
        
        # User's selection is different
        return EDUCATIONAL_NOTES['lower_than_smart_bet'].format(
            user_conf=user_probability,
            smart_conf=smart_bet['probability'],
            smart_market=smart_bet['selection_name']
        )
    
    def _is_same_selection(
        self,
        market_id: str,
        selection_id: str,
        smart_bet: Dict
    ) -> bool:
        """Check if user's selection matches Smart Bet"""
        return (
            market_id == smart_bet.get('market_id') and
            selection_id == smart_bet.get('selection_id')
        )
    
    def _format_selection_name(self, market_id: str, selection_id: str) -> str:
        """Format selection name for display"""
        market_name = SUPPORTED_MARKETS[market_id]['name']
        
        if market_id == 'total_goals':
            return f"{'Over' if 'over' in selection_id else 'Under'} 2.5 Goals"
        elif market_id == 'total_cards':
            return f"{'Over' if 'over' in selection_id else 'Under'} 3.5 Cards"
        elif market_id == 'total_corners':
            return f"{'Over' if 'over' in selection_id else 'Under'} 9.5 Corners"
        elif market_id == 'btts':
            return f"BTTS {'Yes' if selection_id == 'yes' else 'No'}"
        
        return selection_id.replace('_', ' ').title()
