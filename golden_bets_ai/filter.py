"""
Golden Bets AI - 85%+ Confidence Filtering
Identifies the safest, highest-confidence bets from Smart Bets predictions
"""
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class GoldenBetsFilter:
    """Filters Smart Bets predictions to find Golden Bets (85%+ confidence)"""
    
    CONFIDENCE_THRESHOLD = 0.85  # 85% minimum probability
    MIN_ENSEMBLE_AGREEMENT = 0.90  # 90% model agreement required
    MAX_DAILY_PICKS = 3  # Maximum 1-3 picks per day
    
    def __init__(self):
        self.confidence_threshold = self.CONFIDENCE_THRESHOLD
        self.min_agreement = self.MIN_ENSEMBLE_AGREEMENT
        self.max_picks = self.MAX_DAILY_PICKS
    
    def filter_golden_bets(
        self, 
        smart_bets_predictions: List[Dict[str, Any]],
        model_probabilities: Optional[Dict[str, np.ndarray]] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter Smart Bets to find Golden Bets
        
        Args:
            smart_bets_predictions: List of Smart Bets predictions with probabilities
            model_probabilities: Optional dict of individual model probabilities for ensemble agreement
            
        Returns:
            List of Golden Bets (top 1-3 highest confidence picks)
        """
        golden_candidates = []
        
        for prediction in smart_bets_predictions:
            probability = prediction.get('probability', 0)
            match_id = prediction.get('match_id')
            
            # Check confidence threshold
            if probability < self.confidence_threshold:
                logger.debug(f"Match {match_id}: Below confidence threshold ({probability:.2%} < {self.confidence_threshold:.2%})")
                continue
            
            # Calculate ensemble agreement if model probabilities provided
            agreement_score = 1.0  # Default full agreement
            if model_probabilities and match_id in model_probabilities:
                agreement_score = self._calculate_ensemble_agreement(
                    model_probabilities[match_id]
                )
                
                if agreement_score < self.min_agreement:
                    logger.debug(f"Match {match_id}: Low ensemble agreement ({agreement_score:.2%})")
                    continue
            
            # Add golden bet candidate with metadata
            golden_candidate = {
                **prediction,
                'confidence_score': probability,
                'ensemble_agreement': agreement_score,
                'golden_score': self._calculate_golden_score(probability, agreement_score)
            }
            golden_candidates.append(golden_candidate)
        
        # Sort by golden score and return top picks
        golden_candidates.sort(key=lambda x: x['golden_score'], reverse=True)
        golden_bets = golden_candidates[:self.max_picks]
        
        logger.info(f"Golden Bets: Found {len(golden_bets)} picks from {len(smart_bets_predictions)} candidates")
        
        return golden_bets
    
    def _calculate_ensemble_agreement(self, model_probs: np.ndarray) -> float:
        """
        Calculate agreement between models
        Uses coefficient of variation (lower = better agreement)
        """
        if len(model_probs) < 2:
            return 1.0
        
        mean_prob = np.mean(model_probs)
        std_prob = np.std(model_probs)
        
        if mean_prob == 0:
            return 0.0
        
        # Convert coefficient of variation to agreement score (0-1)
        cv = std_prob / mean_prob
        agreement = max(0, 1 - cv)
        
        return agreement
    
    def _calculate_golden_score(self, probability: float, agreement: float) -> float:
        """
        Calculate composite golden score
        Weighted: 70% probability + 30% agreement
        """
        return (0.7 * probability) + (0.3 * agreement)
    
    def generate_reasoning(self, golden_bet: Dict[str, Any]) -> str:
        """Generate detailed reasoning for a Golden Bet selection"""
        prob = golden_bet['confidence_score']
        agreement = golden_bet['ensemble_agreement']
        market = golden_bet.get('market_name', 'Unknown')
        
        reasoning = (
            f"🏆 Golden Bet Selection Criteria Met:\n"
            f"• AI Confidence: {prob:.1%} (≥85% threshold)\n"
            f"• Model Agreement: {agreement:.1%} (≥90% threshold)\n"
            f"• Market: {market}\n"
            f"• This represents one of the top 1-3 safest bets identified today"
        )
        
        return reasoning
