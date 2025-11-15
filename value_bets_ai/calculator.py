"""
Value Calculation Engine
Calculates betting value and expected value from AI probabilities and odds
"""

from typing import Dict, Optional
from config import (
    MIN_VALUE_THRESHOLD,
    MIN_PROBABILITY,
    MIN_EXPECTED_VALUE,
    MIN_ODDS,
    MAX_ODDS,
    VALUE_SCORE_WEIGHTS
)


class ValueCalculator:
    """Calculates betting value metrics"""
    
    @staticmethod
    def decimal_to_implied_probability(decimal_odds: float) -> float:
        """
        Convert decimal odds to implied probability
        
        Args:
            decimal_odds: Decimal odds (e.g., 2.50)
            
        Returns:
            Implied probability (0.0 to 1.0)
        """
        if decimal_odds <= 1.0:
            return 0.0
        return 1.0 / decimal_odds
    
    @staticmethod
    def calculate_value_percentage(ai_probability: float, decimal_odds: float) -> float:
        """
        Calculate value percentage
        
        Formula: Value% = AI_Probability - Implied_Probability
        
        Args:
            ai_probability: AI predicted probability (0.0 to 1.0)
            decimal_odds: Bookmaker decimal odds
            
        Returns:
            Value percentage (can be negative)
        """
        implied_prob = ValueCalculator.decimal_to_implied_probability(decimal_odds)
        return ai_probability - implied_prob
    
    @staticmethod
    def calculate_expected_value(ai_probability: float, decimal_odds: float) -> float:
        """
        Calculate expected value (EV)
        
        Formula: EV = (AI_Probability × Decimal_Odds) - 1
        
        Args:
            ai_probability: AI predicted probability (0.0 to 1.0)
            decimal_odds: Bookmaker decimal odds
            
        Returns:
            Expected value as decimal (e.g., 0.15 = 15% EV)
        """
        return (ai_probability * decimal_odds) - 1.0
    
    @staticmethod
    def calculate_value_score(
        ai_probability: float,
        decimal_odds: float,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate composite value score
        
        Combines value%, EV, and probability into single score
        
        Args:
            ai_probability: AI predicted probability
            decimal_odds: Bookmaker decimal odds
            weights: Custom weights (default from config)
            
        Returns:
            Composite value score (0.0 to 1.0+)
        """
        if weights is None:
            weights = VALUE_SCORE_WEIGHTS
        
        value_pct = ValueCalculator.calculate_value_percentage(ai_probability, decimal_odds)
        ev = ValueCalculator.calculate_expected_value(ai_probability, decimal_odds)
        
        # Normalize components to 0-1 range for scoring
        # Value% normalized: assume max useful value is 30%
        value_component = min(max(value_pct, 0) / 0.30, 1.0)
        
        # EV normalized: assume max useful EV is 50%
        ev_component = min(max(ev, 0) / 0.50, 1.0)
        
        # Probability component (already 0-1)
        prob_component = ai_probability
        
        # Weighted composite score
        score = (
            weights['value_percentage'] * value_component +
            weights['expected_value'] * ev_component +
            weights['probability'] * prob_component
        )
        
        return score
    
    @staticmethod
    def is_value_bet(
        ai_probability: float,
        decimal_odds: float,
        min_value: float = MIN_VALUE_THRESHOLD,
        min_prob: float = MIN_PROBABILITY,
        min_ev: float = MIN_EXPECTED_VALUE
    ) -> bool:
        """
        Determine if a bet qualifies as a value bet
        
        Args:
            ai_probability: AI predicted probability
            decimal_odds: Bookmaker decimal odds
            min_value: Minimum value percentage threshold
            min_prob: Minimum AI probability threshold
            min_ev: Minimum expected value threshold
            
        Returns:
            True if bet meets all value criteria
        """
        # Check odds range
        if decimal_odds < MIN_ODDS or decimal_odds > MAX_ODDS:
            return False
        
        # Check minimum probability
        if ai_probability < min_prob:
            return False
        
        # Calculate metrics
        value_pct = ValueCalculator.calculate_value_percentage(ai_probability, decimal_odds)
        ev = ValueCalculator.calculate_expected_value(ai_probability, decimal_odds)
        
        # Must meet both value% and EV thresholds
        return value_pct >= min_value and ev >= min_ev
    
    @staticmethod
    def calculate_all_metrics(ai_probability: float, decimal_odds: float) -> Dict[str, float]:
        """
        Calculate all value metrics for a bet
        
        Args:
            ai_probability: AI predicted probability
            decimal_odds: Bookmaker decimal odds
            
        Returns:
            Dictionary with all calculated metrics
        """
        implied_prob = ValueCalculator.decimal_to_implied_probability(decimal_odds)
        value_pct = ValueCalculator.calculate_value_percentage(ai_probability, decimal_odds)
        ev = ValueCalculator.calculate_expected_value(ai_probability, decimal_odds)
        value_score = ValueCalculator.calculate_value_score(ai_probability, decimal_odds)
        is_value = ValueCalculator.is_value_bet(ai_probability, decimal_odds)
        
        return {
            'ai_probability': ai_probability,
            'decimal_odds': decimal_odds,
            'implied_probability': implied_prob,
            'value_percentage': value_pct,
            'expected_value': ev,
            'value_score': value_score,
            'is_value_bet': is_value
        }
