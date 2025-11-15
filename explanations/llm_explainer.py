"""
LLM-Based Bet Explanation Service
Uses OpenAI API to generate natural language explanations for bets
"""

import os
import json
from typing import Dict, Optional
import warnings


class BetExplanationService:
    """
    Generates natural language explanations for betting recommendations
    using LLM (OpenAI API) - ONLY for narrative, not for predictions
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize explanation service
        
        Args:
            openai_api_key: OpenAI API key (optional, reads from env if not provided)
        """
        self.api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.client = None
        
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                warnings.warn("OpenAI package not installed. Install with: pip install openai")
        else:
            warnings.warn("OpenAI API key not provided. Explanations will use fallback logic.")
    
    def explain_bet(self, analysis_payload: Dict) -> str:
        """
        Generate natural language explanation for a bet
        
        Args:
            analysis_payload: Structured data with:
                - teams (home_team, away_team)
                - league
                - date
                - market
                - model_probability
                - important_features (stats)
                - odds (optional)
                - value_pct (optional)
                
        Returns:
            Natural language explanation
        """
        if not self.client:
            return self._fallback_explanation(analysis_payload)
        
        try:
            # Construct prompt that ONLY asks for interpretation
            prompt = self._build_prompt(analysis_payload)
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a football betting analyst. You MUST ONLY interpret "
                            "the provided statistics and probabilities. DO NOT invent or "
                            "calculate any numbers. Use ONLY the exact values provided in "
                            "the data. Your role is to explain WHY the model made this "
                            "prediction based on the given statistics."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            explanation = response.choices[0].message.content.strip()
            return explanation
        
        except Exception as e:
            warnings.warn(f"Error generating LLM explanation: {e}")
            return self._fallback_explanation(analysis_payload)
    
    def _build_prompt(self, payload: Dict) -> str:
        """Build prompt for LLM"""
        home_team = payload.get('home_team', 'Home Team')
        away_team = payload.get('away_team', 'Away Team')
        league = payload.get('league', 'League')
        market = payload.get('market', 'Unknown')
        probability = payload.get('model_probability', 0.5)
        features = payload.get('important_features', {})
        odds = payload.get('odds')
        value_pct = payload.get('value_pct')
        
        # Market descriptions
        market_desc = {
            'goals': 'Over 2.5 Goals',
            'btts': 'Both Teams To Score',
            'cards': 'Over 3.5 Cards',
            'corners': 'Over 9.5 Corners'
        }.get(market, market)
        
        prompt = f"""
Explain this betting recommendation in 2-3 sentences:

Match: {home_team} vs {away_team} ({league})
Market: {market_desc}
AI Probability: {probability:.1%}
"""
        
        if odds:
            prompt += f"Odds: {odds:.2f}\n"
        if value_pct:
            prompt += f"Value: {value_pct:.1f}%\n"
        
        prompt += "\nKey Statistics:\n"
        for key, value in features.items():
            prompt += f"- {key}: {value}\n"
        
        prompt += """
Explain WHY this bet is recommended based ONLY on these statistics. 
Do not invent any numbers or statistics not provided above.
Focus on the key factors that support this prediction.
"""
        
        return prompt
    
    def _fallback_explanation(self, payload: Dict) -> str:
        """
        Generate fallback explanation without LLM
        
        Args:
            payload: Analysis payload
            
        Returns:
            Basic explanation string
        """
        market = payload.get('market', 'Unknown')
        probability = payload.get('model_probability', 0.5)
        features = payload.get('important_features', {})
        
        market_names = {
            'goals': 'Over 2.5 Goals',
            'btts': 'Both Teams To Score',
            'cards': 'Over 3.5 Cards',
            'corners': 'Over 9.5 Corners'
        }
        
        market_name = market_names.get(market, market)
        
        explanation = (
            f"Our AI model predicts {market_name} with {probability:.1%} confidence. "
            f"This recommendation is based on comprehensive analysis of team statistics "
            f"including recent form, scoring patterns, and head-to-head data."
        )
        
        # Add key stat if available
        if 'combined_goals_avg' in features:
            explanation += f" The teams average {features['combined_goals_avg']:.1f} combined goals per match."
        elif 'combined_btts_rate' in features:
            explanation += f" Both teams have a combined BTTS rate of {features['combined_btts_rate']:.1%}."
        
        return explanation
    
    def explain_smart_bet(self, smart_bet_data: Dict) -> str:
        """
        Generate explanation for Smart Bet
        
        Args:
            smart_bet_data: Smart Bet recommendation data
            
        Returns:
            Explanation string
        """
        return self.explain_bet(smart_bet_data)
    
    def explain_golden_bet(self, golden_bet_data: Dict) -> str:
        """
        Generate explanation for Golden Bet
        
        Args:
            golden_bet_data: Golden Bet recommendation data
            
        Returns:
            Explanation string
        """
        payload = golden_bet_data.copy()
        payload['explanation_type'] = 'golden'
        return self.explain_bet(payload)
    
    def explain_value_bet(self, value_bet_data: Dict) -> str:
        """
        Generate explanation for Value Bet
        
        Args:
            value_bet_data: Value Bet recommendation data
            
        Returns:
            Explanation string
        """
        payload = value_bet_data.copy()
        payload['explanation_type'] = 'value'
        
        # Add value-specific context
        if 'value_pct' in value_bet_data:
            payload['value_context'] = (
                f"This bet offers {value_bet_data['value_pct']:.1f}% value, "
                f"meaning the AI believes the true probability is higher than "
                f"what the odds suggest."
            )
        
        return self.explain_bet(payload)


# Convenience function
def create_explainer(api_key: Optional[str] = None) -> BetExplanationService:
    """Create and return a bet explanation service"""
    return BetExplanationService(api_key)


if __name__ == "__main__":
    # Test explainer
    explainer = BetExplanationService()
    
    test_payload = {
        'home_team': 'Manchester United',
        'away_team': 'Liverpool',
        'league': 'Premier League',
        'market': 'goals',
        'model_probability': 0.72,
        'important_features': {
            'combined_goals_avg': 3.9,
            'home_goals_avg': 1.8,
            'away_goals_avg': 2.1,
            'home_attack_vs_away_defense': 0.8
        },
        'odds': 1.85,
        'value_pct': 12.5
    }
    
    print("\n" + "=" * 60)
    print("LLM EXPLAINER TEST")
    print("=" * 60)
    
    explanation = explainer.explain_bet(test_payload)
    print(f"\nExplanation:\n{explanation}")
