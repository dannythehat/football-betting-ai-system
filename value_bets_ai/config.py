"""
Value Bets AI Configuration
"""

# Value calculation thresholds
MIN_VALUE_THRESHOLD = 0.10  # Minimum 10% positive value
MIN_PROBABILITY = 0.50  # Minimum 50% AI probability
MAX_DAILY_PICKS = 3  # Maximum 3 value bets per day

# Expected Value (EV) calculation
# EV = (AI_Probability * Decimal_Odds) - 1
MIN_EXPECTED_VALUE = 0.05  # Minimum 5% positive EV

# Odds validation
MIN_ODDS = 1.50  # Minimum decimal odds to consider
MAX_ODDS = 5.00  # Maximum decimal odds to consider

# Value score weighting (must sum to 1.0)
VALUE_SCORE_WEIGHTS = {
    'value_percentage': 0.40,  # 40% weight on value %
    'expected_value': 0.35,    # 35% weight on EV
    'probability': 0.25        # 25% weight on AI probability
}
