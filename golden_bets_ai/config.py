"""
Golden Bets AI Configuration
Centralized settings for confidence filtering and selection criteria
"""

# Confidence Thresholds
CONFIDENCE_THRESHOLD = 0.85  # 85% minimum probability
MIN_ENSEMBLE_AGREEMENT = 0.90  # 90% model agreement required

# Selection Limits
MAX_DAILY_PICKS = 3  # Maximum 1-3 picks per day
MIN_DAILY_PICKS = 1  # Minimum picks (0 if no qualifying bets)

# Scoring Weights
PROBABILITY_WEIGHT = 0.7  # 70% weight on probability
AGREEMENT_WEIGHT = 0.3   # 30% weight on ensemble agreement

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Validation
REQUIRED_FIELDS = [
    'match_id',
    'probability',
    'market_name',
    'selection_name'
]
