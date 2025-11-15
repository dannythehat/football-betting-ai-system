"""
Configuration for Custom Bet Analysis
"""

# Confidence level thresholds for verdicts
CONFIDENCE_LEVELS = {
    'very_high': 0.80,  # 80%+
    'high': 0.70,       # 70-79%
    'moderate': 0.60,   # 60-69%
    'low': 0.50,        # 50-59%
    'very_low': 0.0     # <50%
}

# Verdict messages based on confidence
VERDICT_MESSAGES = {
    'very_high': 'Excellent',
    'high': 'Good',
    'moderate': 'Moderate',
    'low': 'Weak',
    'very_low': 'Not Recommended'
}

# Market definitions (must match Smart Bets AI)
SUPPORTED_MARKETS = {
    'total_goals': {
        'name': 'Total Goals',
        'options': ['over_2.5', 'under_2.5']
    },
    'total_cards': {
        'name': 'Total Cards',
        'options': ['over_3.5', 'under_3.5']
    },
    'total_corners': {
        'name': 'Total Corners',
        'options': ['over_9.5', 'under_9.5']
    },
    'btts': {
        'name': 'Both Teams To Score',
        'options': ['yes', 'no']
    }
}

# Educational messages
EDUCATIONAL_NOTES = {
    'lower_than_smart_bet': (
        "Note: This confidence ({user_conf:.1%}) is lower than our Smart Bet "
        "recommendation for this fixture ({smart_conf:.1%} for {smart_market}). "
        "Smart Bets analyze all 4 markets to find the highest probability option."
    ),
    'matches_smart_bet': (
        "✅ Your selection matches our Smart Bet recommendation! "
        "This is the highest probability option we identified for this fixture."
    ),
    'no_smart_bet': (
        "We don't have a Smart Bet recommendation for this fixture to compare against."
    )
}
