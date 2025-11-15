# Value Bets AI Module

**Dynamic value bet identification based on odds vs AI probability**

## Overview

The Value Bets AI module identifies betting opportunities where the AI's predicted probability significantly exceeds the bookmaker's implied probability, indicating positive expected value (EV) for long-term profitability.

## Purpose

Dynamically calculate and identify value bets by comparing AI probabilities against real-time bookmaker odds across all 4 target markets.

## Key Features

- **Top 3 Daily Picks**: Maximum 3 value bets per day
- **Positive EV Focus**: Minimum 10% value, 5% expected value
- **Real-time Calculation**: Dynamic recalculation as odds change
- **Educational Reasoning**: Detailed explanations of value concept
- **Long-term Profitability**: Focus on EV over individual bet success rate

## Target Markets

Same 4 markets as Smart Bets and Golden Bets:
1. **Goals**: Over/Under 2.5
2. **Cards**: Over/Under 3.5
3. **Corners**: Over/Under 9.5
4. **BTTS**: Yes/No

## Value Calculation

### Core Formulas

**Implied Probability:**
```
Implied_Probability = 1 / Decimal_Odds
```

**Value Percentage:**
```
Value% = AI_Probability - Implied_Probability
```

**Expected Value (EV):**
```
EV = (AI_Probability × Decimal_Odds) - 1
```

**Value Score (Composite):**
```
Value_Score = (0.40 × Value%) + (0.35 × EV) + (0.25 × AI_Probability)
```

### Thresholds

- **Minimum Value**: 10% (AI prob must exceed implied prob by 10%+)
- **Minimum EV**: 5% positive expected value
- **Minimum AI Probability**: 50%
- **Odds Range**: 1.50 to 5.00 decimal odds

## Module Structure

```
value-bets-ai/
├── __init__.py          # Module initialization
├── config.py            # Configuration and thresholds
├── calculator.py        # Value calculation engine
├── predict.py           # Main prediction pipeline
└── README.md           # This file
```

## Usage

### Basic Usage

```python
from value_bets_ai.predict import ValueBetsPredictor

# Initialize predictor
predictor = ValueBetsPredictor()

# Prepare match data with odds
matches_with_odds = [
    {
        'match_id': '12345',
        'home_team': 'Team A',
        'away_team': 'Team B',
        # ... match features ...
        'odds': {
            'goals_over_2_5': 2.10,
            'goals_under_2_5': 1.75,
            'cards_over_3_5': 2.50,
            'cards_under_3_5': 1.55,
            'corners_over_9_5': 1.90,
            'corners_under_9_5': 1.95,
            'btts_yes': 1.85,
            'btts_no': 2.00
        }
    }
]

# Get value bets
value_bets = predictor.predict(matches_with_odds)

# Results: Top 3 value bets ranked by value score
for bet in value_bets:
    print(f"{bet['home_team']} vs {bet['away_team']}")
    print(f"Market: {bet['market_name']} - {bet['selection_name']}")
    print(f"AI Probability: {bet['ai_probability']:.1%}")
    print(f"Odds: {bet['decimal_odds']:.2f}")
    print(f"Value: +{bet['value_percentage']:.1%}")
    print(f"Expected Value: +{bet['expected_value']:.1%}")
    print(f"Reasoning: {bet['reasoning']}\n")
```

### API Endpoint

```bash
POST /api/v1/predictions/value-bets
```

**Request Body:**
```json
{
  "matches": [
    {
      "match_id": "12345",
      "home_team": "Team A",
      "away_team": "Team B",
      "home_goals_avg": 1.5,
      "away_goals_avg": 1.2,
      "home_goals_conceded_avg": 1.1,
      "away_goals_conceded_avg": 1.3,
      "home_corners_avg": 5.2,
      "away_corners_avg": 4.8,
      "home_cards_avg": 2.1,
      "away_cards_avg": 2.3,
      "home_btts_rate": 0.60,
      "away_btts_rate": 0.55,
      "odds": {
        "goals_over_2_5": 2.10,
        "goals_under_2_5": 1.75,
        "cards_over_3_5": 2.50,
        "cards_under_3_5": 1.55,
        "corners_over_9_5": 1.90,
        "corners_under_9_5": 1.95,
        "btts_yes": 1.85,
        "btts_no": 2.00
      }
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "predictions": [
    {
      "match_id": "12345",
      "home_team": "Team A",
      "away_team": "Team B",
      "market_name": "Both Teams To Score",
      "selection_name": "Yes",
      "ai_probability": 0.68,
      "decimal_odds": 1.85,
      "implied_probability": 0.54,
      "value_percentage": 0.14,
      "expected_value": 0.259,
      "value_score": 0.742,
      "bet_category": "value",
      "reasoning": "AI probability (68.0%) significantly exceeds bookmaker's implied probability (54.1%). Value: +14.0%. Expected Value: +25.9%. At odds of 1.85, this bet offers strong long-term profit potential. Value bets focus on long-term profitability rather than individual bet success rate."
    }
  ],
  "count": 1,
  "max_daily": 3
}
```

## How It Works

### Pipeline

1. **Get Smart Bets Predictions**: Obtain AI probabilities for all markets
2. **Extract Odds**: Get bookmaker odds for each market
3. **Calculate Value Metrics**: Compute value%, EV, and value score
4. **Filter Value Bets**: Keep only bets meeting thresholds
5. **Rank by Value Score**: Sort by composite value score
6. **Return Top 3**: Return highest-value opportunities

### Value Score Weighting

The composite value score balances three factors:
- **40%** - Value Percentage (how much AI exceeds bookmaker)
- **35%** - Expected Value (long-term profit potential)
- **25%** - AI Probability (confidence in prediction)

## Example Output

```
VALUE BETS PREDICTIONS
============================================================

Found 3 Value Bets:

1. Manchester United vs Liverpool
   Market: Both Teams To Score
   Selection: Yes
   AI Probability: 68.0%
   Odds: 1.85
   Value: +14.0%
   Expected Value: +25.9%
   AI probability (68.0%) significantly exceeds bookmaker's 
   implied probability (54.1%). Value: +14.0%. Expected Value: 
   +25.9%. At odds of 1.85, this bet offers strong long-term 
   profit potential.

2. Arsenal vs Chelsea
   Market: Total Corners
   Selection: Over 9.5
   AI Probability: 72.0%
   Odds: 1.95
   Value: +20.7%
   Expected Value: +40.4%
   AI probability (72.0%) significantly exceeds bookmaker's 
   implied probability (51.3%). Value: +20.7%. Expected Value: 
   +40.4%. At odds of 1.95, this bet offers strong long-term 
   profit potential.

3. Barcelona vs Real Madrid
   Market: Total Goals
   Selection: Over 2.5
   AI Probability: 65.0%
   Odds: 2.10
   Value: +17.4%
   Expected Value: +36.5%
   AI probability (65.0%) significantly exceeds bookmaker's 
   implied probability (47.6%). Value: +17.4%. Expected Value: 
   +36.5%. At odds of 2.10, this bet offers strong long-term 
   profit potential.
```

## Configuration

Edit `config.py` to adjust thresholds:

```python
# Value calculation thresholds
MIN_VALUE_THRESHOLD = 0.10  # Minimum 10% positive value
MIN_PROBABILITY = 0.50      # Minimum 50% AI probability
MAX_DAILY_PICKS = 3         # Maximum 3 value bets per day
MIN_EXPECTED_VALUE = 0.05   # Minimum 5% positive EV

# Odds validation
MIN_ODDS = 1.50  # Minimum decimal odds
MAX_ODDS = 5.00  # Maximum decimal odds

# Value score weighting
VALUE_SCORE_WEIGHTS = {
    'value_percentage': 0.40,
    'expected_value': 0.35,
    'probability': 0.25
}
```

## Key Differences from Other Features

### vs Smart Bets
- **Smart Bets**: Highest probability per match (no odds consideration)
- **Value Bets**: Best value across all matches (odds-dependent)

### vs Golden Bets
- **Golden Bets**: 85%+ confidence, safety-focused
- **Value Bets**: 50%+ probability, profit-focused, lower win rate but higher ROI

## Important Notes

1. **Long-term Focus**: Value bets optimize for long-term profitability, not individual bet success
2. **Lower Win Rate**: Expect lower win rate than Golden Bets but higher ROI over time
3. **Odds Required**: Must provide current bookmaker odds for all 8 selections
4. **Dynamic Recalculation**: Value changes as odds change - recalculate frequently
5. **Educational**: Helps users understand expected value and betting strategy

## Status

✅ **Production Ready** - Phase 4 Complete

## Dependencies

- `smart_bets_ai` - For AI probability predictions
- Python 3.8+
- NumPy, Pandas (inherited from Smart Bets)

## Testing

Run the test script:
```bash
python value-bets-ai/predict.py
```

Requires test data file: `test-data/upcoming_matches_with_odds_sample.json`

## Next Steps

- Integrate with odds-updater module for real-time odds
- Add historical value bet tracking
- Implement dynamic recalculation triggers
- Add value bet performance analytics
