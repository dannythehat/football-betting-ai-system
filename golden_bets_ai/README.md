# Golden Bets AI

**Status:** ✅ Complete (Phase 3)  
**Purpose:** Filter Smart Bets predictions to identify 1-3 daily picks with 85%+ confidence

---

## Overview

Golden Bets AI applies strict confidence filtering to Smart Bets predictions, identifying the safest, highest-probability betting opportunities each day across the 4 target markets.

### Selection Criteria

1. **Confidence Threshold:** ≥85% probability
2. **Ensemble Agreement:** ≥90% model consensus
3. **Daily Limit:** Maximum 1-3 picks per day
4. **Ranking:** Composite golden score (70% probability + 30% agreement)

---

## Usage

### Quick Prediction (Recommended)

```python
from golden_bets_ai import GoldenBetsPredictor

# Initialize predictor (handles Smart Bets integration automatically)
predictor = GoldenBetsPredictor()

# Get Golden Bets predictions
golden_bets = predictor.predict(matches)

# Display results
for bet in golden_bets:
    print(f"{bet['home_team']} vs {bet['away_team']}")
    print(f"Market: {bet['market_name']}")
    print(f"Confidence: {bet['confidence_score']:.1%}")
    print(f"Reasoning: {bet['reasoning']}")
```

### Manual Filtering (Advanced)

```python
from golden_bets_ai import GoldenBetsFilter

# Initialize filter
filter = GoldenBetsFilter()

# Filter Smart Bets predictions
golden_bets = filter.filter_golden_bets(
    smart_bets_predictions=smart_predictions,
    model_probabilities=ensemble_probs  # Optional
)

# Generate reasoning
for bet in golden_bets:
    reasoning = filter.generate_reasoning(bet)
    print(reasoning)
```

### Input Format

```python
matches = [
    {
        'match_id': '12345',
        'home_team': 'Team A',
        'away_team': 'Team B',
        'stats': {
            'home_goals_avg': 1.4,
            'away_goals_avg': 1.1,
            'home_corners_avg': 5.2,
            'away_corners_avg': 4.8,
            'home_cards_avg': 2.1,
            'away_cards_avg': 1.8,
            'home_btts_rate': 0.6,
            'away_btts_rate': 0.5
        },
        'odds': {
            'total_goals': {'over_2.5': 2.10, 'under_2.5': 1.75},
            'total_cards': {'over_3.5': 1.95, 'under_3.5': 1.85},
            'total_corners': {'over_9.5': 1.90, 'under_9.5': 1.90},
            'btts': {'yes': 1.85, 'no': 2.00}
        }
    }
]
```

### Output Format

```python
golden_bets = [
    {
        'match_id': '12345',
        'home_team': 'Team A',
        'away_team': 'Team B',
        'probability': 0.87,
        'market_name': 'Total Corners',
        'selection_name': 'Over 9.5',
        'confidence_score': 0.87,
        'ensemble_agreement': 0.95,
        'golden_score': 0.894,  # (0.7 * 0.87) + (0.3 * 0.95)
        'bet_category': 'golden',
        'reasoning': '🏆 Golden Bet Selection Criteria Met...'
    }
]
```

---

## Configuration

Edit `config.py` to adjust thresholds:

```python
CONFIDENCE_THRESHOLD = 0.85      # Minimum probability
MIN_ENSEMBLE_AGREEMENT = 0.90    # Minimum model agreement
MAX_DAILY_PICKS = 3              # Maximum daily selections
PROBABILITY_WEIGHT = 0.7         # Probability weight in golden score
AGREEMENT_WEIGHT = 0.3           # Agreement weight in golden score
```

---

## Algorithm Details

### Ensemble Agreement Calculation

Uses coefficient of variation to measure model consensus:

```python
cv = std(model_probs) / mean(model_probs)
agreement = max(0, 1 - cv)
```

- **Low CV** → High agreement → Score near 1.0
- **High CV** → Low agreement → Score near 0.0

### Golden Score Calculation

Composite score combining probability and agreement:

```python
golden_score = (0.7 * probability) + (0.3 * agreement)
```

### Selection Process

1. Filter predictions with probability ≥ 85%
2. Calculate ensemble agreement (if available)
3. Filter predictions with agreement ≥ 90%
4. Calculate golden score for each candidate
5. Sort by golden score (descending)
6. Return top 1-3 picks

---

## Integration with Smart Bets AI

### Using Predictor (Automatic)

```python
from golden_bets_ai import GoldenBetsPredictor

# Predictor handles Smart Bets integration automatically
predictor = GoldenBetsPredictor()
golden_bets = predictor.predict(matches)
```

### Using Filter (Manual)

```python
from smart_bets_ai import SmartBetsPredictor
from golden_bets_ai import GoldenBetsFilter

# Get Smart Bets predictions
predictor = SmartBetsPredictor()
smart_predictions = predictor.predict(matches)

# Extract model probabilities for ensemble agreement
model_probs = predictor.get_model_probabilities(matches)

# Filter for Golden Bets
filter = GoldenBetsFilter()
golden_bets = filter.filter_golden_bets(
    smart_bets_predictions=smart_predictions,
    model_probabilities=model_probs
)
```

---

## Testing

```bash
# Run prediction pipeline
python golden-bets-ai/predict.py

# Test filter with sample data
python golden-bets-ai/test_filter.py

# Run unit tests
python -m pytest golden-bets-ai/tests/
```

---

## API Integration

Golden Bets are automatically included in the API response:

```bash
POST /api/v1/predictions/golden-bets
```

Response:
```json
{
  "golden_bets": [
    {
      "match_id": "12345",
      "home_team": "Team A",
      "away_team": "Team B",
      "market_name": "Total Corners",
      "selection_name": "Over 9.5",
      "confidence_score": 0.87,
      "ensemble_agreement": 0.95,
      "golden_score": 0.894,
      "bet_category": "golden",
      "reasoning": "🏆 Golden Bet Selection Criteria Met..."
    }
  ],
  "total_candidates": 15,
  "selected_count": 3
}
```

---

## Performance Metrics

Track Golden Bets performance:

- **Win Rate:** Target ≥85% (matches confidence threshold)
- **Daily Coverage:** 1-3 picks per day
- **Model Agreement:** Average ≥90%
- **Golden Score:** Average composite score

---

## Troubleshooting

### No Golden Bets Found

**Cause:** No predictions meet 85% threshold or 90% agreement  
**Solution:** Lower thresholds in `config.py` or improve Smart Bets model accuracy

### Too Many Golden Bets

**Cause:** Many predictions exceed thresholds  
**Solution:** Increase `CONFIDENCE_THRESHOLD` or reduce `MAX_DAILY_PICKS`

### Low Ensemble Agreement

**Cause:** Models disagree significantly  
**Solution:** Retrain models with consistent data or adjust `MIN_ENSEMBLE_AGREEMENT`

---

## Module Files

- **`predict.py`** - Main prediction pipeline (recommended entry point)
- **`filter.py`** - Core filtering logic
- **`config.py`** - Configuration settings
- **`test_filter.py`** - Test script with sample data
- **`__init__.py`** - Module exports

---

## Next Steps

- **Phase 4:** Value Bets AI (EV calculations)
- **Phase 5:** Custom Bet Analysis (user-selected analysis)
- **Enhancement:** Historical validation tracking
- **Enhancement:** Dynamic threshold adjustment based on performance
