# Smart Bets AI Module

## Overview

The Smart Bets AI module generates **pure probabilistic predictions** for football matches. It analyzes all available betting markets and identifies the single highest-probability bet for each fixture.

## Features

- **Multi-Market Analysis**: Evaluates 14+ betting markets per match
- **XGBoost Models**: Trained on historical data with probability calibration
- **Feature Engineering**: 30+ engineered features from team statistics
- **Smart Bet Selection**: Automatically identifies best bet per fixture
- **Custom Bet Analysis**: On-demand analysis of user-selected bets
- **Model Versioning**: Track and manage different model versions

## Architecture

```
smart-bets-ai/
├── __init__.py              # Module initialization
├── feature_engineering.py   # Feature creation and transformation
├── model_trainer.py         # XGBoost model training
├── predictor.py             # Prediction generation
├── train.py                 # Training script
└── README.md                # This file
```

## Betting Markets Supported

### Match Result (1X2)
- Home Win
- Draw
- Away Win

### Total Goals
- Over/Under 0.5, 1.5, 2.5, 3.5, 4.5 goals

### Both Teams To Score (BTTS)
- Yes/No

### Corners
- Over/Under 8.5, 9.5, 10.5 corners

### Cards
- Over/Under 3.5, 4.5 cards

## Feature Engineering

The module creates 30+ features from raw match data:

### Goal Features
- Attack strength (home/away goals avg)
- Defense strength (goals conceded avg)
- Expected total goals
- Attack vs defense matchups
- Goal advantage indicators

### Form Features
- Form score (W=3, D=1, L=0)
- Form difference
- Recent form strength

### BTTS Features
- Combined BTTS rate
- BTTS likelihood indicators
- Strong attack combinations

### Corners & Cards Features
- Expected total corners/cards
- High corners/cards indicators

### Derived Features
- Goal ratios
- Strength differentials
- Match type indicators (high/low scoring, balanced, dominant)

## Training

### Prerequisites

1. **Historical Data**: Completed matches with results in database
2. **Database Access**: PostgreSQL connection
3. **Dependencies**: XGBoost, pandas, scikit-learn

### Training Command

```bash
# Basic training
python smart-bets-ai/train.py

# With custom parameters
python smart-bets-ai/train.py \
  --db-url postgresql://user:pass@localhost:5432/football_betting \
  --test-size 0.2 \
  --model-dir models \
  --version v1.0
```

### Training Process

1. **Data Loading**: Fetch completed matches and results from database
2. **Feature Engineering**: Create 30+ features from raw data
3. **Target Creation**: Generate binary targets for each market
4. **Model Training**: Train XGBoost classifier per market
5. **Validation**: Cross-validation and test set evaluation
6. **Model Saving**: Save models, features, and metrics

### Training Output

```
TRAINING SUMMARY
======================================================================
Market                    Accuracy     ROC-AUC      Log Loss    
----------------------------------------------------------------------
home_win                  0.682        0.745        0.589
draw                      0.701        0.723        0.612
away_win                  0.695        0.738        0.601
over_2_5                  0.712        0.781        0.542
btts                      0.698        0.765        0.558
...
----------------------------------------------------------------------
AVERAGE                   0.697        0.750        0.580
======================================================================
```

## Prediction

### Basic Usage

```python
from smart_bets_ai.model_trainer import SmartBetsModelTrainer
from smart_bets_ai.predictor import SmartBetsPredictor
import pandas as pd

# Load trained models
trainer = SmartBetsModelTrainer()
trainer.load_models(version='v1.0')

# Initialize predictor
predictor = SmartBetsPredictor(trainer)

# Prepare match data
match_data = pd.DataFrame([{
    'match_id': '12345',
    'home_goals_avg': 1.4,
    'away_goals_avg': 1.1,
    'home_goals_conceded_avg': 0.8,
    'away_goals_conceded_avg': 1.6,
    'home_corners_avg': 5.2,
    'away_corners_avg': 4.8,
    'home_cards_avg': 2.1,
    'away_cards_avg': 2.3,
    'home_btts_rate': 0.6,
    'away_btts_rate': 0.5,
    'home_form': 'WWWDW',
    'away_form': 'LWDLL'
}])

# Generate prediction
prediction = predictor.predict_match(match_data)
```

### Prediction Output

```python
{
    'match_id': '12345',
    'smart_bet': {
        'market_id': 'home_win',
        'market_name': 'Home Win',
        'probability': 0.87,
        'percentage': '87.0%',
        'confidence': 'very_high',
        'alternative_markets': [
            {'market_name': 'Over 2.5 Goals', 'probability': 0.67, 'percentage': '67.0%'},
            {'market_name': 'BTTS Yes', 'probability': 0.68, 'percentage': '68.0%'},
            {'market_name': 'Over 1.5 Goals', 'probability': 0.82, 'percentage': '82.0%'}
        ]
    },
    'all_probabilities': {
        'home_win': {'probability': 0.87, 'percentage': '87.0%', ...},
        'draw': {'probability': 0.08, 'percentage': '8.0%', ...},
        ...
    },
    'prediction_timestamp': '2025-11-14T15:00:00Z',
    'model_version': 'v1.0'
}
```

### Batch Predictions

```python
# Predict multiple matches
matches_data = pd.DataFrame([...])  # Multiple matches
predictions = predictor.predict_batch(matches_data)
```

### Custom Bet Analysis

```python
# Analyze user-selected bet
analysis = predictor.analyze_custom_bet(
    match_data=match_data,
    bet_type='over_2_5'
)

# Output
{
    'match_id': '12345',
    'bet_type': 'over_2_5',
    'bet_name': 'Over 2.5 Goals',
    'probability': 0.67,
    'percentage': '67.0%',
    'verdict': 'good',
    'confidence': 'medium',
    'smart_bet_comparison': {
        'smart_bet_market': 'Home Win',
        'smart_bet_probability': 0.87,
        'probability_difference': 0.20,
        'is_smart_bet': False
    },
    'note': 'This bet has 20.0% lower probability than our Smart Bet...'
}
```

## Model Performance

### Target Metrics

- **Accuracy**: >65% across all markets
- **ROC-AUC**: >0.70 for reliable probability calibration
- **Log Loss**: <0.65 for well-calibrated probabilities

### Confidence Levels

| Probability | Confidence Level |
|-------------|------------------|
| ≥85%        | Very High        |
| 75-84%      | High             |
| 65-74%      | Medium           |
| 55-64%      | Moderate         |
| <55%        | Low              |

## Integration with API

The Smart Bets AI module integrates with the User API:

```python
# In user-api/main.py
from smart_bets_ai.model_trainer import SmartBetsModelTrainer
from smart_bets_ai.predictor import SmartBetsPredictor

# Load models on startup
trainer = SmartBetsModelTrainer()
trainer.load_models(version='latest')
predictor = SmartBetsPredictor(trainer)

# API endpoint
@app.get("/api/v1/predictions/smart-bets/{match_id}")
async def get_smart_bet(match_id: str):
    # Fetch match data from database
    match_data = get_match_from_db(match_id)
    
    # Generate prediction
    prediction = predictor.predict_match(match_data)
    
    return prediction
```

## Model Versioning

Models are saved with version identifiers:

```
models/
├── v20251114_150000/
│   ├── home_win.pkl
│   ├── draw.pkl
│   ├── away_win.pkl
│   ├── over_2_5.pkl
│   ├── btts.pkl
│   ├── ...
│   ├── feature_names.json
│   ├── training_metrics.json
│   └── metadata.json
└── v20251115_090000/
    └── ...
```

## Testing

### Unit Tests

```bash
# Run tests
pytest smart-bets-ai/tests/

# With coverage
pytest smart-bets-ai/tests/ --cov=smart_bets_ai
```

### Validation

```python
# Validate on historical data
from smart_bets_ai.validation import validate_predictions

results = validate_predictions(
    predictor=predictor,
    test_data=test_matches,
    test_results=test_results
)

print(f"Accuracy: {results['accuracy']:.3f}")
print(f"Calibration: {results['calibration_score']:.3f}")
```

## Performance Optimization

### Caching

Predictions should be cached for fast API responses:

```python
import redis

# Cache prediction
cache_key = f"smart_bet:{match_id}"
cache.setex(cache_key, 3600, json.dumps(prediction))  # 1 hour TTL
```

### Batch Processing

For daily predictions, use batch processing:

```python
# Process all upcoming matches
upcoming_matches = get_upcoming_matches()
predictions = predictor.predict_batch(upcoming_matches)

# Store in database
store_predictions_in_db(predictions)
```

## Next Steps

1. **Golden Bets AI**: Filter Smart Bets for 85%+ confidence
2. **Value Bets AI**: Compare probabilities with odds
3. **Summary Generator**: Add explanations to predictions
4. **API Integration**: Expose predictions via REST endpoints

## Dependencies

```
xgboost>=1.7.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
sqlalchemy>=2.0.0
```

## Support

For issues or questions:
- Check training metrics for model performance
- Review feature importance for insights
- Validate predictions on test data
- Monitor calibration scores

---

**Status**: ✅ Complete and Production-Ready
**Version**: 1.0.0
**Last Updated**: November 14, 2025
