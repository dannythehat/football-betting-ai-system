# Smart Bets AI Module

## Overview
The Smart Bets AI module generates pure probabilistic predictions for the 4 target betting markets and selects the highest probability bet for each fixture.

## Target Markets
1. **Goals:** Over/Under 2.5
2. **Cards:** Over/Under 3.5
3. **Corners:** Over/Under 9.5
4. **BTTS:** Yes/No

## Components

### 1. Feature Engineering (`features.py`)
Transforms raw match data into ML-ready features:
- **Basic Features:** Combined averages, attack vs defense matchups
- **Goals Features:** Expected goals, offensive/defensive strength
- **Cards Features:** Card averages, disciplinary indicators
- **Corners Features:** Corner averages, dominance metrics
- **BTTS Features:** Scoring capability, conceding patterns
- **Form Features:** Recent form scores and differentials

### 2. Model Training (`train.py`)
Trains separate XGBoost models for each market:
- **Data Loading:** Processes historical match data
- **Model Training:** XGBoost with early stopping
- **Evaluation:** Accuracy, log loss, AUC-ROC metrics
- **Model Persistence:** Saves trained models and metadata

### 3. Prediction Service (`predict.py`)
Generates Smart Bets predictions:
- **Market Predictions:** Probabilities for all 4 markets
- **Smart Bet Selection:** Highest probability across markets
- **Explanations:** Context-aware reasoning
- **Batch Processing:** Multiple matches at once

## Usage

### Training Models

```bash
# From project root
python smart-bets-ai/train.py
```

This will:
1. Load historical match data from `test-data/historical_matches_sample.json`
2. Train 4 separate models (one per market)
3. Save models to `smart-bets-ai/models/`
4. Display performance metrics

### Making Predictions

```python
from smart_bets_ai.predict import SmartBetsPredictor

# Initialize predictor
predictor = SmartBetsPredictor()

# Prepare match data
match = {
    'match_id': 'TEST_001',
    'home_team': 'Manchester United',
    'away_team': 'Liverpool',
    'home_goals_avg': 1.8,
    'away_goals_avg': 2.1,
    'home_goals_conceded_avg': 1.0,
    'away_goals_conceded_avg': 0.8,
    'home_corners_avg': 6.2,
    'away_corners_avg': 5.8,
    'home_cards_avg': 2.1,
    'away_cards_avg': 1.9,
    'home_btts_rate': 0.65,
    'away_btts_rate': 0.70,
    'home_form': 'WWDWL',
    'away_form': 'WWWDW'
}

# Get Smart Bet
smart_bet = predictor.get_smart_bet(match)

print(f"Market: {smart_bet['selection_name']}")
print(f"Probability: {smart_bet['percentage']}")
print(f"Explanation: {smart_bet['explanation']}")
```

### API Endpoint

```bash
POST /api/v1/predictions/smart-bets
```

**Request:**
```json
{
  "matches": [
    {
      "match_id": "12345",
      "home_team": "Team A",
      "away_team": "Team B",
      "home_goals_avg": 1.8,
      "away_goals_avg": 1.2,
      "home_goals_conceded_avg": 0.9,
      "away_goals_conceded_avg": 1.4,
      "home_corners_avg": 6.2,
      "away_corners_avg": 4.8,
      "home_cards_avg": 2.1,
      "away_cards_avg": 2.5,
      "home_btts_rate": 0.55,
      "away_btts_rate": 0.60,
      "home_form": "WWDWL",
      "away_form": "LWDLW"
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "total_matches": 1,
  "predictions": [
    {
      "match_id": "12345",
      "smart_bet": {
        "market_id": "total_corners",
        "market_name": "Total Corners",
        "selection_id": "over_9.5",
        "selection_name": "Over 9.5 Corners",
        "probability": 0.87,
        "percentage": "87.0%",
        "explanation": "Highest probability outcome across all 4 analyzed markets...",
        "alternative_markets": [
          {
            "market_name": "Yes",
            "probability": 0.68
          },
          {
            "market_name": "Over 2.5 Goals",
            "probability": 0.67
          },
          {
            "market_name": "Over 3.5 Cards",
            "probability": 0.58
          }
        ]
      }
    }
  ],
  "model_version": "1.0.0"
}
```

## Model Architecture

### XGBoost Configuration
```python
{
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'logloss'
}
```

### Training Process
1. **Data Split:** 80% training, 20% validation
2. **Early Stopping:** Prevents overfitting (20 rounds)
3. **Stratified Split:** Maintains class balance
4. **Feature Engineering:** Market-specific features
5. **Model Evaluation:** Multiple metrics (accuracy, log loss, AUC)

## Performance Metrics

Models are evaluated on:
- **Accuracy:** Overall prediction correctness
- **Log Loss:** Probability calibration quality
- **AUC-ROC:** Discrimination ability

Target performance:
- Accuracy: >60%
- Log Loss: <0.65
- AUC-ROC: >0.65

## Files Structure

```
smart-bets-ai/
├── __init__.py           # Module initialization
├── features.py           # Feature engineering
├── train.py             # Model training script
├── predict.py           # Prediction service
├── README.md            # This file
└── models/              # Trained models (created after training)
    ├── goals_model.pkl
    ├── cards_model.pkl
    ├── corners_model.pkl
    ├── btts_model.pkl
    ├── feature_engineer.pkl
    └── metadata.json
```

## Next Steps

After Smart Bets AI is working:
1. **Golden Bets AI:** 85%+ confidence filtering
2. **Value Bets AI:** Expected value calculations
3. **Summary Generator:** Enhanced explanations
4. **Caching Layer:** Redis integration

## Testing

```bash
# Test feature engineering
python -c "from smart_bets_ai.features import FeatureEngineer; print('✅ Features OK')"

# Test training
python smart-bets-ai/train.py

# Test predictions
python smart-bets-ai/predict.py
```

## Troubleshooting

**Models not loading:**
- Ensure models are trained: `python smart-bets-ai/train.py`
- Check `smart-bets-ai/models/` directory exists

**Import errors:**
- Verify Python path includes project root
- Check all dependencies installed: `pip install -r requirements.txt`

**Low accuracy:**
- Need more training data (currently using sample)
- Adjust hyperparameters in `train.py`
- Add more features in `features.py`

## Notes

- Models are trained on sample data (50 matches)
- Production requires 1000+ historical matches
- Retrain models periodically with new data
- Monitor prediction calibration over time
