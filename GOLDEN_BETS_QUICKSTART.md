# Golden Bets AI - Quick Start Guide

**⏱️ Setup Time:** 5 minutes  
**✅ Status:** Phase 3 Complete  
**🎯 Output:** 1-3 daily picks with 85%+ confidence

---

## What Are Golden Bets?

Golden Bets are the **safest, highest-confidence betting picks** identified by the AI system each day. They represent the top 1-3 opportunities across all analyzed matches and the 4 target markets.

### Key Features

- **85%+ Win Probability:** Only picks with exceptional confidence
- **90%+ Model Agreement:** Multiple AI models must agree
- **1-3 Daily Picks:** Quality over quantity
- **Transparent Reasoning:** Clear explanations for each selection

---

## Quick Start (5 Minutes)

### 1. Prerequisites

Ensure Smart Bets AI is working:

```bash
# Test Smart Bets first
cd smart-bets-ai
python predict.py
```

### 2. Install Dependencies

```bash
cd golden-bets-ai
pip install -r requirements.txt
```

### 3. Test Golden Bets Predictor

```bash
# Run full prediction pipeline
python predict.py
```

**Or test just the filter:**

```bash
python test_filter.py
```

**Expected Output:**
```
============================================================
GOLDEN BETS PREDICTIONS
============================================================

Found 3 Golden Bets:

1. Team C vs Team D
   Market: Total Goals
   Prediction: Over 2.5
   Confidence: 92.0%
   Agreement: 95.2%
   🏆 Golden Bet Selection Criteria Met:
   • AI Confidence: 92.0% (≥85% threshold)
   • Model Agreement: 95.2% (≥90% threshold)
   • Market: Total Goals
   • This represents one of the top 1-3 safest bets identified today
```

### 4. Use in Python (Recommended)

```python
from golden_bets_ai import GoldenBetsPredictor

# Initialize predictor (handles Smart Bets automatically)
predictor = GoldenBetsPredictor()

# Get Golden Bets predictions
golden_bets = predictor.predict(matches)

# Display results
for bet in golden_bets:
    print(f"🏆 {bet['home_team']} vs {bet['away_team']}")
    print(f"   {bet['market_name']}: {bet['selection_name']}")
    print(f"   Confidence: {bet['confidence_score']:.1%}")
    print(f"   Golden Score: {bet['golden_score']:.3f}")
    print(f"   {bet['reasoning']}")
```

### 4b. Manual Filtering (Advanced)

```python
from golden_bets_ai import GoldenBetsFilter
from smart_bets_ai.predict import SmartBetsPredictor

# Get Smart Bets predictions
predictor = SmartBetsPredictor()
smart_predictions = predictor.predict(matches)

# Get model probabilities
model_probs = predictor.get_model_probabilities(matches)

# Filter for Golden Bets
filter = GoldenBetsFilter()
golden_bets = filter.filter_golden_bets(
    smart_bets_predictions=smart_predictions,
    model_probabilities=model_probs
)

# Display results
for bet in golden_bets:
    print(f"🏆 {bet['home_team']} vs {bet['away_team']}")
    print(f"   {bet['market_name']}: {bet['selection_name']}")
    print(f"   Confidence: {bet['confidence_score']:.1%}\"")
    print(f"   Golden Score: {bet['golden_score']:.3f}")
```

### 5. API Usage

Start the API server:

```bash
cd user-api
python main.py
```

Make a request:

```bash
curl -X POST http://localhost:8000/api/v1/predictions/golden-bets \
  -H "Content-Type: application/json" \
  -d '{
    "matches": [
      {
        "match_id": "12345",
        "datetime": "2025-11-15T14:00:00Z",
        "home_team": "Team A",
        "away_team": "Team B",
        "stats": {
          "home_goals_avg": 1.4,
          "away_goals_avg": 1.1,
          "home_corners_avg": 5.2,
          "away_corners_avg": 4.8,
          "home_cards_avg": 2.1,
          "away_cards_avg": 1.8,
          "home_btts_rate": 0.6,
          "away_btts_rate": 0.5
        },
        "odds": {
          "total_goals": {"over_2.5": 2.10, "under_2.5": 1.75},
          "total_cards": {"over_3.5": 1.95, "under_3.5": 1.85},
          "total_corners": {"over_9.5": 1.90, "under_9.5": 1.90},
          "btts": {"yes": 1.85, "no": 2.00}
        }
      }
    ]
  }'
```

**Response:**

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
      "reasoning": "🏆 Golden Bet Selection Criteria Met:\n• AI Confidence: 87.0% (≥85% threshold)\n• Model Agreement: 95.0% (≥90% threshold)\n• Market: Total Corners\n• This represents one of the top 1-3 safest bets identified today"
    }
  ],
  "total_candidates": 15,
  "selected_count": 1
}
```

---

## Understanding the Output

### Golden Score Calculation

```
Golden Score = (0.7 × Probability) + (0.3 × Agreement)
```

**Example:**
- Probability: 87%
- Agreement: 95%
- Golden Score: (0.7 × 0.87) + (0.3 × 0.95) = **0.894**

### Selection Process

1. **Filter by Confidence:** Only predictions ≥85%
2. **Check Agreement:** Only predictions with ≥90% model consensus
3. **Rank by Golden Score:** Sort candidates by composite score
4. **Select Top 1-3:** Return highest-scoring picks

---

## Configuration

Adjust thresholds in `golden-bets-ai/config.py`:

```python
# Stricter filtering (fewer picks, higher confidence)
CONFIDENCE_THRESHOLD = 0.90  # 90% minimum
MIN_ENSEMBLE_AGREEMENT = 0.95  # 95% agreement

# More lenient filtering (more picks, lower confidence)
CONFIDENCE_THRESHOLD = 0.80  # 80% minimum
MIN_ENSEMBLE_AGREEMENT = 0.85  # 85% agreement
```

---

## Common Scenarios

### Scenario 1: No Golden Bets Found

**Cause:** No predictions meet thresholds  
**Solution:**
```python
# Lower thresholds temporarily
filter = GoldenBetsFilter()
filter.confidence_threshold = 0.80
filter.min_agreement = 0.85
```

### Scenario 2: Too Many Golden Bets

**Cause:** Many high-confidence predictions  
**Solution:**
```python
# Increase thresholds or reduce max picks
filter = GoldenBetsFilter()
filter.confidence_threshold = 0.90
filter.max_picks = 2
```

### Scenario 3: Low Ensemble Agreement

**Cause:** Models disagree on predictions  
**Action:** Review model training data consistency

---

## Integration with Your App

### Daily Golden Bets Workflow (Using Predictor)

```python
import schedule
import time
from datetime import datetime
from golden_bets_ai import GoldenBetsPredictor

def fetch_daily_golden_bets():
    """Fetch Golden Bets for today's matches"""
    
    # 1. Get today's fixtures from your app
    matches = get_todays_fixtures()
    
    # 2. Get Golden Bets (handles Smart Bets automatically)
    predictor = GoldenBetsPredictor()
    golden_bets = predictor.predict(matches)
    
    # 3. Send to your app
    send_to_app(golden_bets)
    
    print(f"[{datetime.now()}] Sent {len(golden_bets)} Golden Bets")

# Schedule daily at 9 AM
schedule.every().day.at("09:00").do(fetch_daily_golden_bets)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### Manual Workflow (Using Filter)

```python
import schedule
import time
from datetime import datetime
from smart_bets_ai.predict import SmartBetsPredictor
from golden_bets_ai import GoldenBetsFilter

def fetch_daily_golden_bets():
    """Fetch Golden Bets for today's matches"""
    
    # 1. Get today's fixtures from your app
    matches = get_todays_fixtures()
    
    # 2. Get Smart Bets predictions
    predictor = SmartBetsPredictor()
    smart_predictions = predictor.predict(matches)
    model_probs = predictor.get_model_probabilities(matches)
    
    # 3. Filter for Golden Bets
    filter = GoldenBetsFilter()
    golden_bets = filter.filter_golden_bets(
        smart_bets_predictions=smart_predictions,
        model_probabilities=model_probs
    )
    
    # 4. Send to your app
    send_to_app(golden_bets)
    
    print(f"[{datetime.now()}] Sent {len(golden_bets)} Golden Bets")

# Schedule daily at 9 AM
schedule.every().day.at("09:00").do(fetch_daily_golden_bets)

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## Performance Tracking

Monitor Golden Bets performance:

```python
def track_golden_bets_performance(golden_bets, results):
    """Track win rate and accuracy"""
    
    total = len(golden_bets)
    wins = sum(1 for bet, result in zip(golden_bets, results) if result == 'win')
    
    win_rate = wins / total if total > 0 else 0
    avg_confidence = sum(bet['confidence_score'] for bet in golden_bets) / total
    avg_agreement = sum(bet['ensemble_agreement'] for bet in golden_bets) / total
    
    print(f"Golden Bets Performance:")
    print(f"  Win Rate: {win_rate:.1%} (Target: ≥85%)")
    print(f"  Avg Confidence: {avg_confidence:.1%}")
    print(f"  Avg Agreement: {avg_agreement:.1%}")
    print(f"  Total Picks: {total}")
```

---

## Next Steps

✅ **Phase 3 Complete:** Golden Bets AI  
🔄 **Phase 4 Next:** Value Bets AI (EV calculations)  
⏳ **Phase 5:** Custom Bet Analysis

### Recommended Actions

1. **Test with Real Data:** Run on actual fixtures
2. **Monitor Performance:** Track win rates over 30 days
3. **Adjust Thresholds:** Fine-tune based on results
4. **Integrate with App:** Add to your frontend

---

## Troubleshooting

### Import Errors

```bash
# Ensure all dependencies installed
pip install -r golden-bets-ai/requirements.txt
pip install -r smart-bets-ai/requirements.txt
```

### No Model Probabilities

```python
# Golden Bets works without ensemble agreement
golden_bets = filter.filter_golden_bets(
    smart_bets_predictions=smart_predictions
    # model_probabilities not required
)
```

### API Connection Issues

```bash
# Check API is running
curl http://localhost:8000/api/v1/predictions/golden-bets/config
```

---

## Module Files

- **`predict.py`** - Main prediction pipeline (recommended)
- **`filter.py`** - Core filtering logic
- **`config.py`** - Configuration settings
- **`test_filter.py`** - Test script with sample data

---

## Support

- **Documentation:** [golden-bets-ai/README.md](golden-bets-ai/README.md)
- **Configuration:** [golden-bets-ai/config.py](golden-bets-ai/config.py)
- **Prediction Pipeline:** [golden-bets-ai/predict.py](golden-bets-ai/predict.py)
- **Test Script:** [golden-bets-ai/test_filter.py](golden-bets-ai/test_filter.py)
- **Main README:** [README.md](README.md)

**🎉 You're ready to use Golden Bets AI!**
