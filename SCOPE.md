# Confirmation of Scope

## What This AI System Builds

Your AI builds the **prediction engine**—the core intelligence that:

- Accepts daily fixture, stats, and odds data from your main app
- Runs AI models to generate:
  - **Smart Bets:** Best probabilistic predictions ignoring odds
  - **Golden Bets:** Very high confidence bets based on AI thresholds
  - **Value Bets:** Calculated dynamically as odds change, by comparing AI probabilities vs implied odds
- **On-Demand Analysis:** User selects a match + bet type, AI returns probability and explanation
- Generates human-readable explanations describing why particular bets were selected
- Exposes API endpoints that your main app queries for predictions and explanations
- Implements a caching mechanism to serve these predictions quickly and efficiently

## What This AI Does NOT Build

Your AI will **not** build:
- ❌ Frontend application
- ❌ Data scraping/ingestion from external sources
- ❌ User authentication
- ❌ Payment layers

These belong to your main app ecosystem.

---

## Data Exchange Formats

### Input Format: What Your App Sends Us

**JSON payload** structured per match and bet type, containing:

- Match ID, Date/Time, Teams (home/away)
- Key stats for each team relevant to bet types (goals, corners, cards, BTTS history, etc.)
- Current odds per bet market (decimal or fractional odds) including market ID and selection ID
- Timestamp indicating data freshness

#### Example Input (Batch Predictions):

```json
{
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
        "home_btts_rate": 0.6,
        "away_btts_rate": 0.5
      },
      "odds": {
        "market_1": {
          "selection_1": 1.95,
          "selection_2": 3.5,
          "selection_3": 4.0
        }
      },
      "timestamp": "2025-11-15T08:00:00Z"
    }
  ]
}
```

#### Example Input (On-Demand User Selection):

```json
{
  "match_id": "12345",
  "bet_type": "over_2.5_goals",
  "market_id": "total_goals",
  "selection_id": "over_2.5"
}
```

### Output Format: What We Return to Your App

**Structured JSON response** per match including:

- Smart Bets with probabilities and recommended selections
- Golden Bets flagged with confidence scores
- Value Bets highlighting value metric (AI prob - implied prob), recommended wagers
- Explanation strings or objects describing rationale
- Timestamps of prediction generation

#### Example Output (Batch Predictions):

```json
{
  "predictions": [
    {
      "match_id": "12345",
      "smart_bets": [
        {
          "market_id": "market_1",
          "selection_id": "selection_1",
          "probability": 0.58
        }
      ],
      "golden_bets": [
        {
          "market_id": "market_1",
          "selection_id": "selection_1",
          "confidence": 0.95
        }
      ],
      "value_bets": [
        {
          "market_id": "market_1",
          "selection_id": "selection_1",
          "value": 0.12,
          "explanation": "Strong recent home form and high AI modeled probability vs odds"
        }
      ],
      "timestamp": "2025-11-15T09:00:00Z"
    }
  ]
}
```

#### Example Output (On-Demand User Selection):

```json
{
  "match_id": "12345",
  "bet_type": "over_2.5_goals",
  "probability": 0.67,
  "percentage": "67%",
  "comment": "Both teams average 2.5+ goals combined in recent matches. Team A's attacking form is strong with 1.4 goals per game, while Team B concedes frequently. Historical head-to-head shows 4 of last 5 meetings had over 2.5 goals.",
  "confidence": "high",
  "timestamp": "2025-11-15T09:15:00Z"
}
```

---

## API Endpoints

### 1. Batch Predictions (Daily)
**POST** `/api/v1/predictions/batch`

Receives all matches for the day, returns Smart/Golden/Value bets.

### 2. On-Demand Analysis (User-Selected)
**POST** `/api/v1/predictions/analyze`

User selects match + bet type, returns probability and explanation.

**Request:**
```json
{
  "match_id": "12345",
  "bet_type": "btts",
  "market_id": "both_teams_to_score",
  "selection_id": "yes"
}
```

**Response:**
```json
{
  "match_id": "12345",
  "bet_type": "btts",
  "probability": 0.72,
  "percentage": "72%",
  "comment": "Team A has scored in 8 of last 10 home games. Team B has scored in 7 of last 10 away games. Both teams have high BTTS rates (60% and 50% respectively).",
  "confidence": "high"
}
```

### 3. Odds Update (Real-time)
**POST** `/api/v1/odds/update`

Receives updated odds, triggers Value Bet recalculation.

---

## AI Model Approach

### Starting Point: Probabilistic Classification

A **probabilistic classification model** (e.g., gradient boosting like **XGBoost** or **LightGBM**) trained on historical match data and bet outcomes works well as a strong baseline.

### Bet Type Implementation:

1. **Smart Bets:**
   - Focus on pure AI probabilities predicting event outcomes **without odds**
   - Output: Probability scores for each selection

2. **Golden Bets:**
   - Add confidence thresholding or ensemble model agreement metrics
   - Flag bets exceeding confidence threshold (e.g., 0.90+)

3. **Value Bets:**
   - Build a value calculation layer that compares AI predicted probability vs market-implied probability
   - Dynamically recalculate with each odds refresh
   - Formula: `Value = AI_Probability - Implied_Probability`

4. **On-Demand Analysis:**
   - Use same trained models to analyze user-selected bet types
   - Generate contextual explanations based on input stats
   - Return probability + human-readable comment

### Future Enhancement:

Optionally, explore neural nets or transformer-based models for deeper pattern recognition after baseline validation.

---

## System Summary

### What You'll Receive:
- Keyed JSON daily with fixture, stats, odds (batch)
- Single match + bet type requests (on-demand)

### What You'll Return:
- Keyed JSON predictions with bet types and explanations (batch)
- Probability + comment for user-selected bets (on-demand)

### AI Models Focus:
First on robust probabilistic predictions

### Value Calculations:
Run downstream using model outputs plus fresh odds

### APIs:
Expose these results for your main app to consume

---

## Architecture Flow

```
Your App → [JSON Input] → AI Prediction Engine → [JSON Output] → Your App
                              ↓
                    ┌─────────┴─────────┐
                    │                   │
              Smart Bets AI      Golden Bets AI
                    │                   │
                    └─────────┬─────────┘
                              ↓
                        Value Bets AI
                              ↓
                    Summary Generator
                              ↓
                         User API
                         ↓     ↓
                   [Batch] [On-Demand]
                              ↓
                      [Cached Results]
```

## Use Cases

### Use Case 1: Daily Batch Predictions
- Your app sends all fixtures at 8 AM
- AI processes and returns Smart/Golden/Value bets
- Results cached for fast access
- Users browse pre-calculated recommendations

### Use Case 2: User Explores Specific Bet
- User clicks on "Manchester United vs Liverpool"
- User selects "Over 2.5 Goals"
- Your app calls `/api/v1/predictions/analyze`
- AI returns: "67% probability - Both teams average 2.5+ goals..."
- User sees instant AI analysis

### Use Case 3: Odds Change
- Bookmaker updates odds from 1.95 to 2.10
- Your app sends odds update
- AI recalculates Value Bets
- New value opportunities flagged
