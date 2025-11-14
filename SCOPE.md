# Confirmation of Scope

## What This AI System Builds

Your AI builds the **prediction engine**—the core intelligence that delivers four distinct types of betting intelligence:

### 1. Golden Bets (Premium)
- **Daily 1-3 picks with 85%+ win probability**
- Focus: Safety and consistency, regardless of odds
- Target: Premium users seeking high-certainty bets
- Value: Builds credibility through high success rates

### 2. Value Bets (Premium)
- **Daily top 3 picks with positive expected value**
- Focus: Long-term profitability (AI prob > implied prob)
- Target: Strategic bettors understanding EV concepts
- Value: Educates on market inefficiencies and value betting

### 3. Smart Bets (Per Fixture)
- **Best single bet per match across all markets**
- Focus: Match-specific optimization
- Target: Users betting on specific games
- Value: Data-backed guidance for individual fixtures

### 4. Custom Bet Analysis (Interactive)
- **User-selected fixture + bet type analysis**
- Focus: User hypothesis testing and education
- Target: Advanced users wanting control
- Value: Interactive learning and transparency

### Core Capabilities
- Accepts daily fixture, stats, and odds data from your main app
- Runs AI models to generate all four bet types
- Generates transparent, educational explanations for every recommendation
- Exposes API endpoints that your main app queries
- Implements caching for fast response times
- Dynamically recalculates Value Bets as odds change

## What This AI Does NOT Build

Your AI will **not** build:
- ❌ Frontend application
- ❌ Data scraping/ingestion from external sources
- ❌ User authentication
- ❌ Payment layers

These belong to your main app ecosystem.

---

## Strategic Feature Positioning

### Feature Comparison Matrix

| Feature | Focus | Confidence Level | Frequency | User Access |
|---------|-------|------------------|-----------|-------------|
| Golden Bets | Win rate | Highest (85%+) | 1-3 daily | Premium |
| Value Bets | ROI/EV | Variable | Top 3 daily | Premium |
| Smart Bets | Per-match | High | Per fixture | All users |
| Custom Analysis | Education | Variable | On-demand | All users |

### User Journey Design

**Free Users:**
- Access Smart Bets (quality hook demonstrating AI capability)
- See Custom Analysis (interactive engagement)

**Premium Users:**
- Unlock Golden Bets (safe daily picks)
- Unlock Value Bets (profit-focused picks)
- Full access to all features

### Confidence Level Expectations

**Golden Bets:** 85%+ probability threshold
- Highest confidence
- Focus on win rate over value
- May have lower odds but higher success rate

**Value Bets:** Variable probability (typically 55-75%)
- Focus on positive expected value
- May have lower win rates than Golden Bets
- Higher long-term ROI potential

**Smart Bets:** High probability (typically 60-80%)
- Best option per fixture across all markets
- Pure probabilistic analysis

**Custom Analysis:** Variable (typically lower than Smart Bets)
- User-selected bets often have lower confidence
- Educational tool showing why certain bets are weaker
- Exception: When user's choice matches AI's top pick

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
        "away_btts_rate": 0.5,
        "home_form": "WWWDW",
        "away_form": "LWDLL",
        "home_goals_conceded_avg": 0.8,
        "away_goals_conceded_avg": 1.6
      },
      "odds": {
        "match_result": {
          "home_win": 1.95,
          "draw": 3.5,
          "away_win": 4.0
        },
        "total_goals": {
          "over_2.5": 2.10,
          "under_2.5": 1.75
        },
        "btts": {
          "yes": 1.85,
          "no": 2.00
        }
      },
      "timestamp": "2025-11-15T08:00:00Z"
    }
  ]
}
```

#### Example Input (Custom Bet Analysis):

```json
{
  "match_id": "12345",
  "bet_type": "over_2.5_goals",
  "market_id": "total_goals",
  "selection_id": "over_2.5"
}
```

### Output Format: What We Return to Your App

**Structured JSON response** per match including all four bet types with transparent explanations.

#### Example Output (Batch Predictions):

```json
{
  "predictions": [
    {
      "match_id": "12345",
      "timestamp": "2025-11-15T09:00:00Z",
      
      "golden_bets": [
        {
          "market_id": "match_result",
          "market_name": "Match Result",
          "selection_id": "home_win",
          "selection_name": "Team A Win",
          "probability": 0.87,
          "percentage": "87%",
          "confidence": "very_high",
          "explanation": "Team A has won 9 of last 10 home games with consistent defensive performance (0.8 goals conceded/game). Current form is excellent (WWWDW) while Team B struggles away (LWDLL). Historical head-to-head shows Team A dominance at home.",
          "key_factors": [
            "Exceptional home form (90% win rate)",
            "Strong defensive record",
            "Opponent's poor away form"
          ]
        }
      ],
      
      "value_bets": [
        {
          "market_id": "btts",
          "market_name": "Both Teams To Score",
          "selection_id": "yes",
          "selection_name": "Yes",
          "ai_probability": 0.68,
          "implied_probability": 0.54,
          "bookmaker_odds": 1.85,
          "value": 0.14,
          "value_percentage": "14%",
          "expected_value": "+25.9%",
          "explanation": "AI probability (68%) significantly exceeds bookmaker's implied probability (54%), indicating strong positive expected value. Both teams have high BTTS rates (60% and 50%) and score consistently. Team A averages 1.4 goals at home, Team B averages 1.1 away despite poor form.",
          "key_factors": [
            "High BTTS historical rates for both teams",
            "Consistent scoring patterns",
            "Market undervaluing BTTS probability"
          ],
          "risk_note": "Value bets focus on long-term profitability. Individual bet success rate may be lower than Golden Bets."
        }
      ],
      
      "smart_bets": [
        {
          "market_id": "match_result",
          "market_name": "Match Result",
          "selection_id": "home_win",
          "selection_name": "Team A Win",
          "probability": 0.87,
          "percentage": "87%",
          "explanation": "Highest probability outcome across all analyzed markets for this fixture. Team A's home dominance and Team B's away struggles create clear probability advantage.",
          "alternative_markets": [
            {
              "market_name": "Total Goals Over 2.5",
              "probability": 0.67
            },
            {
              "market_name": "BTTS Yes",
              "probability": 0.68
            }
          ]
        }
      ]
    }
  ],
  
  "summary": {
    "total_matches": 1,
    "golden_bets_count": 1,
    "value_bets_count": 1,
    "processing_time_ms": 245
  }
}
```

#### Example Output (Custom Bet Analysis):

```json
{
  "match_id": "12345",
  "bet_type": "over_2.5_goals",
  "market_id": "total_goals",
  "selection_id": "over_2.5",
  "probability": 0.67,
  "percentage": "67%",
  "verdict": "moderate",
  "confidence": "medium",
  "explanation": "Both teams average 2.5+ goals combined in recent matches. Team A's attacking form is strong with 1.4 goals per game at home, while Team B concedes frequently (1.6 goals/game away). Historical head-to-head shows 4 of last 5 meetings had over 2.5 goals.",
  "key_factors": [
    "Combined goals average: 2.5 per game",
    "Team A strong attack (1.4 goals/game)",
    "Team B weak defense (1.6 conceded/game)",
    "Recent head-to-head trend supports over 2.5"
  ],
  "comparison_note": "This confidence (67%) is lower than our Smart Bet recommendation for this fixture (87% for Team A Win), which analyzed all available markets. Your selected bet is still viable but not our top pick.",
  "smart_bet_alternative": {
    "market_name": "Match Result - Team A Win",
    "probability": 0.87,
    "reason": "Higher probability based on comprehensive market analysis"
  },
  "timestamp": "2025-11-15T09:15:00Z"
}
```

---

## API Endpoints

### 1. Batch Predictions (Daily)
**POST** `/api/v1/predictions/batch`

Receives all matches for the day, returns Golden/Value/Smart bets.

**Request:**
```json
{
  "matches": [/* array of match objects */]
}
```

**Response:**
```json
{
  "predictions": [/* array of prediction objects with all bet types */],
  "summary": {/* processing summary */}
}
```

### 2. Custom Bet Analysis (User-Selected)
**POST** `/api/v1/predictions/analyze`

User selects match + bet type, returns probability, verdict, and educational explanation.

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
  "verdict": "good",
  "confidence": "high",
  "explanation": "Team A has scored in 8 of last 10 home games. Team B has scored in 7 of last 10 away games. Both teams have high BTTS rates (60% and 50% respectively).",
  "comparison_note": "This aligns closely with our Smart Bet analysis for this fixture.",
  "key_factors": [/* array of key factors */]
}
```

### 3. Odds Update (Real-time)
**POST** `/api/v1/odds/update`

Receives updated odds, triggers Value Bet recalculation.

**Request:**
```json
{
  "match_id": "12345",
  "odds": {
    "btts": {
      "yes": 1.95,
      "no": 1.90
    }
  },
  "timestamp": "2025-11-15T10:30:00Z"
}
```

**Response:**
```json
{
  "match_id": "12345",
  "value_bets_updated": true,
  "new_value_bets": [/* updated value bet objects */]
}
```

---

## AI Model Approach

### Starting Point: Probabilistic Classification

A **probabilistic classification model** (e.g., gradient boosting like **XGBoost** or **LightGBM**) trained on historical match data and bet outcomes works well as a strong baseline.

### Bet Type Implementation:

#### 1. Golden Bets
- **Confidence threshold:** 85%+ probability
- **Ensemble validation:** Multiple model agreement
- **Historical verification:** Backtest win rate validation
- **Selection criteria:** Highest confidence picks regardless of odds
- **Daily limit:** 1-3 bets maximum to maintain quality

#### 2. Value Bets
- **Formula:** `Value = AI_Probability - Implied_Probability`
- **Threshold:** Minimum 10% positive value
- **Dynamic recalculation:** Updates with each odds refresh
- **Educational focus:** Detailed explanations of value concept
- **Risk communication:** Clear messaging about variance vs Golden Bets

#### 3. Smart Bets
- **Market analysis:** Evaluate all tracked bet types per fixture
- **Selection:** Highest probability outcome
- **Pure probability:** No odds consideration
- **Transparency:** Show alternative markets with probabilities

#### 4. Custom Bet Analysis
- **Same models:** Use trained models on user-selected bets
- **Contextual explanations:** Generate reasoning based on input stats
- **Comparison:** Show how it compares to Smart Bet recommendation
- **Educational tone:** Explain why certain bets are stronger/weaker
- **Confidence context:** Set expectations about typical confidence levels

### Model Training Focus

- **Accuracy:** Minimize prediction error
- **Calibration:** Ensure probabilities reflect true likelihood
- **Explainability:** Feature importance for transparent reasoning
- **Robustness:** Validate across different leagues and seasons

### Future Enhancement

Optionally, explore neural nets or transformer-based models for deeper pattern recognition after baseline validation.

---

## System Summary

### What You'll Receive:
- Keyed JSON daily with fixture, stats, odds (batch)
- Single match + bet type requests (custom analysis)
- Odds updates for value recalculation

### What You'll Return:
- Golden Bets: 1-3 daily high-confidence picks (85%+)
- Value Bets: Top 3 daily positive EV opportunities
- Smart Bets: Best bet per fixture across all markets
- Custom Analysis: Probability + educational explanation for user-selected bets

### AI Models Focus:
Robust probabilistic predictions with transparent reasoning

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
              Smart Bets AI      Golden Bets AI (85%+ threshold)
                    │                   │
                    └─────────┬─────────┘
                              ↓
                        Value Bets AI (EV calculation)
                              ↓
                    Summary Generator (Educational explanations)
                         ↓     ↓
                   [Batch] [Custom Analysis]
                              ↓
                         User API
                              ↓
                      [Cached Results]
```

---

## Use Cases

### Use Case 1: Daily Batch Predictions
- Your app sends all fixtures at 8 AM
- AI processes and returns Golden/Value/Smart bets
- Results cached for fast access
- Premium users browse Golden and Value Bets
- All users see Smart Bets per fixture

### Use Case 2: User Explores Custom Bet
- User clicks on "Manchester United vs Liverpool"
- User selects "Over 2.5 Goals"
- Your app calls `/api/v1/predictions/analyze`
- AI returns: "67% probability - Both teams average 2.5+ goals..."
- AI notes: "Our Smart Bet for this fixture is Home Win (78% probability)"
- User gains educational insight into bet quality

### Use Case 3: Odds Change
- Bookmaker updates odds from 1.95 to 2.10
- Your app sends odds update
- AI recalculates Value Bets
- New value opportunities flagged
- Premium users notified of updated Value Bets

### Use Case 4: User Learning Journey
- Free user explores Smart Bets (sees AI quality)
- User tests own hypotheses via Custom Analysis
- User learns why certain bets are stronger
- User upgrades to Premium for Golden/Value Bets
- User combines all features for informed betting strategy

---

## Key Differentiators

### Transparency
Every prediction includes detailed, human-readable explanations of reasoning and key factors.

### Education
Custom Analysis teaches users about betting dynamics, not just providing picks.

### Tiered Intelligence
Four distinct features serve different user needs and betting strategies.

### Confidence Context
Clear communication about confidence levels and expected performance of each bet type.

### Dynamic Updates
Value Bets recalculate as odds change, capturing market inefficiencies in real-time.

---

**This AI system delivers a complete betting intelligence ecosystem combining safety (Golden), profitability (Value), match-specific guidance (Smart), and interactive learning (Custom Analysis).**
