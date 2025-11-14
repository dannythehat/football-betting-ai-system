# Confirmation of Scope

## What This AI System Builds

Your AI builds the **prediction engine**—the core intelligence that delivers four distinct types of betting intelligence across **4 specific betting markets only**.

## Target Markets (ONLY THESE 4)

This system focuses exclusively on the following markets:

1. **Goals: Over/Under 2.5**
2. **Cards: Over/Under 3.5** 
3. **Corners: Over/Under 9.5**
4. **BTTS (Both Teams To Score): Yes/No**

**Note:** We do NOT predict match results (home win/draw/away win), first/second half markets, or any other markets. The system is laser-focused on these 4 markets only.

---

### 1. Golden Bets (Premium)
- **Daily 1-3 picks with 85%+ win probability**
- Focus: Safety and consistency, regardless of odds
- Target: Premium users seeking high-certainty bets
- Value: Builds credibility through high success rates
- **Markets:** Any of the 4 target markets with 85%+ confidence

### 2. Value Bets (Premium)
- **Daily top 3 picks with positive expected value**
- Focus: Long-term profitability (AI prob > implied prob)
- Target: Strategic bettors understanding EV concepts
- Value: Educates on market inefficiencies and value betting
- **Markets:** Any of the 4 target markets with positive EV

### 3. Smart Bets (Per Fixture)
- **Best single bet per match across all 4 markets**
- Focus: Match-specific optimization
- Target: Users betting on specific games
- Value: Data-backed guidance for individual fixtures
- **Markets:** Evaluates all 4 markets, returns highest probability

### 4. Custom Bet Analysis (Interactive)
- **User-selected fixture + bet type analysis**
- Focus: User hypothesis testing and education
- Target: Advanced users wanting control
- Value: Interactive learning and transparency
- **Markets:** User selects from any of the 4 markets

### Core Capabilities
- Accepts daily fixture, stats, and odds data from your main app
- Runs AI models to generate predictions for the 4 target markets
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
- ❌ Match result predictions (home/draw/away)
- ❌ First/second half markets
- ❌ Any markets outside the 4 specified

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
- Best option per fixture across all 4 markets
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
- Key stats for each team relevant to the 4 markets (goals, cards, corners, BTTS history)
- Current odds for all 4 markets (decimal odds) including market ID and selection ID
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
        "home_goals_conceded_avg": 0.8,
        "away_goals_conceded_avg": 1.6,
        "home_corners_avg": 5.2,
        "away_corners_avg": 4.8,
        "home_cards_avg": 2.1,
        "away_cards_avg": 1.8,
        "home_btts_rate": 0.6,
        "away_btts_rate": 0.5,
        "home_form": "WWWDW",
        "away_form": "LWDLL"
      },
      "odds": {
        "total_goals": {
          "over_2.5": 2.10,
          "under_2.5": 1.75
        },
        "total_cards": {
          "over_3.5": 1.95,
          "under_3.5": 1.85
        },
        "total_corners": {
          "over_9.5": 1.90,
          "under_9.5": 1.90
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
          "market_id": "total_corners",
          "market_name": "Total Corners",
          "selection_id": "over_9.5",
          "selection_name": "Over 9.5 Corners",
          "probability": 0.87,
          "percentage": "87%",
          "confidence": "very_high",
          "explanation": "Both teams average 10 corners combined per match. Team A averages 5.2 corners at home, Team B averages 4.8 away. Last 8 meetings between these teams averaged 11.3 corners. Both teams play attacking football with high corner rates.",
          "key_factors": [
            "Combined corners average: 10.0 per match",
            "Historical head-to-head: 11.3 corners average",
            "Both teams have attacking playing styles",
            "Recent form supports high corner counts"
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
            "Market undervaluing BTTS probability",
            "Both teams have scored in 6 of last 8 matches"
          ],
          "risk_note": "Value bets focus on long-term profitability. Individual bet success rate may be lower than Golden Bets."
        }
      ],
      
      "smart_bets": [
        {
          "market_id": "total_corners",
          "market_name": "Total Corners",
          "selection_id": "over_9.5",
          "selection_name": "Over 9.5 Corners",
          "probability": 0.87,
          "percentage": "87%",
          "explanation": "Highest probability outcome across all 4 analyzed markets for this fixture. Combined corners average of 10.0 per match with consistent historical data supporting this prediction.",
          "alternative_markets": [
            {
              "market_name": "BTTS Yes",
              "probability": 0.68
            },
            {
              "market_name": "Total Goals Over 2.5",
              "probability": 0.67
            },
            {
              "market_name": "Total Cards Over 3.5",
              "probability": 0.58
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
  "comparison_note": "This confidence (67%) is lower than our Smart Bet recommendation for this fixture (87% for Over 9.5 Corners), which analyzed all 4 available markets. Your selected bet is still viable but not our top pick.",
  "smart_bet_alternative": {
    "market_name": "Total Corners - Over 9.5",
    "probability": 0.87,
    "reason": "Higher probability based on comprehensive market analysis across all 4 markets"
  },
  "timestamp": "2025-11-15T09:15:00Z"
}
```

---

## API Endpoints

### 1. Batch Predictions (Daily)
**POST** `/api/v1/predictions/batch`

Receives all matches for the day, returns Golden/Value/Smart bets across the 4 markets.

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

User selects match + one of the 4 bet markets, returns probability, verdict, and educational explanation.

**Request:**
```json
{
  "match_id": "12345",
  "bet_type": "over_9.5_corners",
  "market_id": "total_corners",
  "selection_id": "over_9.5"
}
```

**Response:**
```json
{
  "match_id": "12345",
  "bet_type": "over_9.5_corners",
  "probability": 0.87,
  "percentage": "87%",
  "verdict": "excellent",
  "confidence": "very_high",
  "explanation": "Both teams average 10 corners combined. Historical data strongly supports this prediction.",
  "comparison_note": "This matches our Smart Bet recommendation for this fixture.",
  "key_factors": [/* array of key factors */]
}
```

### 3. Odds Update (Real-time)
**POST** `/api/v1/odds/update`

Receives updated odds for any of the 4 markets, triggers Value Bet recalculation.

**Request:**
```json
{
  "match_id": "12345",
  "odds": {
    "btts": {
      "yes": 1.95,
      "no": 1.90
    },
    "total_corners": {
      "over_9.5": 2.00,
      "under_9.5": 1.85
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

### Market-Specific Models

The system trains **4 separate models**, one for each market:

1. **Goals Model:** Predicts Over/Under 2.5 goals
2. **Cards Model:** Predicts Over/Under 3.5 cards
3. **Corners Model:** Predicts Over/Under 9.5 corners
4. **BTTS Model:** Predicts Both Teams To Score Yes/No

Each model is trained on relevant features:
- **Goals:** Team scoring averages, defensive records, form, head-to-head
- **Cards:** Team discipline records, referee strictness, match importance
- **Corners:** Team attacking styles, possession stats, corner averages
- **BTTS:** Both teams' scoring/conceding rates, clean sheet records

### Bet Type Implementation:

#### 1. Golden Bets
- **Confidence threshold:** 85%+ probability
- **Ensemble validation:** Multiple model agreement
- **Historical verification:** Backtest win rate validation
- **Selection criteria:** Highest confidence picks across all 4 markets regardless of odds
- **Daily limit:** 1-3 bets maximum to maintain quality

#### 2. Value Bets
- **Formula:** `Value = AI_Probability - Implied_Probability`
- **Threshold:** Minimum 10% positive value
- **Dynamic recalculation:** Updates with each odds refresh
- **Educational focus:** Detailed explanations of value concept
- **Risk communication:** Clear messaging about variance vs Golden Bets

#### 3. Smart Bets
- **Market analysis:** Evaluate all 4 markets per fixture
- **Selection:** Highest probability outcome across the 4 markets
- **Pure probability:** No odds consideration
- **Transparency:** Show alternative markets with probabilities

#### 4. Custom Bet Analysis
- **Same models:** Use trained models on user-selected bets from the 4 markets
- **Contextual explanations:** Generate reasoning based on input stats
- **Comparison:** Show how it compares to Smart Bet recommendation
- **Educational tone:** Explain why certain bets are stronger/weaker
- **Confidence context:** Set expectations about typical confidence levels

### Model Training Focus

- **Accuracy:** Minimize prediction error for each market
- **Calibration:** Ensure probabilities reflect true likelihood
- **Explainability:** Feature importance for transparent reasoning
- **Robustness:** Validate across different leagues and seasons
- **Market-specific optimization:** Each model optimized for its specific market

### Future Enhancement

Optionally, explore neural nets or transformer-based models for deeper pattern recognition after baseline validation.

---

## System Summary

### What You'll Receive:
- Keyed JSON daily with fixture, stats, odds for the 4 markets (batch)
- Single match + bet type requests from the 4 markets (custom analysis)
- Odds updates for value recalculation

### What You'll Return:
- Golden Bets: 1-3 daily high-confidence picks (85%+) from the 4 markets
- Value Bets: Top 3 daily positive EV opportunities from the 4 markets
- Smart Bets: Best bet per fixture across the 4 markets
- Custom Analysis: Probability + educational explanation for user-selected bets from the 4 markets

### The 4 Markets:
1. Goals: Over/Under 2.5
2. Cards: Over/Under 3.5
3. Corners: Over/Under 9.5
4. BTTS: Yes/No

### AI Models Focus:
Robust probabilistic predictions with transparent reasoning for each of the 4 markets

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
           (4 Market Models)      (4 Market Models)
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
- Your app sends all fixtures at 8 AM with stats and odds for the 4 markets
- AI processes and returns Golden/Value/Smart bets across the 4 markets
- Results cached for fast access
- Premium users browse Golden and Value Bets
- All users see Smart Bets per fixture

### Use Case 2: User Explores Custom Bet
- User clicks on "Manchester United vs Liverpool"
- User selects "Over 9.5 Corners" from the 4 available markets
- Your app calls `/api/v1/predictions/analyze`
- AI returns: "87% probability - Both teams average 10 corners combined..."
- AI notes: "This matches our Smart Bet recommendation for this fixture"
- User gains educational insight into bet quality

### Use Case 3: Odds Change
- Bookmaker updates corners odds from 1.90 to 2.10
- Your app sends odds update
- AI recalculates Value Bets for affected markets
- New value opportunities flagged
- Premium users notified of updated Value Bets

### Use Case 4: User Learning Journey
- Free user explores Smart Bets (sees AI quality across 4 markets)
- User tests own hypotheses via Custom Analysis
- User learns why certain markets are stronger for specific fixtures
- User upgrades to Premium for Golden/Value Bets
- User combines all features for informed betting strategy

---

## Key Differentiators

### Transparency
Every prediction includes detailed, human-readable explanations of reasoning and key factors.

### Education
Custom Analysis teaches users about betting dynamics across the 4 markets, not just providing picks.

### Tiered Intelligence
Four distinct features serve different user needs and betting strategies.

### Confidence Context
Clear communication about confidence levels and expected performance of each bet type.

### Dynamic Updates
Value Bets recalculate as odds change, capturing market inefficiencies in real-time.

### Focused Market Coverage
Laser-focused on 4 specific markets with dedicated models for each, ensuring high-quality predictions.

---

**This AI system delivers a complete betting intelligence ecosystem combining safety (Golden), profitability (Value), match-specific guidance (Smart), and interactive learning (Custom Analysis) across 4 carefully selected betting markets: Goals O/U 2.5, Cards O/U 3.5, Corners O/U 9.5, and BTTS Y/N.**