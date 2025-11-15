# Custom Bet Analysis - 5 Minute Quickstart

## What is Custom Bet Analysis?

Custom Bet Analysis lets users test their own betting ideas and get AI-powered feedback. Unlike Smart Bets (which automatically picks the best bet), Custom Analysis evaluates whatever bet the user wants to explore.

**Key Features:**
- ✅ User chooses fixture + bet type
- ✅ AI analyzes that specific bet
- ✅ Educational feedback on bet quality
- ✅ Comparison with Smart Bet recommendation
- ✅ Transparent probability and reasoning

## Quick Start

### 1. Prerequisites

Ensure Smart Bets models are trained:
```bash
python smart-bets-ai/train.py
```

### 2. Test Custom Analysis

```bash
python custom-analysis/test_analyzer.py
```

Expected output:
```
============================================================
CUSTOM BET ANALYSIS TEST
============================================================

1. Initializing Custom Bet Analyzer...
✅ Analyzer initialized successfully

2. Testing 5 different bet selections...
   Match: Manchester United vs Liverpool

============================================================
Test Case 1: Over 2.5 Goals
============================================================

📊 Analysis Results:
   Market: Total Goals
   Selection: Over 2.5 Goals
   Probability: 72.5%
   Confidence: HIGH
   Verdict: Good

💡 Explanation:
   Both teams average 3.4 goals combined per match...

🔍 Comparison:
   Note: This confidence (72.5%) is lower than our Smart Bet...

✨ Smart Bet Alternative:
   Over 9.5 Corners (87.0%)

✅ Test passed
```

### 3. Use via API

Start the API server:
```bash
cd user-api
python main.py
```

Make a request:
```bash
curl -X POST http://localhost:8000/api/v1/predictions/custom-analysis \
  -H "Content-Type: application/json" \
  -d '{
    "match_data": {
      "match_id": "12345",
      "home_team": "Manchester United",
      "away_team": "Liverpool",
      "home_goals_avg": 1.8,
      "away_goals_avg": 1.6,
      "home_goals_conceded_avg": 1.0,
      "away_goals_conceded_avg": 1.2,
      "home_corners_avg": 6.2,
      "away_corners_avg": 5.8,
      "home_cards_avg": 2.3,
      "away_cards_avg": 2.1,
      "home_btts_rate": 0.65,
      "away_btts_rate": 0.60
    },
    "market_id": "total_goals",
    "selection_id": "over_2.5"
  }'
```

Response:
```json
{
  "success": true,
  "analysis": {
    "match_id": "12345",
    "home_team": "Manchester United",
    "away_team": "Liverpool",
    "user_selection": {
      "market_id": "total_goals",
      "market_name": "Total Goals",
      "selection_id": "over_2.5",
      "selection_name": "Over 2.5 Goals"
    },
    "analysis": {
      "probability": 0.725,
      "percentage": "72.5%",
      "confidence_level": "high",
      "verdict": "Good",
      "explanation": "Both teams average 3.4 goals combined per match...",
      "comparison": "Note: This confidence (72.5%) is lower than our Smart Bet..."
    },
    "smart_bet_alternative": {
      "market_id": "total_corners",
      "market_name": "Total Corners",
      "selection_id": "over_9.5",
      "selection_name": "Over 9.5 Corners",
      "probability": 0.87,
      "percentage": "87.0%"
    }
  }
}
```

## Supported Markets

### 1. Total Goals
```json
{
  "market_id": "total_goals",
  "selection_id": "over_2.5"  // or "under_2.5"
}
```

### 2. Total Cards
```json
{
  "market_id": "total_cards",
  "selection_id": "over_3.5"  // or "under_3.5"
}
```

### 3. Total Corners
```json
{
  "market_id": "total_corners",
  "selection_id": "over_9.5"  // or "under_9.5"
}
```

### 4. Both Teams To Score
```json
{
  "market_id": "btts",
  "selection_id": "yes"  // or "no"
}
```

## Confidence Levels

| Probability | Level | Verdict | Meaning |
|-------------|-------|---------|---------|
| 80%+ | Very High | Excellent | Strong bet |
| 70-79% | High | Good | Solid bet |
| 60-69% | Moderate | Moderate | Reasonable bet |
| 50-59% | Low | Weak | Weak bet |
| <50% | Very Low | Not Recommended | Avoid |

## Understanding the Output

### When Your Selection Matches Smart Bet
```
✅ Your selection matches our Smart Bet recommendation! 
This is the highest probability option we identified for this fixture.
```
**Meaning:** You picked the same bet the AI would recommend. High confidence!

### When Your Selection Differs
```
Note: This confidence (67.0%) is lower than our Smart Bet 
recommendation for this fixture (87.0% for Over 9.5 Corners).
```
**Meaning:** Your bet is viable but not the AI's top pick. Consider the alternative.

### Smart Bet Alternative
If your selection differs from the Smart Bet, you'll see:
```json
"smart_bet_alternative": {
  "market_name": "Total Corners",
  "selection_name": "Over 9.5 Corners",
  "probability": 0.87,
  "percentage": "87.0%"
}
```
**Meaning:** This is what the AI would recommend instead.

## Use Cases

### 1. Test Your Hypothesis
"I think this match will have over 2.5 goals. Let me check what the AI thinks."

### 2. Learn About Markets
"Why does the AI prefer corners over goals for this fixture?"

### 3. Validate Your Research
"I've done my analysis. Does the AI agree with my pick?"

### 4. Explore Alternatives
"I want BTTS Yes, but what's the AI's top recommendation?"

## Key Differences from Smart Bets

| Feature | Smart Bets | Custom Analysis |
|---------|------------|-----------------|
| **Who Chooses** | AI picks best | User picks |
| **Markets Analyzed** | All 4 markets | User's choice only |
| **Typical Confidence** | 60-80% | Variable (often lower) |
| **Purpose** | Recommendation | Education |
| **When to Use** | Want best bet | Want to test idea |

## Python Usage

```python
from custom_analysis import CustomBetAnalyzer

# Initialize
analyzer = CustomBetAnalyzer()

# Analyze a bet
result = analyzer.analyze_custom_bet(
    match_data={
        "match_id": "12345",
        "home_team": "Team A",
        "away_team": "Team B",
        "home_goals_avg": 1.5,
        "away_goals_avg": 1.2,
        "home_goals_conceded_avg": 1.0,
        "away_goals_conceded_avg": 1.1,
        "home_corners_avg": 5.0,
        "away_corners_avg": 4.5,
        "home_cards_avg": 2.0,
        "away_cards_avg": 1.8,
        "home_btts_rate": 0.5,
        "away_btts_rate": 0.45
    },
    market_id="total_goals",
    selection_id="over_2.5"
)

# Access results
print(f"Probability: {result['analysis']['percentage']}")
print(f"Verdict: {result['analysis']['verdict']}")
print(f"Explanation: {result['analysis']['explanation']}")
```

## Tips for Best Results

### ✅ Do:
- Use Custom Analysis to learn about betting dynamics
- Compare your picks with Smart Bet recommendations
- Explore different markets for the same fixture
- Use it as an educational tool

### ❌ Don't:
- Expect same confidence as Smart Bets (it analyzes only your choice)
- Ignore the Smart Bet alternative if confidence is low
- Use it as the only decision factor
- Forget that Smart Bets analyze ALL 4 markets

## Troubleshooting

### Error: "Models not loaded"
```bash
# Train Smart Bets models first
python smart-bets-ai/train.py
```

### Error: "Unsupported market"
Check that `market_id` is one of:
- `total_goals`
- `total_cards`
- `total_corners`
- `btts`

### Error: "Invalid selection"
Check that `selection_id` matches the market:
- Goals: `over_2.5` or `under_2.5`
- Cards: `over_3.5` or `under_3.5`
- Corners: `over_9.5` or `under_9.5`
- BTTS: `yes` or `no`

## Next Steps

1. ✅ Test with different fixtures
2. ✅ Compare multiple markets for same match
3. ✅ Integrate with your frontend
4. ✅ Track which user selections match Smart Bets
5. ✅ Use insights to educate users

## Documentation

- Full docs: [custom-analysis/README.md](custom-analysis/README.md)
- API reference: [user-api/README.md](user-api/README.md)
- Smart Bets: [SMART_BETS_QUICKSTART.md](SMART_BETS_QUICKSTART.md)

---

**🎉 Custom Bet Analysis is ready! Start testing your betting ideas with AI-powered feedback.**
