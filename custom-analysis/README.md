# Custom Bet Analysis Module

## Overview
The Custom Bet Analysis module provides on-demand analysis for user-selected fixtures and bet types. It empowers users to test their own betting hypotheses and receive AI-powered feedback with educational context.

## Purpose
- **User Empowerment**: Let users analyze their own bet ideas
- **Education**: Teach users about betting dynamics across the 4 markets
- **Transparency**: Show why certain bets are stronger/weaker
- **Comparison**: Compare user selections with AI's top recommendations

## Features

### ✅ Supported Markets
1. **Total Goals**: Over/Under 2.5
2. **Total Cards**: Over/Under 3.5
3. **Total Corners**: Over/Under 9.5
4. **BTTS**: Yes/No

### ✅ Analysis Components
- **Probability Calculation**: AI-predicted probability for user's selection
- **Confidence Level**: Very High (80%+), High (70-79%), Moderate (60-69%), Low (50-59%), Very Low (<50%)
- **Verdict**: Excellent, Good, Moderate, Weak, Not Recommended
- **Explanation**: Market-specific reasoning based on team stats
- **Comparison**: How user's selection compares to Smart Bet recommendation
- **Alternative**: Smart Bet suggestion if different from user's choice

## How It Works

### 1. User Input
```python
{
    "match_data": {
        "match_id": "12345",
        "home_team": "Team A",
        "away_team": "Team B",
        "home_goals_avg": 1.4,
        "away_goals_avg": 1.1,
        # ... other stats
    },
    "market_id": "total_goals",
    "selection_id": "over_2.5"
}
```

### 2. Analysis Process
1. Validate market and selection
2. Get Smart Bet prediction for comparison
3. Calculate probability for user's selection
4. Determine confidence level and verdict
5. Generate market-specific explanation
6. Compare with Smart Bet recommendation
7. Provide educational context

### 3. Output
```json
{
    "match_id": "12345",
    "home_team": "Team A",
    "away_team": "Team B",
    "user_selection": {
        "market_id": "total_goals",
        "market_name": "Total Goals",
        "selection_id": "over_2.5",
        "selection_name": "Over 2.5 Goals"
    },
    "analysis": {
        "probability": 0.67,
        "percentage": "67.0%",
        "confidence_level": "moderate",
        "verdict": "Moderate",
        "explanation": "Both teams average 2.5 goals combined per match...",
        "comparison": "Note: This confidence (67.0%) is lower than our Smart Bet..."
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
```

## Confidence Levels

| Level | Probability | Verdict | Meaning |
|-------|-------------|---------|---------|
| Very High | 80%+ | Excellent | Strong bet, high confidence |
| High | 70-79% | Good | Solid bet, good confidence |
| Moderate | 60-69% | Moderate | Reasonable bet, moderate confidence |
| Low | 50-59% | Weak | Weak bet, low confidence |
| Very Low | <50% | Not Recommended | Poor bet, avoid |

## Educational Approach

### When User's Selection Matches Smart Bet
```
✅ Your selection matches our Smart Bet recommendation! 
This is the highest probability option we identified for this fixture.
```

### When User's Selection Differs
```
Note: This confidence (67.0%) is lower than our Smart Bet 
recommendation for this fixture (87.0% for Over 9.5 Corners). 
Smart Bets analyze all 4 markets to find the highest probability option.
```

### When No Smart Bet Available
```
We don't have a Smart Bet recommendation for this fixture to compare against.
```

## Usage

### Python
```python
from custom_analysis import CustomBetAnalyzer

# Initialize analyzer
analyzer = CustomBetAnalyzer()

# Analyze custom bet
result = analyzer.analyze_custom_bet(
    match_data={
        "match_id": "12345",
        "home_team": "Team A",
        "away_team": "Team B",
        "home_goals_avg": 1.4,
        "away_goals_avg": 1.1,
        "home_goals_conceded_avg": 0.8,
        "away_goals_conceded_avg": 1.6,
        "home_corners_avg": 5.2,
        "away_corners_avg": 4.8,
        "home_cards_avg": 2.1,
        "away_cards_avg": 1.8,
        "home_btts_rate": 0.6,
        "away_btts_rate": 0.5
    },
    market_id="total_goals",
    selection_id="over_2.5"
)

print(result)
```

### API Endpoint
```bash
curl -X POST http://localhost:8000/api/v1/predictions/custom-analysis \
  -H "Content-Type: application/json" \
  -d '{
    "match_data": {...},
    "market_id": "total_goals",
    "selection_id": "over_2.5"
  }'
```

## Key Differences from Smart Bets

| Feature | Smart Bets | Custom Analysis |
|---------|------------|-----------------|
| **Selection** | AI chooses best bet | User chooses bet |
| **Markets Analyzed** | All 4 markets | User's selected market only |
| **Typical Confidence** | 60-80% | Variable (often lower) |
| **Purpose** | Recommendation | Education & validation |
| **Output** | Best single bet | Analysis of user's choice |

## Important Notes

### Lower Confidence Expected
Custom Analysis typically yields lower confidence than Smart Bets because:
- Smart Bets analyze ALL 4 markets and pick the best
- Custom Analysis evaluates only the user's chosen market
- Exception: When user's choice matches Smart Bet recommendation

### Educational Focus
This feature is designed to:
- Help users understand betting dynamics
- Show why certain markets are stronger for specific fixtures
- Build trust through transparency
- Encourage informed decision-making

## Dependencies
- `smart-bets-ai`: Uses Smart Bets predictor for probability calculations
- Trained models required (run `smart-bets-ai/train.py` first)

## Status
✅ **Phase 5 - Complete**

## Next Steps
- Integration with user-api
- Add caching for frequently analyzed matches
- Track user analysis patterns
- Enhance explanations based on user feedback
