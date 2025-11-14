# Test Data Documentation

## Current Status

### ✅ Completed
1. **teams.json** - 400 teams across 20 European leagues
2. **schema.sql** - Complete PostgreSQL database schema
3. **historical_matches_sample.json** - 5 sample matches showing full structure
4. **DATA_GENERATION_GUIDE.md** - Comprehensive guide for generating full dataset

### 🔄 In Progress / To Do
1. **Historical matches** - Need 300-3600 matches (currently have 5 samples)
2. **Upcoming fixtures** - Need 50-200 fixtures
3. **Team statistics** - Need 400 team stat records

## What Your Main App API Will Provide

Your main app will send data to this AI prediction engine in this format:

### 1. Historical Match Data (For Training)
**Purpose:** Train AI models on past results
**Time Range:** 3 years back (2022-2025)
**Volume:** 1,200-3,600 matches

**Data per match:**
- Match details (teams, datetime, league)
- Team statistics at time of match
- Historical odds from bookmakers
- Actual results and outcomes

### 2. Upcoming Fixtures (For Predictions)
**Purpose:** Generate predictions for future matches
**Time Range:** Next 7-30 days
**Volume:** 50-200 matches

**Data per match:**
- Match details (teams, datetime, league)
- Current team statistics
- Current bookmaker odds
- No results (to be predicted)

### 3. Team Statistics
**Purpose:** Provide context for predictions
**Updates:** Weekly or after each match

**Data per team:**
- Overall stats (goals, wins, form)
- Home/away splits
- Recent form (last 5 matches)
- Seasonal trends

## Data Structure

### Teams Database (400 teams)
```json
{
  "team_id": 1,
  "team_name": "Manchester United",
  "league": "Premier League",
  "tier": "top"
}
```

### Historical Match Format
```json
{
  "match_id": "HM_2022_001",
  "match_datetime": "2022-01-15T15:00:00Z",
  "home_team": "Manchester United",
  "away_team": "Aston Villa",
  "team_stats_at_match_time": {
    "home_goals_avg": 1.8,
    "away_goals_avg": 1.2,
    "home_form": "WWDWL",
    "away_form": "LWDLW"
  },
  "odds": {
    "home_win": 1.65,
    "draw": 3.80,
    "away_win": 5.50,
    "over_2_5": 1.95,
    "btts_yes": 1.75
  },
  "result": {
    "home_goals": 2,
    "away_goals": 1,
    "result": "home_win",
    "btts": true,
    "over_2_5": true
  }
}
```

## Betting Markets Covered

All test data includes odds and outcomes for:

1. **Match Result (1X2)** - Home Win, Draw, Away Win
2. **Total Goals** - Over/Under 0.5, 1.5, 2.5, 3.5, 4.5
3. **Both Teams To Score (BTTS)** - Yes, No
4. **Double Chance** - Home or Draw, Away or Draw, Home or Away
5. **Corners** - Over/Under 8.5, 9.5, 10.5
6. **Cards** - Over/Under 3.5, 4.5

## Database Schema

The `schema.sql` file provides complete PostgreSQL schema with:

- **teams** - Team information
- **team_statistics** - Aggregated team stats per season
- **matches** - All matches with team stats snapshot
- **match_results** - Results for completed matches
- **match_odds** - Current and historical odds
- **predictions** - AI model outputs

**Views:**
- `upcoming_matches_with_odds` - Fixtures ready for prediction
- `historical_matches_with_results` - Training data with outcomes

## Data Realism

### Statistical Distributions
- **Home advantage:** 55% home win rate
- **Goals per match:** 2.7 average (0-6 range)
- **BTTS rate:** 55% of matches
- **Over 2.5 goals:** 50% of matches
- **Corners:** 8-12 average per match
- **Cards:** 3-5 average per match

### Team Tiers
- **Top tier (80 teams):** Strong home/away, consistent form
- **Mid tier (120 teams):** Balanced performance
- **Lower tier (200 teams):** Inconsistent, home advantage matters more

## Usage

### For Development (Phase 1)
```bash
# 1. Create database
psql -U postgres -c "CREATE DATABASE football_betting_ai;"

# 2. Load schema
psql -U postgres -d football_betting_ai -f test-data/schema.sql

# 3. Import teams (manual or script)
# 4. Import historical matches
# 5. Import upcoming fixtures
```

### For Model Training
```python
# Load historical data
import json
with open('test-data/historical_matches_sample.json') as f:
    data = json.load(f)

# Extract features and labels
# Train XGBoost/LightGBM models
# Validate on test set
```

### For Prediction Testing
```python
# Load upcoming fixtures
# Generate predictions using trained models
# Compare output format with expected structure
```

## Next Steps

### Option 1: Generate More Test Data (Recommended for Now)
I can generate additional batches of:
- 50-100 historical matches at a time
- 50 upcoming fixtures
- 400 team statistics

**Pros:** Free, immediate, sufficient for development
**Cons:** Time-consuming, not real data

### Option 2: Purchase Football Data API (Recommended for Production)
APIs like API-Football, Football-Data.org provide:
- Real historical match data (3+ years)
- Actual bookmaker odds
- Live fixtures and updates
- Team statistics

**Pros:** Real data, continuous updates, production-ready
**Cons:** Costs money, requires integration

## Files in This Directory

1. **README.md** (this file) - Overview and documentation
2. **schema.sql** - PostgreSQL database schema
3. **teams.json** - 400 teams database ✅
4. **historical_matches_sample.json** - 5 sample matches with full structure
5. **DATA_GENERATION_GUIDE.md** - Guide for generating full dataset

## Questions?

**Q: How much data do I need to start development?**
A: Minimum 300 historical matches + 50 upcoming fixtures. This is enough to build and test the data ingestion module and basic prediction pipeline.

**Q: Can I use this test data for production?**
A: No. This is for development and testing only. Production requires real data from a football data API.

**Q: How do I generate more test data?**
A: See `DATA_GENERATION_GUIDE.md` for options. I can help generate batches in this conversation.

**Q: What's the recommended approach?**
A: 
1. Use test data for Phase 1 development (data ingestion, basic models)
2. Purchase real API for Phase 2+ (model training, production)
3. This gets you started quickly without upfront API costs

---

**Ready to generate more test data?** Let me know and I'll create the next batch!
