# Test Data Generation Guide

## Overview
This guide explains the structure and generation of the complete test dataset for the Football Betting AI System.

## Complete Dataset Structure

### 1. Teams Database
**File:** `teams.json`
- **Status:** ✅ COMPLETE
- **Records:** 400 teams
- **Leagues:** 20 leagues across Europe
- **Distribution:**
  - Top tier: 80 teams (20%)
  - Mid tier: 120 teams (30%)
  - Lower tier: 200 teams (50%)

### 2. Historical Matches (Training Data)
**Target:** 3,600 matches over 3 years (2022-2025)
- **2022:** 1,200 matches
- **2023:** 1,200 matches
- **2024-2025:** 1,200 matches

**Current Status:** Sample of 5 matches provided in `historical_matches_sample.json`

**Full Dataset Requirements:**
Each match must include:
- Match metadata (ID, datetime, teams, league, season)
- Team statistics at time of match (goals avg, form, etc.)
- Complete odds for all betting markets
- Actual results (goals, corners, cards)
- Market outcomes (boolean results for each bet type)

### 3. Upcoming Fixtures (Prediction Data)
**Target:** 200 upcoming matches
- Matches scheduled for next 30 days
- Same structure as historical but without results
- Used for testing prediction generation

### 4. Team Statistics Database
**Target:** 400 team records with seasonal stats
- Overall performance metrics
- Home/away splits
- Form indicators
- Historical trends

## Data Generation Approach

Since you cannot run Python scripts locally, here are your options:

### Option A: Use Online JSON Generator Tools
1. **JSONGenerator.com** or similar tools
2. Use the sample structure from `historical_matches_sample.json`
3. Generate batches of 100-200 matches at a time
4. Combine into single file

### Option B: AI-Assisted Generation
1. Use this conversation to generate batches
2. I can create 50-100 matches per file
3. You combine them manually
4. **Limitation:** May take multiple sessions

### Option C: Minimal Viable Dataset
For initial development and testing:
- **300 historical matches** (100 per year) - Enough for basic model training
- **50 upcoming fixtures** - For prediction testing
- **400 teams** - Already complete ✅

### Option D: Use Existing Football Data APIs (Recommended)
Once you're ready to purchase real data:
- **API-Football** (RapidAPI)
- **Football-Data.org**
- **TheSportsDB**
- **Sportmonks**

These provide:
- Real historical match data
- Actual odds from bookmakers
- Team statistics
- Live fixtures

## Data Realism Requirements

### Team Statistics (at match time)
```json
{
  "home_goals_avg": 1.5-2.5,      // Realistic range
  "away_goals_avg": 1.0-2.0,      // Away teams score less
  "home_goals_conceded_avg": 0.7-1.5,
  "away_goals_conceded_avg": 1.0-1.8,
  "home_corners_avg": 5.0-8.0,
  "away_corners_avg": 4.0-6.5,
  "home_cards_avg": 1.5-2.5,
  "away_cards_avg": 1.8-2.8,
  "home_btts_rate": 0.40-0.65,
  "away_btts_rate": 0.45-0.70,
  "home_form": "WWDLW",           // Last 5 matches
  "away_form": "LWDWL"
}
```

### Odds Ranges (Decimal)
```json
{
  "home_win": 1.30-3.50,          // Varies by team strength
  "draw": 3.00-4.50,
  "away_win": 2.00-8.00,
  "over_2.5": 1.60-2.40,
  "under_2.5": 1.60-2.40,
  "btts_yes": 1.60-2.20,
  "btts_no": 1.70-2.30
}
```

### Match Results Distribution
- **Home wins:** ~45%
- **Draws:** ~25%
- **Away wins:** ~30%
- **Total goals:** 0-6 (avg 2.7)
- **BTTS:** ~55% of matches
- **Over 2.5 goals:** ~50% of matches

## Recommended Approach for Your Project

### Phase 1: Minimal Dataset (Start Development)
✅ **400 teams** - Complete
🔄 **300 historical matches** - Generate next
🔄 **50 upcoming fixtures** - Generate after historical
🔄 **400 team statistics** - Generate from match data

**Timeline:** Can be completed in this conversation

### Phase 2: Expanded Dataset (Model Training)
- **1,200 historical matches** - Sufficient for initial model
- **100 upcoming fixtures** - Better prediction testing
- Use AI-assisted generation or online tools

### Phase 3: Production Dataset (Real API)
- Purchase football data API subscription
- Ingest real historical data (3+ years)
- Use real-time odds feeds
- Continuous data updates

## Next Steps

**Immediate Action Items:**
1. ✅ Teams database complete (400 teams)
2. Generate 300 historical matches (I can help with batches)
3. Generate 50 upcoming fixtures
4. Create team statistics from match data
5. Set up database and import data

**Would you like me to:**
- A) Generate the next batch of 50 historical matches?
- B) Create the upcoming fixtures file?
- C) Generate team statistics database?
- D) Create SQL import scripts?

Let me know which you'd like to tackle next!
