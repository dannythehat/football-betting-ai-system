# Quick Start Guide

## Get Started Building the Football Betting AI System

This guide will help you understand the system and start development quickly.

---

## 📚 Essential Reading Order

1. **README.md** - System overview and architecture
2. **FEATURES.md** - Detailed feature descriptions and use cases
3. **SCOPE.md** - Technical specifications and data formats
4. **ROADMAP.md** - Implementation plan and milestones
5. **This guide** - Getting started with development

---

## 🎯 What You're Building

An **AI prediction engine** that analyzes football fixtures and generates four types of betting intelligence:

1. **Golden Bets** - 1-3 daily picks with 85%+ win probability (safety-focused)
2. **Value Bets** - Top 3 daily picks with positive expected value (profit-focused)
3. **Smart Bets** - Best single bet per fixture across all markets (match-specific)
4. **Custom Bet Analysis** - User-selected fixture + bet type analysis (interactive learning)

**Key Point:** This is the AI brain, not the full app. Your main app handles frontend, user management, and payments.

---

## 🏗️ System Architecture

```
Main App → [JSON] → AI Engine → [JSON] → Main App
                         ↓
            ┌────────────┴────────────┐
            │                         │
      Smart Bets AI          Golden Bets AI
            │                         │
            └────────────┬────────────┘
                         ↓
                   Value Bets AI
                         ↓
               Summary Generator
                         ↓
                     User API
                         ↓
                 [Cached Results]
```

---

## 🚀 Development Setup

### Prerequisites
- Python 3.9+
- PostgreSQL 13+
- Redis 6+
- Docker (optional but recommended)

### Initial Setup

```bash
# Clone repository
git clone https://github.com/dannythehat/football-betting-ai-system.git
cd football-betting-ai-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your database and Redis credentials

# Initialize database
python scripts/init_db.py

# Run tests
pytest

# Start development server
uvicorn user-api.main:app --reload
```

---

## 📊 Data Flow

### Input from Your App
```json
{
  "matches": [{
    "match_id": "12345",
    "datetime": "2025-11-15T14:00:00Z",
    "home_team": "Team A",
    "away_team": "Team B",
    "stats": {
      "home_goals_avg": 1.4,
      "away_goals_avg": 1.1,
      "home_corners_avg": 5.2,
      "away_corners_avg": 4.8
    },
    "odds": {
      "match_result": {
        "home_win": 1.95,
        "draw": 3.5,
        "away_win": 4.0
      }
    }
  }]
}
```

### Output to Your App
```json
{
  "predictions": [{
    "match_id": "12345",
    "golden_bets": [{
      "market_id": "match_result",
      "selection_id": "home_win",
      "probability": 0.87,
      "explanation": "Team A has won 9 of last 10 home games..."
    }],
    "value_bets": [{
      "market_id": "btts",
      "selection_id": "yes",
      "value": 0.14,
      "explanation": "AI probability (68%) exceeds implied (54%)..."
    }],
    "smart_bets": [{
      "market_id": "match_result",
      "selection_id": "home_win",
      "probability": 0.87,
      "explanation": "Highest probability across all markets..."
    }]
  }]
}
```

---

## 🔧 Key Components

### 1. Data Ingestion (`/data-ingestion`)
**Purpose:** Receive and validate fixture data from your app

**Key Files:**
- `validator.py` - Data validation schemas
- `ingestion.py` - Data processing logic
- `models.py` - Database models

**API Endpoint:** `POST /api/v1/data/ingest`

### 2. Smart Bets AI (`/smart-bets-ai`)
**Purpose:** Generate pure probabilistic predictions

**Key Files:**
- `model.py` - XGBoost/LightGBM model
- `features.py` - Feature engineering
- `predictor.py` - Prediction pipeline

**Output:** Probability for each bet type per fixture

### 3. Golden Bets AI (`/golden-bets-ai`)
**Purpose:** Filter high-confidence bets (85%+)

**Key Files:**
- `filter.py` - Confidence threshold logic
- `ensemble.py` - Multi-model agreement
- `selector.py` - Daily selection algorithm

**Output:** 1-3 highest confidence bets per day

### 4. Value Bets AI (`/value-bets-ai`)
**Purpose:** Calculate expected value vs market odds

**Key Files:**
- `calculator.py` - Value calculation logic
- `ranker.py` - Value ranking algorithm
- `updater.py` - Dynamic recalculation

**Formula:** `Value = AI_Probability - Implied_Probability`

### 5. Odds Updater (`/odds-updater`)
**Purpose:** Process odds updates and trigger recalculations

**Key Files:**
- `processor.py` - Odds processing logic
- `history.py` - Odds history tracking
- `trigger.py` - Recalculation triggers

**API Endpoint:** `POST /api/v1/odds/update`

### 6. Summary Generator (`/summary-generator`)
**Purpose:** Create human-readable explanations

**Key Files:**
- `generator.py` - Explanation generation
- `templates.py` - Explanation templates
- `formatter.py` - Response formatting

**Output:** Transparent, educational explanations

### 7. User API (`/user-api`)
**Purpose:** Serve predictions to your main app

**Key Endpoints:**
- `POST /api/v1/predictions/batch` - Daily batch predictions
- `POST /api/v1/predictions/analyze` - Custom bet analysis
- `POST /api/v1/odds/update` - Odds updates
- `GET /api/v1/health` - Health check

---

## 🧪 Testing Your Setup

### 1. Test Data Ingestion
```bash
curl -X POST http://localhost:8000/api/v1/data/ingest \
  -H "Content-Type: application/json" \
  -d @test_data/sample_fixtures.json
```

### 2. Test Batch Predictions
```bash
curl -X POST http://localhost:8000/api/v1/predictions/batch \
  -H "Content-Type: application/json" \
  -d @test_data/sample_fixtures.json
```

### 3. Test Custom Analysis
```bash
curl -X POST http://localhost:8000/api/v1/predictions/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "match_id": "12345",
    "bet_type": "over_2.5_goals",
    "market_id": "total_goals",
    "selection_id": "over_2.5"
  }'
```

### 4. Test Odds Update
```bash
curl -X POST http://localhost:8000/api/v1/odds/update \
  -H "Content-Type: application/json" \
  -d '{
    "match_id": "12345",
    "odds": {
      "btts": {"yes": 1.95, "no": 1.90}
    }
  }'
```

---

## 📈 Development Workflow

### Phase 1: Foundation (Week 1-2)
- Set up project structure
- Configure database and Redis
- Build data ingestion module
- Create basic API endpoints

### Phase 2: Core Model (Week 3-4)
- Collect historical data
- Train baseline XGBoost model
- Implement Smart Bets logic
- Test predictions

### Phase 3: Golden Bets (Week 5)
- Implement confidence filtering
- Build ensemble validation
- Create selection algorithm

### Phase 4: Value Bets (Week 6-7)
- Build odds processing
- Implement value calculation
- Add dynamic recalculation

### Phase 5: Explanations (Week 8)
- Create explanation templates
- Generate transparent reasoning
- Add educational context

### Phase 6: Custom Analysis (Week 9)
- Build on-demand analysis
- Add comparison logic
- Test user scenarios

### Phase 7: Optimization (Week 10)
- Implement batch processing
- Add comprehensive caching
- Optimize performance

### Phase 8: Testing (Week 11)
- Unit and integration tests
- Model validation
- Load testing

### Phase 9: Deployment (Week 12)
- Deploy to production
- Set up monitoring
- Create documentation

---

## 🎓 Key Concepts

### Probability vs Value
- **Probability:** Likelihood of outcome (Smart Bets, Golden Bets)
- **Value:** Expected profit vs market odds (Value Bets)
- **Golden Bets focus on winning** (high probability)
- **Value Bets focus on profit** (positive EV)

### Confidence Levels
- **Golden Bets:** 85%+ (very high confidence)
- **Smart Bets:** 60-80% (high confidence)
- **Value Bets:** 55-75% (variable, focus on EV)
- **Custom Analysis:** Variable (educational tool)

### Model Calibration
Ensure predicted probabilities match actual outcomes:
- 70% predictions should win ~70% of the time
- Use calibration plots to validate
- Adjust model if miscalibrated

---

## 🔍 Debugging Tips

### Model Not Predicting Well
- Check feature engineering
- Validate training data quality
- Review model hyperparameters
- Test on different leagues/seasons

### API Slow Response
- Check cache hit rates
- Profile database queries
- Optimize model inference
- Review connection pooling

### Explanations Not Clear
- Review explanation templates
- Add more context
- Test with real users
- Iterate based on feedback

### Value Bets Not Updating
- Check odds update flow
- Verify recalculation triggers
- Review cache invalidation
- Test with odds changes

---

## 📚 Additional Resources

### Documentation
- [README.md](README.md) - System overview
- [FEATURES.md](FEATURES.md) - Feature details
- [SCOPE.md](SCOPE.md) - Technical specs
- [ROADMAP.md](ROADMAP.md) - Implementation plan

### External Resources
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Redis Documentation](https://redis.io/documentation)

---

## 🤝 Getting Help

### Common Issues
1. **Database connection errors** - Check PostgreSQL is running and credentials are correct
2. **Redis connection errors** - Verify Redis is running on correct port
3. **Model training fails** - Ensure sufficient historical data
4. **API returns 500 errors** - Check logs for detailed error messages

### Development Tips
- Start with small dataset for faster iteration
- Use Docker for consistent environment
- Write tests as you develop
- Monitor logs during development
- Use API documentation (Swagger UI at `/docs`)

---

## ✅ Checklist: Ready to Start?

- [ ] Read README.md and understand system architecture
- [ ] Review FEATURES.md to understand all four bet types
- [ ] Check SCOPE.md for data format specifications
- [ ] Set up development environment (Python, PostgreSQL, Redis)
- [ ] Clone repository and install dependencies
- [ ] Run initial tests to verify setup
- [ ] Review ROADMAP.md for implementation phases
- [ ] Start with Phase 1: Foundation & Infrastructure

---

## 🚀 Next Steps

1. **Set up your development environment**
2. **Run the test suite to verify setup**
3. **Start with Phase 1 from ROADMAP.md**
4. **Build data ingestion module first**
5. **Iterate and test frequently**

---

**Ready to build the AI brain for your betting intelligence platform!**

For questions or issues, refer to the documentation or create an issue in the repository.
