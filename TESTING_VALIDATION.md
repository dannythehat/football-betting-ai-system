# 🧪 Testing & Validation Guide

## Automated Testing System

All tests run automatically via GitHub Actions. No manual execution required.

## Test Coverage

### 1. Smart Bets AI ✅
**File:** `smart-bets-ai/predict.py`
**Tests:**
- Model loading and initialization
- Feature engineering pipeline
- Predictions for all 4 markets (Goals, Cards, Corners, BTTS)
- Confidence score calculation
- Best bet selection logic

**Expected Output:**
```
Match: Team A vs Team B
Best Bet: Goals Over 2.5 (Confidence: 87.3%)
Reasoning: [AI explanation]
```

### 2. Golden Bets AI ✅
**File:** `golden-bets-ai/test_filter.py`
**Tests:**
- High-confidence filtering (85%+ threshold)
- Daily limit enforcement (1-3 picks)
- Market diversity validation
- Reasoning generation

**Expected Output:**
```
Golden Bets (1-3 picks):
1. Match X - Goals Over 2.5 (89.2% confidence)
2. Match Y - BTTS Yes (86.7% confidence)
```

### 3. Value Bets AI ✅
**File:** `value-bets-ai/predict.py`
**Tests:**
- Expected value calculation
- Positive EV filtering
- Top 3 selection by EV
- Profit potential estimation

**Expected Output:**
```
Value Bets (Top 3):
1. Match A - Corners Over 9.5 (EV: +15.3%)
2. Match B - Cards Over 3.5 (EV: +12.1%)
3. Match C - Goals Over 2.5 (EV: +10.8%)
```

### 4. Custom Analysis ✅
**File:** `custom-analysis/test_analyzer.py`
**Tests:**
- User-selected fixture analysis
- Market-specific predictions
- Educational reasoning
- Confidence scoring

**Expected Output:**
```
Analysis: Team X vs Team Y - Goals Over 2.5
Prediction: YES (Confidence: 78.4%)
Reasoning: [Detailed AI explanation]
Key Factors: [Statistical insights]
```

## Validation Criteria

### Model Performance
- **Accuracy:** >70% on test data
- **Precision:** >75% for high-confidence predictions
- **Recall:** Balanced across all markets
- **F1 Score:** >0.72

### Business Logic
- Golden Bets: Only 85%+ confidence
- Value Bets: Only positive EV
- Smart Bets: Best single bet per match
- Custom Analysis: Educational and transparent

### API Response Time
- Predictions: <500ms
- Batch processing: <2s for 10 matches
- Cache hit: <50ms

## Test Execution Schedule

### Automatic Runs
- **Every 6 hours:** Full test suite
- **On every push:** Quick validation
- **Weekly:** Model retraining + full validation

### Manual Triggers
```bash
# Trigger full test suite
gh workflow run full-test-deploy.yml

# Trigger model training
gh workflow run train-models.yml
```

## Test Results Location

All test results are automatically committed to:
```
test-results/
├── TEST_REPORT.md          # Comprehensive report
├── smart-bets-test.txt     # Smart Bets output
├── golden-bets-test.txt    # Golden Bets output
├── value-bets-test.txt     # Value Bets output
└── custom-analysis-test.txt # Custom Analysis output
```

## Model Metrics

Training metrics saved to:
```
smart-bets-ai/models/metadata.json
```

Contains:
- Training date
- Model versions
- Performance metrics per market
- Feature importance
- Cross-validation scores

## Validation Checklist

Before considering system "production-ready":

- [x] Models train successfully
- [x] All 4 AI features work
- [x] Tests pass automatically
- [x] Results committed to repo
- [x] Documentation complete
- [x] Deployment automated
- [x] Monitoring in place

## Continuous Validation

The system validates itself:
1. **Every 6 hours:** Run all tests
2. **Commit results:** Automatic git push
3. **Track metrics:** Model performance logged
4. **Alert on failure:** GitHub Actions notifications

## How to Verify System Health

### 1. Check Latest Test Run
```bash
# View workflow status
https://github.com/dannythehat/football-betting-ai-system/actions

# Check test results
cat test-results/TEST_REPORT.md
```

### 2. Verify Model Metrics
```bash
cat smart-bets-ai/models/metadata.json
```

### 3. Test API Endpoints
```bash
# Health check
curl https://football-betting-ai-system-production.up.railway.app/health

# Test prediction
curl -X POST https://football-betting-ai-system-production.up.railway.app/api/v1/predictions/smart-bets \
  -H "Content-Type: application/json" \
  -d '{"matches": [...]}'
```

## Troubleshooting

### Tests Failing?
1. Check workflow logs
2. Review test output files
3. Verify model files exist
4. Check data format

### Models Not Training?
1. Verify training data exists
2. Check workflow permissions
3. Review training logs
4. Ensure dependencies installed

### API Not Responding?
1. Check Railway deployment status
2. Verify environment variables
3. Review application logs
4. Test database connectivity

## Success Metrics

System is considered **validated and working** when:
- ✅ All tests pass in latest run
- ✅ Models trained within last 7 days
- ✅ Test results committed to repo
- ✅ API responds to health checks
- ✅ Predictions return valid results

**Current Status: All criteria met ✅**
