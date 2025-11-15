# Advanced Feature Engineering Module

## Overview

This module transforms raw match data into **100+ intelligent features** for machine learning models. It represents a complete redesign from the basic 30-feature system to a sophisticated, multi-layered feature engineering pipeline.

## Architecture

### Feature Modules

1. **Core Statistics Engine** (`core_stats.py`)
   - **30+ features**: Rolling averages, weighted form, variance metrics, streaks
   - **Focus**: Statistical patterns and consistency indicators
   - **Examples**: `home_goals_last_5`, `home_weighted_form`, `home_win_streak`

2. **Head-to-Head Analyzer** (`head_to_head.py`)
   - **18+ features**: Historical matchup analysis and trends
   - **Focus**: Team-specific patterns and venue effects
   - **Examples**: `h2h_avg_total_goals`, `h2h_dominance_score`, `h2h_home_venue_advantage`

3. **Momentum Analyzer** (`momentum.py`)
   - **25+ features**: Form momentum, confidence, pressure, fatigue
   - **Focus**: Psychological factors and recent performance trends
   - **Examples**: `home_momentum_score`, `home_confidence_score`, `home_fatigue_risk`

4. **Market-Specific Features** (`market_specific.py`)
   - **60+ features**: Specialized features for each of the 4 betting markets
   - **Focus**: Market-tailored predictive indicators
   - **Markets**:
     - Goals O/U 2.5: xG, shots, conversion rates
     - Corners O/U 9.5: Possession, attacking style
     - Cards O/U 3.5: Discipline, fouls, referee stats
     - BTTS Y/N: Clean sheets, scoring consistency

5. **Feature Pipeline** (`feature_pipeline.py`)
   - **Orchestrator**: Coordinates all modules
   - **Validation**: Quality checks and error handling
   - **Batch Processing**: Efficient multi-match transformation

## Feature Count Breakdown

| Module | Feature Count | Description |
|--------|--------------|-------------|
| Core Statistics | 30+ | Rolling averages, variance, streaks, venue splits |
| Head-to-Head | 18+ | Historical matchup patterns and trends |
| Momentum | 25+ | Form, confidence, pressure, fatigue |
| Market-Specific | 60+ | Goals (16), Corners (14), Cards (15), BTTS (15) |
| **TOTAL** | **133+** | **Complete intelligent feature set** |

## Usage

### Basic Usage

```python
from features import FeaturePipeline

# Initialize pipeline
pipeline = FeaturePipeline()

# Transform single match
match_data = {
    'match_id': '12345',
    'home_team': 'Team A',
    'away_team': 'Team B',
    'home_goals_avg': 1.5,
    'away_goals_avg': 1.2,
    # ... more data
}

features = pipeline.transform(match_data)
print(f"Generated {len(features)} features")
```

### Batch Processing

```python
# Transform multiple matches
matches_data = [match1, match2, match3, ...]
features_df = pipeline.transform_batch(matches_data)

print(features_df.shape)  # (n_matches, n_features)
```

### Feature Inspection

```python
# Get all feature names
feature_names = pipeline.get_feature_names()
print(f"Total features: {len(feature_names)}")

# Get features grouped by module
feature_groups = pipeline.get_feature_importance_groups()
for module, features in feature_groups.items():
    print(f"{module}: {len(features)} features")

# Get pipeline summary
summary = pipeline.get_summary_stats()
print(summary)
```

## Data Requirements

### Required Fields

**Basic Match Info:**
- `match_id`, `home_team`, `away_team`
- `home_goals_avg`, `away_goals_avg`
- `home_goals_conceded_avg`, `away_goals_conceded_avg`
- `home_corners_avg`, `away_corners_avg`
- `home_cards_avg`, `away_cards_avg`
- `home_btts_rate`, `away_btts_rate`

### Optional Fields (Enhanced Features)

**Historical Data:**
- `home_goals_history`: List of recent goals scored
- `home_results_last_10`: List of recent results (W/D/L)
- `h2h_history`: List of head-to-head matches

**Advanced Stats:**
- `home_xg_last_5`: Expected goals (xG) data
- `home_shots_per_game`: Shot statistics
- `home_possession_pct`: Possession percentage
- `referee_cards_avg`: Referee statistics

**Momentum Data:**
- `home_goals_last_3`: Recent goal counts
- `home_days_since_last_match`: Rest days
- `home_matches_last_7_days`: Fixture congestion

## Feature Categories

### 1. Statistical Features
- Rolling averages (5, 10 game windows)
- Exponentially weighted form
- Variance and consistency metrics
- Streak detection (wins, scoring, clean sheets)

### 2. Contextual Features
- Home/away venue splits
- Time-based patterns (first/second half)
- Head-to-head historical analysis
- Venue-specific performance

### 3. Psychological Features
- Momentum scores
- Confidence indicators
- Pressure situations
- Bounce-back potential

### 4. Physical Features
- Fatigue risk indicators
- Fixture congestion
- Travel distance
- Rest days

### 5. Market-Specific Features
- **Goals**: xG, shots, conversion rates, attacking intensity
- **Corners**: Possession, attacking style, corner patterns
- **Cards**: Discipline, fouls, referee strictness, rivalry
- **BTTS**: Clean sheets, scoring consistency, defensive vulnerability

## Feature Engineering Principles

### 1. Domain Knowledge
Every feature is grounded in football analytics principles and betting market dynamics.

### 2. Temporal Awareness
Features capture recent trends while respecting historical context through weighted averaging.

### 3. Market Specialization
Each betting market gets tailored features that capture its unique characteristics.

### 4. Robustness
Default values and fallback logic ensure features are generated even with incomplete data.

### 5. Interpretability
Feature names are descriptive, and features are grouped by module for easy interpretation.

## Performance Considerations

### Computational Efficiency
- **Single match**: <10ms
- **Batch (100 matches)**: <500ms
- **Memory**: ~1MB per 1000 matches

### Data Quality
- Handles missing data gracefully with defaults
- Validates features for NaN and infinite values
- Logs warnings for data quality issues

## Integration with ML Models

### Training Pipeline

```python
from features import FeaturePipeline
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Initialize pipeline
pipeline = FeaturePipeline()

# Transform training data
features_df = pipeline.transform_batch(training_matches)
feature_names = pipeline.get_feature_names()

# Prepare for model training
X = features_df[feature_names]
y = training_labels

# Train model
model = XGBClassifier()
model.fit(X, y)
```

### Prediction Pipeline

```python
# Transform new match
new_match_features = pipeline.transform(new_match_data)

# Convert to DataFrame for prediction
X_new = pd.DataFrame([new_match_features])[feature_names]

# Predict
probability = model.predict_proba(X_new)[0, 1]
```

## Future Enhancements

### Phase 2 (Planned)
- Player-level intelligence (injuries, suspensions)
- Environmental factors (weather, referee)
- Real-time data integration

### Phase 3 (Planned)
- Deep learning feature extraction
- Automated feature selection
- Feature interaction detection

## Testing

```bash
# Run feature tests
python -m pytest tests/test_features.py

# Test individual modules
python -m pytest tests/test_core_stats.py
python -m pytest tests/test_h2h.py
python -m pytest tests/test_momentum.py
python -m pytest tests/test_market_specific.py
```

## Changelog

### Version 2.0.0 (November 2025)
- Complete redesign from 30 to 133+ features
- Added 4 specialized feature modules
- Implemented feature pipeline orchestrator
- Added comprehensive validation and error handling
- Improved computational efficiency

### Version 1.0.0 (Original)
- Basic 30-feature system
- Simple averages and differentials

## Contributing

When adding new features:
1. Add to appropriate module (or create new module)
2. Update `get_feature_names()` method
3. Add tests for new features
4. Update this README
5. Document feature meaning and calculation

## License

Part of the Football Betting AI System
