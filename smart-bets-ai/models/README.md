## Deep Learning Ensemble Models

Complete multi-model architecture for intelligent football betting predictions.

---

## Architecture Overview

The ensemble combines **4 different model types** to create robust, high-accuracy predictions:

```
┌─────────────────────────────────────────────────────────────┐
│                    ENSEMBLE PREDICTOR                        │
│                                                              │
│  ┌──────────────┐  ┌──────────┐  ┌─────────────┐  ┌──────┐ │
│  │  Gradient    │  │  LSTM    │  │ Transformer │  │ DNN  │ │
│  │  Boosting    │  │  Network │  │  Network    │  │      │ │
│  │  (3 models)  │  │          │  │             │  │      │ │
│  └──────┬───────┘  └────┬─────┘  └──────┬──────┘  └───┬──┘ │
│         │               │                │              │    │
│         └───────────────┴────────────────┴──────────────┘    │
│                              │                               │
│                    ┌─────────▼─────────┐                     │
│                    │  Ensemble Voting  │                     │
│                    │  - Weighted avg   │                     │
│                    │  - Confidence     │                     │
│                    │  - Agreement      │                     │
│                    └─────────┬─────────┘                     │
│                              │                               │
│                    ┌─────────▼─────────┐                     │
│                    │  Final Prediction │                     │
│                    │  + Confidence     │                     │
│                    └───────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Model Components

### 1. Gradient Boosting Ensemble

**File**: `gradient_boosting.py`

**Models**:
- **XGBoost**: Fast, accurate, proven performance
- **LightGBM**: Speed optimized, handles large datasets
- **CatBoost**: Superior categorical feature handling

**Strengths**:
- Excellent baseline performance
- Fast training and inference
- Feature importance analysis
- Robust to overfitting

**Configuration**:
```python
{
    'n_estimators': 300,
    'max_depth': 8,
    'learning_rate': 0.03,
    'early_stopping_rounds': 50
}
```

---

### 2. LSTM Network

**File**: `lstm_model.py`

**Architecture**:
- Bidirectional LSTM layers
- Sequence length: 10 matches
- Hidden size: 128 units
- 2 layers with dropout

**Strengths**:
- Captures time-series patterns
- Models momentum and form
- Sequential match analysis
- Temporal dependencies

**Use Case**: Best for detecting trends and momentum shifts

---

### 3. Transformer Network

**File**: `transformer_model.py`

**Architecture**:
- Multi-head attention (8 heads)
- 4 encoder layers
- Positional encoding
- d_model: 128 dimensions

**Strengths**:
- Complex relationship modeling
- Attention mechanisms
- Parallel processing
- Long-range dependencies

**Use Case**: Best for understanding complex team interactions

---

### 4. Deep Neural Network

**File**: `neural_net.py`

**Architecture**:
- 4 hidden layers: [256, 128, 64, 32]
- Batch normalization
- ReLU activation
- Dropout: 0.3

**Strengths**:
- Non-linear feature interactions
- Deep feature learning
- Flexible architecture
- Fast inference

**Use Case**: Best for capturing complex patterns in tabular data

---

## Ensemble Voting System

**File**: `voting.py`

### Weighted Voting

Models are weighted based on their strengths:

```python
{
    'gradient_boosting': 0.35,  # Highest - proven baseline
    'lstm': 0.25,               # Time-series expertise
    'transformer': 0.20,        # Complex relationships
    'dnn': 0.20                 # Feature interactions
}
```

### Confidence Calculation

```python
confidence = probability_distance × model_agreement

# Boosts:
- High agreement (90%+): +10% confidence
- Low agreement (<70%): -5% confidence penalty
```

### Agreement Analysis

- **Standard Deviation**: Measures prediction variance
- **Consensus**: Counts models agreeing on outcome
- **Unanimous**: All models agree (highest confidence)

---

## Training Pipeline

**File**: `train_ensemble.py`

### Complete Workflow

```python
from smart_bets_ai.train_ensemble import EnsembleTrainer

# Initialize
trainer = EnsembleTrainer('data/historical_matches.json')

# Train all markets
results = trainer.train_all_markets()

# Results include:
# - Training metrics per model
# - Validation performance
# - Test set evaluation
# - Model comparison
```

### Data Splits

- **Training**: 80%
- **Validation**: 10%
- **Test**: 10%

### Training Features

- Early stopping (prevents overfitting)
- Learning rate scheduling
- Gradient clipping
- Cross-validation support

---

## Usage Examples

### Basic Prediction

```python
from smart_bets_ai.models import EnsemblePredictor
import pandas as pd

# Initialize
ensemble = EnsemblePredictor('goals', input_size=133)

# Load trained model
ensemble.load('models/saved/ensemble')

# Predict
X = pd.DataFrame([...])  # Your features
probability = ensemble.predict_proba(X)
```

### Prediction with Confidence

```python
result = ensemble.predict_with_confidence(X)

print(f"Probability: {result['probability'][0]:.2%}")
print(f"Confidence: {result['confidence'][0]:.2%}")
print(f"Agreement: {result['agreement'][0]:.2%}")
```

### High-Confidence Filtering

```python
# Get only 85%+ confidence predictions
high_conf_probs, indices = ensemble.get_high_confidence_predictions(
    X, min_confidence=0.85
)

print(f"Found {len(indices)} high-confidence predictions")
```

### Detailed Analysis

```python
details = ensemble.predict_single(X, idx=0)

print(f"Ensemble: {details['ensemble_probability']:.2%}")
print(f"Confidence: {details['confidence_score']:.2%}")
print(f"Agreement: {details['model_agreement']:.2%}")

print("\nIndividual Models:")
for model, prob in details['individual_predictions'].items():
    print(f"  {model}: {prob:.2%}")

print(f"\nConsensus: {details['consensus']}")
```

---

## Performance Metrics

### Expected Performance (on sufficient data)

| Metric | Target | Current Baseline |
|--------|--------|------------------|
| **Accuracy** | 75-80% | 65-70% |
| **AUC-ROC** | 0.80+ | 0.70-0.75 |
| **Log Loss** | <0.50 | 0.50-0.60 |
| **Calibration** | <5% error | 5-10% error |

### Model Comparison

Typical performance hierarchy:
1. **Ensemble** (best overall)
2. **Gradient Boosting** (strong baseline)
3. **Transformer** (complex patterns)
4. **LSTM** (time-series)
5. **DNN** (feature interactions)

---

## Configuration

**File**: `config.py`

### Key Settings

```python
# Model hyperparameters
GRADIENT_BOOSTING_CONFIG
LSTM_CONFIG
TRANSFORMER_CONFIG
DNN_CONFIG

# Ensemble voting
ENSEMBLE_VOTING_CONFIG

# Training settings
TRAINING_CONFIG

# Model paths
MODEL_PATHS
```

### Customization

```python
from smart_bets_ai.models.config import get_model_config

# Get specific model config
lstm_config = get_model_config('lstm')

# Modify
lstm_config['hidden_size'] = 256
lstm_config['num_layers'] = 3
```

---

## Model Persistence

### Saving

```python
# Save entire ensemble
ensemble.save('models/saved/ensemble')

# Saves:
# - gradient_boosting models (.pkl)
# - lstm model (.pth)
# - transformer model (.pth)
# - dnn model (.pth)
# - ensemble metadata (.pkl)
```

### Loading

```python
# Load entire ensemble
ensemble.load('models/saved/ensemble')

# All models loaded and ready
```

---

## Advanced Features

### Feature Importance

```python
importance = ensemble.get_feature_importance()

print("Top 10 Features:")
print(importance.head(10))
```

### Model Comparison

```python
comparison = ensemble.get_model_comparison(X_test, y_test)

print(comparison)
# Shows accuracy, log_loss, auc_roc for each model
```

### Calibration Analysis

```python
from smart_bets_ai.models.voting import EnsembleVoting

voting = EnsembleVoting()
calibration = voting.calibrate_probabilities(predictions, y_true)

print(f"Mean Calibration Error: {calibration['mean_calibration_error']:.4f}")
```

---

## Requirements

```
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0
```

---

## Performance Tips

### GPU Acceleration

```python
# Automatically uses GPU if available
# Set in config.py:
TRAINING_CONFIG['use_gpu'] = True
```

### Batch Processing

```python
# Process multiple matches efficiently
predictions = ensemble.predict_proba(X_batch)
```

### Memory Optimization

```python
# For large datasets, use batch prediction
batch_size = 1000
for i in range(0, len(X), batch_size):
    batch = X[i:i+batch_size]
    preds = ensemble.predict_proba(batch)
```

---

## Troubleshooting

### Issue: Low Agreement

**Cause**: Models disagree significantly  
**Solution**: Check data quality, retrain with more data

### Issue: Poor Calibration

**Cause**: Probabilities don't match actual outcomes  
**Solution**: Use calibration techniques (Platt scaling, isotonic regression)

### Issue: Slow Training

**Cause**: Large dataset or complex models  
**Solution**: Enable GPU, reduce sequence length, use fewer epochs

---

## Next Steps

1. **Train Models**: Run `train_ensemble.py`
2. **Evaluate**: Check test set performance
3. **Deploy**: Integrate with prediction API
4. **Monitor**: Track real-world performance
5. **Retrain**: Update models with new data

---

## Support

For issues or questions:
- Check configuration in `config.py`
- Review training logs
- Verify data format
- Test with sample data first
