"""
Train Cards Over 3.5 Model
Uses the same training pipeline as goals model
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.train_btts import train_btts_model
from training.config import TRAINING_DATA_PATHS


def train_cards_model(data_path: str = None):
    """Train Cards model using shared training pipeline"""
    if data_path is None:
        data_path = str(TRAINING_DATA_PATHS['cards'])
    
    print("\n" + "=" * 60)
    print("TRAINING CARDS OVER 3.5 MODEL")
    print("=" * 60)
    
    # Reuse BTTS training logic (which reuses goals logic)
    from training.train_goals import prepare_data, train_single_model
    from training.utils import (
        ensemble_predictions, fit_calibration_model, apply_calibration,
        calculate_metrics, save_model_with_metadata
    )
    from training.config import (
        DEFAULT_MODELS, ENSEMBLE_WEIGHTS, CALIBRATION_METHOD, MODELS_DIR
    )
    import json
    import pickle
    
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = prepare_data(data_path)
    
    models = {}
    val_predictions = {}
    all_metrics = {}
    
    for model_type in DEFAULT_MODELS:
        try:
            model, val_proba, metrics = train_single_model(
                model_type, X_train, y_train, X_val, y_val
            )
            models[model_type] = model
            val_predictions[model_type] = val_proba
            all_metrics[model_type] = metrics
        except Exception as e:
            print(f"❌ Error training {model_type}: {e}")
    
    ensemble_proba = ensemble_predictions(val_predictions, ENSEMBLE_WEIGHTS)
    calibration_model = fit_calibration_model(ensemble_proba, y_val.values, CALIBRATION_METHOD)
    calibrated_proba = apply_calibration(calibration_model, ensemble_proba, CALIBRATION_METHOD)
    
    test_predictions = {mt: m.predict_proba(X_test)[:, 1] for mt, m in models.items()}
    test_ensemble = ensemble_predictions(test_predictions, ENSEMBLE_WEIGHTS)
    test_calibrated = apply_calibration(calibration_model, test_ensemble, CALIBRATION_METHOD)
    test_metrics = calculate_metrics(y_test, test_calibrated)
    
    print(f"\n✅ Test set performance:")
    print(f"   Log Loss:    {test_metrics['log_loss']:.4f}")
    print(f"   Brier Score: {test_metrics['brier_score']:.4f}")
    
    for model_type, model in models.items():
        save_model_with_metadata(
            model=model, market='cards', metrics=all_metrics[model_type],
            feature_columns=feature_cols, model_type=model_type
        )
    
    ensemble_path = MODELS_DIR / 'cards' / 'ensemble_metadata.json'
    ensemble_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ensemble_path, 'w') as f:
        json.dump({
            'market': 'cards', 'model_type': 'ensemble', 'version': 'v1.0.0',
            'base_models': list(models.keys()), 'weights': ENSEMBLE_WEIGHTS,
            'calibration_method': CALIBRATION_METHOD,
            'metrics': {'test': test_metrics}, 'feature_columns': feature_cols
        }, f, indent=2)
    
    with open(MODELS_DIR / 'cards' / 'ensemble_calibration.pkl', 'wb') as f:
        pickle.dump(calibration_model, f)
    
    print("\n✅ CARDS MODEL TRAINING COMPLETE")
    return {'models': models, 'test_metrics': test_metrics}


if __name__ == "__main__":
    train_cards_model()
