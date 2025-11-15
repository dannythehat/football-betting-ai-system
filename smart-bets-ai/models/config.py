"""
Model Configuration
Hyperparameters and settings for all ensemble models
"""

from typing import Dict, Any

# Gradient Boosting Ensemble Configuration
GRADIENT_BOOSTING_CONFIG = {
    'xgboost': {
        'n_estimators': 300,
        'max_depth': 8,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'eval_metric': 'logloss',
        'early_stopping_rounds': 50,
        'random_state': 42
    },
    'lightgbm': {
        'n_estimators': 300,
        'max_depth': 8,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'metric': 'binary_logloss',
        'early_stopping_rounds': 50,
        'random_state': 42,
        'verbose': -1
    },
    'catboost': {
        'iterations': 300,
        'depth': 8,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bylevel': 0.8,
        'min_data_in_leaf': 20,
        'l2_leaf_reg': 3.0,
        'eval_metric': 'Logloss',
        'early_stopping_rounds': 50,
        'random_state': 42,
        'verbose': False
    }
}

# LSTM Configuration
LSTM_CONFIG = {
    'sequence_length': 10,  # Last 10 matches
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.3,
    'bidirectional': True,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'early_stopping_patience': 15,
    'weight_decay': 1e-5
}

# Transformer Configuration
TRANSFORMER_CONFIG = {
    'sequence_length': 10,
    'd_model': 128,  # Embedding dimension
    'nhead': 8,  # Number of attention heads
    'num_encoder_layers': 4,
    'dim_feedforward': 512,
    'dropout': 0.2,
    'learning_rate': 0.0001,
    'batch_size': 32,
    'epochs': 100,
    'early_stopping_patience': 15,
    'weight_decay': 1e-5
}

# Deep Neural Network Configuration
DNN_CONFIG = {
    'hidden_layers': [256, 128, 64, 32],
    'activation': 'relu',
    'dropout': 0.3,
    'batch_normalization': True,
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 150,
    'early_stopping_patience': 20,
    'weight_decay': 1e-4
}

# Ensemble Voting Configuration
ENSEMBLE_VOTING_CONFIG = {
    'voting_method': 'weighted',  # 'weighted' or 'soft'
    'weights': {
        'gradient_boosting': 0.35,  # Highest weight - proven performance
        'lstm': 0.25,  # Time-series patterns
        'transformer': 0.20,  # Complex relationships
        'dnn': 0.20  # Feature interactions
    },
    'min_agreement_threshold': 0.70,  # 70% models must agree for high confidence
    'confidence_boost_threshold': 0.90,  # 90% agreement = confidence boost
    'disagreement_penalty': 0.05  # Reduce confidence when models disagree
}

# Training Configuration
TRAINING_CONFIG = {
    'train_split': 0.8,
    'validation_split': 0.1,
    'test_split': 0.1,
    'stratify': True,
    'random_state': 42,
    'cross_validation_folds': 5,
    'use_gpu': True,  # Use GPU if available
    'num_workers': 4,  # Data loading workers
    'pin_memory': True
}

# Feature Selection Configuration
FEATURE_CONFIG = {
    'use_feature_selection': True,
    'selection_method': 'importance',  # 'importance', 'correlation', 'mutual_info'
    'max_features': 100,  # Use top 100 features
    'importance_threshold': 0.001,
    'correlation_threshold': 0.95  # Remove highly correlated features
}

# Model Persistence Configuration
MODEL_PATHS = {
    'gradient_boosting': 'models/saved/gradient_boosting',
    'lstm': 'models/saved/lstm',
    'transformer': 'models/saved/transformer',
    'dnn': 'models/saved/dnn',
    'ensemble': 'models/saved/ensemble',
    'metadata': 'models/saved/metadata'
}

# Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    'min_accuracy': 0.70,  # Minimum 70% accuracy
    'min_auc': 0.75,  # Minimum 0.75 AUC-ROC
    'max_logloss': 0.55,  # Maximum 0.55 log loss
    'calibration_tolerance': 0.05  # 5% calibration error tolerance
}

def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific model
    
    Args:
        model_name: Name of the model ('xgboost', 'lightgbm', 'catboost', 'lstm', 'transformer', 'dnn')
        
    Returns:
        Configuration dictionary
    """
    configs = {
        'xgboost': GRADIENT_BOOSTING_CONFIG['xgboost'],
        'lightgbm': GRADIENT_BOOSTING_CONFIG['lightgbm'],
        'catboost': GRADIENT_BOOSTING_CONFIG['catboost'],
        'lstm': LSTM_CONFIG,
        'transformer': TRANSFORMER_CONFIG,
        'dnn': DNN_CONFIG
    }
    
    return configs.get(model_name, {})

def get_ensemble_config() -> Dict[str, Any]:
    """Get ensemble voting configuration"""
    return ENSEMBLE_VOTING_CONFIG

def get_training_config() -> Dict[str, Any]:
    """Get training configuration"""
    return TRAINING_CONFIG
