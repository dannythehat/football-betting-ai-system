"""
Model Training for Smart Bets AI
Trains XGBoost models for different betting markets
"""

import os
import json
import pickle
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score, 
    classification_report, confusion_matrix
)
import xgboost as xgb


class SmartBetsModelTrainer:
    """
    Trains and manages XGBoost models for different betting markets
    """
    
    # Market definitions
    MARKETS = {
        'match_result': ['home_win', 'draw', 'away_win'],
        'total_goals': ['over_0_5', 'over_1_5', 'over_2_5', 'over_3_5', 'over_4_5'],
        'btts': ['btts'],
        'corners': ['corners_over_8_5', 'corners_over_9_5', 'corners_over_10_5'],
        'cards': ['cards_over_3_5', 'cards_over_4_5']
    }
    
    def __init__(self, model_dir: str = 'models'):
        """
        Initialize model trainer
        
        Args:
            model_dir: Directory to save trained models
        """
        self.model_dir = model_dir
        self.models = {}
        self.feature_names = []
        self.training_metrics = {}
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def train_all_markets(
        self, 
        X: pd.DataFrame, 
        targets: Dict[str, pd.Series],
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Dict]:
        """
        Train models for all betting markets
        
        Args:
            X: Feature DataFrame
            targets: Dictionary of target variables
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary of training metrics per market
        """
        self.feature_names = X.columns.tolist()
        all_metrics = {}
        
        print(f"Training models for {len(targets)} markets...")
        print(f"Features: {len(self.feature_names)}")
        print(f"Samples: {len(X)}")
        print(f"Train/Test split: {1-test_size:.0%}/{test_size:.0%}\n")
        
        for market_name, target_name in targets.items():
            print(f"Training {market_name}...")
            
            # Get target data
            y = targets[target_name]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Train model
            model, metrics = self._train_single_model(
                X_train, X_test, y_train, y_test, market_name
            )
            
            # Store model and metrics
            self.models[market_name] = model
            all_metrics[market_name] = metrics
            
            print(f"  ✓ Accuracy: {metrics['accuracy']:.3f}")
            print(f"  ✓ ROC-AUC: {metrics['roc_auc']:.3f}")
            print(f"  ✓ Log Loss: {metrics['log_loss']:.3f}\n")
        
        self.training_metrics = all_metrics
        return all_metrics
    
    def _train_single_model(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        market_name: str
    ) -> Tuple[xgb.XGBClassifier, Dict]:
        """
        Train a single XGBoost model for one market
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            market_name: Name of the betting market
            
        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        # XGBoost parameters optimized for probability calibration
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'use_label_encoder': False
        }
        
        # Initialize model
        model = xgb.XGBClassifier(**params)
        
        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Generate predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'log_loss': log_loss(y_test, y_pred_proba),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'positive_rate': y_train.mean(),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'feature_importance': self._get_feature_importance(model),
            'best_iteration': model.best_iteration if hasattr(model, 'best_iteration') else None
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1
        )
        metrics['cv_roc_auc_mean'] = cv_scores.mean()
        metrics['cv_roc_auc_std'] = cv_scores.std()
        
        return model, metrics
    
    def _get_feature_importance(self, model: xgb.XGBClassifier, top_n: int = 10) -> Dict:
        """Get top N most important features"""
        importance_dict = dict(zip(
            self.feature_names,
            model.feature_importances_
        ))
        
        # Sort by importance
        sorted_features = sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return dict(sorted_features[:top_n])
    
    def save_models(self, version: str = None) -> str:
        """
        Save all trained models to disk
        
        Args:
            version: Model version string (defaults to timestamp)
            
        Returns:
            Path to saved models directory
        """
        if not version:
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        version_dir = os.path.join(self.model_dir, f'v{version}')
        os.makedirs(version_dir, exist_ok=True)
        
        # Save each model
        for market_name, model in self.models.items():
            model_path = os.path.join(version_dir, f'{market_name}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save feature names
        features_path = os.path.join(version_dir, 'feature_names.json')
        with open(features_path, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        
        # Save training metrics
        metrics_path = os.path.join(version_dir, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.training_metrics, f, indent=2)
        
        # Save metadata
        metadata = {
            'version': version,
            'created_at': datetime.now().isoformat(),
            'num_models': len(self.models),
            'num_features': len(self.feature_names),
            'markets': list(self.models.keys())
        }
        metadata_path = os.path.join(version_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Models saved to: {version_dir}")
        print(f"  - {len(self.models)} market models")
        print(f"  - {len(self.feature_names)} features")
        print(f"  - Version: {version}")
        
        return version_dir
    
    def load_models(self, version: str) -> None:
        """
        Load trained models from disk
        
        Args:
            version: Model version to load
        """
        version_dir = os.path.join(self.model_dir, f'v{version}')
        
        if not os.path.exists(version_dir):
            raise FileNotFoundError(f"Model version not found: {version}")
        
        # Load feature names
        features_path = os.path.join(version_dir, 'feature_names.json')
        with open(features_path, 'r') as f:
            self.feature_names = json.load(f)
        
        # Load training metrics
        metrics_path = os.path.join(version_dir, 'training_metrics.json')
        with open(metrics_path, 'r') as f:
            self.training_metrics = json.load(f)
        
        # Load each model
        self.models = {}
        for market_name in self.training_metrics.keys():
            model_path = os.path.join(version_dir, f'{market_name}.pkl')
            with open(model_path, 'rb') as f:
                self.models[market_name] = pickle.load(f)
        
        print(f"✓ Models loaded from: {version_dir}")
        print(f"  - {len(self.models)} market models")
        print(f"  - {len(self.feature_names)} features")
    
    def get_model_summary(self) -> Dict:
        """Get summary of all trained models"""
        summary = {
            'num_models': len(self.models),
            'num_features': len(self.feature_names),
            'markets': list(self.models.keys()),
            'metrics_summary': {}
        }
        
        for market, metrics in self.training_metrics.items():
            summary['metrics_summary'][market] = {
                'accuracy': metrics['accuracy'],
                'roc_auc': metrics['roc_auc'],
                'log_loss': metrics['log_loss'],
                'cv_roc_auc': f"{metrics['cv_roc_auc_mean']:.3f} ± {metrics['cv_roc_auc_std']:.3f}"
            }
        
        return summary


def print_training_summary(metrics: Dict[str, Dict]) -> None:
    """
    Print a formatted summary of training results
    
    Args:
        metrics: Dictionary of training metrics per market
    """
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    print(f"\n{'Market':<25} {'Accuracy':<12} {'ROC-AUC':<12} {'Log Loss':<12}")
    print("-"*70)
    
    for market, m in metrics.items():
        print(f"{market:<25} {m['accuracy']:<12.3f} {m['roc_auc']:<12.3f} {m['log_loss']:<12.3f}")
    
    # Calculate averages
    avg_accuracy = np.mean([m['accuracy'] for m in metrics.values()])
    avg_roc_auc = np.mean([m['roc_auc'] for m in metrics.values()])
    avg_log_loss = np.mean([m['log_loss'] for m in metrics.values()])
    
    print("-"*70)
    print(f"{'AVERAGE':<25} {avg_accuracy:<12.3f} {avg_roc_auc:<12.3f} {avg_log_loss:<12.3f}")
    print("="*70)
    
    # Top features across all models
    print("\nTOP FEATURES (by average importance):")
    all_features = {}
    for market, m in metrics.items():
        for feature, importance in m['feature_importance'].items():
            if feature not in all_features:
                all_features[feature] = []
            all_features[feature].append(importance)
    
    avg_importance = {
        feature: np.mean(importances) 
        for feature, importances in all_features.items()
    }
    
    sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
    
    for i, (feature, importance) in enumerate(sorted_features[:10], 1):
        print(f"  {i}. {feature:<40} {importance:.4f}")
    
    print("\n" + "="*70 + "\n")
