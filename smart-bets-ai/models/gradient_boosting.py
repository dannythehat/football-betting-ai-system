"""
Gradient Boosting Ensemble
XGBoost + LightGBM + CatBoost for robust predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import train_test_split
import pickle
import os

from .config import GRADIENT_BOOSTING_CONFIG, TRAINING_CONFIG


class GradientBoostingEnsemble:
    """
    Ensemble of gradient boosting models
    Combines XGBoost, LightGBM, and CatBoost for robust predictions
    """
    
    def __init__(self, market: str):
        """
        Initialize gradient boosting ensemble
        
        Args:
            market: Target market ('goals', 'cards', 'corners', 'btts')
        """
        self.market = market
        self.models = {
            'xgboost': None,
            'lightgbm': None,
            'catboost': None
        }
        self.feature_names = []
        self.is_trained = False
        
    def train(self, X: pd.DataFrame, y: pd.Series, 
              validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None) -> Dict[str, Any]:
        """
        Train all three gradient boosting models
        
        Args:
            X: Training features
            y: Training labels
            validation_data: Optional (X_val, y_val) tuple
            
        Returns:
            Training metrics for all models
        """
        self.feature_names = list(X.columns)
        
        # Split validation if not provided
        if validation_data is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, 
                test_size=0.2, 
                random_state=TRAINING_CONFIG['random_state'],
                stratify=y if TRAINING_CONFIG['stratify'] else None
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = validation_data
        
        metrics = {}
        
        # Train XGBoost
        print(f"Training XGBoost for {self.market}...")
        self.models['xgboost'] = self._train_xgboost(X_train, y_train, X_val, y_val)
        metrics['xgboost'] = self._evaluate_model(self.models['xgboost'], X_val, y_val, 'xgboost')
        
        # Train LightGBM
        print(f"Training LightGBM for {self.market}...")
        self.models['lightgbm'] = self._train_lightgbm(X_train, y_train, X_val, y_val)
        metrics['lightgbm'] = self._evaluate_model(self.models['lightgbm'], X_val, y_val, 'lightgbm')
        
        # Train CatBoost
        print(f"Training CatBoost for {self.market}...")
        self.models['catboost'] = self._train_catboost(X_train, y_train, X_val, y_val)
        metrics['catboost'] = self._evaluate_model(self.models['catboost'], X_val, y_val, 'catboost')
        
        self.is_trained = True
        
        # Calculate ensemble metrics
        ensemble_preds = self.predict_proba(X_val)
        metrics['ensemble'] = self._calculate_metrics(y_val, ensemble_preds)
        
        return metrics
    
    def _train_xgboost(self, X_train, y_train, X_val, y_val) -> xgb.XGBClassifier:
        """Train XGBoost model"""
        config = GRADIENT_BOOSTING_CONFIG['xgboost'].copy()
        early_stopping = config.pop('early_stopping_rounds')
        
        model = xgb.XGBClassifier(**config)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        return model
    
    def _train_lightgbm(self, X_train, y_train, X_val, y_val) -> lgb.LGBMClassifier:
        """Train LightGBM model"""
        config = GRADIENT_BOOSTING_CONFIG['lightgbm'].copy()
        
        model = lgb.LGBMClassifier(**config)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        return model
    
    def _train_catboost(self, X_train, y_train, X_val, y_val) -> cb.CatBoostClassifier:
        """Train CatBoost model"""
        config = GRADIENT_BOOSTING_CONFIG['catboost'].copy()
        
        model = cb.CatBoostClassifier(**config)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False
        )
        
        return model
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get ensemble probability predictions
        
        Args:
            X: Features
            
        Returns:
            Probability predictions (averaged across models)
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        predictions = []
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            if model is not None:
                pred = model.predict_proba(X)[:, 1]  # Probability of positive class
                predictions.append(pred)
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        
        return ensemble_pred
    
    def get_model_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get individual predictions from each model
        
        Args:
            X: Features
            
        Returns:
            Dictionary of predictions per model
        """
        predictions = {}
        
        for model_name, model in self.models.items():
            if model is not None:
                predictions[model_name] = model.predict_proba(X)[:, 1]
        
        return predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from all models
        
        Returns:
            DataFrame with feature importance scores
        """
        importance_data = []
        
        for model_name, model in self.models.items():
            if model is not None:
                if model_name == 'xgboost':
                    importance = model.feature_importances_
                elif model_name == 'lightgbm':
                    importance = model.feature_importances_
                elif model_name == 'catboost':
                    importance = model.feature_importances_
                
                for feature, score in zip(self.feature_names, importance):
                    importance_data.append({
                        'feature': feature,
                        'model': model_name,
                        'importance': score
                    })
        
        df = pd.DataFrame(importance_data)
        
        # Calculate average importance across models
        avg_importance = df.groupby('feature')['importance'].mean().reset_index()
        avg_importance = avg_importance.sort_values('importance', ascending=False)
        
        return avg_importance
    
    def _evaluate_model(self, model, X_val, y_val, model_name: str) -> Dict[str, float]:
        """Evaluate a single model"""
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        return self._calculate_metrics(y_val, y_pred_proba)
    
    def _calculate_metrics(self, y_true, y_pred_proba) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss
        
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'log_loss': log_loss(y_true, y_pred_proba),
            'auc_roc': roc_auc_score(y_true, y_pred_proba),
            'brier_score': brier_score_loss(y_true, y_pred_proba)
        }
        
        return metrics
    
    def save(self, save_dir: str):
        """
        Save all models to disk
        
        Args:
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            if model is not None:
                model_path = os.path.join(save_dir, f'{self.market}_{model_name}.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
        
        # Save metadata
        metadata = {
            'market': self.market,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        metadata_path = os.path.join(save_dir, f'{self.market}_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    def load(self, save_dir: str):
        """
        Load all models from disk
        
        Args:
            save_dir: Directory containing saved models
        """
        # Load metadata
        metadata_path = os.path.join(save_dir, f'{self.market}_metadata.pkl')
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.feature_names = metadata['feature_names']
        self.is_trained = metadata['is_trained']
        
        # Load models
        for model_name in self.models.keys():
            model_path = os.path.join(save_dir, f'{self.market}_{model_name}.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
