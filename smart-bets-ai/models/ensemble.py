"""
Ensemble Predictor
Main orchestrator for all deep learning models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import os

from .gradient_boosting import GradientBoostingEnsemble
from .lstm_model import LSTMPredictor
from .transformer_model import TransformerPredictor
from .neural_net import DeepNeuralNetwork
from .voting import EnsembleVoting
from .config import TRAINING_CONFIG, MODEL_PATHS


class EnsemblePredictor:
    """
    Complete ensemble prediction system
    Orchestrates gradient boosting, LSTM, Transformer, and DNN models
    """
    
    def __init__(self, market: str, input_size: int):
        """
        Initialize ensemble predictor
        
        Args:
            market: Target market ('goals', 'cards', 'corners', 'btts')
            input_size: Number of input features
        """
        self.market = market
        self.input_size = input_size
        
        # Initialize all models
        self.models = {
            'gradient_boosting': GradientBoostingEnsemble(market),
            'lstm': LSTMPredictor(market, input_size),
            'transformer': TransformerPredictor(market, input_size),
            'dnn': DeepNeuralNetwork(market, input_size)
        }
        
        # Initialize voting system
        self.voting = EnsembleVoting()
        
        self.is_trained = False
        self.training_metrics = {}
    
    def train(self, X: pd.DataFrame, y: pd.Series,
              validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
              train_models: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train all ensemble models
        
        Args:
            X: Training features
            y: Training labels
            validation_data: Optional (X_val, y_val) tuple
            train_models: Optional list of models to train (default: all)
            
        Returns:
            Training metrics for all models
        """
        if train_models is None:
            train_models = list(self.models.keys())
        
        print(f"\n{'='*60}")
        print(f"Training Ensemble for {self.market.upper()} Market")
        print(f"{'='*60}\n")
        
        metrics = {}
        
        # Train each model
        for model_name in train_models:
            print(f"\n{'-'*60}")
            print(f"Training {model_name.upper().replace('_', ' ')}")
            print(f"{'-'*60}")
            
            try:
                model = self.models[model_name]
                model_metrics = model.train(X, y, validation_data)
                metrics[model_name] = model_metrics
                
                print(f"✓ {model_name} training complete")
                if 'final_val_accuracy' in model_metrics:
                    print(f"  Validation Accuracy: {model_metrics['final_val_accuracy']:.4f}")
                if 'ensemble' in model_metrics:
                    print(f"  Ensemble Accuracy: {model_metrics['ensemble']['accuracy']:.4f}")
                
            except Exception as e:
                print(f"✗ Error training {model_name}: {str(e)}")
                metrics[model_name] = {'error': str(e)}
        
        # Calculate ensemble metrics
        if validation_data is not None:
            X_val, y_val = validation_data
            ensemble_metrics = self._evaluate_ensemble(X_val, y_val)
            metrics['ensemble'] = ensemble_metrics
            
            print(f"\n{'='*60}")
            print(f"ENSEMBLE PERFORMANCE")
            print(f"{'='*60}")
            print(f"Accuracy: {ensemble_metrics['accuracy']:.4f}")
            print(f"AUC-ROC: {ensemble_metrics['auc_roc']:.4f}")
            print(f"Log Loss: {ensemble_metrics['log_loss']:.4f}")
            print(f"Mean Agreement: {ensemble_metrics['mean_agreement']:.4f}")
            print(f"{'='*60}\n")
        
        self.is_trained = True
        self.training_metrics = metrics
        
        return metrics
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get ensemble probability predictions
        
        Args:
            X: Features
            
        Returns:
            Ensemble probability predictions
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before prediction")
        
        # Get predictions from all models
        predictions = self._get_all_predictions(X)
        
        # Aggregate using voting system
        result = self.voting.aggregate_predictions(predictions)
        
        return result['ensemble_probability']
    
    def predict_with_confidence(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get predictions with confidence scores
        
        Args:
            X: Features
            
        Returns:
            Dictionary with probabilities, confidence, and agreement
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before prediction")
        
        predictions = self._get_all_predictions(X)
        result = self.voting.aggregate_predictions(predictions)
        
        return {
            'probability': result['ensemble_probability'],
            'confidence': result['confidence_score'],
            'agreement': result['model_agreement'],
            'individual_predictions': result['individual_predictions']
        }
    
    def predict_single(self, X: pd.DataFrame, idx: int = 0) -> Dict[str, Any]:
        """
        Get detailed prediction for a single sample
        
        Args:
            X: Features (can contain multiple samples)
            idx: Index of sample to analyze
            
        Returns:
            Detailed prediction information
        """
        predictions = self._get_all_predictions(X)
        details = self.voting.get_prediction_details(predictions, idx)
        
        return details
    
    def get_high_confidence_predictions(self, X: pd.DataFrame,
                                       min_confidence: float = 0.85) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get only high-confidence predictions
        
        Args:
            X: Features
            min_confidence: Minimum confidence threshold
            
        Returns:
            Tuple of (probabilities, indices)
        """
        predictions = self._get_all_predictions(X)
        return self.voting.filter_high_confidence(predictions, min_confidence)
    
    def _get_all_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get predictions from all models"""
        predictions = {}
        
        for model_name, model in self.models.items():
            if model.is_trained:
                try:
                    predictions[model_name] = model.predict_proba(X)
                except Exception as e:
                    print(f"Warning: Could not get predictions from {model_name}: {str(e)}")
        
        if len(predictions) == 0:
            raise ValueError("No trained models available for prediction")
        
        return predictions
    
    def _evaluate_ensemble(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """Evaluate ensemble performance"""
        from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss
        
        result = self.predict_with_confidence(X_val)
        
        y_pred_proba = result['probability']
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'log_loss': log_loss(y_val, y_pred_proba),
            'auc_roc': roc_auc_score(y_val, y_pred_proba),
            'brier_score': brier_score_loss(y_val, y_pred_proba),
            'mean_confidence': float(np.mean(result['confidence'])),
            'mean_agreement': float(np.mean(result['agreement']))
        }
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from gradient boosting models
        
        Returns:
            DataFrame with feature importance
        """
        if 'gradient_boosting' in self.models:
            return self.models['gradient_boosting'].get_feature_importance()
        else:
            return pd.DataFrame()
    
    def save(self, save_dir: Optional[str] = None):
        """
        Save all models to disk
        
        Args:
            save_dir: Directory to save models (default from config)
        """
        if save_dir is None:
            save_dir = MODEL_PATHS['ensemble']
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save each model
        for model_name, model in self.models.items():
            if model.is_trained:
                try:
                    model.save(save_dir)
                    print(f"✓ Saved {model_name}")
                except Exception as e:
                    print(f"✗ Error saving {model_name}: {str(e)}")
        
        # Save ensemble metadata
        import pickle
        metadata = {
            'market': self.market,
            'input_size': self.input_size,
            'is_trained': self.is_trained,
            'training_metrics': self.training_metrics
        }
        metadata_path = os.path.join(save_dir, f'{self.market}_ensemble_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"\n✓ Ensemble saved to {save_dir}")
    
    def load(self, save_dir: Optional[str] = None):
        """
        Load all models from disk
        
        Args:
            save_dir: Directory containing saved models (default from config)
        """
        if save_dir is None:
            save_dir = MODEL_PATHS['ensemble']
        
        # Load metadata
        import pickle
        metadata_path = os.path.join(save_dir, f'{self.market}_ensemble_metadata.pkl')
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.is_trained = metadata['is_trained']
        self.training_metrics = metadata['training_metrics']
        
        # Load each model
        for model_name, model in self.models.items():
            try:
                model.load(save_dir)
                print(f"✓ Loaded {model_name}")
            except Exception as e:
                print(f"✗ Error loading {model_name}: {str(e)}")
        
        print(f"\n✓ Ensemble loaded from {save_dir}")
    
    def get_model_comparison(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Compare performance of all models
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            DataFrame with model comparison
        """
        from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
        
        comparison = []
        
        for model_name, model in self.models.items():
            if model.is_trained:
                try:
                    y_pred_proba = model.predict_proba(X)
                    y_pred = (y_pred_proba >= 0.5).astype(int)
                    
                    comparison.append({
                        'model': model_name,
                        'accuracy': accuracy_score(y, y_pred),
                        'log_loss': log_loss(y, y_pred_proba),
                        'auc_roc': roc_auc_score(y, y_pred_proba)
                    })
                except Exception as e:
                    print(f"Error evaluating {model_name}: {str(e)}")
        
        # Add ensemble
        result = self.predict_with_confidence(X)
        y_pred = (result['probability'] >= 0.5).astype(int)
        
        comparison.append({
            'model': 'ensemble',
            'accuracy': accuracy_score(y, y_pred),
            'log_loss': log_loss(y, result['probability']),
            'auc_roc': roc_auc_score(y, result['probability'])
        })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('accuracy', ascending=False)
        
        return df
