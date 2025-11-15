"""
Ensemble Training Script
Train all deep learning models for the 4 betting markets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime

from features.feature_pipeline import FeaturePipeline
from models.ensemble import EnsemblePredictor
from models.config import TRAINING_CONFIG


class EnsembleTrainer:
    """
    Complete training pipeline for ensemble models
    Handles data loading, feature engineering, model training, and evaluation
    """
    
    def __init__(self, data_path: str = 'data/historical_matches.json'):
        """
        Initialize ensemble trainer
        
        Args:
            data_path: Path to historical match data
        """
        self.data_path = data_path
        self.feature_pipeline = FeaturePipeline()
        self.markets = ['goals', 'cards', 'corners', 'btts']
        self.ensembles = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Load historical match data
        
        Returns:
            DataFrame with match data
        """
        print("Loading historical match data...")
        
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        print(f"Loaded {len(df)} matches")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Engineer features for all markets
        
        Args:
            df: Raw match data
            
        Returns:
            Dictionary of features per market
        """
        print("\nEngineering features...")
        
        # Generate all features
        features_df = self.feature_pipeline.create_features_batch(df.to_dict('records'))
        
        print(f"Generated {len(features_df.columns)} features")
        
        return features_df
    
    def create_labels(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Create target labels for all markets
        
        Args:
            df: Match data with outcomes
            
        Returns:
            Dictionary of labels per market
        """
        labels = {}
        
        # Goals O/U 2.5
        if 'total_goals' in df.columns:
            labels['goals'] = (df['total_goals'] > 2.5).astype(int)
        
        # Cards O/U 3.5
        if 'total_cards' in df.columns:
            labels['cards'] = (df['total_cards'] > 3.5).astype(int)
        
        # Corners O/U 9.5
        if 'total_corners' in df.columns:
            labels['corners'] = (df['total_corners'] > 9.5).astype(int)
        
        # BTTS Y/N
        if 'home_goals' in df.columns and 'away_goals' in df.columns:
            labels['btts'] = ((df['home_goals'] > 0) & (df['away_goals'] > 0)).astype(int)
        
        return labels
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """
        Split data into train/validation/test sets
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=TRAINING_CONFIG['test_split'],
            random_state=TRAINING_CONFIG['random_state'],
            stratify=y if TRAINING_CONFIG['stratify'] else None
        )
        
        # Second split: train vs val
        val_size = TRAINING_CONFIG['validation_split'] / (1 - TRAINING_CONFIG['test_split'])
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=TRAINING_CONFIG['random_state'],
            stratify=y_temp if TRAINING_CONFIG['stratify'] else None
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_market(self, market: str, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train ensemble for a single market
        
        Args:
            market: Market name ('goals', 'cards', 'corners', 'btts')
            X: Features
            y: Labels
            
        Returns:
            Training results
        """
        print(f"\n{'='*70}")
        print(f"TRAINING {market.upper()} MARKET ENSEMBLE")
        print(f"{'='*70}\n")
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        print(f"Data split:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")
        
        # Initialize ensemble
        ensemble = EnsemblePredictor(market, input_size=X.shape[1])
        
        # Train ensemble
        metrics = ensemble.train(
            X_train, y_train,
            validation_data=(X_val, y_val)
        )
        
        # Evaluate on test set
        print(f"\n{'='*70}")
        print(f"TEST SET EVALUATION")
        print(f"{'='*70}")
        
        test_result = ensemble.predict_with_confidence(X_test)
        test_metrics = self._calculate_test_metrics(y_test, test_result)
        
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"AUC-ROC: {test_metrics['auc_roc']:.4f}")
        print(f"Log Loss: {test_metrics['log_loss']:.4f}")
        print(f"Mean Confidence: {test_metrics['mean_confidence']:.4f}")
        print(f"Mean Agreement: {test_metrics['mean_agreement']:.4f}")
        
        # Model comparison
        print(f"\n{'='*70}")
        print(f"MODEL COMPARISON")
        print(f"{'='*70}")
        comparison = ensemble.get_model_comparison(X_test, y_test)
        print(comparison.to_string(index=False))
        
        # Save ensemble
        ensemble.save()
        
        # Store ensemble
        self.ensembles[market] = ensemble
        
        return {
            'training_metrics': metrics,
            'test_metrics': test_metrics,
            'model_comparison': comparison.to_dict('records')
        }
    
    def train_all_markets(self) -> Dict[str, Dict]:
        """
        Train ensembles for all 4 markets
        
        Returns:
            Dictionary of results per market
        """
        # Load data
        df = self.load_data()
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Create labels
        labels = self.create_labels(df)
        
        # Train each market
        results = {}
        
        for market in self.markets:
            if market in labels:
                try:
                    results[market] = self.train_market(market, X, labels[market])
                except Exception as e:
                    print(f"\n✗ Error training {market}: {str(e)}")
                    results[market] = {'error': str(e)}
            else:
                print(f"\n⚠ Skipping {market} - no labels available")
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _calculate_test_metrics(self, y_true, prediction_result) -> Dict[str, float]:
        """Calculate test set metrics"""
        from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss
        
        y_pred_proba = prediction_result['probability']
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'log_loss': log_loss(y_true, y_pred_proba),
            'auc_roc': roc_auc_score(y_true, y_pred_proba),
            'brier_score': brier_score_loss(y_true, y_pred_proba),
            'mean_confidence': float(np.mean(prediction_result['confidence'])),
            'mean_agreement': float(np.mean(prediction_result['agreement']))
        }
    
    def _save_results(self, results: Dict):
        """Save training results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = f'models/saved/training_results_{timestamp}.json'
        
        os.makedirs('models/saved', exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {results_path}")


def main():
    """Main training function"""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║         DEEP LEARNING ENSEMBLE TRAINING PIPELINE                 ║
    ║         Football Betting AI System - Phase 2                     ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize trainer
    trainer = EnsembleTrainer()
    
    # Train all markets
    results = trainer.train_all_markets()
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}\n")
    
    # Summary
    for market, result in results.items():
        if 'error' not in result:
            test_metrics = result['test_metrics']
            print(f"{market.upper()}: Accuracy={test_metrics['accuracy']:.4f}, "
                  f"AUC={test_metrics['auc_roc']:.4f}")
    
    print("\n✓ All models saved and ready for deployment")


if __name__ == '__main__':
    main()
