"""
Ensemble Voting System
Intelligent aggregation of predictions from multiple models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats

from .config import ENSEMBLE_VOTING_CONFIG


class EnsembleVoting:
    """
    Intelligent ensemble voting system
    Combines predictions from multiple models with confidence analysis
    """
    
    def __init__(self):
        self.config = ENSEMBLE_VOTING_CONFIG
        self.voting_method = self.config['voting_method']
        self.weights = self.config['weights']
    
    def aggregate_predictions(self, predictions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Aggregate predictions from multiple models
        
        Args:
            predictions: Dictionary of model predictions
                {
                    'gradient_boosting': np.array([0.75, 0.82, ...]),
                    'lstm': np.array([0.71, 0.79, ...]),
                    'transformer': np.array([0.73, 0.81, ...]),
                    'dnn': np.array([0.74, 0.80, ...])
                }
        
        Returns:
            Dictionary containing:
                - ensemble_probability: Final aggregated probability
                - confidence_score: Confidence in the prediction (0-1)
                - model_agreement: Agreement level between models (0-1)
                - individual_predictions: Original predictions from each model
        """
        # Convert to numpy arrays
        model_names = list(predictions.keys())
        pred_arrays = [predictions[name] for name in model_names]
        
        # Stack predictions
        stacked_preds = np.stack(pred_arrays, axis=0)  # Shape: (num_models, num_samples)
        
        # Calculate ensemble probability
        if self.voting_method == 'weighted':
            ensemble_prob = self._weighted_average(predictions, model_names)
        else:  # soft voting
            ensemble_prob = np.mean(stacked_preds, axis=0)
        
        # Calculate model agreement
        agreement = self._calculate_agreement(stacked_preds)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(ensemble_prob, agreement)
        
        return {
            'ensemble_probability': ensemble_prob,
            'confidence_score': confidence,
            'model_agreement': agreement,
            'individual_predictions': predictions
        }
    
    def _weighted_average(self, predictions: Dict[str, np.ndarray], 
                         model_names: List[str]) -> np.ndarray:
        """Calculate weighted average of predictions"""
        weighted_sum = np.zeros_like(predictions[model_names[0]])
        total_weight = 0.0
        
        for model_name in model_names:
            weight = self.weights.get(model_name, 0.25)  # Default equal weight
            weighted_sum += predictions[model_name] * weight
            total_weight += weight
        
        return weighted_sum / total_weight
    
    def _calculate_agreement(self, stacked_preds: np.ndarray) -> np.ndarray:
        """
        Calculate agreement between models
        
        Args:
            stacked_preds: Shape (num_models, num_samples)
            
        Returns:
            Agreement score for each sample (0-1)
        """
        # Calculate standard deviation across models
        std_dev = np.std(stacked_preds, axis=0)
        
        # Convert to agreement score (lower std = higher agreement)
        # Max std for binary classification is ~0.5, so we normalize
        agreement = 1.0 - (std_dev / 0.5)
        agreement = np.clip(agreement, 0.0, 1.0)
        
        return agreement
    
    def _calculate_confidence(self, ensemble_prob: np.ndarray, 
                             agreement: np.ndarray) -> np.ndarray:
        """
        Calculate confidence score based on probability and agreement
        
        Args:
            ensemble_prob: Ensemble probability predictions
            agreement: Model agreement scores
            
        Returns:
            Confidence scores (0-1)
        """
        # Base confidence from probability distance from 0.5
        prob_confidence = np.abs(ensemble_prob - 0.5) * 2  # Scale to 0-1
        
        # Boost confidence when agreement is high
        confidence = prob_confidence * agreement
        
        # Apply agreement threshold boost
        high_agreement_mask = agreement >= self.config['confidence_boost_threshold']
        confidence[high_agreement_mask] *= 1.1  # 10% boost
        
        # Apply disagreement penalty
        low_agreement_mask = agreement < self.config['min_agreement_threshold']
        confidence[low_agreement_mask] *= (1.0 - self.config['disagreement_penalty'])
        
        # Clip to valid range
        confidence = np.clip(confidence, 0.0, 1.0)
        
        return confidence
    
    def get_prediction_details(self, predictions: Dict[str, np.ndarray], 
                              sample_idx: int) -> Dict[str, Any]:
        """
        Get detailed prediction information for a single sample
        
        Args:
            predictions: Dictionary of model predictions
            sample_idx: Index of the sample
            
        Returns:
            Detailed prediction information
        """
        result = self.aggregate_predictions(predictions)
        
        # Extract single sample
        details = {
            'ensemble_probability': float(result['ensemble_probability'][sample_idx]),
            'confidence_score': float(result['confidence_score'][sample_idx]),
            'model_agreement': float(result['model_agreement'][sample_idx]),
            'individual_predictions': {
                model: float(preds[sample_idx])
                for model, preds in predictions.items()
            }
        }
        
        # Add agreement analysis
        individual_probs = list(details['individual_predictions'].values())
        details['prediction_range'] = {
            'min': float(np.min(individual_probs)),
            'max': float(np.max(individual_probs)),
            'std': float(np.std(individual_probs))
        }
        
        # Determine consensus
        binary_preds = [1 if p >= 0.5 else 0 for p in individual_probs]
        consensus_count = sum(binary_preds)
        total_models = len(binary_preds)
        
        details['consensus'] = {
            'positive_votes': consensus_count,
            'negative_votes': total_models - consensus_count,
            'unanimous': consensus_count == total_models or consensus_count == 0,
            'majority': consensus_count > total_models / 2
        }
        
        return details
    
    def filter_high_confidence(self, predictions: Dict[str, np.ndarray],
                               min_confidence: float = 0.85) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter predictions to only high-confidence ones
        
        Args:
            predictions: Dictionary of model predictions
            min_confidence: Minimum confidence threshold
            
        Returns:
            Tuple of (filtered_probabilities, filtered_indices)
        """
        result = self.aggregate_predictions(predictions)
        
        high_conf_mask = result['confidence_score'] >= min_confidence
        high_conf_indices = np.where(high_conf_mask)[0]
        high_conf_probs = result['ensemble_probability'][high_conf_mask]
        
        return high_conf_probs, high_conf_indices
    
    def get_uncertainty_estimate(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate uncertainty estimate for predictions
        
        Args:
            predictions: Dictionary of model predictions
            
        Returns:
            Uncertainty scores (higher = more uncertain)
        """
        result = self.aggregate_predictions(predictions)
        
        # Uncertainty is inverse of confidence
        uncertainty = 1.0 - result['confidence_score']
        
        return uncertainty
    
    def calibrate_probabilities(self, predictions: Dict[str, np.ndarray],
                                true_labels: np.ndarray) -> Dict[str, Any]:
        """
        Analyze calibration of ensemble predictions
        
        Args:
            predictions: Dictionary of model predictions
            true_labels: True binary labels
            
        Returns:
            Calibration analysis
        """
        result = self.aggregate_predictions(predictions)
        ensemble_prob = result['ensemble_probability']
        
        # Bin predictions
        bins = np.linspace(0, 1, 11)  # 10 bins
        bin_indices = np.digitize(ensemble_prob, bins) - 1
        
        calibration = {
            'bins': [],
            'predicted_prob': [],
            'actual_prob': [],
            'count': []
        }
        
        for i in range(10):
            mask = bin_indices == i
            if mask.sum() > 0:
                calibration['bins'].append(f"{bins[i]:.1f}-{bins[i+1]:.1f}")
                calibration['predicted_prob'].append(float(ensemble_prob[mask].mean()))
                calibration['actual_prob'].append(float(true_labels[mask].mean()))
                calibration['count'].append(int(mask.sum()))
        
        # Calculate calibration error
        if len(calibration['predicted_prob']) > 0:
            calibration['mean_calibration_error'] = float(np.mean([
                abs(pred - actual) 
                for pred, actual in zip(calibration['predicted_prob'], 
                                       calibration['actual_prob'])
            ]))
        else:
            calibration['mean_calibration_error'] = 0.0
        
        return calibration
