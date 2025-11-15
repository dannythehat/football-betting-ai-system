"""
Feature Pipeline Orchestrator
Coordinates all feature engineering modules to generate 100+ intelligent features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

from .core_stats import CoreStatisticsEngine
from .head_to_head import HeadToHeadAnalyzer
from .momentum import MomentumAnalyzer
from .market_specific import MarketSpecificFeatures

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Orchestrates all feature engineering modules
    Generates 100+ features from raw match data
    """
    
    def __init__(self):
        self.core_stats = CoreStatisticsEngine()
        self.h2h_analyzer = HeadToHeadAnalyzer()
        self.momentum_analyzer = MomentumAnalyzer()
        self.market_features = MarketSpecificFeatures()
        
        self.feature_count = 0
        self.feature_names = []
    
    def transform(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Transform raw match data into engineered features
        
        Args:
            match_data: Dictionary containing all match and team data
            
        Returns:
            Dictionary of 100+ engineered features
        """
        features = {}
        
        try:
            # Core statistical features (30+ features)
            logger.debug("Generating core statistics features...")
            features.update(self.core_stats.create_features(match_data))
            
            # Head-to-head features (18+ features)
            logger.debug("Generating head-to-head features...")
            features.update(self.h2h_analyzer.analyze_h2h(match_data))
            
            # Momentum and psychological features (25+ features)
            logger.debug("Generating momentum features...")
            features.update(self.momentum_analyzer.analyze_momentum(match_data))
            
            # Market-specific features (60+ features)
            logger.debug("Generating market-specific features...")
            features.update(self.market_features.create_all_market_features(match_data))
            
            # Add basic features from original system (for compatibility)
            features.update(self._add_legacy_features(match_data))
            
            self.feature_count = len(features)
            logger.info(f"Generated {self.feature_count} features successfully")
            
        except Exception as e:
            logger.error(f"Error in feature generation: {str(e)}")
            raise
        
        return features
    
    def transform_batch(self, matches_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Transform multiple matches into feature DataFrame
        
        Args:
            matches_data: List of match data dictionaries
            
        Returns:
            DataFrame with features for all matches
        """
        features_list = []
        
        for i, match_data in enumerate(matches_data):
            try:
                features = self.transform(match_data)
                features['match_id'] = match_data.get('match_id', f'match_{i}')
                features_list.append(features)
            except Exception as e:
                logger.error(f"Error processing match {match_data.get('match_id', i)}: {str(e)}")
                continue
        
        if not features_list:
            raise ValueError("No features generated from input data")
        
        df = pd.DataFrame(features_list)
        
        # Set match_id as index
        if 'match_id' in df.columns:
            df = df.set_index('match_id')
        
        logger.info(f"Generated features for {len(df)} matches with {len(df.columns)} features each")
        
        return df
    
    def _add_legacy_features(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Add basic features from original system for backward compatibility
        """
        features = {}
        
        # Basic averages (if not already added)
        features['home_goals_avg'] = match_data.get('home_goals_avg', 0)
        features['away_goals_avg'] = match_data.get('away_goals_avg', 0)
        features['home_goals_conceded_avg'] = match_data.get('home_goals_conceded_avg', 0)
        features['away_goals_conceded_avg'] = match_data.get('away_goals_conceded_avg', 0)
        features['home_corners_avg'] = match_data.get('home_corners_avg', 0)
        features['away_corners_avg'] = match_data.get('away_corners_avg', 0)
        features['home_cards_avg'] = match_data.get('home_cards_avg', 0)
        features['away_cards_avg'] = match_data.get('away_cards_avg', 0)
        features['home_btts_rate'] = match_data.get('home_btts_rate', 0)
        features['away_btts_rate'] = match_data.get('away_btts_rate', 0)
        
        # Combined metrics
        features['combined_goals_avg'] = features['home_goals_avg'] + features['away_goals_avg']
        features['combined_corners_avg'] = features['home_corners_avg'] + features['away_corners_avg']
        features['combined_cards_avg'] = features['home_cards_avg'] + features['away_cards_avg']
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names across all modules
        
        Returns:
            List of feature names
        """
        if self.feature_names:
            return self.feature_names
        
        feature_names = []
        
        # Collect from all modules
        feature_names.extend(self.core_stats.get_feature_names())
        feature_names.extend(self.h2h_analyzer.get_feature_names())
        feature_names.extend(self.momentum_analyzer.get_feature_names())
        feature_names.extend(self.market_features.get_feature_names())
        
        # Add legacy features
        legacy_features = [
            'home_goals_avg', 'away_goals_avg',
            'home_goals_conceded_avg', 'away_goals_conceded_avg',
            'home_corners_avg', 'away_corners_avg',
            'home_cards_avg', 'away_cards_avg',
            'home_btts_rate', 'away_btts_rate',
            'combined_goals_avg', 'combined_corners_avg', 'combined_cards_avg'
        ]
        feature_names.extend(legacy_features)
        
        # Remove duplicates while preserving order
        seen = set()
        self.feature_names = [x for x in feature_names if not (x in seen or seen.add(x))]
        
        return self.feature_names
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Group features by their source module for interpretability
        
        Returns:
            Dictionary mapping module names to their feature lists
        """
        return {
            'core_statistics': self.core_stats.get_feature_names(),
            'head_to_head': self.h2h_analyzer.get_feature_names(),
            'momentum': self.momentum_analyzer.get_feature_names(),
            'market_specific': self.market_features.get_feature_names(),
            'legacy': [
                'home_goals_avg', 'away_goals_avg',
                'home_goals_conceded_avg', 'away_goals_conceded_avg',
                'home_corners_avg', 'away_corners_avg',
                'home_cards_avg', 'away_cards_avg',
                'home_btts_rate', 'away_btts_rate',
                'combined_goals_avg', 'combined_corners_avg', 'combined_cards_avg'
            ]
        }
    
    def validate_features(self, features: Dict[str, float]) -> bool:
        """
        Validate generated features for quality and completeness
        
        Args:
            features: Dictionary of features
            
        Returns:
            True if features pass validation
        """
        # Check for NaN values
        nan_count = sum(1 for v in features.values() if pd.isna(v))
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in features")
        
        # Check for infinite values
        inf_count = sum(1 for v in features.values() if np.isinf(v))
        if inf_count > 0:
            logger.warning(f"Found {inf_count} infinite values in features")
            return False
        
        # Check minimum feature count
        if len(features) < 50:
            logger.error(f"Insufficient features generated: {len(features)} < 50")
            return False
        
        return True
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics about the feature pipeline
        
        Returns:
            Dictionary with pipeline statistics
        """
        feature_groups = self.get_feature_importance_groups()
        
        return {
            'total_features': len(self.get_feature_names()),
            'feature_groups': {
                name: len(features) 
                for name, features in feature_groups.items()
            },
            'modules': {
                'core_statistics': 'Rolling averages, variance, streaks, venue splits',
                'head_to_head': 'Historical matchup analysis and trends',
                'momentum': 'Form, confidence, pressure, fatigue indicators',
                'market_specific': 'Specialized features for each betting market'
            }
        }
