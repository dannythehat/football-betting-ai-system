"""
Advanced Feature Engineering Module
Transforms raw match data into 100+ intelligent features for ML models
"""

from .core_stats import CoreStatisticsEngine
from .head_to_head import HeadToHeadAnalyzer
from .player_intelligence import PlayerIntelligenceEngine
from .environmental import EnvironmentalAnalyzer
from .momentum import MomentumAnalyzer
from .market_specific import MarketSpecificFeatures
from .feature_pipeline import FeaturePipeline

__all__ = [
    'CoreStatisticsEngine',
    'HeadToHeadAnalyzer',
    'PlayerIntelligenceEngine',
    'EnvironmentalAnalyzer',
    'MomentumAnalyzer',
    'MarketSpecificFeatures',
    'FeaturePipeline'
]

__version__ = '2.0.0'
