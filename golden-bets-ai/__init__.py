"""
Golden Bets AI Module
Filters Smart Bets predictions to identify 1-3 daily picks with 85%+ confidence
"""
from .filter import GoldenBetsFilter
from .predict import GoldenBetsPredictor

__all__ = ['GoldenBetsFilter', 'GoldenBetsPredictor']
