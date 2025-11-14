"""
Tests for Feature Engineering Module
"""

import pytest
import pandas as pd
import numpy as np

from smart_bets_ai.feature_engineering import FeatureEngineer, prepare_training_data


@pytest.fixture
def sample_match_data():
    """Create sample match data for testing"""
    return pd.DataFrame([
        {
            'match_id': 'test_001',
            'home_team_id': 1,
            'away_team_id': 2,
            'match_datetime': '2025-11-15 14:00:00',
            'league': 'Premier League',
            'season': '2024-25',
            'status': 'scheduled',
            'home_goals_avg': 1.5,
            'away_goals_avg': 1.2,
            'home_goals_conceded_avg': 0.9,
            'away_goals_conceded_avg': 1.4,
            'home_corners_avg': 5.5,
            'away_corners_avg': 4.8,
            'home_cards_avg': 2.2,
            'away_cards_avg': 2.5,
            'home_btts_rate': 0.6,
            'away_btts_rate': 0.55,
            'home_form': 'WWDWL',
            'away_form': 'LWDLD'
        }
    ])


@pytest.fixture
def sample_results_data():
    """Create sample results data for testing"""
    return pd.DataFrame([
        {
            'match_id': 'test_001',
            'home_goals': 2,
            'away_goals': 1,
            'result': 'home_win',
            'total_goals': 3,
            'btts': True,
            'over_0_5': True,
            'over_1_5': True,
            'over_2_5': True,
            'over_3_5': False,
            'over_4_5': False,
            'corners_over_8_5': True,
            'corners_over_9_5': True,
            'corners_over_10_5': False,
            'cards_over_3_5': True,
            'cards_over_4_5': False
        }
    ])


class TestFeatureEngineer:
    """Test FeatureEngineer class"""
    
    def test_initialization(self):
        """Test FeatureEngineer initialization"""
        fe = FeatureEngineer()
        assert fe.feature_columns == []
    
    def test_create_features(self, sample_match_data):
        """Test feature creation"""
        fe = FeatureEngineer()
        features = fe.create_features(sample_match_data)
        
        # Check that features were created
        assert len(features) == 1
        assert len(features.columns) > len(sample_match_data.columns)
        
        # Check specific features exist
        assert 'home_attack_strength' in features.columns
        assert 'expected_total_goals' in features.columns
        assert 'home_form_score' in features.columns
        assert 'btts_combined_rate' in features.columns
    
    def test_goal_features(self, sample_match_data):
        """Test goal-related features"""
        fe = FeatureEngineer()
        features = fe.create_features(sample_match_data)
        
        # Check goal features
        assert features['home_attack_strength'].iloc[0] == 1.5
        assert features['away_attack_strength'].iloc[0] == 1.2
        assert features['expected_total_goals'].iloc[0] == 2.7
        assert features['home_goal_advantage'].iloc[0] == pytest.approx(0.3)
    
    def test_defensive_features(self, sample_match_data):
        """Test defensive features"""
        fe = FeatureEngineer()
        features = fe.create_features(sample_match_data)
        
        # Check defensive features
        assert features['home_defense_strength'].iloc[0] == 0.9
        assert features['away_defense_strength'].iloc[0] == 1.4
        assert features['combined_defensive_weakness'].iloc[0] == 2.3
    
    def test_form_features(self, sample_match_data):
        """Test form-based features"""
        fe = FeatureEngineer()
        features = fe.create_features(sample_match_data)
        
        # WWDWL = 3+3+1+3+0 = 10
        # LWDLD = 0+3+1+0+1 = 5
        assert features['home_form_score'].iloc[0] == 10
        assert features['away_form_score'].iloc[0] == 5
        assert features['form_difference'].iloc[0] == 5
    
    def test_btts_features(self, sample_match_data):
        """Test BTTS features"""
        fe = FeatureEngineer()
        features = fe.create_features(sample_match_data)
        
        # Check BTTS features
        expected_btts_rate = (0.6 + 0.55) / 2
        assert features['btts_combined_rate'].iloc[0] == pytest.approx(expected_btts_rate)
        assert features['btts_likely'].iloc[0] == 1  # Both > 0.5
    
    def test_corners_cards_features(self, sample_match_data):
        """Test corners and cards features"""
        fe = FeatureEngineer()
        features = fe.create_features(sample_match_data)
        
        # Check corners/cards features
        assert features['expected_total_corners'].iloc[0] == 10.3
        assert features['expected_total_cards'].iloc[0] == 4.7
        assert features['high_corners_expected'].iloc[0] == 1  # > 10.0
        assert features['high_cards_expected'].iloc[0] == 1  # > 4.0
    
    def test_derived_features(self, sample_match_data):
        """Test derived features"""
        fe = FeatureEngineer()
        features = fe.create_features(sample_match_data)
        
        # Check derived features
        assert 'home_goal_ratio' in features.columns
        assert 'overall_strength_diff' in features.columns
        assert 'high_scoring_expected' in features.columns
        assert features['high_scoring_expected'].iloc[0] == 1  # 2.7 > 2.5
    
    def test_form_to_score(self):
        """Test form string to score conversion"""
        assert FeatureEngineer._form_to_score('WWWWW') == 15
        assert FeatureEngineer._form_to_score('LLLLL') == 0
        assert FeatureEngineer._form_to_score('WDWDW') == 11
        assert FeatureEngineer._form_to_score('') == 7
        assert FeatureEngineer._form_to_score(None) == 7
    
    def test_create_target_variables(self, sample_results_data):
        """Test target variable creation"""
        fe = FeatureEngineer()
        targets = fe.create_target_variables(sample_results_data)
        
        # Check all targets created
        assert 'home_win' in targets
        assert 'draw' in targets
        assert 'away_win' in targets
        assert 'over_2_5' in targets
        assert 'btts' in targets
        
        # Check values
        assert targets['home_win'].iloc[0] == 1
        assert targets['draw'].iloc[0] == 0
        assert targets['away_win'].iloc[0] == 0
        assert targets['over_2_5'].iloc[0] == 1
        assert targets['btts'].iloc[0] == 1
    
    def test_get_feature_names(self, sample_match_data):
        """Test getting feature names"""
        fe = FeatureEngineer()
        features = fe.create_features(sample_match_data)
        feature_names = fe.get_feature_names()
        
        # Check feature names returned
        assert len(feature_names) > 0
        assert 'match_id' not in feature_names
        assert 'status' not in feature_names
        assert 'home_attack_strength' in feature_names


class TestPrepareTrainingData:
    """Test prepare_training_data function"""
    
    def test_prepare_training_data(self, sample_match_data, sample_results_data):
        """Test complete training data preparation"""
        X, targets, fe = prepare_training_data(sample_match_data, sample_results_data)
        
        # Check outputs
        assert isinstance(X, pd.DataFrame)
        assert isinstance(targets, dict)
        assert isinstance(fe, FeatureEngineer)
        
        # Check data shapes
        assert len(X) == 1
        assert len(targets) > 0
        
        # Check all targets present
        assert 'home_win' in targets
        assert 'over_2_5' in targets
        assert 'btts' in targets
    
    def test_feature_target_alignment(self, sample_match_data, sample_results_data):
        """Test that features and targets are properly aligned"""
        X, targets, fe = prepare_training_data(sample_match_data, sample_results_data)
        
        # All targets should have same length as features
        for target_name, target_series in targets.items():
            assert len(target_series) == len(X)


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_missing_form_data(self):
        """Test handling of missing form data"""
        data = pd.DataFrame([{
            'match_id': 'test_001',
            'home_goals_avg': 1.5,
            'away_goals_avg': 1.2,
            'home_goals_conceded_avg': 0.9,
            'away_goals_conceded_avg': 1.4,
            'home_corners_avg': 5.5,
            'away_corners_avg': 4.8,
            'home_cards_avg': 2.2,
            'away_cards_avg': 2.5,
            'home_btts_rate': 0.6,
            'away_btts_rate': 0.55,
            'home_form': None,
            'away_form': ''
        }])
        
        fe = FeatureEngineer()
        features = fe.create_features(data)
        
        # Should handle missing form gracefully
        assert features['home_form_score'].iloc[0] == 7
        assert features['away_form_score'].iloc[0] == 7
    
    def test_zero_goals_conceded(self):
        """Test handling of zero goals conceded (division by zero)"""
        data = pd.DataFrame([{
            'match_id': 'test_001',
            'home_goals_avg': 2.0,
            'away_goals_avg': 1.5,
            'home_goals_conceded_avg': 0.0,
            'away_goals_conceded_avg': 0.0,
            'home_corners_avg': 5.5,
            'away_corners_avg': 4.8,
            'home_cards_avg': 2.2,
            'away_cards_avg': 2.5,
            'home_btts_rate': 0.6,
            'away_btts_rate': 0.55,
            'home_form': 'WWWWW',
            'away_form': 'WWWWW'
        }])
        
        fe = FeatureEngineer()
        features = fe.create_features(data)
        
        # Should handle zero division with small epsilon
        assert not np.isnan(features['home_goal_ratio'].iloc[0])
        assert not np.isinf(features['home_goal_ratio'].iloc[0])
