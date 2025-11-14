"""
Training Script for Smart Bets AI
Trains models on historical match data
"""

import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sqlalchemy import create_engine

from smart_bets_ai.feature_engineering import prepare_training_data
from smart_bets_ai.model_trainer import SmartBetsModelTrainer, print_training_summary


def load_data_from_db(db_url: str) -> tuple:
    """
    Load training data from database
    
    Args:
        db_url: Database connection URL
        
    Returns:
        Tuple of (matches_df, results_df)
    """
    engine = create_engine(db_url)
    
    print("Loading data from database...")
    
    # Load matches with stats
    matches_query = """
        SELECT 
            m.*,
            ht.team_name as home_team_name,
            at.team_name as away_team_name
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.team_id
        JOIN teams at ON m.away_team_id = at.team_id
        WHERE m.status = 'completed'
        ORDER BY m.match_datetime
    """
    matches_df = pd.read_sql(matches_query, engine)
    
    # Load results
    results_query = """
        SELECT * FROM match_results
    """
    results_df = pd.read_sql(results_query, engine)
    
    print(f"✓ Loaded {len(matches_df)} completed matches")
    print(f"✓ Loaded {len(results_df)} match results")
    
    return matches_df, results_df


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Smart Bets AI models')
    parser.add_argument(
        '--db-url',
        type=str,
        default=os.getenv('DATABASE_URL', 'postgresql://user:password@localhost:5432/football_betting'),
        help='Database connection URL'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for testing (default: 0.2)'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory to save trained models (default: models)'
    )
    parser.add_argument(
        '--version',
        type=str,
        default=None,
        help='Model version name (default: timestamp)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("SMART BETS AI - MODEL TRAINING")
    print("="*70 + "\n")
    
    # Load data
    matches_df, results_df = load_data_from_db(args.db_url)
    
    # Prepare training data
    print("\nPreparing training data...")
    X, targets, feature_engineer = prepare_training_data(matches_df, results_df)
    
    print(f"✓ Created {len(X.columns)} features")
    print(f"✓ Created {len(targets)} target variables")
    print(f"✓ Training samples: {len(X)}")
    
    # Initialize trainer
    trainer = SmartBetsModelTrainer(model_dir=args.model_dir)
    
    # Train models
    print("\n" + "="*70)
    print("TRAINING MODELS")
    print("="*70 + "\n")
    
    metrics = trainer.train_all_markets(
        X=X,
        targets=targets,
        test_size=args.test_size
    )
    
    # Print summary
    print_training_summary(metrics)
    
    # Save models
    print("Saving models...")
    version_dir = trainer.save_models(version=args.version)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nModels saved to: {version_dir}")
    print("\nNext steps:")
    print("  1. Review training metrics above")
    print("  2. Test predictions with the predictor module")
    print("  3. Integrate with API endpoints")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
