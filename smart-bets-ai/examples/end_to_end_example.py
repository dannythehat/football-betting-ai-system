"""
End-to-End Example: Training and Using Smart Bets AI

This example demonstrates the complete workflow:
1. Load historical data from database
2. Train Smart Bets AI models
3. Generate predictions for upcoming matches
4. Analyze custom bets
"""

import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime

# Import Smart Bets AI modules
from smart_bets_ai.feature_engineering import prepare_training_data
from smart_bets_ai.model_trainer import SmartBetsModelTrainer, print_training_summary
from smart_bets_ai.predictor import SmartBetsPredictor


def example_1_train_models():
    """
    Example 1: Train Smart Bets AI models on historical data
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: TRAINING SMART BETS AI MODELS")
    print("="*70 + "\n")
    
    # Database connection
    db_url = "postgresql://user:password@localhost:5432/football_betting"
    engine = create_engine(db_url)
    
    # Load historical matches
    print("Loading historical data...")
    matches_query = """
        SELECT m.* 
        FROM matches m
        WHERE m.status = 'completed'
        ORDER BY m.match_datetime
    """
    matches_df = pd.read_sql(matches_query, engine)
    
    # Load results
    results_query = "SELECT * FROM match_results"
    results_df = pd.read_sql(results_query, engine)
    
    print(f"✓ Loaded {len(matches_df)} completed matches")
    print(f"✓ Loaded {len(results_df)} match results\n")
    
    # Prepare training data
    print("Preparing training data...")
    X, targets, feature_engineer = prepare_training_data(matches_df, results_df)
    
    print(f"✓ Created {len(X.columns)} features")
    print(f"✓ Created {len(targets)} target variables")
    print(f"✓ Training samples: {len(X)}\n")
    
    # Initialize trainer
    trainer = SmartBetsModelTrainer(model_dir='models')
    
    # Train models
    print("Training models...\n")
    metrics = trainer.train_all_markets(
        X=X,
        targets=targets,
        test_size=0.2,
        random_state=42
    )
    
    # Print summary
    print_training_summary(metrics)
    
    # Save models
    version_dir = trainer.save_models(version='v1.0')
    
    print(f"\n✓ Models saved to: {version_dir}")
    print("\nTraining complete! Models ready for predictions.\n")
    
    return trainer


def example_2_single_match_prediction():
    """
    Example 2: Generate Smart Bet prediction for a single match
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: SINGLE MATCH PREDICTION")
    print("="*70 + "\n")
    
    # Load trained models
    print("Loading trained models...")
    trainer = SmartBetsModelTrainer()
    trainer.load_models(version='v1.0')
    
    # Initialize predictor
    predictor = SmartBetsPredictor(trainer)
    print("✓ Predictor initialized\n")
    
    # Sample match data
    match_data = pd.DataFrame([{
        'match_id': 'EPL_2025_001',
        'home_goals_avg': 1.8,
        'away_goals_avg': 1.3,
        'home_goals_conceded_avg': 0.7,
        'away_goals_conceded_avg': 1.5,
        'home_corners_avg': 6.2,
        'away_corners_avg': 4.5,
        'home_cards_avg': 2.0,
        'away_cards_avg': 2.8,
        'home_btts_rate': 0.65,
        'away_btts_rate': 0.55,
        'home_form': 'WWWDW',  # Strong home form
        'away_form': 'LWDLL'   # Poor away form
    }])
    
    print("Match Statistics:")
    print(f"  Home Team: Strong form (WWWDW), 1.8 goals/game, 0.7 conceded/game")
    print(f"  Away Team: Poor form (LWDLL), 1.3 goals/game, 1.5 conceded/game\n")
    
    # Generate prediction
    print("Generating prediction...")
    prediction = predictor.predict_match(match_data)
    
    # Display Smart Bet
    smart_bet = prediction['smart_bet']
    print("\n" + "-"*70)
    print("SMART BET RECOMMENDATION")
    print("-"*70)
    print(f"Market: {smart_bet['market_name']}")
    print(f"Probability: {smart_bet['percentage']}")
    print(f"Confidence: {smart_bet['confidence'].upper()}")
    
    print("\nAlternative Markets:")
    for i, alt in enumerate(smart_bet['alternative_markets'], 1):
        print(f"  {i}. {alt['market_name']}: {alt['percentage']}")
    
    print("\n" + "="*70 + "\n")
    
    return prediction


def example_3_batch_predictions():
    """
    Example 3: Generate predictions for multiple upcoming matches
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: BATCH PREDICTIONS FOR UPCOMING MATCHES")
    print("="*70 + "\n")
    
    # Load trained models
    trainer = SmartBetsModelTrainer()
    trainer.load_models(version='v1.0')
    predictor = SmartBetsPredictor(trainer)
    
    # Sample upcoming matches
    upcoming_matches = pd.DataFrame([
        {
            'match_id': 'EPL_2025_001',
            'home_goals_avg': 1.8, 'away_goals_avg': 1.3,
            'home_goals_conceded_avg': 0.7, 'away_goals_conceded_avg': 1.5,
            'home_corners_avg': 6.2, 'away_corners_avg': 4.5,
            'home_cards_avg': 2.0, 'away_cards_avg': 2.8,
            'home_btts_rate': 0.65, 'away_btts_rate': 0.55,
            'home_form': 'WWWDW', 'away_form': 'LWDLL'
        },
        {
            'match_id': 'EPL_2025_002',
            'home_goals_avg': 1.2, 'away_goals_avg': 1.1,
            'home_goals_conceded_avg': 1.3, 'away_goals_conceded_avg': 1.2,
            'home_corners_avg': 4.8, 'away_corners_avg': 5.0,
            'home_cards_avg': 2.5, 'away_cards_avg': 2.3,
            'home_btts_rate': 0.58, 'away_btts_rate': 0.52,
            'home_form': 'DWDLD', 'away_form': 'DLDWD'
        },
        {
            'match_id': 'EPL_2025_003',
            'home_goals_avg': 2.1, 'away_goals_avg': 1.8,
            'home_goals_conceded_avg': 1.4, 'away_goals_conceded_avg': 1.3,
            'home_corners_avg': 7.5, 'away_corners_avg': 6.8,
            'home_cards_avg': 2.2, 'away_cards_avg': 2.6,
            'home_btts_rate': 0.72, 'away_btts_rate': 0.68,
            'home_form': 'WWLWW', 'away_form': 'WDWLW'
        }
    ])
    
    print(f"Processing {len(upcoming_matches)} upcoming matches...\n")
    
    # Generate batch predictions
    predictions = predictor.predict_batch(upcoming_matches)
    
    # Display results
    print("-"*70)
    print("SMART BET RECOMMENDATIONS")
    print("-"*70)
    
    for i, pred in enumerate(predictions, 1):
        smart_bet = pred['smart_bet']
        print(f"\nMatch {i} ({pred['match_id']}):")
        print(f"  Smart Bet: {smart_bet['market_name']}")
        print(f"  Probability: {smart_bet['percentage']}")
        print(f"  Confidence: {smart_bet['confidence'].upper()}")
    
    print("\n" + "="*70 + "\n")
    
    return predictions


def example_4_custom_bet_analysis():
    """
    Example 4: Analyze a user-selected custom bet
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: CUSTOM BET ANALYSIS")
    print("="*70 + "\n")
    
    # Load trained models
    trainer = SmartBetsModelTrainer()
    trainer.load_models(version='v1.0')
    predictor = SmartBetsPredictor(trainer)
    
    # Match data
    match_data = pd.DataFrame([{
        'match_id': 'EPL_2025_001',
        'home_goals_avg': 1.8,
        'away_goals_avg': 1.3,
        'home_goals_conceded_avg': 0.7,
        'away_goals_conceded_avg': 1.5,
        'home_corners_avg': 6.2,
        'away_corners_avg': 4.5,
        'home_cards_avg': 2.0,
        'away_cards_avg': 2.8,
        'home_btts_rate': 0.65,
        'away_btts_rate': 0.55,
        'home_form': 'WWWDW',
        'away_form': 'LWDLL'
    }])
    
    # User wants to bet on "Over 2.5 Goals"
    print("User's bet selection: Over 2.5 Goals")
    print("Analyzing...\n")
    
    analysis = predictor.analyze_custom_bet(match_data, 'over_2_5')
    
    # Display analysis
    print("-"*70)
    print("CUSTOM BET ANALYSIS")
    print("-"*70)
    print(f"Bet: {analysis['bet_name']}")
    print(f"Probability: {analysis['percentage']}")
    print(f"Verdict: {analysis['verdict'].upper()}")
    print(f"Confidence: {analysis['confidence'].upper()}")
    
    print("\nComparison with Smart Bet:")
    comp = analysis['smart_bet_comparison']
    print(f"  Smart Bet: {comp['smart_bet_market']}")
    print(f"  Smart Bet Probability: {comp['smart_bet_percentage']}")
    print(f"  Difference: {comp['probability_difference']*100:.1f}%")
    
    if 'note' in analysis:
        print(f"\nNote: {analysis['note']}")
    
    print("\n" + "="*70 + "\n")
    
    return analysis


def example_5_model_performance():
    """
    Example 5: View model performance metrics
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: MODEL PERFORMANCE METRICS")
    print("="*70 + "\n")
    
    # Load trained models
    trainer = SmartBetsModelTrainer()
    trainer.load_models(version='v1.0')
    
    # Get model summary
    summary = trainer.get_model_summary()
    
    print(f"Total Models: {summary['num_models']}")
    print(f"Total Features: {summary['num_features']}")
    print(f"\nMarkets: {', '.join(summary['markets'][:5])}...\n")
    
    print("-"*70)
    print("PERFORMANCE METRICS")
    print("-"*70)
    print(f"{'Market':<25} {'Accuracy':<12} {'ROC-AUC':<12} {'Log Loss':<12}")
    print("-"*70)
    
    for market, metrics in list(summary['metrics_summary'].items())[:10]:
        print(f"{market:<25} {metrics['accuracy']:<12.3f} {metrics['roc_auc']:<12.3f} {metrics['log_loss']:<12.3f}")
    
    print("\n" + "="*70 + "\n")


def main():
    """
    Run all examples
    """
    print("\n" + "="*70)
    print("SMART BETS AI - END-TO-END EXAMPLES")
    print("="*70)
    
    # Note: Uncomment the example you want to run
    
    # Example 1: Train models (run this first)
    # trainer = example_1_train_models()
    
    # Example 2: Single match prediction
    # prediction = example_2_single_match_prediction()
    
    # Example 3: Batch predictions
    # predictions = example_3_batch_predictions()
    
    # Example 4: Custom bet analysis
    # analysis = example_4_custom_bet_analysis()
    
    # Example 5: Model performance
    # example_5_model_performance()
    
    print("\nTo run examples:")
    print("  1. Uncomment the example you want to run in main()")
    print("  2. Ensure you have trained models (run example_1 first)")
    print("  3. Update database connection string if needed")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
