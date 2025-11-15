"""
Golden Bets AI Test Script
Tests filtering logic with sample predictions
"""
import numpy as np
from filter import GoldenBetsFilter

def test_golden_bets_filter():
    """Test Golden Bets filtering with sample data"""
    
    # Sample Smart Bets predictions
    smart_predictions = [
        {
            'match_id': '001',
            'home_team': 'Team A',
            'away_team': 'Team B',
            'probability': 0.88,
            'market_name': 'Total Corners',
            'selection_name': 'Over 9.5'
        },
        {
            'match_id': '002',
            'home_team': 'Team C',
            'away_team': 'Team D',
            'probability': 0.92,
            'market_name': 'Total Goals',
            'selection_name': 'Over 2.5'
        },
        {
            'match_id': '003',
            'home_team': 'Team E',
            'away_team': 'Team F',
            'probability': 0.78,  # Below threshold
            'market_name': 'BTTS',
            'selection_name': 'Yes'
        },
        {
            'match_id': '004',
            'home_team': 'Team G',
            'away_team': 'Team H',
            'probability': 0.86,
            'market_name': 'Total Cards',
            'selection_name': 'Over 3.5'
        },
        {
            'match_id': '005',
            'home_team': 'Team I',
            'away_team': 'Team J',
            'probability': 0.90,
            'market_name': 'Total Corners',
            'selection_name': 'Under 9.5'
        }
    ]
    
    # Sample model probabilities for ensemble agreement
    model_probs = {
        '001': np.array([0.87, 0.89, 0.88, 0.87]),  # High agreement
        '002': np.array([0.91, 0.93, 0.92, 0.92]),  # High agreement
        '003': np.array([0.75, 0.80, 0.78, 0.79]),  # Below threshold
        '004': np.array([0.82, 0.90, 0.85, 0.87]),  # Moderate agreement
        '005': np.array([0.88, 0.91, 0.90, 0.91])   # High agreement
    }
    
    # Initialize filter
    filter = GoldenBetsFilter()
    
    print("=" * 60)
    print("GOLDEN BETS AI - TEST RUN")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Confidence Threshold: {filter.confidence_threshold:.0%}")
    print(f"  Min Ensemble Agreement: {filter.min_agreement:.0%}")
    print(f"  Max Daily Picks: {filter.max_picks}")
    
    print(f"\nInput: {len(smart_predictions)} Smart Bets predictions")
    
    # Filter Golden Bets
    golden_bets = filter.filter_golden_bets(
        smart_bets_predictions=smart_predictions,
        model_probabilities=model_probs
    )
    
    print(f"\nOutput: {len(golden_bets)} Golden Bets selected")
    print("\n" + "=" * 60)
    print("GOLDEN BETS RESULTS")
    print("=" * 60)
    
    for i, bet in enumerate(golden_bets, 1):
        print(f"\n🏆 Golden Bet #{i}")
        print(f"  Match: {bet['home_team']} vs {bet['away_team']}")
        print(f"  Market: {bet['market_name']}")
        print(f"  Selection: {bet['selection_name']}")
        print(f"  Confidence: {bet['confidence_score']:.1%}")
        print(f"  Agreement: {bet['ensemble_agreement']:.1%}")
        print(f"  Golden Score: {bet['golden_score']:.3f}")
        print(f"\n  {filter.generate_reasoning(bet)}")
    
    print("\n" + "=" * 60)
    print("FILTERED OUT PREDICTIONS")
    print("=" * 60)
    
    golden_ids = {bet['match_id'] for bet in golden_bets}
    filtered_out = [p for p in smart_predictions if p['match_id'] not in golden_ids]
    
    for pred in filtered_out:
        match_id = pred['match_id']
        prob = pred['probability']
        
        # Calculate agreement if available
        agreement = 1.0
        if match_id in model_probs:
            agreement = filter._calculate_ensemble_agreement(model_probs[match_id])
        
        reason = []
        if prob < filter.confidence_threshold:
            reason.append(f"Low confidence ({prob:.1%} < {filter.confidence_threshold:.0%})")
        if agreement < filter.min_agreement:
            reason.append(f"Low agreement ({agreement:.1%} < {filter.min_agreement:.0%})")
        
        print(f"\n❌ {pred['home_team']} vs {pred['away_team']}")
        print(f"   Market: {pred['market_name']} - {pred['selection_name']}")
        print(f"   Reason: {', '.join(reason)}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    
    return golden_bets

if __name__ == "__main__":
    test_golden_bets_filter()
