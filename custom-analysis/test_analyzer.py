"""
Test Custom Bet Analyzer
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from custom_analysis import CustomBetAnalyzer


def test_custom_analysis():
    """Test custom bet analysis with sample data"""
    
    print("=" * 60)
    print("CUSTOM BET ANALYSIS TEST")
    print("=" * 60)
    
    # Initialize analyzer
    print("\n1. Initializing Custom Bet Analyzer...")
    try:
        analyzer = CustomBetAnalyzer()
        print("✅ Analyzer initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Test match data
    match_data = {
        "match_id": "test_12345",
        "home_team": "Manchester United",
        "away_team": "Liverpool",
        "home_goals_avg": 1.8,
        "away_goals_avg": 1.6,
        "home_goals_conceded_avg": 1.0,
        "away_goals_conceded_avg": 1.2,
        "home_corners_avg": 6.2,
        "away_corners_avg": 5.8,
        "home_cards_avg": 2.3,
        "away_cards_avg": 2.1,
        "home_btts_rate": 0.65,
        "away_btts_rate": 0.60,
        "home_form": "WWDWL",
        "away_form": "WDWWL"
    }
    
    # Test cases
    test_cases = [
        {
            "name": "Over 2.5 Goals",
            "market_id": "total_goals",
            "selection_id": "over_2.5"
        },
        {
            "name": "Under 2.5 Goals",
            "market_id": "total_goals",
            "selection_id": "under_2.5"
        },
        {
            "name": "Over 9.5 Corners",
            "market_id": "total_corners",
            "selection_id": "over_9.5"
        },
        {
            "name": "BTTS Yes",
            "market_id": "btts",
            "selection_id": "yes"
        },
        {
            "name": "Over 3.5 Cards",
            "market_id": "total_cards",
            "selection_id": "over_3.5"
        }
    ]
    
    print(f"\n2. Testing {len(test_cases)} different bet selections...")
    print(f"   Match: {match_data['home_team']} vs {match_data['away_team']}")
    print()
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 60}")
        print(f"Test Case {i}: {test_case['name']}")
        print(f"{'=' * 60}")
        
        try:
            result = analyzer.analyze_custom_bet(
                match_data=match_data,
                market_id=test_case['market_id'],
                selection_id=test_case['selection_id']
            )
            
            # Display results
            print(f"\n📊 Analysis Results:")
            print(f"   Market: {result['user_selection']['market_name']}")
            print(f"   Selection: {result['user_selection']['selection_name']}")
            print(f"   Probability: {result['analysis']['percentage']}")
            print(f"   Confidence: {result['analysis']['confidence_level'].upper()}")
            print(f"   Verdict: {result['analysis']['verdict']}")
            
            print(f"\n💡 Explanation:")
            print(f"   {result['analysis']['explanation']}")
            
            print(f"\n🔍 Comparison:")
            print(f"   {result['analysis']['comparison']}")
            
            if 'smart_bet_alternative' in result:
                alt = result['smart_bet_alternative']
                print(f"\n✨ Smart Bet Alternative:")
                print(f"   {alt['selection_name']} ({alt['percentage']})")
            
            results.append({
                'test': test_case['name'],
                'probability': result['analysis']['probability'],
                'verdict': result['analysis']['verdict'],
                'success': True
            })
            
            print(f"\n✅ Test passed")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            results.append({
                'test': test_case['name'],
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print(f"\n\n{'=' * 60}")
    print("TEST SUMMARY")
    print(f"{'=' * 60}")
    
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"\nTests Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed successfully!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
    
    print("\n" + "=" * 60)
    
    # Display probability rankings
    successful_results = [r for r in results if r['success']]
    if successful_results:
        print("\n📈 Probability Rankings:")
        sorted_results = sorted(
            successful_results,
            key=lambda x: x['probability'],
            reverse=True
        )
        for i, r in enumerate(sorted_results, 1):
            print(f"   {i}. {r['test']}: {r['probability']:.1%} ({r['verdict']})")
    
    print("\n" + "=" * 60)


def test_invalid_inputs():
    """Test error handling with invalid inputs"""
    
    print("\n\n" + "=" * 60)
    print("INVALID INPUT TESTS")
    print("=" * 60)
    
    analyzer = CustomBetAnalyzer()
    
    match_data = {
        "match_id": "test_12345",
        "home_team": "Team A",
        "away_team": "Team B",
        "home_goals_avg": 1.5,
        "away_goals_avg": 1.2,
        "home_goals_conceded_avg": 1.0,
        "away_goals_conceded_avg": 1.1,
        "home_corners_avg": 5.0,
        "away_corners_avg": 4.5,
        "home_cards_avg": 2.0,
        "away_cards_avg": 1.8,
        "home_btts_rate": 0.5,
        "away_btts_rate": 0.45
    }
    
    # Test invalid market
    print("\n1. Testing invalid market...")
    try:
        analyzer.analyze_custom_bet(
            match_data=match_data,
            market_id="invalid_market",
            selection_id="over_2.5"
        )
        print("❌ Should have raised ValueError")
    except ValueError as e:
        print(f"✅ Correctly raised ValueError: {e}")
    
    # Test invalid selection
    print("\n2. Testing invalid selection...")
    try:
        analyzer.analyze_custom_bet(
            match_data=match_data,
            market_id="total_goals",
            selection_id="invalid_selection"
        )
        print("❌ Should have raised ValueError")
    except ValueError as e:
        print(f"✅ Correctly raised ValueError: {e}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Run tests
    test_custom_analysis()
    test_invalid_inputs()
    
    print("\n✅ All tests completed!")
