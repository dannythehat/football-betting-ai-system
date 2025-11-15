"""
Backtesting Utilities
Helper functions for backtesting and performance evaluation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


def calculate_roi(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    odds: np.ndarray,
    stake: float = 1.0
) -> Dict[str, float]:
    """
    Calculate Return on Investment (ROI) for betting strategy
    
    Args:
        predictions: Binary predictions (1 = bet, 0 = no bet)
        outcomes: Actual outcomes (1 = win, 0 = loss)
        odds: Decimal odds for each bet
        stake: Stake per bet
        
    Returns:
        Dictionary with ROI metrics
    """
    # Filter to only bets placed
    bet_mask = predictions == 1
    bet_outcomes = outcomes[bet_mask]
    bet_odds = odds[bet_mask]
    
    if len(bet_outcomes) == 0:
        return {
            'total_bets': 0,
            'total_staked': 0.0,
            'total_return': 0.0,
            'profit': 0.0,
            'roi': 0.0,
            'win_rate': 0.0
        }
    
    # Calculate returns
    total_staked = len(bet_outcomes) * stake
    returns = bet_outcomes * bet_odds * stake
    total_return = returns.sum()
    profit = total_return - total_staked
    roi = (profit / total_staked) * 100 if total_staked > 0 else 0.0
    win_rate = bet_outcomes.mean()
    
    return {
        'total_bets': int(len(bet_outcomes)),
        'total_staked': float(total_staked),
        'total_return': float(total_return),
        'profit': float(profit),
        'roi': float(roi),
        'win_rate': float(win_rate)
    }


def calculate_kelly_stake(
    probability: float,
    odds: float,
    bankroll: float,
    kelly_fraction: float = 0.25
) -> float:
    """
    Calculate Kelly Criterion stake size
    
    Args:
        probability: Estimated probability of winning
        odds: Decimal odds
        bankroll: Current bankroll
        kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly)
        
    Returns:
        Stake size
    """
    # Kelly formula: f = (bp - q) / b
    # where b = odds - 1, p = probability, q = 1 - p
    b = odds - 1
    p = probability
    q = 1 - p
    
    kelly = (b * p - q) / b
    
    # Apply fractional Kelly
    kelly = kelly * kelly_fraction
    
    # Ensure non-negative and cap at bankroll
    kelly = max(0, min(kelly, 1.0))
    
    return kelly * bankroll


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio for betting returns
    
    Args:
        returns: Array of returns per bet
        risk_free_rate: Risk-free rate (default 0)
        
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / returns.std()


def calculate_max_drawdown(cumulative_returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown from cumulative returns
    
    Args:
        cumulative_returns: Array of cumulative returns
        
    Returns:
        Maximum drawdown as percentage
    """
    if len(cumulative_returns) == 0:
        return 0.0
    
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return float(max_drawdown * 100)


def walk_forward_split(
    df: pd.DataFrame,
    initial_train_months: int,
    step_months: int,
    date_column: str = 'date'
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create walk-forward train/test splits
    
    Args:
        df: DataFrame with date column
        initial_train_months: Number of months for initial training
        step_months: Number of months to step forward
        date_column: Name of date column
        
    Returns:
        List of (train_df, test_df) tuples
    """
    df = df.sort_values(date_column).reset_index(drop=True)
    df[date_column] = pd.to_datetime(df[date_column])
    
    min_date = df[date_column].min()
    max_date = df[date_column].max()
    
    splits = []
    current_train_end = min_date + pd.DateOffset(months=initial_train_months)
    
    while current_train_end < max_date:
        test_end = current_train_end + pd.DateOffset(months=step_months)
        
        train_df = df[df[date_column] < current_train_end]
        test_df = df[
            (df[date_column] >= current_train_end) & 
            (df[date_column] < test_end)
        ]
        
        if len(train_df) > 0 and len(test_df) > 0:
            splits.append((train_df, test_df))
        
        current_train_end = test_end
    
    return splits


def print_backtest_summary(results: Dict):
    """
    Print formatted backtest summary
    
    Args:
        results: Dictionary with backtest results
    """
    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)
    print(f"Total Periods:  {results.get('total_periods', 0)}")
    print(f"Total Bets:     {results.get('total_bets', 0):,}")
    print(f"Win Rate:       {results.get('win_rate', 0):.2%}")
    print(f"Total Staked:   ${results.get('total_staked', 0):,.2f}")
    print(f"Total Return:   ${results.get('total_return', 0):,.2f}")
    print(f"Profit:         ${results.get('profit', 0):,.2f}")
    print(f"ROI:            {results.get('roi', 0):.2f}%")
    
    if 'sharpe_ratio' in results:
        print(f"Sharpe Ratio:   {results['sharpe_ratio']:.3f}")
    if 'max_drawdown' in results:
        print(f"Max Drawdown:   {results['max_drawdown']:.2f}%")
    
    print("=" * 60)
