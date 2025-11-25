"""
Complete example: Using DataLoader and PairsTradingStrategy together
"""

from data_loader import DataLoader
from pairs_strategy import PairsTradingStrategy
from back_test import Backtester
import pandas as pd
import matplotlib.pyplot as plt

def run_pairs_trading_analysis(ticker_a, ticker_b, start_date, end_date):
    """
    Complete pairs trading workflow from data loading to results.
    
    Args:
        ticker_a: First stock ticker (e.g., 'KO')
        ticker_b: Second stock ticker (e.g., 'PEP')
        start_date: Start date for data (e.g., '2020-01-01')
        end_date: End date for data (e.g., '2024-01-01')
    
    Returns:
        Dictionary with results and metrics
    """
    
    print("=" * 70)
    print(f"PAIRS TRADING ANALYSIS: {ticker_a} vs {ticker_b}")
    print("=" * 70)

    # ============================================================
    # STEP 1: LOAD DATA
    # ============================================================
    print("\n[1/7] Loading data...")
    
    loader = DataLoader(start_date=start_date, end_date=end_date, interval='1d')
    
    try:
        # Fetch data for both stocks
        prices_clean = loader.fetch_data([ticker_a, ticker_b])
        
        # Preprocess (handle missing values, outliers)
        #prices_clean = loader.preprocess_data(prices)
        
        print(f"  ✓ Loaded {len(prices_clean)} days of data")
        print(f"  ✓ Date range: {prices_clean.index[0].date()} to {prices_clean.index[-1].date()}")
        
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        return None
    
    # Extract individual series
    print(prices_clean)
    prices_a = prices_clean['Open'][ticker_a]
    prices_b = prices_clean['Open'][ticker_b]

    print(prices_a)

    # ============================================================
    # STEP 2: INITIALIZE STRATEGY
    # ============================================================
    print("\n[2/7] Initializing strategy...")
    
    strategy = PairsTradingStrategy(
        lookback_period=60,
        entry_threshold=2.0,
        exit_threshold=0.5
    )
    
    print(f"  ✓ Lookback period: {strategy.lookback_period} days")
    print(f"  ✓ Entry threshold: ±{strategy.entry_threshold} std")
    print(f"  ✓ Exit threshold: ±{strategy.exit_threshold} std")

    # ============================================================
    # STEP 3: TEST FOR COINTEGRATION
    # ============================================================
    print("\n[3/7] Testing for cointegration...")
    
    is_cointegrated, pvalue = strategy.test_cointegration(prices_a, prices_b)
    
    print(f"  {'✓' if is_cointegrated else '✗'} Cointegrated: {is_cointegrated}")
    print(f"    P-value: {pvalue:.4f}")
    
    if not is_cointegrated:
        print("\n  ⚠ WARNING: Stocks are not cointegrated!")
        print("    This pair may not be suitable for pairs trading.")
        print("    Consider trying a different pair.\n")
        # You can choose to continue or return here   
    # ============================================================
    # STEP 4: CALCULATE HEDGE RATIO AND SPREAD
    # ============================================================
    print("\n[4/7] Calculating hedge ratio and spread...")
    
    hedge_ratio = strategy.calculate_hedge_ratio(prices_a, prices_b)
    spread = strategy.calculate_spread(prices_a, prices_b, hedge_ratio)
    
    print(f"  ✓ Hedge ratio: {hedge_ratio:.4f}")
    print(f"    For every $1 in {ticker_a}, hold ${hedge_ratio:.2f} in {ticker_b}")
    print(f"  ✓ Spread mean: ${spread.mean():.2f}")
    print(f"    Spread std: ${spread.std():.2f}")

    # ============================================================
    # STEP 5: CALCULATE Z-SCORE AND GENERATE SIGNALS
    # ============================================================
    print("\n[5/7] Generating trading signals...")
    
    zscore = strategy.calculate_zscore(spread)
    signals = strategy.generate_signals(zscore)
    
    # Count signals
    num_long = (signals == 1).sum()
    num_short = (signals == -1).sum()
    num_flat = (signals == 0).sum()
    num_trades = (signals.diff() != 0).sum()
    
    print(f"  ✓ Z-score calculated")
    print(f"    Valid z-scores: {zscore.notna().sum()}/{len(zscore)}")
    print(f"  ✓ Signals generated")
    print(f"    Long positions: {num_long} days")
    print(f"    Short positions: {num_short} days")
    print(f"    Flat positions: {num_flat} days")
    print(f"    Total trades: {num_trades}")
    
    if num_trades == 0:
        print("\n  ⚠ WARNING: No trades generated!")
        print("    Try adjusting entry_threshold or lookback_period.\n")
        return None
    
    # =============================================================================
    # STEP 7: BACKTEST
    # =============================================================================

    backtester = Backtester(
        initial_capital=100000,
        transaction_cost=0.001
    )

    results = backtester.run_backtest(prices_a, prices_b, signals, hedge_ratio)
    metrics = backtester.calculate_performance_metrics(results['returns'])

    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        if 'Return' in key or 'Drawdown' in key or 'Rate' in key:
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value:.2f}")

    # =============================================================================
    # STEP 8: VISUALIZE
    # =============================================================================

    print(results['equity'])
    backtester.plot_results(
        equity_curve=results['equity'],
        signals=signals,
        spread=spread,
        zscore=zscore,
        prices_a=prices_a,
        prices_b=prices_b
    )    

if __name__ == "__main__":
    # Example 1: Basic usage
    print("\n" + "="*70)
    print("EXAMPLE 1: KO vs PEP")
    print("="*70)
    analysis = run_pairs_trading_analysis(
        ticker_a='SPY',
        ticker_b='IVV',
        start_date='2013-01-01',
        end_date='2015-01-01'
    )