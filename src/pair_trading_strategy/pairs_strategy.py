"""
Pairs Trading Strategy with Cointegration Analysis

A market-neutral statistical arbitrage strategy that identifies cointegrated 
stock pairs and trades mean reversion using z-score based signals.

Author: tperera
Date: 2025-11-23
License: MIT
"""
import pandas as pd
from statsmodels.tsa.stattools import coint

class PairsTradingStrategy:
    """
    Market-neutral pairs trading using cointegration and mean reversion.
    
    This class implements the core strategy logic without storing data,
    making it reusable across multiple pairs.
    """
    def __init__(self, 
                 lookback_period: int = 60,
                 entry_threshold: float = 2.0,
                 exit_threshold: float = 0.5):
        """
        Initialize strategy with parameters.
        
        Args:
            lookback_period: Window for rolling statistics (default: 60 days)
            entry_threshold: Z-score to enter positions (default: 2.0)
            exit_threshold: Z-score to exit positions (default: 0.5)
        """
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self._validate_parameters()

    def _validate_parameters(self):
        """Validate strategy parameters"""
        if self.entry_threshold <= self.exit_threshold:
            raise ValueError(
                f"Entry threshold ({self.entry_threshold}) must be > "
                f"exit threshold ({self.exit_threshold})"
            )
        if self.lookback_period < 20:
            raise ValueError("Lookback period should be at least 20 days")

    def test_cointegration(self, prices_a, prices_b, significance_level = 0.05):
        """
        Test for cointegration using Engle-Granger method.
        
        Implementation:
        1. Run Ordinary Least Squares regression: prices_b = alpha + beta * prices_a + error
        2. Extract residuals
        3. Perform Augmented Dickey-Fuller test on residuals
        4. Check if p-value < significance_level
        
        Args:
            prices_a: Price series for stock A
            prices_b: Price series for stock B
            significance_level: P-value threshold (default: 0.05)
        
        Returns:
            Tuple of (is_cointegrated: bool, p_value: float)
        """
        # Align series (remove NaN)
        # aligned_a, aligned_b = self._align_series(prices_a, prices_b)
        aligned_a = prices_a
        aligned_b = prices_b
        # Perform cointegration test
        score, pvalue, _ = coint(aligned_a, aligned_b)
        is_cointegrated = pvalue < significance_level
        return is_cointegrated, pvalue


strategy = PairsTradingStrategy()
ko_prices = pd.Series([60.0, 61.5, 62.0])
pep_prices = pd.Series([180.0, 184.5, 186.0])
is_coint, pval = strategy.test_cointegration(ko_prices, pep_prices)
print(f"Cointegrated: {is_coint}, p-value: {pval:.4f}")