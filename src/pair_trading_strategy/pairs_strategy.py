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
    def __init__(self, lookback_period = 60, entry_threshold = 2.0, exit_threshold = 0.5):
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

    def _align_series(self, prices_a, prices_b):
        """Using explicit index intersection"""
        # Find common dates
        common_dates = prices_a.index.intersection(prices_b.index)
        # Select only common dates
        aligned_a = prices_a.loc[common_dates]
        aligned_b = prices_b.loc[common_dates]
        # Drop NaN values
        combined = pd.DataFrame({'a': aligned_a, 'b': aligned_b})
        combined = combined.dropna()
        # return combined['a'], combined['b']
        return prices_a, prices_b

    
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
        aligned_a, aligned_b = self._align_series(prices_a, prices_b)
        # Perform cointegration test
        score, pvalue, _ = coint(aligned_a, aligned_b)
        is_cointegrated = pvalue < significance_level
        return is_cointegrated, pvalue

    def calculate_hedge_ratio(self, prices_a, prices_b):
        """
        Calculate optimal hedge ratio using OLS regression.
        
        Implementation:
        1. Set up regression: prices_b = alpha + beta * prices_a + error
        2. Estimate beta (hedge ratio) using OLS
        3. Return beta
        
        The hedge ratio tells us: for every $1 in stock A,
        hold $beta in stock B to create a market-neutral spread.
        
        Args:
            prices_a: Price series for stock A (independent variable)
            prices_b: Price series for stock B (dependent variable)
        
        Returns:
            Hedge ratio (beta coefficient)
        """
        from scipy import stats
        
        # Align series
        aligned_a, aligned_b = self._align_series(prices_a, prices_b)
        # OLS regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            aligned_a, aligned_b
        )
        
        return slope


    def calculate_spread(self, prices_a, prices_b, hedge_ratio):
        """
        Calculate the spread between two price series.
        
        Spread = prices_b - hedge_ratio * prices_a
        
        This represents the "distance" between the two stocks after
        accounting for their typical relationship.
        
        Args:
            prices_a: Price series for stock A
            prices_b: Price series for stock B
            hedge_ratio: The hedge ratio from regression
        
        Returns:
            Spread time series
        """
        aligned_a, aligned_b = self._align_series(prices_a, prices_b)
        spread = aligned_b - hedge_ratio * aligned_a
        return spread
    
    def calculate_zscore(self, spread):
        """
        Calculate rolling z-score of the spread.
        
        Z-score = (spread - rolling_mean) / rolling_std
        
        This normalizes the spread so we can use consistent thresholds
        (e.g., enter at |z| > 2) regardless of the stock prices.
        
        Args:
            spread: Spread time series
        
        Returns:
            Z-score time series
        """
        rolling_mean = spread.rolling(window=self.lookback_period).mean()
        rolling_std = spread.rolling(window=self.lookback_period).std()
        zscore = (spread - rolling_mean) / rolling_std
        return zscore

    def generate_signals(self, zscore):
        """
        Generate trading signals based on z-score thresholds.
        
        Signal Logic:
        - zscore > entry_threshold: SHORT spread (short B, long A) → signal = -1
        - zscore < -entry_threshold: LONG spread (long B, short A) → signal = 1
        - |zscore| < exit_threshold: CLOSE positions → signal = 0
        - Otherwise: HOLD current position
        
        Args:
            zscore: Z-score time series
        
        Returns:
            Signal series (1: long spread, -1: short spread, 0: no position)
        """
        signals = pd.Series(0, index=zscore.index)
        # Initialize position
        current_position = 0
        for i in range(len(zscore)):
            z = zscore.iloc[i]
            if pd.isna(z):
                signals.iloc[i] = current_position
                continue
            # Entry logic
            if current_position == 0:
                if z > self.entry_threshold:
                    current_position = -1  # Short spread
                elif z < -self.entry_threshold:
                    current_position = 1   # Long spread
            # Exit logic
            elif abs(z) < self.exit_threshold:
                current_position = 0  # Close position
            signals.iloc[i] = current_position
        return signals
    
if __name__ == "__main__":
    #Example use case of PairsTradingStrategy
    strategy = PairsTradingStrategy()
    ko_prices = pd.Series([60.5, 62.0, 65.0])
    pep_prices = pd.Series([180.0, 184.5, 186.0])
    is_coint, pval = strategy.test_cointegration(ko_prices, pep_prices)
    print(f"Cointegrated: {is_coint}, p-value: {pval:.4f}")

    hedge_ratio = strategy.calculate_hedge_ratio(ko_prices, pep_prices)
    print(f"For every $1 in KO, hold ${hedge_ratio:.2f} in PEP")

    spread = strategy.calculate_spread(ko_prices, pep_prices, hedge_ratio)
    print(spread)
    print(spread.describe())

    zscore = strategy.calculate_zscore(spread)
    print(zscore)
    print(f"Current z-score: {zscore.iloc[-1]:.2f}")

    signals = strategy.generate_signals(zscore)
    print(signals)
    print(f"Number of trades: {(signals.diff() != 0).sum()}")