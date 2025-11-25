"""
Back testing for Pairs Trading Strategy.

Author: tperera
Date: 2025-11-23
License: MIT
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Backtester:
    """Runs backtest given strategy, data, and signals"""
    """
    Backtest pairs trading strategy with realistic assumptions.
    
    Includes transaction costs, proper position sizing, and
    comprehensive performance metrics.
    """
    
    def __init__(self, initial_capital = 100000, transaction_cost = 0.001, max_position_size = 0.5, entry_threshold = 2.0, exit_threshold = 0.5):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital (default: $100,000)
            transaction_cost: Cost per trade as decimal (default: 0.1%)
            max_position_size: Max fraction of capital per leg (default: 50%)
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    def run_backtest(self, prices_a, prices_b, signals, hedge_ratio):
        """
        Execute backtest with transaction costs.
        
        Implementation Steps:
        1. Calculate returns for both stocks
        2. Calculate position sizes based on signals
        3. Calculate portfolio returns (long/short)
        4. Apply transaction costs when positions change
        5. Calculate equity curve
        
        Args:
            prices_a: Price series for stock A
            prices_b: Price series for stock B
            signals: Trading signals (1, -1, or 0)
            hedge_ratio: Hedge ratio for position sizing
        
        Returns:
            Dictionary with:
            - 'equity': Portfolio value over time
            - 'returns': Daily returns
            - 'positions': Position sizes
            - 'trades': Entry/exit points
        """
        # Calculate returns
        returns_a = prices_a.pct_change()
        returns_b = prices_b.pct_change()
        
        # Align everything
        aligned_data = pd.DataFrame({
            'signal': signals,
            'return_a': returns_a,
            'return_b': returns_b
        }).dropna()
        
        # Calculate positions (dollar amount in each stock)
        # When signal = 1 (long spread): long B, short A
        # When signal = -1 (short spread): short B, long A
        position_value = self.initial_capital * self.max_position_size
        
        positions_a = -aligned_data['signal'] * position_value
        positions_b = aligned_data['signal'] * position_value * hedge_ratio
        
        # Calculate portfolio returns
        portfolio_returns = (
            positions_a.shift(1) * aligned_data['return_a'] +
            positions_b.shift(1) * aligned_data['return_b']
        ) / self.initial_capital
        
        # Apply transaction costs
        position_changes = aligned_data['signal'].diff().abs()
        transaction_costs = position_changes * self.transaction_cost
        portfolio_returns -= transaction_costs
        
        # Calculate equity curve
        equity = self.initial_capital * (1 + portfolio_returns).cumprod()
        
        # Track trades
        trades = aligned_data['signal'].diff()
        trade_dates = trades[trades != 0].index
        
        return {
            'equity': equity,
            'returns': portfolio_returns,
            'positions_a': positions_a,
            'positions_b': positions_b,
            'trades': trade_dates
        }
    
    def calculate_performance_metrics(self, returns):
        """
        Calculate comprehensive performance metrics.
        
        Metrics:
        - Total Return: Cumulative return over period
        - Annual Return: Annualized return
        - Annual Volatility: Annualized standard deviation
        - Sharpe Ratio: Risk-adjusted return
        - Max Drawdown: Largest peak-to-trough decline
        - Win Rate: Percentage of profitable days
        - Profit Factor: Gross profit / Gross loss
        - Calmar Ratio: Annual return / Max drawdown
        
        Args:
            returns: Daily return series
        
        Returns:
            Dictionary of performance metrics
        """
        # Total return
        total_return = (1 + returns).prod() - 1
        
        # Annualized return (assuming 252 trading days)
        n_days = len(returns)
        n_years = n_days / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1
        
        # Annual volatility
        annual_vol = returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annual_return / annual_vol if annual_vol != 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns)
        
        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Annual Volatility': annual_vol,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Win Rate': win_rate,
            'Profit Factor': profit_factor,
            'Calmar Ratio': calmar_ratio
        }
    
    def plot_results(self, equity_curve, signals, spread, zscore, prices_a, prices_b):
        """
        Create comprehensive visualization of backtest results.
        
        Creates 6 subplots:
        1. Equity curve with drawdowns
        2. Price series for both stocks
        3. Spread with entry/exit signals
        4. Z-score with threshold lines
        5. Returns distribution
        6. Rolling Sharpe ratio
        """

        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Pairs Trading Strategy Performance', fontsize=16, y=1.00)
        
        # 1. Equity curve
        ax = axes[0, 0]
        equity_curve.plot(ax=ax, label='Portfolio Value', linewidth=2)
        ax.set_title('Equity Curve')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add drawdown shading
        cumulative = equity_curve / equity_curve.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        ax.fill_between(equity_curve.index, 
                        equity_curve, 
                        running_max * equity_curve.iloc[0],
                        where=(drawdown < 0),
                        color='red', alpha=0.3, label='Drawdown')
        
        # 2. Price series
        ax = axes[0, 1]
        (prices_a / prices_a.iloc[0]).plot(ax=ax, label='Stock A', linewidth=1.5)
        (prices_b / prices_b.iloc[0]).plot(ax=ax, label='Stock B', linewidth=1.5)
        ax.set_title('Normalized Price Series')
        ax.set_ylabel('Normalized Price')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 3. Spread with signals
        ax = axes[1, 0]
        spread.plot(ax=ax, label='Spread', color='black', linewidth=1)
        
        # Mark entries/exits
        entries = signals[signals.diff() != 0]
        ax.scatter(entries.index, spread[entries.index], 
                c='green', marker='^', s=100, label='Entry/Exit', zorder=5)
        
        ax.set_title('Spread with Trading Signals')
        ax.set_ylabel('Spread Value')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 4. Z-score
        ax = axes[1, 1]
        zscore.plot(ax=ax, label='Z-Score', color='blue', linewidth=1)
        ax.axhline(y=self.entry_threshold, color='r', linestyle='--', 
                label=f'Entry (±{self.entry_threshold})')
        ax.axhline(y=-self.entry_threshold, color='r', linestyle='--')
        ax.axhline(y=self.exit_threshold, color='g', linestyle='--',
                label=f'Exit (±{self.exit_threshold})')
        ax.axhline(y=-self.exit_threshold, color='g', linestyle='--')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_title('Z-Score with Thresholds')
        ax.set_ylabel('Z-Score')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 5. Returns distribution
        ax = axes[2, 0]
        returns = equity_curve.pct_change().dropna()
        sns.histplot(returns, bins=50, ax=ax, kde=True)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.set_title('Returns Distribution')
        ax.set_xlabel('Daily Returns')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # 6. Rolling Sharpe
        ax = axes[2, 1]
        rolling_sharpe = (
            returns.rolling(60).mean() / returns.rolling(60).std() * np.sqrt(252)
        )
        rolling_sharpe.plot(ax=ax, linewidth=2)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1')
        ax.set_title('Rolling 60-Day Sharpe Ratio')
        ax.set_ylabel('Sharpe Ratio')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        # plt.savefig('results/backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()