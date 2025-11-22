"""
Data loading for Pairs Trading Strategy.

Author: tperera
Date: 2025-11-23
License: MIT
"""
import yfinance as yf
import pandas as pd
from datetime import datetime

class DataLoader:
    """Handles data fetching and preprocessing"""
    def __init__(self, start_date, end_date, interval):
        """
        Initialize data loader with date range.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
        """
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self._validate_inputs()

    def _validate_inputs(self):
        """Ensure date range is valid"""
        valid_intervals = {'1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'}
        intraday_intervals = {'1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h'}

        start_date_obj = pd.to_datetime(self.start_date, format='%Y-%m-%d')
        end_date_obj = pd.to_datetime(self.end_date, format='%Y-%m-%d')

        if start_date_obj >= end_date_obj:
            raise ValueError("Start date must be before end date")
        if end_date_obj > pd.Timestamp.now():
            raise ValueError("End date cannot be in the future")
        if self.interval not in valid_intervals:
            raise ValueError("Invalid interval")
        if self.interval in intraday_intervals:
            duration = end_date_obj - start_date_obj
            if duration.days > 60:
                raise ValueError("For intra-day intervals, duration cannot exceed 60 days.")

        
    def fetch_data(self, tickers):
        """Fetch historical data for list of tickers"""
        data = yf.download(tickers, start=self.start_date, end=self.end_date, interval=self.interval)
        return data    