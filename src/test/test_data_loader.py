"""
Unit tests for Data loading class for Pairs Trading Strategy.
"""

import unittest
from ..pair_trading_strategy.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):

    def test_setup(self):
        print("----------------------------")
        print("Starting a data loader setup")
        DL = DataLoader('2025-07-25', '2025-08-25', '90m')
        self.assertEqual(DL.start_date, '2025-07-25')
        self.assertEqual(DL.end_date, '2025-08-25')
        self.assertEqual(DL.interval, '90m')

if __name__ == '__main__':
    unittest.main()