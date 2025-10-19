import unittest
import pandas as pd
import numpy as np
from src.features import Features

"""
Unit tests for Features class in src.rawdata module.
Tests cover sin and cos seasonality additions and data preparation.
"""

class TestFeatures(unittest.TestCase):

    """
    Test that add_sin_seasonality adds correct sin values.
    """
    def test_add_sin_seasonality_values(self):
        df = pd.DataFrame({'dummy': [0, 1, 2, 3]})
        features = Features(df.copy()).add_sin_seasonality(period=4).prepare_data()
        expected = np.sin(2 * np.pi * np.arange(4) / 4)
        actual = features['sin_season'].to_numpy()
        self.assertTrue(np.allclose(actual, expected),
                        msg="Sin seasonality values are incorrect")

    """
    Test that add_cos_seasonality adds correct cos values.
    """
    def test_add_cos_seasonality_values(self):
        df = pd.DataFrame({'dummy': [0, 1, 2, 3]})
        features = Features(df.copy()).add_cos_seasonality(period=4).prepare_data()
        expected = np.cos(2 * np.pi * np.arange(4) / 4)
        actual = features['cos_season'].to_numpy()
        self.assertTrue(np.allclose(actual, expected),
                        msg="Cos seasonality values are incorrect")

    """
    Test that methods return self for chaining.
    """
    def test_methods_return_self(self):
        df = pd.DataFrame({'dummy': [0, 1]})
        obj = Features(df.copy())
        self.assertIs(obj.add_sin_seasonality(2), obj,
                      msg="add_sin_seasonality does not return self")
        self.assertIs(obj.add_cos_seasonality(2), obj,
                      msg="add_cos_seasonality does not return self")

    """
    Test that prepare_data returns the modified DataFrame.
    """
    def test_prepare_data_returns_dataframe(self):
        df = pd.DataFrame({'dummy': [5, 6, 7]})
        obj = Features(df.copy())
        obj.add_sin_seasonality(period=3).add_cos_seasonality(period=3)
        result = obj.prepare_data()
        self.assertIsInstance(result, pd.DataFrame,
                              msg="prepare_data did not return DataFrame")
        self.assertIn('sin_season', result.columns,
                      msg="prepare_data missing sin_season column")
        self.assertIn('cos_season', result.columns,
                      msg="prepare_data missing cos_season column")


if __name__ == "__main__":
    unittest.main()