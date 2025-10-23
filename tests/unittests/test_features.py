import unittest
import pandas as pd
import numpy as np
from src.features import Features

"""
Unit tests for Features class in src.rawdata module.
Tests cover sin and cos seasonality additions, column selection and data preparation.
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

    """
    Test that choose_columns keeps only specified columns plus Month.
    """
    def test_choose_columns_keeps_specified_columns(self):
        df = pd.DataFrame({'Month': [1, 2, 3], 'feature_A': [10, 20, 30],
                          'feature_B': [100, 200, 300], 'feature_C': [5, 15, 25]})
        result = Features(df.copy()).choose_columns(['feature_A', 'feature_C']).prepare_data()
        expected_columns = ['Month', 'feature_A', 'feature_C']
        self.assertListEqual(list(result.columns), expected_columns,
                            msg="choose_columns did not filter columns correctly")

    """
    Test that choose_columns returns self for chaining.
    """
    def test_choose_columns_returns_self(self):
        df = pd.DataFrame({'Month': [1, 2], 'col_X': [7, 8]})
        obj = Features(df.copy())
        self.assertIs(obj.choose_columns(['col_X']), obj,
                      msg="choose_columns does not return self")

    """
    Test that choose_columns ignores nonexistent columns without error.
    """
    def test_choose_columns_ignores_nonexistent_columns(self):
        df = pd.DataFrame({'Month': [4, 5, 6], 'existing': [11, 22, 33]})
        result = Features(df.copy()).choose_columns(['existing', 'nonexistent']).prepare_data()
        self.assertListEqual(list(result.columns), ['Month', 'existing'],
                            msg="choose_columns did not handle nonexistent columns properly")

    """
    Test that choose_columns preserves Month column always.
    """
    def test_choose_columns_always_includes_month(self):
        df = pd.DataFrame({'Month': [7, 8], 'data': [99, 88]})
        result = Features(df.copy()).choose_columns(['data']).prepare_data()
        self.assertIn('Month', result.columns,
                     msg="choose_columns did not preserve Month column")

    """
    Test that choose_columns with empty list keeps only Month.
    """
    def test_choose_columns_with_empty_list_keeps_only_month(self):
        df = pd.DataFrame({'Month': [9, 10], 'extra': [44, 55]})
        result = Features(df.copy()).choose_columns([]).prepare_data()
        self.assertListEqual(list(result.columns), ['Month'],
                            msg="choose_columns with empty list did not keep only Month")


if __name__ == "__main__":
    unittest.main()
