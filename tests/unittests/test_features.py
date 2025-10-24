import unittest
import pandas as pd
import numpy as np
from src.features import Features


class TestFeaturesCanAddSeasonalityAndManageColumns(unittest.TestCase):
    """
    Unit tests for Features class in src.features module.
    Tests cover sin and cos seasonality additions, column selection, column dropping and data preparation.
    """

    def test_add_sin_seasonality_values(self):
        """
        Checks that add_sin_seasonality adds correct sin values.
        """
        df = pd.DataFrame({'dummy': [0, 1, 2, 3]})
        features = Features(df.copy()).add_sin_seasonality(period=4).prepare_data()
        expected = np.sin(2 * np.pi * np.arange(4) / 4)
        actual = features['sin_season'].to_numpy()
        self.assertTrue(np.allclose(actual, expected), "Sin seasonality values are incorrect")

    def test_add_cos_seasonality_values(self):
        """
        Checks that add_cos_seasonality adds correct cos values.
        """
        df = pd.DataFrame({'dummy': [0, 1, 2, 3]})
        features = Features(df.copy()).add_cos_seasonality(period=4).prepare_data()
        expected = np.cos(2 * np.pi * np.arange(4) / 4)
        actual = features['cos_season'].to_numpy()
        self.assertTrue(np.allclose(actual, expected), "Cos seasonality values are incorrect")

    def test_add_sin_seasonality_returns_self(self):
        """
        Checks that add_sin_seasonality returns self for chaining.
        """
        df = pd.DataFrame({'dummy': [0, 1]})
        obj = Features(df.copy())
        self.assertIs(obj.add_sin_seasonality(2), obj, "add_sin_seasonality does not return self")

    def test_add_cos_seasonality_returns_self(self):
        """
        Checks that add_cos_seasonality returns self for chaining.
        """
        df = pd.DataFrame({'dummy': [0, 1]})
        obj = Features(df.copy())
        self.assertIs(obj.add_cos_seasonality(2), obj, "add_cos_seasonality does not return self")

    def test_prepare_data_returns_dataframe(self):
        """
        Checks that prepare_data returns a DataFrame.
        """
        df = pd.DataFrame({'dummy': [5, 6, 7]})
        obj = Features(df.copy())
        obj.add_sin_seasonality(period=3).add_cos_seasonality(period=3)
        result = obj.prepare_data()
        self.assertIsInstance(result, pd.DataFrame, "prepare_data did not return DataFrame")

    def test_prepare_data_contains_sin_season(self):
        """
        Checks that prepare_data contains sin_season column after adding.
        """
        df = pd.DataFrame({'dummy': [5, 6, 7]})
        obj = Features(df.copy())
        obj.add_sin_seasonality(period=3)
        result = obj.prepare_data()
        self.assertIn('sin_season', result.columns, "prepare_data missing sin_season column")

    def test_prepare_data_contains_cos_season(self):
        """
        Checks that prepare_data contains cos_season column after adding.
        """
        df = pd.DataFrame({'dummy': [5, 6, 7]})
        obj = Features(df.copy())
        obj.add_cos_seasonality(period=3)
        result = obj.prepare_data()
        self.assertIn('cos_season', result.columns, "prepare_data missing cos_season column")

    def test_choose_columns_keeps_specified_columns(self):
        """
        Checks that choose_columns keeps only specified columns plus Month.
        """
        df = pd.DataFrame({'Month': [1, 2, 3], 'feature_A': [10, 20, 30],
                          'feature_B': [100, 200, 300], 'feature_C': [5, 15, 25]})
        result = Features(df.copy()).choose_features(['feature_A', 'feature_C']).prepare_data()
        expected_columns = ['Month', 'feature_A', 'feature_C']
        self.assertListEqual(list(result.columns), expected_columns, "choose_columns did not filter columns correctly")

    def test_choose_columns_returns_self(self):
        """
        Checks that choose_columns returns self for chaining.
        """
        df = pd.DataFrame({'Month': [1, 2], 'col_X': [7, 8]})
        obj = Features(df.copy())
        self.assertIs(obj.choose_features(['col_X']), obj, "choose_columns does not return self")

    def test_choose_columns_ignores_nonexistent_columns(self):
        """
        Checks that choose_columns ignores nonexistent columns without error.
        """
        df = pd.DataFrame({'Month': [4, 5, 6], 'existing': [11, 22, 33]})
        result = Features(df.copy()).choose_features(['existing', 'nonexistent']).prepare_data()
        self.assertListEqual(list(result.columns), ['Month', 'existing'], "choose_columns did not handle nonexistent columns properly")

    def test_choose_columns_always_includes_month(self):
        """
        Checks that choose_columns preserves Month column always.
        """
        df = pd.DataFrame({'Month': [7, 8], 'data': [99, 88]})
        result = Features(df.copy()).choose_features(['data']).prepare_data()
        self.assertIn('Month', result.columns, "choose_columns did not preserve Month column")

    def test_choose_columns_with_empty_list_keeps_only_month(self):
        """
        Checks that choose_columns with empty list keeps only Month.
        """
        df = pd.DataFrame({'Month': [9, 10], 'extra': [44, 55]})
        result = Features(df.copy()).choose_features([]).prepare_data()
        self.assertListEqual(list(result.columns), ['Month'], "choose_columns with empty list did not keep only Month")

    def test_drop_columns_removes_specified_columns(self):
        """
        Checks that drop_columns removes specified columns from DataFrame.
        """
        df = pd.DataFrame({'Month': [1, 2, 3], 'feature_A': [10, 20, 30],
                          'feature_B': [100, 200, 300], 'feature_C': [5, 15, 25]})
        result = Features(df.copy()).drop_features(['feature_A', 'feature_C']).prepare_data()
        expected_columns = ['Month', 'feature_B']
        self.assertListEqual(list(result.columns), expected_columns, "drop_columns did not remove columns correctly")

    def test_drop_columns_returns_self(self):
        """
        Checks that drop_columns returns self for chaining.
        """
        df = pd.DataFrame({'Month': [1, 2], 'col_X': [7, 8], 'col_Y': [9, 10]})
        obj = Features(df.copy())
        self.assertIs(obj.drop_features(['col_X']), obj, "drop_columns does not return self")

    def test_drop_columns_ignores_nonexistent_columns(self):
        """
        Checks that drop_columns ignores nonexistent columns without error.
        """
        df = pd.DataFrame({'Month': [4, 5, 6], 'existing': [11, 22, 33]})
        result = Features(df.copy()).drop_features(['nonexistent']).prepare_data()
        expected_columns = ['Month', 'existing']
        self.assertListEqual(list(result.columns), expected_columns, "drop_columns did not handle nonexistent columns properly")

    def test_drop_columns_with_empty_list_keeps_all_columns(self):
        """
        Checks that drop_columns with empty list keeps all columns.
        """
        df = pd.DataFrame({'Month': [9, 10], 'extra': [44, 55]})
        result = Features(df.copy()).drop_features([]).prepare_data()
        expected_columns = ['Month', 'extra']
        self.assertListEqual(list(result.columns), expected_columns, "drop_columns with empty list did not keep all columns")

    def test_drop_columns_can_drop_month_column(self):
        """
        Checks that drop_columns can remove Month column if specified.
        """
        df = pd.DataFrame({'Month': [1, 2], 'data': [99, 88]})
        result = Features(df.copy()).drop_features(['Month']).prepare_data()
        self.assertNotIn('Month', result.columns, "drop_columns did not remove Month column")

    def test_drop_columns_with_multiple_columns(self):
        """
        Checks that drop_columns can remove multiple columns at once.
        """
        df = pd.DataFrame({'Month': [1, 2, 3], 'A': [10, 20, 30], 'B': [100, 200, 300], 'C': [5, 15, 25], 'D': [7, 8, 9]})
        result = Features(df.copy()).drop_features(['A', 'C', 'D']).prepare_data()
        expected_columns = ['Month', 'B']
        self.assertListEqual(list(result.columns), expected_columns, "drop_columns did not remove multiple columns correctly")


if __name__ == "__main__":
    unittest.main()
