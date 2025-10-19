import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.rawdata import RawData

"""
Unit tests for RawData class in demandpredictor.raw_data module.
Tests cover public and private methods without shared state between tests.
"""

class TestRawData(unittest.TestCase):

    """
    Test that make_features first lag value matches first original value.
    """
    def test_make_features_first_lag_value_matches_first_original(self):
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, 'тест_данные_①.csv')
        frame = pd.DataFrame({
            'Month': ['2020-01-01', '2020-02-01', '2020-03-01'],
            'Deals': [17, 23, 31],
            'Value': [137, 211, 307]
        })
        frame.to_csv(path, index=False)
        raw = RawData(path)
        result = raw.make_features()
        os.remove(path)
        os.rmdir(directory)
        self.assertEqual(result.at[0, 'Deals lag1'], 17, msg="First lag value does not match first original")

    """
    Test that make_features last lag value matches last original value.
    """
    def test_make_features_last_lag_value_matches_last_original(self):
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, 'тест_данные_②.csv')
        frame = pd.DataFrame({
            'Month': ['2020-01-01', '2020-02-01', '2020-03-01'],
            'Deals': [19, 29, 41],
            'Value': [149, 223, 313]
        })
        frame.to_csv(path, index=False)
        raw = RawData(path)
        result = raw.make_features()
        os.remove(path)
        os.rmdir(directory)
        self.assertEqual(result.at[2, 'Deals lag1'], 41, msg="Last lag value does not match last original")

    """
    Test that make_features returns DataFrame with Month, Deals lag1, and Value lag1.
    """
    def test_make_features_contains_expected_columns(self):
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, 'тест_данные_③.csv')
        frame = pd.DataFrame({
            'Month': ['2020-01-01', '2020-02-01', '2020-03-01'],
            'Deals': [13, 27, 43],
            'Value': [131, 227, 317]
        })
        frame.to_csv(path, index=False)
        raw = RawData(path)
        result = raw.make_features()
        os.remove(path)
        os.rmdir(directory)
        expected = {'Month', 'Deals lag1', 'Value lag1'}
        self.assertTrue(expected.issubset(result.columns), msg="make_features missing expected columns")

    """
    Test that make_features returns correct number of rows.
    """
    def test_make_features_row_count_is_correct(self):
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, 'тест_данные_④.csv')
        frame = pd.DataFrame({
            'Month': ['2020-01-01', '2020-02-01', '2020-03-01'],
            'Deals': [11, 25, 37],
            'Value': [127, 229, 331]
        })
        frame.to_csv(path, index=False)
        raw = RawData(path)
        result = raw.make_features()
        os.remove(path)
        os.rmdir(directory)
        self.assertEqual(result.shape[0], 3, msg="make_features row count is incorrect")

    """
    Test that target returns DataFrame with Month and target columns.
    """
    def test_target_returns_dataframe_with_month_and_target_columns(self):
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, 'тест_данные_⑤.csv')
        frame = pd.DataFrame({
            'Month': ['2020-01-01', '2020-02-01', '2020-03-01'],
            'Deals': [47, 53, 59],
            'Value': [347, 353, 359]
        })
        frame.to_csv(path, index=False)
        raw = RawData(path)
        result = raw.target('Deals')
        os.remove(path)
        os.rmdir(directory)
        self.assertEqual(list(result.columns), ['Month', 'Deals'], msg="Target columns are incorrect")

    """
    Test that target returns correct number of rows.
    """
    def test_target_returns_correct_number_of_rows(self):
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, 'тест_данные_⑥.csv')
        frame = pd.DataFrame({
            'Month': ['2020-01-01', '2020-02-01', '2020-03-01'],
            'Deals': [61, 67, 71],
            'Value': [361, 367, 373]
        })
        frame.to_csv(path, index=False)
        raw = RawData(path)
        result = raw.target('Deals')
        os.remove(path)
        os.rmdir(directory)
        self.assertEqual(result.shape[0], 3, msg="Target row count is incorrect")

    """
    Test that Month column in target is datetime type.
    """
    def test_target_month_column_is_datetime(self):
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, 'тест_данные_⑦.csv')
        frame = pd.DataFrame({
            'Month': ['2020-01-01', '2020-02-01', '2020-03-01'],
            'Deals': [73, 79, 83],
            'Value': [373, 379, 383]
        })
        frame.to_csv(path, index=False)
        raw = RawData(path)
        result = raw.target('Deals')
        os.remove(path)
        os.rmdir(directory)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result['Month']), msg="Month column is not datetime")

    """
    Test that private load_csv returns correct DataFrame shape.
    """
    def test_private_load_csv_returns_correct_shape(self):
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, 'тест_данные_⑧.csv')
        frame = pd.DataFrame({
            'Month': ['2020-01-01', '2020-02-01', '2020-03-01'],
            'Deals': [89, 97, 101],
            'Value': [389, 397, 401]
        })
        frame.to_csv(path, index=False)
        raw = RawData(path)
        result = raw._RawData__load_csv()
        os.remove(path)
        os.rmdir(directory)
        self.assertEqual(result.shape, (3, 3), msg="Loaded DataFrame shape is incorrect")

    """
    Test that Month column from load_csv is datetime type.
    """
    def test_private_load_csv_month_column_is_datetime(self):
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, 'тест_данные_⑨.csv')
        frame = pd.DataFrame({
            'Month': ['2020-01-01', '2020-02-01', '2020-03-01'],
            'Deals': [103, 107, 109],
            'Value': [403, 409, 419]
        })
        frame.to_csv(path, index=False)
        raw = RawData(path)
        result = raw._RawData__load_csv()
        os.remove(path)
        os.rmdir(directory)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result['Month']), msg="Month from load_csv is not datetime")

    """
    Test that add_next_month increases row count by one.
    """
    def test_private_add_next_month_increases_row_count(self):
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, 'тест_данные_⑩.csv')
        frame = pd.DataFrame({
            'Month': ['2020-01-01', '2020-02-01', '2020-03-01'],
            'Deals': [113, 127, 131],
            'Value': [421, 433, 439]
        })
        frame.to_csv(path, index=False)
        raw = RawData(path)
        loaded = raw._RawData__load_csv()
        result = raw._RawData__add_next_month(loaded)
        os.remove(path)
        os.rmdir(directory)
        self.assertEqual(result.shape[0], 4, msg="Row count after add_next_month is incorrect")

    """
    Test that last row Deals value after add_next_month is NaN.
    """
    def test_private_add_next_month_last_row_deals_is_nan(self):
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, 'тест_данные_⑪.csv')
        frame = pd.DataFrame({
            'Month': ['2020-01-01', '2020-02-01', '2020-03-01'],
            'Deals': [137, 139, 149],
            'Value': [443, 449, 457]
        })
        frame.to_csv(path, index=False)
        raw = RawData(path)
        loaded = raw._RawData__load_csv()
        result = raw._RawData__add_next_month(loaded)
        os.remove(path)
        os.rmdir(directory)
        self.assertTrue(np.isnan(result.iloc[-1]['Deals']), msg="Last Deals value is not NaN")

    """
    Test that last row Month after add_next_month is next month.
    """
    def test_private_add_next_month_last_row_month_is_next_month(self):
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, 'тест_данные_⑫.csv')
        frame = pd.DataFrame({
            'Month': ['2020-01-01', '2020-02-01', '2020-03-01'],
            'Deals': [151, 157, 163],
            'Value': [461, 463, 467]
        })
        frame.to_csv(path, index=False)
        raw = RawData(path)
        loaded = raw._RawData__load_csv()
        result = raw._RawData__add_next_month(loaded)
        os.remove(path)
        os.rmdir(directory)
        self.assertEqual(result.iloc[-1]['Month'], pd.Timestamp('2020-04-01'), msg="Last Month is not next month")

    """
    Test that create_feature contains lag columns.
    """
    def test_private_create_feature_contains_lag_columns(self):
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, 'тест_данные_⑬.csv')
        frame = pd.DataFrame({
            'Month': ['2020-01-01', '2020-02-01', '2020-03-01'],
            'Deals': [167, 173, 179],
            'Value': [479, 487, 491]
        })
        frame.to_csv(path, index=False)
        raw = RawData(path)
        loaded = raw._RawData__load_csv()
        next_month = raw._RawData__add_next_month(loaded)
        result = raw._RawData__create_feature(next_month)
        os.remove(path)
        os.rmdir(directory)
        self.assertIn('Deals lag1', result.columns, msg="Deals lag1 missing in create_feature")

    """
    Test that create_feature returns correct row count.
    """
    def test_private_create_feature_row_count_is_correct(self):
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, 'тест_данные_⑭.csv')
        frame = pd.DataFrame({
            'Month': ['2020-01-01', '2020-02-01', '2020-03-01'],
            'Deals': [181, 191, 193],
            'Value': [499, 503, 509]
        })
        frame.to_csv(path, index=False)
        raw = RawData(path)
        loaded = raw._RawData__load_csv()
        next_month = raw._RawData__add_next_month(loaded)
        result = raw._RawData__create_feature(next_month)
        os.remove(path)
        os.rmdir(directory)
        self.assertEqual(result.shape[0], 3, msg="Row count in create_feature is incorrect")


if __name__ == "__main__":
    unittest.main()