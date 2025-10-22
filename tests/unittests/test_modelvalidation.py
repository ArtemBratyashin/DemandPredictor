import unittest
import pandas as pd
import numpy as np
import sys
from xgboost import XGBRegressor
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.model_validator import ModelValidator

"""
Unit tests for the ModelValidator class.
Covers validation logic, walk-forward splits, MAPE calculation and month processing.
"""

class TestModelValidator(unittest.TestCase):
    """
    Checks if intersection of months works correctly.
    """
    def test_process_to_equel_months_retains_common_months(self):
        features = pd.DataFrame({
            'Month': [2,3,4,5,6],
            'f1': np.arange(5)
        })
        target = pd.DataFrame({
            'Month': [1,2,3,4,5],
            'target': np.arange(5)
        })
        validator = ModelValidator(XGBRegressor(), features)
        x, y = validator._ModelValidator__process_to_equel_months(features, target)
        self.assertListEqual(list(x.index), [0,1,2,3], "Features index isn't rebuilt correctly")
        self.assertListEqual(list(y), [1,2,3,4], "Target values aren't built correctly")

    """
    Checks that validate returns DataFrame with correct columns.
    """
    def test_validate_returns_dataframe_with_correct_columns(self):
        features = pd.DataFrame({
            'Month': [1,2,3,4,5,6,7,8],
            'f1': np.random.rand(8)
        })
        target = pd.DataFrame({
            'Month': [1,2,3,4,5,6,7,8],
            'target': np.random.rand(8)
        })
        validator = ModelValidator(XGBRegressor(), features)
        result = validator.validate(target)
        self.assertListEqual(list(result.columns), ['TrainPart', 'Actual', 'Predicted', 'MAPE'], "Output columns are incorrect")

    """
    Checks that validate returns correct number of rows based on splits.
    """
    def test_validate_returns_correct_number_of_rows(self):
        features = pd.DataFrame({
            'Month': [1,2,3,4,5,6,7,8,9,10,11,12],
            'f1': np.random.rand(12)
        })
        target = pd.DataFrame({
            'Month': [1,2,3,4,5,6,7,8,9,10,11,12],
            'target': np.random.rand(12)
        })
        validator = ModelValidator(XGBRegressor(), features)
        result = validator.validate(target)
        self.assertEqual(len(result), 4, "Validate should return 4 rows for 4 splits")

    """
    Checks that TrainPart column has correct format.
    """
    def test_validate_trainpart_has_correct_format(self):
        features = pd.DataFrame({
            'Month': [1,2,3,4,5,6,7,8],
            'f1': np.random.rand(8)
        })
        target = pd.DataFrame({
            'Month': [1,2,3,4,5,6,7,8],
            'target': np.random.rand(8)
        })
        validator = ModelValidator(XGBRegressor(), features)
        result = validator.validate(target)
        for part in result['TrainPart']:
            self.assertIn('/', part, "TrainPart should contain '/' separator")

    """
    Checks that MAPE is None when actual value is zero.
    """
    def test_mape_returns_none_when_actual_is_zero(self):
        validator = ModelValidator(XGBRegressor(), pd.DataFrame({'Month':[1],'f1':[1]}))
        mape = validator._ModelValidator__mape(0, 10)
        self.assertIsNone(mape, "MAPE should be None when actual is zero")

    """
    Checks that MAPE is calculated correctly for non-zero actual values.
    """
    def test_mape_calculates_correctly_for_non_zero_actual(self):
        validator = ModelValidator(XGBRegressor(), pd.DataFrame({'Month':[1],'f1':[1]}))
        mape = validator._ModelValidator__mape(100, 90)
        self.assertAlmostEqual(mape, 10.0, places=2, msg="MAPE calculation is incorrect")

    """
    Checks that get_splits returns correct split indices.
    """
    def test_get_splits_returns_correct_indices(self):
        validator = ModelValidator(XGBRegressor(), pd.DataFrame({'Month':[1],'f1':[1]}))
        splits = validator._ModelValidator__get_splits(12)
        self.assertListEqual(splits, [6, 9, 10, 11], "Split indices are incorrect for total=12")

    """
    Checks that validate handles mismatched months correctly.
    """
    def test_validate_handles_mismatched_months(self):
        features = pd.DataFrame({
            'Month': [1,2,3,4,5,6,7,8,9,10],
            'f1': np.random.rand(10)
        })
        target = pd.DataFrame({
            'Month': [3,4,5,6,7,8],
            'target': np.random.rand(6)
        })
        validator = ModelValidator(XGBRegressor(), features)
        result = validator.validate(target)
        self.assertGreater(len(result), 0, "Validate should handle mismatched months")

    """
    Checks that validate works with non-ASCII feature names.
    """
    def test_validate_works_with_non_ascii_features(self):
        features = pd.DataFrame({
            'Month': [1,2,3,4,5,6,7,8],
            '容量': np.random.rand(8),
            'шаг': np.random.rand(8)
        })
        target = pd.DataFrame({
            'Month': [1,2,3,4,5,6,7,8],
            'target': np.random.rand(8)
        })
        validator = ModelValidator(XGBRegressor(), features)
        result = validator.validate(target)
        self.assertFalse(result.isnull().any().any(), "Result should not contain null values")

    """
    Checks that validate works with minimum dataset size.
    """
    def test_validate_works_with_minimum_data(self):
        features = pd.DataFrame({
            'Month': [1,2,3,4],
            'f1': [1.0,2.0,3.0,4.0]
        })
        target = pd.DataFrame({
            'Month': [1,2,3,4],
            'target': [10.0,20.0,30.0,40.0]
        })
        validator = ModelValidator(XGBRegressor(), features)
        result = validator.validate(target)
        self.assertGreater(len(result), 0, "Validate should work with minimum data")

if __name__ == "__main__":
    unittest.main()
