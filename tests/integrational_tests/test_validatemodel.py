import os
import unittest
import pandas as pd
from xgboost import XGBRegressor
from scripts.validate_model import validate_model

"""
Integration test verifying the full model validation pipeline.

This test validates that the RawData → Features → ModelValidator pipeline
executes without errors and produces valid validation results.
"""

class TestValidateModelCanReturnDataFrame(unittest.TestCase):

    """
    Checks that validate_model returns a pandas DataFrame.
    """
    def test_validate_model_returns_dataframe(self):
        data_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../data/raw_data.csv")
        )
        result = validate_model(
            model=XGBRegressor(),
            data_path=data_path,
            target="Deals"
        )
        self.assertIsInstance(result, pd.DataFrame, "Result should be a pandas DataFrame")

    """
    Checks that DataFrame contains all required columns.
    """
    def test_validate_model_dataframe_has_required_columns(self):
        data_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../data/raw_data.csv")
        )
        result = validate_model(
            model=XGBRegressor(),
            data_path=data_path,
            target="Deals"
        )
        required_columns = ['TrainPart', 'Actual', 'Predicted', 'MAPE']
        self.assertListEqual(list(result.columns), required_columns, "DataFrame should have required columns")

    """
    Checks that validation result DataFrame contains at least one row.
    """
    def test_validate_model_dataframe_has_rows(self):
        data_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../data/raw_data.csv")
        )
        result = validate_model(
            model=XGBRegressor(),
            data_path=data_path,
            target="Deals"
        )
        self.assertGreater(len(result), 0, "Validation result should contain at least one row")

    """
    Checks that Actual column has numeric data type.
    """
    def test_validate_model_actual_column_is_numeric(self):
        data_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../data/raw_data.csv")
        )
        result = validate_model(
            model=XGBRegressor(),
            data_path=data_path,
            target="Deals"
        )
        self.assertTrue(
            pd.api.types.is_numeric_dtype(result['Actual']),
            "Actual column should be numeric"
        )

    """
    Checks that Predicted column has numeric data type.
    """
    def test_validate_model_predicted_column_is_numeric(self):
        data_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../data/raw_data.csv")
        )
        result = validate_model(
            model=XGBRegressor(),
            data_path=data_path,
            target="Deals"
        )
        self.assertTrue(
            pd.api.types.is_numeric_dtype(result['Predicted']),
            "Predicted column should be numeric"
        )

    """
    Checks that MAPE column has numeric data type.
    """
    def test_validate_model_mape_column_is_numeric(self):
        data_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../data/raw_data.csv")
        )
        result = validate_model(
            model=XGBRegressor(),
            data_path=data_path,
            target="Deals"
        )
        self.assertTrue(
            pd.api.types.is_numeric_dtype(result['MAPE']),
            "MAPE column should be numeric"
        )

    """
    Checks that all MAPE values are greater than or equal to zero.
    """
    def test_validate_model_mape_values_are_non_negative(self):
        data_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../data/raw_data.csv")
        )
        result = validate_model(
            model=XGBRegressor(),
            data_path=data_path,
            target="Deals"
        )
        self.assertTrue(
            (result['MAPE'] >= 0).all(),
            "All MAPE values should be greater than or equal to zero"
        )

    """
    Checks that all MAPE values are less than or equal to 100.
    """
    def test_validate_model_mape_values_are_bounded(self):
        data_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../data/raw_data.csv")
        )
        result = validate_model(
            model=XGBRegressor(),
            data_path=data_path,
            target="Deals"
        )
        self.assertTrue(
            (result['MAPE'] <= 100).all(),
            "All MAPE values should be less than or equal to 100"
        )

    """
    Checks that all Actual values are positive.
    """
    def test_validate_model_actual_values_are_positive(self):
        data_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../data/raw_data.csv")
        )
        result = validate_model(
            model=XGBRegressor(),
            data_path=data_path,
            target="Deals"
        )
        self.assertTrue(
            (result['Actual'] > 0).all(),
            "All Actual values should be positive"
        )

    """
    Checks that all Predicted values are positive.
    """
    def test_validate_model_predicted_values_are_positive(self):
        data_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../data/raw_data.csv")
        )
        result = validate_model(
            model=XGBRegressor(),
            data_path=data_path,
            target="Deals"
        )
        self.assertTrue(
            (result['Predicted'] > 0).all(),
            "All Predicted values should be positive"
        )


if __name__ == "__main__":
    unittest.main()
