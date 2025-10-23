import os
import unittest
from scripts.predict_target import predict_target

"""
Integration test verifying the full prediction pipeline.

This test validates that the model loads and runs inference correctly
to predict the target value for the next month.
"""

class TestPredictTargetCanPredictValue(unittest.TestCase):

    """
    Checks that predict_target returns a numeric value.
    """
    def test_predict_target_returns_numeric(self):
        model_folder = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../saved_models/xgb_testing_model")
        )
        result = predict_target(model_folder)
        self.assertIsInstance(result, (int, float), "Result should be numeric")

    """
    Checks that predicted value is not NaN.
    """
    def test_predict_target_returns_not_nan(self):
        model_folder = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../saved_models/xgb_testing_model")
        )
        result = predict_target(model_folder)
        self.assertTrue(result == result, "Result should not be NaN")

    """
    Checks that predicted value is positive.
    """
    def test_predict_target_returns_positive(self):
        model_folder = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../saved_models/xgb_testing_model")
        )
        result = predict_target(model_folder)
        self.assertGreater(result, 0, "Result should be positive")

    """
    Checks that predicted value matches expected snapshot value.
    """
    def test_predict_target_returns_expected_value(self):
        model_folder = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../saved_models/xgb_testing_model")
        )
        result = predict_target(model_folder)
        self.assertEqual(int(result), 1669, f"Expected prediction 1669, got {result}")

if __name__ == "__main__":
    unittest.main()