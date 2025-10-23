import os
import unittest
from scripts.predict_target import predict_target

"""
Integration test verifying the full prediction pipeline end-to-end.
"""

class TestPredictTargetIntegration(unittest.TestCase):

    """
    Checks:
    - Model loads and predicts successfully.
    - Output type is numeric.
    - Output is valid (not NaN, positive).
    - Prediction matches expected snapshot (1669).
    """
    def test_predict_target_returns_expected_value(self):
        model_folder = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../saved_models/xgb_testing_model")
        )
        result = predict_target(model_folder)

        self.assertIsInstance(result, (int, float), "Result should be numeric")
        self.assertFalse(result != result, "Result should not be NaN")
        self.assertGreater(result, 0, "Result should be positive")
        self.assertEqual(int(result), 1669, f"Expected prediction 1669, got {result}")

if __name__ == "__main__":
    unittest.main()
