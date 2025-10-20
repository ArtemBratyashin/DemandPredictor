import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.xgbtrainer import XGBTrainer

"""
Unit tests for XGBTrainer class.

Tests covering fit, process, save-model logic and month intersection.
"""

class TestXGBTrainer(unittest.TestCase):
    """
    Test that only common months are returned after processing.
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
        # Only months 2,3,4,5 should remain
        trainer = XGBTrainer()
        processed_features, processed_target = trainer._XGBTrainer__process_to_equel_months(features, target)
        self.assertListEqual(
            list(processed_features['Month']),
            [2,3,4,5],
            "Features do not contain correct common months"
        )
        self.assertListEqual(
            list(processed_target['Month']),
            [2,3,4,5],
            "Target does not contain correct common months"
        )

    """
    Test that train returns self for chaining and fits model.
    """
    def test_train_returns_self_and_fits(self):
        features = pd.DataFrame({
            'Month': [2,3,4],
            'f1': [1,2,3],
            'f2': [3,2,1]
        })
        target = pd.DataFrame({
            'Month': [2,3,4],
            'target': [7,5,3]
        })
        trainer = XGBTrainer()
        returned = trainer.train(features, target)
        self.assertIs(returned, trainer, "train does not return self")

    """
    Test that save_model persists model with provided name.
    """
    def test_save_model_creates_file_with_right_name(self):
        trainer = XGBTrainer()
        folder = tempfile.mkdtemp()
        model_name = "unit_test_model"
        trainer.train(
            pd.DataFrame({'Month':[2,3],'f':[1,2]}),
            pd.DataFrame({'Month':[2,3],'target':[1,2]})
        )
        path = trainer.save_model(folder, model_name)
        self.assertTrue(os.path.exists(path), "Model file was not saved")
        self.assertTrue(path.endswith(f"{model_name}.joblib"), "Model name missing in saved file path")
        os.remove(path)
        os.rmdir(folder)

    """
    Test that internal fit_model can fit on the given small dataset.
    """
    def test_internal_fit_model_trains_without_error(self):
        trainer = XGBTrainer()
        x = pd.DataFrame({'f':[0,1,2]})
        y = pd.Series([0,1,2])
        # Should not raise error
        trainer._XGBTrainer__fit_model(x, y)

if __name__ == "__main__":
    unittest.main()