import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from xgboost import XGBRegressor
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.model_trainer import ModelTrainer

"""
Unit tests for the updated ModelTrainer class.
Covers intersections, train, save logic and .fit call.
"""

class TestModelTrainer(unittest.TestCase):
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
        trainer = ModelTrainer(XGBRegressor(), features)
        x, y = trainer._ModelTrainer__process_to_equel_months(features, target)
        self.assertListEqual(list(x.index), [0,1,2,3], "Features index isn't rebuilt correctly")
        self.assertListEqual(list(y), [1,2,3,4], "Target values aren't built correctly")

    """
    Checks that train returns self and trains the model.
    """
    def test_train_returns_self_and_fits(self):
        features = pd.DataFrame({
            'Month': [2,3,4],
            'f1': [10,11,12]
        })
        target = pd.DataFrame({
            'Month': [2,3,4],
            'target': [5,4,3]
        })
        trainer = ModelTrainer(XGBRegressor(), features)
        returned = trainer.train(target)
        self.assertIs(returned, trainer, "train doesn't return self")

    """
    Checks that save_model saves files in the correct subfolder.
    """
    def test_save_model_creates_files_in_folder(self):
        features = pd.DataFrame({'Month':[2,3],'f':[1,2]})
        target = pd.DataFrame({'Month':[2,3],'target':[1,2]})
        trainer = ModelTrainer(XGBRegressor(), features)
        trainer.train(target)

        folder = tempfile.mkdtemp()
        model_name = "unit_test_model"
        model_path = trainer.save_model(folder, model_name)

        model_dir = os.path.join(folder, model_name)
        joblib_file = os.path.join(model_dir, model_name + '.joblib')
        csv_file = os.path.join(model_dir, 'features.csv')

        self.assertTrue(os.path.exists(joblib_file), "Model file was not saved in subfolder")
        self.assertTrue(os.path.exists(csv_file), "Features CSV file was not saved in subfolder")
        self.assertTrue(model_path == model_dir, "Returned model path is not correct")

        # Cleanup
        if os.path.exists(joblib_file):
            os.remove(joblib_file)
        if os.path.exists(csv_file):
            os.remove(csv_file)
        if os.path.exists(model_dir):
            os.rmdir(model_dir)
        os.rmdir(folder)

    """
    Checks that the private fit method works without errors.
    """
    def test_internal_fit_model_trains_without_error(self):
        features = pd.DataFrame({'f':[0,1,2], 'Month':[1,2,3]})
        trainer = ModelTrainer(XGBRegressor(), features)
        x = features.drop(columns=['Month'])
        y = pd.Series([0,1,2])
        trainer._ModelTrainer__fit_model(x, y)

if __name__ == "__main__":
    unittest.main()

