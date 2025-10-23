import unittest
import tempfile
import os
from xgboost import XGBRegressor
from scripts.save_model import save_model

"""
Integration test verifying the full model training and saving pipeline.

This test validates that the RawData → Features → ModelTrainer → save
pipeline executes without errors and produces correct artifacts.
"""

class TestSaveModelCanSaveValidArtifacts(unittest.TestCase):

    """
    Checks that model file (.joblib) is created in correct location.
    """
    def test_save_model_produces_joblib_file(self):
        temp_dir = tempfile.mkdtemp()
        data_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../data/raw_data.csv")
        )
        model_name = "integration_test_model_001"
        result_path = save_model(
            model=XGBRegressor(),
            data_path=data_path,
            target="Deals",
            models_folder_path=temp_dir,
            model_name=model_name
        )
        joblib_file = os.path.join(result_path, f"{model_name}.joblib")
        self.assertTrue(os.path.exists(joblib_file), "Model joblib file was not created")

    """
    Checks that features CSV file is created in correct location.
    """
    def test_save_model_produces_features_file(self):
        temp_dir = tempfile.mkdtemp()
        data_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../data/raw_data.csv")
        )
        model_name = "integration_test_model_002"
        result_path = save_model(
            model=XGBRegressor(),
            data_path=data_path,
            target="Deals",
            models_folder_path=temp_dir,
            model_name=model_name
        )
        features_file = os.path.join(result_path, "features.csv")
        self.assertTrue(os.path.exists(features_file), "Features CSV file was not created")

    """
    Checks that returned path matches expected model directory.
    """
    def test_save_model_returns_correct_path(self):
        temp_dir = tempfile.mkdtemp()
        data_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../data/raw_data.csv")
        )
        model_name = "integration_test_model_003"
        result_path = save_model(
            model=XGBRegressor(),
            data_path=data_path,
            target="Deals",
            models_folder_path=temp_dir,
            model_name=model_name
        )
        expected_path = os.path.join(temp_dir, model_name)
        self.assertEqual(result_path, expected_path, "Returned path does not match expected directory")

    """
    Checks that joblib file is not empty.
    """
    def test_save_model_produces_non_empty_joblib(self):
        temp_dir = tempfile.mkdtemp()
        data_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../data/raw_data.csv")
        )
        model_name = "integration_test_model_004"
        result_path = save_model(
            model=XGBRegressor(),
            data_path=data_path,
            target="Deals",
            models_folder_path=temp_dir,
            model_name=model_name
        )
        joblib_file = os.path.join(result_path, f"{model_name}.joblib")
        self.assertGreater(os.path.getsize(joblib_file), 0, "Model joblib file is empty")

    """
    Checks that features CSV file is not empty.
    """
    def test_save_model_produces_non_empty_features(self):
        temp_dir = tempfile.mkdtemp()
        data_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../data/raw_data.csv")
        )
        model_name = "integration_test_model_005"
        result_path = save_model(
            model=XGBRegressor(),
            data_path=data_path,
            target="Deals",
            models_folder_path=temp_dir,
            model_name=model_name
        )
        features_file = os.path.join(result_path, "features.csv")
        self.assertGreater(os.path.getsize(features_file), 0, "Features CSV file is empty")


if __name__ == "__main__":
    unittest.main()
