import unittest
import pandas as pd
import tempfile
import os
import sys
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.predictor import Predictor

"""
Unit tests for Predictor class.
Tests cover predict, load_model and last_month_features methods.
"""

class TestPredictor(unittest.TestCase):

    """
    Test that predict returns a valid scalar prediction.
    """
    def test_predict_returns_scalar(self):
        folder = tempfile.mkdtemp()
        model_name = "тест_модель_①"
        model_folder = os.path.join(folder, model_name)
        os.makedirs(model_folder, exist_ok=True)
        features = pd.DataFrame({
            'Month': ['2020-01-01', '2020-02-01'],
            'признак_α': [17, 23],
            'признак_β': [31, 37]
        })
        features_path = os.path.join(model_folder, "features.csv")
        features.to_csv(features_path, index=False)
        X_train = features.drop(columns=['Month'])
        y_train = [41, 43]
        model = LinearRegression()
        model.fit(X_train, y_train)
        model_path = os.path.join(model_folder, model_name + ".joblib")
        joblib.dump(model, model_path)
        predictor = Predictor(model_folder)
        result = predictor.predict()
        os.remove(model_path)
        os.remove(features_path)
        os.rmdir(model_folder)
        os.rmdir(folder)
        self.assertIsInstance(result, (float, int), msg="Prediction is not numeric")

    """
    Test that last_month_features returns last row without Month.
    """
    def test_last_month_features_returns_last_row_without_month(self):
        folder = tempfile.mkdtemp()
        model_name = "тест_модель_②"
        model_folder = os.path.join(folder, model_name)
        os.makedirs(model_folder, exist_ok=True)
        features = pd.DataFrame({
            'Month': ['2020-01-01', '2020-02-01', '2020-03-01'],
            'признак_γ': [47, 53, 59],
            'признак_δ': [61, 67, 71]
        })
        features_path = os.path.join(model_folder, "features.csv")
        features.to_csv(features_path, index=False)
        predictor = Predictor(model_folder)
        result = predictor._Predictor__last_month_features(model_folder)
        os.remove(features_path)
        os.rmdir(model_folder)
        os.rmdir(folder)
        self.assertEqual(result.shape[0], 1, msg="Last features row count is not 1")

    """
    Test that last_month_features does not contain month column.
    """
    def test_last_month_features_does_not_contain_month_column(self):
        folder = tempfile.mkdtemp()
        model_name = "тест_модель_③"
        model_folder = os.path.join(folder, model_name)
        os.makedirs(model_folder, exist_ok=True)
        features = pd.DataFrame({
            'Month': ['2020-01-01', '2020-02-01'],
            'признак_ε': [73, 79],
            'признак_ζ': [83, 89]
        })
        features_path = os.path.join(model_folder, "features.csv")
        features.to_csv(features_path, index=False)
        predictor = Predictor(model_folder)
        result = predictor._Predictor__last_month_features(model_folder)
        os.remove(features_path)
        os.rmdir(model_folder)
        os.rmdir(folder)
        self.assertNotIn('Month', result.columns, msg="Month column was not dropped")

    """
    Test that last_month_features returns correct last value.
    """
    def test_last_month_features_returns_correct_last_value(self):
        folder = tempfile.mkdtemp()
        model_name = "тест_модель_④"
        model_folder = os.path.join(folder, model_name)
        os.makedirs(model_folder, exist_ok=True)
        features = pd.DataFrame({
            'Month': ['2020-01-01', '2020-02-01', '2020-03-01'],
            'признак_η': [97, 101, 103],
            'признак_θ': [107, 109, 113]
        })
        features_path = os.path.join(model_folder, "features.csv")
        features.to_csv(features_path, index=False)
        predictor = Predictor(model_folder)
        result = predictor._Predictor__last_month_features(model_folder)
        os.remove(features_path)
        os.rmdir(model_folder)
        os.rmdir(folder)
        self.assertEqual(result.iloc[0]['признак_η'], 103, msg="Last value is incorrect")

    """
    Test that load_model returns a valid model object.
    """
    def test_load_model_returns_model_object(self):
        folder = tempfile.mkdtemp()
        model_name = "тест_модель_⑤"
        model_folder = os.path.join(folder, model_name)
        os.makedirs(model_folder, exist_ok=True)
        model = LinearRegression()
        model_path = os.path.join(model_folder, model_name + ".joblib")
        joblib.dump(model, model_path)
        predictor = Predictor(model_folder)
        loaded_model = predictor._Predictor__load_model(model_folder)
        os.remove(model_path)
        os.rmdir(model_folder)
        os.rmdir(folder)
        self.assertIsInstance(loaded_model, LinearRegression, msg="Loaded object is not a model")

    """
    Test that load_model correctly extracts model name from folder path.
    """
    def test_load_model_extracts_correct_name_from_path(self):
        folder = tempfile.mkdtemp()
        model_name = "мой_xgb_модель_⑥"
        model_folder = os.path.join(folder, model_name)
        os.makedirs(model_folder, exist_ok=True)
        model = LinearRegression()
        model_path = os.path.join(model_folder, model_name + ".joblib")
        joblib.dump(model, model_path)
        predictor = Predictor(model_folder)
        loaded_model = predictor._Predictor__load_model(model_folder)
        os.remove(model_path)
        os.rmdir(model_folder)
        os.rmdir(folder)
        self.assertIsNotNone(loaded_model, msg="Model was not loaded")


if __name__ == "__main__":
    unittest.main()