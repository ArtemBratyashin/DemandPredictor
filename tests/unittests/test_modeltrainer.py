import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from xgboost import XGBRegressor
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.modeltrainer import ModelTrainer

"""
Unit tests for ModelTrainer class.
Covers intersections, train/save logic and .fit call.
"""

class TestModelTrainer(unittest.TestCase):
    """
    Проверяет, что пересечение месяцев работает корректно.
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
        trainer = ModelTrainer(XGBRegressor())
        x, y = trainer._ModelTrainer__process_to_equel_months(features, target)
        self.assertListEqual(x.index.tolist(), [0,1,2,3], "Features index isn't rebuilt correctly")
        self.assertListEqual(list(y), [1,2,3,4], "Target values aren't built correctly")

    """
    Проверяет, что train возвращает self и обучает модель.
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
        trainer = ModelTrainer(XGBRegressor())
        returned = trainer.train(features, target)
        self.assertIs(returned, trainer, "train doesn't return self")

    """
    Проверяет, что save_model сохраняет файл с верным именем.
    """
    def test_save_model_creates_file_with_right_name(self):
        trainer = ModelTrainer(XGBRegressor())
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
    Проверяет, что приватный метод fit работает без ошибок.
    """
    def test_internal_fit_model_trains_without_error(self):
        trainer = ModelTrainer(XGBRegressor())
        x = pd.DataFrame({'f':[0,1,2]})
        y = pd.Series([0,1,2])
        trainer._ModelTrainer__fit_model(x, y)

if __name__ == "__main__":
    unittest.main()
