import joblib
import os
import pandas as pd
from typing import Any, Self

"""
Model class for prediction target that trains and saves model.

Example:
    (
        ModelTrainer(
            XGBRegressor(),
            (
                Features(...)
                ...
                .prepare_data()
            )
        )
        .train(
            (
                RawData(...)
                .target(...)
            )
        )
        .save_model(folder_path = "../saved_models", model_name = "xgb_model"))
    )
"""

class ModelTrainer:

    def __init__(self, model: Any, features: pd.DataFrame):
        self.__model = model
        self.__features = features

    """
    Trains the model using time-series logic.
    """
    def train(self, target: pd.DataFrame) -> Self:
        x, y = self.__process_to_equel_months(self.__features, target)
        self.__fit_model(x, y)
        return self

    """
    Saves the trained model to the directory for using it in prediction class.
    """
    def save_model(self, folder_path: str, model_name: str) -> str:
        model_folder = os.path.join(folder_path, model_name)
        os.makedirs(model_folder, exist_ok=True)
        model_file = os.path.join(model_folder, model_name + '.joblib')
        joblib.dump(self.__model, model_file)
        features_file = os.path.join(model_folder, 'features.csv')
        self.__features.to_csv(features_file, index=False)
        abs_path = os.path.abspath(model_folder)
        return abs_path

    """
    Fits the model.
    """
    def __fit_model(self, x, y):
        self.__model.fit(x, y)

    """
    Prepares feature and target for prediction.
    """
    def __process_to_equel_months(self, features: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        common_months = sorted(set(features['Month']) & set(target['Month']))
        features = features[features['Month'].isin(common_months)].reset_index(drop=True)
        target = target[target['Month'].isin(common_months)].reset_index(drop=True)
        x = features.drop(columns=['Month'])
        y = target.iloc[:, 1]
        return x, y