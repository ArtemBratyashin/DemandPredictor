import joblib
import os
from xgboost import XGBRegressor
import pandas as pd
from typing import Self

"""
XGBoost model class for deal volume prediction that trains and saves model.

Example:
    (
        XGBTrainer()
        .train(
            (
                Features(...)
                ...
                .prepare_data()
            ), 
            (
                RawData(...)
                .target(...)
            )
        )
        .save_model(folder_path = "../saved_models", model_name = "xgb_model"))
    )
"""

class XGBTrainer:

    def __init__(self):
        self.__model = XGBRegressor()

    """
    Trains the model using time-series logic.
    """
    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> Self:
        x, y = self.__process_to_equel_months(features, target)
        self.__fit_model(x, y)
        return self

    """
    Saves the trained model to the directory for using it in prediction class.
    """
    def save_model(self, folder_path:str, model_name: str) -> str:
        model_name += '.joblib'
        model_path = os.path.join(folder_path, model_name)
        joblib.dump(
            self.__model,
            model_path
        )
        return model_path

    """
    Fits the model.
    """
    def __fit_model(self, x, y):
        self.__model.fit(x, y)

    """
    Splits the dataframe into train and test sets based on the ratio of months.
    """

    def __process_to_equel_months(self, features: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        common_months = sorted(set(features['Month']) & set(target['Month']))
        features = features[features['Month'].isin(common_months)].reset_index(drop=True)
        target = target[target['Month'].isin(common_months)].reset_index(drop=True)
        x = features.drop(columns=['Month'])
        y = target.iloc[:, 1]
        return x, y