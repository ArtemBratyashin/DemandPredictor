import joblib
import os
from xgboost import XGBRegressor
import pandas as pd
from typing import Self

"""
XGBoost model class for deal volume prediction that trains and saves model.

Example:
    XGBTrainer(
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
    .train()
    .save_model(folder_path = "../saved_models", model_name = "xgb_model"))
"""

class XGBTrainer:

    def __init__(self, features: pd.DataFrame, target: pd.DataFrame):
        self.__features = features.copy()
        self.__target = target.copy()
        self.__model = XGBRegressor()

    """
    Trains the model using time-series logic.
    """
    def train(self) -> Self:
        self.__fit_model(
            self.__process_to_equel_months(
                self.__features,
                self.__target
            )
        )
        return self

    """
    Saves the trained model to the directory for using it in prediction class.
    """
    def save_model(self, folder_path:str, model_name: str) -> str:
        model_path = os.path.join(folder_path, "xgb_model.joblib")
        joblib.dump(
            self.__model,
            model_path
        )
        return model_path

    #Needs tests
    """
    Fits the model.
    """
    def __fit_model(self, x, y):
        self.__model.fit(x, y)

    #This code needs to be fixed
    """
    Splits the dataframe into train and test sets based on the ratio of months.
    """
    def __process_to_equel_months(self, features: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        features = features.copy()
        target = target.copy()
        return (features, target)

    #It will be used for prediction class
    """
    Predicts target values for the given feature set.
    """
    def _predict(self, X):
        return self.__model.predict(X)