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
            Features(
                Rawdata(...)
                .make_features(...)
            )
            .add_sin_seasonality(period=12)
            .add_cos_seasonality(period=12)
            .prepare_data()
        ), 
        (
            RawData("../data/raw_data.csv")
            .target('Deals')
        )
    )
    .train()
    .save_model(folder_path = "../saved_models", model_name = "xgb_model"))
"""

class XGBTrainer:

    def __init__(self, features: pd.DataFrame, target: pd.DataFrame):
        self.features = features.copy()
        self.target = target.copy()
        self.model = XGBRegressor()

    """
    Trains the model using time-series logic.
    """
    def train(self) -> Self:
        self.model.fit(
            self.__process_to_equel_months(
                self.features,
                self.target
            )
        )
        return self

    """
    Saves the trained model to the directory for using it in prediction class.
    """
    def save_model(self, folder_path:str, model_name: str) -> str:
        model_path = os.path.join(folder_path, "xgb_model.joblib")
        joblib.dump(
            self.model,
            model_path
        )
        return model_path

    """
    Splits the dataframe into train and test sets based on the ratio of months.
    """
    def __process_to_equel_months(self, features: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        features = features.copy()
        target = target.copy()
        return (features, target)

    """
    Predicts target values for the given feature set.
    """
    def _predict(self, X):
        return self.model.predict(X)