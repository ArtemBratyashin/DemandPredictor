import joblib
import os
import pandas as pd
from pathlib import Path

"""
Loads a trained model and predicts for the latest row of the provided DataFrame.

Example:
    Predictor(
        models_folder = "../saved_models/xgb_model"
    )
    .predict()
"""

class Predictor:

    def __init__(self, model_folder: str|Path):
        self.__model_folder = Path(model_folder)

    """
    Predicts the target for the next month.
    """
    def predict(self) -> float:
        model = self.__load_model(self.__model_folder)
        features = self.__last_month_features(self.__model_folder)
        prediction = model.predict(features)
        return float(prediction[0])

    """
    Gives last month features for prediction.
    """
    def __last_month_features(self, model_folder: str|Path) -> pd.DataFrame:
        features_path = os.path.join(model_folder, "features.csv")
        features = pd.read_csv(features_path)
        last_features = features.drop(columns=['Month'], errors='ignore').iloc[[-1]]
        return last_features

    """
    Loads model from folder.
    """
    def __load_model(self, model_folder: str|Path) -> joblib.load:
        model_name = os.path.basename(os.path.normpath(model_folder))
        model_path = os.path.join(model_folder, model_name + ".joblib")
        model = joblib.load(model_path)
        return model