import joblib
import os
import pandas as pd

"""
Loads a trained model and predicts for the latest row of the provided DataFrame.

Example:
    Predictor(
        models_folder = "../saved_models/xgb_model"
    )
    .predict()
"""

class Predictor:

    def __init__(self, model_folder: str):
        self.__model_folder = model_folder

    def predict(self) -> float:
        model_name = os.path.basename(os.path.normpath(self.__model_folder))
        model_path = os.path.join(self.__model_folder, model_name + ".joblib")
        features_path = os.path.join(self.__model_folder, "features.csv")
        model = joblib.load(model_path)
        features = pd.read_csv(features_path)
        x_last = features.drop(columns=['Month'], errors='ignore').iloc[[-1]]
        prediction = model.predict(x_last)
        return float(prediction[0])