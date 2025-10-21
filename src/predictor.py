import joblib
import os
import pandas as pd

"""
Loads a trained model and predicts for the latest row of the provided DataFrame.

Example:
    Predictor(
        models_folder = "../saved_models/xgb_model",
        model_name = "xgb_model"
    )
    .predict(
        
    )
"""

class Predictor:

    def __init__(self, models_folder: str, model_name: str):
        self.__models_folder = models_folder
        self.__model_name = model_name

    def predict_last(self) -> float:
        model_path = os.path.join(self.__models_folder, self.__model_name, self.__model_name + ".joblib")
        features_path = os.path.join(self.__models_folder, self.__model_name, "features.csv")
        model = joblib.load(model_path)
        features = pd.read_csv(features_path)
        x_last = features.drop(columns=['Month'], errors='ignore').iloc[[-1]]
        prediction = model.predict(x_last)
        return float(prediction[0])