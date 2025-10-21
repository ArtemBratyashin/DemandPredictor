import joblib
import pandas as pd

"""
Loads a trained model and predicts for the latest row of the provided DataFrame.

Example:
    Predictor("../saved_models/xgb_model.joblib")
    .predict(
        Features(
            RawData(data_path)
            .make_features()
        )
        .add_sin_seasonality(period=12)
        .add_cos_seasonality(period=12)
        .prepare_data()
    )
"""

class Predictor:

    def __init__(self, model_path: str):
        self.__model_path = model_path

    """
    Predict using the model for the last row in the DataFrame.
    Returns prediction as float.
    """
    def predict(self, features: pd.DataFrame) -> float:
        model = joblib.load(self.__model_path)
        x_last = features.drop(columns=['Month'], errors='ignore').iloc[[-1]]
        print(x_last)
        prediction = model.predict(x_last)
        return float(prediction[0])