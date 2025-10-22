from xgboost import XGBRegressor
from typing import Any

from src.rawdata import RawData
from src.features import Features
from src.model_trainer import ModelTrainer
from src.predictor import Predictor
from src.model_tester import ModelValidation
from scripts.predict_target import predict_target
from scripts.save_model import save_model

"""
This is main file. Here you can see the examples of using projects code.
Just choose short or long version via '#'
"""

def long_example(model:Any, data_path:str, target:str, models_folder_path:str, model_name:str) -> str:
    result = (
        Predictor(
            model_folder=(
                ModelTrainer(
                    model=model,
                    features=(
                        Features(
                            RawData(data_path)
                            .make_features()
                        )
                        .add_sin_seasonality(period=12)
                        .add_cos_seasonality(period=12)
                        .prepare_data()
                    )
                )
                .train(
                    target=(
                        RawData(data_path)
                        .target(target)
                    )
                )
                .save_model(
                    folder_path=models_folder_path,
                    model_name=model_name
                )
            )
        )
        .predict()
    )
    return result

def short_example(model:Any, data_path:str, target:str, models_folder_path:str, model_name:str) -> str:
    result =(
        predict_target(
            model_folder = (
                save_model(
                    model=model,
                    data_path=data_path,
                    target=target,
                    models_folder_path=models_folder_path,
                    model_name=model_name
                )
            )
        )
    )
    return result

if __name__ == "__main__":
    print(
        f"There will be {
        int(
            #short_example(
            long_example(
                model=XGBRegressor(),
                data_path="data/raw_data.csv", 
                target="Deals", 
                models_folder_path="../saved_models", 
                model_name="xgb_model"
            )
        )
        } deals in september 2025."
    )