import sys
from xgboost import XGBRegressor
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.modeltrainer import ModelTrainer
from src.features import Features
from src.rawdata import RawData

"""
Saves model to folder
"""

def save_model(data_path, target, models_folder_path, model_name):
    model_path=(
        ModelTrainer(
            XGBRegressor(),
            (
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
            RawData(data_path)
            .target(target)
        )
        .save_model(folder_path=models_folder_path, model_name=model_name)
    )
    return model_path

if __name__ == "__main__":
    print(
        f"Model was saved to {
            save_model(
                data_path="../data/raw_data.csv", 
                target="Deals", 
                models_folder_path="../saved_models", 
                model_name="xgb_model"
            )
        }"
    )