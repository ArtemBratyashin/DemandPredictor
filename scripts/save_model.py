import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.xgbpredictor import XGBPredictor
from src.features import csv_file

"""
Saves model to folder
"""

def save_model(data_path, target_column, model_path):
    (
        XGBPredictor(
            df=(
                csv_file(data_path)
                .load_df()
            ),
            target_column=target_column,
        )
        .train(
            train_ratio=12/12
        )
        .save_model(
            path=model_path
        )
    )

if __name__ == "__main__":
    save_model(data_path="../data/prepared_data.csv", target_column="Deals", model_path="../saved_models/xgb_model.joblib")