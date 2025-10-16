import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.csv_file import csv_file
from src.xgb_predictor import xgb_predictor

"""
This is mvp file. I edit it and later it will be removed.
"""

if __name__ == "__main__":

    (
        xgb_predictor(
            df=(
                csv_file("../data/prepared_data.csv")
                .load_df()
            ),
            target_column="Deals"
        )
        .train(
            train_ratio = 12/13
        )
        .save_model(
            path = "../saved_models"
        )
    )