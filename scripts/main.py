import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.excel_file import excel_file
from src.csv_file import csv_file
from src.xgb_predictor import xgb_predictor

"""
This is mvp file. I edit it and later it will be removed.
"""

if __name__ == "__main__":
    (
        xgb_predictor(
            df=(
                csv_file(csv_path="../data/prepared_data.csv")
                .save(
                    excel_file(excel_path="../data/raw_data.xlsx", sheet_name="Data")
                    .process_time()
                    .add_sin_seasonality(period=12)
                    .add_cos_seasonality(period=12)
                    .add_lags()
                    .prepared_data()
                )
                .load_df()
            ),
            target_column="Deals"
        )
        .train_and_test(
            train_ratio = 12/13
        )
        .save_model(
            path = "../saved_models/test_model.joblib"
        )
    )