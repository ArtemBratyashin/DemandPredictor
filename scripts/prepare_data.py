import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.excel_file import excel_file
from src.csv_file import csv_file

'''
Prepares raw data to prepare data for using it in models
'''

def prepare_data(excel_path, sheet_name, csv_path):
    (
        csv_file(csv_path)
        .save(
            excel_file(excel_path, sheet_name)
            .process_time()
            .add_sin_seasonality(period=12)
            .add_cos_seasonality(period=12)
            .add_lags()
            .prepared_data()
        )
    )

if __name__ == "__main__":
    prepare_data(excel_path = "../data/raw_data.xlsx", sheet_name = "Data", csv_path = "../data/prepared_data.csv")