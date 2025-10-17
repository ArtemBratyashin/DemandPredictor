import pandas as pd
from pathlib import Path

"""
Saves processed data from DataProcessor to CSV format and loads CSV files.

Example:
    df = (
        csv_file("data/prepared_data.csv")
        .save(prepared_data)
        .load_df()
    )
"""

class csv_file:

    def __init__(self, csv_path):
        self.__path = Path(csv_path)

    """
    Saves data from excel_file to CSV file.
    """
    def save(self, prepared_data):
        data = prepared_data
        data.to_csv(self.__path, index=False)
        return self

    """
    Loads and returns dataframe from CSV file.
    """
    def load_df(self):
        df = pd.read_csv(self.__path)
        df['Month'] = pd.to_datetime(df['Month'])
        return df