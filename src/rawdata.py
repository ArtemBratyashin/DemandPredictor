import pandas as pd
import numpy as np
from pathlib import Path

"""
Processes Excel data from raw folder with time series transformations.

Example:
    df = (
        excel_file("../data/raw_data.xlsx", "Data")
        .process_time()
        .add_lags()
        .prepared_data()
    )
"""

class RawData:

    def __init__(self, file_path):
        self.__path = Path(file_path)

    """
    Creates 1-month lag features for all numerical columns to use it for Features class.
    """
    def features(self) -> pd.DataFrame:
        df = pd.read_csv(self.__path)
        df['Month'] = pd.to_datetime(self.df['Month'])
        df = df.set_index('Month')
        for col in df.select_dtypes(include=[np.number]).columns:
            df[f'{col}_lag1'] = df[col].shift(1)
        return df

    """
    Creates target column to use it for training models.
    """
    def target(self, target_param:str) -> pd.DataFrame:
        df = pd.read_csv(self.__path, index_col='Month')
        df['Month'] = pd.to_datetime(self.df['Month'])
        df = df.set_index('Month')
        df = df[target_param]
        return df