import pandas as pd
import numpy as np
from pathlib import Path

"""
Processes Excel data from raw folder with time series transformations.

Examples:
    df = (
        RawData("../data/raw_data.csv")
        .make_features()
    )
    df = (
        RawData("../data/raw_data.csv")
        .target('Deals')
    )
"""

class RawData:

    def __init__(self, file_path):
        self.__path = Path(file_path)

    """
    Creates 1-month lag features for all numerical columns to use it for Features class.
    """
    def make_features(self) -> pd.DataFrame:
        df = self.__load_csv()
        features = pd.DataFrame()
        features['Month'] = df['Month']
        for col in df.select_dtypes(include=[np.number]).columns:
            features[f'{col} lag1'] = df[col].shift(1)
        return features

    """
    Creates target column to use it for training models.
    """
    def target(self, target_param:str) -> pd.DataFrame:
        df = self.__load_csv()
        df = df['Month', target_param]
        return df

    """
    Private function for loading data. 
    """
    def __load_csv(self) -> pd.DataFrame:
        df = pd.read_csv(self.__path)
        df['Month'] = pd.to_datetime(df['Month'])
        return df