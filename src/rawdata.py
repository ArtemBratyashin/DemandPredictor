import pandas as pd
import numpy as np
from pathlib import Path

"""
Processes raw data to features and target values for prediction model.

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
        df = self.__load_csv(self.__path)
        df = self.__add_next_month(df)
        features = self.__create_feature(df)
        return features

    """
    Creates target column to use it for training models.
    """
    def target(self, target_param:str) -> pd.DataFrame:
        df = self.__load_csv(self.__path)
        df = df[['Month', target_param]]
        return df

    """
    Private function for loading data. 
    """
    def __load_csv(self, path:str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df['Month'] = pd.to_datetime(df['Month'])
        return df

    """
    Private function to add next month row with NaN values for numerical columns.
    """
    def __add_next_month(self, df) -> pd.DataFrame:
        next_month = df['Month'].max() + pd.DateOffset(months=1)
        next_row = {col: np.nan for col in df.columns}
        next_row['Month'] = next_month
        return pd.concat([df, pd.DataFrame([next_row])], ignore_index=True)

    """
    Private function for feature dataframe template. 
    """
    def __create_feature(self, df) -> pd.DataFrame:
        features = pd.DataFrame()
        features['Month'] = df['Month']
        for col in df.select_dtypes(include=[np.number]).columns:
            features[f'{col} lag1'] = df[col].shift(1)
        features = features.iloc[1:].reset_index(drop=True)
        return features