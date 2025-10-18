import pandas as pd
import numpy as np
from typing import Self

"""
Works with features and adds seasonality.

Example:
    df = (
        Features(
            Rawdata(...)
            .make_features(...)
        )
        .add_sin_seasonality(period=12)
        .add_cos_seasonality(period=12)
        .prepare_data()
    )
"""

class Features:

    def __init__(self, features: pd.DataFrame):
        self.__features = features

    """
    Adds sin seasonal feature based on monthly index.
    """
    def add_sin_seasonality(self, period: int) -> Self:
        month_index = np.arange(len(self.__features))
        self.__features['sin_season'] = np.sin(2*np.pi*month_index/period)
        return self

    """
    Adds cos seasonal feature based on monthly index.
    """
    def add_cos_seasonality(self, period: int) -> Self:
        month_index = np.arange(len(self.__features))
        self.__features['cos_season'] = np.cos(2*np.pi*month_index/period)
        return self

    """
    Returns dataframe from Features class for using it for training.
    """
    def prepare_data(self) -> pd.DataFrame:
        return self.__features