import pandas as pd
import numpy as np
from pathlib import Path

"""
Processes Excel data from raw folder with time series transformations.

Example:
    repared_data = (DataProcessor("../data/raw.xlsx", "Данные")
                    .load()
                    .process_time()
                    .add_sin_seasonality(period=12)
                    .add_cos_seasonality(period=12)
                    .add_lags()
                    .prepare_for_save()
"""

class DataProcessor:

    def __init__(self, path, sheet):
        self.__path = Path(path)
        self.__sheet = sheet
        self.__data = None

    """
    Loads Excel data from specified path and sheet.
    """
    def load(self):
        if not self.__path.exists():
            raise FileNotFoundError(f"Excel file not found at {self.__path}")
        self.__data = pd.read_excel(self.__path, sheet_name=self.__sheet)
        return self

    """
    Converts time column to datetime format.
    """
    def process_time(self):
        self.__data['Время'] = pd.to_datetime(self.__data['Время'])
        return self

    """
    Adds sin seasonal feature based on monthly index.
    """
    def add_sin_seasonality(self, period:int):
        self.__data = self.__data.reset_index(drop=True)
        month_index = np.arange(len(self.__data))
        self.__data['sin_season'] = np.sin(2*np.pi*month_index/period)
        return self

    """
    Adds cos seasonal feature based on monthly index.
    """
    def add_cos_seasonality(self, period:int):
        self.__data = self.__data.reset_index(drop=True)
        month_index = np.arange(len(self.__data))
        self.__data['cos_season'] = np.cos(2*np.pi*month_index/period)
        return self

    """
    Creates 1-month lag features for all numerical columns.
    """
    def add_lags(self):
        for col in self.__data.select_dtypes(include=[np.number]).columns:
            if col not in ['sin_season', 'cos_season']:
                self.__data[f'{col}_lag1'] = self.__data[col].shift(1)
        return self

    """
    Removes NaN values and resets index for clean dataset ready for saving.
    """
    def prepare_for_save(self):
        self.__data = self.__data.dropna().reset_index(drop=True)
        return self