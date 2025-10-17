import joblib
from xgboost import XGBRegressor
import pandas as pd
import numpy as np

"""
XGBoost model class for deal volume prediction.

Example:
    xgb_predictor(df, "Deals")
    .train(train_ratio = 12/13)
    .save_model(path = "../saved_models")
"""

class XGBPredictor:

    def __init__(self, df, target_column):
        self.df = df.copy()
        self.target_column = target_column
        self.model = None

    """
    Trains the model using time-series logic.
    The initial part (train_ratio fraction) of months is used for training.
    """
    def train(self, train_ratio):
        X_train, y_train, X_test, y_test = self._split_train_test(train_ratio)
        self._fit_model(X_train, y_train)
        return self

    """
    Trains and tests the model using time-series logic.
    The initial part (train_ratio fraction) of months is used for training,
    The rest for testing. Prints test MAPE.
    """
    def train_and_test(self, train_ratio):
        X_train, y_train, X_test, y_test = self._split_train_test(train_ratio)
        self._fit_model(X_train, y_train)
        y_pred = self._predict(X_test)
        self._disp_mape(y_test, y_pred)
        return self

    """
    Predicts the number of deals for the next month (using the last row of features).
    """
    def predict_next_month(self):
        lag1_cols = self._get_lag1_columns()
        last_row = self.df[lag1_cols].iloc[[-1]]
        return float(self.model.predict(last_row)[0])

    """
    Saves the trained model to the specified or default directory.
    """
    def save_model(self, path):
        joblib.dump(self.model, path)
        return self

    """
    Splits the dataframe into train and test sets based on the ratio of months.
    """
    def _split_train_test(self, train_ratio):
        months = self.df['Month'].sort_values().unique()
        split = int(len(months)*train_ratio)

        train = self.df[self.df['Month'].isin(months[:split])]
        test = self.df[self.df['Month'].isin(months[split:])]

        lag1_cols = self._get_lag1_columns()

        X_train, y_train = train[lag1_cols], train[self.target_column]
        X_test, y_test = test[lag1_cols], test[self.target_column]

        return X_train, y_train, X_test, y_test

    """
    Returns list of columns with lag_1 in their name.
    """
    def _get_lag1_columns(self):
        return [col for col in self.df.columns if 'lag1' in col]

    """
    Fits the XGBoost model on the training data.
    """
    def _fit_model(self, X_train, y_train):
        self.model = XGBRegressor()
        self.model.fit(X_train, y_train)

    """
    Predicts target values for the given feature set.
    """
    def _predict(self, X):
        return self.model.predict(X)

    """
    Calculates Mean Absolute Percentage Error.
    """
    def _disp_mape(self, y_true, y_pred):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        print(f"Real amount of deals: {' '.join(str(x) for x in y_true)}")
        print(f"Predicted amount: {' '.join(str(x) for x in y_pred)}")
        print(f"Test MAPE: {mape:.2f}%")