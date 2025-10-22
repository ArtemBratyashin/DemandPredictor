import pandas as pd
from typing import Any

"""
Model validation class that tests model performance using time-series walk-forward validation.

Example:
    (
        ModelValidation(
            XGBRegressor(),
            (
                Features(...)
                ...
                .prepare_data()
            )
        )
        .validate(
            (
                RawData(...)
                .target(...)
            )
        )
    )
"""

class ModelTester:

    def __init__(self, model: Any, features: pd.DataFrame):
        self.__model = model
        self.__features = features

    """
    Validates the model using walk-forward time-series approach and returns predictions.
    """
    def validate(self, target: pd.DataFrame) -> pd.DataFrame:
        x, y, months = self.__process_to_equel_months(self.__features, target)
        result = self.__walk_forward(x, y, months)
        return result

    """
    Performs walk-forward validation by training on incremental subsets and predicting next month.
    """
    def __walk_forward(self, x: pd.DataFrame, y: pd.Series, months: list) -> pd.DataFrame:
        predictions = []
        actuals = []
        tested_months = []
        total = len(months)
        splits = [total // 2, (3 * total) // 4, total - 1]
        for split in splits:
            train_x = x.iloc[:split]
            train_y = y.iloc[:split]
            test_x = x.iloc[split:split + 1]
            test_y = y.iloc[split:split + 1]
            test_month = months[split]
            self.__model.fit(train_x, train_y)
            prediction = self.__model.predict(test_x)[0]
            predictions.append(prediction)
            actuals.append(test_y.values[0])
            tested_months.append(test_month)
        return pd.DataFrame({
            'Month': tested_months,
            'Actual': actuals,
            'Predicted': predictions
        })

    """
    Aligns features and target dataframes by common months and extracts relevant columns.
    """
    def __process_to_equel_months(self, features: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list]:
        common_months = sorted(set(features['Month']) & set(target['Month']))
        features = features[features['Month'].isin(common_months)].reset_index(drop=True)
        target = target[target['Month'].isin(common_months)].reset_index(drop=True)
        months = features['Month'].tolist()
        x = features.drop(columns=['Month'])
        y = target.iloc[:, 1]
        return x, y, months
