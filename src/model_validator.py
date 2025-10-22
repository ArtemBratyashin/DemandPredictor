import pandas as pd
from typing import Any

"""
Model validation class that tests model performance using time-series walk-forward validation.
Returns proportion of training samples, predictions, actuals and MAPE for every step.

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

class ModelValidator:

    def __init__(self, model: Any, features: pd.DataFrame):
        self.__model = model
        self.__features = features

    """
    Validates the model using walk-forward time-series approach and returns predictions, proportions and MAPE.
    """
    def validate(self, target: pd.DataFrame) -> pd.DataFrame:
        x, y = self.__process_to_equel_months(self.__features, target)
        result = self.__walk_forward(x, y)
        return result

    """
    Performs walk-forward validation by training on incremental subsets and predicting next month.
    Adds proportion column like '6/12', '9/12', etc. and also MAPE column.
    """
    def __walk_forward(self, x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        predictions = []
        actuals = []
        mape_scores = []
        proportions = []
        total = len(x)
        splits = [total // 2, (3 * total) // 4, total - 2, total - 1]
        for split in splits:
            train_x = x.iloc[:split]
            train_y = y.iloc[:split]
            test_x = x.iloc[split:split + 1]
            test_y = y.iloc[split:split + 1]
            proportion = f"{split}/{total}"
            self.__model.fit(train_x, train_y)
            prediction = self.__model.predict(test_x)[0]
            actual = test_y.values[0]
            mape = self.__mape(actual, prediction)
            predictions.append(prediction)
            actuals.append(actual)
            mape_scores.append(mape)
            proportions.append(proportion)
        return pd.DataFrame({
            'TrainPart': proportions,
            'Actual': actuals,
            'Predicted': predictions,
            'MAPE': mape_scores
        })

    """
    Calculates MAPE for a single actual and predicted value.
    """
    def __mape(self, actual, predicted):
        if actual == 0:
            return None
        return abs((actual - predicted) / actual) * 100

    """
    Aligns features and target dataframes by common months and extracts relevant columns.
    """
    def __process_to_equel_months(self, features: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        common_months = sorted(set(features['Month']) & set(target['Month']))
        features = features[features['Month'].isin(common_months)].reset_index(drop=True)
        target = target[target['Month'].isin(common_months)].reset_index(drop=True)
        x = features.drop(columns=['Month'])
        y = target.iloc[:, 1]
        return x, y
