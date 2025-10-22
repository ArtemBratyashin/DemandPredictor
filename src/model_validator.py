import pandas as pd
from typing import Any, List, Tuple

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
        x, y, _ = self.__process_to_equel_months(self.__features, target)
        splits = self.__get_splits(len(x))
        results = [self.__train_and_predict(x, y, split, len(x)) for split in splits]
        return pd.DataFrame(results, columns=['TrainPart', 'Actual', 'Predicted', 'MAPE'])

    """
    Splits features into training and test sets for walk-forward validation.
    """
    def __get_splits(self, total: int) -> List[int]:
        return [total // 2, (3 * total) // 4, total - 2, total - 1]

    """
    Trains split and gets metrics of current model.
    """
    def __train_and_predict(self, x:pd.DataFrame, y:pd.Series, split:int, total:int) -> Tuple[str, float, float, float]:
        train_x, train_y = x.iloc[:split], y.iloc[:split]
        test_x, test_y = x.iloc[split:split + 1], y.iloc[split:split + 1]
        proportion = f"{split}/{total}"
        self.__model.fit(train_x, train_y)
        pred = self.__model.predict(test_x)[0]
        actual = test_y.values[0]
        mape = self.__mape(actual, pred)
        return proportion, actual, pred, mape

    """
    Calculates MAPE for a single actual and predicted value.
    """
    def __mape(self, actual, predicted) -> float:
        return None if actual == 0 else abs((actual - predicted)/actual)*100

    """
    Prepares feature and target for prediction.
    """
    def __process_to_equel_months(self, features: pd.DataFrame, target: pd.DataFrame) -> tuple:
        common_months = sorted(set(features['Month']) & set(target['Month']))
        features = features[features['Month'].isin(common_months)].reset_index(drop=True)
        target = target[target['Month'].isin(common_months)].reset_index(drop=True)
        months = features['Month'].tolist()
        x = features.drop(columns=['Month'])
        y = target.iloc[:, 1]
        return x, y, months