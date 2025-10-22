import sys
from pathlib import Path
from xgboost import XGBRegressor
from typing import Any
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from src.model_validator import ModelValidator
from src.features import Features
from src.rawdata import RawData

def validate_model(model:Any, data_path:str, target:str) -> pd.DataFrame:
    result = (
        ModelValidator(
            model,
            (
                Features(
                    RawData(data_path)
                    .make_features()
                )
                .add_sin_seasonality(period=12)
                .add_cos_seasonality(period=12)
                .prepare_data()
            )
        )
        .validate(
            (
                RawData(data_path)
                .target(target)
            )
        )
    )
    return result

if __name__ == "__main__":
    print(
        f" Result of the test:\n{
        validate_model(
            model=XGBRegressor(),
            data_path="../data/raw_data.csv",
            target="Deals"
        )
        }"
    )