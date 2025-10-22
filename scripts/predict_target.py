import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.predictor import Predictor

def predict_target(model_folder:str) -> int|float:
    result = (
        Predictor(
            model_folder = model_folder
        )
        .predict()
    )
    return result

if __name__ == "__main__":
    print(
        f"Predicted number is: {
            int(
                predict_target(
                    model_folder="../saved_models/xgb_model"
                )
            )
        }."
    )