import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.predictor import Predictor

def predict_target(models_folder:str, model_name:str):
    result = (
        Predictor(
            models_folder = models_folder,
            model_name = model_name
        )
        .predict()
    )
    return result

if __name__ == "__main__":
    print(
        f"Predicted number is: {
            predict_target(
                models_folder="../saved_models", 
                model_name="xgb_model"
            )
        }"
    )