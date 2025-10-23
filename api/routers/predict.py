from fastapi import APIRouter, HTTPException
from api.models.response_models import PredictionResponse
from api.config import Config
from src.predictor import Predictor

"""
Router for prediction endpoints.

Handles /predict requests that return next month's deal count.
"""

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"]
)

"""
Returns predicted number of deals for next month.

Returns:
    PredictionResponse: JSON object with predicted_deals field.

Raises:
    HTTPException: If model or data files are missing or prediction fails.
"""

@router.get("/", response_model=PredictionResponse)
def get_prediction():
    try:
        config = Config()
        predictor = Predictor(config.model())
        prediction = predictor.predict()
        return PredictionResponse(predicted_deals=prediction)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")