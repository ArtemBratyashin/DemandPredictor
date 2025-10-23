from pydantic import BaseModel, Field

"""
Response models for FastAPI endpoints.
Defines structures returned in API responses.
"""

class PredictionResponse(BaseModel):
    predicted_deals: float = Field(
        description="Predicted number of deals for the next month.",
        example=128.4
    )