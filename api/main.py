import uvicorn
from fastapi import FastAPI
from api.routers import predict
from fastapi.middleware.cors import CORSMiddleware

"""
FastAPI application entry point.

Usage:
    uvicorn api.main:app --reload
"""

app = FastAPI(
    title="Demand Predictor API",
    description="API for predicting number of deals for next month",
    version="1.0.0"
)
app.include_router(predict.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)