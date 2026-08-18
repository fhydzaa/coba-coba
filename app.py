# ============================================================
# app.py
# ============================================================

from typing import List, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from predictor import RULPredictor


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="KAN RUL Prediction API",
    description=(
        "Battery RUL prediction using "
        "pruned KAN and symbolic formula."
    ),
    version="1.0.0"
)


# ============================================================
# LOAD MODEL SEKALI
# ============================================================

predictor = RULPredictor(
    model_dir="."
)


# ============================================================
# REQUEST
# ============================================================

class PredictionRequest(BaseModel):

    data: List[Dict[str, float]]


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "status": "ok",
        "service": "KAN RUL Prediction API"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }

# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(
    request: PredictionRequest
):

    try:

        result = predictor.predict(
            request.data
        )

        return {
            "success": True,
            **result
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )