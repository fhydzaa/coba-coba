# ============================================================
# app.py
# ============================================================
import os
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from predictor import RULPredictor

import shutil


# ============================================================
# APP
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(
    title="KAN RUL Prediction API",
    description=(
        "Battery RUL prediction using "
        "pruned KAN and symbolic formula."
    ),
    version="1.0.0"
)

app.mount(
    "/static",
    StaticFiles(
        directory=str(
            BASE_DIR / "static"
        )
    ),
    name="static"
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
    return FileResponse(
        str(
            BASE_DIR /
            "static" /
            "index.html"
        )
    )


<<<<<<< HEAD
=======
@app.get("/health")
def health():
    return {
        "status": "ok"
    }

@app.get("/disk")
def disk():
    total, used, free = shutil.disk_usage("/")
    return {
        "total_gb": round(total / 1024**3, 2),
        "used_gb": round(used / 1024**3, 2),
        "free_gb": round(free / 1024**3, 2),
    }
>>>>>>> 208cf7dfdf92847ac8f05e3f93a415356d243f7c


@app.get("/storage")
def storage():
    paths = [
        "/app",
        "/opt/render/project/src",
    ]

    result = {}

    for path in paths:
        if os.path.exists(path):
            total_size = 0

            for root, dirs, files in os.walk(path):
                for file in files:
                    try:
                        total_size += os.path.getsize(
                            os.path.join(root, file)
                        )
                    except OSError:
                        pass

            result[path] = round(
                total_size / (1024 ** 3), 2
            )

    return result

@app.get("/storage/files")
def storage_files():
    base = "/opt/render/project/src"

    files = []

    for root, dirs, filenames in os.walk(base):
        for filename in filenames:
            path = os.path.join(root, filename)

            try:
                size = os.path.getsize(path)

                files.append({
                    "file": os.path.relpath(
                        path, base
                    ),
                    "size_mb": round(
                        size / (1024 ** 2),
                        2
                    )
                })

            except OSError:
                pass

    files.sort(
        key=lambda x: x["size_mb"],
        reverse=True
    )

    return files[:20]
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
