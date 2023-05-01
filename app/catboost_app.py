import sys
from datetime import datetime
from functools import wraps
from http import HTTPStatus
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, Request

from app.schemas import Payload

# TODO: Fix this!
sys.path.append(Path(__file__).parent.parent.absolute().as_posix())

from config import config
from config.config import logger
from vin_decoder import main, predict

# Define application
app = FastAPI(
    title="VIN decoder",
    description="Identify model of a vehicle based on its VIN number.",
    version="0.1",
)


@app.on_event("startup")
def load_artifacts():
    global vectorizer, model
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read().strip()
    vectorizer, model = main.load_artifacts(run_id=run_id)
    logger.info("Ready for inference!")


def construct_response(f):
    """Construct a JSON response for an endpoint."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs) -> Dict:
        results = f(request, *args, **kwargs)
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }
        if "data" in results:
            response["data"] = results["data"]
        return response

    return wrap


@app.get("/", tags=["General"])
@construct_response
def _index(request: Request) -> Dict:
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response


@app.post("/predict", tags=["Prediction"])
@construct_response
def _predict(request: Request, payload: Payload) -> Dict:
    """Predict vehicle models for a list of vins."""
    vins = [item.vin for item in payload.vins]
    predictions = predict.predict(vectorizer=vectorizer, model=model, vins=vins)
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "vins": vins,
            "predictions": predictions,
        },
    }
    logger.info(response)
    return response
