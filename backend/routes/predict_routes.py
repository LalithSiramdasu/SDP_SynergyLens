from __future__ import annotations

from flask import Blueprint, request

from backend.services.prediction_service import predict_from_payload
from backend.utils.response import success_response


predict_bp = Blueprint("predict", __name__)


@predict_bp.post("/api/predict")
def api_predict():
    result = predict_from_payload(request.get_json(silent=True), delay_quantum_response=True)
    return success_response(result, **result)
