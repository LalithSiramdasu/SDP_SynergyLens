from __future__ import annotations

from flask import Blueprint

from backend.config import Config
from backend.services import artifact_loader
from backend.utils.response import success_response


health_bp = Blueprint("health", __name__)


@health_bp.get("/api/health")
def api_health():
    status = artifact_loader.artifact_status(load=False)
    quantum_status = artifact_loader.quantum_artifact_status(load=False)
    artifacts = {
        "model_loaded": bool(status.get("model_loaded")),
        "quantum_model_available": bool(quantum_status.get("quantum_model_exists")),
        "feature_columns_loaded": bool(status.get("feature_columns_loaded")),
        "drug_lookup_loaded": bool(status.get("drug_lookup_loaded")),
        "cell_line_lookup_loaded": bool(status.get("cell_line_lookup_loaded")),
        "molecules_loaded": bool(status.get("molecules_loaded")),
    }
    errors = status.get("errors", [])

    available_drugs = 0
    available_cell_lines = 0
    feature_count = 0
    if artifacts["drug_lookup_loaded"]:
        available_drugs = int(artifact_loader.load_drug_name_map().shape[0])
    if artifacts["cell_line_lookup_loaded"]:
        available_cell_lines = int(artifact_loader.load_cell_line_features().shape[0])
    if artifacts["feature_columns_loaded"]:
        feature_count = int(len(artifact_loader.load_feature_columns()))

    data = {
        "status": "ok" if not errors else "degraded",
        "app": Config.APP_NAME,
        "artifacts": artifacts,
    }
    return success_response(
        data,
        status="success" if not errors else "error",
        available_drugs=available_drugs,
        available_cell_lines=available_cell_lines,
        feature_column_count=feature_count,
        model_count=artifact_loader.deployed_model_count(),
        model_loaded=artifacts["model_loaded"],
        shap_available=artifacts["model_loaded"],
        chat_backend="Gemini + Built-in Guide" if artifact_loader.configured_gemini_key_count() else "Built-in Guide",
        errors=errors,
    )
