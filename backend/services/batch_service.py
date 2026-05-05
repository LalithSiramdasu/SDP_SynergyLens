from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from backend.config import Config
from backend.services.prediction_service import predict_from_payload
from backend.services.validation_service import normalize_batch_columns
from backend.utils.errors import ValidationError
from backend.utils.paths import PREDICTIONS_DIR, UPLOADS_DIR, safe_runtime_path


def process_upload(file_storage: FileStorage | None) -> dict:
    if file_storage is None:
        raise ValidationError("No file uploaded.", code="MISSING_FILE")
    if not file_storage.filename:
        raise ValidationError("Uploaded file must have a filename.", code="MISSING_FILENAME")
    if not file_storage.filename.lower().endswith(".csv"):
        raise ValidationError("Only CSV uploads are supported.", code="INVALID_FILE_TYPE")

    upload_path = _save_upload(file_storage)
    try:
        frame = pd.read_csv(upload_path)
    except Exception as exc:
        raise ValidationError(f"Could not read CSV: {exc}", code="INVALID_CSV") from exc

    normalized = normalize_batch_columns(frame)
    output = predict_rows(normalized)
    output_path = _output_path()
    output.to_csv(output_path, index=False)

    successful = int((output["status"] == "success").sum()) if "status" in output.columns else 0
    total = int(len(output))
    failed = total - successful
    preview_frame = output.head(50).where(pd.notna(output.head(50)), None)
    preview = preview_frame.to_dict(orient="records")

    return {
        "total_rows": total,
        "successful_rows": successful,
        "failed_rows": failed,
        "output_file": output_path.name,
        "download_url": f"/api/download/{output_path.name}",
        "preview": preview,
        "uploaded_file": upload_path.name,
    }


def predict_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, row in frame.iterrows():
        output_row = row.to_dict()
        try:
            prediction = predict_from_payload(output_row)
            output_row.update(
                {
                    "prediction_NSC1_to_NSC2": prediction["prediction_NSC1_to_NSC2"],
                    "prediction_NSC2_to_NSC1": prediction["prediction_NSC2_to_NSC1"],
                    "final_predicted_COMBOSCORE": prediction["final_predicted_COMBOSCORE"],
                    "prediction_label": prediction["prediction_label"],
                    "status": "success",
                    "error": "",
                }
            )
        except Exception as exc:
            output_row.update(
                {
                    "prediction_NSC1_to_NSC2": None,
                    "prediction_NSC2_to_NSC1": None,
                    "final_predicted_COMBOSCORE": None,
                    "prediction_label": "Error",
                    "status": "failed",
                    "error": str(exc),
                }
            )
        rows.append(output_row)
    return pd.DataFrame(rows)


def resolve_download(filename: str) -> Path:
    path = safe_runtime_path(PREDICTIONS_DIR, filename)
    if not path.exists() or not path.is_file():
        raise ValidationError("Requested prediction output was not found.", code="DOWNLOAD_NOT_FOUND", status_code=404)
    return path


def _save_upload(file_storage: FileStorage) -> Path:
    original = secure_filename(file_storage.filename or "batch.csv")
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    filename = f"upload_{timestamp}_{uuid4().hex[:8]}_{original}"
    path = safe_runtime_path(UPLOADS_DIR, filename)
    file_storage.save(path)
    return path


def _output_path() -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    filename = f"batch_predictions_{timestamp}_{uuid4().hex[:8]}.csv"
    return safe_runtime_path(Config.PREDICTIONS_DIR, filename)

