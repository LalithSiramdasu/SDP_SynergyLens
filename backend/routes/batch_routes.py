from __future__ import annotations

from flask import Blueprint, request, send_file

from backend.services.batch_service import process_upload, resolve_download
from backend.utils.response import success_response


batch_bp = Blueprint("batch", __name__)


@batch_bp.post("/api/batch-predict")
@batch_bp.post("/api/batch")
def api_batch_predict():
    result = process_upload(request.files.get("file"))
    return success_response(result, **result)


@batch_bp.get("/api/download/<path:filename>")
def api_download(filename: str):
    path = resolve_download(filename)
    return send_file(path, mimetype="text/csv", as_attachment=True, download_name=path.name)

