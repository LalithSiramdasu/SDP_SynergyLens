from __future__ import annotations

from flask import Blueprint, request

from backend.services import metadata_service
from backend.utils.response import success_response


about_bp = Blueprint("about", __name__)


@about_bp.get("/api/about")
def api_about():
    data = metadata_service.about_metadata()
    return success_response(data, **data)


@about_bp.get("/api/system-summary")
def api_system_summary():
    data = metadata_service.system_summary()
    return success_response(data, **data)


@about_bp.get("/api/model-performance-summary")
def api_model_performance_summary():
    data = metadata_service.model_performance_summary()
    return success_response(data, **data)


@about_bp.get("/api/drugs")
def api_drugs():
    query = request.args.get("q", "")
    limit = request.args.get("limit", "all")
    drugs = metadata_service.drug_records(query=query, limit=limit)
    return success_response({"drugs": drugs, "count": len(drugs)}, drugs=drugs, items=drugs, count=len(drugs))


@about_bp.get("/api/cell-lines")
def api_cell_lines():
    cell_lines = metadata_service.available_cell_lines()
    return success_response(
        {"cell_lines": cell_lines, "count": len(cell_lines)},
        cell_lines=cell_lines,
        items=cell_lines,
        count=len(cell_lines),
    )


@about_bp.get("/api/demo-cases")
def api_demo_cases():
    data = metadata_service.demo_cases()
    return success_response(data, **data)
