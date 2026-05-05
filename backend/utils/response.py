from __future__ import annotations

from typing import Any

from flask import jsonify


def success_response(data: Any = None, status_code: int = 200, **extra: Any):
    payload: dict[str, Any] = {"success": True, "data": data if data is not None else {}}
    payload.update(extra)
    return jsonify(payload), status_code


def error_response(
    code: str,
    message: str,
    status_code: int = 400,
    details: Any = None,
    **extra: Any,
):
    error: dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        error["details"] = details

    payload: dict[str, Any] = {
        "success": False,
        "error": error,
        "message": message,
    }
    payload.update(extra)
    return jsonify(payload), status_code

