from __future__ import annotations

from flask import Blueprint, request

from backend.services.explain_service import answer_chat_question, explain_prediction
from backend.utils.response import success_response


explain_bp = Blueprint("explain", __name__)


@explain_bp.post("/api/explain")
def api_explain():
    result = explain_prediction(request.get_json(silent=True) or {})
    return success_response(result, **result)


@explain_bp.post("/api/chat")
def api_chat():
    result = answer_chat_question(request.get_json(silent=True) or {})
    return success_response(result, **result)

