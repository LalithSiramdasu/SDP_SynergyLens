from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import requests

from backend.config import Config


PLACEHOLDER_KEYS = {
    "",
    "your_gemini_api_key",
    "your_gemini_api_key_here",
    "your_google_api_key",
    "paste_your_gemini_key_here",
    "your_first_gemini_api_key_here",
    "your_second_gemini_api_key_here",
    "your-api-key",
    "none",
    "null",
}


@dataclass(frozen=True)
class GeminiChatResult:
    answer: str
    attempted_key_count: int
    successful_key_index: int
    model_name: str


def configured_api_keys() -> list[str]:
    values: list[str] = []
    for env_name in ("GEMINI_API_KEYS", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        raw = os.getenv(env_name, "")
        values.extend(str(raw).replace("\n", ",").split(","))

    local_path = Config.GEMINI_KEYS_LOCAL_PATH
    if local_path.exists():
        try:
            values.extend(
                line.strip()
                for line in local_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            )
        except OSError:
            pass

    keys: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = str(value).strip()
        if key and key.lower() not in PLACEHOLDER_KEYS and key not in seen:
            keys.append(key)
            seen.add(key)
    return keys


def configured_key_count() -> int:
    return len(configured_api_keys())


def answer_with_gemini(
    *,
    question: str,
    mode: str,
    payload: dict[str, Any],
    fallback_answer: str,
    intent: str,
) -> GeminiChatResult | None:
    keys = configured_api_keys()
    if not keys:
        return None

    prompt = _build_chat_prompt(
        question=question,
        mode=mode,
        payload=payload,
        fallback_answer=fallback_answer,
        intent=intent,
    )
    model_name = Config.GEMINI_MODEL_NAME
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{quote(model_name, safe='')}:generateContent"
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.9,
            "maxOutputTokens": 900,
        },
    }

    for index, api_key in enumerate(keys, start=1):
        try:
            response = requests.post(
                url,
                params={"key": api_key},
                json=body,
                timeout=Config.GEMINI_REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code >= 400:
                continue
            answer = _extract_text(response.json())
        except (requests.RequestException, ValueError, KeyError, TypeError):
            continue

        if _is_usable_answer(answer):
            return GeminiChatResult(
                answer=answer,
                attempted_key_count=len(keys),
                successful_key_index=index,
                model_name=model_name,
            )
    return None


def _build_chat_prompt(
    *,
    question: str,
    mode: str,
    payload: dict[str, Any],
    fallback_answer: str,
    intent: str,
) -> str:
    context = _compact_payload(payload)
    return (
        "You are the SynergyLens assistant for a drug-combination synergy screening Flask app.\n"
        "Answer the user's question using only the supplied project or prediction context.\n"
        "Be specific, practical, and clear. Do not invent biological mechanisms, model metrics, citations, or clinical claims.\n"
        "Never give a shorter or less useful answer than the built-in fallback answer.\n"
        "Return complete sentences only; do not stop mid-sentence.\n"
        "For prediction mode, answer with these compact sections when relevant:\n"
        "Result: name the NSC pair, cell line, ComboScore, and label.\n"
        "Model context: name the model and directional scores if supplied.\n"
        "Drivers: summarize supplied feature or SHAP context; if absent, say Explain AI is needed for feature-level drivers.\n"
        "Next checks: give 2-3 experimental or workflow checks.\n"
        "Safety: state this is screening support, not medical advice.\n"
        "For project mode, explain the local app behavior and files/APIs only.\n\n"
        f"Mode: {mode or 'auto'}\n"
        f"Intent: {intent}\n"
        f"Question: {question}\n"
        f"Built-in fallback answer to preserve factual project behavior: {fallback_answer}\n"
        f"Context JSON: {json.dumps(context, ensure_ascii=True, default=str)}\n"
    )


def _compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    prediction = payload.get("prediction") if isinstance(payload, dict) else None
    explanation = payload.get("explanation") if isinstance(payload, dict) else None
    compact: dict[str, Any] = {
        "mode": payload.get("mode"),
        "question": payload.get("question"),
    }

    if isinstance(prediction, dict):
        compact["prediction"] = {
            "input": prediction.get("input"),
            "score": prediction.get("final_predicted_COMBOSCORE") or prediction.get("score"),
            "label": prediction.get("label") or prediction.get("prediction_label"),
            "model_used": prediction.get("model_used"),
            "model_type": prediction.get("model_type"),
            "forward": prediction.get("prediction_NSC1_to_NSC2"),
            "reverse": prediction.get("prediction_NSC2_to_NSC1"),
        }
    else:
        for key in ("NSC1", "NSC2", "CELLNAME", "drug1_id", "drug2_id", "cell_line"):
            if key in payload:
                compact[key] = payload[key]

    if isinstance(explanation, dict):
        compact["explanation"] = {
            "summary": explanation.get("explanation_summary") or explanation.get("plain_english_explanation"),
            "top_positive_contributors": _compact_features(explanation.get("top_positive_contributors")),
            "top_negative_contributors": _compact_features(explanation.get("top_negative_contributors")),
            "features": _compact_features(explanation.get("features")),
        }
    return compact


def _compact_features(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    compacted: list[dict[str, Any]] = []
    for item in value[:5]:
        if not isinstance(item, dict):
            continue
        compacted.append(
            {
                "feature": item.get("readable_feature") or item.get("feature") or item.get("feature_name"),
                "shap_value": item.get("shap_value") or item.get("impact"),
                "direction": item.get("direction"),
            }
        )
    return compacted


def _extract_text(data: dict[str, Any]) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    text = "\n".join(str(part.get("text", "")).strip() for part in parts if isinstance(part, dict))
    return text.strip()


def _is_usable_answer(answer: str) -> bool:
    text = str(answer or "").strip()
    if len(text) < 80:
        return False
    terminal = text.rstrip()[-1:]
    if terminal and terminal not in ".!?)]":
        return False
    lowered = text.lower()
    weak_fragments = (
        "i cannot answer",
        "i do not have enough context",
        "as an ai language model",
    )
    return not any(fragment in lowered for fragment in weak_fragments)
