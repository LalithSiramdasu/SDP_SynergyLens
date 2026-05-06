from __future__ import annotations

from typing import Any

import numpy as np

from backend.services import artifact_loader, gemini_service
from backend.services.prediction_service import (
    build_feature_vector,
    get_drug_name,
    predict_from_payload,
    predict_pair,
    simple_interpret,
)
from backend.services.validation_service import normalize_prediction_payload

try:
    import xgboost as xgb
except Exception:  # pragma: no cover - optional dependency guard
    xgb = None


DISCLAIMER = "This is not medical advice. Predictions should be validated experimentally."
LIMITATION_NOTE = (
    "This explanation reflects model feature contributions for the deployed artifact; "
    "it does not prove a biological mechanism."
)
QUANTUM_SURROGATE_LABEL = "Quantum prediction + classical surrogate XAI"
QUANTUM_SURROGATE_NOTE = (
    "The prediction score is from the quantum model. Feature impacts are shown using a classical "
    "surrogate explanation because native quantum SHAP is not available."
)
QUANTUM_SURROGATE_SUMMARY = (
    "Quantum prediction was selected. Since native feature-level quantum SHAP is not available, "
    "this XAI map uses the classical model as a surrogate explanation for the same drug pair and "
    "cell line. The prediction score remains from the quantum pipeline."
)
QUANTUM_SURROGATE_LIMITATION = "This is a surrogate explanation, not a native explanation of the quantum model."


def explain_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_prediction_payload(payload)
    prediction = predict_pair(**normalized)
    if normalized.get("model_type") == "quantum":
        final_score = float(prediction["final_predicted_COMBOSCORE"])
        try:
            input_frame = build_feature_vector(
                normalized["nsc1"],
                normalized["nsc2"],
                normalized["cell_line"],
                normalized.get("cancer_type") or None,
            )
            features, base_value = get_feature_contributions(input_frame, limit=14)
            positive = [feature for feature in features if feature["shap_value"] >= 0][:7]
            negative = [feature for feature in features if feature["shap_value"] < 0][:7]
        except Exception as exc:
            return {
                "input": prediction["input"],
                "prediction": round(final_score, 3),
                "final_predicted_COMBOSCORE": round(final_score, 3),
                "prediction_label": prediction["prediction_label"],
                "model_used": prediction["model_used"],
                "model_type": "quantum",
                "base_value": None,
                "expected_value": None,
                "features": [],
                "top_positive_contributors": [],
                "top_negative_contributors": [],
                "top_synergy_drivers": [],
                "top_antagonism_drivers": [],
                "plain_english_explanation": QUANTUM_SURROGATE_SUMMARY,
                "explanation_summary": QUANTUM_SURROGATE_SUMMARY,
                "model_limitation_note": QUANTUM_SURROGATE_LIMITATION,
                "limitation_note": QUANTUM_SURROGATE_LIMITATION,
                "surrogate_note": QUANTUM_SURROGATE_NOTE,
                "disclaimer": DISCLAIMER,
                "suggestion": QUANTUM_SURROGATE_NOTE,
                "explanation_available": False,
                "explanation_type": "classical_surrogate_for_quantum",
                "explanation_method": "Classical surrogate XAI",
                "explanation_label": QUANTUM_SURROGATE_LABEL,
                "quantum_prediction_used": True,
                "surrogate_explanation_used": False,
                "surrogate_error": str(exc),
            }

        return {
            "input": prediction["input"],
            "prediction": round(final_score, 3),
            "final_predicted_COMBOSCORE": round(final_score, 3),
            "prediction_label": prediction["prediction_label"],
            "model_used": prediction["model_used"],
            "model_type": "quantum",
            "base_value": round(base_value, 3) if base_value is not None else None,
            "expected_value": round(base_value, 3) if base_value is not None else None,
            "features": features,
            "top_positive_contributors": positive,
            "top_negative_contributors": negative,
            "top_synergy_drivers": positive,
            "top_antagonism_drivers": negative,
            "plain_english_explanation": f"{QUANTUM_SURROGATE_SUMMARY} {QUANTUM_SURROGATE_NOTE}",
            "explanation_summary": QUANTUM_SURROGATE_SUMMARY,
            "model_limitation_note": QUANTUM_SURROGATE_LIMITATION,
            "limitation_note": QUANTUM_SURROGATE_LIMITATION,
            "surrogate_note": QUANTUM_SURROGATE_NOTE,
            "disclaimer": DISCLAIMER,
            "suggestion": QUANTUM_SURROGATE_NOTE,
            "explanation_available": bool(features),
            "explanation_type": "classical_surrogate_for_quantum",
            "explanation_method": "Classical surrogate XAI",
            "explanation_label": QUANTUM_SURROGATE_LABEL,
            "quantum_prediction_used": True,
            "surrogate_explanation_used": bool(features),
        }

    input_frame = build_feature_vector(
        normalized["nsc1"],
        normalized["nsc2"],
        normalized["cell_line"],
        normalized.get("cancer_type") or None,
    )
    features, base_value = get_feature_contributions(input_frame, limit=14)
    positive = [feature for feature in features if feature["shap_value"] >= 0][:7]
    negative = [feature for feature in features if feature["shap_value"] < 0][:7]
    final_score = float(prediction["final_predicted_COMBOSCORE"])

    result = {
        "input": prediction["input"],
        "prediction": round(final_score, 3),
        "final_predicted_COMBOSCORE": round(final_score, 3),
        "prediction_label": prediction["prediction_label"],
        "model_used": prediction["model_used"],
        "model_type": normalized.get("model_type", "classical"),
        "base_value": round(base_value, 3) if base_value is not None else None,
        "expected_value": round(base_value, 3) if base_value is not None else None,
        "features": features,
        "top_positive_contributors": positive,
        "top_negative_contributors": negative,
        "top_synergy_drivers": positive,
        "top_antagonism_drivers": negative,
        "plain_english_explanation": build_plain_explanation(final_score, positive, negative),
        "explanation_summary": build_plain_explanation(final_score, positive, negative),
        "model_limitation_note": LIMITATION_NOTE,
        "disclaimer": DISCLAIMER,
        "suggestion": "Review the largest positive and negative contributors, then validate promising combinations experimentally.",
    }
    return result


def get_feature_contributions(input_frame, limit: int = 10) -> tuple[list[dict[str, Any]], float | None]:
    feature_columns = artifact_loader.load_feature_columns()
    booster = artifact_loader.get_booster()
    base_value = None

    if booster is not None and xgb is not None:
        contribs = booster.predict(xgb.DMatrix(input_frame), pred_contribs=True)
        values = np.asarray(contribs[0, :-1], dtype=float)
        base_value = float(contribs[0, -1])
    else:
        model = artifact_loader.load_model()
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            values = np.zeros(len(feature_columns), dtype=float)
        else:
            values = np.asarray(importances, dtype=float)
            if len(values) != len(feature_columns):
                values = np.resize(values, len(feature_columns))

    top_indexes = np.argsort(np.abs(values))[-limit:]
    pairs = sorted(
        [(feature_columns[index], values[index]) for index in top_indexes],
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    row = input_frame.iloc[0]
    return [
        feature_record(feature_name, shap_value, row.get(feature_name, ""))
        for feature_name, shap_value in pairs
    ], base_value


def feature_record(feature_name: str, shap_value: float, feature_value: Any) -> dict[str, Any]:
    readable = label_feature(feature_name)
    value = round(float(shap_value), 4)
    return {
        "feature_name": feature_name,
        "raw_feature": feature_name,
        "readable_feature": readable,
        "feature": readable,
        "feature_value": _json_value(feature_value),
        "value": _json_value(feature_value),
        "shap_value": value,
        "shap": value,
        "impact": value,
        "direction": "increases synergy score" if value >= 0 else "decreases synergy score",
        "description": get_feature_description(feature_name),
    }


def label_feature(column_name: str) -> str:
    if column_name == "drug1":
        return "Drug 1 identity"
    if column_name == "drug2":
        return "Drug 2 identity"
    if column_name == "cancer_encoded":
        return "Cancer type"
    if column_name.startswith("DD_maccs_"):
        return f"Drug fingerprint bit {column_name.replace('DD_maccs_', '')}"
    if column_name.startswith("RNAi_"):
        return column_name.replace("RNAi_", "").replace(" - Homo sapiens (human)", "").strip()
    if column_name.startswith("DDP_"):
        return column_name.replace("DDP_", "").replace(" - Homo sapiens (human)", "").strip()
    if column_name.startswith("DD_tox_"):
        return column_name.replace("DD_tox_", "Drug toxicity association: ")
    return column_name.replace("_", " ").title()


def get_feature_description(feature_name: str) -> str:
    feature_info = artifact_loader.load_feature_info()
    row = feature_info[feature_info["Feature Name"].astype(str) == str(feature_name)]
    if row.empty:
        return ""
    return str(row.iloc[0]["Description"])


def build_plain_explanation(score: float, positive: list[dict[str, Any]], negative: list[dict[str, Any]]) -> str:
    direction = simple_interpret(score)
    positive_names = ", ".join(item["readable_feature"] for item in positive[:3]) or "no strong positive contributors"
    negative_names = ", ".join(item["readable_feature"] for item in negative[:3]) or "no strong negative contributors"
    return (
        f"The model predicts a {direction} interaction with ComboScore {score:.3f}. "
        f"The strongest upward contributors include {positive_names}. "
        f"The strongest downward contributors include {negative_names}. "
        f"{LIMITATION_NOTE} {DISCLAIMER}"
    )


def answer_chat_question(payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = payload or {}
    question = str(payload.get("question") or "Explain this prediction").strip()
    mode = str(payload.get("mode") or "").lower()

    if mode == "project" or "prediction" not in payload and not any(key in payload for key in ("drug1_id", "NSC1")):
        local_answer = project_answer(question)
    else:
        local_answer = prediction_answer(payload, question)

    intent = classify_question(question)
    suggestions = suggested_questions(mode, question)
    gemini_result = gemini_service.answer_with_gemini(
        question=question,
        mode=mode,
        payload=payload,
        fallback_answer=local_answer,
        intent=intent,
    )

    if gemini_result is not None:
        return {
            "question": question,
            "intent": intent,
            "answer": gemini_result.answer,
            "context": gemini_result.answer,
            "used_fallback": False,
            "llm_used": True,
            "llm_backend": "gemini",
            "provider_label": "AI Enhanced",
            "attempted_key_count": gemini_result.attempted_key_count,
            "successful_key_index": gemini_result.successful_key_index,
            "suggested_questions": suggestions,
        }

    return {
        "question": question,
        "intent": intent,
        "answer": local_answer,
        "context": local_answer,
        "used_fallback": True,
        "llm_used": False,
        "llm_backend": "local-fallback",
        "provider_label": "Built-in Guide",
        "attempted_key_count": artifact_loader.configured_gemini_key_count(),
        "successful_key_index": None,
        "suggested_questions": suggestions,
    }


def prediction_answer(payload: dict[str, Any], question: str) -> str:
    prediction = payload.get("prediction")
    explanation = payload.get("explanation") or {}

    if prediction:
        input_block = prediction.get("input", {})
        nsc1 = input_block.get("NSC1")
        nsc2 = input_block.get("NSC2")
        cell_line = input_block.get("CELLNAME")
        score = float(prediction.get("final_predicted_COMBOSCORE") or prediction.get("score") or 0)
        label = prediction.get("label") or prediction.get("prediction_label") or simple_interpret(score)
        model = prediction.get("model_used") or prediction.get("model_type") or "selected model"
        model_type = prediction.get("model_type") or input_block.get("model_type") or "classical"
        forward = prediction.get("prediction_NSC1_to_NSC2")
        reverse = prediction.get("prediction_NSC2_to_NSC1")
    else:
        computed = predict_from_payload(payload)
        nsc1 = computed["NSC1"]
        nsc2 = computed["NSC2"]
        cell_line = computed["CELLNAME"]
        score = float(computed["final_predicted_COMBOSCORE"])
        label = computed["prediction_label"]
        model = computed.get("model_used") or computed.get("model_type") or "selected model"
        model_type = computed.get("model_type") or "classical"
        forward = computed.get("prediction_NSC1_to_NSC2")
        reverse = computed.get("prediction_NSC2_to_NSC1")

    features = (
        explanation.get("top_positive_contributors")
        or explanation.get("features")
        or explanation.get("top_synergy_drivers")
        or []
    )[:3]
    negative_features = (explanation.get("top_negative_contributors") or explanation.get("top_antagonism_drivers") or [])[:2]
    feature_text = ", ".join(
        str(item.get("readable_feature") or item.get("feature") or item.get("feature_name"))
        for item in features
        if item
    )
    negative_feature_text = ", ".join(
        str(item.get("readable_feature") or item.get("feature") or item.get("feature_name"))
        for item in negative_features
        if item
    )
    lowered = str(question or "").lower()
    score_meaning = (
        "Positive scores above +4 are interpreted as synergistic, scores from -4 to +4 as neutral or weak, "
        "and scores below -4 as antagonistic."
    )
    direction_sentence = (
        f" The directional estimates were NSC1 -> NSC2: {forward} and NSC2 -> NSC1: {reverse}; "
        "the displayed ComboScore is the averaged final score."
        if forward is not None and reverse is not None
        else " Directional estimates were not included in this chat payload."
    )
    feature_sentence = (
        f" The strongest supplied upward drivers are {feature_text}."
        if feature_text
        else " Feature-level drivers were not included in this chat payload; run Explain AI for SHAP-style contributors."
    )
    if negative_feature_text:
        feature_sentence += f" Supplied downward drivers include {negative_feature_text}."

    if "clinical" in lowered or "advice" in lowered or "safe" in lowered or "treat" in lowered:
        return (
            f"Safety: this result is screening support only for NSC {nsc1} + NSC {nsc2} in {cell_line}. "
            f"The model predicts ComboScore {score:.3f} ({label}), but it is not biological proof, dosing guidance, "
            f"or a treatment recommendation. {DISCLAIMER}"
        )
    if "direction" in lowered or "both" in lowered or "average" in lowered:
        return (
            f"Direction check: SynergyLens evaluates NSC {nsc1} -> NSC {nsc2} and NSC {nsc2} -> NSC {nsc1} because "
            "drug-pair feature ordering can change the model input. "
            f"{direction_sentence} This makes the final result less dependent on which compound was entered first. {DISCLAIMER}"
        )
    if "model" in lowered:
        return (
            f"Model context: this prediction used {model} ({model_type}). For NSC {nsc1} + NSC {nsc2} in {cell_line}, "
            f"it returned ComboScore {score:.3f}, labeled {label}.{direction_sentence} {DISCLAIMER}"
        )
    if "feature" in lowered or "shap" in lowered or "caused" in lowered or "driver" in lowered:
        return (
            f"Drivers: for NSC {nsc1} + NSC {nsc2} in {cell_line}, ComboScore {score:.3f} was labeled {label}."
            f"{feature_sentence} These contributors describe model behavior, not proven biological mechanisms. {DISCLAIMER}"
        )

    return (
        f"Result: NSC {nsc1} + NSC {nsc2} in {cell_line} has predicted ComboScore {score:.3f}, labeled {label}. "
        f"That is a positive synergy screen for this cell-line context. "
        f"Model context: {model} ({model_type}) generated the result.{direction_sentence} "
        f"Interpretation: {score_meaning}{feature_sentence} "
        "Next checks: confirm the NSC IDs and cell line, run Explain AI for feature drivers, compare against known controls, "
        f"and validate promising combinations experimentally. Safety: {DISCLAIMER}"
    )


def project_answer(question: str) -> str:
    lowered = question.lower()
    if "batch" in lowered or "csv" in lowered:
        return "Batch prediction accepts a CSV with NSC1, NSC2, and CELLNAME columns, processes each row independently, and saves a result CSV for download."
    if "molecule" in lowered or "compound" in lowered or "structure" in lowered:
        return "Molecule lookup uses the local molecules/drug_mols.pkl RDKit artifact when a matching NSC is available, then returns structure SVG and molecular metadata."
    if "shap" in lowered or "xai" in lowered or "explain" in lowered:
        return f"Explain AI ranks feature contributions from the deployed model and provides a plain-English fallback explanation. {LIMITATION_NOTE} {DISCLAIMER}"
    if "score" in lowered or "comboscore" in lowered or "threshold" in lowered:
        return "ComboScore is interpreted with the deployed thresholds: scores above 10 are strong synergy, above 5 moderate synergy, above 0 mild synergy, above -5 neutral/additive, and -5 or below antagonistic."
    if "model" in lowered or "backend" in lowered or "file" in lowered:
        return "The backend loads the XGBoost model, ordered feature columns, drug fingerprints, cell-line features, drug-name map, molecule pickle, and explanation CSV files from project-relative folders."
    if "clinical" in lowered or "advice" in lowered or "trust" in lowered or "safety" in lowered:
        return f"SynergyLens is a screening tool, not clinical advice. {DISCLAIMER}"
    return "SynergyLens is a Flask workspace for drug-combination synergy screening with prediction, molecule lookup, Explain AI, batch CSV processing, health, and about metadata APIs."


def classify_question(question: str) -> str:
    lowered = str(question or "").lower()
    if "why" in lowered or "explain" in lowered:
        return "explanation"
    if "score" in lowered or "mean" in lowered or "comboscore" in lowered:
        return "score"
    if "factor" in lowered or "feature" in lowered or "shap" in lowered or "contribute" in lowered:
        return "features"
    if "how" in lowered:
        return "workflow"
    return "general"


def suggested_questions(mode: str, question: str) -> list[str]:
    if mode == "project":
        return [
            "How does prediction work?",
            "What CSV columns are required?",
            "What does Explain AI show?",
            "Is this clinical advice?",
        ]
    return [
        "What does this score mean?",
        "Which features caused this?",
        "Why predict both directions?",
        "Is this clinical advice?",
    ]


def drug_context(nsc: int) -> dict[str, Any]:
    drug_info = artifact_loader.load_explain_drug_info()
    row = drug_info[drug_info["drug_id"].astype(int) == int(nsc)]
    if row.empty:
        return {"name": get_drug_name(nsc), "mechanism": "Unknown", "class": "Unknown"}
    return {
        "name": str(row.iloc[0].get("name", get_drug_name(nsc))),
        "mechanism": str(row.iloc[0].get("mechanism", "Unknown")),
        "class": str(row.iloc[0].get("class", "Unknown")),
    }


def _json_value(value: Any):
    try:
        if np.isnan(value):
            return None
    except Exception:
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value
