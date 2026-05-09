from __future__ import annotations

from typing import Any

import pandas as pd

from backend.config import Config
from backend.services import artifact_loader
from backend.services.prediction_service import score_thresholds
from backend.services.validation_service import (
    get_drug_display_name,
    is_predictable_cell_line,
    is_predictable_drug,
    normalize_drug_id,
    normalize_model_type,
    resolve_cell_line,
)


DEFAULT_CANCER_TYPES = [
    "Bladder Cancer",
    "Breast Cancer",
    "Colon Cancer",
    "Gastric Cancer",
    "Leukemia",
    "Lung Cancer",
    "Melanoma",
    "Ovarian Cancer",
    "Prostate Cancer",
]


CURATED_DEMO_CASES = [
    {
        "case_type": "antagonistic",
        "NSC1": 740,
        "NSC2": 754143,
        "CELLNAME": "OVCAR-3",
        "cancer_type": "Ovarian Cancer",
        "real_score": -10.44444444,
        "cut4": 0,
    },
    {
        "case_type": "neutral",
        "NSC1": 82151,
        "NSC2": 119875,
        "CELLNAME": "OVCAR-3",
        "cancer_type": "Ovarian Cancer",
        "real_score": 1.111111111,
        "cut4": 0,
    },
    {
        "case_type": "synergistic",
        "NSC1": 3053,
        "NSC2": 180973,
        "CELLNAME": "OVCAR-3",
        "cancer_type": "Ovarian Cancer",
        "real_score": 30.66666667,
        "cut4": 1,
    },
]


def home_context() -> dict[str, Any]:
    return {
        "cell_lines": [],
        "cancers": DEFAULT_CANCER_TYPES,
        "drugs": [],
    }


def quantum_supported_cell_lines() -> list[str]:
    available = set(artifact_loader.load_cell_line_directory().values())
    return [cell_line for cell_line in Config.QUANTUM_SUPPORTED_CELL_LINES if cell_line in available]


def available_cell_lines(model_type: str = "classical") -> list[str]:
    normalized_model_type = normalize_model_type(model_type)
    if normalized_model_type == "quantum":
        return quantum_supported_cell_lines()
    return sorted(artifact_loader.load_cell_line_directory().values())


def drug_records(query: str = "", limit: str | int = "20") -> list[dict[str, Any]]:
    records = list(artifact_loader.load_drug_directory().values())
    if query:
        search = str(query).strip().lower().replace("nsc", "").strip()
        records = [
            record
            for record in records
            if search in record["id"].lower() or search in record["name"].lower()
        ]
    records = sorted(records, key=lambda item: item["name"].lower())

    if str(limit).lower() != "all":
        try:
            max_rows = max(1, min(int(limit), 100))
        except ValueError:
            max_rows = 20
        records = records[:max_rows]

    return [
        {
            "id": str(record["id"]),
            "name": str(record["name"]),
            "NSC": int(record["id"]),
            "nsc": str(record["id"]),
            "label": f"{record['name']} - NSC {record['id']}",
            "sources": record.get("sources", []),
        }
        for record in records
    ]


def known_cancers() -> list[str]:
    values = set(DEFAULT_CANCER_TYPES)
    encoder = artifact_loader.load_cancer_encoder()
    if encoder is not None:
        values.update(str(item) for item in encoder.classes_)

    sample_path = Config.DATA_DIR / Config.SAMPLE_BATCH_FILE
    if sample_path.exists():
        try:
            sample = pd.read_csv(sample_path)
            for column in ("cancer_type", "cancer"):
                if column in sample.columns:
                    values.update(sample[column].dropna().astype(str).tolist())
        except Exception:
            pass

    cleaned = sorted({item.strip() for item in values if str(item).strip()})
    return cleaned


def about_metadata() -> dict[str, Any]:
    drug_directory = artifact_loader.load_drug_directory()
    cell_directory = artifact_loader.load_cell_line_directory()
    feature_columns = artifact_loader.load_feature_columns()
    molecule_status = artifact_loader.molecule_artifact_status(load=False)

    artifacts = artifact_loader.detected_artifacts()
    return {
        "app_name": Config.APP_NAME,
        "architecture": "Modular monolithic Flask backend using app factory, blueprints, services, utilities, validation, and cached artifact loading.",
        "detected_model_files": [item["name"] for item in artifacts.get("models", []) if item["name"].endswith(".pkl")],
        "detected_dataset_files": [item["name"] for group in ("models", "data") for item in artifacts.get(group, []) if item["name"].endswith(".csv")],
        "number_of_available_drugs": int(len(drug_directory)),
        "number_of_available_cell_lines": int(len(cell_directory)),
        "quantum_supported_cell_lines": quantum_supported_cell_lines(),
        "quantum_supported_cell_line_count": len(quantum_supported_cell_lines()),
        "number_of_feature_columns": int(len(feature_columns)),
        "molecule_data_availability": {
            "available": bool(molecule_status["available"]),
            "loaded": bool(molecule_status["loaded"]),
            "count": molecule_status["count"],
            "file": Config.MOLECULES_FILE,
            "size_bytes": molecule_status["size_bytes"],
            "lazy_loaded": True,
        },
        "explanation_data_availability": {
            "feature_dictionary_loaded": True,
            "drug_explanation_loaded": True,
            "feature_dictionary_file": Config.FEATURE_INFO_FILE,
            "drug_explanation_file": Config.EXPLAIN_DRUG_FILE,
        },
        "artifacts": artifacts,
    }


def model_performance_summary() -> dict[str, Any]:
    drug_count = len(artifact_loader.load_drug_directory())
    cell_count = len(artifact_loader.load_cell_line_directory())
    feature_count = len(artifact_loader.load_feature_columns())
    model_type_counts = artifact_loader.deployed_model_type_counts()
    model_count = sum(model_type_counts.values())
    model_info = {
        "model_type": "XGBoost + Quantum" if model_type_counts.get("Quantum") else "XGBoost",
        "model_file": Config.PRIMARY_MODEL_FILE,
        "feature_column_count": feature_count,
        "uses_xgboost_booster": True,
    }
    return {
        "metrics_available": False,
        "explanation": "No validated performance metrics were found in the current project artifacts; this section reports deployed model/data transparency only.",
        "assets": {
            "total_cell_lines": cell_count,
            "total_drugs": drug_count,
            "feature_vector": feature_count,
            "final_model_count": model_count,
        },
        "model_summary": {
            "total_models": model_count,
            "model_type": model_info["model_type"],
            "count_per_model_type": model_type_counts,
        },
        "performance": {
            "deployed_final_average": {},
            "by_model_type": [],
        },
        **model_info,
    }


def system_summary() -> dict[str, Any]:
    return {
        "project": Config.APP_NAME,
        "source_of_truth": "Current SDP_SynergyLens backend, model artifacts, data format, and route contracts.",
        "prediction_flow": [
            "Accept NSC1, NSC2, CELLNAME, and compatible aliases such as drug1_id, drug2_id, and cell_line.",
            "Resolve drug names or NSC identifiers through the canonical drug directory, with drug_name_id_map.csv as the name authority.",
            "Build the deployed feature vector in the exact order from feature_columns.pkl.",
            "Run the deployed XGBoost model in both drug orders and average the final ComboScore.",
            "Interpret the final score using cut4-aligned bands: antagonistic below -4, neutral/weak from -4 to +4, synergistic above +4.",
        ],
        "score_thresholds": score_thresholds(),
        "quantum_supported_cell_lines": quantum_supported_cell_lines(),
        "quantum_supported_cell_line_count": len(quantum_supported_cell_lines()),
        "endpoints": endpoint_summary(),
    }


def endpoint_summary() -> list[dict[str, str]]:
    return [
        {"method": "GET", "path": "/", "purpose": "Render the workspace UI."},
        {"method": "GET", "path": "/api/health", "purpose": "Backend readiness and artifact status."},
        {"method": "GET", "path": "/api/about", "purpose": "Architecture and artifact metadata."},
        {"method": "GET", "path": "/api/drugs", "purpose": "Drug search/autocomplete."},
        {"method": "GET", "path": "/api/cell-lines", "purpose": "Valid cell-line options."},
        {"method": "GET", "path": "/api/demo-cases", "purpose": "Demo prediction inputs for the UI."},
        {"method": "POST", "path": "/api/predict", "purpose": "Single drug-pair prediction."},
        {"method": "POST", "path": "/api/explain", "purpose": "Feature contribution explanation."},
        {"method": "POST", "path": "/api/molecule-pair", "purpose": "Two-drug molecule lookup."},
        {"method": "POST", "path": "/api/batch-predict", "purpose": "CSV batch prediction."},
        {"method": "GET", "path": "/api/download/<filename>", "purpose": "Secure prediction CSV download."},
        {"method": "POST", "path": "/api/chat", "purpose": "Project and prediction assistant fallback answers."},
    ]


def demo_cases() -> dict[str, Any]:
    curated_cases = [_build_curated_demo_case(case) for case in CURATED_DEMO_CASES]
    if all(curated_cases):
        return {"demo_cases": curated_cases}

    frame = artifact_loader.load_demo_source_rows()
    cases = [
        _pick_demo_case(frame, "synergistic", lambda score: score > 20, lambda score: score > 4, ascending=False),
        _pick_demo_case(frame, "neutral", lambda score: -4 <= score <= 4, None, neutral=True),
        _pick_demo_case(frame, "antagonistic", lambda score: score < -20, lambda score: score < -4, ascending=True),
    ]
    return {"demo_cases": [case for case in cases if case]}


def _build_curated_demo_case(case: dict[str, Any]) -> dict[str, Any] | None:
    nsc1 = normalize_drug_id(case.get("NSC1"))
    nsc2 = normalize_drug_id(case.get("NSC2"))
    cell_line_raw = case.get("CELLNAME")
    if not nsc1 or not nsc2 or not is_predictable_drug(nsc1) or not is_predictable_drug(nsc2):
        return None
    if not is_predictable_cell_line(cell_line_raw):
        return None

    score = float(case["real_score"])
    return {
        "case_type": str(case["case_type"]),
        "display_label": _demo_display_label(str(case["case_type"])),
        "NSC1": int(nsc1),
        "NSC2": int(nsc2),
        "CELLNAME": resolve_cell_line(cell_line_raw),
        "drug1_name": get_drug_display_name(nsc1),
        "drug2_name": get_drug_display_name(nsc2),
        "cancer_type": str(case.get("cancer_type") or "").strip(),
        "real_score": round(score, 2),
        "score": round(score, 2),
        "cut4": int(case["cut4"]) if case.get("cut4") is not None else None,
    }


def _pick_demo_case(
    frame: pd.DataFrame,
    case_type: str,
    preferred_condition,
    fallback_condition=None,
    ascending: bool = False,
    neutral: bool = False,
) -> dict[str, Any] | None:
    if frame.empty or "score" not in frame.columns:
        return None

    working = frame.copy()
    working["score_numeric"] = pd.to_numeric(working["score"], errors="coerce")
    working = working.dropna(subset=["score_numeric"])

    preferred = working[working["score_numeric"].map(preferred_condition)]
    if preferred.empty and fallback_condition is not None:
        preferred = working[working["score_numeric"].map(fallback_condition)]
    if preferred.empty:
        return None

    if neutral:
        preferred = preferred.assign(_distance=preferred["score_numeric"].abs()).sort_values("_distance")
    else:
        preferred = preferred.sort_values("score_numeric", ascending=ascending)

    for _, row in preferred.iterrows():
        nsc1 = normalize_drug_id(row.get("drug1"))
        nsc2 = normalize_drug_id(row.get("drug2"))
        cell_line_raw = row.get("cell line")
        if not nsc1 or not nsc2 or not is_predictable_drug(nsc1) or not is_predictable_drug(nsc2):
            continue
        if not is_predictable_cell_line(cell_line_raw):
            continue
        cell_line = resolve_cell_line(cell_line_raw)
        score = float(row["score_numeric"])
        return {
            "case_type": case_type,
            "display_label": _demo_display_label(case_type),
            "NSC1": int(nsc1),
            "NSC2": int(nsc2),
            "CELLNAME": cell_line,
            "drug1_name": get_drug_display_name(nsc1),
            "drug2_name": get_drug_display_name(nsc2),
            "real_score": round(score, 2),
            "score": round(score, 2),
            "cut4": int(row["cut4"]) if pd.notna(row.get("cut4")) else None,
        }
    return None


def _demo_display_label(case_type: str) -> str:
    return {
        "synergistic": "Synergistic",
        "neutral": "Neutral / weak",
        "antagonistic": "Antagonistic",
    }.get(case_type, case_type)
