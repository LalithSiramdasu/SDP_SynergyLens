from __future__ import annotations

import re
from typing import Any

import pandas as pd

from backend.services import artifact_loader
from backend.utils.errors import ValidationError


DRUG_1_KEYS = ("NSC1", "nsc1", "drug1_id", "drug1", "drug_1", "drug1_name")
DRUG_2_KEYS = ("NSC2", "nsc2", "drug2_id", "drug2", "drug_2", "drug2_name")
CELL_KEYS = ("CELLNAME", "cellname", "cell_line", "cell line", "cell", "CELL_LINE")
CANCER_KEYS = ("cancer_type", "cancer", "CANCER_TYPE")
MODEL_TYPE_KEYS = ("model_type", "modelType", "MODEL_TYPE", "model")
SUPPORTED_MODEL_TYPES = {"classical", "quantum"}


def _first_value(data: dict[str, Any], keys: tuple[str, ...]):
    for key in keys:
        if key in data and data[key] not in (None, ""):
            return data[key]
    return None


def _extract_numeric_id(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    if re.fullmatch(r"\d+(?:\.0+)?", text):
        return int(float(text))
    match = re.search(r"\b(\d{2,})\b", text)
    if match:
        return int(match.group(1))
    return None


def normalize_drug_id(value: Any) -> str | None:
    numeric_id = _extract_numeric_id(value)
    return str(numeric_id) if numeric_id is not None else None


def normalize_drug_name(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def normalize_model_type(value: Any = None) -> str:
    requested = str(value or "classical").strip().lower()
    aliases = {
        "": "classical",
        "existing": "classical",
        "existing model": "classical",
        "classical model": "classical",
        "quantum model": "quantum",
    }
    normalized = aliases.get(requested, requested)
    if normalized not in SUPPORTED_MODEL_TYPES:
        raise ValidationError(
            "model_type must be either 'classical' or 'quantum'.",
            code="INVALID_MODEL_TYPE",
        )
    return normalized


def _predictable_drug_ids() -> set[str]:
    fingerprints = artifact_loader.load_drug_fingerprints()
    return {
        str(int(value))
        for value in fingerprints["drug_id"].dropna().tolist()
    }


def is_predictable_drug(value: Any) -> bool:
    nsc = normalize_drug_id(value)
    return bool(nsc and nsc in _predictable_drug_ids())


def get_drug_display_name(nsc: Any) -> str:
    drug_id = normalize_drug_id(nsc)
    if not drug_id:
        return f"NSC {nsc}"
    record = artifact_loader.load_drug_directory().get(drug_id)
    return str(record["name"]) if record else f"NSC {drug_id}"


def resolve_drug(value: Any, require_predictable: bool = False) -> dict[str, Any]:
    if value in (None, ""):
        raise ValidationError("Drug value is required.", code="MISSING_DRUG")

    original = str(value).strip()
    numeric_id = _extract_numeric_id(value)
    directory = artifact_loader.load_drug_directory()

    if numeric_id is not None:
        nsc_key = str(numeric_id)
        record = directory.get(nsc_key)
        if record is None:
            raise ValidationError(f"Drug NSC {numeric_id} is not available.", code="DRUG_NOT_FOUND")
    else:
        text = normalize_drug_name(value)
        matches = [
            record
            for record in directory.values()
            if normalize_drug_name(record["name"]) == text
        ]
        if not matches:
            raise ValidationError(f"Drug '{value}' is not available.", code="DRUG_NOT_FOUND")
        record = matches[0]
        nsc_key = str(record["id"])

    if require_predictable and nsc_key not in _predictable_drug_ids():
        raise ValidationError(f"Drug NSC {nsc_key} has no fingerprint features.", code="DRUG_NOT_FOUND")

    return {
        "nsc": int(nsc_key),
        "nsc_str": nsc_key,
        "name": str(record["name"]),
        "input": original,
    }


def resolve_cell_line(value: Any) -> str:
    if value in (None, ""):
        raise ValidationError("Cell line is required.", code="MISSING_CELL_LINE")

    requested = str(value).strip()
    values = list(artifact_loader.load_cell_line_directory().values())
    exact = {item.strip(): item for item in values}
    if requested in exact:
        return exact[requested]

    lowered = {item.lower(): item for item in values}
    match = lowered.get(requested.lower())
    if match:
        return match

    raise ValidationError(f"Cell line '{requested}' is not available.", code="CELL_LINE_NOT_FOUND")


def is_predictable_cell_line(value: Any) -> bool:
    try:
        resolved = resolve_cell_line(value)
    except ValidationError:
        return False
    lookup = artifact_loader.load_cell_line_features()
    feature_cells = {str(item).strip() for item in lookup["cell line"].dropna().tolist()}
    return resolved in feature_cells


def normalize_prediction_payload(data: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValidationError("JSON request body is required.", code="MISSING_JSON")

    drug1_value = _first_value(data, DRUG_1_KEYS)
    drug2_value = _first_value(data, DRUG_2_KEYS)
    cell_value = _first_value(data, CELL_KEYS)
    missing = []
    if drug1_value in (None, ""):
        missing.append("NSC1/drug1_id")
    if drug2_value in (None, ""):
        missing.append("NSC2/drug2_id")
    if cell_value in (None, ""):
        missing.append("CELLNAME/cell_line")
    if missing:
        raise ValidationError(f"Missing required field(s): {', '.join(missing)}", code="MISSING_FIELDS")

    drug1 = resolve_drug(drug1_value, require_predictable=True)
    drug2 = resolve_drug(drug2_value, require_predictable=True)
    cell_line = resolve_cell_line(cell_value)
    cancer_type = _first_value(data, CANCER_KEYS)
    model_type = normalize_model_type(_first_value(data, MODEL_TYPE_KEYS))

    return {
        "nsc1": drug1["nsc"],
        "nsc2": drug2["nsc"],
        "drug1_name": drug1["name"],
        "drug2_name": drug2["name"],
        "cell_line": cell_line,
        "cancer_type": str(cancer_type).strip() if cancer_type not in (None, "") else "",
        "model_type": model_type,
    }


def normalize_batch_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping: dict[str, str] = {}
    for column in df.columns:
        normalized = re.sub(r"[\s_\-]+", "", str(column).strip().lower())
        if normalized in {"drug1id", "drug1", "nsc1", "drugname1", "drug1name"}:
            mapping[column] = "NSC1"
        elif normalized in {"drug2id", "drug2", "nsc2", "drugname2", "drug2name"}:
            mapping[column] = "NSC2"
        elif normalized in {"cellline", "cellname", "cell"}:
            mapping[column] = "CELLNAME"
        elif normalized in {"cancertype", "cancer"}:
            mapping[column] = "cancer_type"

    renamed = df.rename(columns=mapping).copy()
    required = {"NSC1", "NSC2", "CELLNAME"}
    missing = sorted(required - set(renamed.columns))
    if missing:
        raise ValidationError(f"CSV missing required column(s): {', '.join(missing)}", code="CSV_MISSING_COLUMNS")
    return renamed


def _drug_name_from_id(name_map: pd.DataFrame, nsc: int) -> str:
    return get_drug_display_name(nsc)
