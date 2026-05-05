from __future__ import annotations

import json
import pickle
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from backend.config import Config
from backend.utils.errors import ArtifactError


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise ArtifactError(f"Required artifact is missing: {path.name}", code="ARTIFACT_MISSING")
    return path


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(_require_file(path))
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError(f"Could not load CSV artifact {path.name}: {exc}") from exc


def _load_joblib(path: Path) -> Any:
    try:
        return joblib.load(_require_file(path))
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError(f"Could not load model artifact {path.name}: {exc}") from exc


def quantum_models_dir() -> Path:
    for directory in (
        Config.QUANTUM_MODELS_DIR,
        Config.LEGACY_QUANTUM_MODELS_DIR,
        Config.ROOT_QUANTUM_MODELS_DIR,
    ):
        if (directory / Config.QUANTUM_MODEL_FILE).exists():
            return directory
    return Config.QUANTUM_MODELS_DIR


def quantum_artifact_paths() -> dict[str, Path]:
    directory = quantum_models_dir()
    return {
        "model": directory / Config.QUANTUM_MODEL_FILE,
        "feature_matrix": directory / Config.QUANTUM_FEATURE_MATRIX_FILE,
        "target_csv": directory / Config.QUANTUM_TARGET_CSV_FILE,
        "target_pickle": directory / Config.QUANTUM_TARGET_PICKLE_FILE,
    }


@lru_cache(maxsize=1)
def load_quantum_model():
    return _load_joblib(quantum_models_dir() / Config.QUANTUM_MODEL_FILE)


@lru_cache(maxsize=1)
def load_quantum_feature_matrix():
    return _load_joblib(quantum_models_dir() / Config.QUANTUM_FEATURE_MATRIX_FILE)


def quantum_model_info(load: bool = False) -> dict[str, Any]:
    directory = quantum_models_dir()
    paths = quantum_artifact_paths()
    info: dict[str, Any] = {
        "model_type": "Quantum",
        "model_file": Config.QUANTUM_MODEL_FILE,
        "model_dir": str(directory),
        "preferred_model_dir": str(Config.QUANTUM_MODELS_DIR),
        "feature_format": "4 PCA features",
        "uses_classical_feature_columns": False,
        "shap_available": False,
        "artifacts": {key: path.name for key, path in paths.items() if path.exists()},
        "missing_artifacts": [path.name for path in paths.values() if not path.exists()],
    }
    if not load:
        return info

    model = load_quantum_model()
    info["model_type"] = type(model).__name__
    info["required_feature_count"] = int(getattr(model, "n_features_in_", 4))
    return info


def quantum_artifact_status(load: bool = False) -> dict[str, Any]:
    paths = quantum_artifact_paths()
    status = {
        "quantum_models_dir": str(quantum_models_dir()),
        "preferred_quantum_models_dir": str(Config.QUANTUM_MODELS_DIR),
        "quantum_model_exists": paths["model"].exists(),
        "quantum_feature_matrix_exists": paths["feature_matrix"].exists(),
        "quantum_target_csv_exists": paths["target_csv"].exists(),
        "quantum_target_pickle_exists": paths["target_pickle"].exists(),
        "quantum_model_loaded": False,
        "errors": [],
    }
    if load and status["quantum_model_exists"]:
        try:
            load_quantum_model()
            status["quantum_model_loaded"] = True
        except Exception as exc:
            status["errors"].append(str(exc))
    return status


def _patch_xgboost_base_score(model: Any) -> None:
    if not hasattr(model, "get_booster"):
        return

    try:
        booster = model.get_booster()
        config = json.loads(booster.save_config())
        params = config.get("learner", {}).get("learner_model_param", {})
        base_score = params.get("base_score")
        if isinstance(base_score, str) and "[" in base_score:
            params["base_score"] = base_score.strip("[]")
            booster.load_config(json.dumps(config))
    except Exception:
        # The model can still predict even if this compatibility patch is not needed.
        return


@lru_cache(maxsize=1)
def load_model():
    model = _load_joblib(Config.MODELS_DIR / Config.PRIMARY_MODEL_FILE)
    _patch_xgboost_base_score(model)
    return model


@lru_cache(maxsize=1)
def load_feature_columns() -> list[str]:
    columns = _load_joblib(Config.MODELS_DIR / Config.FEATURE_COLUMNS_FILE)
    return [str(column) for column in columns]


@lru_cache(maxsize=1)
def load_drug_fingerprints() -> pd.DataFrame:
    return _read_csv(Config.MODELS_DIR / Config.DRUG_FINGERPRINTS_FILE)


@lru_cache(maxsize=1)
def load_cell_line_features() -> pd.DataFrame:
    return _read_csv(Config.MODELS_DIR / Config.CELL_LINE_FEATURES_FILE)


@lru_cache(maxsize=1)
def load_drug_name_map() -> pd.DataFrame:
    return _read_csv(Config.MODELS_DIR / Config.DRUG_NAME_MAP_FILE)


def _normalize_nsc_value(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d+(?:\.0+)?", text):
        return str(int(float(text)))
    match = re.search(r"\b(\d{2,})\b", text)
    return match.group(1) if match else None


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


@lru_cache(maxsize=1)
def load_train500_summary() -> pd.DataFrame:
    return _read_dataset_summary(Config.DATASETS_DIR / "train_500_rows.csv")


@lru_cache(maxsize=1)
def load_train10_summary() -> pd.DataFrame:
    return _read_dataset_summary(Config.DATASETS_DIR / Config.TRAIN10_FILE)


def _read_dataset_summary(path: Path) -> pd.DataFrame:
    wanted = ["drugname1", "drugname2", "drug1", "drug2", "cell line", "score", "cut4"]
    if not path.exists():
        return pd.DataFrame(columns=wanted)
    try:
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        usecols = [column for column in wanted if column in columns]
        if not usecols:
            return pd.DataFrame(columns=wanted)
        frame = pd.read_csv(path, usecols=usecols)
        for column in wanted:
            if column not in frame.columns:
                frame[column] = None
        return frame[wanted]
    except Exception:
        return pd.DataFrame(columns=wanted)


@lru_cache(maxsize=1)
def load_drug_directory() -> dict[str, dict[str, Any]]:
    directory: dict[str, dict[str, Any]] = {}

    def add_drug(raw_id: Any, raw_name: Any = "", source: str = "") -> None:
        nsc = _normalize_nsc_value(raw_id)
        if not nsc:
            return
        name = _clean_text(raw_name)
        record = directory.setdefault(
            nsc,
            {
                "id": nsc,
                "nsc": nsc,
                "name": f"NSC {nsc}",
                "sources": set(),
            },
        )
        if source:
            record["sources"].add(source)
        if name and not re.fullmatch(r"\d+(?:\.0+)?", name):
            # Keep the explicit id,name map as the strongest name source by adding it first.
            if record["name"] == f"NSC {nsc}" or source == Config.DRUG_NAME_MAP_FILE:
                record["name"] = name

    try:
        name_map = load_drug_name_map()
        for _, row in name_map.iterrows():
            add_drug(row.get("id"), row.get("name"), Config.DRUG_NAME_MAP_FILE)
    except Exception:
        pass

    try:
        explain_drugs = load_explain_drug_info()
        for _, row in explain_drugs.iterrows():
            add_drug(row.get("drug_id"), row.get("name"), Config.EXPLAIN_DRUG_FILE)
    except Exception:
        pass

    try:
        fingerprints = load_drug_fingerprints()
        for value in fingerprints.get("drug_id", pd.Series(dtype=object)).tolist():
            add_drug(value, "", Config.DRUG_FINGERPRINTS_FILE)
    except Exception:
        pass

    for frame, source in ((load_train500_summary(), "train_500_rows.csv"),):
        for _, row in frame.iterrows():
            add_drug(row.get("drug1"), row.get("drugname1"), source)
            add_drug(row.get("drug2"), row.get("drugname2"), source)

    return {
        nsc: {
            **record,
            "sources": sorted(record["sources"]),
            "label": f"{record['name']} - NSC {nsc}",
        }
        for nsc, record in sorted(directory.items(), key=lambda item: (item[1]["name"].lower(), int(item[0])))
    }


@lru_cache(maxsize=1)
def load_cell_line_directory() -> dict[str, str]:
    values: dict[str, str] = {}

    def add_cell(value: Any) -> None:
        text = _clean_text(value)
        if text:
            values.setdefault(text.lower(), text)

    try:
        lookup = load_cell_line_features()
        for value in lookup.get("cell line", pd.Series(dtype=object)).tolist():
            add_cell(value)
    except Exception:
        pass

    for frame in (load_train500_summary(),):
        for value in frame.get("cell line", pd.Series(dtype=object)).dropna().tolist():
            add_cell(value)

    return dict(sorted(values.items(), key=lambda item: item[1].lower()))


def load_demo_source_rows() -> pd.DataFrame:
    train500 = load_train500_summary()
    if not train500.empty:
        return train500.copy()
    return load_train10_summary().copy()


@lru_cache(maxsize=1)
def load_feature_info() -> pd.DataFrame:
    return _read_csv(Config.MODELS_DIR / Config.FEATURE_INFO_FILE)


@lru_cache(maxsize=1)
def load_explain_drug_info() -> pd.DataFrame:
    return _read_csv(Config.MODELS_DIR / Config.EXPLAIN_DRUG_FILE)


@lru_cache(maxsize=1)
def load_molecules() -> dict[Any, Any]:
    path = _require_file(Config.MOLECULES_DIR / Config.MOLECULES_FILE)
    try:
        with path.open("rb") as handle:
            molecules = pickle.load(handle)
    except Exception as exc:
        raise ArtifactError(f"Could not load molecule artifact {path.name}: {exc}") from exc

    if not isinstance(molecules, dict):
        raise ArtifactError("Molecule artifact must contain a dictionary.", code="INVALID_ARTIFACT")
    return molecules


def molecule_artifact_status(load: bool = False) -> dict[str, Any]:
    path = Config.MOLECULES_DIR / Config.MOLECULES_FILE
    status: dict[str, Any] = {
        "available": path.exists(),
        "loaded": False,
        "count": None,
        "file": Config.MOLECULES_FILE,
        "path": str(path),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "error": "",
    }
    if not load or not path.exists():
        return status

    try:
        molecules = load_molecules()
    except Exception as exc:
        status["error"] = str(exc)
        return status

    status["loaded"] = True
    status["count"] = len(molecules)
    return status


@lru_cache(maxsize=1)
def load_cancer_encoder():
    path = Config.MODELS_DIR / Config.CANCER_ENCODER_FILE
    if not path.exists():
        return None
    return _load_joblib(path)


@lru_cache(maxsize=1)
def load_train10_score_lookup() -> dict[tuple[int, int, str], float]:
    path = Config.DATASETS_DIR / Config.TRAIN10_FILE
    if not path.exists():
        return {}

    try:
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        drug1_column = _first_existing_column(columns, ("drug1", "NSC1", "nsc1"))
        drug2_column = _first_existing_column(columns, ("drug2", "NSC2", "nsc2"))
        cell_column = _first_existing_column(columns, ("cell line", "CELLNAME", "cell_line", "cellname"))
        score_column = _first_existing_column(columns, ("score", "COMBOSCORE", "ComboScore", "comboscore"))
        if not all((drug1_column, drug2_column, cell_column, score_column)):
            return {}
        frame = pd.read_csv(path, usecols=[drug1_column, drug2_column, cell_column, score_column])
        frame = frame.rename(
            columns={
                drug1_column: "drug1",
                drug2_column: "drug2",
                cell_column: "cell_line",
                score_column: "score",
            }
        )[["drug1", "drug2", "cell_line", "score"]]
    except Exception:
        return {}

    lookup: dict[tuple[int, int, str], float] = {}
    for row in frame.itertuples(index=False, name=None):
        drug1, drug2, cell_line, score = row
        try:
            left = int(drug1)
            right = int(drug2)
            real_score = float(score)
        except (TypeError, ValueError):
            continue

        key = (*sorted((left, right)), str(cell_line).strip())
        if key not in lookup:
            lookup[key] = real_score
    return lookup


def _first_existing_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    by_lower = {str(column).strip().lower(): column for column in columns}
    for candidate in candidates:
        found = by_lower.get(candidate.lower())
        if found is not None:
            return found
    return None


def train10_lookup_key_count() -> int:
    return len(load_train10_score_lookup())


def get_booster():
    model = load_model()
    if hasattr(model, "get_booster"):
        return model.get_booster()
    return None


def model_info(load: bool = True) -> dict[str, Any]:
    if not load:
        return {
            "model_type": "XGBoost",
            "model_file": Config.PRIMARY_MODEL_FILE,
            "feature_column_count": len(load_feature_columns()),
            "uses_xgboost_booster": True,
        }

    model = load_model()
    return {
        "model_type": type(model).__name__,
        "model_file": Config.PRIMARY_MODEL_FILE,
        "feature_column_count": len(load_feature_columns()),
        "uses_xgboost_booster": hasattr(model, "get_booster"),
    }


def artifact_status(load: bool = True) -> dict[str, Any]:
    checks = {
        "model_loaded": load_model,
        "feature_columns_loaded": load_feature_columns,
        "drug_lookup_loaded": load_drug_fingerprints,
        "cell_line_lookup_loaded": load_cell_line_features,
        "drug_name_map_loaded": load_drug_name_map,
        "molecules_loaded": load_molecules,
        "feature_info_loaded": load_feature_info,
        "explain_drug_loaded": load_explain_drug_info,
    }
    status: dict[str, Any] = {}
    errors: list[str] = []

    for key, loader in checks.items():
        try:
            if load:
                loader()
                status[key] = True
            else:
                status[key] = _artifact_exists_for_status(key)
        except Exception as exc:
            status[key] = False
            errors.append(str(exc))

    status["errors"] = errors
    return status


def _artifact_exists_for_status(key: str) -> bool:
    paths = {
        "model_loaded": Config.MODELS_DIR / Config.PRIMARY_MODEL_FILE,
        "feature_columns_loaded": Config.MODELS_DIR / Config.FEATURE_COLUMNS_FILE,
        "drug_lookup_loaded": Config.MODELS_DIR / Config.DRUG_FINGERPRINTS_FILE,
        "cell_line_lookup_loaded": Config.MODELS_DIR / Config.CELL_LINE_FEATURES_FILE,
        "drug_name_map_loaded": Config.MODELS_DIR / Config.DRUG_NAME_MAP_FILE,
        "molecules_loaded": Config.MOLECULES_DIR / Config.MOLECULES_FILE,
        "feature_info_loaded": Config.MODELS_DIR / Config.FEATURE_INFO_FILE,
        "explain_drug_loaded": Config.MODELS_DIR / Config.EXPLAIN_DRUG_FILE,
    }
    path = paths.get(key)
    return bool(path and path.exists())


def detected_artifacts() -> dict[str, list[dict[str, Any]]]:
    groups = {
        "models": Config.MODELS_DIR,
        "molecules": Config.MOLECULES_DIR,
        "data": Config.DATA_DIR,
    }
    result: dict[str, list[dict[str, Any]]] = {}
    for label, directory in groups.items():
        if not directory.exists():
            result[label] = []
            continue
        result[label] = [
            {"name": path.name, "size_bytes": path.stat().st_size}
            for path in sorted(directory.iterdir())
            if path.is_file()
        ]
    return result


def configured_gemini_key_count() -> int:
    from backend.services import gemini_service

    return gemini_service.configured_key_count()
