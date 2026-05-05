from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency guard
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")


class Config:
    APP_NAME = "SDP_SynergyLens"
    PROJECT_ROOT = PROJECT_ROOT
    MODELS_DIR = PROJECT_ROOT / "models"
    QUANTUM_MODELS_DIR = MODELS_DIR / "quantum"
    LEGACY_QUANTUM_MODELS_DIR = MODELS_DIR / "quantum_model"
    ROOT_QUANTUM_MODELS_DIR = PROJECT_ROOT / "quantum_model"
    MOLECULES_DIR = PROJECT_ROOT / "molecules"
    DATA_DIR = PROJECT_ROOT / "data"
    DATASETS_DIR = PROJECT_ROOT / "datasets"
    UPLOADS_DIR = PROJECT_ROOT / "uploads"
    PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    TEMPLATES_DIR = PROJECT_ROOT / "templates"
    STATIC_DIR = PROJECT_ROOT / "static"

    MAX_CONTENT_LENGTH = 10 * 1024 * 1024
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-me")

    PRIMARY_MODEL_FILE = "drug_synergy_xgb_model.pkl"
    SMALL_MODEL_FILE = "drug_synergy_small_model.pkl"
    FEATURE_COLUMNS_FILE = "feature_columns.pkl"
    DRUG_FINGERPRINTS_FILE = "drug_fingerprints_lookup.csv"
    CELL_LINE_FEATURES_FILE = "cell_line_features_lookup.csv"
    DRUG_NAME_MAP_FILE = "drug_name_id_map.csv"
    FEATURE_INFO_FILE = "Feature_info - Data Dictionary.csv"
    EXPLAIN_DRUG_FILE = "explain_drug - Sheet1 (1).csv"
    MOLECULES_FILE = "drug_mols.pkl"
    SAMPLE_BATCH_FILE = "sample_batch.csv"
    TRAIN10_FILE = "train10.csv"
    CANCER_ENCODER_FILE = "cancer_label_encoder.pkl"
    QUANTUM_MODEL_FILE = "quantum_model.pkl"
    QUANTUM_FEATURE_MATRIX_FILE = "X_small_pca.pkl"
    QUANTUM_TARGET_CSV_FILE = "y_small.csv"
    QUANTUM_TARGET_PICKLE_FILE = "y_small.pkl"

    GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME") or os.getenv("GEMINI_MODEL") or "gemini-3-flash-preview"
    GEMINI_REQUEST_TIMEOUT_SECONDS = float(os.getenv("GEMINI_REQUEST_TIMEOUT_SECONDS", "12"))
    GEMINI_KEYS_LOCAL_PATH = PROJECT_ROOT / ".gemini_keys.local"
