from __future__ import annotations

from pathlib import Path

from werkzeug.utils import secure_filename

from backend.config import Config
from backend.utils.errors import ValidationError


PROJECT_ROOT = Config.PROJECT_ROOT
MODELS_DIR = Config.MODELS_DIR
MOLECULES_DIR = Config.MOLECULES_DIR
DATA_DIR = Config.DATA_DIR
UPLOADS_DIR = Config.UPLOADS_DIR
PREDICTIONS_DIR = Config.PREDICTIONS_DIR
OUTPUTS_DIR = Config.OUTPUTS_DIR


def ensure_runtime_directories() -> None:
    for directory in (UPLOADS_DIR, PREDICTIONS_DIR, OUTPUTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def safe_runtime_path(directory: Path, filename: str) -> Path:
    cleaned = secure_filename(Path(str(filename)).name)
    if not cleaned:
        raise ValidationError("A valid filename is required.", code="INVALID_FILENAME")

    base = directory.resolve()
    candidate = (base / cleaned).resolve()
    if base not in candidate.parents and candidate != base:
        raise ValidationError("Invalid filename.", code="INVALID_FILENAME")
    return candidate


def artifact_path(directory: Path, filename: str) -> Path:
    return directory / filename

