"""Flask test-client verification for the modular SDP_SynergyLens backend."""

from __future__ import annotations

import io
import sys
from pathlib import Path

from app import app
from backend.config import Config
from backend.services import artifact_loader


ROOT = Path(__file__).resolve().parent
SAMPLE_BATCH_PATH = ROOT / "data" / "sample_batch.csv"


def sample_payload() -> dict[str, object]:
    drug_ids = artifact_loader.load_drug_fingerprints()["drug_id"].dropna().astype(int).head(2).tolist()
    cell_line = str(artifact_loader.load_cell_line_features()["cell line"].dropna().iloc[0])
    if len(drug_ids) < 2 or not cell_line:
        raise RuntimeError("Need at least two drugs and one cell line for verification.")
    return {"NSC1": int(drug_ids[0]), "NSC2": int(drug_ids[1]), "CELLNAME": cell_line}


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_success_json(response, route: str) -> dict:
    data = response.get_json(silent=True) or {}
    expect(response.status_code < 400 and data.get("success") is True, f"{route} failed: {response.status_code} {data}")
    print(f"PASS {route}")
    return data


def expect_error_json(response, route: str) -> dict:
    data = response.get_json(silent=True) or {}
    expect(response.status_code >= 400 and data.get("success") is False, f"{route} should have failed cleanly")
    expect(isinstance(data.get("error"), dict), f"{route} missing structured error")
    print(f"PASS {route}")
    return data


def expect_molecule_graceful(response, route: str) -> dict:
    data = expect_success_json(response, route)
    molecule = data.get("data") if isinstance(data.get("data"), dict) else data
    expect(molecule.get("found") is True, f"{route} did not resolve molecule metadata")
    expect(
        molecule.get("molecule_found") is True
        or (
            molecule.get("rdkit_available") is False
            and bool(molecule.get("structure_message"))
        ),
        f"{route} did not return structure data or a graceful RDKit fallback",
    )
    return data


def run() -> int:
    payload = sample_payload()
    quantum_payload = {"NSC1": 740, "NSC2": 754143, "CELLNAME": "OVCAR-3", "model_type": "quantum"}

    with app.test_client() as client:
        response = client.get("/")
        expect(response.status_code == 200 and b"SynergyLens" in response.data, "/ did not render")
        print("PASS /")

        health = expect_success_json(client.get("/api/health"), "/api/health")
        expect(health["data"]["artifacts"]["model_loaded"] is True, "/api/health reports model not loaded")
        expect(health.get("model_count") == artifact_loader.deployed_model_count(), "/api/health model count mismatch")

        expect_success_json(client.get("/api/about"), "/api/about")
        expect_success_json(client.get("/api/system-summary"), "/api/system-summary")
        performance = expect_success_json(client.get("/api/model-performance-summary"), "/api/model-performance-summary")
        expect(performance["assets"]["final_model_count"] == artifact_loader.deployed_model_count(), "/api/model-performance-summary model count mismatch")
        expect_success_json(client.get("/api/drugs?limit=all"), "/api/drugs")
        cell_lines = expect_success_json(client.get("/api/cell-lines"), "/api/cell-lines")
        classical_cell_lines = expect_success_json(client.get("/api/cell-lines?model_type=classical"), "/api/cell-lines classical")
        quantum_cell_lines = expect_success_json(client.get("/api/cell-lines?model_type=quantum"), "/api/cell-lines quantum")
        expect(cell_lines["cell_lines"] == classical_cell_lines["cell_lines"], "/api/cell-lines classical should match default")
        expect(quantum_cell_lines["cell_lines"] == list(Config.QUANTUM_SUPPORTED_CELL_LINES), "/api/cell-lines quantum list mismatch")
        expect_success_json(client.get("/api/demo-cases"), "/api/demo-cases")

        prediction = expect_success_json(client.post("/api/predict", json=payload), "/api/predict")
        expect("score" in prediction and "label" in prediction, "/api/predict missing score/label")
        quantum = expect_success_json(client.post("/api/predict", json=quantum_payload), "/api/predict quantum")
        expect(quantum.get("model_type") == "quantum" and "score" in quantum, "/api/predict quantum missing quantum score")

        expect_error_json(client.post("/api/predict", json={"NSC1": 999999999, "NSC2": payload["NSC2"], "CELLNAME": payload["CELLNAME"]}), "/api/predict invalid drug")
        expect_error_json(client.post("/api/predict", json={"NSC1": payload["NSC1"]}), "/api/predict missing fields")

        expect_molecule_graceful(client.get(f"/api/molecule/{payload['NSC1']}"), "/api/molecule/<nsc>")
        expect_error_json(client.get("/api/molecule/999999999"), "/api/molecule invalid drug")
        molecule_pair = expect_success_json(client.post("/api/molecule-pair", json={"NSC1": payload["NSC1"], "NSC2": payload["NSC2"]}), "/api/molecule-pair")
        expect(molecule_pair["molecule_1"].get("found") is True and molecule_pair["molecule_2"].get("found") is True, "/api/molecule-pair did not resolve both molecules")
        expect_success_json(client.get(f"/api/drug_info/{payload['NSC1']}"), "/api/drug_info/<id>")

        explanation = expect_success_json(client.post("/api/explain", json=payload), "/api/explain")
        expect(explanation.get("features"), "/api/explain missing features")

        chat = expect_success_json(client.post("/api/chat", json={**payload, "question": "Explain this prediction"}), "/api/chat")
        expect(chat.get("answer"), "/api/chat missing answer")
        project_chat = expect_success_json(client.post("/api/chat", json={"mode": "project", "question": "What CSV format is required?"}), "/api/chat project")
        expect(project_chat.get("answer"), "/api/chat project missing answer")

        csv_bytes = SAMPLE_BATCH_PATH.read_bytes()
        response = client.post(
            "/api/batch-predict",
            data={"file": (io.BytesIO(csv_bytes), "sample_batch.csv")},
            content_type="multipart/form-data",
        )
        batch = expect_success_json(response, "/api/batch-predict")
        expect(batch.get("output_file"), "/api/batch-predict missing output file")
        download = client.get(f"/api/download/{batch['output_file']}")
        expect(download.status_code == 200 and b"final_predicted_COMBOSCORE" in download.data, "/api/download failed")
        print("PASS /api/download/<filename>")

    print("ALL_FEATURE_TESTS_PASSED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except Exception as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        raise SystemExit(1)
