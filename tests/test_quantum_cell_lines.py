from __future__ import annotations

import unittest

from app import app
from backend.config import Config
from backend.services import artifact_loader


class QuantumCellLineTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_cell_lines_default_and_classical_return_full_list(self):
        default_response = self.client.get("/api/cell-lines")
        classical_response = self.client.get("/api/cell-lines?model_type=classical")

        self.assertEqual(default_response.status_code, 200)
        self.assertEqual(classical_response.status_code, 200)
        default_data = default_response.get_json()
        classical_data = classical_response.get_json()

        self.assertTrue(default_data["success"])
        self.assertTrue(classical_data["success"])
        self.assertEqual(default_data["cell_lines"], classical_data["cell_lines"])
        self.assertGreater(len(default_data["cell_lines"]), len(Config.QUANTUM_SUPPORTED_CELL_LINES))

    def test_quantum_cell_lines_returns_configured_subset(self):
        response = self.client.get("/api/cell-lines?model_type=quantum")

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["cell_lines"], list(Config.QUANTUM_SUPPORTED_CELL_LINES))
        self.assertEqual(data["count"], 5)

    def test_classical_prediction_accepts_regular_valid_cell_line(self):
        payload = self._classical_payload()

        response = self.client.post("/api/predict", json=payload)

        data = response.get_json()
        self.assertEqual(response.status_code, 200, data)
        self.assertTrue(data["success"])
        self.assertEqual(data["model_type"], "classical")
        self.assertIn("score", data)

    def test_quantum_prediction_accepts_supported_cell_line(self):
        response = self.client.post(
            "/api/predict",
            json={"NSC1": 740, "NSC2": 754143, "CELLNAME": "OVCAR-3", "model_type": "quantum"},
        )

        data = response.get_json()
        self.assertEqual(response.status_code, 200, data)
        self.assertTrue(data["success"])
        self.assertEqual(data["model_type"], "quantum")
        self.assertIn("score", data)

    def test_quantum_prediction_rejects_unsupported_cell_line_cleanly(self):
        unsupported_cell_line = self._unsupported_quantum_cell_line()
        response = self.client.post(
            "/api/predict",
            json={"NSC1": 740, "NSC2": 754143, "CELLNAME": unsupported_cell_line, "model_type": "quantum"},
        )

        data = response.get_json()
        self.assertEqual(response.status_code, 400)
        self.assertFalse(data["success"])
        self.assertEqual(data["error"]["code"], "QUANTUM_CELL_LINE_UNSUPPORTED")
        self.assertEqual(data["message"], "Quantum model currently supports only selected cell lines.")

    def _classical_payload(self) -> dict[str, object]:
        drug_ids = artifact_loader.load_drug_fingerprints()["drug_id"].dropna().astype(int).head(2).tolist()
        if len(drug_ids) < 2:
            raise AssertionError("Need at least two drugs for prediction tests.")
        return {
            "NSC1": int(drug_ids[0]),
            "NSC2": int(drug_ids[1]),
            "CELLNAME": self._unsupported_quantum_cell_line(),
            "model_type": "classical",
        }

    def _unsupported_quantum_cell_line(self) -> str:
        quantum_lines = set(Config.QUANTUM_SUPPORTED_CELL_LINES)
        for value in artifact_loader.load_cell_line_features()["cell line"].dropna().astype(str):
            cell_line = value.strip()
            if cell_line and cell_line not in quantum_lines:
                return cell_line
        raise AssertionError("Need at least one classical-only cell line for validation tests.")


if __name__ == "__main__":
    unittest.main()
