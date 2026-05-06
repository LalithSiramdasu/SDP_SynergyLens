from __future__ import annotations

import unittest
from unittest import mock

from app import app


class QuantumSurrogateExplainTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.payload = {
            "drug1_id": "3053",
            "drug2_id": "180973",
            "cell_line": "HL-60(TB)",
        }

    def test_classical_explain_still_returns_contributors(self):
        response = self.client.post("/api/explain", json={**self.payload, "model_type": "classical"})

        data = response.get_json()
        self.assertEqual(response.status_code, 200, data)
        self.assertTrue(data["success"])
        self.assertEqual(data["model_type"], "classical")
        self.assertTrue(data["features"])
        self.assertTrue(data["top_positive_contributors"] or data["top_negative_contributors"])
        self.assertNotEqual(data.get("explanation_type"), "classical_surrogate_for_quantum")

    def test_quantum_explain_uses_quantum_score_and_classical_surrogate_contributors(self):
        with mock.patch("backend.services.prediction_service.time.sleep"):
            prediction = self.client.post("/api/predict", json={**self.payload, "model_type": "quantum"}).get_json()
        response = self.client.post("/api/explain", json={**self.payload, "model_type": "quantum"})

        data = response.get_json()
        self.assertEqual(response.status_code, 200, data)
        self.assertTrue(data["success"])
        self.assertEqual(data["model_type"], "quantum")
        self.assertEqual(data["final_predicted_COMBOSCORE"], prediction["final_predicted_COMBOSCORE"])
        self.assertEqual(data["explanation_type"], "classical_surrogate_for_quantum")
        self.assertEqual(data["explanation_method"], "Classical surrogate XAI")
        self.assertTrue(data["quantum_prediction_used"])
        self.assertTrue(data["surrogate_explanation_used"])
        self.assertTrue(data["explanation_available"])
        self.assertTrue(data["features"])
        self.assertTrue(data["top_positive_contributors"] or data["top_negative_contributors"])
        self.assertIn("prediction score remains from the quantum pipeline", data["explanation_summary"])
        self.assertIn("surrogate explanation", data["limitation_note"])

    def test_quantum_explain_surrogate_failure_returns_clean_fallback(self):
        with mock.patch(
            "backend.services.explain_service.get_feature_contributions",
            side_effect=RuntimeError("forced surrogate failure"),
        ):
            response = self.client.post("/api/explain", json={**self.payload, "model_type": "quantum"})

        data = response.get_json()
        self.assertEqual(response.status_code, 200, data)
        self.assertTrue(data["success"])
        self.assertEqual(data["model_type"], "quantum")
        self.assertEqual(data["explanation_type"], "classical_surrogate_for_quantum")
        self.assertTrue(data["quantum_prediction_used"])
        self.assertFalse(data["surrogate_explanation_used"])
        self.assertFalse(data["explanation_available"])
        self.assertEqual(data["features"], [])
        self.assertEqual(data["top_positive_contributors"], [])
        self.assertEqual(data["top_negative_contributors"], [])
        self.assertEqual(data["surrogate_error"], "forced surrogate failure")


if __name__ == "__main__":
    unittest.main()
