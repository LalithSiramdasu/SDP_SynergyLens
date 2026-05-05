from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests

from app import app
from backend.config import Config


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class GeminiChatTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.keys_path_patch = mock.patch.object(
            Config,
            "GEMINI_KEYS_LOCAL_PATH",
            Path(self.tempdir.name) / ".gemini_keys.local",
        )
        self.keys_path_patch.start()
        self.addCleanup(self.keys_path_patch.stop)

    def test_no_keys_returns_local_fallback(self):
        with mock.patch.dict(
            "os.environ",
            {"GEMINI_API_KEYS": "", "GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""},
        ):
            response = self.client.post("/api/chat", json={"mode": "project", "question": "What CSV format is required?"})

        data = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data["success"])
        self.assertFalse(data["llm_used"])
        self.assertTrue(data["used_fallback"])
        self.assertEqual(data["provider_label"], "Built-in Guide")
        self.assertEqual(data["attempted_key_count"], 0)
        self.assertTrue(data["answer"])

    def test_key_fallback_uses_second_key_after_first_fails(self):
        gemini_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": (
                                    "Gemini enhanced answer: SynergyLens uses local model artifacts to score drug pairs, "
                                    "then explains the result with project-safe context."
                                )
                            }
                        ],
                    },
                }
            ],
        }
        with mock.patch.dict(
            "os.environ",
            {"GEMINI_API_KEYS": "bad-key,good-key", "GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""},
        ), mock.patch(
            "backend.services.gemini_service.requests.post",
            side_effect=[FakeResponse(429), FakeResponse(200, gemini_payload)],
        ) as post:
            response = self.client.post("/api/chat", json={"mode": "project", "question": "What is this app?"})

        data = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data["llm_used"])
        self.assertFalse(data["used_fallback"])
        self.assertEqual(data["provider_label"], "AI Enhanced")
        self.assertEqual(data["attempted_key_count"], 2)
        self.assertEqual(data["successful_key_index"], 2)
        self.assertIn("Gemini enhanced answer", data["answer"])
        self.assertEqual(post.call_count, 2)

    def test_gemini_timeout_falls_back_cleanly(self):
        with mock.patch.dict(
            "os.environ",
            {"GEMINI_API_KEYS": "timeout-key", "GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""},
        ), mock.patch(
            "backend.services.gemini_service.requests.post",
            side_effect=requests.Timeout("timed out"),
        ):
            response = self.client.post("/api/chat", json={"mode": "project", "question": "Is this clinical advice?"})

        data = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertFalse(data["llm_used"])
        self.assertTrue(data["used_fallback"])
        self.assertEqual(data["attempted_key_count"], 1)
        self.assertIsNone(data["successful_key_index"])
        self.assertTrue(data["answer"])

    def test_incomplete_gemini_prediction_answer_uses_local_fallback(self):
        gemini_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "For the combination of NSC 3053 and NSC 180973 in the O"}],
                    },
                }
            ],
        }
        prediction_payload = {
            "mode": "prediction",
            "question": "Explain this result",
            "prediction": {
                "input": {"NSC1": 3053, "NSC2": 180973, "CELLNAME": "OVCAR-3"},
                "prediction_NSC1_to_NSC2": 27.37,
                "prediction_NSC2_to_NSC1": 26.37,
                "final_predicted_COMBOSCORE": 26.87,
                "label": "synergistic",
                "model_used": "XGBRegressor",
                "model_type": "classical",
            },
        }
        with mock.patch.dict(
            "os.environ",
            {"GEMINI_API_KEYS": "weak-key", "GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""},
        ), mock.patch(
            "backend.services.gemini_service.requests.post",
            return_value=FakeResponse(200, gemini_payload),
        ):
            response = self.client.post("/api/chat", json=prediction_payload)

        data = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertFalse(data["llm_used"])
        self.assertTrue(data["used_fallback"])
        self.assertEqual(data["provider_label"], "Built-in Guide")
        self.assertIn("ComboScore 26.870", data["answer"])
        self.assertIn("XGBRegressor", data["answer"])
        self.assertIn("Next checks", data["answer"])

    def test_prediction_fallback_is_specific_and_actionable(self):
        prediction_payload = {
            "mode": "prediction",
            "question": "Explain this result",
            "prediction": {
                "input": {"NSC1": 3053, "NSC2": 180973, "CELLNAME": "OVCAR-3"},
                "prediction_NSC1_to_NSC2": 27.37,
                "prediction_NSC2_to_NSC1": 26.37,
                "final_predicted_COMBOSCORE": 26.87,
                "label": "synergistic",
                "model_used": "XGBRegressor",
                "model_type": "classical",
            },
        }
        with mock.patch.dict(
            "os.environ",
            {"GEMINI_API_KEYS": "", "GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""},
        ):
            response = self.client.post("/api/chat", json=prediction_payload)

        data = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertIn("NSC 3053 + NSC 180973", data["answer"])
        self.assertIn("OVCAR-3", data["answer"])
        self.assertIn("NSC1 -> NSC2", data["answer"])
        self.assertIn("not medical advice", data["answer"])

    def test_project_and_prediction_modes_return_answer_and_suggestions(self):
        prediction_payload = {
            "mode": "prediction",
            "question": "Explain this result",
            "prediction": {
                "input": {"NSC1": 1, "NSC2": 2, "CELLNAME": "A549"},
                "final_predicted_COMBOSCORE": 6.25,
                "label": "moderate synergy",
                "model_used": "XGBoost",
                "model_type": "classical",
            },
            "explanation": {
                "features": [{"readable_feature": "Drug fingerprint bit 42", "shap_value": 1.2}],
            },
        }
        with mock.patch.dict(
            "os.environ",
            {"GEMINI_API_KEYS": "", "GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""},
        ):
            project = self.client.post("/api/chat", json={"mode": "project", "question": "How does prediction work?"})
            prediction = self.client.post("/api/chat", json=prediction_payload)

        for response in (project, prediction):
            data = response.get_json()
            self.assertEqual(response.status_code, 200)
            self.assertTrue(data["answer"])
            self.assertTrue(data["suggested_questions"])


if __name__ == "__main__":
    unittest.main()
