from __future__ import annotations

import unittest

from app import app


class MoleculeLookupTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_molecule_pair_returns_local_metadata_for_quantum_demo_drugs(self):
        response = self.client.post(
            "/api/molecule-pair",
            json={"NSC1": 3053, "NSC2": 180973, "model_type": "quantum"},
        )

        data = response.get_json()
        self.assertEqual(response.status_code, 200, data)
        self.assertTrue(data["success"])
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["NSC1"]["drug_name"], "Dactinomycin")
        self.assertEqual(data["NSC1"]["mechanism"], "RNA synthesis inhibitor")
        self.assertEqual(data["NSC2"]["drug_name"], "Tamoxifen citrate")
        self.assertIn("Selective estrogen receptor", data["NSC2"]["mechanism"])
        self.assertIn("description", data["NSC1"])
        self.assertIn("administration_route", data["NSC2"])

    def test_drug_info_route_returns_profile_metadata(self):
        response = self.client.get("/api/drug_info/740")

        data = response.get_json()
        self.assertEqual(response.status_code, 200, data)
        self.assertTrue(data["success"])
        self.assertEqual(data["name"], "Methotrexate")
        self.assertEqual(data["mechanism"], "Antifolate")
        self.assertEqual(data["targets"], "DHFR")
        self.assertEqual(data["class"], "Chemotherapy")


if __name__ == "__main__":
    unittest.main()
