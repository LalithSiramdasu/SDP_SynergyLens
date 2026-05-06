from __future__ import annotations

from flask import Blueprint, request

from backend.services import molecule_service
from backend.utils.response import success_response


molecule_bp = Blueprint("molecules", __name__)


@molecule_bp.post("/api/molecule-pair")
def api_molecule_pair():
    data = molecule_service.molecule_pair(request.get_json(silent=True))
    return success_response(data, **data)


@molecule_bp.get("/api/molecule/<value>")
@molecule_bp.get("/api/molecules/<value>")
def api_molecule(value: str):
    data = molecule_service.require_molecule(value)
    return success_response(data, **data)


@molecule_bp.get("/api/drug_info/<int:drug_id>")
def api_drug_info(drug_id: int):
    molecule = molecule_service.lookup_molecule(drug_id)
    data = {
        "id": drug_id,
        "name": molecule.get("drug_name") or molecule.get("name") or f"NSC {drug_id}",
        "img_url": f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{drug_id}/PNG",
        "formula": molecule.get("molecular_formula") or "n/a",
        "weight": molecule.get("molecular_weight") or "n/a",
        "iupac": "n/a",
        "metadata_source": molecule.get("source") or "local fallback",
        "mechanism": molecule.get("mechanism") or "",
        "targets": molecule.get("targets") or "",
        "class": molecule.get("drug_class") or molecule.get("class") or "",
        "indications": molecule.get("indications") or "",
        "side_effects": molecule.get("side_effects") or "",
        "administration_route": molecule.get("administration_route") or "",
        "description": molecule.get("description") or "",
        "structure_available": bool(molecule.get("structure_available")),
        "structure_message": molecule.get("structure_message") or molecule.get("message") or "",
    }
    return success_response(data, **data)
