from __future__ import annotations

from functools import lru_cache
from typing import Any

from backend.services import artifact_loader
from backend.services.prediction_service import get_drug_name
from backend.services.validation_service import normalize_drug_id, resolve_drug
from backend.utils.errors import ArtifactError, NotFoundError, ValidationError


RDKIT_MISSING_MESSAGE = "Structure image is unavailable because the RDKit dependency is not installed."
RDKIT_INIT_MESSAGE = "Structure image is unavailable because the RDKit dependency could not be initialized."
MOLECULE_ARTIFACT_MESSAGE = "Structure image is unavailable because the molecule artifact could not be loaded."
MOLECULE_NOT_FOUND_MESSAGE = "Structure image unavailable for this drug."


@lru_cache(maxsize=1)
def _rdkit_modules() -> dict[str, Any]:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdDepictor, rdMolDescriptors
        from rdkit.Chem.Draw import rdMolDraw2D
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised when deployment lacks RDKit
        if str(getattr(exc, "name", "")).split(".", 1)[0] == "rdkit":
            return {
                "available": False,
                "reason": "rdkit_missing",
                "message": RDKIT_MISSING_MESSAGE,
                "Chem": None,
                "Descriptors": None,
                "rdDepictor": None,
                "rdMolDescriptors": None,
                "rdMolDraw2D": None,
            }
        return {
            "available": False,
            "reason": "rdkit_init_failed",
            "message": RDKIT_INIT_MESSAGE,
            "Chem": None,
            "Descriptors": None,
            "rdDepictor": None,
            "rdMolDescriptors": None,
            "rdMolDraw2D": None,
        }
    except Exception:  # pragma: no cover - dependency guard
        return {
            "available": False,
            "reason": "rdkit_init_failed",
            "message": RDKIT_INIT_MESSAGE,
            "Chem": None,
            "Descriptors": None,
            "rdDepictor": None,
            "rdMolDescriptors": None,
            "rdMolDraw2D": None,
        }

    return {
        "available": True,
        "reason": "",
        "message": "",
        "Chem": Chem,
        "Descriptors": Descriptors,
        "rdDepictor": rdDepictor,
        "rdMolDescriptors": rdMolDescriptors,
        "rdMolDraw2D": rdMolDraw2D,
    }


def lookup_molecule(value: Any) -> dict[str, Any]:
    original_value = str(value or "").strip()
    validation_error: ValidationError | None = None
    resolved_from_directory = False

    try:
        resolved = resolve_drug(value)
        nsc = int(resolved["nsc"])
        drug_name = resolved["name"]
        original_value = resolved.get("input") or original_value
        resolved_from_directory = True
    except ValidationError as exc:
        validation_error = exc
        nsc_value = normalize_drug_id(value)
        if not nsc_value:
            raise
        nsc = int(nsc_value)
        drug_name = get_drug_name(nsc)

    rdkit = _rdkit_modules()
    if not rdkit["available"]:
        if not resolved_from_directory and validation_error is not None:
            raise validation_error
        return _structure_unavailable_payload(
            nsc,
            drug_name,
            requested_nsc=nsc,
            input_value=original_value,
            reason=rdkit["reason"],
            message=rdkit["message"],
            rdkit_available=False,
        )

    try:
        molecules = artifact_loader.load_molecules()
    except ArtifactError:
        if not resolved_from_directory and validation_error is not None:
            raise validation_error
        return _structure_unavailable_payload(
            nsc,
            drug_name,
            requested_nsc=nsc,
            input_value=original_value,
            reason="artifact_unavailable",
            message=MOLECULE_ARTIFACT_MESSAGE,
            rdkit_available=True,
        )

    molecule = molecules.get(nsc) or molecules.get(str(nsc))
    if molecule is None:
        if not resolved_from_directory and validation_error is not None:
            raise validation_error
        return _not_found_payload(nsc, drug_name, input_value=original_value)

    return molecule_payload(nsc, molecule, requested_nsc=nsc, drug_name=drug_name, input_value=original_value)


def molecule_pair(payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = payload or {}
    left_value = (
        payload.get("drug1_input")
        or payload.get("NSC1")
        or payload.get("nsc1")
        or payload.get("drug1_id")
        or payload.get("drug1")
    )
    right_value = (
        payload.get("drug2_input")
        or payload.get("NSC2")
        or payload.get("nsc2")
        or payload.get("drug2_id")
        or payload.get("drug2")
    )

    left = lookup_molecule(left_value)
    right = lookup_molecule(right_value)
    status = "success" if left.get("found") and right.get("found") else "error"
    return {
        "status": status,
        "NSC1": left,
        "NSC2": right,
        "molecule_1": left,
        "molecule_2": right,
        "all_structures_available": bool(left.get("structure_available") and right.get("structure_available")),
    }


def molecule_payload(
    nsc: int,
    molecule: Any,
    requested_nsc: int | None = None,
    drug_name: str | None = None,
    input_value: str | None = None,
) -> dict[str, Any]:
    display_name = drug_name or get_drug_name(nsc)
    rdkit = _rdkit_modules()
    if not rdkit["available"]:
        return _structure_unavailable_payload(
            nsc,
            display_name,
            requested_nsc=requested_nsc,
            input_value=input_value,
            reason=rdkit["reason"],
            message=rdkit["message"],
            rdkit_available=False,
        )

    Chem = rdkit["Chem"]
    Descriptors = rdkit["Descriptors"]
    rdMolDescriptors = rdkit["rdMolDescriptors"]

    smiles = Chem.MolToSmiles(molecule) if Chem is not None and molecule is not None else ""
    formula = rdMolDescriptors.CalcMolFormula(molecule) if rdMolDescriptors is not None and molecule is not None else ""
    molecular_weight = float(Descriptors.MolWt(molecule)) if Descriptors is not None and molecule is not None else None
    svg = _molecule_svg(molecule)
    structure_available = bool(svg)
    structure_message = "" if structure_available else MOLECULE_NOT_FOUND_MESSAGE
    return {
        "status": "success",
        "found": True,
        "molecule_found": True,
        "requested_nsc": requested_nsc or nsc,
        "used_nsc": nsc,
        "resolved_nsc": nsc,
        "canonical_nsc": nsc,
        "alias_used": bool(requested_nsc and int(requested_nsc) != int(nsc)),
        "input_value": input_value or str(requested_nsc or nsc),
        "requested_value": input_value or str(requested_nsc or nsc),
        "drug_name": display_name,
        "name": display_name,
        "display_name": display_name,
        "canonical_drug_name": display_name,
        "nsc": nsc,
        "NSC": nsc,
        "smiles": smiles,
        "SMILES": smiles,
        "molecular_formula": formula,
        "formula": formula,
        "molecular_weight": round(molecular_weight, 3) if molecular_weight is not None else None,
        "source": "molecules/drug_mols.pkl",
        "structure_svg": svg,
        "svg": svg,
        "structure_available": structure_available,
        "structure_status": "available" if structure_available else "unavailable",
        "structure_message": structure_message,
        "message": structure_message,
        "error": "",
        "rdkit_available": True,
    }


def require_molecule(value: Any) -> dict[str, Any]:
    result = lookup_molecule(value)
    if not result.get("found"):
        raise NotFoundError(result.get("error", "Molecule was not found."), code="MOLECULE_NOT_FOUND")
    return result


def _molecule_svg(molecule: Any) -> str:
    rdkit = _rdkit_modules()
    Chem = rdkit["Chem"]
    rdDepictor = rdkit["rdDepictor"]
    rdMolDraw2D = rdkit["rdMolDraw2D"]
    if rdMolDraw2D is None or Chem is None:
        return ""
    try:
        draw_molecule = Chem.Mol(molecule)
        if rdDepictor is not None:
            rdDepictor.Compute2DCoords(draw_molecule)
        drawer = rdMolDraw2D.MolDraw2DSVG(360, 260)
        drawer.DrawMolecule(draw_molecule)
        drawer.FinishDrawing()
        return drawer.GetDrawingText().replace("svg:", "")
    except Exception:
        return ""


def _base_profile(
    nsc: int,
    drug_name: str | None = None,
    requested_nsc: int | None = None,
    input_value: str | None = None,
) -> dict[str, Any]:
    display_name = drug_name or get_drug_name(nsc)
    requested = requested_nsc or nsc
    return {
        "status": "success",
        "found": True,
        "metadata_found": True,
        "requested_nsc": requested,
        "used_nsc": nsc,
        "resolved_nsc": nsc,
        "canonical_nsc": nsc,
        "alias_used": bool(requested_nsc and int(requested_nsc) != int(nsc)),
        "input_value": input_value or str(requested),
        "requested_value": input_value or str(requested),
        "drug_name": display_name,
        "name": display_name,
        "display_name": display_name,
        "canonical_drug_name": display_name,
        "nsc": nsc,
        "NSC": nsc,
        "source": "molecules/drug_mols.pkl",
    }


def _structure_unavailable_payload(
    nsc: int,
    drug_name: str | None = None,
    requested_nsc: int | None = None,
    input_value: str | None = None,
    reason: str = "unavailable",
    message: str = MOLECULE_NOT_FOUND_MESSAGE,
    rdkit_available: bool = False,
) -> dict[str, Any]:
    return {
        **_base_profile(nsc, drug_name, requested_nsc, input_value),
        "molecule_found": False,
        "structure_available": False,
        "structure_status": reason,
        "structure_message": message,
        "message": message,
        "structure_svg": "",
        "svg": "",
        "molecular_formula": "",
        "formula": "",
        "molecular_weight": None,
        "smiles": "",
        "SMILES": "",
        "error": "",
        "rdkit_available": rdkit_available,
    }


def _not_found_payload(nsc: int, drug_name: str | None = None, input_value: str | None = None) -> dict[str, Any]:
    return {
        **_structure_unavailable_payload(
            nsc,
            drug_name,
            requested_nsc=nsc,
            input_value=input_value,
            reason="not_found",
            message=MOLECULE_NOT_FOUND_MESSAGE,
            rdkit_available=True,
        ),
        "molecule_found": False,
    }
