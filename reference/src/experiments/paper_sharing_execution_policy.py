"""Hash-bound opt-in execution amendment; no launch or mutable recipe rewrite."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POLICY_ID = "canonical-training-v1"
POLICY_PATH = ROOT / "configs/paper_sharing_validation_2k/canonical-training-policy.v1.json"
POLICY_SHA256 = "b79bc0893d2fbd7c04fcded199dc5f9e89d27e50148b8c1109b5988f0e985909"
CANONICAL_CONTRACT_PATH = ROOT / "configs/paper_sharing_validation_2k/cuda-preflight-canonical.v3.json"
CANONICAL_CONTRACT_SHA256 = "d260e83b886b76fba3b61dfa90ec0734c588dc22417cb6e64092fb2bb2ce46b9"


def _read(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("execution-policy/contract byte identity changed")
    return json.loads(raw)


def canonical_policy():
    value = _read(POLICY_PATH, POLICY_SHA256)
    if (value["policy_id"] != POLICY_ID or value["router_backend"] != "canonical"
            or value["replaces_only"] != "router_backend_intent"
            or value["paid_execution_authorized"] or value["fresh_scientific_training_authorized"]):
        raise ValueError("invalid canonical execution amendment")
    return value


def policy_binding(config):
    identifier = getattr(config, "paper_sharing_execution_policy", None)
    expected = getattr(config, "paper_sharing_execution_policy_sha256", None)
    if identifier is None and expected is None:
        return None
    canonical_policy()
    if identifier != POLICY_ID or expected != POLICY_SHA256 or config.paper_sharing_router_backend != "canonical":
        raise ValueError("execution policy/backend binding differs")
    return {"policy_id": POLICY_ID, "sha256": POLICY_SHA256}


def canonical_contract():
    """Derive v3 from immutable v2; no numeric or scientific field overrides."""
    extension = _read(CANONICAL_CONTRACT_PATH, CANONICAL_CONTRACT_SHA256)
    policy = canonical_policy()
    base = ROOT / "configs/paper_sharing_validation_2k/cuda-preflight-contract.v2.json"
    value = _read(base, extension["base_contract_sha256"])
    if extension["execution_policy"] != POLICY_ID or policy["failed_preflight_contract_sha256"] != extension["base_contract_sha256"]:
        raise ValueError("canonical contract inheritance differs")
    value["contract_id"] = extension["contract_id"]
    value["execution_policy"] = {"policy_id": POLICY_ID, "sha256": POLICY_SHA256}
    value["comparison_semantics"] = extension["full_loss_and_vjp_semantics"]
    # Retain every numeric comparison limit. Scope wording must not claim a
    # successful comparison against the rejected optimized backend.
    for name in ("router_bf16_forward", "router_bf16_vjp", "full_model_sequence_loss", "full_model_vjp_and_update"):
        value["comparisons"][name]["scope"] = extension["full_loss_and_vjp_semantics"] if name.startswith("full_model") else "actual canonical condition dispatcher versus independent canonical equations"
    value["execution_status"] = "canonical implementation prepared; CUDA, runtime, paid approval and new campaign remain unverified"
    return value
