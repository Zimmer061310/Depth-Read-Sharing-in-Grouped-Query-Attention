"""Canonical numerical execution settings for the evaluated model classes."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# These identify the original evaluated configuration on saved model configs;
# they are not hashes of the public, scientifically scoped settings projection.
POLICY_ID = "canonical-training-v1"
POLICY_SHA256 = "b79bc0893d2fbd7c04fcded199dc5f9e89d27e50148b8c1109b5988f0e985909"


def _settings():
    return json.loads((ROOT.parent / "configs/execution.json").read_text())


def canonical_policy():
    value = _settings()
    identity = value["historical_config_identities"]
    if (identity["execution_policy_id"] != POLICY_ID
            or identity["execution_policy_sha256"] != POLICY_SHA256
            or value["router_backend"] != "canonical"
            or value["query_projection_output_widths"] != [640, 640]
            or value["logical_query_head_order"] != "group_major_q2_then_q1"
            or value["attention_implementation"] != "sdpa"):
        raise ValueError("evaluated numerical execution settings differ")
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
