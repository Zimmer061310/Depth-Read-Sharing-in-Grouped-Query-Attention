"""Deterministic numerical runtime used by the evaluated model classes."""
from contextlib import contextmanager
import os

from src.experiments.paper_sharing_execution_policy import (
    POLICY_ID, POLICY_SHA256, _settings,
)

RUNTIME_POLICY_ID = "strict-determinism-v1"
RUNTIME_POLICY_SHA256 = "f940caa0ac27a48aa179909425a420e1e5004a03b2f0f72e306cb92aa9568185"


def runtime_policy():
    value = _settings()
    identity = value["historical_config_identities"]
    if (identity["runtime_policy_id"] != RUNTIME_POLICY_ID
            or identity["runtime_policy_sha256"] != RUNTIME_POLICY_SHA256
            or value["deterministic_algorithms"] is not True
            or value["deterministic_warn_only"] is not False
            or value["cudnn_deterministic"] is not True
            or value["cudnn_benchmark"] is not False
            or value["cublas_workspace_config"] != ":4096:8"):
        raise ValueError("evaluated deterministic runtime settings differ")
    return value


def runtime_binding(config, *, check_runtime=False):
    identifier = getattr(config, "paper_sharing_runtime_policy", None)
    digest = getattr(config, "paper_sharing_runtime_policy_sha256", None)
    if identifier is None and digest is None:
        return None
    runtime_policy()
    if (identifier != RUNTIME_POLICY_ID or digest != RUNTIME_POLICY_SHA256
            or getattr(config, "paper_sharing_execution_policy", None) != POLICY_ID
            or getattr(config, "paper_sharing_execution_policy_sha256", None) != POLICY_SHA256
            or config._attn_implementation != "sdpa"):
        raise ValueError("strict runtime requires exact canonical and runtime bindings")
    if check_runtime:
        require_runtime(fixture=config.paper_sharing_test_fixture)
    return {"policy_id": RUNTIME_POLICY_ID, "sha256": RUNTIME_POLICY_SHA256}


def require_runtime(*, fixture=False):
    import torch
    spec = runtime_policy()
    if (not torch.are_deterministic_algorithms_enabled()
            or torch.is_deterministic_algorithms_warn_only_enabled()
            or not torch.backends.cudnn.deterministic or torch.backends.cudnn.benchmark):
        raise ValueError("strict runtime flags changed or warn-only enabled")
    if not fixture and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != spec["cublas_workspace_config"]:
        raise ValueError("strict CUDA runtime requires CUBLAS_WORKSPACE_CONFIG=:4096:8")


def apply_runtime(*, fixture=False):
    """Owned worker startup only. No silent backend/precision fallback."""
    import torch
    spec = runtime_policy()
    if not fixture and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != spec["cublas_workspace_config"]:
        raise ValueError("set strict cuBLAS environment before CUDA initialization")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    require_runtime(fixture=fixture)


@contextmanager
def strict_runtime(*, fixture=False):
    """Local fixtures/explicit callers restore the prior global flags on exit."""
    import torch
    saved = (torch.are_deterministic_algorithms_enabled(),
             torch.is_deterministic_algorithms_warn_only_enabled(),
             torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    try:
        apply_runtime(fixture=fixture)
        yield
    finally:
        torch.use_deterministic_algorithms(saved[0], warn_only=saved[1])
        torch.backends.cudnn.deterministic = saved[2]
        torch.backends.cudnn.benchmark = saved[3]
