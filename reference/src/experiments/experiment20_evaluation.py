"""Experiment 20 multi-slot route-count fixed evaluation and paired analysis.

This module evaluates the trained R=3/R=4/R=8 multi-slot models on the exact
fixed discovery and confirmation artifact shared with Experiments 9-19,
compares them against the accepted R=2 RS-HQ8, HQ8, and M8 references only
after exact identity validation, and measures per-slot router differentiation.

Training identity is pinned to ``training_source_commit``.  Evaluation code may
advance in a descendant commit; the descendant relation is verified and the
instrumentation commit is recorded separately.
"""

from __future__ import annotations

import math
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.experiments.experiment1_partition_compatibility import (
    atomic_write_json,
    load_fixed_eval_artifact,
    sha256_file,
    sha256_path,
    utc_now,
)
from src.experiments.experiment6_screening import evaluate_split
from src.experiments.experiment20_multislot_routing import (
    Experiment20MHARForCausalLM,
    experiment20_parameter_report,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/experiment20/evaluation_protocol.json"
CONDITIONS = ("r3", "r4", "r8")
SPLITS = ("discovery", "confirmation")
REFERENCE_ORDER = ("r2", "hq8", "m8")


def read_json(path: Path) -> dict:
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_protocol(path: Path = PROTOCOL) -> dict:
    spec = read_json(path)
    if spec.get("experiment") != "20-multislot-role-specific-mhar-v1":
        raise ValueError("Experiment 20 evaluation protocol name changed")
    if (spec.get("seed"), spec.get("milestone"), spec.get("schedule_steps")) != (
        42, 2000, 20000
    ):
        raise ValueError("Experiment 20 seed or milestone changed")
    if tuple(spec.get("conditions", {})) != CONDITIONS:
        raise ValueError("Experiment 20 condition set changed")
    expected_routes = {"r3": 3, "r4": 4, "r8": 8}
    for name, route_count in expected_routes.items():
        if spec["conditions"][name]["route_count"] != route_count:
            raise ValueError(f"Experiment 20 {name} route count changed")
    if spec["bootstrap"] != {"samples": 10000, "seed": 20042, "confidence": 0.95}:
        raise ValueError("Experiment 20 bootstrap protocol changed")
    for name, reference in spec["references"].items():
        if name not in REFERENCE_ORDER:
            raise ValueError("unknown Experiment 20 reference")
        if len(reference["checkpoint_sha256"]) != 64:
            raise ValueError("Experiment 20 reference hash is malformed")
    if spec["fixed_evaluation"]["sha256"] != (
        "29e545dd9399e9eaea6f5abf38f20ef76ba232466a01b6ddf13c3a6a287a3691"
    ):
        raise ValueError("Experiment 20 fixed evaluation artifact changed")
    if not isinstance(spec["training_source_commit"], str) or len(
        spec["training_source_commit"]
    ) != 40:
        raise ValueError("Experiment 20 training source commit is malformed")
    return spec


def protocol_sha256(path: Path = PROTOCOL) -> str:
    return sha256_file(path)


def git_commit() -> str:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if len(commit) != 40:
        raise RuntimeError("Experiment 20 requires a pinned evaluation commit")
    return commit


def require_clean_worktree() -> str:
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if status.strip():
        raise RuntimeError("Experiment 20 evaluation requires a clean worktree")
    return git_commit()


def require_descendant(training_source_commit: str) -> str:
    """Require the evaluation commit to descend from the frozen training commit."""
    evaluation_commit = require_clean_worktree()
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", training_source_commit, evaluation_commit],
        cwd=ROOT,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Experiment 20 evaluation commit does not descend from the training commit")
    return evaluation_commit


def _verified_reconstruction(payload: dict, observed: str, spec: dict) -> dict:
    """Fail closed: the historical fixture hashes cover only four discovery rows.

    They cannot establish full discovery or confirmation identity. A future
    recovery must supply independently anchored full-split proof through a
    separately reviewed verifier, or recover the exact frozen artifact.
    """
    raise ValueError(
        "Experiment 20 artifact identity unverified: serialized hash mismatch; "
        "partial Experiment 18 fingerprints do not prove full-split identity")


def verify_artifact(artifact: Path, spec: dict) -> tuple[dict, dict]:
    payload, observed = load_fixed_eval_artifact(Path(artifact))
    for split in SPLITS:
        tensor = payload[f"{split}_input_ids"]
        expected = (
            spec["fixed_evaluation"][f"{split}_sequences"],
            spec["fixed_evaluation"]["sequence_length"],
        )
        if tuple(tensor.shape) != expected:
            raise ValueError(f"Experiment 20 {split} artifact shape mismatch")
    frozen = spec["fixed_evaluation"]["sha256"]
    if observed == frozen:
        return payload, {
            "frozen_sha256": frozen,
            "observed_sha256": observed,
            "identity": "exact",
        }
    return payload, _verified_reconstruction(payload, observed, spec)


def _validate_split_row(run_id, split, metrics, spec, *, has_per_sequence_tokens):
    count = spec["fixed_evaluation"][f"{split}_sequences"]
    length = spec["fixed_evaluation"]["sequence_length"] - 1
    losses = metrics["sequence_losses"]
    if (len(losses) != count
            or metrics["valid_tokens"] != count * length
            or not all(math.isfinite(float(value)) for value in losses)):
        raise ValueError(f"{run_id} {split} sequence mismatch")
    if has_per_sequence_tokens:
        if metrics["sequence_valid_tokens"] != [length] * count:
            raise ValueError(f"{run_id} {split} per-sequence token mismatch")
    recomputed = float(np.mean(np.asarray(losses, dtype=np.float64)))
    if not math.isclose(recomputed, float(metrics["loss"]), abs_tol=5e-6, rel_tol=0):
        raise ValueError(f"{run_id} {split} aggregate loss mismatch")
    return {
        "sequence_losses": list(losses),
        "sequence_valid_tokens": [length] * count,
        "loss": float(metrics["loss"]),
    }


def validate_reference(model_id: str, spec: dict, *, split: str | None = None) -> dict:
    """Return validated per-split metrics for an accepted reference.

    The reference result files are committed repository artifacts; the
    scheduled checkpoint itself need not remain on disk.
    """
    if model_id not in REFERENCE_ORDER:
        raise ValueError("unknown Experiment 20 reference")
    reference = spec["references"][model_id]
    splits = (split,) if split else SPLITS

    if model_id == "r2":
        out = {}
        for name in splits:
            result = read_json(ROOT / reference["result"][name])
            if (result.get("artifact_sha256") != spec["fixed_evaluation"]["sha256"]
                    or result.get("checkpoint_sha256") != reference["checkpoint_sha256"]
                    or result.get("run_id") != f"{reference['run_id_prefix']}-{name}"
                    or result.get("condition") != reference["condition"]
                    or result.get("split") != name
                    or result.get("milestone") != spec["milestone"]):
                raise ValueError(f"accepted R2 reference identity mismatch ({name})")
            out[name] = _validate_split_row(
                result["run_id"], name, result["metrics"], spec,
                has_per_sequence_tokens=True)
        return out

    result = read_json(ROOT / reference["result"])
    if (result.get("artifact_sha256") != spec["fixed_evaluation"]["sha256"]
            or result.get("checkpoint_sha256") != reference["checkpoint_sha256"]):
        raise ValueError(f"accepted {model_id} reference identity mismatch")
    provenance = read_json(ROOT / reference["provenance"])
    if model_id == "hq8":
        identity = provenance.get("run_identity", {})
        if (provenance.get("global_step") != 2000
                or provenance.get("chunks_consumed") != 64_000
                or identity.get("seed") != 42
                or identity.get("experiment8_variant") != "hq8"
                or identity.get("dataset_revision")
                    != "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
                or identity.get("tokenizer_revision")
                    != "c1899de289a04d12100db370d81485cdf75e47ca"):
            raise ValueError("accepted HQ8 reference training provenance mismatch")
    else:
        identity = provenance.get("run_identity", {})
        architecture = provenance.get("architecture", {})
        if (identity.get("screening_seed") != 42
                or identity.get("variant") != "h8"
                or identity.get("checkpoint_sha256") != reference["checkpoint_sha256"]
                or identity.get("artifact_sha256") != spec["fixed_evaluation"]["sha256"]
                or architecture.get("attnres_mode") != "full_mh"
                or architecture.get("attnres_num_heads") != 8):
            raise ValueError("accepted M8 reference provenance mismatch")
    out = {}
    for name in splits:
        out[name] = _validate_split_row(
            f"{model_id}-{name}", name, result["splits"][name], spec,
            has_per_sequence_tokens=False)
    return out


def _accepted_hq8_data_files(spec: dict) -> dict:
    manifest = read_json(
        ROOT / "results/experiment8/provenance/step-2000/hq8/training_run_manifest.json")
    files = manifest["run_identity"]["data_files"]["files"]
    return {Path(row["path"]).name: (row["bytes"], row["sha256"]) for row in files}


def validate_training_checkpoint(checkpoint: Path, condition: str, spec: dict) -> dict:
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_dir() or checkpoint.name != "step-2000":
        raise ValueError("Experiment 20 requires an atomic step-2000 checkpoint")
    required = ("config.json", "training_manifest.json", "training_state.pt")
    if any(not (checkpoint / name).is_file() for name in required):
        raise ValueError("incomplete Experiment 20 checkpoint")
    if not (checkpoint / "model.safetensors").is_file() and not (
        checkpoint / "model.safetensors.index.json"
    ).is_file():
        raise ValueError("Experiment 20 checkpoint has no model weights")

    route_count = int(spec["conditions"][condition]["route_count"])
    manifest = read_json(checkpoint / "training_manifest.json")
    identity = manifest.get("run_identity", {})
    expected = {
        "mode": "full_mh", "attnres_heads": 8,
        "hidden_size": 1280, "num_layers": 36,
        "num_heads": 16, "num_kv_heads": 8,
        "intermediate_size": 5120, "seq_len": 1024,
        "steps": 20000, "global_batch_size": 32,
        "lr": 5e-4, "lr_min": 5e-5, "warmup": 1000,
        "seed": 42,
        "experiment20_route_count": route_count,
        "experiment20_multislot_router": True,
        "experiment20_router_backend": "tiled-r2-triton",
        "fused": True, "grad_ckpt": True,
        "dataset": "HuggingFaceFW/fineweb-edu",
        "dataset_name": "sample-10BT",
        "dataset_revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "tokenizer": "Qwen/Qwen3-0.6B",
        "tokenizer_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "optimizer": "AdamW", "optimizer_betas": [0.9, 0.95],
        "optimizer_eps": 1e-8, "weight_decay": 0.1,
        "per_gpu_batch_size": 4, "grad_accum": 8, "world_size": 1,
        "experiment8_hybrid_q_groups": 8,
        "experiment8_local_head_position": "even",
        "experiment8_global_head_position": "odd",
        "source_commit": spec["training_source_commit"],
    }
    mismatches = {
        key: {"expected": value, "observed": identity.get(key)}
        for key, value in expected.items() if identity.get(key) != value
    }
    if (manifest.get("global_step") != spec["milestone"]
            or manifest.get("chunks_consumed") != 64_000 or mismatches):
        raise ValueError(f"Experiment 20 training identity mismatch: {mismatches}")

    recorded_files = identity.get("data_files", {}).get("files", [])
    actual_files = {
        Path(row["path"]).name: (row["bytes"], row["sha256"])
        for row in recorded_files
    }
    if actual_files != _accepted_hq8_data_files(spec):
        raise ValueError("Experiment 20 training data bytes differ from accepted HQ8")

    state = torch.load(
        checkpoint / "training_state.pt", map_location="cpu",
        weights_only=False, mmap=True,
    )
    required_state = (
        "global_step", "chunks_consumed", "optimizer", "scheduler",
        "cpu_rng_state", "cuda_rng_state_all", "run_identity",
    )
    if (any(key not in state for key in required_state)
            or state.get("global_step") != spec["milestone"]
            or state.get("chunks_consumed") != 64_000
            or state.get("run_identity") != identity
            or not state["optimizer"].get("state")
            or not state["scheduler"]
            or not isinstance(state["cpu_rng_state"], torch.Tensor)
            or not state["cuda_rng_state_all"]):
        raise ValueError("incomplete Experiment 20 optimizer/RNG/data-position state")
    del state

    config = read_json(checkpoint / "config.json")
    if (config.get("experiment20_route_count") != route_count
            or config.get("experiment20_multislot_router") is not True
            or config.get("experiment20_router_backend") != "tiled-r2-triton"):
        raise ValueError("checkpoint model config omits the multi-slot route graph")
    return {
        "checkpoint_sha256": sha256_path(checkpoint),
        "training_manifest": manifest,
        "route_count": route_count,
    }


def condition_path(root: Path, condition: str, split: str) -> Path:
    if condition not in CONDITIONS or split not in SPLITS:
        raise ValueError("unknown Experiment 20 condition or split")
    return Path(root) / "evaluation" / condition / split / "result.json"


def probe_path(root: Path, condition: str) -> Path:
    if condition not in CONDITIONS:
        raise ValueError("unknown Experiment 20 condition")
    return Path(root) / "probe" / condition / "result.json"


def write_once(path: Path, value: dict) -> None:
    if path.exists() or path.with_suffix(path.suffix + ".tmp").exists():
        raise FileExistsError(f"refusing to replace accepted output: {path}")
    atomic_write_json(path, value)


def evaluate_condition(
    *, checkpoint: Path, artifact: Path, root: Path, split: str,
    condition: str, device: torch.device, dtype: torch.dtype,
    batch_size: int, wandb_mode: str = "disabled",
    wandb_entity: str | None = None,
) -> dict:
    spec = load_protocol()
    if condition not in CONDITIONS or split not in SPLITS:
        raise ValueError("unknown Experiment 20 evaluation condition")
    output = condition_path(root, condition, split)
    if output.exists():
        raise FileExistsError(f"evaluation already exists: {output}")
    evaluation_commit = require_descendant(spec["training_source_commit"])
    checkpoint_info = validate_training_checkpoint(checkpoint, condition, spec)
    payload, artifact_verification = verify_artifact(artifact, spec)
    artifact_hash = artifact_verification["frozen_sha256"]
    if split == "confirmation":
        verify_frozen_discovery(root, spec)

    model = Experiment20MHARForCausalLM.from_pretrained(
        str(checkpoint), dtype=dtype,
    ).to(device=device).eval()
    parameters = experiment20_parameter_report(model)
    run = None
    if wandb_mode != "disabled":
        import wandb
        run = wandb.init(
            project=spec["wandb"]["project"], entity=wandb_entity,
            group=spec["wandb"]["group"],
            name=f"mhar-exp20-{condition}-{split}",
            job_type="fixed-evaluation", mode=wandb_mode,
            tags=["experiment-20", condition, split, "seed-42"],
            config={
                "checkpoint_sha256": checkpoint_info["checkpoint_sha256"],
                "artifact_sha256": artifact_hash,
                "protocol_sha256": protocol_sha256(),
            },
        )
    try:
        metrics = evaluate_split(
            model, payload[f"{split}_input_ids"],
            batch_size=batch_size, device=device,
        )
        length = spec["fixed_evaluation"]["sequence_length"] - 1
        metrics["sequence_valid_tokens"] = [length] * len(metrics["sequence_losses"])
        metrics["sequence_total_losses"] = [
            float(value) * length for value in metrics["sequence_losses"]
        ]
        result = {
            "format_version": 1,
            "created_at": utc_now(),
            "experiment": spec["experiment"],
            "run_id": f"exp20-{condition}-s42",
            "condition": condition,
            "route_count": checkpoint_info["route_count"],
            "split": split,
            "seed": spec["seed"],
            "milestone": spec["milestone"],
            "artifact_sha256": artifact_hash,
            "checkpoint_sha256": checkpoint_info["checkpoint_sha256"],
            "protocol_sha256": protocol_sha256(),
            "training_source_commit": spec["training_source_commit"],
            "evaluation_source_commit": evaluation_commit,
            "artifact_verification": artifact_verification,
            "parameters": parameters,
            "metrics": metrics,
            "wandb": ({"run_id": run.id, "run_url": run.url} if run else None),
        }
        write_once(output, result)
        if run:
            run.log({f"{split}/loss": metrics["loss"],
                     f"{split}/tokens_per_second": metrics["tokens_per_second"],
                     f"{split}/peak_cuda_allocated_bytes":
                         metrics["peak_cuda_allocated_bytes"]})
        return result
    finally:
        if run:
            run.finish()
        del model


def discovery_manifest(root: Path, spec: dict) -> dict:
    root = Path(root)
    paths = {
        condition: condition_path(root, condition, "discovery")
        for condition in CONDITIONS
    }
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("all discovery conditions must finish before confirmation")
    return {
        "format_version": 1,
        "experiment": spec["experiment"],
        "training_source_commit": spec["training_source_commit"],
        "protocol_sha256": protocol_sha256(),
        "discovery_result_sha256": {
            condition: sha256_file(path) for condition, path in paths.items()
        },
        "reference_result_sha256": {
            model_id: sha256_file(ROOT / row["result"])
            for model_id, row in spec["references"].items()
            if model_id in ("hq8", "m8")
        },
        "conditions": list(CONDITIONS),
        "contrasts": {
            condition: {
                "primary": f"{condition}-minus-r2",
                "baseline_hq8": f"{condition}-minus-hq8",
                "baseline_m8": f"{condition}-minus-m8",
            }
            for condition in CONDITIONS
        },
        "bootstrap": spec["bootstrap"],
        "candidate_selection": "none",
    }


def freeze_discovery(root: Path, spec: dict | None = None) -> Path:
    spec = spec or load_protocol()
    require_descendant(spec["training_source_commit"])
    for name in REFERENCE_ORDER:
        validate_reference(name, spec)
    record = discovery_manifest(root, spec)
    destination = Path(root) / "selections" / "no_selection_manifest.json"
    write_once(destination, record)
    return destination


def verify_frozen_discovery(root: Path, spec: dict | None = None) -> dict:
    spec = spec or load_protocol()
    path = Path(root) / "selections" / "no_selection_manifest.json"
    if not path.is_file():
        raise ValueError("confirmation is locked until discovery is frozen")
    recorded = read_json(path)
    expected = discovery_manifest(root, spec)
    if recorded != expected:
        raise ValueError("discovery selection manifest or accepted inputs changed")
    return recorded


def load_condition(root: Path, condition: str, split: str, spec: dict) -> dict:
    result = read_json(condition_path(root, condition, split))
    verification = result.get("artifact_verification", {})
    if (verification.get("identity") != "exact"
            or verification.get("observed_sha256") != spec["fixed_evaluation"]["sha256"]):
        raise ValueError("Experiment 20 reference pairing requires exact artifact identity")
    metrics = result["metrics"]
    count = spec["fixed_evaluation"][f"{split}_sequences"]
    length = spec["fixed_evaluation"]["sequence_length"] - 1
    if (result.get("artifact_sha256") != spec["fixed_evaluation"]["sha256"]
            or result.get("protocol_sha256") != protocol_sha256()
            or result.get("condition") != condition
            or result.get("route_count") != spec["conditions"][condition]["route_count"]
            or result.get("milestone") != spec["milestone"]
            or result.get("training_source_commit") != spec["training_source_commit"]
            or len(metrics["sequence_losses"]) != count
            or metrics["sequence_valid_tokens"] != [length] * count
            or metrics["valid_tokens"] != count * length):
        raise ValueError(f"Experiment 20 {condition}/{split} result mismatch")
    return result


def paired_bootstrap(a: dict, b: dict, *, samples: int = 10000,
                     seed: int = 20042) -> dict:
    left = np.asarray(a["sequence_losses"], dtype=np.float64)
    right = np.asarray(b["sequence_losses"], dtype=np.float64)
    left_n = np.asarray(a["sequence_valid_tokens"], dtype=np.float64)
    right_n = np.asarray(b["sequence_valid_tokens"], dtype=np.float64)
    if (left.shape != right.shape or left.shape != left_n.shape
            or right.shape != right_n.shape or not np.array_equal(left_n, right_n)
            or (left_n <= 0).any() or not np.isfinite(left).all()
            or not np.isfinite(right).all()):
        raise ValueError("paired per-sequence Losses or valid-token counts mismatch")
    delta = left - right
    point = float(np.dot(delta, left_n) / left_n.sum())
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for offset in range(0, samples, 512):
        width = min(512, samples - offset)
        index = rng.integers(0, len(delta), size=(width, len(delta)))
        weights = left_n[index]
        draws[offset:offset + width] = (
            (delta[index] * weights).sum(axis=1) / weights.sum(axis=1)
        )
    confidence = 0.95
    return {
        "delta_loss": point,
        "ci95_low": float(np.quantile(draws, (1 - confidence) / 2)),
        "ci95_high": float(np.quantile(draws, 1 - (1 - confidence) / 2)),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def _contrast_definitions() -> list[tuple[str, str, str]]:
    rows = []
    for condition in CONDITIONS:
        rows.append((f"{condition}-minus-r2", condition, "r2"))
        rows.append((f"{condition}-minus-hq8", condition, "hq8"))
        rows.append((f"{condition}-minus-m8", condition, "m8"))
    rows.append(("r4-minus-r3", "r4", "r3"))
    rows.append(("r8-minus-r4", "r8", "r4"))
    return rows


def _m8_query_macs(spec: dict) -> int:
    # M8 dense query projection: hidden * query_heads * head_dim.
    return 1280 * 16 * 80


def analyze(root: Path, *, require_confirmation: bool = True) -> dict:
    spec = load_protocol()
    require_descendant(spec["training_source_commit"])
    references = {name: validate_reference(name, spec) for name in REFERENCE_ORDER}
    splits = SPLITS if require_confirmation else ("discovery",)
    records = {}
    checkpoint_hashes = {}
    for split in splits:
        conditions = {
            condition: load_condition(root, condition, split, spec)
            for condition in CONDITIONS
        }
        for condition, result in conditions.items():
            checkpoint_hashes.setdefault(condition, result["checkpoint_sha256"])
            if checkpoint_hashes[condition] != result["checkpoint_sha256"]:
                raise ValueError("discovery and confirmation use different checkpoints")
        metrics = {
            condition: {
                "sequence_losses": result["metrics"]["sequence_losses"],
                "sequence_valid_tokens": result["metrics"]["sequence_valid_tokens"],
                "loss": result["metrics"]["loss"],
            }
            for condition, result in conditions.items()
        }
        metrics.update({name: references[name][split] for name in REFERENCE_ORDER})
        contrasts = {}
        for cid, left, right in _contrast_definitions():
            contrasts[cid] = paired_bootstrap(
                metrics[left], metrics[right],
                samples=spec["bootstrap"]["samples"], seed=spec["bootstrap"]["seed"])
        records[split] = {
            "loss": {
                model: metrics[model]["loss"] for model in
                ("r2", "r3", "r4", "r8", "hq8", "m8")
            },
            "contrasts": contrasts,
            "parameters": {
                condition: conditions[condition]["parameters"]
                for condition in CONDITIONS
            },
        }
    if require_confirmation:
        verify_frozen_discovery(root, spec)
        final = records["confirmation"]["contrasts"]
        m8_macs = _m8_query_macs(spec)
        decisions = {}
        for condition in CONDITIONS:
            primary = final[f"{condition}-minus-r2"]
            m8_baseline = final[f"{condition}-minus-m8"]
            macs = records["confirmation"]["parameters"][condition][
                "query_projection_macs_per_token_per_layer"]
            decisions[condition] = {
                "lower_confirmation_loss_than_r2":
                    primary["ci95_high"] < 0,
                "paired_delta_loss_vs_r2": primary["delta_loss"],
                "paired_delta_loss_vs_r2_ci95":
                    [primary["ci95_low"], primary["ci95_high"]],
                "lower_confirmation_loss_than_m8":
                    m8_baseline["ci95_high"] < 0,
                "query_projection_macs_per_token_per_layer": macs,
                "below_m8_query_projection_macs": macs < m8_macs,
                "within_measured_cost_region": True,
            }
        any_advantage = any(
            row["lower_confirmation_loss_than_r2"] and row["below_m8_query_projection_macs"]
            for row in decisions.values())
        decision = ("confirmed_multislot_screening_advantage" if any_advantage
                    else "no_confirmed_multislot_advantage")
    else:
        decisions = None
        decision = None
    return {
        "format_version": 1,
        "created_at": utc_now(),
        "experiment": spec["experiment"],
        "artifact_sha256": spec["fixed_evaluation"]["sha256"],
        "protocol_sha256": protocol_sha256(),
        "training_source_commit": spec["training_source_commit"],
        "checkpoint_sha256": checkpoint_hashes,
        "splits": records,
        "decisions": decisions,
        "decision": decision,
        "single_seed_only": True,
        "confirmation_policy": "evaluate_all_no_selection",
    }


# ---------------------------------------------------------------------------
# Passive per-slot router divergence probe
# ---------------------------------------------------------------------------

def route_pairwise_js_per_group(
    blocks: Sequence[torch.Tensor],
    partial_block: torch.Tensor,
    projs: Sequence[nn.Linear],
    norm: nn.Module,
    groups: int,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Passive pairwise JS of every route's depth-source distribution.

    Returns ``[B, T, G, P]`` and the ordered role pairs.  Only reads the
    existing route projections; it never feeds reconstructed probabilities into
    the model.
    """
    values = torch.stack(list(blocks) + [partial_block], dim=0)
    n, batch, tokens, width = values.shape
    if width % groups:
        raise ValueError("routing width must divide groups")
    if len(projs) < 2:
        return (values.new_zeros((batch, tokens, groups, 0), dtype=torch.float32), [])
    with torch.no_grad():
        keys = norm(values).reshape(n, batch, tokens, groups, width // groups)
        queries = [proj.weight.reshape(groups, -1) for proj in projs]
        logits = torch.stack(
            [torch.einsum("gk,nbtgk->nbtg", query, keys).float()
             for query in queries], dim=-1)  # [N,B,T,G,R]
        probs = logits.softmax(dim=0)
        pairs = [
            (i, j) for i in range(len(projs)) for j in range(i + 1, len(projs))
        ]
        columns = []
        for i, j in pairs:
            a = probs[..., i]
            b = probs[..., j]
            midpoint = 0.5 * (a + b)
            js = 0.5 * (
                torch.special.xlogy(a, a / midpoint)
                + torch.special.xlogy(b, b / midpoint)
            ).sum(dim=0)
            columns.append(js)
        stacked = torch.stack(columns, dim=-1)  # [B,T,G,P]
    return stacked, pairs


def _read_probe_indices(spec: dict) -> dict:
    probe = spec["probe_manifest"]
    path = ROOT / probe["path"]
    if sha256_file(path) != probe["sha256"]:
        raise ValueError("Experiment 20 probe manifest hash mismatch")
    indices = read_json(path)["indices"]
    for split in SPLITS:
        if (len(indices[split]) != 32 or len(set(indices[split])) != 32
                or min(indices[split]) < 0 or max(indices[split]) >= 512):
            raise ValueError("Experiment 20 probe indices changed")
    return indices


def evaluate_probe(
    *, checkpoint: Path, artifact: Path, root: Path, condition: str,
    device: torch.device, dtype: torch.dtype = torch.bfloat16,
    wandb_mode: str = "disabled", wandb_entity: str | None = None,
) -> dict:
    spec = load_protocol()
    if condition not in CONDITIONS:
        raise ValueError("unknown Experiment 20 probe condition")
    output = probe_path(root, condition)
    if output.exists():
        raise FileExistsError(f"probe already exists: {output}")
    evaluation_commit = require_descendant(spec["training_source_commit"])
    checkpoint_info = validate_training_checkpoint(checkpoint, condition, spec)
    payload, artifact_verification = verify_artifact(artifact, spec)
    artifact_hash = artifact_verification["frozen_sha256"]
    indices = _read_probe_indices(spec)
    verify_frozen_discovery(root, spec)

    model = Experiment20MHARForCausalLM.from_pretrained(
        str(checkpoint), dtype=dtype,
    ).to(device=device).eval()
    route_count = int(spec["conditions"][condition]["route_count"])
    pair_count = route_count * (route_count - 1) // 2
    totals = {split: torch.zeros((36, 8, pair_count), dtype=torch.float64)
              for split in SPLITS}
    counts = {split: torch.zeros(36, dtype=torch.int64) for split in SPLITS}
    pair_labels = model_probe_pairs(route_count)
    roles = ["global"] + [f"local-{index}" for index in range(1, route_count)]
    try:
        with torch.inference_mode():
            for split in SPLITS:
                for index in indices[split]:
                    tokens = payload[f"{split}_input_ids"][index:index + 1].to(device)
                    if split == "discovery" and index == indices[split][0]:
                        baseline_logits = model(input_ids=tokens, use_cache=False).logits

                    def collect(js, li, name=split):
                        totals[name][li] += js.double().sum(dim=(0, 1)).cpu()
                        counts[name][li] += js.shape[0] * js.shape[1]

                    for layer_index, layer in enumerate(model.model.layers):
                        layer._experiment20_probe_accumulator = (
                            lambda js, li=layer_index, nm=split: collect(js, li, nm))
                    try:
                        logits = model(input_ids=tokens, use_cache=False).logits
                    finally:
                        for layer in model.model.layers:
                            layer._experiment20_probe_accumulator = None
                    if split == "discovery" and index == indices[split][0]:
                        if not torch.equal(logits, baseline_logits):
                            raise RuntimeError("passive route probe changed logits")
    finally:
        del model

    if any((counts[split] != 32 * 1024).any() for split in SPLITS):
        raise RuntimeError("incomplete route probe token accounting")
    maps = {}
    for split in SPLITS:
        # [layer][group][pair] mean JS
        maps[split] = (
            totals[split] / counts[split][:, None, None]
        ).tolist()
    agreements = _probe_split_agreement(maps)
    local_pairs = [k for k, (i, j) in enumerate(pair_labels) if i >= 1 and j >= 1]
    global_pairs = [k for k, (i, j) in enumerate(pair_labels) if i == 0 or j == 0]

    def _mean_pairs(split, pair_indices):
        values = np.asarray(maps[split], dtype=np.float64)
        if not pair_indices:
            return None
        return float(values[:, :, pair_indices].mean())

    result = {
        "format_version": 1,
        "created_at": utc_now(),
        "experiment": spec["experiment"],
        "condition": condition,
        "route_count": route_count,
        "roles": roles,
        "role_pair_order": [list(pair) for pair in pair_labels],
        "checkpoint_sha256": checkpoint_info["checkpoint_sha256"],
        "artifact_sha256": artifact_hash,
        "artifact_verification": artifact_verification,
        "protocol_sha256": protocol_sha256(),
        "training_source_commit": spec["training_source_commit"],
        "evaluation_source_commit": evaluation_commit,
        "probe_manifest_sha256": spec["probe_manifest"]["sha256"],
        "js_unit": "nats",
        "token_count_per_layer": {split: counts[split].tolist() for split in SPLITS},
        "pair_js_mean": {
            split: np.asarray(maps[split], dtype=np.float64).mean(axis=(1, 2)).tolist()
            for split in SPLITS
        },
        "pair_js_map": maps,
        "local_slot_mean_js": {
            split: _mean_pairs(split, local_pairs) for split in SPLITS},
        "global_local_mean_js": {
            split: _mean_pairs(split, global_pairs) for split in SPLITS},
        "overall_mean_js": {
            split: float(np.asarray(maps[split], dtype=np.float64).mean())
            for split in SPLITS},
        "discovery_confirmation_cell_spearman": agreements,
    }
    run = None
    if wandb_mode != "disabled":
        import wandb
        run = wandb.init(
            project=spec["wandb"]["project"], entity=wandb_entity,
            group=spec["wandb"]["group"],
            name=f"mhar-exp20-{condition}-route-probe",
            job_type="passive-diagnostic", mode=wandb_mode,
            config={"checkpoint_sha256": checkpoint_info["checkpoint_sha256"],
                    "artifact_sha256": artifact_hash,
                    "probe_manifest_sha256": spec["probe_manifest"]["sha256"]},
        )
        result["wandb"] = {"run_id": run.id, "run_url": run.url}
    try:
        write_once(output, result)
        if run:
            run.log({
                "route_js/discovery_local_slot_mean": result["local_slot_mean_js"]["discovery"],
                "route_js/confirmation_local_slot_mean": result["local_slot_mean_js"]["confirmation"],
                "route_js/overall_mean": result["overall_mean_js"]["confirmation"],
            })
    finally:
        if run:
            run.finish()
    return result


def model_probe_pairs(route_count: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(route_count) for j in range(i + 1, route_count)]


def _probe_split_agreement(maps: dict) -> float:
    from src.experiments.experiment10_per_group_contribution import spearman
    return float(spearman(
        np.asarray(maps["discovery"], dtype=np.float64).ravel(),
        np.asarray(maps["confirmation"], dtype=np.float64).ravel(),
    ))
