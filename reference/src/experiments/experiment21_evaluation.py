"""Experiment 21/22 fixed evaluation, data lock, and paired analysis.

Both experiments share one fresh matched R2 control and one fixed evaluation
identity.  This module implements the shared workflow:

* build or recover the fixed evaluation data lock (a *new* identity with
  full-split tensor hashes, explicitly not a prefix-hash reconstruction);
* validate training-checkpoint identity;
* evaluate every condition, including the frozen ``request-off`` and
  ``contrast-off`` interventions, on discovery and confirmation;
* freeze a no-selection discovery manifest before confirmation;
* paired per-sequence token-weighted bootstrap contrasts.

Historical Experiment 19/20 per-sequence numbers are never primary controls.
"""

from __future__ import annotations

import math
import glob
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import torch

from src.experiments.experiment1_partition_compatibility import (
    atomic_write_json,
    load_fixed_eval_artifact,
    materialize_document_disjoint_sequences,
    save_fixed_eval_artifact,
    sha256_file,
    sha256_path,
    utc_now,
)
from src.experiments.experiment6_screening import evaluate_split
from src.experiments.experiment21_request_conditioned import (
    EXPERIMENT21_VARIANTS,
    Experiment21MHARForCausalLM,
    experiment21_parameter_report,
    set_request_off,
)
from src.experiments.experiment22_contrast_readout import (
    Experiment22MHARForCausalLM,
    experiment22_parameter_report,
    set_contrast_off,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_21 = ROOT / "configs/experiment21/protocol.json"
PROTOCOL_22 = ROOT / "configs/experiment22/protocol.json"
SPLITS = ("discovery", "confirmation")
HISTORICAL_ARTIFACT_SHA256 = (
    "29e545dd9399e9eaea6f5abf38f20ef76ba232466a01b6ddf13c3a6a287a3691")

EXPERIMENTS = {
    "experiment21": {
        "protocol": PROTOCOL_21,
        "training": {
            "exp21-r2-s42": ("experiment21", "r2"),
            "exp21-constant-request-s42": ("experiment21", "constant-request"),
            "exp21-request-local-s42": ("experiment21", "request-local"),
        },
        "interventions": {
            "exp21-request-off-s42": ("exp21-request-local-s42", "request-off"),
        },
        "probes": {
            "exp21-probe-r2": "exp21-r2-s42",
            "exp21-probe-constant-request": "exp21-constant-request-s42",
            "exp21-probe-request-local": "exp21-request-local-s42",
        },
        "primary": (
            ("Delta_R2", "exp21-request-local-s42", "exp21-r2-s42"),
            ("Delta_constant", "exp21-request-local-s42",
             "exp21-constant-request-s42"),
        ),
        "diagnostic": (
            ("request_off", "exp21-request-off-s42", "exp21-request-local-s42"),
        ),
        "timing": ("exp21-r2-s42", "exp21-constant-request-s42",
                   "exp21-request-local-s42"),
    },
    "experiment22": {
        "protocol": PROTOCOL_22,
        "training": {
            "exp21-r2-s42": ("experiment21", "r2"),
            "exp22-contrast-local-s42": ("experiment22", "contrast-local"),
        },
        "interventions": {
            "exp22-contrast-off-s42": ("exp22-contrast-local-s42", "contrast-off"),
        },
        "probes": {"exp22-probe-contrast-local": "exp22-contrast-local-s42"},
        "primary": (("Delta", "exp22-contrast-local-s42", "exp21-r2-s42"),),
        "diagnostic": (
            ("contrast_off", "exp22-contrast-off-s42",
             "exp22-contrast-local-s42"),
        ),
        "timing": ("exp21-r2-s42", "exp22-contrast-local-s42"),
    },
}


def read_json(path: Path) -> dict:
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_protocol(experiment: str, path: Path | None = None) -> dict:
    spec = read_json(path or EXPERIMENTS[experiment]["protocol"])
    if spec.get("seed") != 42 or spec.get("milestone") != 2000:
        raise ValueError(f"{experiment} seed or milestone changed")
    if spec.get("schedule_steps") != 20000:
        raise ValueError(f"{experiment} schedule changed")
    return spec


def protocol_sha256(experiment: str, path: Path | None = None) -> str:
    return sha256_file(path or EXPERIMENTS[experiment]["protocol"])


def require_clean_commit() -> str:
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if status.strip():
        raise RuntimeError("evaluation requires a clean worktree")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


# ---------------------------------------------------------------------------
# Fixed evaluation data lock
# ---------------------------------------------------------------------------

def build_data_lock(
    *, artifact: Path, manifest: Path, data_files: str, tokenizer: str,
    tokenizer_revision: str, seed: int = 20260824, shuffle_buffer: int = 10000,
    seq_len: int = 1024, discovery_sequences: int = 512,
    confirmation_sequences: int = 512, experiment: str = "experiment21",
    dataset_revision: str = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
) -> dict:
    """Materialize a new document-disjoint fixed artifact and hash its tensors.

    This records a *new* evaluation identity.  It is intentionally not the
    prefix-fingerprint reconstruction that Experiment 20 used, and it must be
    chosen once, before training or outcome inspection.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from src.experiments.experiment18_compute_audit import tensor_sha256

    if Path(artifact).exists() or Path(manifest).exists():
        raise FileExistsError("refusing to overwrite an existing data lock")
    if (seed, shuffle_buffer, seq_len, discovery_sequences,
            confirmation_sequences) != (20260824, 10000, 1024, 512, 512):
        raise ValueError("data-lock recipe differs from the frozen protocol")
    if (dataset_revision != "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
            or tokenizer != "Qwen/Qwen3-0.6B"
            or tokenizer_revision != "c1899de289a04d12100db370d81485cdf75e47ca"):
        raise ValueError("data-lock dataset/tokenizer revision differs from frozen protocol")
    shards = [Path(p).resolve() for p in sorted(glob.glob(data_files))]
    if not shards:
        raise FileNotFoundError("evaluation shard pattern matched no files")
    shard_records = [{"path": str(p), "bytes": p.stat().st_size,
                      "sha256": sha256_file(p)} for p in shards]
    source_commit = require_clean_commit()
    tokenizer_obj = AutoTokenizer.from_pretrained(
        tokenizer, revision=tokenizer_revision, use_fast=True)
    if tokenizer_obj.eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    dataset = load_dataset(
        "parquet", data_files=data_files, split="train", streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
    document_ids = []

    def identified_documents():
        for sample in dataset:
            text = sample.get("text") or sample.get("content") or sample.get("wikitext") or ""
            if text:
                document_ids.append(hashlib.sha256(text.encode("utf-8")).hexdigest())
                yield sample

    discovery, confirmation, document_counts = (
        materialize_document_disjoint_sequences(
            identified_documents(), tokenizer_obj, seq_len=seq_len,
            discovery_count=discovery_sequences,
            confirmation_count=confirmation_sequences))
    # The packer may fetch one unused document before stopping; exclude it.
    ids = {
        "discovery": document_ids[:document_counts["discovery"]],
        "confirmation": document_ids[document_counts["discovery"]:document_counts["total"]],
    }
    if set(ids["discovery"]) & set(ids["confirmation"]):
        raise ValueError("duplicate document content crosses evaluation splits")
    metadata = {
        "experiment": experiment,
        "data_files": data_files,
        "seed": seed,
        "shuffle_buffer": shuffle_buffer,
        "sequence_length": seq_len,
        "tokenizer": tokenizer,
        "tokenizer_revision": tokenizer_revision,
        "selection": "single shuffled document stream; document-disjoint splits",
        "source_commit": source_commit,
        "dataset_revision": dataset_revision,
        "shards": shard_records,
        "document_identity": "SHA-256 of UTF-8 source text",
        "document_ids": ids,
    }
    save_fixed_eval_artifact(Path(artifact), discovery, confirmation, metadata)
    lock = {
        "format_version": 2,
        "experiment": experiment,
        "created_at": utc_now(),
        "policy": "new-document-disjoint-evaluation-identity",
        "artifact_path": str(Path(artifact).resolve()),
        "artifact_sha256": sha256_file(Path(artifact)),
        "data_files": data_files,
        "dataset_revision": dataset_revision,
        "source_commit": source_commit,
        "shards": shard_records,
        "document_ids": ids,
        "document_identity": metadata["document_identity"],
        "tokenizer": tokenizer,
        "tokenizer_revision": tokenizer_revision,
        "seed": seed,
        "shuffle_buffer": shuffle_buffer,
        "sequence_length": seq_len,
        "documents_by_split": document_counts,
        "splits": {
            split: {
                "shape": list(tensor.shape),
                "dtype": "torch.int32",
                "valid_tokens_per_sequence": seq_len - 1,
                "tensor_sha256": tensor_sha256(tensor.to(torch.int64)),
                "valid_token_mask_sha256": tensor_sha256(
                    torch.ones(tensor.shape[0], seq_len - 1, dtype=torch.bool)),
            }
            for split, tensor in (("discovery", discovery),
                                  ("confirmation", confirmation))
        },
    }
    atomic_write_json(Path(manifest), lock)
    return lock


def verify_data_lock(*, artifact: Path, manifest: Path) -> dict:
    from src.experiments.experiment18_compute_audit import tensor_sha256

    lock = read_json(manifest)
    if lock.get("policy") != "new-document-disjoint-evaluation-identity":
        raise ValueError("data lock policy changed")
    if (lock.get("format_version") != 2 or not lock.get("shards")
            or lock.get("dataset_revision") != "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
            or len(lock.get("source_commit", "")) != 40
            or lock.get("seed") != 20260824 or lock.get("shuffle_buffer") != 10000
            or lock.get("tokenizer") != "Qwen/Qwen3-0.6B"
            or lock.get("tokenizer_revision") != "c1899de289a04d12100db370d81485cdf75e47ca"
            or lock.get("sequence_length") != 1024):
        raise ValueError("incomplete or changed data-lock provenance")
    ids = lock.get("document_ids", {})
    if (any(not ids.get(s) or len(ids[s]) != lock["documents_by_split"][s]
            or any(not isinstance(i, str) or len(i) != 64 for i in ids[s])
            for s in SPLITS) or set(ids["discovery"]) & set(ids["confirmation"])):
        raise ValueError("document disjointness evidence mismatch")
    for shard in lock["shards"]:
        if shard["bytes"] <= 0 or len(shard["sha256"]) != 64:
            raise ValueError("invalid shard identity")
    payload, observed = load_fixed_eval_artifact(Path(artifact))
    if observed != lock["artifact_sha256"]:
        raise ValueError("data-lock artifact hash mismatch")
    metadata = payload.get("metadata", {})
    for name in ("shards", "document_ids", "source_commit", "dataset_revision",
                 "tokenizer", "tokenizer_revision", "seed", "shuffle_buffer", "sequence_length"):
        if metadata.get(name) != lock.get(name):
            raise ValueError(f"data-lock {name} provenance mismatch")
    for split in SPLITS:
        tensor = payload[f"{split}_input_ids"]
        row = lock["splits"][split]
        if list(tensor.shape) != [512, 1024] or list(tensor.shape) != row["shape"]:
            raise ValueError(f"data-lock {split} shape mismatch")
        if tensor_sha256(tensor.to(torch.int64)) != row["tensor_sha256"]:
            raise ValueError(f"data-lock {split} tensor hash mismatch")
        mask = torch.ones(512, 1023, dtype=torch.bool)
        if (row.get("valid_tokens_per_sequence") != 1023
                or row.get("valid_token_mask_sha256") != tensor_sha256(mask)):
            raise ValueError(f"data-lock {split} mask mismatch")
    return lock


def resolve_artifact(*, artifact: Path, manifest: Path | None) -> tuple[dict, dict]:
    """Return the artifact payload and its identity record."""
    observed = sha256_file(Path(artifact))
    payload = load_fixed_eval_artifact(Path(artifact))[0]
    if observed == HISTORICAL_ARTIFACT_SHA256:
        identity = {
            "kind": "recovered_historical",
            "frozen_sha256": HISTORICAL_ARTIFACT_SHA256,
            "observed_sha256": observed,
        }
    else:
        if manifest is None:
            raise ValueError("a non-historical artifact requires a data lock")
        lock = verify_data_lock(artifact=artifact, manifest=manifest)
        identity = {
            "kind": "new_data_lock",
            "frozen_sha256": lock["artifact_sha256"],
            "observed_sha256": observed,
            "lock_manifest": str(Path(manifest).resolve()),
            "lock_manifest_sha256": sha256_file(manifest),
        }
    from src.experiments.experiment18_compute_audit import tensor_sha256
    identity["splits"] = {}
    for split in SPLITS:
        tensor = payload[f"{split}_input_ids"]
        if list(tensor.shape) != [512, 1024]:
            raise ValueError("fixed evaluation must have 512x1024 tokens per split")
        identity["splits"][split] = {
            "tensor_sha256": tensor_sha256(tensor.to(torch.int64)),
            "valid_token_mask_sha256": tensor_sha256(torch.ones(512, 1023, dtype=torch.bool)),
        }
    return payload, identity


# ---------------------------------------------------------------------------
# Checkpoint identity
# ---------------------------------------------------------------------------

def _accepted_hq8_data_files() -> dict:
    manifest = read_json(
        ROOT / "results/experiment8/provenance/step-2000/hq8/training_run_manifest.json")
    rows = manifest["run_identity"]["data_files"]["files"]
    return {Path(row["path"]).name: (row["bytes"], row["sha256"]) for row in rows}


def validate_checkpoint(checkpoint: Path, *, model: str, variant: str) -> dict:
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_dir() or checkpoint.name != "step-2000":
        raise ValueError("requires an atomic step-2000 checkpoint")
    manifest = read_json(checkpoint / "training_manifest.json")
    identity = manifest.get("run_identity", {})
    expected = {
        "mode": "full_mh", "attnres_heads": 8,
        "hidden_size": 1280, "num_layers": 36,
        "num_heads": 16, "num_kv_heads": 8,
        "intermediate_size": 5120, "seq_len": 1024,
        "steps": 20000, "global_batch_size": 32,
        "lr": 5e-4, "lr_min": 5e-5, "warmup": 1000, "seed": 42,
        "fused": True, "grad_ckpt": True,
        "dataset": "HuggingFaceFW/fineweb-edu", "dataset_name": "sample-10BT",
        "dataset_revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "tokenizer": "Qwen/Qwen3-0.6B",
        "tokenizer_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "optimizer": "AdamW", "optimizer_betas": [0.9, 0.95],
        "optimizer_eps": 1e-8, "weight_decay": 0.1,
        "per_gpu_batch_size": 4, "grad_accum": 8, "world_size": 1,
        "experiment8_hybrid_q_groups": 8,
        "experiment8_local_head_position": "even",
        "experiment8_global_head_position": "odd",
    }
    if model == "experiment21":
        expected["experiment21_variant"] = variant
        expected["experiment21_request"] = True
        expected["experiment21_request_rank"] = 4
    elif model == "experiment22":
        expected["experiment22_variant"] = variant
        expected["experiment22_contrast"] = True
    else:
        raise ValueError("unknown experiment model")
    mismatches = {
        key: {"expected": value, "observed": identity.get(key)}
        for key, value in expected.items() if identity.get(key) != value
    }
    if (manifest.get("global_step") != 2000
            or manifest.get("chunks_consumed") != 64_000 or mismatches):
        raise ValueError(f"training identity mismatch: {mismatches}")
    recorded = identity.get("data_files", {}).get("files", [])
    observed_files = {
        Path(row["path"]).name: (row["bytes"], row["sha256"]) for row in recorded}
    if observed_files != _accepted_hq8_data_files():
        raise ValueError("training data bytes differ from accepted HQ8")
    if not isinstance(identity.get("source_commit"), str) or len(
        identity["source_commit"]) != 40:
        raise ValueError("checkpoint source commit is missing")
    state = torch.load(
        checkpoint / "training_state.pt", map_location="cpu",
        weights_only=False, mmap=True)
    required = ("global_step", "chunks_consumed", "optimizer", "scheduler",
                "cpu_rng_state", "cuda_rng_state_all", "run_identity")
    if (any(key not in state for key in required)
            or state.get("global_step") != 2000
            or state.get("chunks_consumed") != 64_000
            or state.get("run_identity") != identity
            or not state["optimizer"].get("state")
            or not state["scheduler"]):
        raise ValueError("incomplete optimizer/RNG/data-position state")
    del state
    return {"checkpoint_sha256": sha256_path(checkpoint), "manifest": manifest}


def _condition_loader(model: str, variant: str):
    if model == "experiment21":
        return Experiment21MHARForCausalLM, experiment21_parameter_report
    if model == "experiment22":
        return Experiment22MHARForCausalLM, experiment22_parameter_report
    raise ValueError("unknown experiment model")


def evaluate_condition(
    *, experiment: str, condition: str, checkpoint: Path, artifact: Path,
    manifest: Path | None, root: Path, split: str, device: torch.device,
    dtype: torch.dtype, batch_size: int, wandb_mode: str = "disabled",
    wandb_entity: str | None = None,
    shared_root: Path | None = None,
) -> dict:
    spec = load_protocol(experiment)
    if experiment == "experiment22" and condition == "exp21-r2-s42":
        raise ValueError("reuse Experiment 21 R2 evaluation via --shared-root; do not relabel it")
    table = EXPERIMENTS[experiment]
    if condition in table["training"]:
        model, variant = table["training"][condition]
        intervention = None
        parent = condition
    elif condition in table["interventions"]:
        parent, intervention = table["interventions"][condition]
        model, variant = table["training"][parent]
    else:
        raise ValueError(f"unknown {experiment} condition: {condition}")
    if split not in SPLITS:
        raise ValueError("unknown split")
    output = Path(root) / "evaluation" / condition / split / "result.json"
    if output.exists():
        raise FileExistsError(f"evaluation already exists: {output}")
    evaluation_commit = require_clean_commit()
    info = validate_checkpoint(checkpoint, model=model, variant=variant)
    payload, artifact_identity = resolve_artifact(artifact=artifact, manifest=manifest)
    if split == "confirmation":
        verify_frozen_discovery(root, experiment, spec, shared_root=shared_root)
        discovery = load_condition(root, experiment, condition, "discovery", spec,
                                   shared_root=shared_root)
        if (discovery["checkpoint_sha256"] != info["checkpoint_sha256"]
                or artifact_key(discovery["artifact_identity"]) != artifact_key(artifact_identity)):
            raise ValueError("confirmation checkpoint/artifact differs from frozen discovery")

    cls, report = _condition_loader(model, variant)
    net = cls.from_pretrained(str(checkpoint), dtype=dtype).to(device=device).eval()
    if intervention == "request-off":
        set_request_off(net, True)
    elif intervention == "contrast-off":
        set_contrast_off(net, True)
    elif intervention is not None:
        raise ValueError(f"unknown intervention: {intervention}")
    parameters = report(net)
    run = None
    if wandb_mode != "disabled":
        import wandb
        run = wandb.init(
            project=spec["wandb"]["project"], entity=wandb_entity,
            group=spec["wandb"]["group"],
            name=f"mhar-{experiment}-{condition}-{split}",
            job_type="fixed-evaluation", mode=wandb_mode,
            tags=[experiment, condition, split, "seed-42"],
            config={"checkpoint_sha256": info["checkpoint_sha256"],
                    "artifact_sha256": artifact_identity["observed_sha256"],
                    "protocol_sha256": protocol_sha256(experiment)},
        )
    try:
        metrics = evaluate_split(
            net, payload[f"{split}_input_ids"],
            batch_size=batch_size, device=device)
        length = 1023
        metrics["sequence_valid_tokens"] = [length] * len(metrics["sequence_losses"])
        metrics["sequence_total_losses"] = [
            float(value) * length for value in metrics["sequence_losses"]]
        result = {
            "format_version": 1,
            "created_at": utc_now(),
            "experiment": spec["experiment"],
            "condition": condition,
            "parent_condition": parent,
            "intervention": intervention,
            "split": split,
            "seed": spec["seed"],
            "milestone": spec["milestone"],
            "artifact_identity": artifact_identity,
            "checkpoint_sha256": info["checkpoint_sha256"],
            "protocol_sha256": protocol_sha256(experiment),
            "evaluation_source_commit": evaluation_commit,
            "parameters": parameters,
            "metrics": metrics,
            "wandb": ({"run_id": run.id, "run_url": run.url} if run else None),
        }
        if output.exists():
            raise FileExistsError(output)
        atomic_write_json(output, result)
        if run:
            run.log({f"{split}/loss": metrics["loss"],
                     f"{split}/tokens_per_second": metrics["tokens_per_second"]})
        return result
    finally:
        if run:
            run.finish()
        del net


# ---------------------------------------------------------------------------
# Discovery freeze
# ---------------------------------------------------------------------------

def result_path(root, experiment, condition, split, shared_root=None):
    if experiment == "experiment22" and condition == "exp21-r2-s42":
        if shared_root is None:
            raise ValueError("Experiment 22 requires --shared-root for accepted fresh R2 results")
        root = shared_root
    return Path(root) / "evaluation" / condition / split / "result.json"


def artifact_key(identity):
    """Path-independent complete ordered-token/mask identity."""
    observed = identity.get("observed_sha256")
    if (not isinstance(observed, str) or len(observed) != 64
            or observed != identity.get("frozen_sha256")
            or set(identity.get("splits", {})) != set(SPLITS)):
        raise ValueError("incomplete artifact identity")
    hashes = []
    for split in SPLITS:
        row = identity["splits"][split]
        for name in ("tensor_sha256", "valid_token_mask_sha256"):
            value = row.get(name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError("incomplete ordered tensor/mask identity")
            hashes.append(value)
    return observed, *hashes


def validate_pairing(records, experiment):
    keys = {artifact_key(row["artifact_identity"]) for row in records.values()}
    if len(keys) != 1:
        raise ValueError("paired conditions have different artifact/order/mask identities")
    for condition, (parent, _) in EXPERIMENTS[experiment]["interventions"].items():
        if (condition in records and records[condition]["checkpoint_sha256"]
                != records[parent]["checkpoint_sha256"]):
            raise ValueError("intervention checkpoint differs from its parent")


def discovery_manifest(root: Path, experiment: str, spec: dict, shared_root=None) -> dict:
    table = EXPERIMENTS[experiment]
    paths = {
        condition: result_path(root, experiment, condition, "discovery", shared_root)
        for condition in (*table["training"], *table["interventions"])
    }
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("all discovery conditions must finish before confirmation")
    records = {c: load_condition(root, experiment, c, "discovery", spec,
                                shared_root=shared_root) for c in paths}
    validate_pairing(records, experiment)
    return {
        "format_version": 1,
        "experiment": spec["experiment"],
        "protocol_sha256": protocol_sha256(experiment),
        "discovery_result_sha256": {
            condition: sha256_file(path) for condition, path in paths.items()},
        "primary_contrasts": [list(row) for row in table["primary"]],
        "candidate_selection": "none",
        "bootstrap": spec["evaluation"]["bootstrap"],
        "artifact_identity": list(artifact_key(next(iter(records.values()))["artifact_identity"])),
        "checkpoint_sha256": {c: r["checkpoint_sha256"] for c, r in records.items()},
        "shared_control": ({"source_experiment": "experiment21",
                            "protocol_sha256": protocol_sha256("experiment21"),
                            "discovery_result_sha256": sha256_file(paths["exp21-r2-s42"])}
                           if experiment == "experiment22" else None),
    }


def freeze_discovery(root: Path, experiment: str, spec: dict | None = None,
                     *, shared_root=None) -> Path:
    spec = spec or load_protocol(experiment)
    require_clean_commit()
    destination = Path(root) / "selections" / "no_selection_manifest.json"
    if destination.exists():
        raise FileExistsError(destination)
    atomic_write_json(destination, discovery_manifest(root, experiment, spec, shared_root))
    return destination


def verify_frozen_discovery(root: Path, experiment: str, spec: dict | None = None,
                            *, shared_root=None) -> dict:
    spec = spec or load_protocol(experiment)
    path = Path(root) / "selections" / "no_selection_manifest.json"
    if not path.is_file():
        raise ValueError("confirmation is locked until discovery is frozen")
    recorded = read_json(path)
    if recorded != discovery_manifest(root, experiment, spec, shared_root):
        raise ValueError("frozen discovery manifest changed")
    return recorded


# ---------------------------------------------------------------------------
# Passive diagnostics probe
# ---------------------------------------------------------------------------

def _softmax_js(a, b, dim=0):
    a, b = a.float(), b.float()
    midpoint = 0.5 * (a + b)
    denominator = midpoint.clamp_min(torch.finfo(torch.float32).tiny)
    return 0.5 * (
        torch.special.xlogy(a, a / denominator)
        + torch.special.xlogy(b, b / denominator)
    ).sum(dim=dim)


def _layer_probe(layer, blocks, partial_block, experiment, variant):
    """Recompute route diagnostics for one layer without changing logits."""
    with torch.no_grad():
        stacked = torch.stack(list(blocks) + [partial_block], dim=0)
        n, batch, tokens, width = stacked.shape
        heads = layer.attnres_num_heads
        chunk = width // heads
        keys = layer.attn_res_norm(stacked).view(n, batch, tokens, heads, chunk)
        values = stacked.view(n, batch, tokens, heads, chunk)
        g_query = layer.attn_res_proj.weight.view(heads, chunk)
        l_query = layer.local_attn_res_proj.weight.view(heads, chunk)
        score_g = torch.einsum("hk,nbthk->nbth", g_query, keys)
        score_l = torch.einsum("hk,nbthk->nbth", l_query, keys)
        out = {name: torch.zeros(heads, device=keys.device) for name in PROBE_MAPS}
        out["js"] = None
        if experiment == "experiment21":
            request_u = getattr(layer, "request_u", None)
            request_v = getattr(layer, "request_v", None)
            if variant != "r2" and request_u is not None:
                projected = torch.einsum("nbthk,hkr->nbthr", keys, request_v)
                if variant == "constant-request":
                    fixed = keys.new_zeros(chunk); fixed[0] = 1.0
                    request = torch.einsum("k,hkr->hr", fixed, request_u)
                    delta = torch.einsum("hr,nbthr->nbth", request, projected)
                else:
                    request = torch.einsum("bthk,hkr->bthr", keys[-1], request_u)
                    delta = torch.einsum("bthr,nbthr->nbth", request, projected)
                delta = delta / math.sqrt(int(layer._experiment21_rank))
                score_l = score_l + delta
                delta32 = delta.float()
                out["added_rms"] = delta32.pow(2).mean(dim=(0, 1, 2)).sqrt()
                centred = delta32 - delta32.mean(dim=0, keepdim=True)
                out["added_centered_std"] = centred.pow(2).mean(dim=(0, 1, 2)).sqrt()
        a_g = score_g.softmax(dim=0)
        a_l = score_l.softmax(dim=0)
        out["js"] = _softmax_js(a_g, a_l, dim=0).mean(dim=(0, 1))  # [G]
        out["entropy_g"] = -torch.special.xlogy(a_g.float(), a_g.float()).sum(dim=0).mean(dim=(0, 1))
        out["entropy_l"] = -torch.special.xlogy(a_l.float(), a_l.float()).sum(dim=0).mean(dim=(0, 1))
        c_g = torch.einsum("nbth,nbthk->bthk", a_g, values).reshape(batch, tokens, width)
        c_l = torch.einsum("nbth,nbthk->bthk", a_l, values).reshape(batch, tokens, width)
        readout = c_l + (c_l - c_g) if experiment == "experiment22" else c_l

        def grouped(value):
            return value.float().reshape(batch, tokens, heads, chunk)

        def mean_norm(value):
            return grouped(value).norm(dim=-1).mean(dim=(0, 1))

        out["contrast_norm"] = mean_norm(c_l - c_g)
        out["local_norm"] = mean_norm(c_l)
        out["readout_norm"] = mean_norm(readout)
        out["disagreement_local_ratio"] = out["contrast_norm"] / out["local_norm"].clamp_min(1e-12)
        out["readout_local_cosine"] = torch.nn.functional.cosine_similarity(
            grouped(readout), grouped(c_l), dim=-1).mean(dim=(0, 1))
        global_input = layer.input_layernorm(layer._apply_gate(partial_block, c_g, "attn"))
        local_input = layer.input_layernorm(layer._apply_gate(partial_block, readout, "attn"))
        plain_input = layer.input_layernorm(layer._apply_gate(partial_block, c_l, "attn"))
        qproj = layer.self_attn.q_proj
        shape = (batch, tokens, heads, 2, layer.self_attn.head_dim)
        raw_q = qproj.forward_dual(global_input, local_input).reshape(shape)
        plain_q = qproj.forward_dual(global_input, plain_input).reshape(shape)
        q = layer.self_attn.q_norm(raw_q).float()
        plain_q = layer.self_attn.q_norm(plain_q).float()
        local, global_q = q[..., 0, :], q[..., 1, :]
        out["query_local_global_angle"] = torch.nn.functional.cosine_similarity(
            local, global_q, dim=-1).clamp(-1, 1).acos().mean(dim=(0, 1))
        out["query_contrast_angle"] = torch.nn.functional.cosine_similarity(
            local, plain_q[..., 0, :], dim=-1).clamp(-1, 1).acos().mean(dim=(0, 1))
        out["near_zero_local_query_fraction"] = (
            raw_q[..., 0, :].float().norm(dim=-1) < 1e-8).float().mean(dim=(0, 1))
        if experiment == "experiment22":
            effective = 2 * a_l.float() - a_g.float()
            out["negative_fraction"] = (effective < 0).float().mean(dim=(0, 1, 2))
            out["negative_mass"] = (-effective).clamp_min(0).sum(dim=0).mean(dim=(0, 1))
        return out


PROBE_MAPS = ("entropy_g", "entropy_l", "added_rms", "added_centered_std",
                 "contrast_norm", "local_norm", "readout_norm",
                 "negative_fraction", "negative_mass", "disagreement_local_ratio",
                 "readout_local_cosine", "query_local_global_angle",
                 "query_contrast_angle", "near_zero_local_query_fraction")


def evaluate_probe(
    *, experiment: str, condition: str, checkpoint: Path, artifact: Path,
    manifest: Path | None, root: Path, device: torch.device,
    dtype: torch.dtype = torch.bfloat16, sequences: int = 32,
    wandb_mode: str = "disabled", wandb_entity: str | None = None,
    shared_root: Path | None = None,
) -> dict:
    spec = load_protocol(experiment)
    if sequences != spec["parity_gates"]["probe_first_sequences_per_split"]:
        raise ValueError("probe sequence count differs from frozen protocol")
    table = EXPERIMENTS[experiment]
    if condition not in table["probes"]:
        raise ValueError(f"unknown probe condition: {condition}")
    parent = table["probes"][condition]
    model_name, variant = table["training"][parent]
    output = Path(root) / "probe" / condition / "result.json"
    if output.exists():
        raise FileExistsError(output)
    evaluation_source = require_clean_commit()
    verify_frozen_discovery(root, experiment, spec, shared_root=shared_root)
    info = validate_checkpoint(checkpoint, model=model_name, variant=variant)
    payload, artifact_identity = resolve_artifact(artifact=artifact, manifest=manifest)
    native = load_condition(root, experiment, parent, "discovery", spec,
                            shared_root=shared_root)
    if (info["checkpoint_sha256"] != native["checkpoint_sha256"]
            or artifact_key(artifact_identity) != artifact_key(native["artifact_identity"])):
        raise ValueError("probe checkpoint/artifact differs from accepted evaluation")
    reference = None
    if experiment == "experiment22":
        reference_path = Path(shared_root) / "probe/exp21-probe-r2/result.json"
        reference = read_json(reference_path)
        r2 = load_condition(root, experiment, "exp21-r2-s42", "discovery", spec,
                            shared_root=shared_root)
        if (reference.get("condition") != "exp21-probe-r2"
                or reference.get("protocol_sha256") != protocol_sha256("experiment21")
                or reference.get("checkpoint_sha256") != r2["checkpoint_sha256"]
                or reference.get("probe_sequences_per_split") != sequences
                or reference.get("logits_bit_identical") is not True
                or artifact_key(reference["artifact_identity"]) != artifact_key(artifact_identity)):
            raise ValueError("shared R2 probe identity mismatch")
    cls, _report = _condition_loader(model_name, variant)
    net = cls.from_pretrained(str(checkpoint), dtype=dtype).to(device=device).eval()
    layers = len(net.model.layers)
    if reference:
        for split in SPLITS:
            for values in (reference["splits"][split]["js_map"],
                           *reference["splits"][split]["maps"].values()):
                array = np.asarray(values, dtype=float)
                if array.shape != (layers, 8) or not np.isfinite(array).all():
                    raise ValueError("shared R2 probe maps are incomplete")
            if set(reference["splits"][split]["maps"]) != set(PROBE_MAPS):
                raise ValueError("shared R2 probe metric set mismatch")
    totals = {split: {name: torch.zeros(layers, 8, dtype=torch.float64) for name in PROBE_MAPS}
              for split in SPLITS}
    js_totals = {split: [[0.0] * 8 for _ in range(layers)] for split in SPLITS}
    parity_sequences = 0
    try:
        with torch.inference_mode():
            for split in SPLITS:
                for index in range(sequences):
                    tokens = payload[f"{split}_input_ids"][index:index + 1].to(device)
                    ordinary_logits = net(input_ids=tokens, use_cache=False).logits
                    captured = {}

                    def make(layer_index):
                        def accumulate(blocks, partial_block, layer):
                            captured[layer_index] = _layer_probe(
                                layer, blocks, partial_block, experiment, variant)
                        return accumulate

                    for layer_index, layer in enumerate(net.model.layers):
                        if model_name == "experiment21":
                            layer._experiment21_probe_accumulator = make(layer_index)
                        else:
                            layer._experiment22_probe_accumulator = make(layer_index)
                    try:
                        observed_logits = net(input_ids=tokens, use_cache=False).logits
                    finally:
                        for layer in net.model.layers:
                            layer._experiment21_probe_accumulator = None
                            layer._experiment22_probe_accumulator = None
                    if not torch.equal(ordinary_logits, observed_logits):
                        raise ValueError("passive probe changed model logits")
                    if set(captured) != set(range(layers)):
                        raise ValueError("probe did not observe every layer")
                    parity_sequences += 1
                    del ordinary_logits, observed_logits
                    for layer_index, out in captured.items():
                        for name in PROBE_MAPS:
                            value = out[name].detach().cpu().double()
                            if value.shape != (8,) or not torch.isfinite(value).all():
                                raise ValueError(f"invalid probe map {name}")
                            totals[split][name][layer_index] += value
                        if out["js"] is not None:
                            row = js_totals[split][layer_index]
                            for group, value in enumerate(out["js"].tolist()):
                                row[group] += value
    finally:
        del net
    per_split = {}
    for split in SPLITS:
        sequences_in_split = sequences
        per_split[split] = {
            "js_map": [[value / sequences_in_split for value in row]
                       for row in js_totals[split]],
            **{name: (totals[split][name].mean(dim=1) / sequences_in_split).tolist()
               for name in PROBE_MAPS},
            "maps": {name: (totals[split][name] / sequences_in_split).tolist()
                     for name in PROBE_MAPS},
        }
    run = None
    if wandb_mode != "disabled":
        import wandb
        run = wandb.init(
            project=spec["wandb"]["project"], entity=wandb_entity,
            group=spec["wandb"]["group"], name=f"mhar-{experiment}-{condition}",
            job_type="passive-diagnostic", mode=wandb_mode,
            config={"checkpoint_sha256": info["checkpoint_sha256"]})
    result = {
        "format_version": 1,
        "created_at": utc_now(),
        "experiment": spec["experiment"],
        "condition": condition,
        "parent_condition": parent,
        "checkpoint_sha256": info["checkpoint_sha256"],
        "artifact_identity": artifact_identity,
        "protocol_sha256": protocol_sha256(experiment),
        "probe_sequences_per_split": sequences,
        "evaluation_source_commit": evaluation_source,
        "logits_bit_identical": True,
        "parity_sequences_checked": parity_sequences,
        "js_unit": "nats",
        "splits": per_split,
        "reference_r2": ({"result_sha256": sha256_file(reference_path),
                           "checkpoint_sha256": reference["checkpoint_sha256"],
                           "protocol_sha256": reference["protocol_sha256"],
                           "splits": reference["splits"]} if reference else None),
        "wandb": ({"run_id": run.id, "run_url": run.url} if run else None),
    }
    try:
        atomic_write_json(output, result)
    finally:
        if run:
            run.finish()
    return result


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def paired_bootstrap(a: dict, b: dict, *, samples: int, seed: int) -> dict:
    left = np.asarray(a["sequence_losses"], dtype=np.float64)
    right = np.asarray(b["sequence_losses"], dtype=np.float64)
    left_n = np.asarray(a["sequence_valid_tokens"], dtype=np.float64)
    right_n = np.asarray(b["sequence_valid_tokens"], dtype=np.float64)
    if (left.shape != right.shape or not np.array_equal(left_n, right_n)
            or (left_n <= 0).any() or not np.isfinite(left).all()
            or not np.isfinite(right).all()):
        raise ValueError("paired per-sequence Losses or token counts mismatch")
    delta = left - right
    point = float(np.dot(delta, left_n) / left_n.sum())
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for offset in range(0, samples, 512):
        width = min(512, samples - offset)
        index = rng.integers(0, len(delta), size=(width, len(delta)))
        weights = left_n[index]
        draws[offset:offset + width] = (
            (delta[index] * weights).sum(axis=1) / weights.sum(axis=1))
    return {
        "delta_loss": point,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def load_condition(root: Path, experiment: str, condition: str, split: str,
                   spec: dict, *, shared_root=None) -> dict:
    path = result_path(root, experiment, condition, split, shared_root)
    result = read_json(path)
    metrics = result["metrics"]
    count = 512
    origin = "experiment21" if condition == "exp21-r2-s42" else experiment
    origin_spec = spec if origin == experiment else load_protocol(origin)
    intervention = EXPERIMENTS[experiment]["interventions"].get(condition)
    parent = intervention[0] if intervention else condition
    losses = np.asarray(metrics["sequence_losses"], dtype=float)
    counts = np.asarray(metrics.get("sequence_valid_tokens", []), dtype=float)
    if (result.get("protocol_sha256") != protocol_sha256(origin)
            or result.get("experiment") != origin_spec["experiment"]
            or result.get("condition") != condition or result.get("split") != split
            or result.get("parent_condition") != parent
            or result.get("intervention") != (intervention[1] if intervention else None)
            or result.get("seed") != spec["seed"]
            or result.get("milestone") != spec["milestone"]
            or len(result.get("checkpoint_sha256", "")) != 64
            or len(result.get("evaluation_source_commit", "")) != 40
            or losses.shape != (count,) or not np.isfinite(losses).all()
            or counts.shape != (count,) or not np.all(counts == 1023)
            or metrics["valid_tokens"] != count * 1023
            # Float32 batch sums versus sequence means have different reduction
            # rounding. This checks serialization consistency, not model parity.
            or not math.isclose(float(metrics["loss"]), float(losses.mean()), abs_tol=1e-6)):
        raise ValueError(f"{condition}/{split} result mismatch")
    artifact_key(result["artifact_identity"])
    return result


def analyze(root: Path, experiment: str, *, require_confirmation: bool = True,
            shared_root=None) -> dict:
    spec = load_protocol(experiment)
    table = EXPERIMENTS[experiment]
    splits = SPLITS if require_confirmation else ("discovery",)
    records = {}
    checkpoint_hashes = {}
    for split in splits:
        conditions = {
            condition: load_condition(root, experiment, condition, split, spec, shared_root=shared_root)
            for condition in table["training"]
        }
        conditions.update({
            condition: load_condition(root, experiment, condition, split, spec, shared_root=shared_root)
            for condition in table["interventions"]
        })
        validate_pairing(conditions, experiment)
        for condition, result in conditions.items():
            previous = checkpoint_hashes.setdefault(condition, result["checkpoint_sha256"])
            if previous != result["checkpoint_sha256"]:
                raise ValueError("checkpoint changed between discovery and confirmation")
            if split == "confirmation":
                if artifact_key(result["artifact_identity"]) != artifact_key(
                        load_condition(root, experiment, condition, "discovery", spec,
                                       shared_root=shared_root)["artifact_identity"]):
                    raise ValueError("artifact changed between discovery and confirmation")
        metrics = {
            condition: {
                "sequence_losses": result["metrics"]["sequence_losses"],
                "sequence_valid_tokens": result["metrics"]["sequence_valid_tokens"],
                "loss": result["metrics"]["loss"],
            }
            for condition, result in conditions.items()
        }
        contrasts = {}
        for cid, left, right in table["primary"]:
            contrasts[cid] = paired_bootstrap(
                metrics[left], metrics[right],
                samples=spec["evaluation"]["bootstrap"]["samples"],
                seed=spec["evaluation"]["bootstrap"]["seed"])
        for cid, left, right in table["diagnostic"]:
            contrasts[cid] = paired_bootstrap(
                metrics[left], metrics[right],
                samples=spec["evaluation"]["bootstrap"]["samples"],
                seed=spec["evaluation"]["bootstrap"]["seed"] + 1)
        records[split] = {
            "loss": {name: metrics[name]["loss"] for name in metrics},
            "contrasts": contrasts,
        }
    decision = None
    if require_confirmation:
        verify_frozen_discovery(root, experiment, spec, shared_root=shared_root)
        final = records["confirmation"]["contrasts"]
        target = spec["evaluation"]["practical_target_nats_per_token"]
        primaries = [final[cid] for cid, _, _ in table["primary"]]
        confirmed = all(
            row["delta_loss"] <= target and row["ci95_high"] < 0
            for row in primaries)
        decision = ("confirmed_single_seed_advancement" if confirmed
                    else "no_confirmed_advancement")
    return {
        "format_version": 1,
        "created_at": utc_now(),
        "experiment": spec["experiment"],
        "protocol_sha256": protocol_sha256(experiment),
        "checkpoint_sha256": checkpoint_hashes,
        "splits": records,
        "decision": decision,
        "single_seed_only": True,
        "confirmation_policy": "evaluate_all_no_selection",
        "shared_control_provenance": ({
            split: {"source_experiment": "experiment21",
                    "protocol_sha256": protocol_sha256("experiment21"),
                    "result_sha256": sha256_file(result_path(
                        root, experiment, "exp21-r2-s42", split, shared_root))}
            for split in splits} if experiment == "experiment22" else None),
    }
