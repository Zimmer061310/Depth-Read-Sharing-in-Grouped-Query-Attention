"""
Train Qwen3 / Qwen3-AttnRes from scratch on FineWeb-Edu.

Usage:
    # Baseline (no AttnRes)
    torchrun --nproc_per_node=8 -m src.training.train_scratch --mode baseline

    # Block AttnRes
    torchrun --nproc_per_node=8 -m src.training.train_scratch --mode block

    # Full AttnRes
    torchrun --nproc_per_node=8 -m src.training.train_scratch --mode full
"""

import argparse
import contextlib
import glob
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.attention_residuals.modeling_qwen3_attnres import (
    Qwen3AttnResConfig,
    Qwen3AttnResForCausalLM,
    enable_compile as enable_attnres_compile,
)
from transformers import AutoTokenizer
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM


@contextlib.contextmanager
def optional_checkpoint_lock():
    """Serialize large atomic saves when an experiment shares a filesystem.

    This is an operational safeguard only. It is opt-in through an environment
    variable so legacy training behavior and scientific identities are unchanged.
    """

    path = os.environ.get("MHAR_CHECKPOINT_LOCK")
    if not path:
        yield
        return
    import fcntl

    lock_path = Path(path).resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="baseline", choices=["baseline", "moe", "block", "block_v", "block_mh", "full", "full_mh", "full_hw", "hyper_connection", "full_split", "full_split_shared", "full_split_shared_v", "full_v", "depth_attn", "full_additive", "delta", "delta_sublayer", "delta_centered", "delta_centered_block", "delta_centered_reset", "delta_avg_block", "delta_replace", "delta_replace_block", "delta_block", "delta_block_v", "delta_v", "first_layer", "pre_gated"],
                   help="baseline, block, block_v, full, full_mh, full_hw, full_split, full_split_shared, full_v, delta, delta_block, delta_block_v, delta_v, first_layer, pre_gated")
    p.add_argument("--hyper_n", type=int, default=4,
                   help="Expansion rate (number of parallel streams) for hyper_connection mode")
    p.add_argument("--moe_experts", type=int, default=8,
                   help="moe mode: number of experts")
    p.add_argument("--moe_topk", type=int, default=2,
                   help="moe mode: experts per token")
    p.add_argument("--moe_ff", type=int, default=768,
                   help="moe mode: per-expert intermediate size (topk*moe_ff ~ dense ff for iso-activated-FLOPs)")
    p.add_argument("--attnres_heads", type=int, default=8,
                   help="Routing heads for full_mh, and for full_hw's MLP/final routing (hidden_size must be divisible)")
    p.add_argument(
        "--qkv_groups", type=int, choices=[4, 8], default=None,
        help="Experiment 6 only: restrict Q/K/V inputs to 4 or 8 contiguous groups; "
             "valid only for baseline or matched full_mh",
    )
    p.add_argument(
        "--experiment6_variant",
        choices=["b", "m4", "c4", "g4", "m8", "c8", "g8"],
        default=None,
        help="Frozen Experiment 6 architecture identity; omitted by all legacy runs",
    )
    p.add_argument(
        "--local_q_groups", type=int, choices=[4, 8], default=None,
        help="Experiment 7 only: make Q block diagonal while K/V remain dense",
    )
    p.add_argument(
        "--experiment7_variant", choices=["lq4", "lq8", "blq4", "blq8"],
        default=None,
        help="Frozen Experiment 7 architecture identity; omitted by all legacy runs",
    )
    p.add_argument(
        "--hybrid_q_groups", type=int, choices=[8], default=None,
        help="Experiment 8 only: one local-even and one global-odd Q per GQA group",
    )
    p.add_argument(
        "--experiment8_variant", choices=["hq8", "bhq8"], default=None,
        help="Frozen Experiment 8 architecture identity; omitted by all legacy runs",
    )
    p.add_argument(
        "--experiment19_variant", choices=["rs-hq8"], default=None,
        help="Experiment 19 opt-in independent local-Q MHAR depth router",
    )
    p.add_argument(
        "--experiment20_route_count", type=int, choices=[3, 4, 8], default=None,
        help="Experiment 20 total routes: one global plus R-1 local slots",
    )
    p.add_argument(
        "--experiment20_router_backend",
        choices=["shared-eager", "role-axis-eager", "tiled-r2-triton"],
        default="role-axis-eager",
        help="Experiment 20 routing backend; use tiled-r2-triton on CUDA",
    )
    p.add_argument(
        "--experiment21_variant",
        choices=["r2", "constant-request", "request-local"], default=None,
        help="Experiment 21 request-conditioned local route variant",
    )
    p.add_argument(
        "--experiment22_variant",
        choices=["contrast-local"], default=None,
        help="Experiment 22 stabilized local-global contrast readout",
    )
    p.add_argument("--experiment25_variant", choices=["abbc-gl64-l16"], default=None)
    p.add_argument("--experiment25_router_backend", choices=["canonical", "tiled-r2-exact"], default="tiled-r2-exact")
    p.add_argument("--experiment25_execution_lock", default=None)
    p.add_argument("--experiment25_budget_lock", default=None)
    p.add_argument("--experiment25_budget_heartbeat", default=None)
    p.add_argument("--experiment24_variant", choices=["global-aabc", "global-abbc"], default=None)
    p.add_argument("--experiment24_router_backend", choices=["canonical", "tiled-r2-exact"],
                   default="tiled-r2-exact")
    p.add_argument("--experiment23_variant", choices=[
        "global-aaaa", "global-abaa", "global-aaba", "global-aaab", "global-aabb",
        "global-abba", "r2-gl-abaa", "hq8-gl-aaaa"], default=None)
    p.add_argument("--experiment23_router_backend", choices=["shared-eager", "r2-triton"],
                   default="shared-eager")
    p.add_argument("--experiment23_projection_backend", choices=["canonical", "packed"],
                   default="packed")
    p.add_argument(
        "--experiment11_run",
        choices=[
            "s2q8-l000", "s2q8-l010", "s2q8-l025", "s2q8-l050",
            "gslq8-l000", "gslq8-l010", "gslq8-l025", "gslq8-l050",
            "m8-l100",
        ],
        default=None,
        help="Frozen Experiment 11 dense-Q soft-specialization run identity",
    )
    p.add_argument(
        "--experiment11_probe_artifact",
        default=None,
        help="Experiment 11 fixed artifact used for the exact pre-step-1 probe",
    )
    p.add_argument(
        "--experiment11_probe_output",
        default=None,
        help="Write-once Experiment 11 step-0 probe result",
    )
    p.add_argument(
        "--mixed_partition", default=None,
        help="Experiment 2 ordered singleton/doubleton atom partition id; "
             "requires full_mh and uses attnres_heads as the atomic count",
    )
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=12)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_kv_heads", type=int, default=4)
    p.add_argument("--intermediate_size", type=int, default=1536)
    p.add_argument("--num_blocks", type=int, default=4,
                   help="Number of AttnRes blocks (for block mode)")
    p.add_argument("--gate_type", default="bias",
                   choices=["bias", "sigmoid_scalar", "sigmoid_vector", "learnable_alpha"],
                   help="Gate type for mixing AttnRes output with residual stream")
    p.add_argument("--null_source", action="store_true",
                   help="Add null source for identity init")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset_name", default="default")
    p.add_argument("--dataset_revision", default=None,
                   help="Immutable Hugging Face dataset commit for reproducibility")
    p.add_argument("--data_files", default=None,
                   help="glob of local parquet shards (overrides --dataset), "
                        "e.g. /mnt/localssd/data/anneal_pt_v3/*.parquet")
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch_size", type=int, default=4, help="per-GPU")
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--expected_global_batch", type=int, default=None,
                   help="Fail if batch_size * grad_accum * world_size differs")
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--lr_min", type=float, default=6e-5)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--max_norm", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--save_steps", default="",
                   help="Additional absolute optimizer steps to save and protect; "
                        "comma-separated. --save_every 0 disables periodic saves.")
    p.add_argument("--model_snapshot_steps", default="",
                   help="Experiment 16A model-only scientific milestones; comma-separated absolute steps")
    p.add_argument("--eval_every", type=int, default=500,
                   help="Run validation every N steps (0 to disable)")
    p.add_argument("--eval_steps", type=int, default=50,
                   help="Number of batches for validation")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--wandb_project", default="residual")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_group", default=None)
    p.add_argument("--wandb_run_id", default=None,
                   help="Pre-registered W&B run ID; required by controlled experiments")
    p.add_argument("--wandb_required", action="store_true",
                   help="Fail instead of training untracked if W&B initialization fails")
    p.add_argument("--run_name", default=None)
    p.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    p.add_argument("--tokenizer_revision", default=None,
                   help="Immutable tokenizer/model commit for reproducibility")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compile", action="store_true",
                   help="Enable torch.compile on AttnRes kernels for faster training")
    p.add_argument("--fused", action="store_true",
                   help="Fused Triton MHAR routing kernels (full_mh only): shared source "
                        "buffer + single fwd/bwd kernel per routing call")
    p.add_argument("--compile_model", action="store_true",
                   help="torch.compile the entire model (fuses attention+MLP+routing)")
    p.add_argument("--fsdp", action="store_true",
                   help="Use FSDP full-shard (ZeRO-3) instead of DDP — required for 7B+")
    p.add_argument("--grad_ckpt", action="store_true",
                   help="Activation checkpointing on each decoder layer (cuts memory for 7B+)")
    continuation = p.add_mutually_exclusive_group()
    continuation.add_argument(
        "--resume_from", default=None,
        help="Checkpoint directory for an exact same-run resume")
    continuation.add_argument(
        "--branch_from", default=None,
        help="Experiment 3 parent checkpoint for a validated new training branch")
    continuation.add_argument(
        "--experiment16_parent", default=None,
        help="Exact M8 step-2000 parent for a fresh Experiment 16A branch")
    continuation.add_argument(
        "--experiment17_parent", default=None,
        help="Exact HQ8 step-2000 parent for a fresh Experiment 17A branch")
    p.add_argument(
        "--resume_source_change_record", default=None,
        help="Write-once, hash-bound exception permitting only the documented "
             "Experiment 19 evaluation-only source correction on exact resume",
    )
    p.add_argument(
        "--resume_validation_only", action="store_true",
        help="Load an exact resume checkpoint, run its scheduled validation, and exit "
             "without an optimizer update (Experiment 19 recovery gate)",
    )
    p.add_argument(
        "--resume_validation_output", default=None,
        help="Write-once JSON result required with --resume_validation_only",
    )
    p.add_argument(
        "--branch_manifest", default=None,
        help="Frozen Experiment 3C branch-selection manifest; required with --branch_from")
    p.add_argument(
        "--branch_role", default=None,
        choices=("predicted-good", "predicted-bad", "random", "unchanged"),
        help="Frozen Experiment 3C role; required with --branch_from")
    p.add_argument(
        "--experiment16_branch", choices=("control", "aligned-localized"), default=None,
        help="Frozen Experiment 16A branch; required with --experiment16_parent")
    p.add_argument(
        "--experiment16_protocol", default=None,
        help="Frozen Experiment 16A protocol; required with --experiment16_parent")
    p.add_argument(
        "--experiment17_branch", choices=("hq8-control", "expanded-global"), default=None,
        help="Frozen Experiment 17A branch; required with --experiment17_parent")
    p.add_argument(
        "--experiment17_protocol", default=None,
        help="Frozen Experiment 17A protocol; required with --experiment17_parent")
    p.add_argument(
        "--stop_after_step", type=int, default=None,
        help="Stop at this global step without changing the full LR schedule in --steps")
    p.add_argument("--keep_last", type=int, default=2,
                   help="Number of resumable step checkpoints to retain")
    p.add_argument(
        "--reuse_step_checkpoint_as_final", action="store_true",
        help="Storage-only: use an existing terminal step-* checkpoint as final "
             "instead of writing an identical second checkpoint",
    )
    p.add_argument(
        "--keep_steps",
        default="",
        help="Comma-separated step checkpoints to protect from rotation, e.g. "
             "2000,5000,10000,20000",
    )
    return p.parse_args()


def cosine_with_warmup(step, warmup, total, lr_min_ratio):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return lr_min_ratio + (1 - lr_min_ratio) * cos


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path, value):
    """Durably append one local metric row so W&B is never the sole record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def truncate_metrics_after_step(path, step):
    """Drop uncheckpointed metric rows before replaying from an atomic resume."""
    path = Path(path)
    if not path.is_file():
        return
    kept = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid training metric JSONL at {path}:{line_number}") from exc
        if int(row["step"]) <= step:
            kept.append(row)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def validate_resume_source_change_exception(
        saved_identity, current_identity, checkpoint, record_path):
    """Validate the one narrow Experiment 19 evaluation-only resume exception."""

    from src.experiments.experiment1_partition_compatibility import sha256_path

    checkpoint = Path(checkpoint).resolve()
    record_path = Path(record_path).resolve()
    required_keys = {
        "format_version", "experiment", "failed_step", "reason",
        "from_source_commit", "to_source_commit", "checkpoint_sha256",
    }
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"invalid resume source-change record: {record_path}") from exc
    if set(record) != required_keys:
        raise RuntimeError(
            "resume source-change record fields differ from the frozen schema")
    expected_constants = {
        "format_version": 1,
        "experiment": "19-role-specific-mhar-v1",
        "failed_step": 500,
        "reason": "evaluation-only-shared-fused-context-disabled",
    }
    for key, expected in expected_constants.items():
        if record.get(key) != expected:
            raise RuntimeError(
                f"resume source-change record has invalid {key}: "
                f"expected {expected!r}, got {record.get(key)!r}")
    if saved_identity.get("experiment19_variant") != "rs-hq8":
        raise RuntimeError("source-change exception is restricted to Experiment 19 RS-HQ8")
    if record["from_source_commit"] != saved_identity.get("source_commit"):
        raise RuntimeError("source-change record does not match saved source commit")
    if record["to_source_commit"] != current_identity.get("source_commit"):
        raise RuntimeError("source-change record does not match current source commit")
    saved_without_source = dict(saved_identity)
    current_without_source = dict(current_identity)
    saved_without_source.pop("source_commit", None)
    current_without_source.pop("source_commit", None)
    saved_without_source.pop("source_change_exception", None)
    current_without_source.pop("source_change_exception", None)
    if saved_without_source != current_without_source:
        raise RuntimeError(
            "resume source-change exception rejected a non-source identity change")
    observed_checkpoint_hash = sha256_path(checkpoint)
    if record["checkpoint_sha256"] != observed_checkpoint_hash:
        raise RuntimeError(
            "resume source-change record checkpoint hash mismatch: "
            f"expected {record['checkpoint_sha256']}, got {observed_checkpoint_hash}")
    return record


def training_identity(args, world_size, *, data_identity=None):
    """Fields that must remain fixed for a scientifically exact resume."""

    identity = {
        "source_commit": git_commit(),
        "mode": args.mode,
        "attnres_heads": args.attnres_heads,
        "mixed_partition": args.mixed_partition,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "intermediate_size": args.intermediate_size,
        "dataset": args.dataset,
        "dataset_name": args.dataset_name,
        "dataset_revision": args.dataset_revision,
        "data_files": data_files_identity(args.data_files) if data_identity is None else data_identity,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "seq_len": args.seq_len,
        "steps": args.steps,
        "per_gpu_batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "world_size": world_size,
        "global_batch_size": args.batch_size * args.grad_accum * world_size,
        "lr": args.lr,
        "lr_min": args.lr_min,
        "warmup": args.warmup,
        "max_norm": args.max_norm,
        "optimizer": "AdamW",
        "optimizer_betas": [0.9, 0.95],
        "optimizer_eps": 1e-8,
        "weight_decay": 0.1,
        "precision": "bf16",
        "seed": args.seed,
        "fused": args.fused,
        "compile": args.compile,
        "compile_model": args.compile_model,
        "fsdp": args.fsdp,
        "grad_ckpt": args.grad_ckpt,
    }
    qkv_groups = getattr(args, "qkv_groups", None)
    if qkv_groups is not None:
        # Omit this key for every legacy run so old exact-resume identities
        # remain byte-for-byte unchanged.
        identity["experiment6_qkv_groups"] = qkv_groups
    experiment6_variant = getattr(args, "experiment6_variant", None)
    if experiment6_variant is not None:
        identity["experiment6_variant"] = experiment6_variant
    local_q_groups = getattr(args, "local_q_groups", None)
    if local_q_groups is not None:
        identity["experiment7_local_q_groups"] = local_q_groups
    experiment7_variant = getattr(args, "experiment7_variant", None)
    if experiment7_variant is not None:
        identity["experiment7_variant"] = experiment7_variant
    hybrid_q_groups = getattr(args, "hybrid_q_groups", None)
    if hybrid_q_groups is not None:
        identity["experiment8_hybrid_q_groups"] = hybrid_q_groups
        identity["experiment8_local_head_position"] = "even"
        identity["experiment8_global_head_position"] = "odd"
    experiment8_variant = getattr(args, "experiment8_variant", None)
    if experiment8_variant is not None:
        identity["experiment8_variant"] = experiment8_variant
    experiment19_variant = getattr(args, "experiment19_variant", None)
    if experiment19_variant is not None:
        identity["experiment19_variant"] = experiment19_variant
        identity["experiment19_role_router"] = True
    experiment20_route_count = getattr(args, "experiment20_route_count", None)
    if experiment20_route_count is not None:
        identity["experiment20_route_count"] = int(experiment20_route_count)
        identity["experiment20_multislot_router"] = True
        identity["experiment20_router_backend"] = args.experiment20_router_backend
    experiment21_variant = getattr(args, "experiment21_variant", None)
    if experiment21_variant is not None:
        identity["experiment21_variant"] = experiment21_variant
        identity["experiment21_request"] = True
        identity["experiment21_request_rank"] = 4
    experiment22_variant = getattr(args, "experiment22_variant", None)
    if experiment22_variant is not None:
        identity["experiment22_variant"] = experiment22_variant
        identity["experiment22_contrast"] = True
    if getattr(args, "experiment23_variant", None) is not None:
        from src.experiments.experiment23_route_topology import RUNS, training_id
        features, routes = RUNS[args.experiment23_variant]
        identity.update(experiment23_variant=args.experiment23_variant,
            experiment23_route_topology=True, experiment23_features=features,
            experiment23_routes=routes, experiment23_run_id=training_id(args.experiment23_variant),
            experiment23_router_backend=args.experiment23_router_backend,
            experiment23_projection_backend=args.experiment23_projection_backend)
        from src.experiments.experiment23_workflow import PROTOCOL, MATRIX, LOCK
        identity.update(experiment23_protocol_sha256=sha256_file(PROTOCOL),
                        experiment23_matrix_sha256=sha256_file(MATRIX),
                        experiment23_data_lock_sha256=sha256_file(LOCK))
    if getattr(args, "experiment24_variant", None) is not None:
        from src.experiments.experiment24_three_route import RUNS, training_id
        from src.experiments.experiment24_workflow import PROTOCOL, MATRIX, LOCK
        identity.update(experiment24_variant=args.experiment24_variant,
            experiment24_three_route=True, experiment24_features="GGGG",
            experiment24_routes=RUNS[args.experiment24_variant],
            experiment24_run_id=training_id(args.experiment24_variant),
            experiment24_router_backend=args.experiment24_router_backend,
            experiment24_protocol_sha256=sha256_file(PROTOCOL),
            experiment24_matrix_sha256=sha256_file(MATRIX),
            experiment24_data_lock_sha256=sha256_file(LOCK))
    if getattr(args, "experiment25_variant", None) is not None:
        from src.experiments.experiment25_workflow import PROTOCOL, MATRIX, LOCK
        identity.update(experiment25_variant=args.experiment25_variant,
            experiment25_global_local=True, experiment25_routes="ABBC",
            experiment25_run_id="exp25-abbc-gl64-l16-seed42-step2000",
            experiment25_router_backend=args.experiment25_router_backend,
            experiment25_protocol_sha256=sha256_file(PROTOCOL),
            experiment25_matrix_sha256=sha256_file(MATRIX),
            experiment25_data_lock_sha256=sha256_file(LOCK))
    experiment11_run = getattr(args, "experiment11_run", None)
    if experiment11_run is not None:
        from src.experiments.experiment11_soft_specialization import run_spec
        row = run_spec(experiment11_run)
        identity.update({
            "experiment11_run_id": row.run_id,
            "experiment11_family": row.family,
            "experiment11_lambda": row.lambda_value,
            "experiment11_soft_q_groups": 8,
        })
    if args.branch_from:
        branch_manifest = json.loads(
            Path(args.branch_manifest).read_text(encoding="utf-8"))
        identity["branch"] = {
            "role": args.branch_role,
            "parent_checkpoint": str(Path(args.branch_from).resolve()),
            "parent_checkpoint_sha256": branch_manifest["parent_checkpoint_sha256"],
            "selection_manifest": str(Path(args.branch_manifest).resolve()),
            "selection_manifest_sha256": sha256_file(Path(args.branch_manifest)),
        }
    if args.experiment16_parent:
        from src.experiments.experiment16_causal_adaptation import protocol_sha256
        spec = json.loads(Path(args.experiment16_protocol).read_text(encoding="utf-8"))
        identity["experiment16"] = {
            "branch": args.experiment16_branch,
            "parent_checkpoint": str(Path(args.experiment16_parent).resolve()),
            "parent_checkpoint_sha256": spec["parent_checkpoint"]["sha256"],
            "protocol": str(Path(args.experiment16_protocol).resolve()),
            "protocol_sha256": protocol_sha256(Path(args.experiment16_protocol)),
            "query_mask": "all-ones-dense" if args.experiment16_branch == "control" else "even-local-odd-global",
        }
    if args.experiment17_parent:
        from src.experiments.experiment17_reverse_adaptation import protocol_sha256
        spec = json.loads(Path(args.experiment17_protocol).read_text(encoding="utf-8"))
        identity["experiment17"] = {
            "branch": args.experiment17_branch,
            "parent_checkpoint": str(Path(args.experiment17_parent).resolve()),
            "parent_checkpoint_sha256": spec["parent_checkpoint"]["sha256"],
            "protocol": str(Path(args.experiment17_protocol).resolve()),
            "protocol_sha256": protocol_sha256(Path(args.experiment17_protocol)),
            "query_expansion": (
                "none-native-hq8" if args.experiment17_branch == "hq8-control"
                else "zero-initialized-cross-chunk"
            ),
        }
    return identity


def wandb_tags(args):
    """Keep legacy run metadata unchanged and describe opt-in experiments."""

    if getattr(args, "experiment25_variant", None) is not None:
        return ["experiment-25", args.experiment25_variant, "cache-neutral",
                "single-seed", "step-2000-screen"]
    if getattr(args, "experiment24_variant", None) is not None:
        return ["experiment-24", args.experiment24_variant, "three-route-topology",
                "single-seed", "step-2000-screen"]
    if getattr(args, "experiment23_variant", None) is not None:
        return ["experiment-23", args.experiment23_variant, "route-topology",
                "single-seed", "step-2000-screen"]
    if getattr(args, "experiment21_variant", None) is not None:
        return [
            "experiment-21", str(args.experiment21_variant),
            "request-conditioned-local", "single-seed", "step-2000-screen",
        ]
    if getattr(args, "experiment22_variant", None) is not None:
        return [
            "experiment-22", str(args.experiment22_variant),
            "contrast-readout", "single-seed", "step-2000-screen",
        ]
    if getattr(args, "experiment20_route_count", None) is not None:
        route_count = int(args.experiment20_route_count)
        return [
            "experiment-20", f"r{route_count}", "multislot-rs-hq8",
            "single-seed", "global-route-kv", "step-2000-screen",
        ]

    if getattr(args, "experiment19_variant", None) is not None:
        return [
            "experiment-19", "rs-hq8", "role-specific-mhar", "single-seed",
            "local-q-depth-router", "global-route-kv", "step-2000-screen",
        ]

    if getattr(args, "experiment17_branch", None) is not None:
        return [
            "experiment-17a", "reverse-adaptation", args.experiment17_branch,
            "hq8-parent-step-2000", "function-preserving-expansion", "single-seed",
        ]

    if getattr(args, "experiment16_branch", None) is not None:
        return [
            "experiment-16a", "causal-adaptation", args.experiment16_branch,
            "m8-parent-step-2000", "masked-dense-q", "single-seed",
        ]

    experiment11_run = getattr(args, "experiment11_run", None)
    if experiment11_run is not None:
        from src.experiments.experiment11_soft_specialization import run_spec
        row = run_spec(experiment11_run)
        return [
            "experiment-11", row.run_id, row.family, "soft-query-specialization",
            f"lambda-{row.lambda_value:g}", "dense-q", "mhar", "full-mh", "h8",
            "1b", "fineweb-edu", "single-seed-screen",
        ]

    experiment8_variant = getattr(args, "experiment8_variant", None)
    if experiment8_variant is not None:
        tags = [
            "experiment-8", experiment8_variant, "hybrid-q-global-kv", "1b",
            "fineweb-edu", "q-local-fraction-0.5", "local-even-global-odd",
        ]
        if args.mode == "full_mh":
            tags.extend(["mhar", "full-mh", "h8"])
        else:
            tags.append("ordinary-residual")
        return tags

    experiment7_variant = getattr(args, "experiment7_variant", None)
    if experiment7_variant is not None:
        tags = [
            "experiment-7", experiment7_variant, "local-q-global-kv", "1b", "fineweb-edu",
            f"q-groups-{args.local_q_groups}",
        ]
        if args.mode == "full_mh":
            tags.extend(["mhar", "full-mh", f"h{args.attnres_heads}"])
        else:
            tags.append("ordinary-residual")
        return tags

    experiment6_variant = getattr(args, "experiment6_variant", None)
    if experiment6_variant is not None:
        tags = [
            "experiment-6",
            experiment6_variant,
            "1b",
            "fineweb-edu",
            f"qkv-groups-{getattr(args, 'qkv_groups', None) or 'dense'}",
        ]
        if args.mode == "full_mh":
            tags.extend(["mhar", "full-mh", f"h{args.attnres_heads}"])
        else:
            tags.append("ordinary-residual")
        return tags
    return [
        "mhar", "full-mh", f"h{args.attnres_heads}", "1b", "fineweb-edu",
        *(["mixed-width", "experiment-2"] if args.mixed_partition else []),
        *(["experiment-3", "actionability", args.branch_role]
          if args.branch_from else []),
    ]


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path):
    path = Path(path).resolve()
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    for candidate in sorted(value for value in path.rglob("*") if value.is_file()):
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with candidate.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def data_files_identity(pattern):
    if not pattern:
        return None
    matches = sorted(Path(value).resolve() for value in glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"--data_files matched no files: {pattern}")
    return {
        "pattern": pattern,
        "files": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in matches
        ],
    }


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_training_state(checkpoint_dir):
    path = Path(checkpoint_dir).resolve() / "training_state.pt"
    if not path.is_file():
        raise FileNotFoundError(f"missing resumable training state: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "global_step", "chunks_consumed", "optimizer", "scheduler",
        "cpu_rng_state", "cuda_rng_state_all", "run_identity",
    }
    missing = required - set(state)
    if missing:
        raise ValueError(f"training state is missing fields: {sorted(missing)}")
    return state


def validate_model_config(model, args):
    expected = {
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.num_layers,
        "num_attention_heads": args.num_heads,
        "num_key_value_heads": args.num_kv_heads,
        "intermediate_size": args.intermediate_size,
        "head_dim": args.hidden_size // args.num_heads,
        "max_position_embeddings": args.seq_len * 2,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
    }
    if args.mode not in ("baseline", "moe"):
        expected.update({
            "attnres_num_heads": args.attnres_heads,
            "attnres_mode": args.mode,
        })
    qkv_groups = getattr(args, "qkv_groups", None)
    if qkv_groups is not None:
        expected["experiment6_qkv_groups"] = qkv_groups
    experiment6_variant = getattr(args, "experiment6_variant", None)
    if experiment6_variant is not None:
        expected["experiment6_variant"] = experiment6_variant
    local_q_groups = getattr(args, "local_q_groups", None)
    if local_q_groups is not None:
        expected["experiment7_local_q_groups"] = local_q_groups
    experiment7_variant = getattr(args, "experiment7_variant", None)
    if experiment7_variant is not None:
        expected["experiment7_variant"] = experiment7_variant
    hybrid_q_groups = getattr(args, "hybrid_q_groups", None)
    if hybrid_q_groups is not None:
        expected["experiment8_hybrid_q_groups"] = hybrid_q_groups
        expected["experiment8_local_head_position"] = "even"
        expected["experiment8_global_head_position"] = "odd"
    experiment8_variant = getattr(args, "experiment8_variant", None)
    if experiment8_variant is not None:
        expected["experiment8_variant"] = experiment8_variant
    if getattr(args, "experiment19_variant", None) is not None:
        expected["experiment19_role_router"] = True
    experiment20_route_count = getattr(args, "experiment20_route_count", None)
    if experiment20_route_count is not None:
        expected["experiment20_route_count"] = int(experiment20_route_count)
        expected["experiment20_multislot_router"] = True
        expected["experiment20_router_backend"] = args.experiment20_router_backend
    if getattr(args, "experiment21_variant", None) is not None:
        expected["experiment21_variant"] = args.experiment21_variant
        expected["experiment21_request"] = True
        expected["experiment21_request_rank"] = 4
    if getattr(args, "experiment22_variant", None) is not None:
        expected["experiment22_variant"] = args.experiment22_variant
        expected["experiment22_contrast"] = True
    if getattr(args, "experiment23_variant", None) is not None:
        expected.update(experiment23_variant=args.experiment23_variant,
            experiment23_route_topology=True,
            experiment23_router_backend=args.experiment23_router_backend,
            experiment23_projection_backend=args.experiment23_projection_backend)
    if getattr(args, "experiment24_variant", None) is not None:
        expected.update(experiment24_variant=args.experiment24_variant,
            experiment24_three_route=True,
            experiment24_router_backend=args.experiment24_router_backend)
    if getattr(args, "experiment25_variant", None) is not None:
        expected.update(experiment25_variant=args.experiment25_variant,
            experiment25_global_local=True,
            experiment25_router_backend=args.experiment25_router_backend)
    experiment11_run = getattr(args, "experiment11_run", None)
    if experiment11_run is not None:
        from src.experiments.experiment11_soft_specialization import run_spec
        row = run_spec(experiment11_run)
        expected.update({
            "experiment11_run_id": row.run_id,
            "experiment11_family": row.family,
            "experiment11_lambda": row.lambda_value,
            "experiment11_soft_q_groups": 8,
        })
    experiment17_branch = getattr(args, "experiment17_branch", None)
    if experiment17_branch is not None:
        expected.update({
            "experiment17_branch": experiment17_branch,
            "experiment17_query_expansion": (
                "none-native-hq8" if experiment17_branch == "hq8-control"
                else "zero-initialized-cross-chunk"
            ),
        })
    observed = {key: getattr(model.config, key, None) for key in expected}
    mismatches = {
        key: {"expected": expected[key], "observed": observed[key]}
        for key in expected if observed[key] != expected[key]
    }
    if mismatches:
        raise ValueError(f"checkpoint/model configuration mismatch: {mismatches}")
    return observed


def token_stream(dataset_name, config_name, tokenizer, seq_len, rank, world_size, seed,
                 data_files=None, dataset_revision=None, skip_chunks=0):
    import time
    from datasets import load_dataset

    def make_ds(n_skip):
        if data_files:
            ds = load_dataset("parquet", data_files=data_files, split="train",
                              streaming=True)
        else:
            ds = load_dataset(dataset_name, name=config_name, split="train",
                              revision=dataset_revision, streaming=True)
        ds = ds.shuffle(seed=seed + rank, buffer_size=10_000)
        return ds.skip(rank + n_skip)

    # Transient HF CDN errors (e.g. 408 mid-stream) bypass datasets' built-in
    # retries; reconnect and fast-forward past the samples already consumed.
    n_consumed = 0
    n_chunks = 0
    buf = []
    while True:
        try:
            for sample in make_ds(n_consumed):
                n_consumed += 1
                text = sample.get("text") or sample.get("content") or sample.get("wikitext") or ""
                if not text:
                    continue
                ids = tokenizer.encode(text, add_special_tokens=False)
                ids.append(tokenizer.eos_token_id)
                buf.extend(ids)
                while len(buf) >= seq_len + 1:
                    chunk = buf[:seq_len + 1]
                    buf = buf[world_size * seq_len:]
                    n_chunks += 1
                    if n_chunks > skip_chunks:
                        yield torch.tensor(chunk, dtype=torch.long)
            return
        except Exception as e:
            print(f"[rank {rank}] data stream error after {n_consumed} samples: "
                  f"{e!r} — reconnecting in 30s", flush=True)
            time.sleep(30)


def build_model(args, device):
    """Build model from scratch based on mode."""
    qkv_groups = getattr(args, "qkv_groups", None)
    experiment6_variant = getattr(args, "experiment6_variant", None)
    local_q_groups = getattr(args, "local_q_groups", None)
    experiment7_variant = getattr(args, "experiment7_variant", None)
    hybrid_q_groups = getattr(args, "hybrid_q_groups", None)
    experiment8_variant = getattr(args, "experiment8_variant", None)
    experiment19_variant = getattr(args, "experiment19_variant", None)
    experiment20_route_count = getattr(args, "experiment20_route_count", None)
    experiment21_variant = getattr(args, "experiment21_variant", None)
    experiment22_variant = getattr(args, "experiment22_variant", None)
    experiment23_variant = getattr(args, "experiment23_variant", None)
    experiment24_variant = getattr(args, "experiment24_variant", None)
    experiment25_variant = getattr(args, "experiment25_variant", None)
    experiment11_run = getattr(args, "experiment11_run", None)
    selected_projections = sum(value is not None for value in (
        qkv_groups, local_q_groups, hybrid_q_groups, experiment11_run))
    if selected_projections > 1:
        raise ValueError(
            "Experiment 6, 7, 8, and 11 projection modes are mutually exclusive")
    selected_identities = sum(value is not None for value in (
        experiment6_variant, experiment7_variant, experiment8_variant,
        experiment11_run, experiment19_variant, experiment20_route_count,
        experiment21_variant, experiment22_variant, experiment23_variant, experiment24_variant,
        experiment25_variant))
    if selected_identities > 1:
        raise ValueError("only one frozen experiment identity may be selected")
    if experiment25_variant is not None:
        if (getattr(args, "experiment16_branch", None) or getattr(args, "experiment17_branch", None)
            or args.mode != "full_mh" or args.attnres_heads != 8 or args.num_heads != 16
            or args.num_kv_heads != 8 or args.hidden_size != 1280 or selected_projections
            or getattr(args, "partition", None) or getattr(args, "mixed_partition", None)
            or getattr(args, "contiguous_partition", None)):
            raise ValueError("Experiment 25 requires frozen ABBC 1280D full_mh H8")
        from src.experiments.experiment25_global_local import Experiment25MHARForCausalLM
    if experiment24_variant is not None:
        if (getattr(args, "experiment16_branch", None) or getattr(args, "experiment17_branch", None)
            or args.mode != "full_mh" or args.attnres_heads != 8 or args.num_heads != 16
            or args.num_kv_heads != 8 or args.hidden_size % 8 or selected_projections
            or getattr(args, "partition", None) or getattr(args, "mixed_partition", None)
            or getattr(args, "contiguous_partition", None)):
            raise ValueError("Experiment 24 requires native full_mh H8 and no other architecture")
        from src.experiments.experiment24_three_route import Experiment24MHARForCausalLM
    if experiment23_variant is not None:
        if getattr(args, "experiment16_branch", None) or getattr(args, "experiment17_branch", None):
            raise ValueError("Experiment 23 cannot use architectural-adaptation branches")
        if (args.mode != "full_mh" or args.attnres_heads != 8 or args.num_heads != 16
            or args.num_kv_heads != 8 or args.hidden_size % 8 or selected_projections
            or getattr(args, "partition", None) or getattr(args, "mixed_partition", None)
            or getattr(args, "contiguous_partition", None)):
            raise ValueError("Experiment 23 requires native full_mh H8 and no other projection mode")
        from src.experiments.experiment23_route_topology import Experiment23MHARForCausalLM
    if experiment11_run is not None:
        if args.mode != "full_mh" or args.attnres_heads != 8:
            raise ValueError("Experiment 11 requires mode=full_mh and attnres_heads=8")
        if (args.hidden_size % 8 or args.num_kv_heads != 8 or
                args.num_heads != 16):
            raise ValueError("Experiment 11 requires H8 chunks, 8 KV heads, and 16 Q heads")
        from src.experiments.experiment11_soft_specialization import (
            Experiment11MHARForCausalLM,
        )
    if experiment19_variant is not None:
        if (experiment19_variant != "rs-hq8" or args.mode != "full_mh"
                or args.attnres_heads != 8 or hybrid_q_groups != 8
                or args.num_kv_heads != 8 or args.num_heads != 16
                or args.hidden_size % 8 or args.mixed_partition is not None):
            raise ValueError(
                "Experiment 19 requires rs-hq8, full_mh H8, 8 KV heads, "
                "16 Q heads, hybrid_q_groups=8, and no runtime partition")
        from src.experiments.experiment19_role_specific_routing import (
            Experiment19MHARForCausalLM,
        )
    if experiment20_route_count is not None:
        if (args.mode != "full_mh" or args.attnres_heads != 8
                or hybrid_q_groups != 8 or args.num_kv_heads != 8
                or args.num_heads != 16 or args.hidden_size % 8
                or args.mixed_partition is not None):
            raise ValueError(
                "Experiment 20 requires full_mh H8, 8 KV heads, 16 Q heads, "
                "hybrid_q_groups=8, and no runtime partition")
        from src.experiments.experiment20_multislot_routing import (
            Experiment20MHARForCausalLM,
        )
    if experiment21_variant is not None:
        if (args.mode != "full_mh" or args.attnres_heads != 8
                or hybrid_q_groups != 8 or args.num_kv_heads != 8
                or args.num_heads != 16 or args.hidden_size % 8
                or args.mixed_partition is not None):
            raise ValueError(
                "Experiment 21 requires full_mh H8, 8 KV heads, 16 Q heads, "
                "hybrid_q_groups=8, and no runtime partition")
        from src.experiments.experiment21_request_conditioned import (
            Experiment21MHARForCausalLM,
        )
    if experiment22_variant is not None:
        if (args.mode != "full_mh" or args.attnres_heads != 8
                or hybrid_q_groups != 8 or args.num_kv_heads != 8
                or args.num_heads != 16 or args.hidden_size % 8
                or args.mixed_partition is not None):
            raise ValueError(
                "Experiment 22 requires full_mh H8, 8 KV heads, 16 Q heads, "
                "hybrid_q_groups=8, and no runtime partition")
        from src.experiments.experiment22_contrast_readout import (
            Experiment22MHARForCausalLM,
        )
    if experiment8_variant is not None:
        frozen8 = {
            "hq8": ("full_mh", 8, 8),
            "bhq8": ("baseline", None, 8),
        }
        expected_mode, expected_heads, expected_groups = frozen8[experiment8_variant]
        observed_heads = args.attnres_heads if args.mode == "full_mh" else None
        observed = (args.mode, observed_heads, hybrid_q_groups)
        expected = (expected_mode, expected_heads, expected_groups)
        if observed != expected:
            raise ValueError(
                f"Experiment 8 variant {experiment8_variant} requires "
                f"mode/heads/hybrid_q_groups={expected}, got {observed}")
    if hybrid_q_groups is not None:
        if args.mode not in ("baseline", "full_mh"):
            raise ValueError("--hybrid_q_groups is valid only with baseline or full_mh")
        if args.mode == "full_mh" and args.attnres_heads != hybrid_q_groups:
            raise ValueError("hybrid-Q MHAR requires attnres_heads == hybrid_q_groups")
        if (args.hidden_size % hybrid_q_groups or
                args.num_kv_heads != hybrid_q_groups or
                args.num_heads != 2 * hybrid_q_groups):
            raise ValueError("hybrid Q requires H8 chunks, 8 KV heads, and 16 Q heads")
        from src.experiments.experiment8_hybrid_q import (
            Experiment8BaselineForCausalLM,
            Experiment8MHARForCausalLM,
        )
    if experiment7_variant is not None:
        frozen7 = {
            "lq4": ("full_mh", 4, 4),
            "lq8": ("full_mh", 8, 8),
            "blq4": ("baseline", None, 4),
            "blq8": ("baseline", None, 8),
        }
        expected_mode, expected_heads, expected_groups = frozen7[experiment7_variant]
        observed_heads = args.attnres_heads if args.mode == "full_mh" else None
        observed = (args.mode, observed_heads, local_q_groups)
        expected = (expected_mode, expected_heads, expected_groups)
        if observed != expected:
            raise ValueError(
                f"Experiment 7 variant {experiment7_variant} requires "
                f"mode/heads/local_q_groups={expected}, got {observed}")
    if local_q_groups is not None:
        if args.mode not in ("baseline", "full_mh"):
            raise ValueError("--local_q_groups is valid only with baseline or full_mh")
        if args.mode == "full_mh" and args.attnres_heads != local_q_groups:
            raise ValueError("local-Q MHAR requires attnres_heads == local_q_groups")
        if args.hidden_size % local_q_groups or args.num_heads % local_q_groups:
            raise ValueError("hidden and Q-head counts must divide local_q_groups")
        from src.experiments.experiment7_local_q import (
            Experiment7BaselineForCausalLM,
            Experiment7MHARForCausalLM,
        )
    if experiment6_variant is not None:
        frozen = {
            "b": ("baseline", None, None),
            "m4": ("full_mh", 4, None),
            "c4": ("full_mh", 4, 4),
            "g4": ("baseline", None, 4),
            "m8": ("full_mh", 8, None),
            "c8": ("full_mh", 8, 8),
            "g8": ("baseline", None, 8),
        }
        expected_mode, expected_heads, expected_groups = frozen[experiment6_variant]
        observed_heads = args.attnres_heads if args.mode == "full_mh" else None
        observed = (args.mode, observed_heads, qkv_groups)
        expected = (expected_mode, expected_heads, expected_groups)
        if observed != expected:
            raise ValueError(
                f"Experiment 6 variant {experiment6_variant} requires "
                f"mode/heads/qkv_groups={expected}, got {observed}")
    if qkv_groups is not None:
        if args.mode not in ("baseline", "full_mh"):
            raise ValueError("--qkv_groups is valid only with --mode baseline or full_mh")
        if args.mode == "full_mh" and args.attnres_heads != qkv_groups:
            raise ValueError("coupled MHAR requires --attnres_heads to equal --qkv_groups")
        if args.hidden_size % qkv_groups or args.num_heads % qkv_groups or args.num_kv_heads % qkv_groups:
            raise ValueError("hidden, Q-head, and KV-head counts must divide --qkv_groups")
        from src.experiments.experiment6_coupled_qkv import (
            Experiment6BaselineForCausalLM,
            Experiment6MHARForCausalLM,
        )
    source_checkpoint = (
        args.resume_from or args.branch_from
        or getattr(args, "experiment16_parent", None)
        or getattr(args, "experiment17_parent", None)
    )
    if source_checkpoint:
        if args.fsdp:
            raise NotImplementedError("checkpoint continuation is currently supported for DDP runs")
        if args.mode == "baseline":
            model_class = (Experiment8BaselineForCausalLM if hybrid_q_groups else
                           Experiment7BaselineForCausalLM if local_q_groups else
                           Experiment6BaselineForCausalLM if qkv_groups else Qwen3ForCausalLM)
        elif args.mode == "moe":
            from transformers import Qwen3MoeForCausalLM
            model_class = Qwen3MoeForCausalLM
        else:
            if args.resume_from and getattr(args, "experiment17_branch", None):
                from src.experiments.experiment17_reverse_adaptation import Experiment17MHARForCausalLM
                model_class = Experiment17MHARForCausalLM
            elif args.resume_from and getattr(args, "experiment16_branch", None):
                from src.experiments.experiment16_causal_adaptation import Experiment16MHARForCausalLM
                model_class = Experiment16MHARForCausalLM
            else:
                model_class = (Experiment25MHARForCausalLM if experiment25_variant else
                           Experiment24MHARForCausalLM if experiment24_variant else
                           Experiment23MHARForCausalLM if experiment23_variant else
                           Experiment11MHARForCausalLM if experiment11_run else
                           Experiment21MHARForCausalLM if experiment21_variant else
                           Experiment22MHARForCausalLM if experiment22_variant else
                           Experiment20MHARForCausalLM if experiment20_route_count else
                           Experiment19MHARForCausalLM if experiment19_variant else
                           Experiment8MHARForCausalLM if hybrid_q_groups else
                           Experiment7MHARForCausalLM if local_q_groups else
                           Experiment6MHARForCausalLM if qkv_groups else Qwen3AttnResForCausalLM)
        model = model_class.from_pretrained(
            source_checkpoint, dtype=torch.bfloat16).to(device=device)
        if getattr(args, "experiment16_parent", None):
            from src.experiments.experiment16_causal_adaptation import install_query_mask
            install_query_mask(model, args.experiment16_branch)
        if getattr(args, "experiment17_parent", None):
            from src.experiments.experiment17_reverse_adaptation import expand_hybrid_query
            expand_hybrid_query(model, args.experiment17_branch)
        validate_model_config(model, args)
        if args.mixed_partition:
            from src.attention_residuals.mhar_partition import parse_mixed_partition_id
            model.set_mhar_mixed_partition(parse_mixed_partition_id(
                args.mixed_partition, num_atomic_blocks=args.attnres_heads))
        return model

    common = dict(
        vocab_size=151936,  # Qwen3 tokenizer vocab
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.seq_len * 2,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        head_dim=args.hidden_size // args.num_heads,
    )

    if args.mode == "baseline":
        config = Qwen3Config(**common)
        if experiment6_variant:
            config.experiment6_variant = experiment6_variant
        if experiment7_variant:
            config.experiment7_variant = experiment7_variant
        if experiment8_variant:
            config.experiment8_variant = experiment8_variant
        if hybrid_q_groups:
            config.experiment8_hybrid_q_groups = hybrid_q_groups
            config.experiment8_local_head_position = "even"
            config.experiment8_global_head_position = "odd"
            model = Experiment8BaselineForCausalLM(config)
        elif local_q_groups:
            config.experiment7_local_q_groups = local_q_groups
            model = Experiment7BaselineForCausalLM(config)
        elif qkv_groups:
            config.experiment6_qkv_groups = qkv_groups
            model = Experiment6BaselineForCausalLM(config)
        else:
            model = Qwen3ForCausalLM(config)
    elif args.mode == "moe":
        # Stock Qwen3-MoE, attention identical to baseline; iso-activated-FLOPs
        # MLP (topk * moe_ff == dense intermediate_size). Aux load-balancing
        # loss is train-only; evaluate() reports pure CE for moe.
        from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
        config = Qwen3MoeConfig(
            num_experts=args.moe_experts,
            num_experts_per_tok=args.moe_topk,
            moe_intermediate_size=args.moe_ff,
            decoder_sparse_step=1,
            mlp_only_layers=[],
            norm_topk_prob=True,
            output_router_logits=True,
            router_aux_loss_coef=0.001,
            **common,
        )
        model = Qwen3MoeForCausalLM(config)
    else:
        config = Qwen3AttnResConfig(
            attnres_num_blocks=args.num_blocks,

            attnres_mode=args.mode,
            attnres_gate_type=args.gate_type,
            attnres_use_null_source=args.null_source,
            attnres_num_heads=args.attnres_heads,
            attnres_hyper_n=args.hyper_n,
            **common,
        )
        if experiment6_variant:
            config.experiment6_variant = experiment6_variant
        if experiment7_variant:
            config.experiment7_variant = experiment7_variant
        if experiment8_variant:
            config.experiment8_variant = experiment8_variant
        if experiment19_variant:
            config.experiment19_role_router = True
        if experiment20_route_count:
            config.experiment20_route_count = int(experiment20_route_count)
            config.experiment20_multislot_router = True
            config.experiment20_router_backend = args.experiment20_router_backend
        if experiment21_variant:
            config.experiment21_variant = experiment21_variant
            config.experiment21_request = True
            config.experiment21_request_rank = 4
        if experiment22_variant:
            config.experiment22_variant = experiment22_variant
            config.experiment22_contrast = True
        if experiment25_variant:
            config._attn_implementation = "sdpa"
            config.experiment25_variant = experiment25_variant
            config.experiment25_global_local = True
            config.experiment25_router_backend = args.experiment25_router_backend
        if experiment24_variant:
            config._attn_implementation = "sdpa"
            config.experiment24_variant = experiment24_variant
            config.experiment24_three_route = True
            config.experiment24_router_backend = args.experiment24_router_backend
        if experiment23_variant:
            config._attn_implementation = "sdpa"
            config.experiment23_variant = experiment23_variant
            config.experiment23_route_topology = True
            config.experiment23_router_backend = args.experiment23_router_backend
            config.experiment23_projection_backend = args.experiment23_projection_backend
        if experiment11_run:
            from src.experiments.experiment11_soft_specialization import run_spec
            row = run_spec(experiment11_run)
            config.experiment11_run_id = row.run_id
            config.experiment11_family = row.family
            config.experiment11_lambda = row.lambda_value
            config.experiment11_soft_q_groups = 8
        if experiment25_variant:
            model = Experiment25MHARForCausalLM(config)
        elif experiment24_variant:
            model = Experiment24MHARForCausalLM(config)
        elif experiment23_variant:
            model = Experiment23MHARForCausalLM(config)
        elif experiment11_run:
            model = Experiment11MHARForCausalLM(config)
        elif experiment21_variant:
            config.experiment8_hybrid_q_groups = hybrid_q_groups
            config.experiment8_local_head_position = "even"
            config.experiment8_global_head_position = "odd"
            model = Experiment21MHARForCausalLM(config)
        elif experiment22_variant:
            config.experiment8_hybrid_q_groups = hybrid_q_groups
            config.experiment8_local_head_position = "even"
            config.experiment8_global_head_position = "odd"
            model = Experiment22MHARForCausalLM(config)
        elif experiment20_route_count:
            config.experiment8_hybrid_q_groups = hybrid_q_groups
            config.experiment8_local_head_position = "even"
            config.experiment8_global_head_position = "odd"
            model = Experiment20MHARForCausalLM(config)
        elif experiment19_variant:
            config.experiment8_hybrid_q_groups = hybrid_q_groups
            config.experiment8_local_head_position = "even"
            config.experiment8_global_head_position = "odd"
            model = Experiment19MHARForCausalLM(config)
        elif hybrid_q_groups:
            config.experiment8_hybrid_q_groups = hybrid_q_groups
            config.experiment8_local_head_position = "even"
            config.experiment8_global_head_position = "odd"
            model = Experiment8MHARForCausalLM(config)
        elif local_q_groups:
            config.experiment7_local_q_groups = local_q_groups
            model = Experiment7MHARForCausalLM(config)
        elif qkv_groups:
            config.experiment6_qkv_groups = qkv_groups
            model = Experiment6MHARForCausalLM(config)
        else:
            model = Qwen3AttnResForCausalLM(config)

    validate_model_config(model, args)
    if args.mixed_partition:
        if args.mode != "full_mh":
            raise ValueError("--mixed_partition requires --mode full_mh")
        from src.attention_residuals.mhar_partition import parse_mixed_partition_id
        model.set_mhar_mixed_partition(parse_mixed_partition_id(
            args.mixed_partition, num_atomic_blocks=args.attnres_heads))

    if getattr(args, "fsdp", False):
        # Keep params fp32; FSDP MixedPrecision does bf16 compute and keeps an
        # fp32 sharded master for the optimizer (stable + memory-fits when sharded).
        model = model.to(device=device)
    else:
        model = model.to(dtype=torch.bfloat16, device=device)
    return model


def unwrap_model(model):
    value = model.module if hasattr(model, "module") else model
    return value._orig_mod if hasattr(value, "_orig_mod") else value


def parse_keep_steps(value):
    if not value:
        return frozenset()
    try:
        steps = frozenset(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise ValueError("--keep_steps must be a comma-separated list of integers") from error
    if any(step < 1 for step in steps):
        raise ValueError("--keep_steps values must be positive")
    return steps


def checkpoint_due(step, save_every, save_steps=()):
    return step in save_steps or (save_every > 0 and step % save_every == 0)


BRANCH_SCIENTIFIC_IDENTITY_KEYS = frozenset({
    "mode", "attnres_heads", "hidden_size", "num_layers", "num_heads",
    "num_kv_heads", "intermediate_size", "dataset", "dataset_name",
    "dataset_revision", "data_files", "tokenizer", "tokenizer_revision",
    "seq_len", "steps", "per_gpu_batch_size", "grad_accum", "world_size",
    "global_batch_size", "lr", "lr_min", "warmup", "max_norm", "optimizer",
    "optimizer_betas", "optimizer_eps", "weight_decay", "precision", "seed",
})


def validate_branch_invocation(args, *, verify_parent_hash=False):
    if not args.branch_from:
        if args.branch_manifest or args.branch_role:
            raise ValueError("--branch_manifest/--branch_role require --branch_from")
        return None
    if not args.branch_manifest or not args.branch_role:
        raise ValueError("--branch_from requires --branch_manifest and --branch_role")
    manifest_path = Path(args.branch_manifest).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    branches = payload.get("branches", {})
    if args.branch_role not in branches:
        raise ValueError(f"branch manifest has no role {args.branch_role!r}")
    row = branches[args.branch_role]
    expected_partition = row.get("partition_id")
    if args.mixed_partition != expected_partition:
        raise ValueError(
            "--mixed_partition does not match frozen branch role: "
            f"expected={expected_partition!r}, observed={args.mixed_partition!r}")
    parent = Path(args.branch_from).resolve()
    if verify_parent_hash:
        observed_hash = sha256_tree(parent)
        if payload.get("parent_checkpoint_sha256") != observed_hash:
            raise RuntimeError("branch parent checkpoint hash differs from frozen manifest")
    return payload


def validate_branch_parent_identity(parent_identity, current_identity):
    mismatches = {
        key: {
            "parent": parent_identity.get(key),
            "branch": current_identity.get(key),
        }
        for key in BRANCH_SCIENTIFIC_IDENTITY_KEYS
        if parent_identity.get(key) != current_identity.get(key)
    }
    if mismatches:
        raise RuntimeError(
            "branch parent scientific identity mismatch:\n"
            + json.dumps(mismatches, indent=2, sort_keys=True))
    if parent_identity.get("mixed_partition") is not None:
        raise RuntimeError("Experiment 3C branches require a native H16 parent")


def prune_step_checkpoints(out_dir, keep_last, keep_steps=()):
    if keep_last < 1:
        raise ValueError("--keep_last must be at least 1")
    out_dir = Path(out_dir).resolve()
    checkpoints = []
    for candidate in out_dir.iterdir():
        match = re.fullmatch(r"step-(\d+)", candidate.name)
        if match and candidate.is_dir():
            checkpoints.append((int(match.group(1)), candidate))
    newest = {step for step, _ in sorted(checkpoints)[-keep_last:]}
    protected = set(keep_steps) | newest
    for step, candidate in checkpoints:
        if step not in protected:
            shutil.rmtree(candidate)


def save_training_checkpoint(
    *, model, tokenizer, optimizer, scheduler, global_step, chunks_consumed,
    run_identity, out_dir, keep_last, wandb_run_id, elapsed_training_seconds,
    keep_steps=(), final=False,
):
    """Atomically save weights plus optimizer/scheduler/RNG state."""

    with optional_checkpoint_lock():
        return _save_training_checkpoint_unlocked(
            model=model, tokenizer=tokenizer, optimizer=optimizer,
            scheduler=scheduler, global_step=global_step,
            chunks_consumed=chunks_consumed, run_identity=run_identity,
            out_dir=out_dir, keep_last=keep_last, wandb_run_id=wandb_run_id,
            elapsed_training_seconds=elapsed_training_seconds,
            keep_steps=keep_steps, final=final,
        )


def _save_training_checkpoint_unlocked(
    *, model, tokenizer, optimizer, scheduler, global_step, chunks_consumed,
    run_identity, out_dir, keep_last, wandb_run_id, elapsed_training_seconds,
    keep_steps=(), final=False,
):
    """Checkpoint implementation called while the optional save lock is held."""

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "final" if final else f"step-{global_step}"
    destination = out_dir / name
    temporary = out_dir / f".{name}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing checkpoint: {destination}")
    temporary.mkdir()

    unwrap_model(model).save_pretrained(temporary)
    tokenizer.save_pretrained(temporary)
    state = {
        "format_version": 1,
        "created_at": utc_now(),
        "global_step": global_step,
        "chunks_consumed": chunks_consumed,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        "run_identity": run_identity,
        "wandb_run_id": wandb_run_id,
        "elapsed_training_seconds": elapsed_training_seconds,
    }
    state_path = temporary / "training_state.pt"
    state_temporary = state_path.with_suffix(".pt.tmp")
    torch.save(state, state_temporary)
    os.replace(state_temporary, state_path)
    atomic_write_json(temporary / "training_manifest.json", {
        "format_version": 1,
        "created_at": state["created_at"],
        "global_step": global_step,
        "chunks_consumed": chunks_consumed,
        "elapsed_training_seconds": elapsed_training_seconds,
        "run_identity": run_identity,
        "wandb_run_id": wandb_run_id,
    })
    os.replace(temporary, destination)
    if not final:
        prune_step_checkpoints(out_dir, keep_last, keep_steps)
    return destination


def save_model_snapshot(*, model, tokenizer, global_step, chunks_consumed,
                        run_identity, out_dir):
    """Atomically save a non-resumable model-only measurement snapshot."""
    with optional_checkpoint_lock():
        root = Path(out_dir).resolve() / "model_snapshots"
        root.mkdir(parents=True, exist_ok=True)
        destination = root / f"step-{global_step}"
        temporary = root / f".step-{global_step}.tmp"
        if destination.exists() or temporary.exists():
            raise FileExistsError(f"refusing to overwrite model snapshot: {destination}")
        temporary.mkdir()
        unwrap_model(model).save_pretrained(temporary)
        tokenizer.save_pretrained(temporary)
        atomic_write_json(temporary / "training_manifest.json", {
            "format_version": 1,
            "checkpoint_kind": "model-only-measurement-snapshot",
            "created_at": utc_now(),
            "global_step": global_step,
            "chunks_consumed": chunks_consumed,
            "run_identity": run_identity,
        })
        os.replace(temporary, destination)
        return destination


def main():
    args = parse_args()
    if args.experiment25_variant:
        from src.experiments.experiment25_workflow import verify_trainer_invocation
        verify_trainer_invocation(args)

    probe_flags = (
        args.experiment11_probe_artifact is not None,
        args.experiment11_probe_output is not None,
    )
    if probe_flags[0] != probe_flags[1]:
        raise ValueError("Experiment 11 step-0 probe artifact/output must be provided together")
    if any(probe_flags) and (
        args.experiment11_run is None or args.resume_from or args.branch_from
        or args.experiment16_parent or args.experiment17_parent
    ):
        raise ValueError("Experiment 11 step-0 probe is valid only for a fresh Experiment 11 run")
    if args.keep_last < 1:
        raise ValueError("--keep_last must be at least 1")
    validate_branch_invocation(args)
    if args.experiment16_parent and args.experiment17_parent:
        raise ValueError("Experiment 16A and Experiment 17A parent modes are mutually exclusive")
    if (args.experiment16_branch or args.experiment16_protocol) and (
            args.experiment17_branch or args.experiment17_protocol):
        raise ValueError("Experiment 16A and Experiment 17A metadata are mutually exclusive")
    exp16_metadata = bool(args.experiment16_branch and args.experiment16_protocol)
    if args.experiment16_parent and not exp16_metadata:
        raise ValueError("--experiment16_parent requires --experiment16_branch and --experiment16_protocol")
    if exp16_metadata and not (args.experiment16_parent or args.resume_from):
        raise ValueError("Experiment 16A branch metadata requires a parent or exact resume")
    if bool(args.experiment16_branch) != bool(args.experiment16_protocol):
        raise ValueError("Experiment 16A branch and protocol must be provided together")
    if args.experiment16_parent:
        from src.experiments.experiment16_causal_adaptation import load_spec, protocol_sha256
        exp16_spec = load_spec(Path(args.experiment16_protocol))
        if sha256_tree(args.experiment16_parent) != exp16_spec["parent_checkpoint"]["sha256"]:
            raise RuntimeError("Experiment 16A parent checkpoint hash mismatch")
        if (args.mode, args.attnres_heads, args.hidden_size, args.num_layers,
                args.num_heads, args.num_kv_heads, args.seed) != (
                    "full_mh", 8, 1280, 36, 16, 8, 42):
            raise ValueError("Experiment 16A requires the frozen M8 architecture and seed")
    exp17_metadata = bool(args.experiment17_branch and args.experiment17_protocol)
    if args.experiment17_parent and not exp17_metadata:
        raise ValueError("--experiment17_parent requires --experiment17_branch and --experiment17_protocol")
    if exp17_metadata and not (args.experiment17_parent or args.resume_from):
        raise ValueError("Experiment 17A branch metadata requires a parent or exact resume")
    if bool(args.experiment17_branch) != bool(args.experiment17_protocol):
        raise ValueError("Experiment 17A branch and protocol must be provided together")
    if args.experiment17_parent:
        from src.experiments.experiment17_reverse_adaptation import load_spec
        exp17_spec = load_spec(Path(args.experiment17_protocol))
        if sha256_tree(args.experiment17_parent) != exp17_spec["parent_checkpoint"]["sha256"]:
            raise RuntimeError("Experiment 17A parent checkpoint hash mismatch")
        if (args.mode, args.attnres_heads, args.hidden_size, args.num_layers,
                args.num_heads, args.num_kv_heads, args.seed,
                args.hybrid_q_groups, args.experiment8_variant) != (
                    "full_mh", 8, 1280, 36, 16, 8, 42, 8, "hq8"):
            raise ValueError("Experiment 17A requires the frozen HQ8 architecture and seed")
    if args.save_every < 0:
        raise ValueError("--save_every must be nonnegative")
    save_steps = parse_keep_steps(args.save_steps)
    model_snapshot_steps = parse_keep_steps(args.model_snapshot_steps)
    if model_snapshot_steps and not (args.experiment16_branch or args.experiment17_branch):
        raise ValueError("--model_snapshot_steps is reserved for Experiment 16A/17A")
    if args.fsdp and save_steps:
        raise ValueError("explicit --save_steps is only supported without FSDP")
    keep_steps = parse_keep_steps(args.keep_steps) | save_steps

    if args.run_name is None:
        args.run_name = f"scratch-{args.mode}-d{args.hidden_size}-L{args.num_layers}-{args.steps//1000}k"
    if args.out_dir is None:
        args.out_dir = f"./output/scratch-{args.mode}-d{args.hidden_size}-L{args.num_layers}-{args.steps//1000}k"

    # ── distributed ──
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    is_main = rank == 0

    if args.branch_from:
        verification = [None]
        if is_main:
            try:
                validate_branch_invocation(args, verify_parent_hash=True)
                verification[0] = {"ok": True}
            except Exception as exc:
                verification[0] = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        dist.broadcast_object_list(verification, src=0)
        if not verification[0]["ok"]:
            raise RuntimeError(
                "Experiment 3 branch verification failed on rank 0: "
                f"{verification[0]['error_type']}: {verification[0]['error']}")

    torch.manual_seed(args.seed + rank)

    continuation_checkpoint = (
        args.resume_from or args.branch_from or args.experiment16_parent
        or args.experiment17_parent
    )
    resume_state = (
        load_training_state(continuation_checkpoint)
        if continuation_checkpoint else None)
    run_identity = training_identity(args, world_size)
    accepted_source_change = None
    if args.resume_from and resume_state is not None:
        saved_branch = resume_state["run_identity"].get("branch")
        if saved_branch is not None:
            run_identity["branch"] = saved_branch
        saved_experiment16 = resume_state["run_identity"].get("experiment16")
        if saved_experiment16 is not None:
            if args.experiment16_branch != saved_experiment16.get("branch"):
                raise RuntimeError("Experiment 16A resume branch differs from checkpoint")
            run_identity["experiment16"] = saved_experiment16
        saved_experiment17 = resume_state["run_identity"].get("experiment17")
        if saved_experiment17 is not None:
            if args.experiment17_branch != saved_experiment17.get("branch"):
                raise RuntimeError("Experiment 17A resume branch differs from checkpoint")
            run_identity["experiment17"] = saved_experiment17
        saved_source_change = resume_state["run_identity"].get(
            "source_change_exception")
        if saved_source_change is not None:
            run_identity["source_change_exception"] = saved_source_change
    if (args.expected_global_batch is not None
            and run_identity["global_batch_size"] != args.expected_global_batch):
        raise ValueError(
            f"expected global batch {args.expected_global_batch}, got "
            f"{run_identity['global_batch_size']}")
    if resume_state is not None:
        if args.branch_from:
            validate_branch_parent_identity(resume_state["run_identity"], run_identity)
        elif args.experiment16_parent:
            validate_branch_parent_identity(resume_state["run_identity"], run_identity)
        elif args.experiment17_parent:
            validate_branch_parent_identity(resume_state["run_identity"], run_identity)
        elif resume_state["run_identity"] != run_identity:
            if not args.resume_source_change_record:
                raise RuntimeError(
                    "resume checkpoint identity does not match this invocation:\n"
                    f"saved={json.dumps(resume_state['run_identity'], sort_keys=True)}\n"
                    f"current={json.dumps(run_identity, sort_keys=True)}")
            accepted_source_change = validate_resume_source_change_exception(
                saved_identity=resume_state["run_identity"],
                current_identity=run_identity,
                checkpoint=args.resume_from,
                record_path=args.resume_source_change_record,
            )
            run_identity["source_change_exception"] = accepted_source_change
        elif args.resume_source_change_record:
            raise RuntimeError(
                "--resume_source_change_record was supplied but no source change exists")
    start_step = int(resume_state["global_step"]) if resume_state else 0
    target_step = args.stop_after_step or args.steps
    if args.resume_validation_only:
        if not args.resume_from or not args.resume_validation_output:
            raise ValueError(
                "--resume_validation_only requires --resume_from and "
                "--resume_validation_output")
        target_step = start_step
    elif not start_step < target_step <= args.steps:
        raise ValueError(
            "--stop_after_step must be greater than the starting step and no greater "
            f"than --steps; start={start_step}, target={target_step}, steps={args.steps}")
    if not args.resume_validation_only and any(
            step <= start_step or step > target_step for step in save_steps):
        raise ValueError("--save_steps must fall after the starting step and at/before the target")
    if any(step <= start_step or step > target_step for step in model_snapshot_steps):
        raise ValueError("--model_snapshot_steps must fall after the starting step and at/before the target")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_path = out_dir / "training_run_manifest.json"
    if run_manifest_path.exists():
        manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        manifest_identity_matches_saved = (
            accepted_source_change is not None
            and resume_state is not None
            and manifest["run_identity"] == resume_state["run_identity"]
        )
        if manifest["run_identity"] != run_identity and not manifest_identity_matches_saved:
            raise RuntimeError("output directory belongs to a different training run")
        if resume_state is None:
            raise RuntimeError(
                "output directory already contains a run manifest; pass --resume_from "
                "for that branch/run or choose a new --out_dir")
    else:
        manifest = {
            "format_version": 1,
            "created_at": utc_now(),
            "run_identity": run_identity,
            "command": sys.argv,
        }
        if is_main:
            atomic_write_json(run_manifest_path, manifest)
    if is_main and accepted_source_change is not None:
        continuation_manifest = out_dir / "source_change_resume_manifest.json"
        if continuation_manifest.exists():
            existing = json.loads(continuation_manifest.read_text(encoding="utf-8"))
            if existing.get("run_identity") != run_identity:
                raise RuntimeError("existing source-change resume manifest differs")
        else:
            atomic_write_json(continuation_manifest, {
                "format_version": 1,
                "created_at": utc_now(),
                "checkpoint": str(Path(args.resume_from).resolve()),
                "run_identity": run_identity,
                "command": sys.argv,
            })
    if is_main and resume_state is not None:
        truncate_metrics_after_step(out_dir / "training_metrics.jsonl", start_step)
        if args.experiment16_branch:
            truncate_metrics_after_step(out_dir / "experiment16_batch_stream.jsonl", start_step)
        if args.experiment17_branch:
            truncate_metrics_after_step(out_dir / "experiment17_batch_stream.jsonl", start_step)
        if args.experiment11_run:
            truncate_metrics_after_step(
                out_dir / "experiment11_weight_metrics.jsonl", start_step)
    dist.barrier()

    # ── W&B ──
    use_wandb = False
    saved_wandb_run_id = resume_state.get("wandb_run_id") if resume_state else None
    if args.resume_from and saved_wandb_run_id and args.wandb_run_id and saved_wandb_run_id != args.wandb_run_id:
        raise RuntimeError("requested W&B run ID differs from resumable checkpoint")
    if args.branch_from or args.experiment16_parent or args.experiment17_parent:
        wandb_run_id = args.wandb_run_id
    else:
        wandb_run_id = saved_wandb_run_id or args.wandb_run_id
    if is_main and not args.resume_validation_only:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                group=args.wandb_group,
                job_type="pretraining",
                tags=wandb_tags(args),
                id=wandb_run_id,
                resume="allow" if wandb_run_id else None,
                name=args.run_name,
                config={**vars(args), **run_identity},
            )
            wandb_run_id = wandb.run.id
            use_wandb = True
        except Exception as e:
            if args.wandb_required:
                raise
            print(f"W&B init failed ({e}), continuing without logging")
    if is_main:
        manifest["wandb"] = {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "group": args.wandb_group,
            "run_name": args.run_name,
            "run_id": wandb_run_id,
            "required": args.wandb_required,
        }
        if args.experiment11_run and use_wandb:
            import wandb
            manifest["wandb"]["run_url"] = wandb.run.url
        atomic_write_json(run_manifest_path, manifest)

    # ── model ──
    if is_main:
        print(f"Building {args.mode} model from scratch...")

    model = build_model(args, device)

    if args.compile and args.mode != "baseline":
        enable_attnres_compile()
        if is_main:
            print("torch.compile enabled for AttnRes kernels")

    if args.fused:
        if args.mode != "full_mh":
            raise ValueError("--fused currently supports --mode full_mh only")
        if args.mixed_partition:
            raise ValueError("--fused does not support --mixed_partition; use the eager path")
        from src.attention_residuals.modeling_qwen3_attnres import enable_fused_mhar
        enable_fused_mhar(True)
        if is_main:
            print("fused Triton MHAR routing kernels enabled")

    if args.grad_ckpt and not args.fsdp:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        if is_main:
            print("activation checkpointing enabled on decoder layers")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    if is_main:
        print(f"Model: {n_params:.1f}M params | mode={args.mode} | d={args.hidden_size} L={args.num_layers}")
        if getattr(args, "experiment25_variant", None):
            from src.experiments.experiment25_global_local import parameter_report
            report25 = parameter_report(model)
            manifest["experiment25_parameters"] = report25
            atomic_write_json(run_manifest_path, manifest)
            if use_wandb:
                import wandb
                wandb.config.update(report25, allow_val_change=False)
        if getattr(args, "experiment24_variant", None):
            from src.experiments.experiment24_three_route import parameter_report
            report24 = parameter_report(model)
            manifest["experiment24_parameters"] = report24
            atomic_write_json(run_manifest_path, manifest)
            if use_wandb:
                import wandb
                wandb.config.update(report24, allow_val_change=False)
        if getattr(args, "experiment23_variant", None):
            from src.experiments.experiment23_route_topology import parameter_report
            report23 = parameter_report(model)
            manifest["experiment23_parameters"] = report23
            atomic_write_json(run_manifest_path, manifest)
            if use_wandb:
                import wandb
                wandb.config.update(report23, allow_val_change=False)
        if args.mode != "baseline":
            n_attnres = sum(p.numel() for n, p in model.named_parameters() if "res_" in n)
            print(f"AttnRes params: {n_attnres/1e3:.1f}K")
        if args.experiment6_variant:
            from src.experiments.experiment6_coupled_qkv import experiment6_parameter_report
            experiment6_parameters = experiment6_parameter_report(model)
            print("Experiment 6 parameter report: " + json.dumps(
                experiment6_parameters, sort_keys=True))
            manifest["experiment6_parameters"] = experiment6_parameters
            atomic_write_json(run_manifest_path, manifest)
            if use_wandb:
                import wandb
                wandb.config.update(experiment6_parameters, allow_val_change=False)
        if args.experiment7_variant:
            from src.experiments.experiment7_local_q import experiment7_parameter_report
            experiment7_parameters = experiment7_parameter_report(model)
            print("Experiment 7 parameter report: " + json.dumps(
                experiment7_parameters, sort_keys=True))
            manifest["experiment7_parameters"] = experiment7_parameters
            atomic_write_json(run_manifest_path, manifest)
            if use_wandb:
                import wandb
                wandb.config.update(experiment7_parameters, allow_val_change=False)
        # Experiment 17 uses the HQ8 construction flag to build its parent
        # graph, but the expanded-global branch replaces HybridQueryLinear
        # before reporting. Its parameterization is validated and reported by
        # the dedicated Experiment 17 block below.
        if args.experiment8_variant and not args.experiment17_branch:
            from src.experiments.experiment8_hybrid_q import experiment8_parameter_report
            experiment8_parameters = experiment8_parameter_report(model)
            print("Experiment 8 parameter report: " + json.dumps(
                experiment8_parameters, sort_keys=True))
            manifest["experiment8_parameters"] = experiment8_parameters
            atomic_write_json(run_manifest_path, manifest)
            if use_wandb:
                import wandb
                wandb.config.update(experiment8_parameters, allow_val_change=False)
        if getattr(args, "experiment19_variant", None):
            from src.experiments.experiment19_role_specific_routing import (
                experiment19_parameter_report,
            )
            experiment19_parameters = experiment19_parameter_report(unwrap_model(model))
            print("Experiment 19 parameter report: " + json.dumps(
                experiment19_parameters, sort_keys=True))
            manifest["experiment19_parameters"] = experiment19_parameters
            atomic_write_json(run_manifest_path, manifest)
            if use_wandb:
                import wandb
                wandb.config.update(experiment19_parameters, allow_val_change=False)
        if getattr(args, "experiment20_route_count", None):
            from src.experiments.experiment20_multislot_routing import (
                experiment20_parameter_report,
            )
            experiment20_parameters = experiment20_parameter_report(
                unwrap_model(model))
            print("Experiment 20 parameter report: " + json.dumps(
                experiment20_parameters, sort_keys=True))
            manifest["experiment20_parameters"] = experiment20_parameters
            atomic_write_json(run_manifest_path, manifest)
            if use_wandb:
                import wandb
                wandb.config.update(experiment20_parameters, allow_val_change=False)
        if args.experiment11_run:
            from src.experiments.experiment11_soft_specialization import (
                experiment11_parameter_report,
            )
            experiment11_parameters = experiment11_parameter_report(model)
            print("Experiment 11 parameter report: " + json.dumps(
                experiment11_parameters, sort_keys=True))
            manifest["experiment11_parameters"] = experiment11_parameters
            atomic_write_json(run_manifest_path, manifest)
            if use_wandb:
                import wandb
                wandb.config.update(experiment11_parameters, allow_val_change=False)
        if args.experiment16_branch:
            from src.experiments.experiment16_causal_adaptation import validate_installed_mask
            validate_installed_mask(unwrap_model(model), args.experiment16_branch)
            experiment16_mask = {
                "branch": args.experiment16_branch,
                "masked_dense_q_parameters": sum(
                    layer.self_attn.q_proj.weight.numel()
                    for layer in unwrap_model(model).model.layers
                ),
                "trainable_parameter_count_unchanged": True,
                "systems_efficiency_claim": False,
            }
            manifest["experiment16_mask"] = experiment16_mask
            atomic_write_json(run_manifest_path, manifest)
            if use_wandb:
                import wandb
                wandb.config.update(experiment16_mask, allow_val_change=False)
        if args.experiment17_branch:
            from src.experiments.experiment17_reverse_adaptation import (
                ExpandedHybridQueryLinear,
                validate_expanded_query,
            )
            validate_expanded_query(unwrap_model(model), args.experiment17_branch)
            queries = [layer.self_attn.q_proj for layer in unwrap_model(model).model.layers]
            experiment17_expansion = {
                "branch": args.experiment17_branch,
                "cross_parameter_count": sum(
                    query.cross_weight.numel()
                    for query in queries if isinstance(query, ExpandedHybridQueryLinear)
                ),
                "cross_parameter_count_expected": (
                    36 * 8 * 80 * 1120 if args.experiment17_branch == "expanded-global" else 0
                ),
                "old_parameter_objects_preserved": True,
                "systems_efficiency_claim": False,
            }
            if experiment17_expansion["cross_parameter_count"] != experiment17_expansion["cross_parameter_count_expected"]:
                raise RuntimeError("Experiment 17A cross-parameter count mismatch")
            manifest["experiment17_expansion"] = experiment17_expansion
            atomic_write_json(run_manifest_path, manifest)
            if use_wandb:
                import wandb
                wandb.config.update(experiment17_expansion, allow_val_change=False)

    if args.experiment11_probe_artifact:
        if world_size != 1:
            raise ValueError("Experiment 11 step-0 probe requires one process per run")
        if is_main:
            from src.experiments.experiment11_workflow import write_step0_probe
            step0 = write_step0_probe(
                model,
                run_id=args.experiment11_run,
                artifact=Path(args.experiment11_probe_artifact),
                output=Path(args.experiment11_probe_output),
                training_identity=run_identity,
                device=device,
            )
            print(f"Experiment 11 step-0 probe: {args.experiment11_probe_output}")
            if use_wandb:
                import wandb
                wandb.log({
                    "experiment11/r_act": step0["activation_metrics"]["metrics"]["r_act"]["mean"],
                    "experiment11/theta_radians": step0["activation_metrics"]["metrics"]["theta_radians"]["mean"],
                }, step=0)
        dist.barrier()

    # torch.compile the full model before DDP wrapping.
    # Gives ~2.5-2.9x throughput improvement for all modes.
    if args.compile_model:
        model = torch.compile(model)
        if is_main:
            print("torch.compile enabled for full model")

    if args.fsdp:
        import functools
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy)
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        # Wrap each transformer decoder layer (works for baseline Qwen3 and AttnRes).
        layer_cls = type(model.model.layers[0])
        wrap_policy = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls={layer_cls})
        mp = MixedPrecision(param_dtype=torch.bfloat16,
                            reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
        model = FSDP(model, auto_wrap_policy=wrap_policy, mixed_precision=mp,
                     sharding_strategy=ShardingStrategy.FULL_SHARD,
                     device_id=local_rank, sync_module_states=True,
                     use_orig_params=True)
        if args.grad_ckpt:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                checkpoint_wrapper, CheckpointImpl, apply_activation_checkpointing)
            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=functools.partial(
                    checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
                check_fn=lambda m: isinstance(m, layer_cls))
            if is_main:
                print("activation checkpointing enabled on decoder layers")
        if is_main:
            print(f"FSDP FULL_SHARD enabled, wrapping {layer_cls.__name__}")
    else:
        if args.mode == "moe":
            # Plain default DDP: with topk*batch tokens every expert is hit
            # each micro-batch (all params used), and both find_unused and
            # static_graph deadlock against grad-accumulation's repeated
            # backwards here.
            model = DDP(model, device_ids=[local_rank])
        else:
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # ── optimizer ──
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8)
    lr_min_ratio = args.lr_min / args.lr
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda s: cosine_with_warmup(s, args.warmup, args.steps, lr_min_ratio),
    )
    if resume_state is not None:
        if args.experiment17_parent and args.experiment17_branch == "expanded-global":
            from src.experiments.experiment17_reverse_adaptation import migrate_hq8_optimizer_state
            migration = migrate_hq8_optimizer_state(
                resume_state["optimizer"], optimizer, model
            )
            if is_main:
                atomic_write_json(out_dir / "optimizer_state_migration.json", migration)
        else:
            optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])

    # ── data ──
    # Controlled parent checkpoints carry the exact tokenizer files used by
    # their accepted runs. Prefer that immutable local copy on rented
    # servers, where outbound Hugging Face access may be unavailable, while
    # retaining the canonical remote tokenizer/revision in run_identity.
    tokenizer_source = args.tokenizer
    tokenizer_revision = args.tokenizer_revision
    controlled_parent = args.experiment16_parent or args.experiment17_parent
    if controlled_parent:
        parent_tokenizer = Path(controlled_parent)
        required_tokenizer_files = ("tokenizer.json", "tokenizer_config.json")
        if all((parent_tokenizer / name).is_file() for name in required_tokenizer_files):
            tokenizer_source = str(parent_tokenizer)
            tokenizer_revision = None
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, revision=tokenizer_revision)
    resume_chunks = int(resume_state["chunks_consumed"]) if resume_state else 0
    stream = token_stream(args.dataset, args.dataset_name, tokenizer,
                          args.seq_len, rank, world_size, args.seed,
                          data_files=args.data_files,
                          dataset_revision=args.dataset_revision,
                          skip_chunks=resume_chunks)

    # ── validation data (use a different seed to avoid overlap) ──
    val_stream = None
    if args.eval_every > 0:
        val_stream = token_stream(args.dataset, args.dataset_name, tokenizer,
                                  args.seq_len, rank, world_size, args.seed + 9999,
                                  data_files=args.data_files,
                                  dataset_revision=args.dataset_revision)

    @torch.no_grad()
    def evaluate(val_iter, n_steps):
        model.eval()
        total_loss = 0.0
        count = 0
        for _ in range(n_steps):
            batch_chunks = []
            for _ in range(args.batch_size):
                try:
                    chunk = next(val_iter)
                    batch_chunks.append(chunk[:-1])
                except StopIteration:
                    break
            if not batch_chunks:
                break
            input_ids = torch.stack(batch_chunks).to(device)
            labels = input_ids
            out = model(input_ids=input_ids, labels=labels, use_cache=False)
            if any(getattr(args,k,None) for k in ("experiment23_variant","experiment24_variant","experiment25_variant")) and not torch.isfinite(out.loss):
                raise RuntimeError("Experiment 23 validation produced non-finite Loss")
            if args.mode == "moe":
                # out.loss includes the router aux term; report pure CE so val
                # is comparable to the dense baseline.
                import torch.nn.functional as F
                lg = out.logits[:, :-1].float()
                ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                     labels[:, 1:].reshape(-1))
                total_loss += ce.item()
            else:
                total_loss += out.loss.item()
            count += 1
        model.train()
        if any(getattr(args,k,None) for k in ("experiment23_variant","experiment24_variant","experiment25_variant")) and count != n_steps:
            raise RuntimeError("Experiment 23 validation stream ended before the frozen batch count")
        avg = torch.tensor(total_loss / max(count, 1), device=device)
        dist.all_reduce(avg, op=dist.ReduceOp.AVG)
        return avg.item()

    # ── training ──
    os.makedirs(args.out_dir, exist_ok=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    val_iter = iter(val_stream) if val_stream is not None else None

    global_step = start_step
    accum_step = 0
    accum_loss = 0.0
    t0 = time.time()
    tokens_seen = 0
    chunks_consumed = resume_chunks
    elapsed_before_resume = (
        float(resume_state.get("elapsed_training_seconds", 0.0)) if resume_state else 0.0)
    run_started = time.time() - elapsed_before_resume
    if resume_state is not None:
        torch.set_rng_state(resume_state["cpu_rng_state"])
        torch.cuda.set_rng_state_all(resume_state["cuda_rng_state_all"])
        if is_main:
            print(
                f"Resumed exact state at step {global_step}; replay-skipping "
                f"{resume_chunks} packed chunks", flush=True)

    if args.resume_validation_only:
        if val_iter is None or args.eval_every <= 0:
            raise RuntimeError("resume validation gate requires validation data")
        val_loss = evaluate(val_iter, args.eval_steps)
        result = {
            "format_version": 1,
            "created_at": utc_now(),
            "checkpoint": str(Path(args.resume_from).resolve()),
            "checkpoint_step": start_step,
            "evaluation_steps": args.eval_steps,
            "validation_loss": val_loss,
            "optimizer_updates": 0,
            "run_identity": run_identity,
        }
        if is_main:
            output = Path(args.resume_validation_output).resolve()
            if output.exists():
                raise RuntimeError(f"resume validation output already exists: {output}")
            atomic_write_json(output, result)
            print(
                f"Resume validation gate passed at step {start_step}: "
                f"val_loss={val_loss:.8f}", flush=True)
            if use_wandb:
                import wandb
                wandb.finish()
        dist.barrier()
        dist.destroy_process_group()
        return

    batch_buf = []
    controlled_branch = args.experiment17_branch or args.experiment16_branch
    controlled_step_batches = []
    controlled_stream_digest = "0" * 64
    controlled_stream_name = "experiment17" if args.experiment17_branch else "experiment16"
    if controlled_branch and is_main:
        stream_path = out_dir / f"{controlled_stream_name}_batch_stream.jsonl"
        if stream_path.is_file():
            existing_rows = [json.loads(line) for line in stream_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if existing_rows:
                controlled_stream_digest = existing_rows[-1]["cumulative_digest"]
    for chunk in stream:
        if global_step >= target_step:
            break
        if getattr(args, "experiment25_variant", None):
            from src.experiments.experiment25_budget import require_work_allowed
            from src.experiments.experiment25_workflow import require_disk_floor
            require_work_allowed(args.experiment25_budget_lock, args.experiment25_budget_heartbeat)
            require_disk_floor(args.out_dir)

        chunks_consumed += 1
        batch_buf.append(chunk[:-1])
        if len(batch_buf) < args.batch_size:
            continue

        input_ids = torch.stack(batch_buf).to(device)  # [batch_size, seq_len]
        if controlled_branch:
            controlled_step_batches.append(input_ids.detach().cpu())
        labels = input_ids
        batch_buf = []

        out = model(input_ids=input_ids, labels=labels, use_cache=False)
        loss = out.loss / args.grad_accum
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite loss at step={global_step} accum={accum_step}: {loss.item()}")
        loss.backward()

        accum_loss += loss.item()
        accum_step += 1
        tokens_seen += args.seq_len * args.batch_size

        if accum_step < args.grad_accum:
            continue

        if args.fsdp:
            grad_norm = model.clip_grad_norm_(args.max_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if any(getattr(args,k,None) for k in ("experiment23_variant","experiment24_variant","experiment25_variant")) and not torch.isfinite(grad_norm):
            raise FloatingPointError("Experiment 23 non-finite gradient norm; optimizer update refused")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1
        accum_step = 0

        if controlled_branch:
            if args.experiment17_branch:
                from src.experiments.experiment17_reverse_adaptation import stream_step_digest
            else:
                from src.experiments.experiment16_causal_adaptation import stream_step_digest
            local_step_digest = stream_step_digest(controlled_step_batches)
            gathered = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, local_step_digest)
            if is_main:
                controlled_stream_digest = hashlib.sha256(
                    bytes.fromhex(controlled_stream_digest)
                    + b"".join(bytes.fromhex(value) for value in gathered)
                ).hexdigest()
                append_jsonl(out_dir / f"{controlled_stream_name}_batch_stream.jsonl", {
                    "created_at": utc_now(), "step": global_step,
                    "step_rank_digests": gathered,
                    "cumulative_digest": controlled_stream_digest,
                    "chunks_consumed": chunks_consumed,
                })
            controlled_step_batches = []

        if global_step % args.log_every == 0:
            loss_t = torch.tensor(accum_loss, device=device)
            dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)

            if is_main:
                elapsed = time.time() - t0
                tok_sec = tokens_seen * world_size / elapsed
                avg_loss = loss_t.item()
                lr_now = scheduler.get_last_lr()[0]
                mem_gb = torch.cuda.max_memory_allocated() / 1e9
                print(f"step {global_step:6d} | loss {avg_loss:.4f} | "
                      f"lr {lr_now:.2e} | grad_norm {grad_norm:.3f} | "
                      f"{tok_sec/1e3:.1f}k tok/s | {mem_gb:.1f}GB")

                metric_row = {
                    "created_at": utc_now(),
                    "step": global_step,
                    "loss": avg_loss,
                    "lr": lr_now,
                    "grad_norm": float(grad_norm),
                    "tokens_per_second": tok_sec,
                    "peak_cuda_allocated_gib": (
                        torch.cuda.max_memory_allocated() / 2**30),
                    "chunks_consumed": chunks_consumed,
                    "tokens_consumed": (
                        global_step * run_identity["global_batch_size"] * args.seq_len),
                    "elapsed_hours": (time.time() - run_started) / 3600,
                    "seed": args.seed,
                    "branch_role": args.branch_role,
                    "mixed_partition": args.mixed_partition,
                }
                append_jsonl(out_dir / "training_metrics.jsonl", metric_row)

                if use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/lr": lr_now,
                        "train/grad_norm": grad_norm,
                        "train/tok_per_s": tok_sec,
                        "train/peak_cuda_allocated_gib": (
                            torch.cuda.max_memory_allocated() / 2**30),
                        "train/chunks_consumed": chunks_consumed,
                        "train/tokens_consumed": (
                            global_step * run_identity["global_batch_size"] * args.seq_len),
                        "train/elapsed_hours": (time.time() - run_started) / 3600,
                    }, step=global_step)

                tokens_seen = 0
                t0 = time.time()
        accum_loss = 0.0

        if is_main and args.experiment11_run and global_step % 100 == 0:
            from src.experiments.experiment11_soft_specialization import (
                weight_specialization_metrics,
            )
            weight_metrics = weight_specialization_metrics(unwrap_model(model))
            weight_row = {
                "created_at": utc_now(),
                "step": global_step,
                "run_id": args.experiment11_run,
                **weight_metrics,
            }
            append_jsonl(out_dir / "experiment11_weight_metrics.jsonl", weight_row)
            if use_wandb:
                import wandb
                wandb.log({
                    "experiment11/r_weight": sum(
                        row["r_weight"] for row in weight_metrics["rows"]
                    ) / len(weight_metrics["rows"]),
                }, step=global_step)

        if not args.fsdp and is_main and checkpoint_due(global_step, args.save_every, save_steps):
            ckpt_dir = save_training_checkpoint(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=global_step,
                chunks_consumed=chunks_consumed,
                run_identity=run_identity,
                out_dir=out_dir,
                keep_last=args.keep_last,
                keep_steps=keep_steps,
                wandb_run_id=wandb_run_id,
                elapsed_training_seconds=time.time() - run_started,
            )
            print(f"Saved checkpoint → {ckpt_dir}")
            if use_wandb:
                import wandb
                wandb.run.summary["latest_checkpoint"] = str(ckpt_dir)
                wandb.run.summary["latest_checkpoint_step"] = global_step

        if not args.fsdp and is_main and global_step in model_snapshot_steps:
            snapshot_dir = save_model_snapshot(
                model=model,
                tokenizer=tokenizer,
                global_step=global_step,
                chunks_consumed=chunks_consumed,
                run_identity=run_identity,
                out_dir=out_dir,
            )
            print(f"Saved model-only measurement snapshot → {snapshot_dir}")

        if args.eval_every > 0 and global_step % args.eval_every == 0 and val_iter is not None:
            val_loss = evaluate(val_iter, args.eval_steps)
            if is_main:
                import math as _math
                val_ppl = _math.exp(val_loss) if val_loss < 20 else float('inf')
                print(f"step {global_step:6d} | val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}")
                if use_wandb:
                    import wandb
                    wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=global_step)

    # ── final validation ──
    if args.eval_every > 0 and val_iter is not None:
        val_loss = evaluate(val_iter, args.eval_steps)
        if is_main:
            import math as _math
            val_ppl = _math.exp(val_loss) if val_loss < 20 else float('inf')
            print(f"FINAL   | val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}")
            if use_wandb:
                import wandb
                wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=global_step)

    if is_main:
        terminal_step_dir = out_dir / f"step-{global_step}"
        if args.reuse_step_checkpoint_as_final:
            required = (
                terminal_step_dir / "training_manifest.json",
                terminal_step_dir / "training_state.pt",
            )
            if not all(path.is_file() for path in required):
                raise RuntimeError(
                    "--reuse_step_checkpoint_as_final requires a complete terminal checkpoint")
            final_dir = terminal_step_dir
        else:
            final_dir = save_training_checkpoint(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=global_step,
                chunks_consumed=chunks_consumed,
                run_identity=run_identity,
                out_dir=out_dir,
                keep_last=args.keep_last,
                keep_steps=keep_steps,
                wandb_run_id=wandb_run_id,
                elapsed_training_seconds=time.time() - run_started,
                final=True,
            )
        print(f"Training done. Final model → {final_dir}")
        if use_wandb:
            import wandb
            wandb.run.summary["final_checkpoint"] = str(final_dir)
            wandb.run.summary["completed_steps"] = global_step
            wandb.run.summary["total_tokens"] = (
                global_step * run_identity["global_batch_size"] * args.seq_len)
            wandb.finish()

        if any(getattr(args, k, None) for k in ("experiment23_variant", "experiment24_variant", "experiment25_variant")):
            variant_key = ("experiment25_variant" if args.experiment25_variant else
                           "experiment24_variant" if args.experiment24_variant else "experiment23_variant")
            atomic_write_json(out_dir / "run_complete.json", {
                variant_key: getattr(args, variant_key),
                "run_identity": run_identity, "global_step": global_step,
                "chunks_consumed": chunks_consumed, "wandb_run_id": wandb_run_id,
                "checkpoint_sha256": sha256_tree(final_dir),
                "scheduled_and_final_validation_completed": True,
            })

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
