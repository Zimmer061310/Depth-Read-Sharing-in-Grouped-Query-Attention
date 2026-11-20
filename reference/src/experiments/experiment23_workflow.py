"""Frozen manifest, production arguments, and fail-closed execution binding."""
from __future__ import annotations

import json
import os
from pathlib import Path

from src.experiments.experiment1_partition_compatibility import sha256_file, atomic_write_json
from src.experiments.experiment21_evaluation import read_json, require_clean_commit
from src.experiments.experiment23_route_topology import RUNS, training_id

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "configs/experiment23/run_matrix.json"
PROTOCOL = ROOT / "configs/experiment23/protocol.json"
LOCK = ROOT / "results/experiment21/provenance/data_lock.json"


def spec():
    matrix = read_json(MATRIX)
    expected = {training_id(v): (features, routes) for v, (features, routes) in RUNS.items()}
    rows = matrix["training_runs"]
    observed = {r["id"]: (r["features"], r["routes"]) for r in rows}
    if len(rows) != 8 or observed != expected:
        raise ValueError("duplicate, missing, alias, or changed training condition")
    new = matrix.get("new_training_variants", [])
    reused = matrix.get("reused_control_variants", [])
    if (new != ["global-abaa", "global-aaba", "global-aaab", "global-aabb", "global-abba"]
        or reused != ["global-aaaa", "r2-gl-abaa", "hq8-gl-aaaa"]):
        raise ValueError("changed five-training/three-reuse execution scope")
    conditions = matrix["evaluation"]["conditions"]
    expected_conditions = {(key, key, "native") for key in expected}
    expected_conditions |= {(key + "-collapse-b", key, "frozen-route-collapse")
                            for key, (_, routes) in expected.items() if "B" in routes}
    if len(conditions) != 14 or {(r["id"], r["parent"], r["kind"]) for r in conditions} != expected_conditions:
        raise ValueError("changed evaluation matrix")
    return matrix


def variant_for_id(run_id):
    for variant in RUNS:
        if training_id(variant) == run_id:
            return variant
    raise ValueError(f"unknown training run: {run_id}")


def validate_training_data_identity(data):
    accepted = read_json(ROOT / "results/experiment8/provenance/step-2000/hq8/training_run_manifest.json")
    def ordered(rows):
        return [(Path(r["path"]).name,r["bytes"],r["sha256"]) for r in rows]
    if ordered(data.get("files",[])) != ordered(accepted["run_identity"]["data_files"]["files"]):
        raise ValueError("ordered training shard bytes differ from the accepted native stream")


def train_arguments(variant, *, output, data_files, resume=None, disposable=False):
    training_id(variant)
    protocol = read_json(PROTOCOL)
    args = ["--experiment23_variant", variant, "--experiment23_router_backend", protocol["router_backend"],
        "--experiment23_projection_backend", protocol["projection_backend"],
        "--mode", "full_mh", "--attnres_heads", "8", "--hidden_size", "1280",
        "--num_layers", "36", "--num_heads", "16", "--num_kv_heads", "8",
        "--intermediate_size", "5120", "--seq_len", "1024", "--steps", "20000",
        "--stop_after_step", "2000", "--batch_size", "4", "--grad_accum", "8",
        "--expected_global_batch", "32", "--lr", "5e-4", "--lr_min", "5e-5",
        "--warmup", "1000", "--max_norm", "1.0", "--seed", "42",
        "--dataset", "HuggingFaceFW/fineweb-edu", "--dataset_name", "sample-10BT",
        "--dataset_revision", "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "--data_files", str(data_files), "--tokenizer", "Qwen/Qwen3-0.6B",
        "--tokenizer_revision", "c1899de289a04d12100db370d81485cdf75e47ca",
        "--grad_ckpt", "--fused", "--save_every", "100", "--keep_last", "1",
        "--keep_steps", "500,2000", "--reuse_step_checkpoint_as_final",
        "--eval_every", "500", "--eval_steps", "50", "--log_every", "10",
        "--out_dir", str(output), "--wandb_project", "MHAR Stuff",
        "--wandb_entity", "zimmer061310-ena", "--wandb_group", "mhar-exp23-global-route-topology-seed42",
        "--run_name", training_id(variant)]
    if not disposable:
        args.append("--wandb_required")
    if resume is not None:
        args += ["--resume_from", str(resume)]
    return args


def identity_for_arguments(arguments, *, data_identity=None):
    import sys
    from src.training import train_scratch
    saved = sys.argv
    try:
        sys.argv = ["experiment23", *arguments]
        args = train_scratch.parse_args()
        return train_scratch.training_identity(args, 1, data_identity=data_identity)
    finally:
        sys.argv = saved


def binding_identity():
    return {"source_commit": require_clean_commit(), "protocol_sha256": sha256_file(PROTOCOL),
            "matrix_sha256": sha256_file(MATRIX), "data_lock_sha256": sha256_file(LOCK),
            "artifact_sha256": spec()["evaluation"]["artifact_sha256"]}


def verify_execution_lock(path, variant, arguments):
    lock = read_json(path)
    bindings = binding_identity()
    if any(lock.get(k) != v for k, v in bindings.items()):
        raise ValueError("execution lock source/protocol/data identity changed")
    if lock.get("variant") != variant or lock.get("training_identity") != identity_for_arguments(arguments):
        raise ValueError("execution lock training recipe or stream changed")
    validate_training_data_identity(lock["training_identity"]["data_files"])
    required = set(read_json(PROTOCOL)["cuda_gates"])
    gates = lock.get("readiness_gate_hashes", {})
    if set(gates) != required:
        raise ValueError("incomplete CUDA readiness gates")
    for name, row in gates.items():
        p = Path(row["path"])
        if sha256_file(p) != row["sha256"]:
            raise ValueError(f"readiness gate modified: {name}")
        report = read_json(p)
        if (report.get("gate") != name or report.get("passed") is not True
            or any(report.get(k) != v for k, v in bindings.items())):
            raise ValueError(f"unbound or failed readiness gate: {name}")
    if not lock.get("environment") or not lock.get("initializer_and_stream_certificate"):
        raise ValueError("missing runtime/initializer/stream certificate")
    from scripts.train.preflight_experiment23 import environment
    deployed = environment()
    # Equivalent idle cards may share a preflight certificate, but not a
    # different runtime, GPU model, VRAM capacity, or backend configuration.
    for key in deployed.keys() - {"device_uuid", "visible_devices"}:
        if lock["environment"].get(key) != deployed[key]:
            raise ValueError(f"execution runtime/device changed: {key}")
    return lock


def launch(variant, *, output, data_files, execution_lock, resume=None):
    """Called only by a separately authorized launch, never automatically."""
    import subprocess
    import sys
    if variant not in spec()["new_training_variants"]:
        raise ValueError("accepted control must be reused, not retrained")
    from src.experiments.experiment23_reuse import verify_launch_resources
    verify_launch_resources(output)
    arguments = train_arguments(variant, output=output, data_files=data_files, resume=resume)
    verify_execution_lock(execution_lock, variant, arguments)
    output = Path(output)
    if resume is not None and Path(resume).resolve().parent != output.resolve():
        raise ValueError("resume must use this run's own output directory")
    if resume is None and output.exists() and any(output.iterdir()):
        raise FileExistsError("fresh training output must be empty")
    env = dict(os.environ, WANDB_MODE="online", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    port = int(env.get("MHAR_MASTER_PORT", 30323 + list(RUNS).index(variant)))
    subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=1",
        "--master_port", str(port), "--module", "src.training.train_scratch", *arguments],
        cwd=ROOT, env=env, check=True)


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    atomic_write_json(path, value)
