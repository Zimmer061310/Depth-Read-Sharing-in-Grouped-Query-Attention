"""Experiment 24 binding, finite matrix, resource locks and explicit launch."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import shutil
from pathlib import Path

from src.experiments.experiment1_partition_compatibility import sha256_file
from src.experiments.experiment21_evaluation import read_json, require_clean_commit
from src.experiments.experiment23_workflow import write_new, validate_training_data_identity
from src.experiments.experiment24_three_route import RUNS, SUBSTITUTIONS, training_id

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "configs/experiment24/run_matrix.json"
PROTOCOL = ROOT / "configs/experiment24/protocol.json"
LOCK = ROOT / "results/experiment23/provenance/data_lock.json"
HISTORICAL = ROOT / "results/experiment23"


def spec():
    matrix = read_json(MATRIX)
    rows = matrix["new_training_runs"]
    if (len(rows) != 2 or {r["id"]: r["routes"] for r in rows}
            != {training_id(v): routes for v, routes in RUNS.items()}):
        raise ValueError("changed two-run scope or duplicate alias")
    if matrix["expected_counts"]["total_fixed_evaluation_records"] != 34:
        raise ValueError("changed evaluation count")
    return matrix


def conditions():
    m = spec()
    rows = [{"id": r["id"], "parent": r["id"], "kind": "native", "imported": True}
            for r in m["reused_native_references"]]
    incumbent = m["analysis"]["primary_reference"]
    rows.append({"id": incumbent+"-collapse-b", "parent": incumbent,
                 "kind": "frozen-route-collapse", "imported": True})
    for r in m["new_training_runs"]:
        rows.append({"id": r["id"], "parent": r["id"], "kind": "native", "imported": False})
        rows.extend({"id": r["id"]+"-"+suffix, "parent": r["id"],
                     "kind": suffix, "imported": False} for suffix in SUBSTITUTIONS)
    rows.extend({"id": r["id"], "parent": incumbent, "kind": "value-interpolation",
                 "coefficient": r["lambda"], "imported": False}
                for r in m["evaluation"]["aaab_value_interpolation"]["new_conditions"])
    if len(rows) != 17 or len({r["id"] for r in rows}) != 17:
        raise ValueError("incomplete/duplicate fixed condition matrix")
    return rows


def condition_row(condition):
    for row in conditions():
        if row["id"] == condition:
            return row
    raise ValueError(f"unknown/alias condition: {condition}")


def variant_for_id(parent):
    for v in RUNS:
        if training_id(v) == parent:
            return v
    if parent in {r["id"] for r in spec()["reused_native_references"]}:
        return parent.removeprefix("exp23-").removesuffix("-seed42-step2000")
    raise ValueError("unknown parent")


def train_arguments(variant, *, output, data_files, resume=None, disposable=False):
    from src.experiments.experiment23_workflow import train_arguments as inherited
    training_id(variant)
    # Reuse the exact native recipe, not a second hand-maintained recipe.
    args = inherited("global-aaab", output=output, data_files=data_files,
                     resume=resume, disposable=disposable)
    for flag in ("--experiment23_variant", "--experiment23_router_backend",
                 "--experiment23_projection_backend"):
        i = args.index(flag)
        del args[i:i+2]
    args += ["--experiment24_variant", variant, "--experiment24_router_backend", "tiled-r2-exact"]
    args[args.index("--run_name")+1] = training_id(variant)
    args[args.index("--wandb_group")+1] = spec()["wandb"]["group"]
    return args


def identity_for_arguments(arguments, *, data_identity=None):
    from src.experiments.experiment23_workflow import identity_for_arguments as identity
    return identity(arguments, data_identity=data_identity)


def binding_identity():
    return {"source_commit": require_clean_commit(), "protocol_sha256": sha256_file(PROTOCOL),
            "matrix_sha256": sha256_file(MATRIX), "data_lock_sha256": sha256_file(LOCK),
            "artifact_sha256": spec()["evaluation"]["artifact_sha256"]}


def verify_execution_lock(path, variant, arguments):
    from scripts.train.preflight_experiment24 import environment
    lock, bindings = read_json(path), binding_identity()
    if any(lock.get(k) != v for k, v in bindings.items()):
        raise ValueError("execution lock changed source/data/protocol")
    if lock.get("variant") != variant or lock.get("training_identity") != identity_for_arguments(arguments):
        raise ValueError("execution lock changed recipe/stream")
    validate_training_data_identity(lock["training_identity"]["data_files"])
    gates = lock.get("readiness_gate_hashes", {})
    if set(gates) != set(read_json(PROTOCOL)["cuda_gates"]):
        raise ValueError("missing readiness gates")
    for name, gate in gates.items():
        report = read_json(gate["path"])
        if (sha256_file(Path(gate["path"])) != gate["sha256"] or report.get("gate") != name
            or report.get("passed") is not True or any(report.get(k) != v for k,v in bindings.items())):
            raise ValueError(f"failed/stale/unbound gate: {name}")
        if any(report.get("environment",{}).get(k)!=v for k,v in lock.get("environment",{}).items()
               if k not in ("device_uuid","visible_devices")):
            raise ValueError(f"gate environment differs from execution lock: {name}")
    env = environment()
    if lock.get("initializer_and_stream_certificate")!=read_json(gates["initializer_stream_alias"]["path"]):
        raise ValueError("missing/changed initializer/stream certificate")
    for key in env.keys() - {"device_uuid", "visible_devices"}:
        if lock.get("environment", {}).get(key) != env[key]:
            raise ValueError(f"runtime changed: {key}")
    return lock


def launch(variant, *, output, data_files, execution_lock, resume=None):
    args = train_arguments(variant, output=output, data_files=data_files, resume=resume)
    verify_execution_lock(execution_lock, variant, args)
    verify_launch_resources(output)
    output = Path(output)
    if resume is not None and Path(resume).resolve().parent != output.resolve():
        raise ValueError("resume must belong to this run")
    if resume is None and output.exists() and any(output.iterdir()):
        raise FileExistsError("fresh training root must be empty; log outside it")
    env = dict(os.environ, WANDB_MODE="online", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    if not env.get("HF_HOME") or not Path(env["HF_HOME"]).is_dir():
        raise ValueError("verify/export the offline tokenizer cache before launch")
    subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=1",
        "--master_port", env.get("MHAR_MASTER_PORT", str(30424+list(RUNS).index(variant))),
        "--module", "src.training.train_scratch", *args], cwd=ROOT, env=env, check=True)


def verify_launch_resources(output):
    """Two new runs only; no Exp23 three-control index dependency."""
    directory = Path(output).resolve().parent
    while not directory.exists(): directory = directory.parent
    occupied = 0
    for variant in RUNS:
        run = directory/training_id(variant)
        if run.is_symlink(): raise ValueError("training root may not be a symlink")
        for checkpoint in run.glob("step-*"):
            if checkpoint.is_symlink() or not checkpoint.is_dir(): raise ValueError("invalid checkpoint storage")
            manifest = checkpoint/"training_manifest.json"
            if not manifest.is_file() or not (checkpoint/"training_state.pt").is_file(): continue
            if read_json(manifest)["run_identity"].get("experiment24_variant") != variant:
                raise ValueError("stored checkpoint belongs to a different run")
            for file in checkpoint.rglob("*"):
                if file.is_symlink(): raise ValueError("checkpoint contains a symlink")
                if file.is_file(): occupied += file.stat().st_size
    if shutil.disk_usage(directory).free < max(20*1024**3,65*1024**3-occupied):
        raise RuntimeError("reserve 65 GiB for two runs including existing complete checkpoints, minimum 20 GiB free")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit(): raise RuntimeError("select exactly one physical GPU per process")
    used = subprocess.check_output(["nvidia-smi","-i",visible,"--query-gpu=memory.used",
        "--format=csv,noheader,nounits"],text=True).strip()
    if int(used)>1024: raise RuntimeError("GPU occupied; do not reset unrelated work")
