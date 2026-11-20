"""Atomic same-run checkpoints for the isolated training adapter.

This is a library, not a launcher, paid approval or scoring-release controller.
Caller retains the run identity and manifest SHA outside the checkpoint. No
primary token/mask/provenance file is opened. Fixture identities never become
production identities. Interrupted saves and stale locks require explicit audit.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import shutil
import subprocess
import sys
import uuid

import numpy as np
import torch
import transformers

from src.experiments import paper_sharing_data as data
from src.experiments.paper_sharing_model import (
    model_fingerprints, load_local_model_checkpoint, runtime_buffer_fingerprints,
)
from src.experiments.paper_sharing_protocol import run_id, sha
from src.experiments.paper_sharing_training import (
    ROOT, RECIPE_SHA256, SUPERVISION_SHA256, UpdateProgress, frozen_recipe,
    validate_training_model, validate_boundary, make_optimization, native_lr_multiplier,
)

SOURCE_FILES = [
    "src/experiments/paper_sharing_checkpoint.py", "src/experiments/paper_sharing_training.py",
    "src/experiments/paper_sharing_model.py", "src/experiments/paper_sharing_supervision.py",
    "src/experiments/paper_sharing_execution_policy.py",
    "src/experiments/paper_sharing_determinism.py",
    "src/experiments/paper_sharing_data.py", "src/experiments/paper_sharing_protocol.py",
    "src/experiments/experiment23_route_topology.py", "src/experiments/experiment24_three_route.py",
    "src/attention_residuals/modeling_qwen3_attnres.py", "src/training/train_scratch.py",
    "configs/paper_sharing_validation_2k/design.json",
    "configs/paper_sharing_validation_2k/training-recipe.frozen.json",
    "configs/paper_sharing_validation_2k/loss-supervision.frozen.json",
    "configs/paper_sharing_validation_2k/checkpoint-local-contract.json",
    "configs/paper_sharing_validation_2k/canonical-training-policy.v1.json",
    "configs/paper_sharing_validation_2k/strict-determinism-policy.v1.json",
]
MANIFEST = "checkpoint-manifest.json"
COMPLETE = "CHECKPOINT_COMPLETE.json"


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _tree_digest(value):
    """Device-independent hash retaining tensor dtype/shape and container/key types."""
    def encode(item):
        if isinstance(item, torch.Tensor):
            t = item.detach().cpu().contiguous()
            return ["tensor", str(t.dtype), list(t.shape),
                    hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()]
        if isinstance(item, dict):
            return ["dict", [[encode(k), encode(v)] for k, v in sorted(
                item.items(), key=lambda pair: (type(pair[0]).__name__, str(pair[0])))]]
        if isinstance(item, (list, tuple)):
            return [type(item).__name__, [encode(v) for v in item]]
        _require(type(item) in (str, int, float, bool, type(None)), "unsupported checkpoint state type")
        return [type(item).__name__, item]
    return _digest(encode(value))


def _source_snapshot():
    from src.experiments.paper_sharing_profile import source_files
    return {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
            "files": {p: data.file_sha256(ROOT / p) for p in [*SOURCE_FILES, *source_files()]}}


def _runtime_snapshot(local_fixture):
    device = []
    if not local_fixture:
        _require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
                 "production requires exactly one visible CUDA device per run")
        properties = torch.cuda.get_device_properties(0)
        device = [{"name": properties.name, "uuid": str(properties.uuid),
                   "capability": list(torch.cuda.get_device_capability(0))}]
    return {"python": sys.version.split()[0], "platform": platform.system(),
            "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
            "cuda": torch.version.cuda, "devices": device,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_autocast": torch.is_autocast_enabled("cuda"),
            "cpu_autocast": torch.is_autocast_enabled("cpu"),
            "cuda_autocast_dtype": str(torch.get_autocast_dtype("cuda")),
            "cpu_autocast_dtype": str(torch.get_autocast_dtype("cpu")),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def _config_digest(model):
    config = model.config.to_dict()
    for key in ("transformers_version", "torch_dtype", "dtype", "_name_or_path", "architectures"):
        config.pop(key, None)
    return _digest(config)


def _parameter_order(model):
    return [{"name": name, "shape": list(p.shape), "dtype": str(p.dtype)}
            for name, p in model.named_parameters() if p.requires_grad]


def _regular(path):
    path = Path(path)
    _require(not path.is_symlink() and path.is_file(), f"regular file required: {path.name}")
    return path


def _bundle_binding(bundle, seed, expected_manifest_sha256, local_fixture):
    """Metadata plus selected TRAINING files only; never inspect primary payload."""
    root = Path(bundle)
    _require(not root.is_symlink() and root.is_dir(), "regular bundle directory required")
    sha(expected_manifest_sha256)
    path = _regular(root / "manifest.json")
    _require(data.file_sha256(path) == expected_manifest_sha256, "data manifest hash mismatch")
    manifest = json.loads(path.read_text())
    marker = json.loads(_regular(root / "COMPLETE.json").read_text())
    _require(marker == {"manifest_sha256": expected_manifest_sha256}, "data bundle incomplete")
    _require(manifest["synthetic_fixture"] is local_fixture, "fixture/production data identity mismatch")
    policy_path = _regular(root / "policy.json")
    policy = json.loads(policy_path.read_text())
    _require(data.digest(policy) == manifest["policy_sha256"], "data policy binding mismatch")
    if not local_fixture:
        from src.experiments import paper_sharing_profile as profile
        contract = json.loads((ROOT / "configs/paper_sharing_validation_2k/loss-supervision.frozen.json").read_text())
        frozen_policy = ROOT / "configs/paper_sharing_validation_2k/gate-b-policy.frozen.json"
        if profile.current():
            profile.verify_bundle_binding(policy, expected_manifest_sha256)
        else:
            _require(expected_manifest_sha256 == contract["bundle_manifest_sha256"]
                 and data.file_sha256(frozen_policy) == contract["data_policy_file_sha256"]
                 and json.loads(frozen_policy.read_text()) == policy, "unfrozen production data")
    stream = manifest["training_streams"][str(seed)]
    _require(stream["sequence_length"] == 1024 and type(stream["sequences"]) is int
             and stream["sequences"] > 0 and (local_fixture or stream["sequences"] == 64000),
             "changed training stream dimensions")
    directory = root / f"training-seed{seed}"
    _require(not directory.is_symlink() and directory.is_dir(), "regular training directory required")
    _require(set(stream["files"]) == {"tokens.u32le", "valid-target-mask.u8", "provenance.jsonl"},
             "changed training file inventory")
    for name, expected in stream["files"].items():
        _require(data.file_sha256(_regular(directory / name)) == sha(expected), "training stream file hash mismatch")
    count = stream["sequences"]
    _require((directory / "tokens.u32le").stat().st_size == count * 1025 * 4
             and (directory / "valid-target-mask.u8").stat().st_size == count * 1024,
             "training file dimensions mismatch")
    for condition in ("AAAB", "AABC", "ABBC"):
        _require(manifest["run_streams"][run_id(condition, seed)] == {
            "seed": seed, "directory": f"training-seed{seed}",
            "tokens_sha256": stream["files"]["tokens.u32le"]}, "within-seed alias mismatch")
    return {"manifest_sha256": expected_manifest_sha256, "policy_file_sha256": data.file_sha256(policy_path),
            "policy_sha256": manifest["policy_sha256"], "exclusion_sha256": manifest["exclusion_sha256"],
            "tokenizer": policy["tokenizer"], "dataset_revision": policy["dataset_revision"],
            "directory": directory.name, "sequences": count, "sequence_length": 1024, "files": stream["files"]}


def create_run_identity(model, progress, *, bundle, expected_data_manifest_sha256,
                        wandb_run_id, microbatch=4, accumulation=8):
    """Bind actual step-zero state; pairing/launch authorization remain external gates."""
    _require(type(progress) is UpdateProgress and not progress.failed
             and type(progress.completed_updates) is int and progress.completed_updates == 0
             and type(progress.chunks_consumed) is int and progress.chunks_consumed == 0,
             "identity must be frozen at complete step zero")
    condition = validate_training_model(model, local_fixture=progress.local_fixture)
    _require(progress.condition == condition, "condition mismatch")
    identifier = run_id(condition, progress.seed)
    _require(isinstance(wandb_run_id, str) and re.fullmatch(r"[a-z0-9]{8}", wandb_run_id), "explicit stable W&B ID required")
    _require(all(type(n) is int and n > 0 for n in (microbatch, accumulation))
             and (progress.local_fixture or (microbatch, accumulation) == (4, 8)), "changed production batch policy")
    _require(all(p.grad is None for p in model.parameters()), "step-zero gradients not clean")
    source = _source_snapshot()
    _require(progress.local_fixture or not source["dirty"], "dirty production execution source")
    frozen_recipe()
    result = {"format_version": 1, "study_id": "paper-sharing-fresh-2k-v1", "run_id": identifier,
            "condition": condition, "seed": progress.seed, "local_fixture": progress.local_fixture,
            "source": source, "runtime": _runtime_snapshot(progress.local_fixture),
            "recipe_sha256": RECIPE_SHA256, "supervision_sha256": SUPERVISION_SHA256,
            "model_config_sha256": _config_digest(model), "initialization": model_fingerprints(model),
            "parameter_order": _parameter_order(model),
            "data": _bundle_binding(bundle, progress.seed, expected_data_manifest_sha256, progress.local_fixture),
            "microbatch": microbatch, "accumulation": accumulation, "stop_after_update": 2000,
            "schedule_steps": 20000, "wandb_run_id": wandb_run_id}
    from src.experiments.paper_sharing_execution_policy import policy_binding
    amendment = policy_binding(model.config)
    if amendment is not None:
        result["execution_policy"] = amendment
    from src.experiments.paper_sharing_determinism import runtime_binding
    strict = runtime_binding(model.config, check_runtime=True)
    if strict is not None:
        result["runtime_policy"] = strict
    return result


def _check_identity(identity):
    _require(type(identity) is dict and type(identity.get("local_fixture")) is bool, "explicit run identity required")
    _require(identity["format_version"] == 1 and identity["study_id"] == "paper-sharing-fresh-2k-v1"
             and identity["run_id"] == run_id(identity["condition"], identity["seed"])
             and identity["recipe_sha256"] == RECIPE_SHA256 and identity["supervision_sha256"] == SUPERVISION_SHA256
             and identity["stop_after_update"] == 2000 and identity["schedule_steps"] == 20000,
             "changed run/recipe/supervision/schedule identity")
    _require(identity["source"] == _source_snapshot(), "execution source changed")
    _require(identity["local_fixture"] or not identity["source"]["dirty"], "dirty production source")
    _require(identity["runtime"] == _runtime_snapshot(identity["local_fixture"]), "runtime/backend/device changed")
    _require(identity["local_fixture"] or (identity["microbatch"], identity["accumulation"]) == (4, 8),
             "changed production batch policy")
    frozen_recipe()
    if "execution_policy" in identity:
        from src.experiments.paper_sharing_execution_policy import POLICY_ID, POLICY_SHA256, canonical_policy
        canonical_policy()
        _require(identity["execution_policy"] == {"policy_id": POLICY_ID, "sha256": POLICY_SHA256},
                 "changed checkpoint execution-policy binding")
    if "runtime_policy" in identity:
        from src.experiments.paper_sharing_determinism import RUNTIME_POLICY_ID, RUNTIME_POLICY_SHA256, require_runtime
        _require(identity["runtime_policy"] == {"policy_id": RUNTIME_POLICY_ID, "sha256": RUNTIME_POLICY_SHA256},
                 "changed checkpoint strict-runtime binding")
        require_runtime(fixture=identity["local_fixture"])


def _fsync_dir(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path, value):
    with Path(path).open("xb") as handle:
        handle.write(_canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def register_run(root, identity):
    """Write-once identity root. Not an active-worker reservation/launch grant."""
    _check_identity(identity)
    root = Path(root)
    _require(not root.is_symlink(), "symlink run root refused")
    root.mkdir(parents=True, exist_ok=True)
    _require(not any(root.iterdir()), "run root already occupied; no duplicate registration")
    _write_json(root / "RUN_IDENTITY.json", identity)
    _fsync_dir(root)
    return _digest(identity)


@contextmanager
def _writer_lock(root):
    lock = root / ".checkpoint-writer.lock"
    lock.mkdir()  # Never steal/remove a stale lock.
    try:
        _write_json(lock / "owner.json", {"pid": os.getpid()})
        yield
    finally:
        (lock / "owner.json").unlink()
        lock.rmdir()


def _capture_rng(local_fixture):
    numpy = np.random.get_state()
    return {"torch_cpu": torch.get_rng_state(),
            "torch_cuda": [] if local_fixture else torch.cuda.get_rng_state_all(),
            "python": random.getstate(),
            "numpy": [numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]]}


def _check_rng(rng, local_fixture):
    _require(set(rng) == {"torch_cpu", "torch_cuda", "python", "numpy"}, "missing RNG state")
    state = rng["torch_cpu"]
    _require(isinstance(state, torch.Tensor) and state.dtype == torch.uint8
             and state.shape == torch.get_rng_state().shape, "invalid CPU RNG state")
    torch.Generator(device="cpu").set_state(state)
    _require(isinstance(rng["torch_cuda"], list) and len(rng["torch_cuda"]) == (0 if local_fixture else 1),
             "CUDA RNG device count mismatch")
    if not local_fixture:
        torch.Generator(device="cuda:0").set_state(rng["torch_cuda"][0])
    random.Random().setstate(rng["python"])
    n = rng["numpy"]
    _require(type(n) is list and len(n) == 5 and n[0] == "MT19937" and len(n[1]) == 624
             and all(type(v) is int and 0 <= v < 2**32 for v in n[1]), "invalid NumPy RNG state")
    np.random.RandomState(0).set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))


def _restore_rng(rng, local_fixture):
    torch.set_rng_state(rng["torch_cpu"])
    if not local_fixture:
        torch.cuda.set_rng_state_all(rng["torch_cuda"])
    random.setstate(rng["python"])
    n = rng["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))


def _file_inventory(directory):
    result = {}
    for path in sorted(directory.rglob("*")):
        _require(not path.is_symlink(), "checkpoint symlink refused")
        if path.is_file() and path.name not in (MANIFEST, COMPLETE):
            result[path.relative_to(directory).as_posix()] = {"bytes": path.stat().st_size, "sha256": data.file_sha256(path)}
    return result


def verify_checkpoint(directory, *, expected_manifest_sha256, expected_identity):
    """Verify bytes BEFORE tensor deserialization. Caller hash is mandatory."""
    sha(expected_manifest_sha256)
    root = Path(directory)
    _require(not root.is_symlink() and root.is_dir() and re.fullmatch(r"step-[1-9][0-9]*", root.name),
             "published step directory required; partial directories cannot resume")
    path = _regular(root / MANIFEST)
    _require(data.file_sha256(path) == expected_manifest_sha256, "checkpoint manifest hash mismatch")
    marker = json.loads(_regular(root / COMPLETE).read_text())
    _require(marker == {"manifest_sha256": expected_manifest_sha256}, "checkpoint completion marker mismatch")
    manifest = json.loads(path.read_text())
    _require(manifest["format_version"] == 1 and manifest["identity_sha256"] == _digest(expected_identity)
             and manifest["identity"] == expected_identity, "checkpoint run identity mismatch")
    _require(root.name == f'step-{manifest["completed_updates"]}', "checkpoint step/path mismatch")
    _require(_file_inventory(root) == manifest["files"], "checkpoint file inventory/size/hash mismatch")
    _require({"config.json", "model.safetensors", "training-state.pt"} <= set(manifest["files"]),
             "missing full model/optimizer checkpoint files")
    _require(isinstance(manifest.get("runtime_buffers"), dict) and bool(manifest["runtime_buffers"]),
             "checkpoint lacks runtime-buffer binding; legacy weights cannot prove exact execution")
    return manifest


def _check_optimizer_state(state, order, step):
    _require(set(state) == {"state", "param_groups"} and len(state["param_groups"]) == 1,
             "invalid optimizer structure")
    ids = state["param_groups"][0]["params"]
    _require(ids == list(range(len(order))) and all(type(i) is int for i in ids), "optimizer parameter order changed")
    _require(bool(state["state"]) and set(state["state"]) <= set(ids), "missing/foreign optimizer state")
    for index, values in state["state"].items():
        _require(set(values) == {"step", "exp_avg", "exp_avg_sq"}, "incomplete AdamW moments")
        counter = values["step"]
        _require(isinstance(counter, torch.Tensor) and counter.ndim == 0 and counter.dtype == torch.float32
                 and counter.device.type == "cpu" and counter.item() == step,
                 "optimizer state/update mismatch")
        for key in ("exp_avg", "exp_avg_sq"):
            tensor = values[key]
            _require(isinstance(tensor, torch.Tensor) and list(tensor.shape) == order[index]["shape"]
                     and str(tensor.dtype) == order[index]["dtype"] and bool(torch.isfinite(tensor).all()),
                     "optimizer moment shape/dtype/nonfinite mismatch")
        _require(bool((values["exp_avg_sq"] >= 0).all()), "negative AdamW squared moment")


def _check_scheduler_state(state, step):
    lr = 5e-4 * native_lr_multiplier(step)
    _require(type(state.get("last_epoch")) is int and state["last_epoch"] == step
             and type(state.get("_step_count")) is int and state["_step_count"] == step + 1
             and state.get("base_lrs") == [5e-4] and state.get("_last_lr") == [lr]
             and state.get("lr_lambdas") == [None], "native scheduler clock/LR state mismatch")


def _check_progress(progress, identity):
    _require(type(progress) is UpdateProgress and type(progress.completed_updates) is int
             and 0 < progress.completed_updates <= 2000 and type(progress.chunks_consumed) is int
             and progress.chunks_consumed == progress.completed_updates * identity["microbatch"] * identity["accumulation"]
             and progress.condition == identity["condition"] and type(progress.seed) is int and progress.seed == identity["seed"]
             and progress.local_fixture is identity["local_fixture"] and progress.failed is False,
             "checkpoint progress/run/data position mismatch")
    _require(identity["local_fixture"] or progress.completed_updates % 100 == 0, "production checkpoint cadence changed")
    _require(progress.chunks_consumed <= identity["data"]["sequences"], "checkpoint past frozen training stream")


def _prune(root, identity, latest):
    """Only same-run verified ordinary checkpoints, after new durable publication."""
    candidates = []
    for path in root.iterdir():
        match = re.fullmatch(r"step-([1-9][0-9]*)", path.name)
        if match is None:
            continue
        step = int(match.group(1))
        _require(step <= latest, "refuse rollback/pruning with a newer checkpoint present")
        digest = data.file_sha256(_regular(path / MANIFEST))
        verify_checkpoint(path, expected_manifest_sha256=digest, expected_identity=identity)
        if step not in (latest, 500, 2000):
            candidates.append((path, digest))
    for path, digest in candidates:
        ledger = root / "retention.jsonl"
        _require(not ledger.is_symlink(), "retention ledger symlink refused")
        with ledger.open("a") as handle:
            handle.write(_canonical({"event": "delete_planned", "name": path.name,
                "manifest_sha256": digest, "new_complete_step": latest}).decode() + "\n")
            handle.flush(); os.fsync(handle.fileno())
        shutil.rmtree(path)  # Exact validated checkpoint inside caller's run root only.
        _fsync_dir(root)
        with ledger.open("a") as handle:
            handle.write(_canonical({"event": "deleted", "name": path.name, "manifest_sha256": digest}).decode() + "\n")
            handle.flush(); os.fsync(handle.fileno())


def save_checkpoint(root, *, model, optimizer, scheduler, progress, identity,
                    tokenizer_snapshot=None, prune=True):
    _check_identity(identity)
    root = Path(root)
    _require(not root.is_symlink() and root.is_dir(), "registered regular run root required")
    _require(json.loads(_regular(root / "RUN_IDENTITY.json").read_text()) == identity, "registered run identity mismatch")
    _check_progress(progress, identity)
    validate_boundary(model, optimizer, scheduler, progress,
                      microbatch=identity["microbatch"], accumulation=identity["accumulation"])
    _require(_config_digest(model) == identity["model_config_sha256"]
             and _parameter_order(model) == identity["parameter_order"], "model config or parameter binding changed")
    from src.experiments.paper_sharing_execution_policy import policy_binding
    _require(policy_binding(model.config) == identity.get("execution_policy"), "checkpoint execution policy missing or changed")
    from src.experiments.paper_sharing_determinism import runtime_binding
    _require(runtime_binding(model.config, check_runtime=True) == identity.get("runtime_policy"), "checkpoint runtime policy missing or changed")
    _require(type(prune) is bool and (identity["local_fixture"] or prune), "production latest-only retention required")
    with _writer_lock(root):
        destination = root / f"step-{progress.completed_updates}"
        _require(not destination.exists() and not destination.is_symlink(), "checkpoint already exists; no overwrite")
        _require(not any(re.fullmatch(r"step-([1-9][0-9]*)", p.name)
                         and int(p.name[5:]) > progress.completed_updates for p in root.iterdir()), "out-of-order checkpoint save")
        partial = root / f".step-{progress.completed_updates}.partial-{uuid.uuid4().hex}"
        partial.mkdir()
        fingerprints = model_fingerprints(model)
        _require(all(not tensor.is_floating_point() or bool(torch.isfinite(tensor).all())
                     for tensor in model.state_dict().values()), "nonfinite model tensors/buffers")
        rng = _capture_rng(identity["local_fixture"])
        state = {"format_version": 1, "identity_sha256": _digest(identity), "progress": asdict(progress),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": rng,
                 "parameter_order": _parameter_order(model)}
        _check_optimizer_state(state["optimizer"], state["parameter_order"], progress.completed_updates)
        _check_scheduler_state(state["scheduler"], progress.completed_updates)
        _check_rng(rng, identity["local_fixture"])
        model.save_pretrained(partial, safe_serialization=True)
        if not identity["local_fixture"]:
            snapshot = Path(tokenizer_snapshot) if tokenizer_snapshot is not None else None
            _require(snapshot is not None and snapshot.is_dir(), "verified tokenizer snapshot required")
            for name, expected in identity["data"]["tokenizer"]["files"].items():
                _require(Path(name).name == name and data.file_sha256(_regular(snapshot / name)) == sha(expected),
                         "tokenizer snapshot hash mismatch")
                shutil.copyfile(snapshot / name, partial / name)
        torch.save(state, partial / "training-state.pt")
        _check_identity(identity)
        manifest = {"format_version": 1, "identity": identity, "identity_sha256": _digest(identity),
                    "completed_updates": progress.completed_updates, "chunks_consumed": progress.chunks_consumed,
                    "model_fingerprints": fingerprints, "state_tree_sha256": _tree_digest(state),
                    "runtime_buffers": runtime_buffer_fingerprints(model),
                    "files": _file_inventory(partial)}
        for path in partial.rglob("*"):
            if path.is_file():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
        _write_json(partial / MANIFEST, manifest)
        manifest_sha = data.file_sha256(partial / MANIFEST)
        _write_json(partial / COMPLETE, {"manifest_sha256": manifest_sha})
        _fsync_dir(partial)
        _require(not destination.exists(), "checkpoint destination collision")
        os.rename(partial, destination)
        _fsync_dir(root)
        verify_checkpoint(destination, expected_manifest_sha256=manifest_sha, expected_identity=identity)
        if prune:
            _prune(root, identity, progress.completed_updates)
        return {"directory": str(destination), "manifest_sha256": manifest_sha,
                "identity_sha256": _digest(identity), "completed_updates": progress.completed_updates}


def verified_training_iterator(bundle, identity, *, skip_chunks):
    """Verify/replay just the bound training files; no primary payload access."""
    observed = _bundle_binding(bundle, identity["seed"], identity["data"]["manifest_sha256"], identity["local_fixture"])
    _require(observed == identity["data"], "training data identity changed")
    _require(type(skip_chunks) is int and 0 <= skip_chunks <= observed["sequences"], "invalid training resume offset")
    directory = Path(bundle) / observed["directory"]
    # Full stream hash/real provenance audit is bound above and from Gate B;
    # replay checks dimensions, token range and causal continuity locally.
    for index, raw in enumerate(data._raw_chunks(directory / "tokens.u32le", 1024,
                                                observed["sequences"], skip_chunks), start=skip_chunks):
        yield {"chunk_index": index, "input_ids": raw[:-1], "targets": raw[1:],
               "valid_target_mask": [True] * 1024}


@dataclass
class ResumedTraining:
    model: object
    optimizer: object
    scheduler: object
    progress: UpdateProgress
    examples: object
    identity: dict


def _load_exact_checkpoint(directory, *, expected_manifest_sha256, expected_identity, bundle):
    """Validate identity/bytes/state, build model/optimizer, restore RNG LAST.

    No silent source exception, step skipping or scientific retry. Callers must
    separately authorize resuming scientific execution and reserve the worker.
    """
    _check_identity(expected_identity)
    manifest = verify_checkpoint(directory, expected_manifest_sha256=expected_manifest_sha256,
                                 expected_identity=expected_identity)
    observed = _bundle_binding(bundle, expected_identity["seed"], expected_identity["data"]["manifest_sha256"],
                               expected_identity["local_fixture"])
    _require(observed == expected_identity["data"], "resume training data mismatch")
    state = torch.load(Path(directory) / "training-state.pt", map_location="cpu", weights_only=True)
    _require(set(state) == {"format_version", "identity_sha256", "progress", "optimizer", "scheduler", "rng", "parameter_order"}
             and state["format_version"] == 1 and state["identity_sha256"] == _digest(expected_identity)
             and _tree_digest(state) == manifest["state_tree_sha256"], "serialized training state mismatch")
    progress = UpdateProgress(**state["progress"])
    _check_progress(progress, expected_identity)
    _require(progress.completed_updates == manifest["completed_updates"]
             and progress.chunks_consumed == manifest["chunks_consumed"]
             and state["parameter_order"] == expected_identity["parameter_order"], "state/manifest counters or parameter order mismatch")
    _check_optimizer_state(state["optimizer"], state["parameter_order"], progress.completed_updates)
    _check_scheduler_state(state["scheduler"], progress.completed_updates)
    _check_rng(state["rng"], expected_identity["local_fixture"])
    model = load_local_model_checkpoint(directory, expected_condition=expected_identity["condition"],
        expected_state_sha256=manifest["model_fingerprints"]["checkpoint_state_sha256"])
    _require(_config_digest(model) == expected_identity["model_config_sha256"]
             and model_fingerprints(model) == manifest["model_fingerprints"], "loaded model/config mismatch")
    from src.experiments.paper_sharing_execution_policy import policy_binding
    _require(policy_binding(model.config) == expected_identity.get("execution_policy"), "restored execution policy missing or changed")
    from src.experiments.paper_sharing_determinism import runtime_binding
    _require(runtime_binding(model.config, check_runtime=True) == expected_identity.get("runtime_policy"), "restored runtime policy missing or changed")
    model.to(device="cpu" if expected_identity["local_fixture"] else "cuda:0")
    _require(runtime_buffer_fingerprints(model) == manifest["runtime_buffers"],
             "restored runtime buffers differ from saved execution state")
    optimizer, scheduler = make_optimization(model, local_fixture=expected_identity["local_fixture"])
    _require(_parameter_order(model) == state["parameter_order"], "optimizer parameter order mismatch")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    _require(_tree_digest(optimizer.state_dict()) == _tree_digest(state["optimizer"])
             and _tree_digest(scheduler.state_dict()) == _tree_digest(state["scheduler"]), "optimizer/scheduler restore changed state")
    validate_boundary(model, optimizer, scheduler, progress,
                      microbatch=expected_identity["microbatch"], accumulation=expected_identity["accumulation"])
    _check_identity(expected_identity)
    # Restore only after all model construction, verification and data setup.
    _restore_rng(state["rng"], expected_identity["local_fixture"])
    return ResumedTraining(model, optimizer, scheduler, progress,
        verified_training_iterator(bundle, expected_identity, skip_chunks=progress.chunks_consumed), expected_identity)


def load_exact_checkpoint(directory, *, expected_manifest_sha256, expected_identity, bundle):
    """Exact load; preserve the caller's RNG if any validation/construction fails."""
    _check_identity(expected_identity)
    before = _capture_rng(expected_identity["local_fixture"])
    try:
        return _load_exact_checkpoint(directory, expected_manifest_sha256=expected_manifest_sha256,
                                      expected_identity=expected_identity, bundle=bundle)
    except BaseException:
        _restore_rng(before, expected_identity["local_fixture"])
        raise
