"""Reuse accepted control bytes, never historical Losses or fabricated run IDs.

The read-only adapter preserves every Parameter and its enumeration order so
disposable timing restores the original optimizer state by the original slots.
An inactive B aliases A; it adds neither parameters nor optimizer slots.
"""
from __future__ import annotations

import copy
import os
import shutil
import subprocess
from pathlib import Path

import torch

from src.experiments.experiment1_partition_compatibility import sha256_path, sha256_file
from src.experiments.experiment21_evaluation import read_json
from src.experiments.experiment23_workflow import (
    spec, binding_identity, validate_training_data_identity, train_arguments,
    identity_for_arguments, write_new, PROTOCOL, MATRIX,
)

CONTROLS = {
    "global-aaaa": {"model": "m8", "checkpoint_sha256":
        "903b7fdba36caab0eff95b4c7844e2f13488c3df03530b6bc1bf0ef2a99f51e1",
        "original_run_id": "env3cmhq"},
    "hq8-gl-aaaa": {"model": "hq8", "checkpoint_sha256":
        "74cff0ab19409dac9f6104e8986e4890c0837bc730d8ac233ae18011dbc58333",
        "original_run_id": "6jjcjmnh"},
    "r2-gl-abaa": {"model": "exp21-r2", "checkpoint_sha256":
        "35b2de712d5d5a254ca5720e596dec68a2f30a9e450e56ab308c6e050f673613",
        "original_run_id": "9lu46pmg"},
}


def native_class(variant):
    if variant == "global-aaaa":
        from src.attention_residuals.modeling_qwen3_attnres import Qwen3AttnResForCausalLM
        return Qwen3AttnResForCausalLM
    if variant == "hq8-gl-aaaa":
        from src.experiments.experiment8_hybrid_q import Experiment8MHARForCausalLM
        return Experiment8MHARForCausalLM
    if variant == "r2-gl-abaa":
        from src.experiments.experiment21_request_conditioned import Experiment21MHARForCausalLM
        return Experiment21MHARForCausalLM
    raise ValueError("not an accepted reused control")


def adapt_control(net, variant):
    """Transient execution adapter; no tensor copies or checkpoint writes."""
    from src.experiments.experiment23_route_topology import (
        Experiment23MHARForCausalLM, Experiment23Attention, RUNS,
    )
    if variant not in CONTROLS:
        raise ValueError("not a reused control")
    if not isinstance(net, native_class(variant)):
        raise TypeError("wrong native control class")
    before = [(n, id(p)) for n, p in net.named_parameters()]
    c = copy.deepcopy(net.config)
    if variant == "r2-gl-abaa" and getattr(c, "experiment21_variant", None) != "r2":
        raise ValueError("only the parameter-free Experiment21 R2 control may be reused")
    for flag in ("experiment8_hybrid_q_groups", "experiment19_role_router",
                 "experiment20_multislot_router", "experiment21_request", "experiment22_contrast"):
        setattr(c, flag, False)
    c.experiment23_route_topology = True
    c.experiment23_variant = variant
    c.experiment23_router_backend = "r2-triton"
    c.experiment23_projection_backend = "packed"
    net.config = net.model.config = c
    routes = RUNS[variant][1]
    for layer in net.model.layers:
        if "B" not in routes:
            # Module sharing is deduplicated by named_parameters, unlike
            # creating a dummy B parameter which would break optimizer restore.
            layer.local_attn_res_proj = layer.attn_res_proj
        elif not hasattr(layer, "local_attn_res_proj"):
            raise ValueError("R2 checkpoint has no independent local router")
        layer._experiment23_route_topology = True
        layer._experiment23_active_b = "B" in routes
        layer._experiment23_collapse = False
        layer._experiment23_router_backend = "r2-triton"
        layer.self_attn.__class__ = Experiment23Attention
        layer.self_attn._experiment23_routes = routes
        layer.self_attn._experiment23_projection = "packed"
        layer.self_attn.config = c
    net.__class__ = Experiment23MHARForCausalLM
    if before != [(n, id(p)) for n, p in net.named_parameters()]:
        raise RuntimeError("control conversion changed optimizer parameter slots")
    return net


def metadata(checkpoint, variant):
    checkpoint = Path(checkpoint)
    accepted = CONTROLS[variant]
    digest = sha256_path(checkpoint)
    if digest != accepted["checkpoint_sha256"]:
        raise ValueError("control checkpoint is not the exact accepted byte tree")
    manifest = read_json(checkpoint / "training_manifest.json")
    identity = manifest["run_identity"]
    expected = identity_for_arguments(train_arguments(variant, output="/unused", data_files="/unused"),
                                      data_identity=identity["data_files"])
    keys = ("seed", "mode", "hidden_size", "num_layers", "num_heads", "num_kv_heads",
            "intermediate_size", "attnres_heads", "seq_len", "global_batch_size", "grad_accum",
            "per_gpu_batch_size", "world_size", "steps", "warmup", "lr", "lr_min",
            "optimizer", "optimizer_betas", "optimizer_eps", "weight_decay", "precision",
            "grad_ckpt", "fsdp", "dataset", "dataset_name", "dataset_revision",
            "tokenizer", "tokenizer_revision", "max_norm")
    if any(identity.get(k) != expected[k] for k in keys):
        raise ValueError("historical control does not match the frozen native recipe")
    validate_training_data_identity(identity["data_files"])
    if (manifest.get("global_step") != 2000 or manifest.get("chunks_consumed") != 64000
        or manifest.get("wandb_run_id") != accepted["original_run_id"]):
        raise ValueError("historical checkpoint step/stream/W&B identity changed")
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False, mmap=True)
    if (state.get("run_identity") != identity or state.get("global_step") != 2000
        or state.get("chunks_consumed") != 64000 or not state.get("optimizer", {}).get("state")
        or state.get("scheduler", {}).get("last_epoch") != 2000
        or not isinstance(state.get("cpu_rng_state"), torch.Tensor)
        or len(state.get("cuda_rng_state_all", [])) != 1
        or state.get("wandb_run_id") != accepted["original_run_id"]):
        raise ValueError("incomplete historical training state")
    del state
    return {"checkpoint_sha256": digest, "training_identity": identity,
            "original_model": accepted["model"], "original_wandb_run_id": accepted["original_run_id"]}


def validate_gate(gate, variant, checkpoint_hash, *, source=None):
    if (gate.get("variant") != variant or gate.get("checkpoint_sha256") != checkpoint_hash
        or gate.get("checkpoint_sha256") != CONTROLS[variant]["checkpoint_sha256"]
        or gate.get("protocol_sha256") != sha256_file(PROTOCOL)
        or gate.get("matrix_sha256") != sha256_file(MATRIX)
        or gate.get("artifact_sha256") != spec()["evaluation"]["artifact_sha256"]
        or gate.get("passed") is not True or gate.get("discovery_sequences") != 512
        or gate.get("parameter_objects_and_order_preserved") is not True
        or not 0 <= gate.get("max_per_sequence_loss_difference", float("inf")) <= 0.0005
        or len(gate.get("source_commit", "")) != 40
        or (source is not None and gate.get("source_commit") != source)):
        raise ValueError("missing, stale, failed, or mismatched control reuse gate")


def reuse_index():
    path = os.environ.get("MHAR_EXP23_REUSE_INDEX")
    if not path:
        raise ValueError("provide MHAR_EXP23_REUSE_INDEX with all three accepted checkpoints/gates")
    index = read_json(path)
    if set(index) != set(CONTROLS):
        raise ValueError("all three reused controls are required; no automatic retraining")
    return index


def validated_control(checkpoint, variant):
    info = metadata(checkpoint, variant)
    row = reuse_index()[variant]
    if Path(row["checkpoint"]).resolve() != Path(checkpoint).resolve():
        raise ValueError("control checkpoint differs from launch-frozen reuse index")
    gate = read_json(row["gate"])
    from src.experiments.experiment23_source_exception import live_exception
    exception = live_exception(gate.get("source_commit", ""))
    validate_gate(gate, variant, info["checkpoint_sha256"], source=gate.get("source_commit", ""))
    return {**info, "control_reuse": gate,
            **({"postprocess_source_exception":exception} if exception is not None else {})}


def load_model(checkpoint, variant, *, dtype, device):
    from src.experiments.experiment23_route_topology import Experiment23MHARForCausalLM
    if variant in CONTROLS:
        net = native_class(variant).from_pretrained(checkpoint, dtype=dtype, attn_implementation="sdpa")
        net = adapt_control(net, variant)
    else:
        net = Experiment23MHARForCausalLM.from_pretrained(checkpoint, dtype=dtype,
                                                        attn_implementation="sdpa")
    return net.to(device)


def verify_launch_resources(output):
    # Covers all five protected checkpoint trees and concurrent atomic saves.
    directory = Path(output).resolve().parent
    while not directory.exists():
        directory = directory.parent
    # In a second wave, this reservation includes checkpoints already written
    # by the same five frozen jobs; do not reserve their occupied bytes twice.
    occupied = 0
    for variant in spec()["new_training_variants"]:
        run = directory / ("exp23-" + variant + "-seed42-step2000")
        if run.is_symlink():
            raise RuntimeError("training storage may not use symlinked run roots")
        for checkpoint in run.glob("step-*"):
            if checkpoint.is_symlink() or not checkpoint.is_dir():
                raise RuntimeError("invalid checkpoint storage target")
            manifest = checkpoint / "training_manifest.json"
            if not manifest.is_file() or not (checkpoint / "training_state.pt").is_file():
                continue  # Atomic/incomplete writes are never credited.
            identity = read_json(manifest).get("run_identity", {})
            if identity.get("experiment23_variant") != variant:
                raise RuntimeError("checkpoint storage identity mismatch")
            for file in checkpoint.rglob("*"):
                if file.is_symlink():
                    raise RuntimeError("checkpoint storage contains a symlink")
                if file.is_file():
                    occupied += file.stat().st_size
    required = max(30 * 1024**3, 115 * 1024**3 - occupied)
    if shutil.disk_usage(directory).free < required:
        raise RuntimeError("five-run launch needs at least 115 GiB free on the output disk")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit():
        raise RuntimeError("select exactly one physical GPU for each training job")
    used = subprocess.check_output(["nvidia-smi", "-i", visible,
        "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip()
    if int(used) > 1024:
        raise RuntimeError("training GPU is occupied; do not reset or kill unrelated work")
    for variant, row in reuse_index().items():
        validated_control(row["checkpoint"], variant)


def certify_control(*, variant, checkpoint, artifact, output, device):
    from src.experiments.experiment23_evaluation import artifact_identity, sequence_metrics
    info = metadata(checkpoint, variant)
    payload, _ = artifact_identity(artifact)
    before = sha256_path(checkpoint)
    net = native_class(variant).from_pretrained(checkpoint, dtype=torch.bfloat16,
                                               attn_implementation="sdpa").to(device).eval()
    baseline = sequence_metrics(net, payload["discovery_input_ids"], batch_size=1, device=device)
    net = adapt_control(net, variant).eval()
    converted = sequence_metrics(net, payload["discovery_input_ids"], batch_size=1, device=device)
    delta = max(abs(a-b) for a,b in zip(baseline["sequence_losses"], converted["sequence_losses"]))
    if delta > 0.0005 or sha256_path(checkpoint) != before:
        raise RuntimeError(f"control conversion parity/checkpoint integrity failed: {delta}")
    gate = {**binding_identity(), **info, "variant": variant, "passed": True,
        "discovery_sequences": 512, "max_per_sequence_loss_difference": delta,
        "parameter_objects_and_order_preserved": True,
        "historical_initialization_fingerprint_available": False,
        "interpretation": "matched recorded recipe, not a newly paired training replicate"}
    validate_gate(gate, variant, info["checkpoint_sha256"])
    write_new(output, gate)
    return gate
