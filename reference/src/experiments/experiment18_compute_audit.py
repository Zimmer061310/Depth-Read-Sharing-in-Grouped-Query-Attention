#!/usr/bin/env python3
"""Experiment 18 native M8/HQ8 compute and timing audit.

This module deliberately does not modify either model graph.  It provides the
frozen analytic ledger, immutable benchmark fixture, graph checks, profiler
workers, and paired same-GPU timing harness used by the controller.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.attention_residuals import mhar_triton
from src.attention_residuals.modeling_qwen3_attnres import (
    Qwen3AttnResForCausalLM,
    enable_fused_mhar,
)
from src.experiments.experiment1_partition_compatibility import (
    append_jsonl,
    atomic_write_json,
    git_commit,
    load_fixed_eval_artifact,
    sha256_file,
    sha256_path,
    utc_now,
)
from src.experiments.experiment8_hybrid_q import (
    Experiment8MHARForCausalLM,
    HybridQueryLinear,
)
from src.training.train_scratch import cosine_with_warmup


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiment18/protocol.json"
MODELS = ("m8", "hq8")
TIMING_MODES = ("forward", "forward-backward", "optimizer-step")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_spec(path: Path = CONFIG) -> dict[str, Any]:
    spec = read_json(path)
    if spec.get("training") is not False or (spec.get("seed"), spec.get("milestone")) != (42, 2000):
        raise ValueError("Experiment 18 must remain a no-training seed-42 step-2000 audit")
    arch = spec["architecture"]
    expected = {
        "hidden_size": 1280,
        "num_layers": 36,
        "num_query_heads": 16,
        "num_kv_heads": 8,
        "head_dimension": 80,
        "intermediate_size": 5120,
        "vocab_size": 151936,
        "mhar_groups": 8,
        "mhar_chunk_width": 160,
        "mhar_routing_calls": 73,
        "mhar_total_source_count": 2701,
    }
    if arch != expected:
        raise ValueError(f"Experiment 18 architecture contract changed: {arch}")
    jobs = spec.get("logical_jobs", [])
    if len(jobs) != 20 or len(set(jobs)) != 20:
        raise ValueError("Experiment 18 requires exactly 20 unique logical jobs")
    if spec["execution"].get("required_gpu_count") != 1:
        raise ValueError("Experiment 18 requires paired sequential work on one GPU")
    return spec


def protocol_sha256(path: Path = CONFIG) -> str:
    return sha256_file(path)


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    dtype = str(value.dtype).encode()
    shape = json.dumps(list(value.shape), separators=(",", ":")).encode()
    digest.update(len(dtype).to_bytes(8, "big")); digest.update(dtype)
    digest.update(len(shape).to_bytes(8, "big")); digest.update(shape)
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def balanced_orders(pair_count: int, seed: int) -> list[list[str]]:
    if pair_count < 2 or pair_count % 2:
        raise ValueError("paired block count must be a positive even number")
    orders = [["m8", "hq8"]] * (pair_count // 2) + [["hq8", "m8"]] * (pair_count // 2)
    random.Random(seed).shuffle(orders)
    return orders


def runtime_manifest() -> dict[str, Any]:
    spec = load_spec()
    primary = spec["benchmark"]["primary"]
    context = spec["benchmark"]["context"]
    payload = {
        "format_version": 1,
        "experiment": "18-native-compute-audit",
        "source_commit": git_commit(),
        "protocol_sha256": protocol_sha256(),
        "logical_jobs": spec["logical_jobs"],
        "primary_orders": balanced_orders(primary["paired_blocks"], primary["order_seed"]),
        "context_orders": balanced_orders(context["paired_blocks"], context["order_seed"]),
        "models": spec["models"],
        "fixed_evaluation": spec["fixed_evaluation"],
        "counter_metric_candidates": spec["hardware_counters"]["preferred_metric_groups"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["content_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _forward_rows(model_id: str, batch_size: int, sequence_length: int,
                  attention_convention: str) -> list[dict[str, Any]]:
    spec = load_spec(); arch = spec["architecture"]
    if model_id not in MODELS:
        raise ValueError(f"unknown model: {model_id}")
    if attention_convention not in spec["analytic"]["attention_conventions"]:
        raise ValueError(f"unknown attention convention: {attention_convention}")
    b, t = int(batch_size), int(sequence_length)
    if b < 1 or t < 2:
        raise ValueError("ledger requires batch >=1 and sequence length >=2")
    d, layers = arch["hidden_size"], arch["num_layers"]
    hq, hkv, dh, ff, vocab = (
        arch["num_query_heads"], arch["num_kv_heads"],
        arch["head_dimension"], arch["intermediate_size"], arch["vocab_size"],
    )
    q_per_token = (
        d * hq * dh if model_id == "m8"
        else hkv * d * dh + hkv * arch["mhar_chunk_width"] * dh
    )
    positions = t * t if attention_convention == "dense_square_equivalent" else t * (t + 1) // 2
    rows = [
        ("q_projection", layers * b * t * q_per_token, True, "matrix_multiply"),
        ("k_projection", layers * b * t * d * hkv * dh, True, "matrix_multiply"),
        ("v_projection", layers * b * t * d * hkv * dh, True, "matrix_multiply"),
        ("attention_qk", layers * b * hq * positions * dh, True, "matrix_multiply"),
        ("attention_av", layers * b * hq * positions * dh, True, "matrix_multiply"),
        ("o_projection", layers * b * t * hq * dh * d, True, "matrix_multiply"),
        ("mlp_gate_up_down", layers * b * t * 3 * d * ff, True, "matrix_multiply"),
        # Source counts 1..72 are inside checkpointed decoder layers.
        ("mhar_layer_routing_score_and_mix", b * t * 2 * sum(range(1, 73)) * d, True, "dot_product"),
        # The final source count 73 is outside the decoder-layer checkpoints.
        ("mhar_final_routing_score_and_mix", b * t * 2 * 73 * d, False, "dot_product"),
        ("lm_output_projection", b * t * d * vocab, False, "matrix_multiply"),
    ]
    return [
        {
            "model": model_id,
            "batch_size": b,
            "sequence_length": t,
            "attention_convention": attention_convention,
            "component": component,
            "forward_macs": int(macs),
            "inside_decoder_checkpoint": checkpointed,
            "operation_kind": kind,
        }
        for component, macs, checkpointed, kind in rows
    ]


def analytic_ledger(model_id: str, batch_size: int, sequence_length: int,
                    mode: str, attention_convention: str) -> list[dict[str, Any]]:
    """Return the declared counted-MAC ledger for one execution mode.

    Checkpointed transformer contractions execute once in the initial forward,
    once in backward recomputation, and twice for their two matmul gradients.
    Non-checkpointed contractions execute once forward and twice backward.
    Elementwise and transcendental work remains an explicit unconverted
    inventory in the published metadata.
    """

    if mode not in TIMING_MODES:
        raise ValueError(f"unknown mode: {mode}")
    rows = _forward_rows(model_id, batch_size, sequence_length, attention_convention)
    accumulation = load_spec()["benchmark"]["primary"]["gradient_accumulation"] if mode == "optimizer-step" else 1
    for row in rows:
        if mode == "forward":
            factor = 1
        else:
            factor = 4 if row["inside_decoder_checkpoint"] else 3
        row["mode"] = mode
        row["gradient_accumulation"] = int(accumulation)
        row["execution_factor"] = int(factor * accumulation)
        row["counted_macs"] = int(row["forward_macs"] * factor * accumulation)
        row["analytic_flops"] = int(row["counted_macs"] * 2)
    return rows


def ledger_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot summarize an empty ledger")
    return {
        "model": materialized[0]["model"],
        "batch_size": materialized[0]["batch_size"],
        "sequence_length": materialized[0]["sequence_length"],
        "mode": materialized[0]["mode"],
        "attention_convention": materialized[0]["attention_convention"],
        "counted_macs": sum(row["counted_macs"] for row in materialized),
        "analytic_flops": sum(row["analytic_flops"] for row in materialized),
        "components": len(materialized),
        "unconverted_inventory": [
            "embedding lookup", "RMSNorm", "RoPE", "causal masking", "softmax",
            "SiLU", "cross entropy", "gradient clipping", "AdamW", "scheduler",
            "reshape/index/stack", "memory copies",
        ],
    }


def all_ledgers() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spec = load_spec(); rows, summaries = [], []
    shapes = [(4, 1024, mode) for mode in TIMING_MODES]
    shapes += [
        (1, length, mode)
        for length in spec["benchmark"]["context"]["sequence_lengths"]
        for mode in ("forward", "forward-backward")
    ]
    for model_id in MODELS:
        for batch, length, mode in shapes:
            for convention in spec["analytic"]["attention_conventions"]:
                part = analytic_ledger(model_id, batch, length, mode, convention)
                rows.extend(part); summaries.append(ledger_summary(part))
    return rows, summaries


def write_ledger(output_root: Path) -> None:
    rows, summaries = all_ledgers()
    target = output_root / "data/derived/operator_ledger.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    atomic_write_json(output_root / "data/derived/operator_ledger_summary.json", {
        "format_version": 1,
        "created_at": utc_now(),
        "protocol_sha256": protocol_sha256(),
        "summaries": summaries,
    })


def build_fixture(artifact: Path, output: Path) -> dict[str, Any]:
    spec = load_spec(); payload, observed = load_fixed_eval_artifact(artifact)
    if observed != spec["fixed_evaluation"]["sha256"]:
        raise ValueError("Experiment 18 fixed-evaluation artifact hash mismatch")
    source = payload["discovery_input_ids"].to(torch.int64).contiguous()
    if source.shape[0] < 4 or source.shape[1] != 1024:
        raise ValueError(f"unexpected discovery tensor shape: {tuple(source.shape)}")
    tensors = {
        "b4_t1024": source[:4].clone(),
        "b1_t0256": source[:1, :256].clone(),
        "b1_t0512": source[:1, :512].clone(),
        "b1_t1024": source[:1].clone(),
        "b1_t2048": torch.cat((source[0], source[1])).reshape(1, 2048).contiguous(),
    }
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite fixture: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save({"format_version": 1, "tensors": tensors}, temporary)
    os.replace(temporary, output)
    manifest = {
        "format_version": 1,
        "created_at": utc_now(),
        "artifact_sha256": observed,
        "fixture_path": str(output),
        "fixture_sha256": sha256_file(output),
        "tensors": {
            name: {
                "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                "sha256": tensor_sha256(tensor), "valid_tokens": int(tensor.numel()),
            }
            for name, tensor in tensors.items()
        },
        "t2048_semantics": "systems-only concatenation of discovery sequences 0 then 1",
    }
    atomic_write_json(output.with_suffix(output.suffix + ".manifest.json"), manifest)
    return manifest


def load_fixture(path: Path, batch_size: int, sequence_length: int) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported Experiment 18 fixture")
    key = f"b{batch_size}_t{sequence_length:04d}"
    try:
        tensor = payload["tensors"][key]
    except KeyError as exc:
        raise ValueError(f"fixture does not contain {key}") from exc
    if tuple(tensor.shape) != (batch_size, sequence_length) or tensor.dtype != torch.int64:
        raise ValueError(f"invalid fixture tensor {key}")
    return tensor.contiguous()


def checkpoint_identity(checkpoint: Path, model_id: str) -> dict[str, Any]:
    spec = load_spec(); checkpoint = checkpoint.resolve()
    observed_hash = sha256_path(checkpoint)
    if observed_hash != spec["models"][model_id]["checkpoint_sha256"]:
        raise ValueError(f"{model_id} checkpoint hash mismatch")
    manifest = read_json(checkpoint / "training_manifest.json")
    identity = manifest.get("run_identity", {})
    expected = {
        "global_step": 2000, "seed": 42, "mode": "full_mh",
        "attnres_heads": 8, "hidden_size": 1280, "num_layers": 36,
        "num_heads": 16, "num_kv_heads": 8, "intermediate_size": 5120,
    }
    observed = {"global_step": manifest.get("global_step"), **identity}
    mismatch = {key: [value, observed.get(key)] for key, value in expected.items() if observed.get(key) != value}
    if model_id == "hq8" and identity.get("experiment8_variant") != "hq8":
        mismatch["experiment8_variant"] = ["hq8", identity.get("experiment8_variant")]
    if model_id == "m8" and identity.get("experiment8_variant") is not None:
        mismatch["experiment8_variant"] = [None, identity.get("experiment8_variant")]
    if mismatch:
        raise ValueError(f"{model_id} checkpoint identity mismatch: {mismatch}")
    return {"checkpoint_sha256": observed_hash, "manifest": manifest, "identity": identity}


def load_model(checkpoint: Path, model_id: str, device: torch.device,
               *, training: bool, gradient_checkpointing: bool = False):
    cls = Qwen3AttnResForCausalLM if model_id == "m8" else Experiment8MHARForCausalLM
    model = cls.from_pretrained(
        str(checkpoint), dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device=device)
    validate_model_graph(model, model_id)
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train(training)
    return model


def validate_model_graph(model: torch.nn.Module, model_id: str) -> dict[str, Any]:
    spec = load_spec(); arch = spec["architecture"]
    cfg = model.config
    observed = {
        "hidden_size": int(cfg.hidden_size),
        "num_layers": int(cfg.num_hidden_layers),
        "num_query_heads": int(cfg.num_attention_heads),
        "num_kv_heads": int(cfg.num_key_value_heads),
        "head_dimension": int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)),
        "intermediate_size": int(cfg.intermediate_size),
        "vocab_size": int(cfg.vocab_size),
        "mhar_groups": int(cfg.attnres_num_heads),
    }
    required = {key: arch[key] for key in observed}
    if observed != required or getattr(cfg, "attnres_mode", None) != "full_mh":
        raise ValueError(f"{model_id} architecture mismatch: {observed}")
    q_parameters = 0
    q_shapes: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(model.model.layers):
        query = layer.self_attn.q_proj
        if model_id == "m8":
            if not isinstance(query, torch.nn.Linear) or tuple(query.weight.shape) != (1280, 1280):
                raise TypeError(f"M8 layer {layer_index} must use physical dense 1280x1280 Q")
            q_parameters += query.weight.numel()
            q_shapes.append({"layer": layer_index, "kind": "dense", "weight": [1280, 1280]})
        else:
            if not isinstance(query, HybridQueryLinear):
                raise TypeError(f"HQ8 layer {layer_index} must use HybridQueryLinear")
            if tuple(query.local_weight.shape) != (8, 80, 160) or tuple(query.global_weight.shape) != (8, 80, 1280):
                raise ValueError(f"HQ8 layer {layer_index} Q shapes changed")
            q_parameters += query.local_weight.numel() + query.global_weight.numel()
            q_shapes.append({
                "layer": layer_index, "kind": "physical_local_global",
                "local_weight": [8, 80, 160], "global_weight": [8, 80, 1280],
                "absent_cross_parameters": 8 * 80 * 1120,
            })
        for name, shape in (("k_proj", (640, 1280)), ("v_proj", (640, 1280)), ("o_proj", (1280, 1280))):
            module = getattr(layer.self_attn, name)
            if not isinstance(module, torch.nn.Linear) or tuple(module.weight.shape) != shape:
                raise ValueError(f"{model_id} layer {layer_index} {name} mismatch")
    if q_parameters != spec["analytic"]["q_parameters"][model_id]:
        raise ValueError(f"{model_id} physical Q parameter mismatch")
    total = sum(parameter.numel() for parameter in model.parameters())
    if total != spec["models"][model_id]["accepted_parameters"]:
        raise ValueError(f"{model_id} total parameter mismatch: {total}")
    return {
        "model": model_id, "total_parameters": total, "q_parameters": q_parameters,
        "q_shapes": q_shapes, "dense_local_cross_path_present": False if model_id == "hq8" else None,
    }


@torch.inference_mode()
def parity_loss(model: torch.nn.Module, input_ids: torch.Tensor) -> float:
    logits = model(input_ids=input_ids, use_cache=False).logits
    values = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1), reduction="none",
    )
    return float(values.mean().item())


def optimizer_from_checkpoint(model: torch.nn.Module, checkpoint: Path):
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    required = {"optimizer", "scheduler", "run_identity", "global_step"}
    if not required.issubset(state):
        raise ValueError(f"incomplete training state: {sorted(required - set(state))}")
    identity = state["run_identity"]
    optimizer = AdamW(
        model.parameters(), lr=float(identity["lr"]), betas=(0.9, 0.95),
        eps=1e-8, weight_decay=0.1,
    )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_with_warmup(
            step, int(identity["warmup"]), int(identity["steps"]),
            float(identity["lr_min"]) / float(identity["lr"]),
        ),
    )
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return optimizer, scheduler, identity


def effective_routing_state(model: torch.nn.Module, mode: str) -> dict[str, Any]:
    requested = mhar_triton.fused_mhar_enabled()
    training = mode != "forward"
    checkpointing = bool(getattr(model.model, "gradient_checkpointing", False)) and training
    return {
        "fused_requested_and_available": bool(requested),
        "model_training": training,
        "gradient_checkpointing": checkpointing,
        "fused_context_expected_active": bool(requested and not checkpointing),
        "reason": "gradient-checkpointed training intentionally uses eager routing" if checkpointing else "eligible for fused routing",
    }


def _run_iteration(model, input_ids: torch.Tensor, mode: str, *, optimizer=None,
                   scheduler=None, accumulation: int = 8, max_norm: float = 1.0):
    if mode == "forward":
        with torch.inference_mode():
            result = model(input_ids=input_ids, use_cache=False).logits
        return result
    if mode == "forward-backward":
        model.zero_grad(set_to_none=True)
        loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
        loss.backward()
        return loss
    if mode == "optimizer-step":
        if optimizer is None or scheduler is None:
            raise ValueError("optimizer-step requires restored optimizer and scheduler")
        optimizer.zero_grad(set_to_none=True)
        total = None
        for _ in range(accumulation):
            loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss / accumulation
            loss.backward()
            total = loss.detach() if total is None else total + loss.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        return total
    raise ValueError(f"unknown timing mode: {mode}")


def gpu_telemetry() -> dict[str, Any]:
    fields = "uuid,name,driver_version,temperature.gpu,utilization.gpu,power.draw,clocks.sm,clocks.mem,memory.used,memory.free"
    try:
        output = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.STDOUT,
        ).strip().splitlines()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
        physical_index = int(visible) if visible.isdigit() else torch.cuda.current_device()
        values = [value.strip() for value in output[physical_index].split(",")]
        return dict(zip(fields.split(","), values))
    except Exception as exc:  # pragma: no cover - provider dependent
        return {"telemetry_error": str(exc)}


def static_bytes(model: torch.nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


def optimizer_bytes(optimizer) -> int:
    if optimizer is None:
        return 0
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += value.numel() * value.element_size()
    return total


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()


def _prepare(checkpoint: Path, model_id: str, device: torch.device, mode: str):
    training = mode != "forward"
    checkpointing = training and load_spec()["benchmark"]["gradient_checkpointing_training_modes"]
    model = load_model(checkpoint, model_id, device, training=training,
                       gradient_checkpointing=checkpointing)
    optimizer = scheduler = identity = None
    if mode == "optimizer-step":
        optimizer, scheduler, identity = optimizer_from_checkpoint(model, checkpoint)
    return model, optimizer, scheduler, identity


def benchmark_model(checkpoint: Path, model_id: str, fixture: torch.Tensor,
                    mode: str, iterations: int, device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        raise ValueError("accepted Experiment 18 timing requires CUDA")
    spec = load_spec(); enable_fused_mhar(True)
    input_ids = fixture.to(device=device, dtype=torch.long)

    # Warm kernels and allocator on a disposable accepted-state copy.
    warm, warm_opt, warm_sched, warm_identity = _prepare(checkpoint, model_id, device, mode)
    _run_iteration(
        warm, input_ids, mode, optimizer=warm_opt, scheduler=warm_sched,
        accumulation=spec["benchmark"]["primary"]["gradient_accumulation"],
        max_norm=float((warm_identity or {}).get("max_norm", 1.0)),
    )
    torch.cuda.synchronize(device)
    del warm, warm_opt, warm_sched, warm_identity
    release_cuda()

    # Reload so every measured block begins from the immutable accepted state.
    model, optimizer, scheduler, identity = _prepare(checkpoint, model_id, device, mode)
    routing = effective_routing_state(model, mode)
    before = gpu_telemetry()
    torch.cuda.reset_peak_memory_stats(device)
    gpu_milliseconds: list[float] = []
    host_milliseconds: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        host_start = time.perf_counter()
        start.record()
        result = _run_iteration(
            model, input_ids, mode, optimizer=optimizer, scheduler=scheduler,
            accumulation=spec["benchmark"]["primary"]["gradient_accumulation"],
            max_norm=float((identity or {}).get("max_norm", 1.0)),
        )
        end.record(); end.synchronize()
        host_milliseconds.append((time.perf_counter() - host_start) * 1000)
        gpu_milliseconds.append(float(start.elapsed_time(end)))
        del result
    after = gpu_telemetry()
    tokens_per_iteration = int(input_ids.numel()) * (
        spec["benchmark"]["primary"]["gradient_accumulation"] if mode == "optimizer-step" else 1
    )
    mean_gpu_ms = sum(gpu_milliseconds) / len(gpu_milliseconds)
    payload = {
        "model": model_id, "mode": mode, "batch_size": int(input_ids.shape[0]),
        "sequence_length": int(input_ids.shape[1]), "iterations": iterations,
        "gpu_milliseconds": gpu_milliseconds, "host_milliseconds": host_milliseconds,
        "mean_gpu_milliseconds": mean_gpu_ms,
        "mean_host_milliseconds": sum(host_milliseconds) / len(host_milliseconds),
        "tokens_per_second": tokens_per_iteration / (mean_gpu_ms / 1000),
        "max_memory_allocated": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved": int(torch.cuda.max_memory_reserved(device)),
        "static_parameter_bytes": static_bytes(model),
        "optimizer_state_bytes": optimizer_bytes(optimizer),
        "routing": routing, "telemetry_before": before, "telemetry_after": after,
    }
    del model, optimizer, scheduler, input_ids
    release_cuda()
    return payload


def benchmark_pairs(*, checkpoints: dict[str, Path], fixture_path: Path,
                    output: Path, batch_size: int, sequence_length: int,
                    mode: str, pair_count: int, order_seed: int,
                    iterations: int, device: torch.device) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite timing output: {output}")
    orders = balanced_orders(pair_count, order_seed)
    fixture = load_fixture(fixture_path, batch_size, sequence_length)
    for pair_index, order in enumerate(orders):
        for position, model_id in enumerate(order):
            row = benchmark_model(
                checkpoints[model_id], model_id, fixture, mode, iterations, device
            )
            row.update({
                "format_version": 1, "created_at": utc_now(),
                "protocol_sha256": protocol_sha256(), "pair_index": pair_index,
                "order": order, "position": position,
                "checkpoint_sha256": load_spec()["models"][model_id]["checkpoint_sha256"],
                "fixture_sha256": sha256_file(fixture_path),
            })
            append_jsonl(output, row)


def _profile_table(profiler) -> list[dict[str, Any]]:
    rows = []
    for item in profiler.key_averages(group_by_input_shape=True):
        rows.append({
            "key": item.key, "input_shapes": str(item.input_shapes),
            "count": int(item.count), "cpu_time_total_us": float(item.cpu_time_total),
            "cuda_time_total_us": float(getattr(item, "device_time_total", 0.0)),
            "profiler_attributed_flops": int(getattr(item, "flops", 0) or 0),
        })
    return rows


def profile_model(*, checkpoint: Path, model_id: str, fixture_path: Path,
                  output_root: Path, kind: str, device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        raise ValueError("Experiment 18 profiler requires CUDA")
    if kind not in {"qproj", "forward", "train"}:
        raise ValueError(f"unknown profile kind: {kind}")
    target = output_root / "data/raw/profiler_tables" / kind / model_id
    if target.exists():
        raise FileExistsError(f"refusing to overwrite profiler output: {target}")
    target.mkdir(parents=True)
    mode = "forward-backward" if kind == "train" else "forward"
    enable_fused_mhar(True)
    model, _, _, _ = _prepare(checkpoint, model_id, device, mode)
    ids = load_fixture(fixture_path, 4, 1024).to(device)
    if kind == "qproj":
        inputs = model.model.embed_tokens(ids).detach()
        callable_ = lambda: model.model.layers[0].self_attn.q_proj(inputs)
    elif kind == "forward":
        callable_ = lambda: model(input_ids=ids, use_cache=False).logits
    else:
        def callable_():
            model.zero_grad(set_to_none=True)
            loss = model(input_ids=ids, labels=ids, use_cache=False).loss
            loss.backward()
            return loss
    callable_(); torch.cuda.synchronize(device)
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities, record_shapes=True, profile_memory=True,
        with_flops=True, with_stack=False,
    ) as profiler:
        with torch.profiler.record_function(f"exp18_{kind}_{model_id}"):
            result = callable_()
        torch.cuda.synchronize(device)
    del result
    trace = target / "trace.json"
    profiler.export_chrome_trace(str(trace))
    table = _profile_table(profiler)
    atomic_write_json(target / "table.json", {
        "format_version": 1, "created_at": utc_now(), "model": model_id,
        "kind": kind, "protocol_sha256": protocol_sha256(), "rows": table,
        "profiler_attributed_flops": sum(row["profiler_attributed_flops"] for row in table),
        "trace_sha256": sha256_file(trace), "routing": effective_routing_state(model, mode),
    })
    del model, ids
    release_cuda()
    return {"trace": str(trace), "trace_sha256": sha256_file(trace), "rows": len(table)}


def write_graph_and_parity(*, checkpoints: dict[str, Path], artifact: Path,
                           output_root: Path, device: torch.device) -> None:
    payload, artifact_hash = load_fixed_eval_artifact(artifact)
    if artifact_hash != load_spec()["fixed_evaluation"]["sha256"]:
        raise ValueError("fixed artifact hash mismatch")
    ids = payload["discovery_input_ids"][:1].to(device=device, dtype=torch.long)
    reports = {}
    enable_fused_mhar(False)  # reproduce the accepted Experiment 2/8 evaluator exactly
    for model_id in MODELS:
        identity = checkpoint_identity(checkpoints[model_id], model_id)
        model = load_model(checkpoints[model_id], model_id, device, training=False)
        graph = validate_model_graph(model, model_id)
        observed = parity_loss(model, ids)
        expected = load_spec()["models"][model_id]["parity_discovery_sequence_0_loss"]
        if observed != expected:
            raise ValueError(f"{model_id} parity Loss mismatch: {observed} != {expected}")
        reports[model_id] = {
            "identity": identity, "graph": graph, "parity_loss": observed,
            "parity_expected": expected, "absolute_error": 0.0,
        }
        del model
        release_cuda()
    atomic_write_json(output_root / "provenance/native_graph_and_parity.json", {
        "format_version": 1, "created_at": utc_now(),
        "protocol_sha256": protocol_sha256(), "artifact_sha256": artifact_hash,
        "reports": reports,
    })


def manifest_command(args) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    atomic_write_json(args.output, runtime_manifest())


def ledger_command(args) -> None:
    write_ledger(args.output_root.resolve())


def fixture_command(args) -> None:
    build_fixture(args.artifact.resolve(), args.output.resolve())


def validate_command(args) -> None:
    checkpoints = {"m8": args.m8_checkpoint.resolve(), "hq8": args.hq8_checkpoint.resolve()}
    write_graph_and_parity(
        checkpoints=checkpoints, artifact=args.artifact.resolve(),
        output_root=args.output_root.resolve(), device=torch.device(args.device),
    )


def benchmark_command(args) -> None:
    benchmark_pairs(
        checkpoints={"m8": args.m8_checkpoint.resolve(), "hq8": args.hq8_checkpoint.resolve()},
        fixture_path=args.fixture.resolve(), output=args.output.resolve(),
        batch_size=args.batch_size, sequence_length=args.sequence_length,
        mode=args.mode, pair_count=args.pair_count, order_seed=args.order_seed,
        iterations=args.iterations, device=torch.device(args.device),
    )


def profile_command(args) -> None:
    for model_id, checkpoint in (("m8", args.m8_checkpoint), ("hq8", args.hq8_checkpoint)):
        profile_model(
            checkpoint=checkpoint.resolve(), model_id=model_id,
            fixture_path=args.fixture.resolve(), output_root=args.output_root.resolve(),
            kind=args.kind, device=torch.device(args.device),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest"); manifest.add_argument("--output", type=Path, required=True); manifest.set_defaults(func=manifest_command)
    ledger = commands.add_parser("ledger"); ledger.add_argument("--output-root", type=Path, required=True); ledger.set_defaults(func=ledger_command)
    fixture = commands.add_parser("fixture"); fixture.add_argument("--artifact", type=Path, required=True); fixture.add_argument("--output", type=Path, required=True); fixture.set_defaults(func=fixture_command)
    validate = commands.add_parser("validate")
    profile = commands.add_parser("profile")
    benchmark = commands.add_parser("benchmark")
    for item in (validate, profile, benchmark):
        item.add_argument("--m8-checkpoint", type=Path, required=True)
        item.add_argument("--hq8-checkpoint", type=Path, required=True)
    validate.add_argument("--artifact", type=Path, required=True); validate.add_argument("--output-root", type=Path, required=True); validate.add_argument("--device", default="cuda"); validate.set_defaults(func=validate_command)
    profile.add_argument("--fixture", type=Path, required=True); profile.add_argument("--output-root", type=Path, required=True); profile.add_argument("--kind", choices=("qproj", "forward", "train"), required=True); profile.add_argument("--device", default="cuda"); profile.set_defaults(func=profile_command)
    benchmark.add_argument("--fixture", type=Path, required=True); benchmark.add_argument("--output", type=Path, required=True); benchmark.add_argument("--batch-size", type=int, required=True); benchmark.add_argument("--sequence-length", type=int, required=True); benchmark.add_argument("--mode", choices=TIMING_MODES, required=True); benchmark.add_argument("--pair-count", type=int, required=True); benchmark.add_argument("--order-seed", type=int, required=True); benchmark.add_argument("--iterations", type=int, required=True); benchmark.add_argument("--device", default="cuda"); benchmark.set_defaults(func=benchmark_command)
    return parser


def main() -> None:
    args = build_parser().parse_args(); args.func(args)


if __name__ == "__main__":
    main()
