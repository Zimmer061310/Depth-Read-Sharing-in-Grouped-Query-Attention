"""Stage-B adapter of the frozen update recipe; old three-condition module untouched.

No corpus/artifact reads, primary scoring, checkpoint writes, DDP, autocast,
GradScaler, or authorization inference. Callers still need a complete source,
data, checkpoint and paid-execution lock. CPU fixtures are explicitly labeled.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

from src.attention_residuals.modeling_qwen3_attnres import Qwen3AttnResConfig
from src.experiments.paper_stage_b_baseline import (
    StageBForCausalLM as PaperSharingForCausalLM,
    validate_config, optimizer_coverage as validate_optimizer_coverage,
    production_config as baseline_config,
)
from src.experiments.paper_sharing_supervision import load_contract, native_training_kwargs
from src.experiments.paper_stage_b_followup import SEEDS
from src.training.train_scratch import cosine_with_warmup


ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "configs/paper_sharing_validation_2k/training-recipe.frozen.json"
RECIPE_SHA256 = "0b303ab2077e19ee1ed9ca18eaa6d92200e81c5936c034b0de36ab6dc9b2360a"
SUPERVISION_SHA256 = "dd3591d3e0cd2d50f362aa5a09ba38c71945a04b0284cb4a53d1a667174a1219"


def frozen_recipe():
    raw = RECIPE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != RECIPE_SHA256:
        raise ValueError("frozen recipe byte identity changed")
    contract = load_contract()
    contract_path = ROOT / "configs/paper_sharing_validation_2k/loss-supervision.frozen.json"
    if hashlib.sha256(contract_path.read_bytes()).hexdigest() != SUPERVISION_SHA256:
        raise ValueError("frozen supervision byte identity changed")
    if contract["training_recipe_intent_sha256"] != RECIPE_SHA256:
        raise ValueError("recipe/supervision binding mismatch")
    return json.loads(raw)


def production_config(condition, *, execution_policy=None, runtime_policy=None):
    return baseline_config(condition)

def validate_training_model(model, *, local_fixture=False):
    if type(local_fixture) is not bool or type(model) is not PaperSharingForCausalLM:
        raise ValueError("explicit Stage-B model and boolean fixture flag required")
    validate_config(model.config)
    condition, backend = model.config.paper_shared_read_condition, "canonical"
    model.validate_execution_dispatch()
    if model.config.paper_sharing_test_fixture != local_fixture:
        raise ValueError("fixture/production model identity mismatch")
    if local_fixture:
        if backend != "canonical" or any(p.device.type != "cpu" or p.dtype != torch.float32
                                         for p in model.parameters()):
            raise ValueError("local execution fixtures must be canonical CPU FP32")
    else:
        expected = production_config(condition, execution_policy=getattr(model.config, "paper_sharing_execution_policy", None),
                                     runtime_policy=getattr(model.config, "paper_sharing_runtime_policy", None)).to_dict()
        # Compare all serialized architecture/execution settings, not just width.
        actual = model.config.to_dict()
        ignored = {"transformers_version", "torch_dtype", "dtype", "_name_or_path", "architectures"}
        if {k: v for k, v in actual.items() if k not in ignored} != {
                k: v for k, v in expected.items() if k not in ignored}:
            raise ValueError("model differs from frozen production recipe")
        if any(p.device.type != "cuda" or p.dtype != torch.bfloat16 for p in model.parameters()):
            raise ValueError("production requires native CUDA BF16 parameters, not autocast FP32 masters")
    return condition


def native_lr_multiplier(step):
    return cosine_with_warmup(step, 1000, 20000, .1)


class FrozenSchedule(torch.optim.lr_scheduler.LambdaLR):
    """Exactly the historical schedule, including initial LR=0 and update order."""
    def __init__(self, optimizer):
        frozen_recipe()
        super().__init__(optimizer, native_lr_multiplier)


def make_optimization(model, *, local_fixture=False):
    validate_training_model(model, local_fixture=local_fixture)
    r = frozen_recipe()
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=r["lr"], betas=tuple(r["betas"]),
                                 eps=r["epsilon"], weight_decay=r["weight_decay"])
    validate_optimizer_coverage(model, optimizer)
    scheduler = FrozenSchedule(optimizer)
    optimizer.zero_grad(set_to_none=True)
    return optimizer, scheduler


@dataclass
class UpdateProgress:
    """Complete-update counters only; NOT sufficient scientific resume evidence."""
    condition: str
    seed: int
    completed_updates: int = 0
    chunks_consumed: int = 0
    failed: bool = False
    local_fixture: bool = False


def validate_boundary(model, optimizer, scheduler, progress, *, microbatch, accumulation):
    r = frozen_recipe()
    if not isinstance(progress, UpdateProgress) or type(progress.local_fixture) is not bool:
        raise ValueError("explicit update progress required")
    condition = validate_training_model(model, local_fixture=progress.local_fixture)
    if (progress.condition != condition or type(progress.seed) is not int or progress.seed not in SEEDS
            or type(progress.failed) is not bool or progress.failed
            or type(progress.completed_updates) is not int or not 0 <= progress.completed_updates <= 2000
            or type(progress.chunks_consumed) is not int
            or any(type(n) is not int or n < 1 for n in (microbatch, accumulation))):
        raise ValueError("changed run identity, poisoned state or invalid counters")
    if not progress.local_fixture and (microbatch, accumulation) != (4, 8):
        raise ValueError("production must retain B4 accumulation8 world-size1")
    if progress.chunks_consumed != progress.completed_updates * microbatch * accumulation:
        raise ValueError("partial update or changed data position")
    validate_optimizer_coverage(model, optimizer)
    if type(optimizer) is not torch.optim.AdamW or type(scheduler) is not FrozenSchedule or scheduler.optimizer is not optimizer:
        raise ValueError("changed optimizer or native schedule")
    if scheduler.last_epoch != progress.completed_updates or scheduler.base_lrs != [r["lr"]]:
        raise ValueError("scheduler/update clock mismatch")
    if scheduler.lr_lambdas != [native_lr_multiplier]:
        raise ValueError("changed native schedule function")
    expected_lr = r["lr"] * cosine_with_warmup(progress.completed_updates, 1000, 20000, .1)
    if len(optimizer.param_groups) != 1:
        raise ValueError("native recipe uses one optimizer group")
    group = optimizer.param_groups[0]
    if (group["betas"] != tuple(r["betas"]) or group["eps"] != r["epsilon"]
            or group["weight_decay"] != r["weight_decay"] or group["lr"] != expected_lr
            or group["amsgrad"] or group["maximize"] or group["capturable"] or group["differentiable"]
            or group["foreach"] is not None or group["fused"] is not None):
        raise ValueError("changed native AdamW policy")
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("boundary has uncheckpointed/partial gradients")
    if (progress.completed_updates == 0 and optimizer.state) or (
            progress.completed_updates > 0 and not optimizer.state):
        raise ValueError("optimizer initialization/update clock mismatch")
    for state in optimizer.state.values():
        step = state.get("step")
        if not isinstance(step, torch.Tensor) or step.numel() != 1 or step.item() != progress.completed_updates:
            raise ValueError("optimizer parameter update clock mismatch")
    if not model.training or not model.is_gradient_checkpointing:
        raise ValueError("training mode and gradient checkpointing required")


def collate_training_window(examples, *, expected_start, device="cpu", microbatch=4,
                            accumulation=8, local_fixture=False):
    """Collate caller's verified training-only reader, rejecting gaps/duplicates.

    Reads exactly a complete window, without prefetching the next example. This
    checks ordering, not corpus identity: the caller must supply the reader's
    frozen manifest hash and resume offset independently. No path is accepted.
    """
    if (type(local_fixture) is not bool or type(expected_start) is not int or expected_start < 0
            or any(type(n) is not int or n < 1 for n in (microbatch, accumulation))
            or (not local_fixture and (microbatch, accumulation) != (4, 8))):
        raise ValueError("invalid window position or changed production batch policy")
    rows = []
    for index in range(expected_start, expected_start + microbatch * accumulation):
        try:
            row = next(examples)
        except StopIteration as exc:
            raise ValueError("training stream ended before a complete update; no partial update allowed") from exc
        if (not isinstance(row, dict) or set(row) != {"chunk_index", "input_ids", "targets", "valid_target_mask"}
                or type(row["chunk_index"]) is not int or row["chunk_index"] != index
                or any(not isinstance(row[key], list) or len(row[key]) != 1024 for key in
                       ("input_ids", "targets", "valid_target_mask"))
                or any(type(value) is not int for key in ("input_ids", "targets") for value in row[key])
                or any(type(value) is not bool for value in row["valid_target_mask"])):
            raise ValueError("malformed, duplicate, missing or out-of-order training example")
        rows.append(row)
    batches = []
    for start in range(0, len(rows), microbatch):
        part = rows[start:start + microbatch]
        batch = (torch.tensor([row["input_ids"] for row in part], dtype=torch.long, device=device),
                 torch.tensor([row["targets"] for row in part], dtype=torch.long, device=device),
                 torch.tensor([row["valid_target_mask"] for row in part], dtype=torch.bool, device=device))
        native_training_kwargs(batch[0], exposed_targets=batch[1], data_validity_mask=batch[2])
        batches.append(batch)
    return batches


def complete_update(model, optimizer, scheduler, progress, batches, *, microbatch=4, accumulation=8):
    """Consume exactly one complete update, never silently retry a partial failure.

    Batch tuples are (input_ids, exposed_targets, data_validity_mask), already
    supplied by the caller. No primary reader exists here. Caller must ensure
    their provenance/stream order; counters alone do not prove that identity.
    """
    validate_boundary(model, optimizer, scheduler, progress, microbatch=microbatch, accumulation=accumulation)
    if progress.completed_updates == 2000:
        raise ValueError("exact 2000-update stop reached; extra work refused")
    if not isinstance(batches, (list, tuple)) or len(batches) != accumulation:
        raise ValueError("one complete accumulation window required")
    prepared = []
    device = next(model.parameters()).device
    for batch in batches:
        if not isinstance(batch, (tuple, list)) or len(batch) != 3:
            raise ValueError("expected input/target/validity batch triple")
        inputs, targets, validity = batch
        kwargs = native_training_kwargs(inputs, exposed_targets=targets, data_validity_mask=validity)
        if inputs.shape[0] != microbatch or inputs.device != device:
            raise ValueError("microbatch or device mismatch")
        prepared.append(kwargs)
    # Once computation begins any failure poisons this controller. No rollback
    # is implied, even if optimizer.step itself throws after a partial mutation.
    try:
        total = 0.0
        for kwargs in prepared:
            loss = model(**kwargs).loss / accumulation
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite or nonscalar training loss; update refused")
            loss.backward()
            total += loss.detach().item()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("nonfinite gradient norm; update refused")
        lr_used = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        progress.completed_updates += 1
        progress.chunks_consumed += microbatch * accumulation
    except BaseException:
        progress.failed = True
        raise
    return {"completed_updates": progress.completed_updates, "chunks_consumed": progress.chunks_consumed,
            "input_positions": progress.chunks_consumed * 1024,
            "supervised_positions": progress.chunks_consumed * 1023,
            "loss": total, "gradient_norm": norm.item(), "lr_used": lr_used,
            "next_lr": scheduler.get_last_lr()[0], "local_fixture": progress.local_fixture,
            "scientific_resume_verified": False, "full_gate_a_passed": False}
