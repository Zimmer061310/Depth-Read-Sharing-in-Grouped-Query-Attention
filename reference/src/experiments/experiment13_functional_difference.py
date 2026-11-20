#!/usr/bin/env python3
"""Frozen observational measurements for Experiment 13A--13C attempt 2.

The accepted HQ8 checkpoint is never modified.  Temporary hooks observe the
interleaved local/global head outputs immediately before dense ``W_O``.  The
32-sequence Experiment 13C probe reconstructs attention probabilities in a
passive side path while the model remains on its original SDPA implementation.
The reconstruction is reduced online without retaining full attention matrices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import repeat_kv

from src.experiments.experiment1_partition_compatibility import (
    atomic_write_json,
    dtype_from_name,
    git_commit,
    load_fixed_eval_artifact,
    sha256_file,
    utc_now,
)
from src.experiments.experiment8_hybrid_q import Experiment8MHARForCausalLM
from src.experiments.experiment9_head_contribution import (
    validate_checkpoint,
    validate_hq8_model,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiment13/protocol.json"
SPLITS = ("discovery", "confirmation")
MEASUREMENT_KINDS = ("reproduce", "ab", "c")
OUTPUT_METRICS = ("a_local", "a_global", "b_local", "b_global")
PROBE_METRICS = OUTPUT_METRICS + (
    "delta_h",
    "js_divergence",
    "s_out",
    "s_res",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_path_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return sha256_file(path)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    """Atomically write one compressed NumPy archive."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_spec(path: Path = CONFIG) -> dict[str, Any]:
    spec = read_json(path)
    if spec.get("training") is not False or spec.get("seed") != 42 or spec.get("milestone") != 2000:
        raise ValueError("Experiment 13 must remain a frozen seed-42 step-2000 evaluation")
    if spec.get("attempt") != {
        "number": 2,
        "method": "passive_sdpa_attention_reconstruction",
        "predecessor_status": "attempt1_failed_discovery_eager_parity_gate",
        "predecessor_max_per_sequence_loss_difference": 0.0009851455688476562,
        "predecessor_source_commit": "b942dbaaff47cba419650c02fbf17439d40c1459",
        "predecessor_protocol_sha256": "fe35da88b72349f8a05de3aac89aba006546bd4c09514ba3be6d00bde708524c",
    }:
        raise ValueError("Experiment 13 attempt-2 identity changed")
    architecture = spec.get("architecture", {})
    expected_architecture = {
        "hidden_size": 1280,
        "num_layers": 36,
        "num_gqa_groups": 8,
        "num_query_heads": 16,
        "num_kv_heads": 8,
        "head_dimension": 80,
        "local_head_rule": "2*g",
        "global_head_rule": "2*g+1",
        "observation_site": "attention head output immediately before dense W_O",
    }
    if architecture != expected_architecture:
        raise ValueError("Experiment 13 architecture contract changed")
    fixed = spec.get("fixed_evaluation", {})
    if (
        fixed.get("splits") != list(SPLITS)
        or fixed.get("sequences_per_split") != 512
        or fixed.get("sequence_length") != 1024
        or fixed.get("valid_tokens_per_sequence") != 1023
    ):
        raise ValueError("Experiment 13 fixed-evaluation contract changed")
    probe = spec.get("probe", {})
    if (
        probe.get("sequences_per_split") != 32
        or probe.get("salt") != "mhar-exp13-functional-difference-v1"
        or probe.get("store_tokenwise_float32") is not True
    ):
        raise ValueError("Experiment 13 probe contract changed")
    numerics = spec.get("numerics", {})
    if (
        numerics.get("epsilon") != 1e-8
        or numerics.get("microbatch_size") != 1
        or numerics.get("quantiles") != [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    ):
        raise ValueError("Experiment 13 numerical contract changed")
    if numerics.get("parity_tolerances") != {
        "repeated_uninstrumented_per_sequence_max_abs": 0.0,
        "accepted_vs_uninstrumented_per_sequence_max_abs": 0.0,
        "uninstrumented_vs_ab_per_sequence_max_abs": 0.0,
        "accepted_probe_vs_passive_c_per_sequence_max_abs": 0.0,
    }:
        raise ValueError("Experiment 13 parity tolerance contract changed")
    if numerics.get("attention_reconstruction") != {
        "model_attention_implementation": "sdpa",
        "query_block_size": 64,
        "score_dtype": "model_dtype",
        "softmax_dtype": "float32",
        "probability_output_dtype": "model_dtype",
        "float32_eager_reference_atol": 1e-6,
        "float32_eager_reference_rtol": 1e-5,
        "retain_full_attention_matrices": False,
    }:
        raise ValueError("Experiment 13 passive attention reconstruction contract changed")
    expected_jobs = [
        "exp13a2-repro-discovery-full512",
        "exp13a2-ab-discovery-full512",
        "exp13a2-c-discovery-probe32",
        "exp13a2-analysis-discovery",
        "exp13a2-repro-confirmation-full512",
        "exp13a2-ab-confirmation-full512",
        "exp13a2-c-confirmation-probe32",
        "exp13a2-analysis-confirmation",
    ]
    if spec.get("logical_jobs") != expected_jobs:
        raise ValueError("Experiment 13 logical job matrix changed")
    if spec.get("wandb_runs") != {
        "discovery_analysis": {
            "id": "e13d42p2",
            "name": "mhar-exp13-discovery-analysis-seed42-step2000-attempt2",
        },
        "confirmation_analysis": {
            "id": "e13c42p2",
            "name": "mhar-exp13-confirmation-analysis-seed42-step2000-attempt2",
        },
    }:
        raise ValueError("Experiment 13 W&B identities changed")
    if spec.get("wandb_group") != "mhar-exp13-functional-difference-seed42-step2000-attempt2":
        raise ValueError("Experiment 13 W&B group changed")
    return spec


def protocol_sha256(path: Path = CONFIG) -> str:
    return sha256_path_file(path)


def select_probe_indices(
    split: str,
    *,
    artifact_sha256: str,
    sequence_count: int = 512,
    selected_count: int = 32,
    salt: str = "mhar-exp13-functional-difference-v1",
) -> list[int]:
    if split not in SPLITS:
        raise ValueError(f"unsupported split {split!r}")
    if not 0 < selected_count <= sequence_count:
        raise ValueError("probe size must be positive and no larger than the split")
    keyed = []
    for index in range(sequence_count):
        message = f"{salt}|{artifact_sha256}|{split}|{index}".encode("utf-8")
        keyed.append((hashlib.sha256(message).hexdigest(), index))
    chosen = [index for _, index in sorted(keyed)[:selected_count]]
    return sorted(chosen)


def build_probe_manifest(artifact_path: Path, spec: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = load_spec() if spec is None else spec
    payload, observed_hash = load_fixed_eval_artifact(artifact_path)
    expected_hash = spec["fixed_evaluation"]["sha256"]
    if observed_hash != expected_hash:
        raise ValueError("fixed evaluation artifact hash differs from Experiment 13 protocol")
    sequence_count = spec["fixed_evaluation"]["sequences_per_split"]
    sequence_length = spec["fixed_evaluation"]["sequence_length"]
    for split in SPLITS:
        tensor = payload[f"{split}_input_ids"]
        if list(tensor.shape) != [sequence_count, sequence_length]:
            raise ValueError(f"unexpected {split} artifact shape {list(tensor.shape)}")
    manifest: dict[str, Any] = {
        "format_version": 1,
        "experiment": "13-functional-difference",
        "attempt": spec["attempt"],
        "artifact_sha256": observed_hash,
        "protocol_sha256": protocol_sha256(),
        "selection_algorithm": (
            "lexicographically smallest SHA256(salt|artifact_sha256|split|decimal(index)); "
            "digest ties by ascending index; output indices sorted ascending"
        ),
        "salt": spec["probe"]["salt"],
        "sequence_count_per_split": sequence_count,
        "probe_count_per_split": spec["probe"]["sequences_per_split"],
        "indices": {
            split: select_probe_indices(
                split,
                artifact_sha256=observed_hash,
                sequence_count=sequence_count,
                selected_count=spec["probe"]["sequences_per_split"],
                salt=spec["probe"]["salt"],
            )
            for split in SPLITS
        },
    }
    manifest["content_sha256"] = canonical_json_sha256(manifest)
    return manifest


def validate_probe_manifest(
    manifest: dict[str, Any],
    artifact_path: Path,
    spec: dict[str, Any] | None = None,
) -> None:
    expected = build_probe_manifest(artifact_path, spec)
    if manifest != expected:
        raise ValueError("probe manifest differs from the frozen deterministic selection")


def write_probe_manifest(path: Path, artifact_path: Path) -> dict[str, Any]:
    expected = build_probe_manifest(artifact_path)
    if path.exists():
        observed = read_json(path)
        if observed != expected:
            raise ValueError("refusing to replace a different Experiment 13 probe manifest")
        return observed
    atomic_write_json(path, expected)
    return expected


def output_projection_blocks(weight: torch.Tensor, *, num_heads: int = 16, head_dim: int = 80) -> torch.Tensor:
    if weight.ndim != 2 or weight.shape[1] != num_heads * head_dim:
        raise ValueError("unexpected dense W_O shape")
    # [output, input] -> [head, head_input, output]
    return weight.float().reshape(weight.shape[0], num_heads, head_dim).permute(1, 2, 0)


def output_metric_tensors(
    values: torch.Tensor,
    output_weight: torch.Tensor,
    *,
    epsilon: float = 1e-8,
    include_similarity: bool = False,
) -> dict[str, torch.Tensor]:
    """Return per-token paired-head measurements without materializing delta-x.

    Residual norms and cosines are computed exactly through the 80x80 Gram
    matrices induced by each head's dense ``W_O`` input-column block.
    """

    if values.ndim != 3 or values.shape[-1] != 16 * 80:
        raise ValueError("expected pre-W_O values with shape [batch, tokens, 1280]")
    heads = values.float().reshape(*values.shape[:-1], 16, 80)
    local = heads[..., 0::2, :]
    global_ = heads[..., 1::2, :]
    blocks = output_projection_blocks(output_weight)
    local_blocks = blocks[0::2]
    global_blocks = blocks[1::2]
    gram_local = torch.einsum("gio,gjo->gij", local_blocks, local_blocks)
    gram_global = torch.einsum("gio,gjo->gij", global_blocks, global_blocks)
    gram_cross = torch.einsum("gio,gjo->gij", local_blocks, global_blocks)

    local_sq = torch.einsum("btgi,gij,btgj->btg", local, gram_local, local)
    global_sq = torch.einsum("btgi,gij,btgj->btg", global_, gram_global, global_)
    b_local = torch.sqrt(torch.clamp(local_sq, min=0.0))
    b_global = torch.sqrt(torch.clamp(global_sq, min=0.0))
    result = {
        "a_local": torch.linalg.vector_norm(local, dim=-1),
        "a_global": torch.linalg.vector_norm(global_, dim=-1),
        "b_local": b_local,
        "b_global": b_global,
    }
    if include_similarity:
        out_dot = torch.sum(local * global_, dim=-1)
        out_denominator = torch.clamp(
            result["a_local"] * result["a_global"], min=epsilon
        )
        residual_dot = torch.einsum(
            "btgi,gij,btgj->btg", local, gram_cross, global_
        )
        residual_denominator = torch.clamp(b_local * b_global, min=epsilon)
        result["s_out"] = out_dot / out_denominator
        result["s_res"] = residual_dot / residual_denominator
    return result


def output_weight_statistics(weight: torch.Tensor) -> dict[str, np.ndarray]:
    blocks = output_projection_blocks(weight)
    local = blocks[0::2]
    global_ = blocks[1::2]

    def stats(values: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        frobenius = torch.linalg.vector_norm(values, dim=(-2, -1))
        rms = torch.sqrt(torch.mean(values.square(), dim=(-2, -1)))
        return frobenius.cpu().numpy().astype(np.float64), rms.cpu().numpy().astype(np.float64)

    local_fro, local_rms = stats(local)
    global_fro, global_rms = stats(global_)
    return {
        "w_o_local_frobenius": local_fro,
        "w_o_global_frobenius": global_fro,
        "w_o_local_rms": local_rms,
        "w_o_global_rms": global_rms,
    }


def attention_metric_tensors(
    attention_weights: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    epsilon: float = 1e-8,
) -> dict[str, torch.Tensor]:
    if attention_weights.ndim != 4 or attention_weights.shape[1] != 16:
        raise ValueError("expected attention probabilities with shape [batch, 16, query, key]")
    probabilities = attention_weights.float()
    local = probabilities[:, 0::2]
    global_ = probabilities[:, 1::2]
    if local.shape != global_.shape:
        raise AssertionError("paired local/global attention shapes differ")

    def entropy(values: torch.Tensor) -> torch.Tensor:
        return -torch.sum(
            torch.where(values > 0, values * torch.log(torch.clamp(values, min=epsilon)), torch.zeros_like(values)),
            dim=-1,
        )

    local_entropy = entropy(local)
    global_entropy = entropy(global_)
    query_length = probabilities.shape[-2]
    if attention_mask is None:
        available = torch.arange(
            1, query_length + 1, device=probabilities.device, dtype=torch.float32
        ).view(1, query_length).expand(probabilities.shape[0], -1)
    else:
        mask = attention_mask[..., :query_length, :probabilities.shape[-1]]
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError("unexpected causal attention-mask shape")
        available = (mask[:, 0].float() >= 0).sum(dim=-1).float()
    normalization = torch.log(torch.clamp(available, min=1.0))
    safe_normalization = torch.where(
        available > 1, normalization, torch.ones_like(normalization)
    )
    local_normalized = local_entropy / safe_normalization[:, None, :]
    global_normalized = global_entropy / safe_normalization[:, None, :]
    local_normalized = torch.where(
        (available > 1)[:, None, :], local_normalized, torch.zeros_like(local_normalized)
    )
    global_normalized = torch.where(
        (available > 1)[:, None, :], global_normalized, torch.zeros_like(global_normalized)
    )

    mixture = 0.5 * (local + global_)
    local_kl = torch.sum(
        torch.where(
            local > 0,
            local * (torch.log(torch.clamp(local, min=epsilon)) - torch.log(torch.clamp(mixture, min=epsilon))),
            torch.zeros_like(local),
        ),
        dim=-1,
    )
    global_kl = torch.sum(
        torch.where(
            global_ > 0,
            global_ * (torch.log(torch.clamp(global_, min=epsilon)) - torch.log(torch.clamp(mixture, min=epsilon))),
            torch.zeros_like(global_),
        ),
        dim=-1,
    )
    # Attention tensors are [B, G, T]; output metric tensors use [B, T, G].
    return {
        "delta_h": (local_normalized - global_normalized).transpose(1, 2),
        "js_divergence": (0.5 * (local_kl + global_kl)).transpose(1, 2),
    }


def _additive_attention_mask(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    query_start: int,
    query_end: int,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """Return the exact additive support mask for one query block."""

    batch = query_states.shape[0]
    key_length = key_states.shape[-2]
    if attention_mask is not None:
        if attention_mask.ndim != 4:
            raise ValueError("passive attention reconstruction requires a 4D mask")
        block = attention_mask[..., query_start:query_end, :key_length]
        if block.shape[-2:] != (query_end - query_start, key_length):
            raise ValueError("attention-mask slice does not match reconstructed block")
        if block.dtype == torch.bool:
            additive = torch.zeros_like(block, dtype=query_states.dtype)
            block = additive.masked_fill(~block, torch.finfo(query_states.dtype).min)
        else:
            block = block.to(device=query_states.device, dtype=query_states.dtype)
        if block.shape[0] not in (1, batch) or block.shape[1] != 1:
            raise ValueError("unexpected passive attention-mask batch/head shape")
        return block

    if query_states.shape[-2] != key_length:
        raise ValueError("mask-free passive reconstruction does not support cached decoding")
    query_positions = torch.arange(
        query_start, query_end, device=query_states.device
    ).view(-1, 1)
    key_positions = torch.arange(key_length, device=query_states.device).view(1, -1)
    allowed = key_positions <= query_positions
    if sliding_window is not None:
        allowed &= key_positions > (query_positions - int(sliding_window))
    block = torch.full(
        (1, 1, query_end - query_start, key_length),
        torch.finfo(query_states.dtype).min,
        device=query_states.device,
        dtype=query_states.dtype,
    )
    return block.masked_fill(allowed.view(1, 1, query_end - query_start, key_length), 0)


def _attention_probability_blocks(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    scaling: float,
    query_block_size: int,
    num_key_value_groups: int,
    sliding_window: int | None = None,
):
    """Yield eager-equivalent probabilities without changing model execution."""

    if query_states.ndim != 4 or key_states.ndim != 4:
        raise ValueError("passive reconstruction expects [batch, heads, tokens, width]")
    if query_states.shape[0] != key_states.shape[0] or query_states.shape[-1] != key_states.shape[-1]:
        raise ValueError("query/key batch or head width mismatch")
    if query_block_size <= 0:
        raise ValueError("query block size must be positive")
    repeated_keys = repeat_kv(key_states, int(num_key_value_groups))
    if repeated_keys.shape[1] != query_states.shape[1]:
        raise ValueError("GQA key repetition does not match the query-head count")
    query_length = query_states.shape[-2]
    for start in range(0, query_length, query_block_size):
        end = min(start + query_block_size, query_length)
        block_mask = _additive_attention_mask(
            query_states,
            repeated_keys,
            attention_mask,
            query_start=start,
            query_end=end,
            sliding_window=sliding_window,
        )
        scores = torch.matmul(
            query_states[:, :, start:end], repeated_keys.transpose(2, 3)
        ) * scaling
        scores = scores + block_mask
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        yield start, end, probabilities, block_mask


def reconstruct_attention_probabilities(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    scaling: float,
    query_block_size: int,
    num_key_value_groups: int,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """Materialize passive probabilities for deterministic reference tests only."""

    blocks = [
        probabilities
        for _, _, probabilities, _ in _attention_probability_blocks(
            query_states,
            key_states,
            attention_mask,
            scaling=scaling,
            query_block_size=query_block_size,
            num_key_value_groups=num_key_value_groups,
            sliding_window=sliding_window,
        )
    ]
    return torch.cat(blocks, dim=-2)


def passive_attention_metric_tensors(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    scaling: float,
    query_block_size: int,
    num_key_value_groups: int,
    sliding_window: int | None = None,
    epsilon: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Reduce attention metrics blockwise without retaining full score matrices."""

    batch, _, query_length, _ = query_states.shape
    output = {
        "delta_h": torch.empty(
            batch, query_length, 8, device=query_states.device, dtype=torch.float32
        ),
        "js_divergence": torch.empty(
            batch, query_length, 8, device=query_states.device, dtype=torch.float32
        ),
    }
    for start, end, probabilities, block_mask in _attention_probability_blocks(
        query_states,
        key_states,
        attention_mask,
        scaling=scaling,
        query_block_size=query_block_size,
        num_key_value_groups=num_key_value_groups,
        sliding_window=sliding_window,
    ):
        metrics = attention_metric_tensors(
            probabilities, attention_mask=block_mask, epsilon=epsilon
        )
        for name in output:
            output[name][:, start:end] = metrics[name]
    return output


class PassiveAttentionInterfaceObserver(AbstractContextManager):
    """Observe exact SDPA inputs while returning the untouched SDPA result."""

    def __init__(self, model, callback, *, expected_implementation: str = "sdpa"):
        self.model = model
        self.callback = callback
        self.expected_implementation = expected_implementation
        self.implementation: str | None = None
        self.original_interface = None
        self.previous_local_override = None
        self.had_local_override = False

    def __enter__(self):
        implementation = self.model.config._attn_implementation
        if implementation != self.expected_implementation or implementation == "eager":
            raise RuntimeError(
                f"passive Experiment 13C requires {self.expected_implementation!r}, "
                f"observed {implementation!r}"
            )
        self.implementation = implementation
        self.had_local_override = implementation in ALL_ATTENTION_FUNCTIONS._local_mapping
        if self.had_local_override:
            self.previous_local_override = ALL_ATTENTION_FUNCTIONS._local_mapping[implementation]
        self.original_interface = ALL_ATTENTION_FUNCTIONS[implementation]

        def observed_interface(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            *args,
            scaling,
            dropout=0.0,
            **kwargs,
        ):
            result = self.original_interface(
                module,
                query_states,
                key_states,
                value_states,
                attention_mask,
                *args,
                scaling=scaling,
                dropout=dropout,
                **kwargs,
            )
            self.callback(
                module,
                query_states,
                key_states,
                attention_mask,
                scaling=scaling,
                sliding_window=kwargs.get("sliding_window"),
            )
            return result

        ALL_ATTENTION_FUNCTIONS[implementation] = observed_interface
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.implementation is not None:
            if self.had_local_override:
                ALL_ATTENTION_FUNCTIONS[self.implementation] = self.previous_local_override
            else:
                del ALL_ATTENTION_FUNCTIONS[self.implementation]
        self.implementation = None
        self.original_interface = None
        self.previous_local_override = None
        return False


class Experiment13Recorder(AbstractContextManager):
    """Collect token-weighted sufficient statistics through temporary hooks."""

    def __init__(
        self,
        model,
        *,
        sequence_count: int,
        sequence_length: int,
        capture_attention: bool,
        store_tokenwise: bool,
        epsilon: float = 1e-8,
    ):
        self.model = model
        self.sequence_count = int(sequence_count)
        self.sequence_length = int(sequence_length)
        self.capture_attention = bool(capture_attention)
        self.store_tokenwise = bool(store_tokenwise)
        self.epsilon = float(epsilon)
        self.metric_names = PROBE_METRICS if capture_attention else OUTPUT_METRICS
        self.sums = {
            name: np.zeros((sequence_count, 36, 8), dtype=np.float64)
            for name in self.metric_names
        }
        self.tokenwise = (
            {
                name: np.zeros((sequence_count, 36, 8, sequence_length - 1), dtype=np.float32)
                for name in self.metric_names
            }
            if store_tokenwise
            else {}
        )
        self.valid_tokens = np.zeros(sequence_count, dtype=np.int64)
        self.weight_stats = {
            name: np.zeros((36, 8), dtype=np.float64)
            for name in (
                "w_o_local_frobenius",
                "w_o_global_frobenius",
                "w_o_local_rms",
                "w_o_global_rms",
            )
        }
        self.handles = []
        self._rows: np.ndarray | None = None
        self._valid_mask: torch.Tensor | None = None
        self._attention_observer: PassiveAttentionInterfaceObserver | None = None
        reconstruction = load_spec()["numerics"]["attention_reconstruction"]
        self._query_block_size = int(reconstruction["query_block_size"])
        self._expected_attention_implementation = reconstruction[
            "model_attention_implementation"
        ]
        self._seen_output_layers: set[int] = set()
        self._seen_attention_layers: set[int] = set()

    def __enter__(self):
        validate_hq8_model(self.model)
        try:
            if self.capture_attention:
                self._attention_observer = PassiveAttentionInterfaceObserver(
                    self.model,
                    self._capture_attention,
                    expected_implementation=self._expected_attention_implementation,
                )
                self._attention_observer.__enter__()
            for layer_index, layer in enumerate(self.model.model.layers):
                stats = output_weight_statistics(layer.self_attn.o_proj.weight.detach())
                for name, values in stats.items():
                    self.weight_stats[name][layer_index] = values

                def before_output_projection(module, args, layer_index=layer_index):
                    self._capture_output(layer_index, args[0], module.weight)

                self.handles.append(
                    layer.self_attn.o_proj.register_forward_pre_hook(before_output_projection)
                )
        except Exception:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self._attention_observer is not None:
            self._attention_observer.__exit__(exc_type, exc_value, traceback)
            self._attention_observer = None
        self._rows = None
        self._valid_mask = None
        return False

    def start_batch(self, rows: np.ndarray, valid_mask: torch.Tensor) -> None:
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1 or len(set(rows.tolist())) != rows.size:
            raise ValueError("batch rows must be a one-dimensional unique index list")
        if np.any(rows < 0) or np.any(rows >= self.sequence_count):
            raise ValueError("batch row outside recorder range")
        if valid_mask.ndim != 2 or valid_mask.shape[0] != rows.size or valid_mask.shape[1] != self.sequence_length:
            raise ValueError("valid-query mask shape mismatch")
        if not torch.all(valid_mask[:, :-1]) or torch.any(valid_mask[:, -1]):
            raise ValueError("frozen artifact must score every position except the final query")
        self._rows = rows
        self._valid_mask = valid_mask
        counts = valid_mask.sum(dim=1).detach().cpu().numpy().astype(np.int64)
        self.valid_tokens[rows] = counts
        self._seen_output_layers.clear()
        self._seen_attention_layers.clear()

    def finish_batch(self) -> None:
        if self._rows is None:
            raise RuntimeError("no active recorder batch")
        expected = set(range(36))
        if self._seen_output_layers != expected:
            raise RuntimeError("not every layer produced pre-W_O observations")
        if self.capture_attention and self._seen_attention_layers != expected:
            raise RuntimeError("not every layer produced passive attention probabilities")
        self._rows = None
        self._valid_mask = None

    def _record(self, name: str, layer: int, values: torch.Tensor) -> None:
        if self._rows is None or self._valid_mask is None:
            raise RuntimeError("recorder hook fired outside an active batch")
        if values.shape != (*self._valid_mask.shape, 8):
            raise ValueError(f"unexpected {name} metric shape {tuple(values.shape)}")
        mask = self._valid_mask.unsqueeze(-1)
        safe = torch.where(mask, values.float(), torch.zeros_like(values, dtype=torch.float32))
        sums = safe.sum(dim=1).detach().cpu().numpy().astype(np.float64)
        self.sums[name][self._rows, layer] = sums
        if self.store_tokenwise:
            token_values = safe[:, :-1].detach().cpu().numpy().astype(np.float32)
            self.tokenwise[name][self._rows, layer] = token_values.transpose(0, 2, 1)

    def _capture_output(self, layer: int, values: torch.Tensor, weight: torch.Tensor) -> None:
        metrics = output_metric_tensors(
            values,
            weight,
            epsilon=self.epsilon,
            include_similarity=self.capture_attention,
        )
        for name, tensor in metrics.items():
            self._record(name, layer, tensor)
        self._seen_output_layers.add(layer)

    def _capture_attention(
        self,
        module,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        scaling: float,
        sliding_window: int | None,
    ) -> None:
        layer = int(module.layer_idx)
        for name, tensor in passive_attention_metric_tensors(
            query_states,
            key_states,
            attention_mask,
            scaling=scaling,
            query_block_size=self._query_block_size,
            num_key_value_groups=module.num_key_value_groups,
            sliding_window=sliding_window,
            epsilon=self.epsilon,
        ).items():
            self._record(name, layer, tensor)
        self._seen_attention_layers.add(layer)


def measurement_paths(output_root: Path, split: str, kind: str) -> tuple[Path, Path | None]:
    if split not in SPLITS or kind not in MEASUREMENT_KINDS:
        raise ValueError("unsupported measurement identity")
    base = output_root / "measurements" / split
    if kind == "reproduce":
        return base / "reproduction.json", None
    if kind == "ab":
        return base / "ab_measurements.npz", base / "ab_measurements.json"
    return base / "c_measurements.npz", base / "c_measurements.json"


def required_job_id(split: str, kind: str) -> str:
    suffix = "full512" if kind in {"reproduce", "ab"} else "probe32"
    label = "repro" if kind == "reproduce" else kind
    return f"exp13a2-{label}-{split}-{suffix}"


def validate_discovery_manifest(path: Path, *, protocol_hash: str, probe_hash: str) -> dict[str, Any]:
    payload = read_json(path)
    claimed = payload.get("content_sha256")
    body = dict(payload)
    body.pop("content_sha256", None)
    if claimed != canonical_json_sha256(body):
        raise ValueError("discovery analysis manifest content hash is invalid")
    if payload.get("stage") != "discovery_frozen":
        raise ValueError("confirmation requires a frozen discovery analysis manifest")
    if payload.get("attempt") != load_spec()["attempt"]:
        raise ValueError("confirmation discovery manifest attempt identity mismatch")
    if payload.get("protocol_sha256") != protocol_hash or payload.get("probe_manifest_sha256") != probe_hash:
        raise ValueError("discovery analysis manifest input identity mismatch")
    return payload


@torch.inference_mode()
def run_reproduction(model, input_ids: torch.Tensor, *, device: torch.device, batch_size: int) -> dict[str, Any]:
    total_loss = 0.0
    sequence_losses: list[float] = []
    valid_tokens = 0
    started = time.perf_counter()
    for offset in range(0, input_ids.shape[0], batch_size):
        batch = input_ids[offset:offset + batch_size].to(device=device, dtype=torch.long)
        logits = model(input_ids=batch, use_cache=False).logits
        losses = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            batch[:, 1:].reshape(-1),
            reduction="none",
        ).view(batch.shape[0], batch.shape[1] - 1)
        total_loss += float(losses.sum().item())
        valid_tokens += int(losses.numel())
        sequence_losses.extend(losses.mean(dim=1).cpu().tolist())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "total_loss": total_loss,
        "valid_tokens": valid_tokens,
        "loss": total_loss / valid_tokens,
        "sequence_losses": sequence_losses,
        "elapsed_seconds": elapsed,
        "tokens_per_second": valid_tokens / elapsed,
    }


@torch.inference_mode()
def run_measurement(
    model,
    input_ids: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    capture_attention: bool,
    store_tokenwise: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    count, length = input_ids.shape
    recorder = Experiment13Recorder(
        model,
        sequence_count=count,
        sequence_length=length,
        capture_attention=capture_attention,
        store_tokenwise=store_tokenwise,
    )
    total_loss = 0.0
    sequence_losses = np.zeros(count, dtype=np.float64)
    started = time.perf_counter()
    with recorder:
        for offset in range(0, count, batch_size):
            end = min(offset + batch_size, count)
            batch = input_ids[offset:end].to(device=device, dtype=torch.long)
            valid_mask = torch.ones(batch.shape, dtype=torch.bool, device=device)
            valid_mask[:, -1] = False
            recorder.start_batch(np.arange(offset, end), valid_mask)
            logits = model(
                input_ids=batch,
                use_cache=False,
                output_attentions=False,
            ).logits
            recorder.finish_batch()
            losses = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                batch[:, 1:].reshape(-1),
                reduction="none",
            ).view(batch.shape[0], batch.shape[1] - 1)
            values = losses.mean(dim=1).cpu().numpy().astype(np.float64)
            sequence_losses[offset:end] = values
            total_loss += float(losses.sum().item())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    arrays = {
        "artifact_indices": np.arange(count, dtype=np.int64),
        "valid_tokens": recorder.valid_tokens,
        "sequence_losses": sequence_losses,
        **recorder.sums,
        **recorder.weight_stats,
    }
    if store_tokenwise:
        arrays.update({f"tokenwise_{name}": values for name, values in recorder.tokenwise.items()})
    metadata = {
        "valid_tokens": int(recorder.valid_tokens.sum()),
        "total_loss": total_loss,
        "loss": total_loss / int(recorder.valid_tokens.sum()),
        "elapsed_seconds": elapsed,
        "tokens_per_second": int(recorder.valid_tokens.sum()) / elapsed,
        "capture_attention": capture_attention,
        "store_tokenwise": store_tokenwise,
        "metric_names": list(recorder.metric_names),
    }
    return arrays, metadata


def validate_existing_output(data_path: Path, metadata_path: Path | None, expected: dict[str, Any]) -> bool:
    if not data_path.exists() and (metadata_path is None or not metadata_path.exists()):
        return False
    if not data_path.is_file() or (metadata_path is not None and not metadata_path.is_file()):
        raise ValueError("partial Experiment 13 output exists; refusing overwrite")
    observed = read_json(data_path if metadata_path is None else metadata_path)
    for key, value in expected.items():
        if observed.get(key) != value:
            raise ValueError(f"existing Experiment 13 output identity mismatch for {key}")
    if metadata_path is not None and observed.get("data_sha256") != sha256_path_file(data_path):
        raise ValueError("existing Experiment 13 measurement archive hash mismatch")
    return True


def measure_command(args: argparse.Namespace) -> None:
    spec = load_spec()
    split = args.split
    kind = args.kind
    if split not in SPLITS or kind not in MEASUREMENT_KINDS:
        raise ValueError("invalid Experiment 13 measurement job")
    if args.batch_size != spec["numerics"]["microbatch_size"]:
        raise ValueError("Experiment 13 production microbatch size differs from protocol")
    checkpoint = Path(args.checkpoint).resolve()
    artifact = Path(args.artifact).resolve()
    probe_path = Path(args.probe_manifest).resolve()
    output_root = Path(args.output_root).resolve()
    probe = read_json(probe_path)
    validate_probe_manifest(probe, artifact, spec)
    probe_hash = probe["content_sha256"]
    p_hash = protocol_sha256()
    if split == "confirmation":
        if not args.discovery_manifest:
            raise ValueError("confirmation requires --discovery-manifest")
        validate_discovery_manifest(
            Path(args.discovery_manifest).resolve(),
            protocol_hash=p_hash,
            probe_hash=probe_hash,
        )

    data_path, metadata_path = measurement_paths(output_root, split, kind)
    identity = {
        "format_version": 1,
        "experiment": "13-functional-difference",
        "attempt": spec["attempt"],
        "job_id": required_job_id(split, kind),
        "kind": kind,
        "split": split,
        "source_commit": git_commit(),
        "protocol_sha256": p_hash,
        "checkpoint_sha256": spec["checkpoint"]["sha256"],
        "artifact_sha256": spec["fixed_evaluation"]["sha256"],
        "probe_manifest_sha256": probe_hash,
    }
    if validate_existing_output(data_path, metadata_path, identity):
        return

    validate_checkpoint(checkpoint, spec)
    payload, artifact_hash = load_fixed_eval_artifact(artifact)
    if artifact_hash != spec["fixed_evaluation"]["sha256"]:
        raise ValueError("fixed artifact hash differs from Experiment 13 protocol")
    all_ids = payload[f"{split}_input_ids"]
    expected_shape = (
        spec["fixed_evaluation"]["sequences_per_split"],
        spec["fixed_evaluation"]["sequence_length"],
    )
    if tuple(all_ids.shape) != expected_shape:
        raise ValueError("fixed split shape differs from Experiment 13 protocol")
    selected_indices = (
        list(range(expected_shape[0])) if kind in {"reproduce", "ab"}
        else probe["indices"][split]
    )
    input_ids = all_ids[selected_indices]
    device = torch.device(args.device)
    model = Experiment8MHARForCausalLM.from_pretrained(
        str(checkpoint), dtype=dtype_from_name(args.dtype)
    ).to(device=device).eval()
    validate_hq8_model(model)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    if kind == "reproduce":
        result = run_reproduction(model, input_ids, device=device, batch_size=args.batch_size)
        repeated = run_reproduction(model, input_ids, device=device, batch_size=args.batch_size)
        repeatability = float(np.max(np.abs(
            np.asarray(result["sequence_losses"], dtype=np.float64)
            - np.asarray(repeated["sequence_losses"], dtype=np.float64)
        )))
        tolerance = spec["numerics"]["parity_tolerances"][
            "repeated_uninstrumented_per_sequence_max_abs"
        ]
        if repeatability > tolerance:
            raise ValueError(
                f"repeated uninstrumented evaluation exceeded tolerance: {repeatability}"
            )
        payload_out = {
            **identity,
            "created_at": utc_now(),
            "artifact_indices": selected_indices,
            "measurement": result,
            "repeat_measurement": repeated,
            "repeatability_per_sequence_max_abs": repeatability,
            "peak_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            ),
        }
        atomic_write_json(data_path, payload_out)
        return

    arrays, metadata = run_measurement(
        model,
        input_ids,
        device=device,
        batch_size=args.batch_size,
        capture_attention=(kind == "c"),
        store_tokenwise=(kind == "c"),
    )
    arrays["artifact_indices"] = np.asarray(selected_indices, dtype=np.int64)
    atomic_save_npz(data_path, **arrays)
    payload_out = {
        **identity,
        "created_at": utc_now(),
        "artifact_indices": selected_indices,
        "array_names": sorted(arrays),
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "measurement": metadata,
        "data_sha256": sha256_path_file(data_path),
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
    }
    atomic_write_json(metadata_path, payload_out)


def manifest_command(args: argparse.Namespace) -> None:
    manifest = write_probe_manifest(Path(args.output).resolve(), Path(args.artifact).resolve())
    print(json.dumps(manifest, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--artifact", required=True)
    manifest.add_argument("--output", required=True)
    manifest.set_defaults(func=manifest_command)
    measure = subparsers.add_parser("measure")
    measure.add_argument("--kind", choices=MEASUREMENT_KINDS, required=True)
    measure.add_argument("--split", choices=SPLITS, required=True)
    measure.add_argument("--checkpoint", required=True)
    measure.add_argument("--artifact", required=True)
    measure.add_argument("--probe-manifest", required=True)
    measure.add_argument("--discovery-manifest")
    measure.add_argument("--output-root", required=True)
    measure.add_argument("--device", default="cuda")
    measure.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    measure.add_argument("--batch-size", type=int, default=1)
    measure.set_defaults(func=measure_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
