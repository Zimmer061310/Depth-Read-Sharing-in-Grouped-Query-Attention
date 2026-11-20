"""Experiment 16A: state-preserving M8 -> aligned HQ8 adaptation.

The intervention keeps the original dense Q parameter object and applies a
persistent binary mask in the forward graph. Even query heads become local to
their aligned 160-D MHAR chunks; odd query heads remain globally dense.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from src.attention_residuals.modeling_qwen3_attnres import Qwen3AttnResForCausalLM


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiment16/protocol.json"
BRANCHES = ("control", "aligned-localized")
OFFSETS = (0, 25, 50, 100, 250, 500, 1000)
BANDS = ((0, 5), (6, 11), (12, 17))


def load_spec(path: Path = CONFIG) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if (spec.get("seed"), spec.get("parent_step"), spec.get("target_step")) != (42, 2000, 3000):
        raise ValueError("Experiment 16A seed/step contract changed")
    if tuple(spec.get("optimizer_offsets", ())) != OFFSETS:
        raise ValueError("Experiment 16A milestone schedule changed")
    if tuple(map(tuple, spec.get("early_depth_bands", ()))) != BANDS:
        raise ValueError("Experiment 16A early-depth bands changed")
    if [row["id"] for row in spec.get("branches", ())] != list(BRANCHES):
        raise ValueError("Experiment 16A branch identities changed")
    if spec["conditions_per_split"]["total"] != 245:
        raise ValueError("Experiment 16A requires exactly 245 conditions per split")
    if len(spec.get("wandb_runs", {})) != 6:
        raise ValueError("Experiment 16A requires six logical W&B identities")
    return spec


def protocol_sha256(path: Path = CONFIG) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aligned_dense_q_mask(
    in_features: int = 1280,
    out_features: int = 1280,
    *,
    groups: int = 8,
) -> torch.Tensor:
    if in_features % groups or out_features % (2 * groups):
        raise ValueError("dense Q dimensions do not match 8-group, two-query GQA")
    chunk = in_features // groups
    head_dim = out_features // (2 * groups)
    mask = torch.ones(out_features, in_features, dtype=torch.bool)
    viewed = mask.view(groups, 2, head_dim, in_features)
    viewed[:, 0].zero_()
    for group in range(groups):
        viewed[group, 0, :, group * chunk:(group + 1) * chunk] = True
    return mask


class MaskedDenseQueryLinear(nn.Linear):
    """Dense Q with a persistent, non-trainable functional connectivity mask."""

    def __init__(self, in_features, out_features, bias=False, *, mask, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        if tuple(mask.shape) != (out_features, in_features):
            raise ValueError("query mask shape mismatch")
        self.register_buffer("experiment16_query_mask", mask.to(dtype=torch.bool), persistent=True)

    @classmethod
    def from_dense(cls, dense: nn.Linear, mask: torch.Tensor) -> "MaskedDenseQueryLinear":
        result = cls(
            dense.in_features, dense.out_features, dense.bias is not None,
            mask=mask, device=dense.weight.device, dtype=dense.weight.dtype,
        )
        # Preserve the exact Parameter objects so optimizer parameter ordering,
        # state lookup, moments, and decay history remain attached to M8.
        result.weight = dense.weight
        result.bias = dense.bias
        return result

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mask = self.experiment16_query_mask
        # The control branch must remain the ordinary dense operation, both
        # for exact parent-logit parity and to avoid introducing an unnecessary
        # autograd node.  For the localized branch, recompute the masked weight
        # in backward instead of retaining a full 1280x1280 copy for every
        # layer; retaining those products caused multi-GB activation growth.
        if bool(mask.all()):
            return F.linear(inputs, self.weight, self.bias)
        return _MaskedLinearFunction.apply(inputs, self.weight, self.bias, mask)

    def materialized_weight(self) -> torch.Tensor:
        return self.weight * self.experiment16_query_mask.to(self.weight.dtype)


class _MaskedLinearFunction(torch.autograd.Function):
    """Memory-bounded masked dense linear with an unchanged forward result."""

    @staticmethod
    def forward(ctx, inputs, weight, bias, mask):
        ctx.save_for_backward(inputs, weight, mask)
        ctx.has_bias = bias is not None
        if ctx.has_bias:
            ctx.bias = bias
        return F.linear(inputs, weight * mask.to(weight.dtype), bias)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight, mask = ctx.saved_tensors
        masked_weight = weight * mask.to(weight.dtype)
        input_2d = inputs.reshape(-1, inputs.shape[-1])
        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])
        grad_input = grad_2d.matmul(masked_weight).reshape_as(inputs)
        grad_weight = grad_2d.transpose(0, 1).matmul(input_2d)
        grad_weight = grad_weight * mask.to(grad_weight.dtype)
        grad_bias = grad_2d.sum(dim=0) if ctx.has_bias else None
        return grad_input, grad_weight, grad_bias, None


def install_query_mask(
    model: nn.Module,
    branch: str,
    *,
    require_parameter_identity: bool = True,
) -> list[tuple[int, int]]:
    """Install the control or aligned mask while preserving Q Parameters.

    Returns ``(old_id, new_id)`` pairs for explicit optimizer-identity checks.
    Identity is required when converting the already-loaded continuation model,
    where optimizer state is attached to the original Parameters.  It is not
    required while ``from_pretrained`` constructs an empty checkpoint-loading
    shell before any optimizer exists; Transformers may replace Parameters in
    that low-memory initialization context.
    """
    if branch not in BRANCHES:
        raise ValueError(f"unknown Experiment 16A branch: {branch}")
    identities = []
    for layer_index, layer in enumerate(model.model.layers):
        query = layer.self_attn.q_proj
        if isinstance(query, MaskedDenseQueryLinear):
            raise ValueError(f"layer {layer_index} already has an Experiment 16 mask")
        if not isinstance(query, nn.Linear):
            raise TypeError(f"layer {layer_index} Q must be dense nn.Linear")
        mask = torch.ones_like(query.weight, dtype=torch.bool)
        if branch == "aligned-localized":
            mask = aligned_dense_q_mask(query.in_features, query.out_features).to(query.weight.device)
        old_id = id(query.weight)
        layer.self_attn.q_proj = MaskedDenseQueryLinear.from_dense(query, mask)
        identities.append((old_id, id(layer.self_attn.q_proj.weight)))
    if require_parameter_identity and any(before != after for before, after in identities):
        raise RuntimeError("Experiment 16A changed a Q Parameter identity")
    model.config.experiment16_branch = branch
    model.config.experiment16_query_mask = (
        "all-ones-dense" if branch == "control" else "even-local-odd-global"
    )
    model.config.experiment16_mask_version = 1
    return identities


def validate_installed_mask(model: nn.Module, branch: str) -> None:
    for layer_index, layer in enumerate(model.model.layers):
        query = layer.self_attn.q_proj
        if not isinstance(query, MaskedDenseQueryLinear):
            raise ValueError(f"layer {layer_index} lacks MaskedDenseQueryLinear")
        expected = torch.ones_like(query.experiment16_query_mask)
        if branch == "aligned-localized":
            expected = aligned_dense_q_mask(query.in_features, query.out_features).to(expected.device)
        if not torch.equal(query.experiment16_query_mask, expected):
            raise ValueError(f"layer {layer_index} query mask differs from frozen branch")


def condition_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset in OFFSETS:
        rows.extend([
            {"id": f"exp16a-control-t{offset:04d}", "branch": "control", "offset": offset, "kind": "reference"},
            {"id": f"exp16a-localized-t{offset:04d}", "branch": "aligned-localized", "offset": offset, "kind": "reference"},
        ])
        for group in range(8):
            rows.extend([
                {"id": f"exp16a-control-zero-g0-g{group}-t{offset:04d}", "branch": "control", "offset": offset, "kind": "group", "slot": "g0", "group": group},
                {"id": f"exp16a-control-zero-g1-g{group}-t{offset:04d}", "branch": "control", "offset": offset, "kind": "group", "slot": "g1", "group": group},
                {"id": f"exp16a-localized-zero-global-g{group}-t{offset:04d}", "branch": "aligned-localized", "offset": offset, "kind": "group", "slot": "global", "group": group},
            ])
        for band, (start, end) in enumerate(BANDS):
            rows.extend([
                {"id": f"exp16a-control-zero-g0-band-b{band}-t{offset:04d}", "branch": "control", "offset": offset, "kind": "band", "slot": "g0", "band": band, "layers": [start, end]},
                {"id": f"exp16a-control-zero-g1-band-b{band}-t{offset:04d}", "branch": "control", "offset": offset, "kind": "band", "slot": "g1", "band": band, "layers": [start, end]},
                {"id": f"exp16a-localized-zero-global-band-b{band}-t{offset:04d}", "branch": "aligned-localized", "offset": offset, "kind": "band", "slot": "global", "band": band, "layers": [start, end]},
            ])
    if len(rows) != 245 or len({row["id"] for row in rows}) != 245:
        raise AssertionError("Experiment 16A condition matrix must be 245 unique rows")
    return rows


def heads_for_condition(row: dict[str, Any]) -> tuple[dict[int, tuple[int, ...]], ...]:
    """Return one layer->head map, wrapped for future multi-mask compatibility."""
    if row["kind"] == "reference":
        return ({},)
    slot = row["slot"]
    head_offset = 0 if slot == "g0" else 1
    if slot == "global":
        head_offset = 1
    if row["kind"] == "group":
        heads = (2 * int(row["group"]) + head_offset,)
        layers: Iterable[int] = range(36)
    elif row["kind"] == "band":
        heads = tuple(2 * group + head_offset for group in range(8))
        layers = range(int(row["layers"][0]), int(row["layers"][1]) + 1)
    else:
        raise ValueError("unknown Experiment 16A condition kind")
    return ({int(layer): heads for layer in layers},)


def stream_step_digest(rank_batches: list[torch.Tensor], previous: str = "0" * 64) -> str:
    """Chain exact packed token batches into a branch-comparable step digest."""
    digest = hashlib.sha256(bytes.fromhex(previous))
    for rank, batch in enumerate(rank_batches):
        value = batch.detach().to(device="cpu", dtype=torch.int64).contiguous()
        digest.update(rank.to_bytes(4, "big"))
        digest.update(tuple(value.shape).__repr__().encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class Experiment16MHARForCausalLM(Qwen3AttnResForCausalLM):
    """Checkpoint-loadable M8 whose dense Q carries the frozen branch mask."""

    def __init__(self, config):
        branch = getattr(config, "experiment16_branch", None)
        if branch not in BRANCHES:
            raise ValueError("Experiment16MHARForCausalLM requires a frozen branch")
        super().__init__(config)
        install_query_mask(self, branch, require_parameter_identity=False)
