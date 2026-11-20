"""Experiment 17A HQ8 -> expanded-GG reverse adaptation primitives.

The even query slot retains its trained aligned local projection and gains a
new zero-initialized projection from all seven non-aligned chunks. The odd
global slot and every other model component remain unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from src.experiments.experiment8_hybrid_q import (
    HYBRID_Q_GROUPS,
    Experiment8MHARForCausalLM,
    HybridQueryLinear,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiment17/protocol.json"
BRANCHES = ("hq8-control", "expanded-global")
OFFSETS = (0, 25, 50, 100, 250, 500, 1000)
BANDS = ((0, 5), (6, 11), (12, 17))


def load_spec(path: Path = CONFIG) -> dict[str, Any]:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec.get("training") is not True:
        raise ValueError("Experiment 17A must remain a training experiment")
    if (spec.get("seed"), spec.get("parent_step"), spec.get("target_step")) != (42, 2000, 3000):
        raise ValueError("Experiment 17A seed/step contract changed")
    if tuple(spec.get("optimizer_offsets", ())) != OFFSETS:
        raise ValueError("Experiment 17A milestone schedule changed")
    if tuple(map(tuple, spec.get("early_depth_bands", ()))) != BANDS:
        raise ValueError("Experiment 17A early-depth bands changed")
    if [row["id"] for row in spec.get("branches", ())] != list(BRANCHES):
        raise ValueError("Experiment 17A branch identities changed")
    if spec.get("conditions_per_split") != {"mandatory": 312, "maximum": 320}:
        raise ValueError("Experiment 17A condition matrix changed")
    run_ids = [row["id"] for row in spec.get("wandb_runs", {}).values()]
    if len(run_ids) != 6 or len(set(run_ids)) != 6:
        raise ValueError("Experiment 17A requires six unique W&B identities")
    follow_up = spec.get("follow_up", {})
    if follow_up.get("experiment") != "17B" or follow_up.get("automatic_continuation") is not False:
        raise ValueError("Experiment 17A must not automatically continue")
    return spec


def protocol_sha256(path: Path = CONFIG) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cross_indices(in_features: int, groups: int) -> torch.Tensor:
    chunk = in_features // groups
    rows = []
    for target in range(groups):
        rows.append([
            index
            for index in range(in_features)
            if not target * chunk <= index < (target + 1) * chunk
        ])
    return torch.tensor(rows, dtype=torch.long)


class _CrossQueryFunction(torch.autograd.Function):
    """Memory-bounded projection from each group's seven non-aligned chunks."""

    @staticmethod
    def forward(ctx, inputs, weight, indices):
        ctx.save_for_backward(inputs, weight, indices)
        flat = inputs.reshape(-1, inputs.shape[-1])
        outputs = []
        for group in range(weight.shape[0]):
            selected = flat.index_select(1, indices[group])
            outputs.append(selected.matmul(weight[group].transpose(0, 1)))
        result = torch.stack(outputs, dim=1)
        return result.reshape(*inputs.shape[:-1], weight.shape[0], weight.shape[1])

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight, indices = ctx.saved_tensors
        flat = inputs.reshape(-1, inputs.shape[-1])
        grad = grad_output.reshape(-1, weight.shape[0], weight.shape[1])
        grad_inputs = torch.zeros_like(flat)
        grad_weight = torch.empty_like(weight)
        for group in range(weight.shape[0]):
            selected = flat.index_select(1, indices[group])
            grad_group = grad[:, group]
            grad_weight[group] = grad_group.transpose(0, 1).matmul(selected)
            grad_inputs.index_add_(
                1,
                indices[group],
                grad_group.matmul(weight[group]),
            )
        return grad_inputs.reshape_as(inputs), grad_weight, None


class ExpandedHybridQueryLinear(nn.Module):
    """HQ8 query projection with zero-initialized cross access for even heads."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        groups: int = HYBRID_Q_GROUPS,
        *,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if groups != HYBRID_Q_GROUPS:
            raise ValueError(f"Experiment 17A requires exactly {HYBRID_Q_GROUPS} groups")
        if in_features % groups or out_features % (2 * groups):
            raise ValueError("expanded query dimensions do not match eight-group GQA")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.groups = int(groups)
        self.in_features_per_group = in_features // groups
        self.cross_features = in_features - self.in_features_per_group
        self.head_dim = out_features // (2 * groups)
        factory = {"device": device, "dtype": dtype}
        # Registration order intentionally preserves all native HQ8 parameter
        # names/order before appending the sole new parameter.
        self.local_weight = nn.Parameter(torch.empty(
            groups, self.head_dim, self.in_features_per_group, **factory))
        self.global_weight = nn.Parameter(torch.empty(
            groups, self.head_dim, in_features, **factory))
        self.local_bias = nn.Parameter(torch.empty(groups, self.head_dim, **factory)) if bias else None
        self.global_bias = nn.Parameter(torch.empty(groups, self.head_dim, **factory)) if bias else None
        self.cross_weight = nn.Parameter(torch.zeros(
            groups, self.head_dim, self.cross_features, **factory))
        # A concrete non-persistent tensor created under Transformers' meta
        # loading context can be materialized as uninitialized data. Keep a
        # registered empty cache and construct the deterministic indices only
        # on the first real-device forward.
        self.register_buffer("experiment17_cross_indices", None, persistent=False)
        self.local_group_permutation: tuple[int, ...] | None = None

    @classmethod
    def from_hybrid(cls, hybrid: HybridQueryLinear) -> "ExpandedHybridQueryLinear":
        if hybrid.local_group_permutation is not None:
            raise ValueError("cannot expand a runtime-permuted HQ8 projection")
        result = cls(
            hybrid.in_features,
            hybrid.out_features,
            hybrid.groups,
            bias=hybrid.local_bias is not None,
            device=hybrid.local_weight.device,
            dtype=hybrid.local_weight.dtype,
        )
        # Preserve Parameter identity so every old optimizer state remains
        # attached by stable name during an in-memory conversion.
        result.local_weight = hybrid.local_weight
        result.global_weight = hybrid.global_weight
        result.local_bias = hybrid.local_bias
        result.global_bias = hybrid.global_bias
        return result

    def query_components(self, inputs: torch.Tensor):
        if inputs.shape[-1] != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, got {inputs.shape[-1]}")
        grouped = inputs.reshape(*inputs.shape[:-1], self.groups, self.in_features_per_group)
        local = torch.einsum("...gi,goi->...go", grouped, self.local_weight)
        global_ = torch.einsum("...i,goi->...go", inputs, self.global_weight)
        if self.local_bias is not None:
            local = local + self.local_bias
            global_ = global_ + self.global_bias
        if self.experiment17_cross_indices is None:
            self.experiment17_cross_indices = _cross_indices(
                self.in_features, self.groups
            ).to(device=inputs.device)
        cross = _CrossQueryFunction.apply(
            inputs,
            self.cross_weight,
            self.experiment17_cross_indices,
        )
        return local, cross, global_

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        local, cross, global_ = self.query_components(inputs)
        interleaved = torch.stack((local + cross, global_), dim=-2)
        return interleaved.reshape(*inputs.shape[:-1], self.out_features)

    def dense_weight(self) -> torch.Tensor:
        dense = self.global_weight.new_zeros(self.out_features, self.in_features)
        viewed = dense.view(self.groups, 2, self.head_dim, self.in_features)
        viewed[:, 1].copy_(self.global_weight)
        indices = _cross_indices(self.in_features, self.groups).to(device=dense.device)
        for group in range(self.groups):
            start = group * self.in_features_per_group
            viewed[group, 0, :, start:start + self.in_features_per_group].copy_(self.local_weight[group])
            viewed[group, 0].index_copy_(1, indices[group], self.cross_weight[group])
        return dense


def expand_hybrid_query(model: nn.Module, branch: str) -> None:
    if branch not in BRANCHES:
        raise ValueError(f"unknown Experiment 17A branch {branch!r}")
    if int(model.config.num_key_value_heads) != 8 or int(model.config.num_attention_heads) != 16:
        raise ValueError("Experiment 17A requires 16 Q heads and 8 KV heads")
    for index, layer in enumerate(model.model.layers):
        query = layer.self_attn.q_proj
        if branch == "hq8-control":
            if not isinstance(query, HybridQueryLinear) or isinstance(query, ExpandedHybridQueryLinear):
                raise TypeError(f"layer {index} control query is not native HQ8")
        elif isinstance(query, ExpandedHybridQueryLinear):
            pass
        elif isinstance(query, HybridQueryLinear):
            layer.self_attn.q_proj = ExpandedHybridQueryLinear.from_hybrid(query)
        else:
            raise TypeError(f"layer {index} query is not an HQ8 hybrid projection")
    model.config.experiment17_branch = branch
    model.config.experiment17_query_expansion = (
        "none-native-hq8" if branch == "hq8-control" else "zero-initialized-cross-chunk"
    )


def validate_expanded_query(model: nn.Module, branch: str) -> None:
    if getattr(model.config, "experiment17_branch", None) != branch:
        raise ValueError("Experiment 17A branch metadata mismatch")
    for index, layer in enumerate(model.model.layers):
        query = layer.self_attn.q_proj
        if branch == "hq8-control":
            if not isinstance(query, HybridQueryLinear) or isinstance(query, ExpandedHybridQueryLinear):
                raise TypeError(f"layer {index} does not contain native HQ8 Q")
        else:
            if not isinstance(query, ExpandedHybridQueryLinear):
                raise TypeError(f"layer {index} does not contain expanded HQ8 Q")
            expected = (query.groups, query.head_dim, query.in_features - query.in_features_per_group)
            if tuple(query.cross_weight.shape) != expected:
                raise ValueError(f"layer {index} cross-weight shape mismatch")


class Experiment17MHARForCausalLM(Experiment8MHARForCausalLM):
    """Checkpoint-loadable Experiment 17A control or expanded branch."""

    def __init__(self, config):
        branch = getattr(config, "experiment17_branch", None)
        if branch not in BRANCHES:
            raise ValueError("Experiment17MHARForCausalLM requires a frozen branch")
        super().__init__(config)
        expand_hybrid_query(self, branch)


def _new_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, _ in model.named_parameters() if name.endswith("q_proj.cross_weight")]


def migrate_hq8_optimizer_state(
    parent_state: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
) -> dict[str, Any]:
    """Map parent HQ8 AdamW slots by stable name and zero new cross slots."""

    current = optimizer.state_dict()
    if len(parent_state.get("param_groups", ())) != 1 or len(current.get("param_groups", ())) != 1:
        raise ValueError("Experiment 17A supports exactly one optimizer parameter group")
    current_names = [name for name, _ in model.named_parameters()]
    current_parameters = dict(model.named_parameters())
    new_names = _new_parameter_names(model)
    if not new_names:
        raise ValueError("expanded branch has no cross parameters")
    old_names = [name for name in current_names if name not in set(new_names)]
    parent_ids = parent_state["param_groups"][0]["params"]
    current_ids = current["param_groups"][0]["params"]
    if len(parent_ids) != len(old_names) or len(current_ids) != len(current_names):
        raise ValueError("optimizer parameter counts do not match stable-name expansion")
    parent_by_name = dict(zip(old_names, parent_ids))
    current_by_name = dict(zip(current_names, current_ids))
    migrated_state = {}
    for name in old_names:
        parent_id = parent_by_name[name]
        if parent_id not in parent_state.get("state", {}):
            raise ValueError(f"missing optimizer state for old parameter {name}")
        migrated_state[current_by_name[name]] = copy.deepcopy(parent_state["state"][parent_id])
    for name in new_names:
        parameter = current_parameters[name]
        migrated_state[current_by_name[name]] = {
            "step": torch.tensor(0.0),
            "exp_avg": torch.zeros_like(parameter, device="cpu"),
            "exp_avg_sq": torch.zeros_like(parameter, device="cpu"),
        }
    migrated_group = copy.deepcopy(parent_state["param_groups"][0])
    migrated_group["params"] = current_ids
    optimizer.load_state_dict({"state": migrated_state, "param_groups": [migrated_group]})
    manifest = {
        "format_version": 1,
        "old_parameter_count": len(old_names),
        "new_parameter_count": len(new_names),
        "old_parameters": old_names,
        "new_parameters": new_names,
        "policy": "stable-name-copy-old-zero-moments-and-step-for-cross",
    }
    validate_optimizer_state_migration(parent_state, optimizer, model, manifest)
    return manifest


def validate_optimizer_state_migration(
    parent_state: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    manifest: dict[str, Any],
) -> None:
    current = optimizer.state_dict()
    names = [name for name, _ in model.named_parameters()]
    ids = current["param_groups"][0]["params"]
    state_by_name = {name: current["state"][pid] for name, pid in zip(names, ids)}
    new_names = manifest.get("new_parameters", [])
    old_names = manifest.get("old_parameters", [])
    parent_ids = parent_state["param_groups"][0]["params"]
    if len(parent_ids) != len(old_names):
        raise ValueError("optimizer migration manifest old-name count mismatch")
    for name, parent_id in zip(old_names, parent_ids):
        if name not in state_by_name or parent_id not in parent_state["state"]:
            raise ValueError(f"missing migrated optimizer state for {name}")
        for key, expected in parent_state["state"][parent_id].items():
            observed = state_by_name[name].get(key)
            if isinstance(expected, torch.Tensor):
                if observed is None or not torch.equal(observed.cpu(), expected.cpu()):
                    raise ValueError(f"optimizer state changed for {name}/{key}")
            elif observed != expected:
                raise ValueError(f"optimizer scalar state changed for {name}/{key}")
    for name in new_names:
        state = state_by_name.get(name, {})
        if float(state.get("step", torch.tensor(-1)).item()) != 0.0:
            raise ValueError(f"new optimizer step is not zero for {name}")
        for key in ("exp_avg", "exp_avg_sq"):
            if key not in state or torch.count_nonzero(state[key]).item() != 0:
                raise ValueError(f"new optimizer {key} is not zero for {name}")


def condition_rows(*, include_optional_m8_pair: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset in OFFSETS:
        for branch in BRANCHES:
            prefix = "control" if branch == "hq8-control" else "reverse"
            rows.append({
                "id": f"exp17a-{prefix}-t{offset:04d}",
                "branch": branch,
                "offset": offset,
                "kind": "reference",
            })
            for group in range(8):
                rows.extend([
                    {
                        "id": f"exp17a-{prefix}-zero-former-local-g{group}-t{offset:04d}",
                        "branch": branch, "offset": offset, "kind": "group",
                        "slot": "former-local", "group": group,
                    },
                    {
                        "id": f"exp17a-{prefix}-zero-old-global-g{group}-t{offset:04d}",
                        "branch": branch, "offset": offset, "kind": "group",
                        "slot": "old-global", "group": group,
                    },
                ])
            for band, (start, end) in enumerate(BANDS):
                rows.append({
                    "id": f"exp17a-{prefix}-zero-old-global-band-b{band}-t{offset:04d}",
                    "branch": branch, "offset": offset, "kind": "band",
                    "slot": "old-global", "band": band, "layers": [start, end],
                })
            if offset in (0, 1000):
                for group in range(8):
                    rows.append({
                        "id": f"exp17a-{prefix}-zero-pair-g{group}-t{offset:04d}",
                        "branch": branch, "offset": offset, "kind": "pair",
                        "slot": "pair", "group": group,
                    })
    if include_optional_m8_pair:
        for group in range(8):
            rows.append({
                "id": f"exp17a-m8-reference-zero-pair-g{group}-t1000",
                "branch": "m8-reference", "offset": 1000, "kind": "pair",
                "slot": "pair", "group": group, "optional": True,
            })
    expected = 320 if include_optional_m8_pair else 312
    if len(rows) != expected or len({row["id"] for row in rows}) != expected:
        raise AssertionError(f"Experiment 17A requires {expected} unique conditions")
    return rows


def heads_for_condition(row: dict[str, Any]) -> tuple[dict[int, tuple[int, ...]], ...]:
    if row["kind"] == "reference":
        return ({},)
    if row["kind"] == "group":
        head = 2 * int(row["group"]) + (1 if row["slot"] == "old-global" else 0)
        layers: Iterable[int] = range(36)
        heads = (head,)
    elif row["kind"] == "band":
        layers = range(int(row["layers"][0]), int(row["layers"][1]) + 1)
        heads = tuple(2 * group + 1 for group in range(8))
    elif row["kind"] == "pair":
        layers = range(36)
        group = int(row["group"])
        heads = (2 * group, 2 * group + 1)
    else:
        raise ValueError("unknown Experiment 17A condition kind")
    return ({int(layer): heads for layer in layers},)


def stream_step_digest(rank_batches: list[torch.Tensor], previous: str = "0" * 64) -> str:
    digest = hashlib.sha256(bytes.fromhex(previous))
    for rank, batch in enumerate(rank_batches):
        value = batch.detach().to(device="cpu", dtype=torch.int64).contiguous()
        digest.update(rank.to_bytes(4, "big"))
        digest.update(tuple(value.shape).__repr__().encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()
