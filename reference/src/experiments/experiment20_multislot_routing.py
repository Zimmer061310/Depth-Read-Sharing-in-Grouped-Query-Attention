"""Experiment 20: one global and multiple local MHAR depth routes.

The global route supplies global Q and dense K/V.  Each added local route
supplies the aligned MHAR chunk to a wider, physically sparse local-Q
projection.  Attention head count and output layout remain HQ8-compatible.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen3Attention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from src.experiments.experiment8_hybrid_q import (
    Experiment8MHARForCausalLM,
    HybridQueryLinear,
)


EXPERIMENT20_ROUTE_COUNTS = (3, 4, 8)
EXPERIMENT20_ROUTER_BACKENDS = (
    "shared-eager", "role-axis-eager", "tiled-r2-triton",
)


class MultiSlotHybridQueryLinear(nn.Module):
    """Interleaved local/global Q with multiple routed inputs per local head."""

    def __init__(self, base: HybridQueryLinear, local_slots: int):
        super().__init__()
        if local_slots < 1:
            raise ValueError("multi-slot local Q requires at least one local slot")
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.groups = base.groups
        self.in_features_per_group = base.in_features_per_group
        self.head_dim = base.head_dim
        self.local_slots = int(local_slots)
        self.local_weight = nn.Parameter(base.local_weight.new_zeros(
            self.groups, self.local_slots, self.head_dim,
            self.in_features_per_group,
        ))
        self.global_weight = nn.Parameter(base.global_weight.detach().clone())
        self.local_bias = (
            nn.Parameter(base.local_bias.detach().clone())
            if base.local_bias is not None else None
        )
        self.global_bias = (
            nn.Parameter(base.global_bias.detach().clone())
            if base.global_bias is not None else None
        )
        with torch.no_grad():
            self.local_weight[:, 0].copy_(base.local_weight)

    @classmethod
    def from_hybrid(cls, base: HybridQueryLinear, local_slots: int):
        return cls(base, local_slots)

    @property
    def local_input_width(self) -> int:
        return self.local_slots * self.in_features_per_group

    def forward_multi(
        self,
        global_inputs: torch.Tensor,
        local_inputs: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if len(local_inputs) != self.local_slots:
            raise ValueError(
                f"expected {self.local_slots} local inputs, got {len(local_inputs)}")
        if global_inputs.shape[-1] != self.in_features:
            raise ValueError("global query input width mismatch")
        grouped = []
        for slot, value in enumerate(local_inputs):
            if value.shape != global_inputs.shape:
                raise ValueError(f"local input {slot} shape differs from global input")
            grouped.append(value.reshape(
                *value.shape[:-1], self.groups, self.in_features_per_group))

        # Keep slot 0's exact Experiment-19 contraction. Added zero-initialized
        # slices are separate contractions so step-0 logits remain bit-identical.
        local = torch.einsum(
            "...gi,goi->...go", grouped[0], self.local_weight[:, 0])
        for slot in range(1, self.local_slots):
            local = local + torch.einsum(
                "...gi,goi->...go", grouped[slot], self.local_weight[:, slot])
        global_ = torch.einsum(
            "...i,goi->...go", global_inputs, self.global_weight)
        if self.local_bias is not None:
            local = local + self.local_bias
            global_ = global_ + self.global_bias
        interleaved = torch.stack((local, global_), dim=-2)
        return interleaved.reshape(*global_inputs.shape[:-1], self.out_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("Experiment 20 requires explicit routed local inputs")


class Experiment20Attention(Qwen3Attention):
    """HQ8 attention with one global and R-1 routed local-Q inputs."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        *,
        local_hidden_states: Sequence[torch.Tensor],
        **kwargs,
    ):
        if not isinstance(self.q_proj, MultiSlotHybridQueryLinear):
            raise TypeError("Experiment 20 requires MultiSlotHybridQueryLinear")
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(
            self.q_proj.forward_multi(
                hidden_states, local_hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling, sliding_window=self.sliding_window, **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


class Experiment20MHARForCausalLM(Experiment8MHARForCausalLM):
    """Trainable R=3/4/8 multi-slot role-specific HQ8."""

    def __init__(self, config):
        route_count = int(getattr(config, "experiment20_route_count", 0))
        if route_count not in EXPERIMENT20_ROUTE_COUNTS:
            raise ValueError(
                f"Experiment 20 route count must be one of {EXPERIMENT20_ROUTE_COUNTS}")
        if getattr(config, "experiment20_multislot_router", None) is not True:
            raise ValueError("Experiment 20 requires experiment20_multislot_router=true")
        backend = getattr(config, "experiment20_router_backend", "shared-eager")
        if backend not in EXPERIMENT20_ROUTER_BACKENDS:
            raise ValueError(f"unsupported Experiment 20 router backend: {backend}")
        super().__init__(config)
        local_slots = route_count - 1
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        try:
            for layer in self.model.layers:
                query = layer.self_attn.q_proj
                if not isinstance(query, HybridQueryLinear):
                    raise TypeError("Experiment 20 requires native HQ8 Q projections")
                layer.self_attn.q_proj = MultiSlotHybridQueryLinear.from_hybrid(
                    query, local_slots)

                routers = nn.ModuleList()
                for slot in range(local_slots):
                    original = layer.attn_res_proj
                    router = nn.Linear(
                        original.in_features, original.out_features, bias=False,
                        device=original.weight.device, dtype=original.weight.dtype,
                    )
                    if slot == 0:
                        with torch.no_grad():
                            router.weight.copy_(original.weight)
                    routers.append(router)
                layer.local_attn_res_projs = routers
                layer._experiment20_multislot_router = True
                layer._experiment20_router_backend = backend
                layer.self_attn.__class__ = Experiment20Attention
        finally:
            # Added capacity must not move the shared initialization/data RNG.
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)


def set_experiment20_router_backend(model: nn.Module, backend: str) -> None:
    if backend not in EXPERIMENT20_ROUTER_BACKENDS:
        raise ValueError(f"unsupported Experiment 20 router backend: {backend}")
    layers = model.model.layers
    if not layers or not all(
        getattr(layer, "_experiment20_multislot_router", False) for layer in layers
    ):
        raise TypeError("Experiment 20 backend requires a multi-slot model")
    for layer in layers:
        layer._experiment20_router_backend = backend


def experiment20_parameter_report(model: nn.Module) -> dict:
    layers = model.model.layers
    if not layers or not all(
        getattr(layer, "_experiment20_multislot_router", False) for layer in layers
    ):
        raise TypeError("parameter report requires an Experiment 20 model")
    route_count = int(model.config.experiment20_route_count)
    local_slots = route_count - 1
    router_parameters = sum(
        router.weight.numel()
        for layer in layers for router in layer.local_attn_res_projs
    )
    local_q_parameters = sum(
        layer.self_attn.q_proj.local_weight.numel() for layer in layers)
    hidden_size = int(model.config.hidden_size)
    groups = int(model.config.attnres_num_heads)
    head_dim = int(model.config.head_dim)
    dense_q_per_layer = hidden_size * int(model.config.num_attention_heads) * head_dim
    experiment20_q_per_layer = (
        hidden_size * groups * head_dim
        + local_slots * (hidden_size // groups) * groups * head_dim
    )
    return {
        "route_count": route_count,
        "local_slot_count": local_slots,
        "local_query_input_width": (
            local_slots * model.config.hidden_size // model.config.attnres_num_heads),
        "attention_router_count": len(layers) * route_count,
        "attention_router_parameters": router_parameters,
        "local_query_parameters": local_q_parameters,
        "query_projection_macs_per_token_per_layer": experiment20_q_per_layer,
        "query_projection_mac_reduction_vs_m8_per_token_per_layer": (
            dense_q_per_layer - experiment20_q_per_layer),
        "query_projection_mac_reduction_vs_m8_fraction": (
            (dense_q_per_layer - experiment20_q_per_layer) / dense_q_per_layer),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad),
    }
