"""Experiment 19's opt-in second attention-depth route for HQ8 local Q.

The trained global route supplies global Q and dense K/V. Only even local Q
receives the new route. Both routes share the original attention RMSNorm.
"""

from __future__ import annotations

from collections.abc import Callable

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


class Experiment19Attention(Qwen3Attention):
    """Qwen3 SDPA path with a second input only for even-numbered Q heads."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        *,
        local_hidden_states: torch.Tensor,
        **kwargs,
    ):
        if not isinstance(self.q_proj, HybridQueryLinear):
            raise TypeError("Experiment 19 requires native HybridQueryLinear")
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(
            self.q_proj.forward_dual(hidden_states, local_hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling, sliding_window=self.sliding_window, **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


class Experiment19MHARForCausalLM(Experiment8MHARForCausalLM):
    """HQ8 with an independently trainable local-Q attention router."""

    def __init__(self, config):
        if getattr(config, "experiment19_role_router", None) is not True:
            raise ValueError("Experiment 19 requires experiment19_role_router=true")
        super().__init__(config)
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        try:
            for layer in self.model.layers:
                if not isinstance(layer.self_attn.q_proj, HybridQueryLinear):
                    raise TypeError("Experiment 19 requires native HQ8 Q projections")
                original = layer.attn_res_proj
                local = nn.Linear(
                    original.in_features, original.out_features, bias=False,
                    device=original.weight.device, dtype=original.weight.dtype,
                )
                with torch.no_grad():
                    local.weight.copy_(original.weight)
                layer.local_attn_res_proj = local
                layer._experiment19_role_router = True
                layer._experiment19_collapse_router = False
                layer._experiment19_router_backend = "legacy-eager"
                # Same module registration and state-dict names, opt-in forward.
                layer.self_attn.__class__ = Experiment19Attention
        finally:
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)


def set_router_collapse(model: nn.Module, enabled: bool) -> None:
    """At frozen evaluation, feed G's route to L without modifying weights."""
    if enabled and model.training:
        raise ValueError("Experiment 19 router collapse is frozen-evaluation only")
    layers = model.model.layers
    if not layers or not all(getattr(layer, "_experiment19_role_router", False)
                             for layer in layers):
        raise TypeError("router collapse requires an Experiment 19 model")
    for layer in layers:
        layer._experiment19_collapse_router = bool(enabled)


EXPERIMENT19_ROUTER_BACKENDS = (
    "legacy-eager", "shared-eager", "role-axis-eager", "r2-triton",
)


def set_router_backend(model: nn.Module, backend: str) -> None:
    """Select an execution-only backend without changing model parameters."""
    if backend not in EXPERIMENT19_ROUTER_BACKENDS:
        raise ValueError(f"unsupported Experiment 19 router backend: {backend}")
    layers = model.model.layers
    if not layers or not all(getattr(layer, "_experiment19_role_router", False)
                             for layer in layers):
        raise TypeError("router backend requires an Experiment 19 model")
    for layer in layers:
        layer._experiment19_router_backend = backend


def get_router_backend(model: nn.Module) -> str:
    layers = model.model.layers
    if not layers or not all(getattr(layer, "_experiment19_role_router", False)
                             for layer in layers):
        raise TypeError("router backend requires an Experiment 19 model")
    values = {getattr(layer, "_experiment19_router_backend", "legacy-eager")
              for layer in layers}
    if len(values) != 1:
        raise RuntimeError("Experiment 19 layers disagree on router backend")
    return values.pop()


def experiment19_parameter_report(model: nn.Module) -> dict:
    layers = model.model.layers
    if not layers or not all(getattr(layer, "_experiment19_role_router", False)
                             for layer in layers):
        raise TypeError("parameter report requires an Experiment 19 model")
    extra = sum(layer.local_attn_res_proj.weight.numel() for layer in layers)
    return {
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "extra_router_parameters": extra,
        "attention_router_count": len(layers) * 2,
        "query_projection_mac_reduction_vs_m8_per_token":
            len(layers) * 8 * 1120 * 80 if model.config.hidden_size == 1280 else None,
    }


def route_js_per_group(
    blocks: list[torch.Tensor], partial_block: torch.Tensor,
    global_proj: nn.Linear, local_proj: nn.Linear, norm: nn.Module,
    groups: int,
) -> torch.Tensor:
    """Passive per-token JS of the two source-layer distributions, in nats.

    Returns ``[B,T,G]``. It never feeds reconstructed probabilities back into
    the model. A one-source first layer has exactly zero divergence.
    """
    values = torch.stack(blocks + [partial_block], dim=0)
    n, batch, tokens, width = values.shape
    if width % groups:
        raise ValueError("routing width must divide groups")
    if n == 1:
        return values.new_zeros((batch, tokens, groups), dtype=torch.float32)
    with torch.no_grad():
        keys = norm(values).reshape(n, batch, tokens, groups, width // groups)
        queries = (
            global_proj.weight.reshape(groups, -1),
            local_proj.weight.reshape(groups, -1),
        )
        logits = [torch.einsum("gk,nbtgk->nbtg", query, keys).float()
                  for query in queries]
        p, q = (value.softmax(dim=0) for value in logits)
        midpoint = 0.5 * (p + q)
        js = 0.5 * (
            torch.special.xlogy(p, p / midpoint)
            + torch.special.xlogy(q, q / midpoint)
        ).sum(dim=0)
    return js
