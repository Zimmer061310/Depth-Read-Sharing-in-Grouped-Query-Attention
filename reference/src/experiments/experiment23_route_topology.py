"""Opt-in global-width Q/K/V route topology; native two-Q/one-KV GQA.

Logical Q1 is the odd slot; Q2 is even. Model construction starts with native
M8 and copies B without consuming RNG. GL references slice that same dense
initialization using the physical HQ8 projection, never a padded dense mask.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention, ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb,
    eager_attention_forward,
)

from src.attention_residuals.modeling_qwen3_attnres import (
    Qwen3AttnResForCausalLM, mh_block_attn_res, mh_block_attn_res_roles,
)
from src.experiments.experiment8_hybrid_q import HybridQueryLinear

RUNS = {
    **{f"global-{routes.lower()}": ("GGGG", routes)
       for routes in ("AAAA", "ABAA", "AABA", "AAAB", "AABB", "ABBA")},
    "r2-gl-abaa": ("GLGG", "ABAA"),
    "hq8-gl-aaaa": ("GLGG", "AAAA"),
}
ROUTER_BACKENDS = ("shared-eager", "r2-triton")
PROJECTION_BACKENDS = ("canonical", "eager-matched", "packed")


def canonical_roles(values, queries, norm_weight, heads, eps):
    """Explicit canonical rounding boundaries, including backward casts."""
    n, batch, tokens, width = values.shape
    values32 = values.float()
    keys = (values32 * torch.rsqrt(values32.pow(2).mean(-1, keepdim=True) + eps)).to(values.dtype)
    keys = (norm_weight * keys).view(n, batch, tokens, heads, width // heads)
    grouped = values.view(n, batch, tokens, heads, width // heads)
    return tuple(torch.einsum("nbth,nbthk->bthk",
        torch.einsum("hk,nbthk->nbth", query, keys).softmax(0), grouped).reshape(batch,tokens,width)
        for query in queries.unbind(0))


class _ExactBackwardRoles(torch.autograd.Function):
    """Fast R2 forward, canonical BF16 VJP; opt-in to Experiment23 only.

    The inherited custom Triton backward recomputes smooth FP32 derivatives
    and does not reproduce eager BF16 intermediate gradient casts. Recompute
    the canonical graph for its VJP, without changing legacy implementations.
    This trades backward speed for the frozen numerical-equivalence contract.
    """
    @staticmethod
    def forward(ctx, values, queries, norm_weight, heads, eps):
        from src.attention_residuals import mhar_triton
        if not values.is_cuda or not mhar_triton.HAS_TRITON:
            raise RuntimeError("Experiment23 R2 forward requires CUDA/Triton")
        outputs, _, _ = mhar_triton._roles2_accurate_forward(
            values.contiguous(), queries.contiguous(), norm_weight, heads, eps)
        ctx.save_for_backward(values, queries, norm_weight)
        ctx.meta = (heads, eps)
        return outputs

    @staticmethod
    def backward(ctx, upstream):
        values, queries, norm_weight = ctx.saved_tensors
        heads, eps = ctx.meta
        with torch.enable_grad():
            inputs = tuple(t.detach().requires_grad_(True) for t in (values,queries,norm_weight))
            outputs = canonical_roles(*inputs, heads, eps)
            gradients = torch.autograd.grad(outputs, inputs, tuple(upstream.unbind(0)))
        return (*tuple(g if needed else None for g,needed in zip(gradients,ctx.needs_input_grad[:3])),
                None, None)


def exact_backward_routes(blocks, partial, projs, norm, heads):
    values = torch.stack(blocks + [partial])
    queries = torch.stack([proj.weight.reshape(heads,-1) for proj in projs])
    outputs = _ExactBackwardRoles.apply(values, queries, norm.weight, heads, norm.variance_epsilon)
    return tuple(outputs.unbind(0))


def training_id(variant: str) -> str:
    if variant not in RUNS:
        raise ValueError(f"unknown Experiment 23 class: {variant}")
    return f"exp23-{variant}-seed42-step2000"


def alias_routes(routes: str) -> str:
    """Q-slot exchange plus A/B relabeling. This is only a GG symmetry."""
    if len(routes) != 4 or set(routes) - set("AB"):
        raise ValueError("route string must specify Q1/Q2/K/V")
    swapped = routes[1] + routes[0] + routes[2:]
    return swapped.translate(str.maketrans("AB", "BA"))


class Experiment23Attention(Qwen3Attention):
    def project_qkv(self, route_inputs):
        q1, q2, k, v = (route_inputs[r] for r in self._experiment23_routes)
        shape = (*q1.shape[:-1], 8, 2, self.head_dim)
        if isinstance(self.q_proj, HybridQueryLinear):
            query = self.q_proj.forward_dual(q1, q2)
        elif (self._experiment23_projection in ("packed", "eager-matched")
              and self._experiment23_routes[0] == self._experiment23_routes[1]):
            # The eager numerical oracle mirrors the deployed GEMM shape.
            # "canonical" remains the common fragmented slot oracle for
            # initialization/graph proofs, not cross-GEMM BF16 bit parity.
            query = (self.q_proj(q1) if self._experiment23_projection == "packed"
                     else F.linear(q1, self.q_proj.weight, self.q_proj.bias))
        else:
            weights = self.q_proj.weight.reshape(8, 2, self.head_dim, q1.shape[-1])
            bias = self.q_proj.bias
            bias = None if bias is None else bias.reshape(8, 2, self.head_dim)
            parts = []
            for slot, inputs in ((0, q2), (1, q1)):
                # Two 640-output GEMMs, not two full Q projections.
                w = weights[:, slot].reshape(8 * self.head_dim, -1)
                b = None if bias is None else bias[:, slot].reshape(-1)
                parts.append(F.linear(inputs, w, b).reshape(*inputs.shape[:-1], 8, self.head_dim))
            query = torch.stack(parts, dim=-2).reshape(*q1.shape[:-1], -1)
        return query.reshape(shape), self.k_proj(k), self.v_proj(v)

    def forward(self, hidden_states, position_embeddings, attention_mask,
                past_key_values=None, cache_position=None, *, route_inputs, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query, key, value = self.project_qkv(route_inputs)
        query = self.q_norm(query.reshape(hidden_shape)).transpose(1, 2)
        key = self.k_norm(key.reshape(hidden_shape)).transpose(1, 2)
        value = value.reshape(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx,
                {"sin": sin, "cos": cos, "cache_position": cache_position})
        interface = (eager_attention_forward if self.config._attn_implementation == "eager"
                     else ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation])
        output, weights = interface(self, query, key, value, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling, sliding_window=self.sliding_window, **kwargs)
        output = output.reshape(*input_shape, -1).contiguous()
        observer = getattr(self, "_experiment23_attention_probe", None)
        if observer is not None:
            with torch.no_grad():
                observer(self, query, key, attention_mask, output)
        return self.o_proj(output), weights


def layer_forward(layer, blocks, partial, *, position_embeddings, attention_mask,
                  position_ids, past_key_values, use_cache, cache_position):
    """One history update per sublayer; no route becomes an extra carrier."""
    if any(getattr(layer, name) is not None for name in (
        "_mhar_partition", "_mhar_mixed_partition", "_mhar_contiguous_partition")):
        raise ValueError("Experiment 23 requires native contiguous H8 chunks")
    if layer._experiment23_active_b and not layer._experiment23_collapse:
        if layer._experiment23_router_backend == "r2-triton":
            a, b = exact_backward_routes(blocks, partial,
                (layer.attn_res_proj, layer.local_attn_res_proj), layer.attn_res_norm,
                layer.attnres_num_heads)
        else:
            a, b = mh_block_attn_res_roles(blocks, partial,
                (layer.attn_res_proj, layer.local_attn_res_proj), layer.attn_res_norm,
                layer.attnres_num_heads, backend=layer._experiment23_router_backend)
    else:
        a = mh_block_attn_res(blocks, partial, layer.attn_res_proj,
                              layer.attn_res_norm, layer.attnres_num_heads)
        b = a
    observer = getattr(layer, "_experiment23_route_probe", None)
    if observer is not None:
        with torch.no_grad():
            observer(layer, blocks, partial, a, b)
    inputs = {"A": layer.input_layernorm(layer._apply_gate(partial, a, "attn"))}
    if layer._experiment23_active_b:
        inputs["B"] = layer.input_layernorm(layer._apply_gate(partial, b, "attn"))
    attn_out, _ = layer.self_attn(hidden_states=inputs["A"], route_inputs=inputs,
        position_embeddings=position_embeddings, attention_mask=attention_mask,
        position_ids=position_ids, past_key_values=past_key_values,
        use_cache=use_cache, cache_position=cache_position)
    blocks = blocks + [partial]
    partial = attn_out
    routed = mh_block_attn_res(blocks, partial, layer.mlp_res_proj,
                               layer.mlp_res_norm, layer.attnres_num_heads)
    mlp = layer.mlp(layer.post_attention_layernorm(layer._apply_gate(partial, routed, "mlp")))
    return blocks + [partial], mlp


class Experiment23MHARForCausalLM(Qwen3AttnResForCausalLM):
    def __init__(self, config):
        variant = getattr(config, "experiment23_variant", None)
        if variant not in RUNS or getattr(config, "experiment23_route_topology", False) is not True:
            raise ValueError("Experiment 23 requires a frozen variant and opt-in flag")
        if (config.attnres_mode != "full_mh" or config.attnres_num_heads != 8
            or config.num_key_value_heads != 8 or config.num_attention_heads != 16
            or config.hidden_size % 8 or config.attention_dropout != 0
            or getattr(config,"attention_bias",False)
            or getattr(config, "attnres_gate_type", "bias") != "bias"
            or getattr(config, "attnres_use_null_source", False)):
            raise ValueError("Experiment 23 requires native H8, two Q per KV, bias gate, no null/dropout")
        for flag in ("experiment19_role_router", "experiment20_multislot_router",
                     "experiment21_request", "experiment22_contrast", "experiment8_hybrid_q_groups"):
            if getattr(config, flag, False):
                raise ValueError(f"incompatible Experiment 23 flag: {flag}")
        router = getattr(config, "experiment23_router_backend", "shared-eager")
        projection = getattr(config, "experiment23_projection_backend", "packed")
        if router not in ROUTER_BACKENDS or projection not in PROJECTION_BACKENDS:
            raise ValueError("unsupported Experiment 23 execution backend")
        super().__init__(config)
        features, routes = RUNS[variant]
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        try:
            for layer in self.model.layers:
                layer.local_attn_res_proj = copy.deepcopy(layer.attn_res_proj)
                layer._experiment23_route_topology = True
                layer._experiment23_active_b = "B" in routes
                layer._experiment23_collapse = False
                layer._experiment23_router_backend = router
                layer.self_attn.__class__ = Experiment23Attention
                layer.self_attn._experiment23_routes = routes
                layer.self_attn._experiment23_projection = projection
                if features == "GLGG":
                    layer.self_attn.q_proj = HybridQueryLinear.from_dense(layer.self_attn.q_proj, 8)
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)


def set_collapse_b(model, enabled):
    if enabled and model.training:
        raise ValueError("route collapse is frozen evaluation only")
    if not isinstance(model, Experiment23MHARForCausalLM):
        raise TypeError("collapse requires Experiment 23")
    for layer in model.model.layers:
        layer._experiment23_collapse = bool(enabled)


@contextmanager
def execution_backend(model, *, router=None, projection=None):
    if router is not None and router not in ROUTER_BACKENDS:
        raise ValueError("unknown router backend")
    if projection is not None and projection not in PROJECTION_BACKENDS:
        raise ValueError("unknown projection backend")
    saved = [(l._experiment23_router_backend, l.self_attn._experiment23_projection)
             for l in model.model.layers]
    try:
        for layer in model.model.layers:
            if router is not None:
                layer._experiment23_router_backend = router
            if projection is not None:
                layer.self_attn._experiment23_projection = projection
        yield
    finally:
        for layer, (r, p) in zip(model.model.layers, saved):
            layer._experiment23_router_backend = r
            layer.self_attn._experiment23_projection = p


def parameter_report(model):
    layers = model.model.layers
    nominal = sum(p.numel() for p in model.parameters())
    inactive = sum(l.local_attn_res_proj.weight.numel() for l in layers
                   if not l._experiment23_active_b and l.local_attn_res_proj is not l.attn_res_proj)
    projections = {key: sum(p.numel() for l in layers
                           for p in getattr(l.self_attn, f"{key}_proj").parameters())
                   for key in ("q", "k", "v", "o")}
    width = model.config.hidden_size
    sources = sum(2 * l + 1 for l in range(len(layers)))
    active = 1 + int(layers[0]._experiment23_active_b)
    return {"nominal_parameters": nominal, "active_parameters": nominal - inactive,
        "inactive_b_parameters": inactive, "projection_parameters": projections,
        "projection_macs_per_token": projections, "active_attention_routes": active,
        "attention_router_weights": sum(p.numel() for p in {id(p):p for l in layers
            for module in (l.attn_res_proj,l.local_attn_res_proj) for p in module.parameters()}.values()),
        "attention_route_score_macs_per_token": sources * width * active,
        "attention_route_mixture_macs_per_token": sources * width * active,
        "arithmetic_exclusions": ["source and input RMSNorm", "softmax", "memory movement",
            "attention score/value matmuls", "MLP", "MLP/final routers", "output logits",
            "backward/recomputation", "optimizer"],
        "hardware_flops": None}
