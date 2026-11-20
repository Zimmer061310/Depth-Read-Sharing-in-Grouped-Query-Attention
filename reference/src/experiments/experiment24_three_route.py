"""Experiment 24: three real depth routes, fixed two-Q/one-K/V GQA.

The legacy two-route implementation is deliberately not modified. CUDA uses
two accepted R2 forward tiles; the VJP replays only the three real BF16 graphs.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager

import torch

from src.attention_residuals.modeling_qwen3_attnres import (
    Qwen3AttnResForCausalLM, mh_block_attn_res,
)
from src.experiments.experiment23_route_topology import (
    Experiment23Attention, Experiment23MHARForCausalLM, canonical_roles,
)

RUNS = {"global-aabc": "AABC", "global-abbc": "ABBC"}
BACKENDS = ("canonical", "tiled-r2-exact")
SUBSTITUTIONS = {"collapse-b-to-a": ("B", "A"),
                 "collapse-c-to-a": ("C", "A"),
                 "collapse-c-to-b": ("C", "B")}


def training_id(variant):
    if variant not in RUNS:
        raise ValueError("only AABC and ABBC are new Experiment 24 runs")
    return f"exp24-{variant}-seed42-step2000"


class _ThreeRoutes(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, queries, norm_weight, heads, eps):
        from src.attention_residuals import mhar_triton
        if not values.is_cuda or not mhar_triton.HAS_TRITON or queries.shape[0] != 3:
            raise RuntimeError("three-route tiled forward requires CUDA/Triton and exactly three queries")
        # One shared history allocation. The fourth query is a non-trainable
        # temporary, never stored in the module or passed to the backward.
        values = values.contiguous()
        ab, _, _ = mhar_triton._roles2_accurate_forward(
            values, queries[:2].contiguous(), norm_weight, heads, eps)
        c0, _, _ = mhar_triton._roles2_accurate_forward(
            values, torch.stack((queries[2], torch.zeros_like(queries[2]))),
            norm_weight, heads, eps)
        ctx.save_for_backward(values, queries, norm_weight)
        ctx.meta = heads, eps
        return torch.cat((ab, c0[:1]), dim=0)

    @staticmethod
    def backward(ctx, upstream):
        values, queries, norm_weight = ctx.saved_tensors
        with torch.enable_grad():
            inputs = tuple(t.detach().requires_grad_(True) for t in (values, queries, norm_weight))
            outputs = canonical_roles(*inputs, *ctx.meta)
            gradients = torch.autograd.grad(outputs, inputs, tuple(upstream.unbind(0)))
        return (*(g if need else None for g, need in zip(gradients, ctx.needs_input_grad[:3])), None, None)


def three_routes(blocks, partial, projections, norm, heads, backend):
    values = torch.stack(blocks + [partial])
    queries = torch.stack([p.weight.reshape(heads, -1) for p in projections])
    if backend == "canonical":
        return canonical_roles(values, queries, norm.weight, heads, norm.variance_epsilon)
    if backend != "tiled-r2-exact":
        raise ValueError("unknown Experiment 24 backend")
    return tuple(_ThreeRoutes.apply(values, queries, norm.weight, heads,
                                   norm.variance_epsilon).unbind(0))


def layer_forward(layer, blocks, partial, *, position_embeddings, attention_mask,
                  position_ids, past_key_values, use_cache, cache_position):
    if any(getattr(layer, name) is not None for name in (
            "_mhar_partition", "_mhar_mixed_partition", "_mhar_contiguous_partition")):
        raise ValueError("Experiment 24 requires native contiguous H8 chunks")
    raw = dict(zip("ABC", three_routes(blocks, partial,
        (layer.attn_res_proj, layer.local_attn_res_proj, layer.third_attn_res_proj),
        layer.attn_res_norm, layer.attnres_num_heads, layer._experiment24_backend)))
    intervention = layer._experiment24_intervention
    if intervention is not None:
        target, source = SUBSTITUTIONS[intervention]
        raw[target] = raw[source]
    observer = getattr(layer, "_experiment24_route_probe", None)
    if observer is not None:
        with torch.no_grad():
            observer(layer, blocks, partial, raw)
    inputs = {r: layer.input_layernorm(layer._apply_gate(partial, x, "attn"))
              for r, x in raw.items()}
    out, _ = layer.self_attn(hidden_states=inputs["A"], route_inputs=inputs,
        position_embeddings=position_embeddings, attention_mask=attention_mask,
        position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache,
        cache_position=cache_position)
    blocks = blocks + [partial]
    routed = mh_block_attn_res(blocks, out, layer.mlp_res_proj,
                               layer.mlp_res_norm, layer.attnres_num_heads)
    mlp = layer.mlp(layer.post_attention_layernorm(layer._apply_gate(out, routed, "mlp")))
    return blocks + [out], mlp


class Experiment24MHARForCausalLM(Qwen3AttnResForCausalLM):
    def __init__(self, config):
        variant = getattr(config, "experiment24_variant", None)
        backend = getattr(config, "experiment24_router_backend", "canonical")
        if variant not in RUNS or getattr(config, "experiment24_three_route", False) is not True:
            raise ValueError("Experiment 24 requires an explicit frozen variant/opt-in")
        if backend not in BACKENDS:
            raise ValueError("unknown Experiment 24 backend")
        if (config.attnres_mode != "full_mh" or config.attnres_num_heads != 8
            or config.num_key_value_heads != 8 or config.num_attention_heads != 16
            or config.hidden_size % 8 or config.attention_dropout != 0
            or getattr(config, "attention_bias", False)
            or getattr(config, "attnres_gate_type", "bias") != "bias"
            or getattr(config, "attnres_use_null_source", False)):
            raise ValueError("Experiment 24 requires full_mh H8, native GQA, bias gate, no dropout/null")
        for flag in ("experiment19_role_router", "experiment20_multislot_router",
                     "experiment21_request", "experiment22_contrast", "experiment23_route_topology",
                     "experiment8_hybrid_q_groups"):
            if getattr(config, flag, False):
                raise ValueError(f"incompatible Experiment 24 flag: {flag}")
        super().__init__(config)
        # deepcopy consumes no RNG and leaves the native initializer untouched.
        for layer in self.model.layers:
            layer.local_attn_res_proj = copy.deepcopy(layer.attn_res_proj)
            layer.third_attn_res_proj = copy.deepcopy(layer.attn_res_proj)
            layer._experiment24_three_route = True
            layer._experiment24_backend = backend
            layer._experiment24_intervention = None
            layer.self_attn.__class__ = Experiment23Attention
            layer.self_attn._experiment23_routes = RUNS[variant]
            layer.self_attn._experiment23_projection = "packed"


@contextmanager
def intervention(model, name=None):
    if not isinstance(model, Experiment24MHARForCausalLM) or model.training:
        raise ValueError("three-route substitutions require frozen Experiment 24 evaluation")
    if name is not None and name not in SUBSTITUTIONS:
        raise ValueError("unknown frozen intervention")
    saved = [l._experiment24_intervention for l in model.model.layers]
    try:
        for layer in model.model.layers:
            layer._experiment24_intervention = name
        yield
    finally:
        for layer, old in zip(model.model.layers, saved):
            layer._experiment24_intervention = old


@contextmanager
def router_backend(model, backend):
    if backend not in BACKENDS:
        raise ValueError("unknown backend")
    saved = [l._experiment24_backend for l in model.model.layers]
    try:
        for layer in model.model.layers:
            layer._experiment24_backend = backend
        yield
    finally:
        for layer, old in zip(model.model.layers, saved):
            layer._experiment24_backend = old


@contextmanager
def aaab_value_interpolation(model, coefficient):
    """No class/weight/config mutation. Endpoints execute the original paths.

    The legacy route observer is called before input processing, making it a
    narrow adapter for the V-only B route. It returns no replacement data to
    the model; copy_ on the fresh routed tensor changes that input in eval only.
    """
    if (not isinstance(model, Experiment23MHARForCausalLM) or model.training
        or model.config.experiment23_variant != "global-aaab"
        or coefficient not in (0, .25, .5, .75, 1)):
        raise ValueError("interpolation requires accepted frozen AAAB and a preregistered lambda")
    saved = [(l._experiment23_collapse, getattr(l, "_experiment23_route_probe", None))
             for l in model.model.layers]
    try:
        for layer, (_, callback) in zip(model.model.layers, saved):
            layer._experiment23_collapse = coefficient == 1
            if coefficient not in (0, 1):
                def observe(l, blocks, partial, a, b, cb=callback):
                    b.copy_((1-coefficient)*b + coefficient*a)
                    if cb is not None:
                        cb(l, blocks, partial, a, b)
                layer._experiment23_route_probe = observe
        with torch.no_grad():
            yield
    finally:
        for layer, (collapse, callback) in zip(model.model.layers, saved):
            layer._experiment23_collapse = collapse
            layer._experiment23_route_probe = callback


def parameter_report(model):
    layers = model.model.layers
    width = model.config.hidden_size
    sources = sum(2*l+1 for l in range(len(layers)))
    projections = {k: sum(p.numel() for l in layers
                         for p in getattr(l.self_attn, f"{k}_proj").parameters()) for k in "qkvo"}
    return {"nominal_parameters": sum(p.numel() for p in model.parameters()),
            "active_attention_routes": 3, "dummy_parameters": 0,
            "attention_router_weights": len(layers)*width*3,
            "extra_router_weights_vs_aaab": len(layers)*width,
            "projection_parameters": projections, "projection_macs_per_token": projections,
            "ideal_route_score_and_mix_macs_per_token": 2*width*sources*3,
            "padded_forward_route_score_and_mix_macs_per_token": 2*width*sources*4,
            "extra_ideal_macs_vs_aaab": 2*width*sources,
            "tiles_per_forward": 2, "shared_history_stacks_per_attention": 1,
            "backward": "canonical_bf16_vjp_three_real_routes",
            "hardware_flops": None}
