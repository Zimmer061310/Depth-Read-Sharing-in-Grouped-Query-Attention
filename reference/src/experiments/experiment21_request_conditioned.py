"""Experiment 21: request-conditioned local MHAR depth routing.

Native R2 RS-HQ8 routes the local (even) Q head over the depth history with a
fixed learned score vector and routes the global (odd) Q head / dense K,V over
the same history.  Experiment 21 keeps that graph and adds a rank-4 bilinear
term to the *local* score only, scoring each candidate source against a request
built from the current pre-routing representation:

    z      = chunk_g(N(partial_block))            # last source key, reused
    u      = z U                                  # [B,T,G,4]
    v_s    = k_s V                                # [S,B,T,G,4]
    score  = score_L0 + dot(u, v_s) / sqrt(4)
    aL     = softmax_sources(score)

The global score, Q/K/V projections, gates, norms, MLP and all source updates
are unchanged.  ``U ~ Normal(0, 0.02)`` and ``V = 0``, so the added score starts
exactly zero and ``U`` still receives gradient once ``V`` moves off zero.

Three variants share one class:

* ``r2``              — native two-route R2 control, no added matrices;
* ``constant-request``— added matrices scored against a fixed request ``e1``
                        (an input-independent vector absorbable into the static
                        score; a parameterization control, not a mechanism);
* ``request-local``   — the proposed token-dependent request.

No implementation of this module is authorized to launch paid compute.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from src.experiments.experiment8_hybrid_q import (
    Experiment8MHARForCausalLM,
    HybridQueryLinear,
)
from src.experiments.experiment19_role_specific_routing import Experiment19Attention


EXPERIMENT21_VARIANTS = ("r2", "constant-request", "request-local")
EXPERIMENT21_ADDED_SCORE_VARIANTS = ("constant-request", "request-local")
EXPERIMENT21_REQUEST_RANK = 4
EXPERIMENT21_U_SEED = 21042
EXPERIMENT21_U_STD = 0.02


def _request_route(
    blocks: list[torch.Tensor],
    partial_block: torch.Tensor,
    local_proj: nn.Linear,
    norm: nn.Module,
    num_heads: int,
    request_u: torch.Tensor | None,
    request_v: torch.Tensor | None,
    *,
    constant_request: bool,
    rank: int,
) -> torch.Tensor:
    """Canonical eager local route with an optional request-conditioned term.

    The no-request path reproduces ``_mh_block_attn_res_kernel`` exactly: the
    same full-width RMSNorm, the same per-head score einsum, the same source
    softmax and the same value-mixing einsum.  When the request term is exactly
    zero this returns a bit-identical local mixture.
    """
    sources = torch.stack(list(blocks) + [partial_block], dim=0)
    n, batch, tokens, width = sources.shape
    if width % num_heads:
        raise ValueError("routing width must divide the routing-head count")
    chunk = width // num_heads
    keys = norm(sources).view(n, batch, tokens, num_heads, chunk)
    values = sources.view(n, batch, tokens, num_heads, chunk)
    query = local_proj.weight.view(num_heads, chunk)
    logits = torch.einsum("hk,nbthk->nbth", query, keys)
    if request_u is not None:
        if request_v is None:
            raise ValueError("request conditioning requires both U and V")
        projected = torch.einsum("nbthk,hkr->nbthr", keys, request_v)
        if constant_request:
            fixed = keys.new_zeros(chunk)
            fixed[0] = 1.0
            request = torch.einsum("k,hkr->hr", fixed, request_u)
            delta = torch.einsum("hr,nbthr->nbth", request, projected)
        else:
            request = torch.einsum("bthk,hkr->bthr", keys[-1], request_u)
            delta = torch.einsum("bthr,nbthr->nbth", request, projected)
        logits = logits + delta / math.sqrt(rank)
    weights = logits.softmax(dim=0)
    mixed = torch.einsum("nbth,nbthk->bthk", weights, values)
    return mixed.reshape(batch, tokens, width)


def request_conditioned_local_route(
    blocks: list[torch.Tensor],
    partial_block: torch.Tensor,
    local_proj: nn.Linear,
    norm: nn.Module,
    num_heads: int,
    request_u: torch.Tensor | None,
    request_v: torch.Tensor | None,
    *,
    variant: str,
    rank: int = EXPERIMENT21_REQUEST_RANK,
) -> torch.Tensor:
    if variant not in EXPERIMENT21_VARIANTS:
        raise ValueError(f"unsupported Experiment 21 variant: {variant}")
    if variant == "r2":
        return _request_route(
            blocks, partial_block, local_proj, norm, num_heads,
            None, None, constant_request=False, rank=rank)
    return _request_route(
        blocks, partial_block, local_proj, norm, num_heads,
        request_u, request_v,
        constant_request=(variant == "constant-request"), rank=rank)


class Experiment21MHARForCausalLM(Experiment8MHARForCausalLM):
    """Native R2 RS-HQ8 with an optional request-conditioned local score."""

    def __init__(self, config):
        variant = getattr(config, "experiment21_variant", None)
        if variant not in EXPERIMENT21_VARIANTS:
            raise ValueError(
                f"Experiment 21 variant must be one of {EXPERIMENT21_VARIANTS}")
        if getattr(config, "experiment21_request", None) is not True:
            raise ValueError("Experiment 21 requires experiment21_request=true")
        super().__init__(config)
        if getattr(config, "experiment20_multislot_router", None):
            raise ValueError("Experiment 21 must not enable the multi-slot router")
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        try:
            added = variant in EXPERIMENT21_ADDED_SCORE_VARIANTS
            rank = int(getattr(config, "experiment21_request_rank",
                               EXPERIMENT21_REQUEST_RANK))
            for layer_index, layer in enumerate(self.model.layers):
                if not isinstance(layer.self_attn.q_proj, HybridQueryLinear):
                    raise TypeError("Experiment 21 requires native HQ8 Q projections")
                original = layer.attn_res_proj
                local = nn.Linear(
                    original.in_features, original.out_features, bias=False,
                    device=original.weight.device, dtype=original.weight.dtype,
                )
                with torch.no_grad():
                    local.weight.copy_(original.weight)
                layer.local_attn_res_proj = local
                num_heads = int(config.attnres_num_heads)
                chunk = int(config.hidden_size) // num_heads
                if added:
                    generator = torch.Generator(device="cpu").manual_seed(
                        EXPERIMENT21_U_SEED + layer_index)
                    request_u = (
                        torch.randn((num_heads, chunk, rank), generator=generator,
                                    dtype=torch.float32) * EXPERIMENT21_U_STD
                    ).to(device=original.weight.device, dtype=original.weight.dtype)
                    request_v = torch.zeros(
                        (num_heads, chunk, rank),
                        device=original.weight.device, dtype=original.weight.dtype)
                    layer.request_u = nn.Parameter(request_u)
                    layer.request_v = nn.Parameter(request_v)
                else:
                    layer.request_u = None
                    layer.request_v = None
                layer._experiment21_request = True
                layer._experiment21_variant = variant
                layer._experiment21_rank = rank
                layer.self_attn.__class__ = Experiment19Attention
        finally:
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)


def set_request_off(model: nn.Module, enabled: bool) -> None:
    """Frozen-evaluation-only switch: drop the added request term."""
    if enabled and model.training:
        raise ValueError("Experiment 21 request-off is frozen-evaluation only")
    layers = model.model.layers
    if not layers or not all(
        getattr(layer, "_experiment21_request", False) for layer in layers
    ):
        raise TypeError("request-off requires an Experiment 21 model")
    for layer in layers:
        layer._experiment21_request_off = bool(enabled)


def experiment21_parameter_report(model: nn.Module) -> dict:
    layers = model.model.layers
    if not layers or not all(
        getattr(layer, "_experiment21_request", False) for layer in layers
    ):
        raise TypeError("parameter report requires an Experiment 21 model")
    variant = model.config.experiment21_variant
    added = variant in EXPERIMENT21_ADDED_SCORE_VARIANTS
    rank = int(getattr(model.config, "experiment21_request_rank",
                       EXPERIMENT21_REQUEST_RANK))
    width = int(model.config.hidden_size)
    chunk = width // int(model.config.attnres_num_heads)
    added_parameters = sum(
        parameter.numel()
        for layer in layers
        for parameter in ((layer.request_u, layer.request_v) if added
                          else ())
        if parameter is not None
    )
    router_parameters = sum(
        layer.local_attn_res_proj.weight.numel() for layer in layers)
    extra_router_macs = 0
    if added:
        for layer_index in range(len(layers)):
            sources = 2 * layer_index + 1
            extra_router_macs += 8 * (
                (sources + 1) * chunk * rank + sources * rank)
    return {
        "variant": variant,
        "added_score_parameters": added_parameters,
        "extra_local_router_parameters": router_parameters,
        "request_rank": rank,
        "added_router_macs_per_token": extra_router_macs,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad),
    }
