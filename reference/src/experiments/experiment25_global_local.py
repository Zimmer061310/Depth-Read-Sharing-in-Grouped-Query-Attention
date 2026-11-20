"""Opt-in Experiment 25: ABBC, physical grouped feature-local KV, fixed GQA.

Construction deliberately initializes the accepted native graph first, converts
it to BF16, then performs a deterministic, RNG-free coordinate transform. This
is an initializer correspondence, NOT forward or training equivalence.
"""
from __future__ import annotations

from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.experiments.experiment24_three_route import Experiment24MHARForCausalLM
from src.experiments.experiment23_route_topology import Experiment23Attention

VARIANT = "abbc-gl64-l16"
RUN_ID = "exp25-abbc-gl64-l16-seed42-step2000"
LOCAL_PAIRS = (2, 7, 12, 17, 22, 27, 32, 37)
GLOBAL_PAIRS = tuple(k for k in range(40) if k not in LOCAL_PAIRS)
GLOBAL_ROWS = GLOBAL_PAIRS + tuple(k + 40 for k in GLOBAL_PAIRS)
LOCAL_ROWS = LOCAL_PAIRS + tuple(k + 40 for k in LOCAL_PAIRS)
PERMUTATION = GLOBAL_ROWS + LOCAL_ROWS


def rms_norm(x, weight, eps):
    dtype = x.dtype
    x = x.float()
    return weight * (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)).to(dtype)


def rotate(x, cos, sin, rows):
    # Native 80D frequencies and cast boundary, sliced by whole split-half pairs.
    cos, sin = cos[..., list(rows)].unsqueeze(1), sin[..., list(rows)].unsqueeze(1)
    a, b = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-b, a), -1) * sin


def absolute_mask(query, key, cache_position, attention_mask):
    """Offset causal mask; non-square SDPA's upper-left is_causal is incorrect."""
    qlen, klen = query.shape[-2], key.shape[-2]
    if cache_position is None:
        cache_position = torch.arange(klen - qlen, klen, device=query.device)
    if cache_position.numel() != qlen:
        raise ValueError("cache positions must identify every current query")
    allowed = torch.arange(klen, device=query.device)[None, :] <= cache_position[:, None]
    allowed = allowed[None, None]
    if attention_mask is None:
        return allowed
    if attention_mask.ndim == 2:
        if attention_mask.shape[-1] != klen:
            raise ValueError("padding mask must cover the complete cached key sequence")
        return allowed & attention_mask[:, None, None, :].bool()
    if attention_mask.ndim != 4 or attention_mask.shape[-1] < klen:
        raise ValueError("unsupported attention mask")
    mask = attention_mask[..., :qlen, :klen]
    if mask.dtype == torch.bool:
        return allowed & mask
    return mask.masked_fill(~allowed, float("-inf"))


class Experiment25Attention(Experiment23Attention):
    def install(self):
        """Transform native BF16 weights; allocates no randomly initialized module."""
        q = self.q_proj.weight.detach().reshape(8, 2, 80, 1280)
        k = self.k_proj.weight.detach().reshape(8, 80, 1280)
        v = self.v_proj.weight.detach().reshape(8, 80, 1280)
        self.q_global_weight = nn.Parameter(q[:, :, list(GLOBAL_ROWS)].clone())
        self.k_global_weight = nn.Parameter(k[:, list(GLOBAL_ROWS)].clone())
        self.v_global_weight = nn.Parameter(v[:, list(GLOBAL_ROWS)].clone())
        def local(w):
            return torch.stack([w[g, ..., list(LOCAL_ROWS), 160*g:160*(g+1)]
                                for g in range(8)]).float().mul(math.sqrt(8)).to(w.dtype)
        self.q_local_weight = nn.Parameter(local(q))
        self.k_local_weight = nn.Parameter(local(k))
        self.v_local_weight = nn.Parameter(local(v))
        for role in ("q", "k"):
            old = getattr(self, role + "_norm").weight.detach()
            self.register_parameter(role + "_global_norm", nn.Parameter(old[list(GLOBAL_ROWS)].clone()))
            self.register_parameter(role + "_local_norm", nn.Parameter(old[list(LOCAL_ROWS)].clone()))
        self.branch_eps = self.q_norm.variance_epsilon
        with torch.no_grad():
            self.o_proj.weight.copy_(self.o_proj.weight.reshape(1280, 16, 80)
                                     [..., list(PERMUTATION)].reshape(1280, 1280))
        for name in ("q_proj", "k_proj", "v_proj", "q_norm", "k_norm"):
            delattr(self, name)
        self._experiment25_ablation = None
        self._experiment25_observer = None

    def project_branches(self, routes):
        # Physical even head = Q2/B, odd head = Q1/A, without changing GQA.
        qg, ql = [], []
        for slot, role in enumerate(("B", "A")):
            x = routes[role]
            qg.append(F.linear(x, self.q_global_weight[:, slot].reshape(512, 1280))
                      .reshape(*x.shape[:-1], 8, 64))
            ql.append(torch.einsum("btgc,gdc->btgd", x.reshape(*x.shape[:-1], 8, 160),
                                   self.q_local_weight[:, slot]))
        qg = torch.stack(qg, -2).flatten(-3, -2).transpose(1, 2)
        ql = torch.stack(ql, -2).flatten(-3, -2).transpose(1, 2)
        outputs = [qg, ql]
        for name, role in (("k", "B"), ("v", "C")):
            x = routes[role]
            g = F.linear(x, getattr(self, name + "_global_weight").reshape(512, 1280))
            l = torch.einsum("btgc,gdc->btgd", x.reshape(*x.shape[:-1], 8, 160),
                             getattr(self, name + "_local_weight"))
            outputs.extend((g.reshape(*x.shape[:-1], 8, 64).transpose(1, 2), l.transpose(1, 2)))
        return tuple(outputs)

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, cache_position=None, route_inputs=None, **kwargs):
        if route_inputs is None:
            raise ValueError("Experiment 25 requires three real ABBC route inputs")
        if self._experiment25_ablation is not None and self.training:
            raise ValueError("branch ablation is evaluation-only")
        qg, ql, kg, kl, vg, vl = self.project_branches(route_inputs)
        cos, sin = position_embeddings
        qg = rotate(rms_norm(qg, self.q_global_norm, self.branch_eps), cos, sin, GLOBAL_ROWS)
        ql = rotate(rms_norm(ql, self.q_local_norm, self.branch_eps), cos, sin, LOCAL_ROWS)
        kg = rotate(rms_norm(kg, self.k_global_norm, self.branch_eps), cos, sin, GLOBAL_ROWS)
        kl = rotate(rms_norm(kl, self.k_local_norm, self.branch_eps), cos, sin, LOCAL_ROWS)
        key, value = torch.cat((kg, kl), -1), torch.cat((vg, vl), -1)
        if past_key_values is not None:
            # Exactly one persistent packed 8-head K/V update, never 16-head copies.
            key, value = past_key_values.update(key, value, self.layer_idx,
                {"cos": cos, "sin": sin, "cache_position": cache_position})
        kg, kl = key[..., :64], key[..., 64:]
        vg, vl = value[..., :64], value[..., 64:]
        mask = absolute_mask(qg, kg, cache_position, attention_mask)
        def attend(q, k, v, dim):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                dropout_p=0.0, is_causal=False, scale=dim**-0.5, enable_gqa=True)
        og, ol = attend(qg, kg, vg, 64), attend(ql, kl, vl, 16)
        if self._experiment25_observer is not None:
            with torch.no_grad():
                self._experiment25_observer(self, qg.detach(), ql.detach(), kg.detach(),
                    kl.detach(), mask, og.detach(), ol.detach())
        if self._experiment25_ablation == "zero-local":
            ol = torch.zeros_like(ol)
        elif self._experiment25_ablation == "zero-global":
            og = torch.zeros_like(og)
        out = torch.cat((og, ol), -1).transpose(1, 2).contiguous()
        out = out.reshape(*hidden_states.shape[:-1], 1280)
        return self.o_proj(out), None


class Experiment25MHARForCausalLM(Experiment24MHARForCausalLM):
    def __init__(self, config):
        if (getattr(config, "experiment25_global_local", False) is not True
            or getattr(config, "experiment25_variant", None) != VARIANT):
            raise ValueError("Experiment 25 requires explicit frozen opt-in")
        if (config.hidden_size != 1280 or config.head_dim != 80 or config.max_position_embeddings != 2048
            or config.rms_norm_eps != 1e-6 or config.rope_theta != 10000.0
            or getattr(config, "rope_scaling", None) is not None):
            raise ValueError("Experiment 25 requires native 1280/80 geometry, norms and RoPE")
        # An internal adapter reuses the accepted three-route layer without adding
        # a fourth route or changing its source/history/MLP graph.
        config.experiment24_three_route = True
        config.experiment24_variant = "global-abbc"
        config.experiment24_router_backend = getattr(config, "experiment25_router_backend", "canonical")
        super().__init__(config)
        construction_dtype = next(self.parameters()).dtype
        self.to(dtype=torch.bfloat16)
        for layer in self.model.layers:
            layer.self_attn.__class__ = Experiment25Attention
            layer.self_attn.install()
        # Preserve HF's requested load dtype so an FP32 diagnostic checkpoint
        # roundtrips without rounding trained values during from_pretrained.
        # Fresh values still have exactly the preregistered BF16 boundary.
        self.to(dtype=construction_dtype)


@contextmanager
def intervention(model, name=None):
    if not isinstance(model, Experiment25MHARForCausalLM) or model.training:
        raise ValueError("Experiment 25 interventions require frozen evaluation")
    if name not in (None, "zero-local", "zero-global"):
        raise ValueError("unknown frozen branch intervention")
    saved = [l.self_attn._experiment25_ablation for l in model.model.layers]
    try:
        for layer in model.model.layers:
            layer.self_attn._experiment25_ablation = name
        yield
    finally:
        for layer, old in zip(model.model.layers, saved):
            layer.self_attn._experiment25_ablation = old


def parameter_report(model):
    attentions = [l.self_attn for l in model.model.layers]
    global_qkv = sum(getattr(a, r + "_global_weight").numel() for a in attentions for r in "qkv")
    local_qkv = sum(getattr(a, r + "_local_weight").numel() for a in attentions for r in "qkv")
    return {"nominal_parameters": sum(p.numel() for p in model.parameters()),
            "global_qkv": global_qkv, "local_qkv": local_qkv,
            "qkv": global_qkv + local_qkv, "qkv_saving_fraction": .175,
            "qk_norm": 160 * len(attentions), "independent_maps_per_layer": 32,
            "cache_bytes_per_token": 2560 * len(attentions),
            "whole_model_or_runtime_saving_claim": False}


def cache_report(cache):
    """Logical payload and deduplicated backing storage, excluding workspace."""
    tensors = []
    for layer in cache.layers:
        tensors.extend((layer.keys, layer.values))
    storages = {(t.device, t.untyped_storage().data_ptr()): t.untyped_storage().nbytes()
                for t in tensors if t is not None}
    return {"logical_bytes": sum(t.numel()*t.element_size() for t in tensors if t is not None),
            "unique_storage_bytes": sum(storages.values()),
            "tensors": [{"shape": list(t.shape), "stride": list(t.stride()), "dtype": str(t.dtype)}
                        for t in tensors if t is not None],
            "persistent_mhar_decode_state_bytes": 0}
