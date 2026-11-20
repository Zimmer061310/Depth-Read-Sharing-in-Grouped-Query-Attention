"""Experiment 22: local-global contrast readout.

Both routers stay native R2.  Only the readout sent to the local (even) Q head
changes, replacing the local route ``cL`` with a stabilized contrast against the
global route ``cG``:

    dL = cL + (cL - cG)          # == 2*cL - cG; the mandated operation order
    xG = input_layernorm(apply_attn_gate(partial_block, cG))
    xL = input_layernorm(apply_attn_gate(partial_block, dL))

The subtraction happens *before* the existing attention gate and input RMSNorm,
on the full ``[B,T,1280]`` route tensors.  ``cG`` is not detached.  When
``cL == cG`` the readout reduces to ``cL``, so shared information is retained
and initialization is not degenerate.

The frozen ``contrast-off`` intervention sets ``dL = cL`` at evaluation time
without changing weights.  There are no new trainable parameters; every common
weight is initialized exactly as the fresh R2 control.

No implementation of this module is authorized to launch paid compute.
"""

from __future__ import annotations

import torch
from torch import nn

from src.experiments.experiment8_hybrid_q import (
    Experiment8MHARForCausalLM,
    HybridQueryLinear,
)
from src.experiments.experiment19_role_specific_routing import Experiment19Attention


EXPERIMENT22_VARIANTS = ("contrast-local",)


def contrast_readout(
    global_route: torch.Tensor,
    local_route: torch.Tensor,
    *,
    off: bool = False,
) -> torch.Tensor:
    """Return the local readout ``dL`` for the given routes."""
    if off:
        return local_route
    return local_route + (local_route - global_route)


class Experiment22MHARForCausalLM(Experiment8MHARForCausalLM):
    """Native R2 RS-HQ8 with the stabilized local-global contrast readout."""

    def __init__(self, config):
        if getattr(config, "experiment22_variant", None) not in EXPERIMENT22_VARIANTS:
            raise ValueError(
                f"Experiment 22 variant must be one of {EXPERIMENT22_VARIANTS}")
        if getattr(config, "experiment22_contrast", None) is not True:
            raise ValueError("Experiment 22 requires experiment22_contrast=true")
        super().__init__(config)
        if getattr(config, "experiment20_multislot_router", None):
            raise ValueError("Experiment 22 must not enable the multi-slot router")
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        try:
            for layer in self.model.layers:
                if not isinstance(layer.self_attn.q_proj, HybridQueryLinear):
                    raise TypeError("Experiment 22 requires native HQ8 Q projections")
                original = layer.attn_res_proj
                local = nn.Linear(
                    original.in_features, original.out_features, bias=False,
                    device=original.weight.device, dtype=original.weight.dtype,
                )
                with torch.no_grad():
                    local.weight.copy_(original.weight)
                layer.local_attn_res_proj = local
                layer._experiment22_contrast = True
                layer._experiment22_contrast_off = False
                layer.self_attn.__class__ = Experiment19Attention
        finally:
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)


def set_contrast_off(model: nn.Module, enabled: bool) -> None:
    """Frozen-evaluation-only switch: feed ``cL`` to local Q instead of ``dL``."""
    if enabled and model.training:
        raise ValueError("Experiment 22 contrast-off is frozen-evaluation only")
    layers = model.model.layers
    if not layers or not all(
        getattr(layer, "_experiment22_contrast", False) for layer in layers
    ):
        raise TypeError("contrast-off requires an Experiment 22 model")
    for layer in layers:
        layer._experiment22_contrast_off = bool(enabled)


def experiment22_parameter_report(model: nn.Module) -> dict:
    layers = model.model.layers
    if not layers or not all(
        getattr(layer, "_experiment22_contrast", False) for layer in layers
    ):
        raise TypeError("parameter report requires an Experiment 22 model")
    width = int(model.config.hidden_size)
    return {
        "variant": model.config.experiment22_variant,
        "added_score_parameters": 0,
        "extra_local_router_parameters": sum(
            layer.local_attn_res_proj.weight.numel() for layer in layers),
        "added_scalar_operations_per_token": 2 * width * len(layers),
        "attention_sites": len(layers),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad),
    }
