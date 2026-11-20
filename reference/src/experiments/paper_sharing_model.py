"""Isolated fresh-study Q execution; no trainer, artifact reader, or GPU launcher.

Reuse archival layer equations without editing them. The native dense Q weight
keeps its disjoint, independently trainable odd/even row blocks; all conditions
execute the existing canonical two-slot projection. CPU fixtures are explicit
and must never be represented as full-size CUDA readiness.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re

import torch
from torch import nn

from src.attention_residuals.modeling_qwen3_attnres import Qwen3AttnResForCausalLM
from src.experiments.experiment23_route_topology import Experiment23Attention
from src.experiments.paper_sharing_protocol import MODELS, Q_HEADS, ROUTE_COUNTS


QUERY_EXECUTION = "two-slot-v1"
ROUTER_BACKENDS = ("canonical", "cuda-exact")


def _legacy_settings(condition, backend):
    if condition == "AAAB":
        return {"experiment23_route_topology": True, "experiment24_three_route": False,
                "experiment23_variant": "global-aaab", "experiment23_projection_backend": "canonical",
                "experiment23_router_backend": "shared-eager" if backend == "canonical" else "r2-triton"}
    return {"experiment23_route_topology": False, "experiment24_three_route": True,
            "experiment24_variant": f"global-{condition.lower()}",
            "experiment24_router_backend": "canonical" if backend == "canonical" else "tiled-r2-exact"}


def configure(config, condition, *, router_backend="canonical", test_fixture=False, execution_policy=None, runtime_policy=None):
    """Make an explicit fresh-study config, without changing the supplied one.

    Seed/RNG policy belongs to the caller's later source-bound execution lock;
    constructing this wrapper consumes exactly the native model's RNG draws.
    """
    if condition not in MODELS or router_backend not in ROUTER_BACKENDS or type(test_fixture) is not bool:
        raise ValueError("unknown condition/backend or invalid fixture flag")
    if any(key.startswith("experiment") and value is not None and value is not False
           for key, value in config.to_dict().items()):
        raise ValueError("start from a clean native config, not another experiment/checkpoint config")
    result = copy.deepcopy(config)
    result.paper_sharing_validation = True
    result.paper_sharing_condition = condition
    result.paper_sharing_query_execution = QUERY_EXECUTION
    result.paper_sharing_router_backend = router_backend
    result.paper_sharing_test_fixture = test_fixture
    if execution_policy is not None:
        from src.experiments.paper_sharing_execution_policy import POLICY_ID, POLICY_SHA256, canonical_policy
        canonical_policy()
        if execution_policy != POLICY_ID or router_backend != "canonical":
            raise ValueError("explicit canonical policy/backend required")
        result.paper_sharing_execution_policy = POLICY_ID
        result.paper_sharing_execution_policy_sha256 = POLICY_SHA256
    if runtime_policy is not None:
        from src.experiments.paper_sharing_determinism import RUNTIME_POLICY_ID, RUNTIME_POLICY_SHA256, runtime_binding
        result.paper_sharing_runtime_policy = runtime_policy
        result.paper_sharing_runtime_policy_sha256 = RUNTIME_POLICY_SHA256
        runtime_binding(result)
    for name, value in _legacy_settings(condition, router_backend).items():
        setattr(result, name, value)
    return result


def validate_config(config):
    condition = getattr(config, "paper_sharing_condition", None)
    backend = getattr(config, "paper_sharing_router_backend", None)
    if (getattr(config, "paper_sharing_validation", False) is not True or condition not in MODELS
            or backend not in ROUTER_BACKENDS
            or getattr(config, "paper_sharing_query_execution", None) != QUERY_EXECUTION
            or type(getattr(config, "paper_sharing_test_fixture", None)) is not bool):
        raise ValueError("explicit paper-sharing config and execution identity required")
    from src.experiments.paper_sharing_execution_policy import policy_binding
    policy_binding(config)
    from src.experiments.paper_sharing_determinism import runtime_binding
    runtime_binding(config)
    expected = {"hidden_size": 1280, "head_dim": 80, "num_attention_heads": 16,
                "num_key_value_heads": 8, "attnres_num_heads": 8}
    if any(type(getattr(config, key, None)) is not int or getattr(config, key) != value
           for key, value in expected.items()):
        raise ValueError("production Q/KV dimensions and fixed GQA are required, even in CPU fixtures")
    if (config.attnres_mode != "full_mh" or config.attnres_gate_type != "bias"
            or config.attention_dropout != 0 or getattr(config, "attention_bias", False)
            or config.attnres_use_null_source):
        raise ValueError("full_mh H8/bias gate with no dropout, QKV bias, or null source required")
    if not config.paper_sharing_test_fixture and (
            config.num_hidden_layers != 36 or config.intermediate_size != 5120 or config.vocab_size != 151936):
        raise ValueError("reduced depth/FFN/vocabulary is only allowed in explicitly labeled local fixtures")
    settings = _legacy_settings(condition, backend)
    if any(getattr(config, key, None) != value for key, value in settings.items()):
        raise ValueError("changed inherited layer dispatch or backend identity")
    for key, value in config.to_dict().items():
        if key.startswith("experiment") and key not in settings and value is not None and value is not False:
            raise ValueError(f"incompatible experiment setting: {key}")
    return condition, backend


class PaperSharingAttention(Experiment23Attention):
    def project_qkv(self, route_inputs):
        condition = self._paper_sharing_condition
        roles = set(condition)
        if (self._experiment23_projection != "canonical" or self._experiment23_routes != condition
                or type(self.q_proj) is not nn.Linear or self.q_proj.in_features != 1280
                or self.q_proj.out_features != 1280 or self.q_proj.bias is not None
                or not self.q_proj.weight.is_contiguous()
                or self.head_dim != 80 or set(route_inputs) != roles):
            raise ValueError("changed two-slot projection, assignment, dimensions or active routes")
        first = route_inputs["A"]
        if first.ndim != 3 or first.shape[-1] != 1280 or any(
                value.shape != first.shape or value.dtype != first.dtype or value.device != first.device
                for value in route_inputs.values()):
            raise ValueError("all route inputs must have matching [B,T,1280] shape/dtype/device")
        # The inherited canonical path performs Q2 then Q1 as two 640-output
        # GEMMs and interleaves group-major. Q norm/RoPE/attention stay inherited.
        return super().project_qkv(route_inputs)


class PaperSharingForCausalLM(Qwen3AttnResForCausalLM):
    def __init__(self, config):
        condition, backend = validate_config(config)
        super().__init__(config)
        for layer in self.model.layers:
            layer.local_attn_res_proj = copy.deepcopy(layer.attn_res_proj)
            if condition == "AAAB":
                layer._experiment23_route_topology = True
                layer._experiment23_active_b = True
                layer._experiment23_collapse = False
                layer._experiment23_router_backend = config.experiment23_router_backend
            else:
                layer.third_attn_res_proj = copy.deepcopy(layer.attn_res_proj)
                layer._experiment24_three_route = True
                layer._experiment24_backend = config.experiment24_router_backend
                layer._experiment24_intervention = None
            layer.self_attn.__class__ = PaperSharingAttention
            layer.self_attn._paper_sharing_condition = condition
            layer.self_attn._experiment23_routes = condition
            layer.self_attn._experiment23_projection = "canonical"
        self._paper_sharing_backend = backend

    def validate_execution_dispatch(self):
        """Canonical amendment cannot silently enter custom/fused router code."""
        from src.experiments.paper_sharing_determinism import runtime_binding
        runtime_binding(self.config, check_runtime=True)
        from src.experiments.paper_sharing_execution_policy import policy_binding
        binding = policy_binding(self.config)
        if binding is None:
            return  # Preserve all legacy/frozen v2 behavior.
        from src.attention_residuals import modeling_qwen3_attnres as native
        fused = native._mhar_fused
        if (self._paper_sharing_backend != "canonical" or native._compiled_mh_block_kernel is not None
                or (fused is not None and (fused._ENABLED or fused.current_context() is not None))):
            raise ValueError("canonical execution refuses compiled/fused routing")
        condition = self.config.paper_sharing_condition
        for layer in self.model.layers:
            if condition == "AAAB":
                valid = (layer._experiment23_router_backend == "shared-eager"
                         and layer._experiment23_active_b and not layer._experiment23_collapse
                         and getattr(layer, "_experiment23_route_topology", False)
                         and not getattr(layer, "_experiment24_three_route", False))
            else:
                valid = (layer._experiment24_backend == "canonical" and layer._experiment24_intervention is None
                         and getattr(layer, "_experiment24_three_route", False)
                         and not getattr(layer, "_experiment23_route_topology", False))
            if (not valid or layer.self_attn._experiment23_projection != "canonical"
                    or layer.self_attn._experiment23_routes != condition
                    or layer.self_attn._paper_sharing_condition != condition
                    or hasattr(layer, "third_attn_res_proj") != (condition != "AAAB")):
                raise ValueError("canonical execution layer dispatch changed")

    def forward(self, *args, **kwargs):
        self.validate_execution_dispatch()
        return super().forward(*args, **kwargs)


def tensor_sha256(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)},
                                      sort_keys=True).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _state_sha256(items):
    digest = hashlib.sha256()
    for name, value in sorted(items):
        digest.update(json.dumps([name, tensor_sha256(value)], separators=(",", ":")).encode())
    return digest.hexdigest()


def model_fingerprints(model):
    """Hash actual current tensors; these are initialization evidence only at step zero."""
    condition, _ = validate_config(model.config)
    slots = {"Q1": [], "Q2": []}
    route_items = []
    q_parameters = set()
    for index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        if (not isinstance(attention, PaperSharingAttention) or attention._experiment23_routes != condition
                or attention._experiment23_projection != "canonical"
                or attention._paper_sharing_condition != condition):
            raise ValueError("layer assignment/execution identity differs from saved config")
        if hasattr(layer, "third_attn_res_proj") != (condition != "AAAB"):
            raise ValueError("dormant or missing C route")
        if not attention.q_proj.weight.is_contiguous() or id(attention.q_proj.weight) in q_parameters:
            raise ValueError("query rows must be dense/disjoint and not tied between layers")
        q_parameters.add(id(attention.q_proj.weight))
        if condition == "AAAB":
            if (getattr(layer, "_experiment24_three_route", False)
                    or not getattr(layer, "_experiment23_route_topology", False)
                    or not layer._experiment23_active_b or layer._experiment23_collapse
                    or layer._experiment23_router_backend != model.config.experiment23_router_backend):
                raise ValueError("changed two-route dispatch/backend")
        elif (getattr(layer, "_experiment23_route_topology", False)
              or not getattr(layer, "_experiment24_three_route", False)
              or layer._experiment24_backend != model.config.experiment24_router_backend
              or layer._experiment24_intervention is not None):
            raise ValueError("changed three-route dispatch/backend or active intervention")
        weight = attention.q_proj.weight.reshape(8, 2, 80, 1280)
        for name, slot in (("Q1", 1), ("Q2", 0)):
            slots[name].append((str(index), weight[:, slot]))
        for name in ("attn_res_proj", "local_attn_res_proj", "third_attn_res_proj"):
            if hasattr(layer, name):
                route_items.append((f"{index}.{name}", getattr(layer, name).weight))
    common = [(name, tensor) for name, tensor in model.state_dict().items() if ".third_attn_res_proj." not in name]
    return {"condition": condition, "common_parameters_sha256": _state_sha256(common),
            "checkpoint_state_sha256": _state_sha256(model.state_dict().items()),
            "attention_route_parameters_sha256": _state_sha256(route_items),
            "q_slot_initial_parameters_sha256": {name: _state_sha256(values) for name, values in slots.items()},
            "q_slot_physical_heads": copy.deepcopy(Q_HEADS), "q_slot_rows_disjoint": True,
            "q_projection_input_width": 1280, "q_projection_output_widths": [640, 640],
            "attention_route_count": ROUTE_COUNTS[condition],
            "test_fixture": model.config.paper_sharing_test_fixture,
            "nominal_parameters": sum(parameter.numel() for parameter in model.parameters())}


def runtime_buffer_fingerprints(model):
    """Bind execution tensors omitted by state_dict, especially nonpersistent RoPE.

    Transformers4.57 keeps original_inv_freq as a plain tensor attribute;
    newer versions register it. Cover either representation without assuming
    that a matching safetensors weight hash implies matching execution state.
    """
    buffers = dict(model.named_buffers())
    for name, module in model.named_modules():
        value = getattr(module, "original_inv_freq", None)
        if isinstance(value, torch.Tensor):
            buffers.setdefault(f"{name}.original_inv_freq" if name else "original_inv_freq", value)
    return {name: {"dtype": str(value.dtype), "shape": list(value.shape),
                   "sha256": tensor_sha256(value)} for name, value in sorted(buffers.items())}


def load_local_model_checkpoint(path, *, expected_condition, expected_state_sha256,
                                attn_implementation="sdpa"):
    """Strict local model reload; no Hub downloads or missing-weight fallback.

    The caller must bind the expected identity independently before loading.
    This is not a scientific optimizer/scheduler/data/RNG resume validator.
    """
    directory = Path(path).resolve(strict=True)
    if not directory.is_dir() or expected_condition not in MODELS:
        raise ValueError("existing local checkpoint directory and known condition required")
    if not isinstance(expected_state_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_state_sha256) is None:
        raise ValueError("complete expected tensor-state identity required")
    config = PaperSharingForCausalLM.config_class.from_pretrained(directory, local_files_only=True)
    if getattr(config, "paper_sharing_runtime_policy", None) is not None:
        # Transformers does not serialize its private attention selector.
        # Bind the explicit loader choice before validating strict metadata;
        # never let from_pretrained silently select another attention interface.
        if attn_implementation != "sdpa":
            raise ValueError("strict checkpoint reload requires explicit sdpa")
        config._attn_implementation = attn_implementation
    condition, _ = validate_config(config)
    if condition != expected_condition:
        raise ValueError("checkpoint condition differs from expected run")
    with torch.device("cpu"):
        model, info = PaperSharingForCausalLM.from_pretrained(directory, config=config, local_files_only=True,
            output_loading_info=True, attn_implementation=attn_implementation, dtype="auto")
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("incomplete or mismatched checkpoint; no silent weight initialization")
    if model_fingerprints(model)["checkpoint_state_sha256"] != expected_state_sha256:
        raise ValueError("checkpoint tensor-state identity differs from expected run")
    # Fresh native construction applies model.to(dtype=BF16) to the entire
    # module, including nonpersistent RoPE buffers. HF dtype="auto" restores
    # BF16 weights but recreates inv_freq in FP32; it is absent from state_dict.
    # Replay the SAME whole-module cast, not a new higher-precision RoPE policy.
    # FP32 fixtures remain FP32; raw checkpoint bytes are never rewritten.
    model.to(dtype=next(model.parameters()).dtype)
    if model_fingerprints(model)["checkpoint_state_sha256"] != expected_state_sha256:
        raise ValueError("native buffer cast changed checkpoint weight identity")
    return model


def validate_optimizer_coverage(model, optimizer):
    """Every trainable native Parameter once; Q slices are not re-registered."""
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    actual = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("optimizer has missing, duplicate or foreign Parameters")
    state = model.state_dict()
    for index, layer in enumerate(model.model.layers):
        parameter = layer.self_attn.q_proj.weight
        if not parameter.requires_grad or actual.count(id(parameter)) != 1:
            raise ValueError("both independent Q row blocks must be optimized exactly once")
        name = f"model.layers.{index}.self_attn.q_proj.weight"
        if name not in state or state[name].shape != (1280, 1280):
            raise ValueError("native checkpoint does not cover both Q row blocks")
    return {"trainable_parameter_objects": len(expected), "q_row_blocks_per_layer": 2,
            "q_parameter_registrations_per_layer": 1, "full_gate_a_passed": False}
