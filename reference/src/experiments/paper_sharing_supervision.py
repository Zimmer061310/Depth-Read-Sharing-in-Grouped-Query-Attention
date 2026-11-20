"""Native T-1 supervision, separate from the immutable T-target data-validity mask.

No artifact loader, model invocation, trainer, scoring release or GPU launcher.
The final lookahead is stored/provenanced but never contributes to this loss.
Actual primary scoring still requires the nine-checkpoint release controller.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn import functional as F

from src.experiments.paper_sharing_protocol import frozen_equal, require


CONTRACT = Path(__file__).resolve().parents[2] / "configs/paper_sharing_validation_2k/loss-supervision.frozen.json"
SEQUENCE_LENGTH = 1024
TOKENIZER_VOCAB = 151669


def validate_contract(contract):
    required = {
        "format_version": 1, "study_id": "paper-sharing-fresh-2k-v1",
        "contract_id": "native-causal-t-minus-one-v1",
        "author_reply": "Preserve historical training loss",
        "data_policy_file_sha256": "9c3c3983c6ee1a628f5d8539fef0c7101013b6e5bf3a06ac244b07702e006ae8",
        "bundle_manifest_sha256": "c69db984bdfcc2958962fd5cddb04065b622e8e71bb2933654f5435c7e36be40",
        "training_recipe_intent_sha256": "0b303ab2077e19ee1ed9ca18eaa6d92200e81c5936c034b0de36ab6dc9b2360a",
        "sequence_length": 1024, "stored_raw_tokens_per_example": 1025,
        "exposed_shifted_targets_per_example": 1024, "supervised_targets_per_example": 1023,
        "data_validity_mask": "unchanged_all_1024_positions_valid_not_a_loss_weight_mask",
        "training": {"input_ids": "raw[:1024]", "model_labels": "input_ids_unshifted", "use_cache": False,
            "internal_label_shift_count": 1, "override_shift_labels": False, "supervised_logit_positions": "0:1023",
            "supervised_raw_target_positions": "1:1024", "ignore_index": -100,
            "ignored_final_logit_position": 1023, "stored_lookahead_raw_position_1024_supervised": False},
        "evaluation": {"logits": "logits[:, :-1].float()", "targets": "input_ids[:, 1:]",
            "per_token_loss": "cross_entropy_reduction_none", "sequence_total_accumulation": "float64",
            "aggregate": "sum_sequence_total_losses/sum_sequence_supervised_token_counts"},
        "effective_supervision_mask": "data_validity_mask.clone(); set final column false",
        "primary_sequences": 1024, "primary_input_positions": 1048576, "primary_supervised_positions": 1047552,
        "training_examples_per_seed": 64000, "training_input_positions_per_run": 65536000,
        "training_supervised_positions_per_run": 65472000,
        "local_fp32_parity": {"loss_atol": 1e-6, "loss_rtol": 1e-6,
            "logit_gradient_atol": 1e-7, "logit_gradient_rtol": 1e-6, "ignored_final_logit_gradient": "exactly_zero"},
        "cuda_bf16_loss_parity_verified": False, "full_trainer_integration_verified": False,
        "stored_artifact_or_stream_bytes_changed": False, "primary_scoring_authorized": False,
        "paid_execution_authorized": False,
    }
    require(all(key in contract and frozen_equal(contract[key], value) for key, value in required.items()),
            "changed native loss contract, identities, counts or authority")
    return contract


def load_contract():
    return validate_contract(json.loads(CONTRACT.read_text()))


def supervision_mask(data_validity_mask):
    require(isinstance(data_validity_mask, torch.Tensor) and data_validity_mask.dtype == torch.bool
            and data_validity_mask.ndim == 2 and data_validity_mask.shape[0] > 0
            and data_validity_mask.shape[1] == SEQUENCE_LENGTH, "expected a nonempty B-by-1024 boolean validity mask")
    require(bool(data_validity_mask.all()), "frozen bundle has no padding or invalid data targets")
    result = data_validity_mask.clone()
    result[:, -1] = False
    return result


def _inputs(input_ids, data_validity_mask):
    mask = supervision_mask(data_validity_mask)
    require(isinstance(input_ids, torch.Tensor) and input_ids.dtype == torch.long
            and input_ids.shape == mask.shape and input_ids.device == mask.device,
            "input IDs must be unshifted B-by-1024 long tensors on the mask device")
    require(bool(((input_ids >= 0) & (input_ids < TOKENIZER_VOCAB)).all()), "input token outside frozen tokenizer range")
    return mask


def native_training_kwargs(input_ids, *, exposed_targets, data_validity_mask):
    _inputs(input_ids, data_validity_mask)
    require(isinstance(exposed_targets, torch.Tensor) and exposed_targets.dtype == torch.long
            and exposed_targets.shape == input_ids.shape and exposed_targets.device == input_ids.device,
            "exposed shifted targets must match input shape/dtype/device")
    require(bool(((exposed_targets >= 0) & (exposed_targets < TOKENIZER_VOCAB)).all()), "exposed target outside frozen tokenizer range")
    require(torch.equal(input_ids[:, 1:], exposed_targets[:, :-1]), "input/target causal alignment mismatch")
    # Deliberately do not pass exposed_targets as labels: the native model will
    # shift labels once internally and ignore its final padded label.
    return {"input_ids": input_ids, "labels": input_ids, "use_cache": False}


def native_loss_statistics(logits, input_ids, data_validity_mask):
    """Historical loss formula for caller-supplied logits; not scoring authority."""
    mask = _inputs(input_ids, data_validity_mask)
    require(isinstance(logits, torch.Tensor) and logits.is_floating_point() and logits.ndim == 3
            and logits.shape[:2] == input_ids.shape and logits.device == input_ids.device
            and logits.shape[-1] > 0, "logit shape/dtype/device mismatch")
    require(bool(torch.isfinite(logits).all()) and bool((input_ids < logits.shape[-1]).all()),
            "nonfinite logits or target outside supplied logit vocabulary")
    per_token = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                input_ids[:, 1:].reshape(-1), reduction="none").reshape(input_ids.shape[0], -1)
    totals = per_token.double().sum(-1)
    counts = mask.sum(-1)
    return {"loss": per_token.mean(), "sequence_total_losses": totals,
            "sequence_valid_tokens": counts, "token_weighted_loss": totals.sum() / counts.sum(),
            "effective_supervision_mask": mask}
