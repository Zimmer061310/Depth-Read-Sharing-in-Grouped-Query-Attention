"""CPU-only preparation guards; no model imports, training or launch entry point.

These functions check supplied records. Synthetic unit fixtures do not prove
actual model initialization, execution equivalence, or real-data disjointness.
"""
from __future__ import annotations

import hashlib
import json
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "configs/paper_sharing_validation_2k/design.json"
BINDINGS = ROOT / "configs/paper_sharing_validation_2k/execution-bindings.pending.json"
MODELS = ("AAAB", "AABC", "ABBC")
SEEDS = (123, 2025, 3407)
UPDATES = 2000
PRIMARY_MARGIN = Decimal("-0.0025")
ALLOCATION_MARGIN = Decimal("-0.0015")
OVERHEAD_PHASES = ("preflight", "checkpoint_io", "evaluation", "transfer_and_idle", "shutdown_grace")
STUDY_ID = "paper-sharing-fresh-2k-v1"
HISTORICAL_SOURCE = "results/experiment24/data/raw/compute/result.json"
HISTORICAL_SHA256 = "4a136ea8556b6e316d645541d07ef45a8b9d6e2ccf0539fbd42265905133d5bb"
Q_HEADS = {"Q1": list(range(1, 16, 2)), "Q2": list(range(0, 16, 2))}
ROUTE_COUNTS = {"AAAB": 2, "AABC": 3, "ABBC": 3}
BINDING_KEYS = (
    "clean_execution_source_commit", "exact_recipe_and_schedule_lock",
    "projection_execution_policy_and_numerical_contract",
    "primary_artifact_bytes_and_tensor_mask_identity",
    "source_document_selection_and_exclusion_manifest",
    "all_nine_filtered_training_stream_identities",
    "all_nine_initialization_pairing_fingerprints",
    "real_model_graph_gradient_optimizer_checkpoint_tests",
    "primary_reservation_and_scoring_release_lock", "canonical_rtx5090_runtime_benchmark",
    "bounded_preflight_handoff_and_authorization",
    "complete_budget_forecast_with_overheads", "provider_offer_and_resource_handoff",
    "separate_paid_execution_authorization",
)


class ProtocolError(ValueError):
    """Invalid or incomplete provenance; never silently drop a record."""


def require(condition, message):
    if not condition:
        raise ProtocolError(message)


def frozen_equal(value, expected):
    """Compare JSON values without accepting True as 1 or 36.0 as 36."""
    return json.dumps(value, sort_keys=True) == json.dumps(expected, sort_keys=True)


def number(value):
    require(not isinstance(value, bool), "boolean is not a numerical measurement")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ProtocolError("invalid numerical measurement") from exc
    require(result.is_finite(), "non-finite numerical measurement")
    return result


def arithmetic_precision(values):
    """Keep finite decimal sums/differences exact before testing a boundary."""
    return max(40, max(value.adjusted() for value in values)
               - min(value.as_tuple().exponent for value in values) + 10)


def sha(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "expected a complete lowercase SHA-256 identity")
    return value


def active_seeds():
    from src.experiments.paper_sharing_profile import seeds
    return seeds(SEEDS)


def design_path():
    from src.experiments.paper_sharing_profile import design_path as selected
    return selected(DESIGN)


def run_id(model, seed):
    require(model in MODELS and type(seed) is int and seed in active_seeds(), "unknown model/seed or rescue seed")
    return f"paper-sharing-{model.lower()}-seed{seed}-step{UPDATES}"


def expected_rows():
    return [{"id": run_id(model, seed), "model": model, "seed": seed, "updates": UPDATES}
            for seed in active_seeds() for model in MODELS]


def expected_preconditions():
    """Author-amended requirements, not assertions that evidence exists."""
    return {
        "query_execution": {
            "projections_per_attention_layer": 2, "input_width": 1280, "output_widths": [640, 640],
            "logical_q1_physical_heads": list(Q_HEADS["Q1"]), "logical_q2_physical_heads": list(Q_HEADS["Q2"]),
            "packed_head_order": "group_major_q2_then_q1", "common_reshape_qnorm_rope": True,
            "corresponding_slot_initial_weights_identical_within_seed": True,
            "slot_weights_tied_to_each_other": False, "disjoint_native_q_proj_weight_rows_allowed": True,
            "attention_route_counts": dict(ROUTE_COUNTS), "dormant_route_padding_allowed": False,
        },
        "real_model_gate": {
            "required_checks": ["fp32_forward_reference", "bf16_forward_reference",
                "projection_and_input_backward_reference", "equivalent_route_backward_reference",
                "equal_route_forward", "interleaved_head_order", "optimizer_and_checkpoint_row_coverage",
                "checkpoint_roundtrip_logits_and_assignments"],
            "numerical_tolerances": "freeze_before_checks_not_yet_bound",
            "ordinary_cross_topology_router_gradient_equality_required": False,
            "synthetic_fixtures_sufficient": False,
        },
        "real_data_gate": {
            "document_identity": "prefer_stable_dataset_id_else_frozen_canonical_text_sha256",
            "manifest_ids": "sha256_of_frozen_document_identity",
            "selection_identity_and_rng_policy": "freeze_before_artifact_construction",
            "training_order": ["exclude_heldout_documents", "shuffle", "tokenize", "pack"],
            "exclude_every_contributing_document": True,
            "required_artifact_fields": ["input_ids", "targets", "valid_target_mask", "sequence_order",
                "document_to_sequence_provenance"],
            "within_seed_stream_equality": "token_for_token_and_order_sha256",
            "near_duplicate_separation_claimed": False, "synthetic_fixtures_sufficient": False,
        },
        "primary_reservation": {
            "release_after": "all_nine_step2000_checkpoints_frozen",
            "scoring_phase": "one_predetermined_final_phase",
            "historical_intermediate_or_debug_model_scoring_allowed": False,
            "pre_release_structure_hash_and_provenance_checks_allowed": True,
            "development_and_preflight_artifact": "existing_development_only",
        },
        "runtime_rebenchmark": {
            "required_models": list(MODELS), "gpu": "RTX 5090",
            "historical_packed_rates_authoritative": False,
            "bounded_preflight_handoff_and_separate_authorization_required": True,
            "preflight_billed_time_counts_toward_cap": True, "scientific_horizon_changes_allowed": False,
        },
        "final_training_handoff_requires": ["real_model_gate", "real_data_gate", "canonical_runtime_rebenchmark",
            "source_protocol_data_stream_and_checkpoint_policy_lock", "within_cap_complete_forecast"],
        "paid_preflight_authorized": False,
    }


def validate_design(design):
    require(type(design["format_version"]) is int and design["format_version"] == 2
            and design["study_id"] == STUDY_ID
            and design["status"] == "author_frozen_design_local_preparation_only",
            "unknown protocol identity/version/status")
    require(frozen_equal(design["mandatory_preconditions"], expected_preconditions()),
            "changed mandatory safeguards or preflight authority")
    require(frozen_equal(design["author_safeguard_amendment"], {
        "record": "docs/plans/mhar-fresh-validation-safeguards-20261014.md",
        "request_sha256": "bdcacd17b41df9f875ce496bf82ee4ae539147af2289b09313ce4c1a0dc06f48",
        "previous_design_commit": "f66f1ead5d358845ac4b564dbc4a7190314f4fa4"}), "changed amendment provenance")
    s = design["scientific_design"]
    require(s["models"] == list(MODELS) and s["fresh_seeds"] == list(active_seeds()), "changed frozen matrix/seeds")
    require(all(type(seed) is int for seed in s["fresh_seeds"]), "non-integer seed")
    require(type(s["updates_per_run"]) is int and s["updates_per_run"] == UPDATES, "changed endpoint")
    require(frozen_equal(s["backbone"], {"num_layers": 36, "hidden_size": 1280, "query_heads": 16, "kv_heads": 8}),
            "changed backbone or fixed GQA")
    require(s["consumers"] == ["Q1", "Q2", "K", "V"], "changed consumer ordering")
    require(s["route_assignments"] == {model: list(model) for model in MODELS}, "changed sharing graph")
    require(type(s["development_seed"]) is int and s["development_seed"] == 42
            and s["development_seed_is_fresh_confirmation"] is False,
            "development seed is not fresh confirmation")
    for key in ("from_scratch", "common_initialization_matched", "three_route_initial_tensors_identical",
                "within_seed_training_order_identical", "optimizer_schedule_backend_identical"):
        require(s[key] is True, f"changed required pairing invariant: {key}")
    for key in ("silently_rescale_schedule", "rescue_seeds_allowed"):
        require(s[key] is False, f"forbidden protocol change: {key}")
    require(frozen_equal(s["primary_evaluation"], {"sequences": 1024, "tokens_per_sequence": 1024,
            "exclude_source_documents_from_all_fresh_streams": True, "archival_confirmation": "secondary_only"}),
            "changed primary artifact scope or archival role")
    require(frozen_equal(s["primary_gate"], {"contrast": "ABBC-AAAB", "all_seed_deltas_strictly_negative": True,
            "mean_delta_loss_max": str(PRIMARY_MARGIN)}), "changed primary promotion rule")
    require(frozen_equal(s["allocation_gate"], {"contrast": "ABBC-AABC", "all_seed_deltas_strictly_negative": True,
            "mean_delta_loss_max": str(ALLOCATION_MARGIN), "requires_primary_pass": True}),
            "changed allocation promotion rule")
    require(frozen_equal(s["reporting"], {"all_seeds": True, "paired_mean": True, "paired_sample_sd_ddof": 1,
            "population_significance_claim": False, "sequence_bootstrap_is_seed_uncertainty": False}),
            "changed reporting/uncertainty boundary")
    require(s["primary_failure"] == "topology_mechanistic_study"
            and s["primary_only_success"] == "drop_better_allocation_claim", "changed fallback interpretation")
    rows = design["run_matrix"]
    require(frozen_equal(rows, expected_rows()), "missing, duplicate, altered or reordered run matrix")
    b = design["budget"]
    require(number(b["max_allocated_gpu_hours"]) == 150 and number(b["assumed_cny_per_gpu_hour"]) == Decimal("2.88"),
            "changed author planning cap/price assumption")
    require(number(b["planning_cost_cny"]) == 432 and b["persistent_disk_fees_included"] is False
            and b["all_billed_gpu_time_counts"] is True, "changed cost scope")
    require(b["historical_seconds_per_update"] == {"AAAB": "24.18", "AABC": "28.25", "ABBC": "28.30"},
            "changed historical rate transcription")
    require(b["historical_source"] == HISTORICAL_SOURCE
            and b["historical_source_sha256"] == HISTORICAL_SHA256, "changed historical timing identity")
    raw = sum(number(v) for v in b["historical_seconds_per_update"].values()) * UPDATES * 3 / 3600
    require(raw == number(b["training_only_gpu_hours"]) == Decimal("134.55")
            and number(b["overhead_and_reserve_gpu_hours"]) == 150 - raw, "incorrect budget arithmetic")
    require(design["execution_authorized"] is False and design["runtime_launcher_implemented"] is False,
            "preparation design cannot grant launch authority")
    return design


def indexed(rows, *, allow_missing=False):
    expected = {row["id"]: row for row in expected_rows()}
    result = {}
    for row in rows:
        identifier = row["id"]
        require(identifier in expected and identifier not in result, "unknown/duplicate run")
        require(row["model"] == expected[identifier]["model"] and type(row["seed"]) is int
                and row["seed"] == expected[identifier]["seed"], "run identity mismatch")
        result[identifier] = row
    if not allow_missing:
        require(set(result) == set(expected), "all nine run identities required")
    return result


def verify_pairing(rows):
    records = indexed(rows)
    for seed in active_seeds():
        group = [records[run_id(model, seed)] for model in MODELS]
        for field in ("common_parameters_sha256", "training_stream_sha256", "shared_recipe_sha256"):
            values = [sha(row[field]) for row in group]
            require(len(set(values)) == 1, f"within-seed pairing mismatch: {field}")
        require(sha(group[1]["three_route_parameters_sha256"]) == sha(group[2]["three_route_parameters_sha256"]),
                "AABC/ABBC initial three-route tensors differ")
        signatures = [row["q_projection_output_widths"] for row in group]
        require(all(frozen_equal(sig, [640, 640]) for sig in signatures), "exactly two 640-output projections required")
        for row in group:
            require(type(row["q_projection_input_width"]) is int and row["q_projection_input_width"] == 1280,
                    "changed projection input width")
            require(frozen_equal(row["q_slot_physical_heads"], Q_HEADS), "changed interleaved query head ordering")
            require(row["q_slot_rows_disjoint"] is True, "query slot parameters must not be tied")
            require(type(row["attention_route_count"]) is int
                    and row["attention_route_count"] == ROUTE_COUNTS[row["model"]], "dormant or missing attention route")
            require(set(row["q_slot_initial_parameters_sha256"]) == {"Q1", "Q2"}, "query slot hash coverage missing")
        for slot in ("Q1", "Q2"):
            require(len({sha(row["q_slot_initial_parameters_sha256"][slot]) for row in group}) == 1,
                    "corresponding query slot initial weights differ")
    shared_recipes = {row["shared_recipe_sha256"] for row in records.values()}
    require(len(shared_recipes) == 1, "optimizer/schedule/backend differs across seeds")


def verify_exclusions(primary_documents, training_documents):
    primary = {sha(value) for value in primary_documents}
    require(primary, "empty primary document set")
    require(set(training_documents) == {row["id"] for row in expected_rows()}, "must audit all nine streams")
    for identifier, values in training_documents.items():
        train = {sha(value) for value in values}
        require(train, f"empty/unaudited training document set: {identifier}")
        require(not primary.intersection(train), f"held-out document leakage: {identifier}")


def budget_forecast(design, overheads, actual_cny_per_gpu_hour, *, canonical_seconds_per_update=None):
    """Arithmetic on supplied benchmark rates; does not verify a GPU benchmark."""
    validate_design(design)
    require(isinstance(canonical_seconds_per_update, dict)
            and set(canonical_seconds_per_update) == set(MODELS), "canonical benchmark rates required; historical rates insufficient")
    rates = [number(canonical_seconds_per_update[model]) for model in MODELS]
    require(all(value > 0 for value in rates), "invalid canonical benchmark rate")
    require(set(overheads) == set(OVERHEAD_PHASES), "all billed overhead categories must be estimated")
    allowances = [number(overheads[key]) for key in OVERHEAD_PHASES]
    require(all(v >= 0 for v in allowances), "negative overhead allowance")
    price = number(actual_cny_per_gpu_hour)
    require(price > 0, "invalid provider price")
    with localcontext() as context:
        context.prec = arithmetic_precision([price, *rates, *allowances]) * 2
        raw = sum(rates) * UPDATES * len(active_seeds()) / 3600
        hours = raw + sum(allowances)
        require(hours <= number(design["budget"]["max_allocated_gpu_hours"]), "forecast exceeds GPU-hour cap")
        cost = hours * price
        require(cost <= number(design["budget"]["planning_cost_cny"]), "offer/forecast exceeds planning cost; refresh approval")
    return {"allocated_gpu_hours": str(hours), "compute_cost_cny": str(cost), "disk_fees_included": False,
            "benchmark_provenance_validated": False}


def verify_primary_scoring_release(checkpoints, *, subject_run_id, purpose):
    """Check supplied release records only; cannot unlock/read/score an artifact."""
    require(purpose == "final_primary_scoring", "primary artifact cannot be used for debugging or preflight")
    records = indexed(checkpoints)
    require(subject_run_id in records, "only the nine fresh final checkpoints may be scored")
    for row in records.values():
        require(row["status"] == "complete" and type(row["optimizer_step"]) is int
                and row["optimizer_step"] == UPDATES and row["checkpoint_frozen"] is True,
                "all nine exact step-2000 checkpoints must be complete and frozen first")
        sha(row["checkpoint_sha256"])
    return {"record_preconditions_satisfied": True, "scientific_provenance_validated": False}


def contrast_summary(values, margin):
    require(len(values) == len(active_seeds()), "all three paired seed differences required")
    values = [number(value) for value in values]
    margin = number(margin)
    with localcontext() as context:
        context.prec = arithmetic_precision([margin, *values])
        total = sum(values)
        mean = total / len(values)
        sd = (sum((value - mean) ** 2 for value in values) / (len(values) - 1)).sqrt()
        passes = all(value < 0 for value in values) and total <= margin * len(values)
    return {"per_seed_delta_loss": {str(seed): str(value) for seed, value in zip(active_seeds(), values)},
            "mean_delta_loss": str(mean), "sample_sd": str(sd), "sample_sd_ddof": 1,
            "all_negative": all(value < 0 for value in values),
            "passes": passes}


def evaluate_records(rows, *, primary_artifact_sha256):
    """Apply gates to supplied summaries, not validate scientific provenance.

    Artifact/checkpoint fields here are claimed identities, not byte rehashes.
    No tensor masks, token-weighted reduction, model execution, pairing or data
    exclusions are verified by this arithmetic helper.
    """
    artifact = sha(primary_artifact_sha256)
    records = indexed(rows, allow_missing=True)
    missing = [row["id"] for row in expected_rows() if row["id"] not in records]
    incomplete = []
    losses = {}
    for identifier, row in records.items():
        require(row["status"] in {"pending", "running", "incomplete", "failed", "complete"}, "unknown run status")
        require(type(row["optimizer_step"]) is int and 0 <= row["optimizer_step"] <= UPDATES,
                "invalid/extra optimizer steps")
        if row["status"] != "complete" or row["optimizer_step"] != UPDATES:
            incomplete.append(identifier)
            continue
        require(sha(row["primary_artifact_sha256"]) == artifact, "primary artifact mismatch")
        sha(row["checkpoint_sha256"])
        value = number(row["loss"])
        require(value >= 0, "negative token NLL")
        losses[identifier] = value
    if missing or incomplete:
        return {"classification": "incomplete", "missing_runs": missing, "incomplete_runs": incomplete,
                "primary_gate_passes": False, "allocation_gate_passes": False,
                "scientific_provenance_validated": False}
    with localcontext() as context:
        context.prec = arithmetic_precision([PRIMARY_MARGIN, ALLOCATION_MARGIN, *losses.values()])
        primary = contrast_summary([losses[run_id("ABBC", s)] - losses[run_id("AAAB", s)] for s in active_seeds()], PRIMARY_MARGIN)
        allocation = contrast_summary([losses[run_id("ABBC", s)] - losses[run_id("AABC", s)] for s in active_seeds()], ALLOCATION_MARGIN)
    promoted = primary["passes"] and allocation["passes"]
    classification = ("bounded_allocation_claim" if promoted else
                      "primary_only_drop_allocation_claim" if primary["passes"] else "topology_mechanistic_study")
    return {"classification": classification, "primary": primary, "allocation": allocation,
            "primary_gate_passes": primary["passes"], "allocation_gate_passes": promoted,
            "scientific_provenance_validated": False,
            "scope": "gate arithmetic on supplied records only; scientific provenance remains unverified; "
                     "2K setting only, not population significance or external-method superiority"}


def inspect_preparation():
    design = validate_design(json.loads(design_path().read_text()))
    historical = ROOT / design["budget"]["historical_source"]
    require(hashlib.sha256(historical.read_bytes()).hexdigest() == design["budget"]["historical_source_sha256"],
            "historical timing source hash mismatch")
    pending = json.loads(BINDINGS.read_text())
    require(type(pending["format_version"]) is int and pending["format_version"] == 2
            and pending["status"] == "pending_real_evidence_not_a_launch_lock"
            and set(pending["bindings"]) == set(BINDING_KEYS), "missing/altered required binding inventory")
    require(pending["study_id"] == design["study_id"] and pending["ready_for_execution"] is False
            and pending["synthetic_fixtures_are_binding_evidence"] is False, "invalid preparation boundary")
    return {"study_id": design["study_id"], "scientific_design_valid": True, "run_count": 9,
            "historical_training_only_gpu_hours": design["budget"]["training_only_gpu_hours"],
            "canonical_benchmark_verified": False, "real_model_gate_passed": False, "real_data_gate_passed": False,
            "paid_preflight_authorized": False,
            "missing_bindings": [key for key in BINDING_KEYS if pending["bindings"][key] is None],
            "unverified_populated_bindings": [key for key in BINDING_KEYS if pending["bindings"][key] is not None],
            "ready_for_execution": False, "runtime_launcher_implemented": False, "execution_authorized": False,
            "boundary": "CPU preparation only; non-null fields alone cannot establish readiness or authority"}
