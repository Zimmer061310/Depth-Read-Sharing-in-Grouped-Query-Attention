"""Exact artifact, complete checkpoint, discovery freeze, and paired analysis."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.experiments.experiment1_partition_compatibility import (
    sha256_file, sha256_path, load_fixed_eval_artifact,
)
from src.experiments.experiment18_compute_audit import tensor_sha256
from src.experiments.experiment21_evaluation import read_json, require_clean_commit, verify_data_lock
from src.experiments.experiment23_route_topology import Experiment23MHARForCausalLM, set_collapse_b
from src.experiments.experiment23_workflow import (
    MATRIX, PROTOCOL, LOCK, spec, variant_for_id, train_arguments, identity_for_arguments, write_new,
    validate_training_data_identity,
)


def artifact_identity(artifact):
    matrix = spec()
    if sha256_file(artifact) != matrix["evaluation"]["artifact_sha256"]:
        raise ValueError("Experiment 23 requires the exact accepted evaluation artifact")
    lock = verify_data_lock(artifact=Path(artifact), manifest=LOCK)
    for split in ("discovery", "confirmation"):
        row = lock["splits"][split]
        if (row["tensor_sha256"] != matrix["evaluation"][f"{split}_tensor_sha256"]
            or row["valid_token_mask_sha256"] != matrix["evaluation"]["valid_token_mask_sha256"]):
            raise ValueError("frozen tensor/mask identity mismatch")
    payload, observed = load_fixed_eval_artifact(Path(artifact))
    return payload, {"artifact_sha256": observed, "lock_sha256": sha256_file(LOCK),
        "tokenizer": lock["tokenizer"], "tokenizer_revision": lock["tokenizer_revision"],
        "dataset_revision": lock["dataset_revision"], "splits": lock["splits"]}


def validate_checkpoint(checkpoint, parent):
    checkpoint = Path(checkpoint)
    variant = variant_for_id(parent)
    from src.experiments.experiment23_reuse import CONTROLS, validated_control
    if variant in CONTROLS:
        return validated_control(checkpoint, variant)
    manifest = read_json(checkpoint / "training_manifest.json")
    identity = manifest["run_identity"]
    args = train_arguments(variant, output=checkpoint.parent,
                           data_files=identity["data_files"].get("pattern", ""))
    # Build expected non-file identity without re-reading unavailable server shards.
    expected = identity_for_arguments(args, data_identity=identity["data_files"])
    from src.experiments.experiment23_source_exception import live_exception
    exception = live_exception(identity.get("source_commit", ""))
    expected["source_commit"] = identity.get("source_commit", "")
    if identity != expected:
        mismatch = {k: [expected.get(k), identity.get(k)] for k in expected.keys() | identity.keys()
                    if expected.get(k) != identity.get(k)}
        raise ValueError(f"checkpoint recipe/source/condition mismatch: {mismatch}")
    if (checkpoint.name != "step-2000" or manifest.get("global_step") != 2000
        or manifest.get("chunks_consumed") != 64000):
        raise ValueError("incomplete or wrong milestone checkpoint")
    files = identity["data_files"].get("files", [])
    validate_training_data_identity(identity["data_files"])
    if not files or any(r.get("bytes", 0) <= 0 or len(r.get("sha256", "")) != 64 for r in files):
        raise ValueError("missing ordered training shard identity")
    state = torch.load(checkpoint / "training_state.pt", weights_only=False, map_location="cpu", mmap=True)
    if (state.get("run_identity") != identity or state.get("global_step") != 2000
        or state.get("chunks_consumed") != 64000 or not state.get("optimizer", {}).get("state")
        or state.get("scheduler", {}).get("last_epoch") != 2000
        or not isinstance(state.get("cpu_rng_state"), torch.Tensor)
        or len(state.get("cuda_rng_state_all", [])) != 1 or not state.get("wandb_run_id")):
        raise ValueError("incomplete optimizer/scheduler/RNG/stream/W&B state")
    c = read_json(checkpoint / "config.json")
    for key, value in {"experiment23_variant": variant, "experiment23_route_topology": True,
        "experiment23_projection_backend": "packed",
        "experiment23_router_backend": "r2-triton", "hidden_size": 1280,
        "num_hidden_layers": 36, "num_attention_heads": 16, "num_key_value_heads": 8,
        "intermediate_size": 5120, "head_dim": 80, "vocab_size": 151936,
        "rms_norm_eps": 1e-6, "attention_dropout": 0.0, "tie_word_embeddings": True}.items():
        if c.get(key) != value:
            raise ValueError(f"checkpoint config differs: {key}")
    from safetensors import safe_open
    from src.attention_residuals.modeling_qwen3_attnres import Qwen3AttnResConfig
    observed_shapes = {}
    for shard in checkpoint.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in observed_shapes:
                    raise ValueError("duplicate tensor in model shards")
                observed_shapes[name] = tuple(handle.get_slice(name).get_shape())
    with torch.device("meta"):
        skeleton = Experiment23MHARForCausalLM(Qwen3AttnResConfig(**c))
    expected_shapes = {name:tuple(t.shape) for name,t in skeleton.state_dict().items()
                       if name != "lm_head.weight"}
    observed_shapes.pop("lm_head.weight",None)
    if observed_shapes != expected_shapes:
        raise ValueError("incomplete or incompatible model shard tensor inventory")
    complete = read_json(checkpoint.parent / "run_complete.json")
    checkpoint_hash = sha256_path(checkpoint)
    if (complete.get("experiment23_variant") != variant or complete.get("global_step") != 2000
        or complete.get("chunks_consumed") != 64000 or complete.get("run_identity") != identity
        or complete.get("checkpoint_sha256") != checkpoint_hash
        or complete.get("scheduled_and_final_validation_completed") is not True
        or complete.get("wandb_run_id") != state["wandb_run_id"]):
        raise ValueError("training did not finish validation/upload with this checkpoint")
    del state
    return {"checkpoint_sha256": checkpoint_hash, "training_identity": identity,
            **({"postprocess_source_exception":exception} if exception is not None else {})}


def condition_row(condition):
    rows = [r for r in spec()["evaluation"]["conditions"] if r["id"] == condition]
    if len(rows) != 1:
        raise ValueError("unknown, duplicate, or alias-only evaluation condition")
    return rows[0]


def result_path(root, condition, split):
    if split not in ("discovery", "confirmation"):
        raise ValueError("unknown split")
    condition_row(condition)
    return Path(root) / "data/raw/evaluation" / condition / split / "result.json"


@torch.no_grad()
def sequence_metrics(model, tokens, *, batch_size, device):
    if batch_size <= 0:
        raise ValueError("positive batch required")
    totals, counts = [], []
    for start in range(0, len(tokens), batch_size):
        batch = tokens[start:start + batch_size].to(device=device, dtype=torch.long)
        logits = model(batch, use_cache=False).logits
        losses = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                 batch[:, 1:].reshape(-1), reduction="none").reshape(len(batch), -1)
        totals.extend(losses.double().sum(-1).cpu().tolist())
        counts.extend([losses.shape[-1]] * len(batch))
    if not np.isfinite(totals).all():
        raise ValueError("non-finite evaluation Loss")
    return {"loss": float(sum(totals) / sum(counts)), "sequence_total_losses": totals,
        "sequence_valid_tokens": counts, "sequence_losses": [t/n for t,n in zip(totals, counts)],
        "sequence_ids": [tensor_sha256(row.to(torch.int64)) for row in tokens]}


def wandb_job(name, kind, config):
    import wandb
    matrix = spec()
    run = wandb.init(entity=matrix["wandb"]["entity"], project=matrix["wandb"]["project"],
        group=matrix["wandb"]["group"], name=name, job_type=kind, mode="online",
        tags=["experiment-23", "seed-42"], config=config)
    if run is None or run.disabled:
        raise RuntimeError("online W&B required")
    return run


def evaluate_condition(*, condition, checkpoint, artifact, root, split,
                       device, batch_size=1, dtype=torch.bfloat16):
    row = condition_row(condition)
    output = result_path(root, condition, split)
    if output.exists():
        raise FileExistsError(output)
    source = require_clean_commit()
    info = validate_checkpoint(checkpoint, row["parent"])
    payload, data = artifact_identity(artifact)
    if split == "confirmation":
        verify_discovery(root)
        old = load_result(root, condition, "discovery")
        if old["checkpoint_sha256"] != info["checkpoint_sha256"] or old["artifact_identity"] != data:
            raise ValueError("confirmation changed frozen checkpoint/data")
    from src.experiments.experiment23_reuse import load_model
    net = load_model(checkpoint, variant_for_id(row["parent"]), dtype=dtype, device=device).eval()
    set_collapse_b(net, row["kind"] == "frozen-route-collapse")
    run = wandb_job(condition + "-" + split, "fixed-evaluation", {**info,
        "artifact_identity": data, "protocol_sha256": sha256_file(PROTOCOL), "source": source})
    try:
        metrics = sequence_metrics(net, payload[f"{split}_input_ids"], batch_size=batch_size, device=device)
        record = {"condition": condition, "parent": row["parent"], "kind": row["kind"],
            "split": split, "seed": 42, "milestone": 2000, "metrics": metrics,
            "protocol_sha256": sha256_file(PROTOCOL), "matrix_sha256": sha256_file(MATRIX),
            "evaluation_source_commit": source, "artifact_identity": data, **info,
            "wandb": {"run_id": run.id, "run_url": run.url}}
        validate_record(record, condition, split)
        write_new(output, record)
        run.log({f"{split}/loss": metrics["loss"]})
        return record
    finally:
        run.finish()
        del net


def validate_record(record, condition, split):
    row = condition_row(condition)
    if (record.get("condition") != condition or record.get("parent") != row["parent"]
        or record.get("kind") != row["kind"] or record.get("split") != split
        or record.get("seed") != 42 or record.get("milestone") != 2000
        or record.get("protocol_sha256") != sha256_file(PROTOCOL)
        or record.get("matrix_sha256") != sha256_file(MATRIX)
        or len(record.get("checkpoint_sha256", "")) != 64
        or len(record.get("evaluation_source_commit", "")) != 40):
        raise ValueError("invalid condition/protocol/checkpoint/result identity")
    data = record["artifact_identity"]
    identity = record.get("training_identity", {})
    variant = variant_for_id(row["parent"])
    expected = identity_for_arguments(train_arguments(variant, output="/unused",
        data_files=identity.get("data_files", {}).get("pattern", "")),
        data_identity=identity.get("data_files", {}))
    # Offline analysis must survive a later result/publication commit. Preserve
    # the original pinned executable identity rather than relabel it as HEAD.
    source = identity.get("source_commit", "")
    if len(source) != 40 or any(c not in "0123456789abcdef" for c in source):
        raise ValueError("result missing original pinned training source")
    from src.experiments.experiment23_reuse import CONTROLS, validate_gate
    from src.experiments.experiment23_source_exception import verify_exception
    if variant in CONTROLS:
        gate = record.get("control_reuse", {})
        gate_source = gate.get("source_commit", "")
        if gate_source != record["evaluation_source_commit"]:
            verify_exception(record.get("postprocess_source_exception"), gate_source,
                             record["evaluation_source_commit"])
        validate_gate(gate, variant, record["checkpoint_sha256"], source=gate_source)
        if (gate.get("training_identity") != identity
            or gate.get("original_wandb_run_id") != CONTROLS[variant]["original_run_id"]):
            raise ValueError("reused result changed its original training identity")
    else:
        if record["evaluation_source_commit"] != source:
            verify_exception(record.get("postprocess_source_exception"), source,
                             record["evaluation_source_commit"])
        expected["source_commit"] = source
        if identity != expected:
            raise ValueError("result training identity differs from the frozen recipe")
    validate_training_data_identity(identity["data_files"])
    if not record.get("wandb", {}).get("run_id") or not record.get("wandb", {}).get("run_url"):
        raise ValueError("result missing online W&B identity")
    ev = spec()["evaluation"]
    if (data.get("artifact_sha256") != ev["artifact_sha256"]
        or data.get("lock_sha256") != sha256_file(LOCK)
        or data.get("splits") != read_json(LOCK)["splits"]
        or data.get("tokenizer_revision") != spec()["training"]["tokenizer_revision"]
        or data.get("tokenizer") != spec()["training"]["tokenizer"]
        or data.get("dataset_revision") != spec()["training"]["dataset_revision"]):
        raise ValueError("result artifact changed")
    m = record["metrics"]
    t, n = np.asarray(m["sequence_total_losses"]), np.asarray(m["sequence_valid_tokens"])
    ids = m.get("sequence_ids", [])
    if (t.shape != (512,) or n.shape != (512,) or not np.isfinite(t).all() or (t < 0).any()
        or not np.array_equal(n, np.full(512, 1023)) or len(ids) != 512
        or any(len(i) != 64 for i in ids) or not np.isfinite(m["loss"])
        or abs(m["loss"] - t.sum() / n.sum()) > 1e-10):
        raise ValueError("invalid per-sequence sums/counts/identities/aggregate")
    return record


def load_result(root, condition, split):
    return validate_record(read_json(result_path(root, condition, split)), condition, split)


def paired_identity(records):
    first = records[0]
    def recipe(r):
        # Topology differs, recipe and ordered stream must not. Remote paths
        # are not sequence identities, but shard order/bytes/hashes are.
        identity = dict(r["training_identity"])
        for key in list(identity):
            if key == "source_commit" or re.match(r"^experiment\d+_",key):
                identity.pop(key)
        data = identity.pop("data_files")
        identity["ordered_data_files"] = [(x["bytes"], x["sha256"]) for x in data["files"]]
        return identity
    for r in records[1:]:
        if (r["artifact_identity"] != first["artifact_identity"] or r["split"] != first["split"]
            or r["metrics"]["sequence_ids"] != first["metrics"]["sequence_ids"]
            or recipe(r) != recipe(first)):
            raise ValueError("paired artifact/order/training recipe differs")
    by_parent = {}
    for r in records:
        parent = r["parent"]
        if parent in by_parent and by_parent[parent] != r["checkpoint_sha256"]:
            raise ValueError("intervention and parent checkpoints differ")
        by_parent[parent] = r["checkpoint_sha256"]


def discovery_body(root):
    matrix = spec()
    records = [load_result(root, r["id"], "discovery") for r in matrix["evaluation"]["conditions"]]
    paired_identity(records)
    globals_ = [r for r in matrix["training_runs"] if r["features"] == "GGGG"]
    losses = {r["condition"]: r["metrics"]["loss"] for r in records}
    selected = min(globals_, key=lambda r: (losses[r["id"]], r["active_routes"], r["id"]))["id"]
    return {"protocol_sha256": sha256_file(PROTOCOL), "matrix_sha256": sha256_file(MATRIX),
        "primary_id": matrix["analysis"]["primary_candidate"], "selected_global_id": selected,
        "discovery_files": {r["condition"]: sha256_file(result_path(root, r["condition"], "discovery"))
                            for r in records},
        "checkpoints": {r["condition"]: r["checkpoint_sha256"] for r in records}}


def freeze_discovery(root):
    body = discovery_body(root)
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    write_new(Path(root) / "data/selections/discovery_selection.json", {**body, "content_sha256": digest})
    return body


def verify_discovery(root):
    record = read_json(Path(root) / "data/selections/discovery_selection.json")
    digest = record.pop("content_sha256", None)
    if digest != hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest() or record != discovery_body(root):
        raise ValueError("discovery selection/files changed")
    return record


def bootstrap_contrast(metrics, coefficients, *, confidence=.95, samples=10000, seed=23042):
    totals = [np.asarray(m["sequence_total_losses"], dtype=np.float64) for m in metrics]
    counts = [np.asarray(m["sequence_valid_tokens"], dtype=np.float64) for m in metrics]
    if (len(coefficients) != len(metrics) or not totals or not 0 < confidence < 1 or samples < 1
        or any(t.shape != totals[0].shape or n.shape != t.shape or (n <= 0).any()
               or not np.isfinite(t).all() for t, n in zip(totals, counts))
        or any(not np.array_equal(n, counts[0]) for n in counts)):
        raise ValueError("invalid paired bootstrap inputs")
    point = sum(c * t.sum()/n.sum() for c,t,n in zip(coefficients,totals,counts))
    rng = np.random.default_rng(seed)
    draws = np.empty(samples)
    for start in range(0, samples, 128):
        indices = rng.integers(0, len(totals[0]), size=(min(128,samples-start), len(totals[0])))
        draws[start:start+len(indices)] = sum(c * t[indices].sum(1)/n[indices].sum(1)
                                           for c,t,n in zip(coefficients,totals,counts))
    tail = (1-confidence)/2
    return {"delta_loss": float(point), "ci_low": float(np.quantile(draws, tail)),
        "ci_high": float(np.quantile(draws, 1-tail)), "confidence": confidence,
        "bootstrap_samples": samples, "bootstrap_seed": seed}


def analyze(root):
    matrix = spec()
    selection = verify_discovery(root)
    result = {"protocol_sha256": sha256_file(PROTOCOL), "selection": selection,
              "analysis_source_commit": subprocess.check_output(
                  ["git","rev-parse","HEAD"],text=True,cwd=PROTOCOL.parents[2]).strip(),
              "splits": {}, "limitations": ["single seed", "step-2000 screen", "reused validation benchmark",
                  "historical controls: matched recorded recipe, not freshly paired training trajectories",
                  "historical initialization fingerprint unavailable; source/backend differences remain possible confounds"]}
    native = [r["id"] for r in matrix["training_runs"]]
    primary = matrix["analysis"]["primary_candidate"]
    m8, r2 = matrix["analysis"]["primary_references"]
    hq8 = "exp23-hq8-gl-aaaa-seed42-step2000"
    for split in ("discovery", "confirmation"):
        records = {r["id"]: load_result(root, r["id"], split) for r in matrix["evaluation"]["conditions"]}
        paired_identity(list(records.values()))
        for condition, record in records.items():
            if split == "confirmation" and record["checkpoint_sha256"] != selection["checkpoints"][condition]:
                raise ValueError("confirmation checkpoint differs from selection")
        def contrast(ids, coeffs, confidence=.95):
            return bootstrap_contrast([records[i]["metrics"] for i in ids], coeffs, confidence=confidence)
        primaries = {ref: contrast([primary, ref], [1,-1]) for ref in (m8,r2)}
        for value in primaries.values():
            value["practical_gate"] = value["delta_loss"] <= -.002 and value["ci_high"] < 0
        secondary = {i: {"descriptive95": contrast([i,m8],[1,-1]),
                         "bonferroni99": contrast([i,m8],[1,-1],.99)}
                     for i in native if i != m8 and variant_for_id(i).startswith("global")}
        for value in secondary.values():
            ci = value["bonferroni99"]
            value["practical_gate"] = ci["delta_loss"] <= -.002 and ci["ci_high"] < 0
        collapse = {i: contrast([i,r["parent"]],[1,-1]) for i,r in records.items()
                    if r["kind"] == "frozen-route-collapse"}
        result["splits"][split] = {"losses": {i:r["metrics"]["loss"] for i,r in records.items()},
            "primary": primaries, "joint_promising": all(v["practical_gate"] for v in primaries.values()),
            "secondary": secondary, "collapse": collapse,
            "interaction": contrast([primary,m8,r2,hq8],[1,-1,-1,1])}
    write_new(Path(root) / "data/derived/paired_analysis.json", result)
    return result
