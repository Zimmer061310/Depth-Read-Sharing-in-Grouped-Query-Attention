"""Immutable historical imports, frozen evaluations and preregistered analysis."""
from __future__ import annotations

import hashlib
import json
import subprocess
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from src.experiments.experiment1_partition_compatibility import sha256_file, sha256_path
from src.experiments.experiment21_evaluation import read_json, require_clean_commit
from src.experiments.experiment23_evaluation import (
    artifact_identity, sequence_metrics, bootstrap_contrast, paired_identity,
    validate_record as historical_record,
)
from src.experiments.experiment24_three_route import (
    Experiment24MHARForCausalLM, intervention, aaab_value_interpolation, RUNS,
)
from src.experiments.experiment24_workflow import (
    ROOT, MATRIX, PROTOCOL, LOCK, HISTORICAL, spec, conditions, condition_row,
    variant_for_id, train_arguments, identity_for_arguments, validate_training_data_identity, write_new,
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def result_path(root, condition, split):
    condition_row(condition)
    if split not in ("discovery", "confirmation"):
        raise ValueError("invalid split")
    return Path(root)/"data/raw/evaluation"/condition/split/"result.json"


def import_references(root):
    """Validate actual old records, then copy their original bytes, never relabel."""
    rows = []
    for condition in conditions():
        if not condition["imported"]:
            continue
        for split in ("discovery", "confirmation"):
            source = result_path(HISTORICAL, condition["id"], split)
            record = historical_record(read_json(source), condition["id"], split)
            expected = next(r["checkpoint_sha256"] for r in spec()["reused_native_references"]
                            if r["id"] == condition["parent"])
            if record["checkpoint_sha256"] != expected:
                raise ValueError("historical checkpoint hash differs from frozen matrix")
            target = result_path(root, condition["id"], split)
            target.parent.mkdir(parents=True, exist_ok=True)
            raw = source.read_bytes()
            if target.exists():
                if target.read_bytes() != raw:
                    raise ValueError("existing historical import was modified")
            else:
                # Exclusive creation; no overwrite even under concurrent recovery.
                with target.open("xb") as handle:
                    handle.write(raw)
            rows.append({"condition": condition["id"], "split": split,
                         "original_path": str(source.relative_to(ROOT)),
                         "sha256": sha256_file(source), "original_wandb": record["wandb"]})
    manifest = {"records": rows, "count": len(rows), "original_ids_preserved": True}
    path = Path(root)/"provenance/reused_records.json"
    if path.exists():
        if read_json(path) != manifest:
            raise ValueError("import manifest changed")
    else:
        write_new(path, manifest)
    return manifest


def validate_source(training, evaluation, exception=None):
    if any(len(s) != 40 or set(s)-set("0123456789abcdef") for s in (training, evaluation)):
        raise ValueError("invalid original source identity")
    if training == evaluation:
        return
    from src.experiments.experiment24_source_exception import verify_exception
    verify_exception(exception, training, evaluation)


def validate_checkpoint(checkpoint, parent):
    """Read-only checkpoint validation; old exceptions never authorize new graphs."""
    checkpoint = Path(checkpoint)
    variant = variant_for_id(parent)
    identity = read_json(checkpoint/"training_manifest.json")["run_identity"]
    manifest = read_json(checkpoint/"training_manifest.json")
    tree_hash = sha256_path(checkpoint)
    if parent.startswith("exp23-"):
        accepted = historical_record(read_json(result_path(HISTORICAL, parent, "discovery")), parent, "discovery")
        expected_hash = next(r["checkpoint_sha256"] for r in spec()["reused_native_references"] if r["id"] == parent)
        if tree_hash != expected_hash or identity != accepted["training_identity"]:
            raise ValueError("accepted reference checkpoint/identity changed")
    else:
        expected = identity_for_arguments(train_arguments(variant, output=checkpoint.parent,
            data_files=identity["data_files"].get("pattern", "")), data_identity=identity["data_files"])
        expected["source_commit"] = identity.get("source_commit")
        if expected != identity:
            raise ValueError("new checkpoint recipe/source/variant differs")
    validate_training_data_identity(identity["data_files"])
    if (checkpoint.name != "step-2000" or manifest.get("global_step") != 2000
        or manifest.get("chunks_consumed") != 64000):
        raise ValueError("not a complete step-2000 checkpoint")
    state = torch.load(checkpoint/"training_state.pt", map_location="cpu", weights_only=False, mmap=True)
    if (state.get("run_identity") != identity or state.get("global_step") != 2000
        or state.get("chunks_consumed") != 64000 or not state.get("optimizer", {}).get("state")
        or state.get("scheduler", {}).get("last_epoch") != 2000
        or not isinstance(state.get("cpu_rng_state"), torch.Tensor)
        or len(state.get("cuda_rng_state_all", [])) != 1 or not state.get("wandb_run_id")):
        raise ValueError("incomplete optimizer/scheduler/RNG/data/W&B state")
    complete = read_json(checkpoint.parent/"run_complete.json")
    key = "experiment24_variant" if parent.startswith("exp24-") else "experiment23_variant"
    if (complete.get(key) != variant or complete.get("checkpoint_sha256") != tree_hash
        or complete.get("run_identity") != identity or complete.get("global_step") != 2000
        or complete.get("chunks_consumed") != 64000
        or complete.get("scheduled_and_final_validation_completed") is not True
        or complete.get("wandb_run_id") != state["wandb_run_id"]):
        raise ValueError("missing accepted completion/upload marker")
    config = read_json(checkpoint/"config.json")
    from safetensors import safe_open
    from src.attention_residuals.modeling_qwen3_attnres import Qwen3AttnResConfig
    from src.experiments.experiment23_route_topology import Experiment23MHARForCausalLM
    cls = Experiment24MHARForCausalLM if parent.startswith("exp24-") else Experiment23MHARForCausalLM
    shape_fields = dict(hidden_size=1280,num_hidden_layers=36,num_attention_heads=16,
        num_key_value_heads=8,head_dim=80,intermediate_size=5120,vocab_size=151936,
        rms_norm_eps=1e-6,attention_dropout=0.,tie_word_embeddings=True,attnres_num_heads=8,
        attnres_mode="full_mh")
    shape_fields.update({"experiment24_variant":variant, "experiment24_three_route":True,
                         "experiment24_router_backend":"tiled-r2-exact"} if parent.startswith("exp24-")
                       else {"experiment23_variant":variant, "experiment23_route_topology":True,
                             "experiment23_router_backend":"r2-triton", "experiment23_projection_backend":"packed"})
    if any(config.get(k) != v for k,v in shape_fields.items()):
        raise ValueError("checkpoint architecture/backend changed")
    with torch.device("meta"):
        skeleton = cls(Qwen3AttnResConfig(**config))
    expected_shapes = {k:tuple(v.shape) for k,v in skeleton.state_dict().items() if k != "lm_head.weight"}
    shapes = {}
    for shard in checkpoint.glob("*.safetensors"):
        with safe_open(shard,framework="pt",device="cpu") as f:
            for k in f.keys():
                if k in shapes:
                    raise ValueError("duplicate checkpoint tensor")
                shapes[k] = tuple(f.get_slice(k).get_shape())
    shapes.pop("lm_head.weight", None)
    if shapes != expected_shapes:
        raise ValueError("incomplete tensor inventory")
    return {"checkpoint_sha256":tree_hash,"training_identity":identity,
            "training_wandb_run_id":state["wandb_run_id"]}


def load_model(checkpoint, parent, *, dtype, device):
    if parent.startswith("exp23-"):
        from src.experiments.experiment23_reuse import load_model as old_loader
        return old_loader(checkpoint,variant_for_id(parent),dtype=dtype,device=device)
    from src.attention_residuals.modeling_qwen3_attnres import enable_fused_mhar
    enable_fused_mhar(True)
    return Experiment24MHARForCausalLM.from_pretrained(checkpoint, dtype=dtype,
                    attn_implementation="sdpa").to(device)


def measurement_binding(parent, info, source_exception=None):
    source = require_clean_commit()
    if parent.startswith("exp24-"):
        validate_source(info["training_identity"]["source_commit"],source,source_exception)
    return {"measurement_source_commit":source,
            **({"source_exception":source_exception} if source_exception else {})}


@contextmanager
def apply_condition(model, row):
    if row["kind"] == "value-interpolation":
        with aaab_value_interpolation(model, row["coefficient"]):
            yield
    else:
        with intervention(model, None if row["kind"] == "native" else row["kind"]):
            yield


def wandb_job(name, kind, config):
    import wandb
    m = spec()["wandb"]
    run = wandb.init(**{k:m[k] for k in ("entity","project","group")}, name=name,
        job_type=kind, mode="online", tags=["experiment-24","seed-42"], config=config)
    if run is None or run.disabled:
        raise RuntimeError("online W&B is required")
    return run


def validate_record(record, condition, split):
    row = condition_row(condition)
    if row["imported"]:
        original = result_path(HISTORICAL,condition,split)
        if record != read_json(original):
            raise ValueError("imported record changed")
        return historical_record(record,condition,split)
    if (record.get("condition") != condition or record.get("parent") != row["parent"]
        or record.get("kind") != row["kind"] or record.get("split") != split
        or record.get("seed") != 42 or record.get("milestone") != 2000
        or record.get("protocol_sha256") != sha256_file(PROTOCOL)
        or record.get("matrix_sha256") != sha256_file(MATRIX)
        or len(record.get("checkpoint_sha256", "")) != 64):
        raise ValueError("invalid fixed-result identity")
    identity = record["training_identity"]
    if row["parent"].startswith("exp24-"):
        expected = identity_for_arguments(train_arguments(variant_for_id(row["parent"]),
            output="/unused",data_files=identity["data_files"].get("pattern","")),
            data_identity=identity["data_files"])
        expected["source_commit"] = identity.get("source_commit")
        if expected != identity:
            raise ValueError("result recipe differs")
        validate_source(identity["source_commit"],record["evaluation_source_commit"],
                        record.get("source_exception"))
    else:
        old = read_json(result_path(HISTORICAL,row["parent"],split))
        if identity != old["training_identity"] or record["checkpoint_sha256"] != old["checkpoint_sha256"]:
            raise ValueError("incumbent interpolation changed reference identity")
        # This is a NEW authorized measurement, not a rewritten old record.
        if record.get("reference_record_sha256") != sha256_file(result_path(HISTORICAL,row["parent"],split)):
            raise ValueError("incumbent measurement missing historical binding")
        validate_source(record["measurement_source_commit"],record["evaluation_source_commit"],
                        record.get("source_exception"))
    reference = read_json(result_path(HISTORICAL,spec()["analysis"]["primary_reference"],split))
    if record["artifact_identity"] != reference["artifact_identity"]:
        raise ValueError("fixed data/order/masks changed")
    metrics = record["metrics"]
    t,n = np.asarray(metrics["sequence_total_losses"]),np.asarray(metrics["sequence_valid_tokens"])
    if (t.shape != (512,) or n.shape != (512,) or not np.isfinite(t).all() or (t<0).any()
        or not np.array_equal(n,np.full(512,1023)) or metrics["sequence_ids"] != reference["metrics"]["sequence_ids"]
        or not np.isfinite(metrics["loss"]) or abs(metrics["loss"]-t.sum()/n.sum())>1e-10
        or not record.get("wandb",{}).get("run_id") or not record["wandb"].get("run_url")):
        raise ValueError("invalid sequence losses/counts/IDs/W&B")
    return record


def load_result(root, condition, split):
    path = result_path(root,condition,split)
    if condition_row(condition)["imported"] and sha256_file(path) != sha256_file(result_path(HISTORICAL,condition,split)):
        raise ValueError("historical bytes changed")
    return validate_record(read_json(path),condition,split)


def evaluate_condition(*, condition, checkpoint, artifact, root, split, device,
                       batch_size=1, source_exception=None):
    row = condition_row(condition)
    if row["imported"]:
        raise ValueError("historical records must be imported, not re-evaluated/relabelled")
    path = result_path(root,condition,split)
    if path.exists():
        old = load_result(root,condition,split)
        if validate_checkpoint(checkpoint,row["parent"])["checkpoint_sha256"] != old["checkpoint_sha256"]:
            raise ValueError("existing evaluation belongs to a different requested checkpoint")
        artifact_identity(artifact)
        return old
    source = require_clean_commit()
    info = validate_checkpoint(checkpoint,row["parent"])
    payload,data = artifact_identity(artifact)
    if split == "confirmation" and row["kind"] != "value-interpolation":
        verify_discovery(root)
        old = load_result(root,condition,"discovery")
        if old["checkpoint_sha256"] != info["checkpoint_sha256"]:
            raise ValueError("confirmation changed frozen checkpoint")
    extra = {"source_exception":source_exception} if source_exception else {}
    if row["parent"].startswith("exp24-"):
        validate_source(info["training_identity"]["source_commit"],source,source_exception)
    else:
        extra.update(reference_record_sha256=sha256_file(result_path(HISTORICAL,row["parent"],split)),
                     measurement_source_commit=source)
    net = load_model(checkpoint,row["parent"],dtype=torch.bfloat16,device=device).eval()
    run = wandb_job(condition+"-"+split,"fixed-evaluation",{**info,"source":source})
    try:
        with apply_condition(net,row):
            metrics = sequence_metrics(net,payload[f"{split}_input_ids"],batch_size=batch_size,device=device)
            telemetry = None
            if row["kind"] == "value-interpolation":
                from src.experiments.experiment24_probe import probe_split
                telemetry = probe_split(net,payload[f"{split}_input_ids"][:32],device,check_noop=False)
        record = {"condition":condition,"parent":row["parent"],"kind":row["kind"],
            "split":split,"seed":42,"milestone":2000,"metrics":metrics,**info,**extra,
            "evaluation_source_commit":source,"protocol_sha256":sha256_file(PROTOCOL),
            "matrix_sha256":sha256_file(MATRIX),"artifact_identity":data,
            "interpolation_telemetry":telemetry,"wandb":{"run_id":run.id,"run_url":run.url}}
        validate_record(record,condition,split)
        run.log({f"{split}/loss":metrics["loss"]})
        run.finish(exit_code=0)
        write_new(path,record)
        return record
    except BaseException:
        run.finish(exit_code=1)
        raise
    finally:
        del net


def discovery_body(root):
    records = [load_result(root,r["id"],"discovery") for r in conditions()]
    paired_identity(records)
    ids = spec()["analysis"]["selection_ids"]
    losses = {r["condition"]:r["metrics"]["loss"] for r in records}
    selected = min(ids,key=lambda i:(losses[i],2 if i.startswith("exp23-") else 3,i))
    return {"protocol_sha256":sha256_file(PROTOCOL),"matrix_sha256":sha256_file(MATRIX),
            "primary_id":spec()["analysis"]["primary_candidate"],"selected_id":selected,
            "discovery_files":{r["condition"]:sha256_file(result_path(root,r["condition"],"discovery")) for r in records},
            "checkpoints":{r["condition"]:r["checkpoint_sha256"] for r in records}}


def freeze_discovery(root):
    path = Path(root)/"data/selections/discovery_selection.json"
    body = discovery_body(root)
    value = {**body,"content_sha256":digest(body)}
    if path.exists():
        if read_json(path) != value:
            raise ValueError("frozen discovery changed")
    else:
        write_new(path,value)
    return value


def verify_discovery(root):
    record = read_json(Path(root)/"data/selections/discovery_selection.json")
    body = discovery_body(root)
    if record != {**body,"content_sha256":digest(body)}:
        raise ValueError("discovery files/selection/checkpoints changed")
    return record


def analyze(root):
    m = spec()
    selection = verify_discovery(root)
    ref = m["analysis"]["primary_reference"]
    candidates = [r["id"] for r in m["new_training_runs"]]
    result = {"selection":selection,"splits":{},"protocol_sha256":sha256_file(PROTOCOL),
        "limitations":["single seed and step-2000 screen", "reused/adaptively studied validation benchmark",
                       "historical controls, not newly paired training trajectories", "no seed uncertainty"]}
    for split in ("discovery","confirmation"):
        records = {r["id"]:load_result(root,r["id"],split) for r in conditions()}
        paired_identity(list(records.values()))
        if any(r["checkpoint_sha256"] != selection["checkpoints"][i] for i,r in records.items()):
            raise ValueError("checkpoint changed after discovery")
        def contrast(a,b,confidence=.95):
            return bootstrap_contrast([records[a]["metrics"],records[b]["metrics"]],[1,-1],
                confidence=confidence,samples=10000,seed=24042)
        result["splits"][split] = {"losses":{i:r["metrics"]["loss"] for i,r in records.items()},
            "candidate_contrasts":{i:{"descriptive95":contrast(i,ref),"family97_5":contrast(i,ref,.975)} for i in candidates},
            "abbc_minus_aabc":contrast(candidates[1],candidates[0]),
            "context_vs_aaab":{i:contrast(i,ref) for i,r in records.items() if r["kind"]=="native"},
            "interventions":{i:contrast(i,r["parent"]) for i,r in records.items() if r["kind"]!="native"}}
    result["family_promising"] = {}
    for i in candidates:
        d,c = (result["splits"][s]["candidate_contrasts"][i] for s in ("discovery","confirmation"))
        meaningful = d["descriptive95"]["delta_loss"] <= -.002 and c["descriptive95"]["delta_loss"] <= -.002
        result["family_promising"][i] = meaningful and c["family97_5"]["ci_high"] < 0
        if i == m["analysis"]["primary_candidate"]:
            result["primary_promising"] = meaningful and c["descriptive95"]["ci_high"] < 0
    path = Path(root)/"data/derived/paired_analysis.json"
    if path.exists():
        if read_json(path) != result:
            raise ValueError("existing analysis differs; preserve it and use a fresh derived root")
    else:
        write_new(path,result)
    return result
