"""Passive, split-preserving route/content diagnostics; original SDPA stays live."""
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path
import math

import torch
from torch.nn import functional as F

from src.experiments.experiment23_probe import mean_valid, head_metrics
from src.experiments.experiment24_workflow import PROTOCOL, spec, write_new
from src.experiments.experiment24_evaluation import (
    artifact_identity, validate_checkpoint, load_model, wandb_job, measurement_binding,
)
from src.experiments.experiment1_partition_compatibility import sha256_file


def route_metrics(layer, blocks, partial, raw):
    source = torch.stack(blocks+[partial])
    n,b,t,d = source.shape
    h = layer.attnres_num_heads
    keys = layer.attn_res_norm(source).reshape(n,b,t,h,d//h)
    projs = {"A":layer.attn_res_proj,"B":layer.local_attn_res_proj}
    if "C" in raw:
        projs["C"] = layer.third_attn_res_proj
    probabilities = {r:torch.einsum("hk,nbthk->nbth",p.weight.reshape(h,-1),keys).softmax(0).float()
                     for r,p in projs.items()}
    grouped = {r:v.float().reshape(b,t,h,-1) for r,v in raw.items()}
    out = {}
    for r,p in probabilities.items():
        out[f"route_entropy_{r}"] = mean_valid(-torch.special.xlogy(p,p).sum(0)/(math.log(n) if n>1 else 1))
        out[f"route_rms_{r}"] = mean_valid(grouped[r].square().mean(-1).sqrt())
    for r,s in combinations(raw,2):
        p,q = probabilities[r],probabilities[s]
        m = (p+q)/2
        js = .5*(torch.special.xlogy(p,p.clamp_min(1e-30)/m.clamp_min(1e-30))
                  +torch.special.xlogy(q,q.clamp_min(1e-30)/m.clamp_min(1e-30))).sum(0)
        out[f"route_js_{r}_{s}"] = mean_valid(js)
        out[f"route_cosine_{r}_{s}"] = mean_valid(F.cosine_similarity(grouped[r],grouped[s],dim=-1))
        out[f"route_difference_norm_{r}_{s}"] = mean_valid((grouped[r]-grouped[s]).norm(dim=-1))
    out["effective_V_K_route_js"] = out["route_js_B_C"] if "C" in raw else out["route_js_A_B"]
    return out


def content_metrics(layer, partial, raw):
    inputs = {r:layer.input_layernorm(layer._apply_gate(partial,x,"attn")) for r,x in raw.items()}
    out = {f"input_rms_{r}":mean_valid(x.float().reshape(*x.shape[:2],8,-1).square().mean(-1).sqrt())
           for r,x in inputs.items()}
    # Same trained V projection, native histories only; these are observers.
    values = {r:layer.self_attn.v_proj(x).float().reshape(*x.shape[:2],8,-1) for r,x in inputs.items()}
    for r,v in values.items():
        out[f"value_norm_{r}"] = mean_valid(v.norm(dim=-1))
        out[f"value_rms_{r}"] = mean_valid(v.square().mean(-1).sqrt())
    for r,s in combinations(values,2):
        out[f"value_cosine_{r}_{s}"] = mean_valid(F.cosine_similarity(values[r],values[s],dim=-1))
    return out


@contextmanager
def observe(model, rows):
    if model.training or model.config._attn_implementation != "sdpa":
        raise ValueError("passive observations require frozen SDPA")
    saved = []
    def accumulate(i, values):
        row = rows.setdefault(i,{})
        for k,v in values.items():
            if not torch.isfinite(v).all():
                raise ValueError(f"non-finite probe: {k}")
            row.setdefault(k,[]).append(v.detach().cpu().double())
    try:
        for i,l in enumerate(model.model.layers):
            name = "_experiment24_route_probe" if hasattr(l,"_experiment24_three_route") else "_experiment23_route_probe"
            old = getattr(l,name,None)
            old_attn = getattr(l.self_attn,"_experiment23_attention_probe",None)
            saved.append((name,old,old_attn))
            def capture(layer,blocks,partial,raw,index=i):
                accumulate(index,{**route_metrics(layer,blocks,partial,raw),**content_metrics(layer,partial,raw)})
            if name == "_experiment24_route_probe":
                def cb(layer,blocks,partial,raw,prior=old,record=capture):
                    if prior is not None:
                        prior(layer,blocks,partial,raw)
                    record(layer,blocks,partial,raw)
            else:
                def cb(layer,blocks,partial,a,b,prior=old,record=capture):
                    # Preserve an active interpolation adapter before observing.
                    if prior is not None:
                        prior(layer,blocks,partial,a,b)
                    record(layer,blocks,partial,{"A":a,"B":b})
            setattr(l,name,cb)
            def attn(module,q,k,mask,out,index=i,prior=old_attn):
                if prior is not None:
                    prior(module,q,k,mask,out)
                metrics = head_metrics(module,q,k,mask,out)
                heads = out.float().reshape(*out.shape[:2],8,2,module.head_dim)
                weights = module.o_proj.weight.float().reshape(module.o_proj.out_features,8,2,module.head_dim)
                # Norm of the summed two writes, including their cross term.
                w = weights.permute(1,2,3,0).reshape(8,2*module.head_dim,-1)
                pair = heads.flatten(-2)
                gram = w @ w.transpose(-1,-2)
                norm = torch.einsum("btgi,gij,btgj->btg",pair,gram,pair).clamp_min(0).sqrt()
                metrics["true_pair_residual_norm"] = mean_valid(norm)
                metrics["attention_output_rms"] = mean_valid(heads.square().mean(dim=(-1,-2)).sqrt())
                metrics["residual_write_rms"] = mean_valid(norm/math.sqrt(module.o_proj.out_features))
                accumulate(index,metrics)
            l.self_attn._experiment23_attention_probe = attn
        yield saved
    finally:
        for l,(name,old,attn) in zip(model.model.layers,saved):
            setattr(l,name,old)
            l.self_attn._experiment23_attention_probe = attn


def probe_split(model, tokens, device, check_noop=True):
    rows,logits_rows = {},[]
    with torch.no_grad(), observe(model,rows) as saved:
        for x in tokens.split(1):
            x = x.to(device=device,dtype=torch.long)
            logits = model(x,use_cache=False).logits
            values = logits[:,:-1].float()
            logits_rows.append({"rms":float(values.square().mean().sqrt()),"max_abs":float(values.abs().max()),
                                "finite":bool(torch.isfinite(values).all())})
            if not logits_rows[-1]["finite"]:
                raise ValueError("non-finite probe logits")
            if check_noop:
                callbacks = []
                try:
                    for l,(name,prior,attn) in zip(model.model.layers,saved):
                        callbacks.append((getattr(l,name),l.self_attn._experiment23_attention_probe))
                        setattr(l,name,prior); l.self_attn._experiment23_attention_probe=attn
                    plain = model(x,use_cache=False).logits
                    if not torch.equal(logits,plain):
                        raise RuntimeError("passive instrumentation changed native logits")
                finally:
                    for l,(name,_,_), (route,attn) in zip(model.model.layers,saved,callbacks):
                        setattr(l,name,route); l.self_attn._experiment23_attention_probe=attn
    if set(rows) != set(range(len(model.model.layers))):
        raise ValueError("incomplete probe layers")
    keys = set(rows[0])
    if any(set(row)!=keys or any(len(v)!=len(tokens) for v in row.values()) for row in rows.values()):
        raise ValueError("incomplete/mixed probe reduction")
    return {"maps":{key:[torch.stack(rows[i][key]).mean(0).tolist() for i in range(len(rows))] for key in sorted(keys)},
            "logit_statistics":logits_rows,"sequences":len(tokens),"passive_logits_bit_identical":check_noop}


def evaluate_probe(*, parent, checkpoint, artifact, root, device, source_exception=None):
    allowed = spec()["compute"]["conditions"]
    if parent not in allowed:
        raise ValueError("only AAAB/AABC/ABBC have probe jobs")
    path = Path(root)/"data/raw/probe"/parent/"result.json"
    info = validate_checkpoint(checkpoint,parent)
    payload,data = artifact_identity(artifact)
    if path.exists():
        old = __import__("json").loads(path.read_text())
        if any(old.get(k)!=v for k,v in info.items()) or old["artifact_identity"]!=data:
            raise ValueError("existing probe identity changed")
        return old
    binding = measurement_binding(parent,info,source_exception)
    net = load_model(checkpoint,parent,dtype=torch.bfloat16,device=device).eval()
    splits = {s:probe_split(net,payload[f"{s}_input_ids"][:32],device) for s in ("discovery","confirmation")}
    job = "exp24-probe-reference-aaab" if parent.startswith("exp23-") else "exp24-probe-"+parent.split("exp24-",1)[1].split("-seed42",1)[0]
    run = wandb_job(job,"passive-probe",info)
    try:
        result = {"parent":parent,**info,**binding,"artifact_identity":data,"protocol_sha256":sha256_file(PROTOCOL),
                  "splits":splits,"wandb":{"run_id":run.id,"run_url":run.url}}
        run.log({"probe/complete":1}); run.finish(exit_code=0)
        write_new(path,result)
        return result
    except BaseException:
        run.finish(exit_code=1)
        raise
