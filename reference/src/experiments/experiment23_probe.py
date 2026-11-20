"""Passive source-route and paired-head observations; never change SDPA."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import math

import torch
from torch.nn import functional as F

from src.experiments.experiment13_functional_difference import _attention_probability_blocks
from src.experiments.experiment23_evaluation import (
    artifact_identity, validate_checkpoint, sequence_metrics, wandb_job,
)
from src.experiments.experiment23_route_topology import Experiment23MHARForCausalLM
from src.experiments.experiment23_workflow import spec, training_id, write_new, PROTOCOL
from src.experiments.experiment1_partition_compatibility import sha256_file


def mean_valid(tensor):
    # Full immutable sequences, valid immediate-next-token query positions.
    return tensor[:, :-1].float().mean(dim=(0,1))


def route_metrics(layer, blocks, partial, a, b):
    source = torch.stack(blocks + [partial])
    n, batch, tokens, width = source.shape
    groups = layer.attnres_num_heads
    keys = layer.attn_res_norm(source).reshape(n, batch, tokens, groups, width//groups)
    def probabilities(proj):
        return torch.einsum("gk,nbtgk->nbtg", proj.weight.reshape(groups,-1), keys).softmax(0).float()
    p = probabilities(layer.attn_res_proj)
    entropy_a = -torch.special.xlogy(p,p).sum(0) / (math.log(n) if n > 1 else 1)
    result = {"route_entropy_a": mean_valid(entropy_a), "route_js": None,
              "route_entropy_b": None, "route_difference_norm": None}
    if layer._experiment23_active_b:
        q = probabilities(layer.local_attn_res_proj)
        m = (p+q)/2
        js = .5 * (torch.special.xlogy(p,p.clamp_min(1e-30)/m.clamp_min(1e-30))
                  + torch.special.xlogy(q,q.clamp_min(1e-30)/m.clamp_min(1e-30))).sum(0)
        result.update(route_js=mean_valid(js), route_entropy_b=mean_valid(
            -torch.special.xlogy(q,q).sum(0)/(math.log(n) if n > 1 else 1)),
            route_difference_norm=mean_valid((b-a).float().reshape(batch,tokens,groups,-1).norm(dim=-1)))
    return result


def head_metrics(module, query, key, mask, output):
    batch, tokens, _ = output.shape
    heads = output.float().reshape(batch,tokens,8,2,module.head_dim)
    q2, q1 = heads[...,0,:], heads[...,1,:]
    # Exact isolated residual norms via each head's small Gram matrix.
    w = module.o_proj.weight.float().reshape(module.o_proj.out_features,8,2,module.head_dim)
    w2, w1 = w[:,:,0,:].permute(1,2,0), w[:,:,1,:].permute(1,2,0)
    gram2 = torch.einsum("gio,gjo->gij",w2,w2)
    gram1 = torch.einsum("gio,gjo->gij",w1,w1)
    cross = torch.einsum("gio,gjo->gij",w2,w1)
    n2 = torch.einsum("btgi,gij,btgj->btg",q2,gram2,q2).clamp_min(0).sqrt()
    n1 = torch.einsum("btgi,gij,btgj->btg",q1,gram1,q1).clamp_min(0).sqrt()
    cos = torch.einsum("btgi,gij,btgj->btg",q2,cross,q1)/(n2*n1).clamp_min(1e-12)
    result = {"pre_norm_q1": mean_valid(q1.norm(dim=-1)), "pre_norm_q2": mean_valid(q2.norm(dim=-1)),
        "residual_norm_q1": mean_valid(n1), "residual_norm_q2": mean_valid(n2),
        "output_cosine": mean_valid(F.cosine_similarity(q2,q1,dim=-1)),
        "residual_cosine": mean_valid(cos)}
    entropy = torch.zeros(batch,16,tokens,device=query.device)
    js = torch.zeros(batch,8,tokens,device=query.device)
    for start,end,p,block_mask in _attention_probability_blocks(query,key,mask,
            scaling=module.scaling,query_block_size=64,num_key_value_groups=2,
            sliding_window=module.sliding_window):
        p = p.float()
        available = (block_mask[:,0].float() >= 0).sum(-1).float()
        normalizer = available.clamp_min(2).log()[:,None,:]
        entropy[:,:,start:end] = torch.where(available[:,None,:]>1,
            -torch.special.xlogy(p,p).sum(-1)/normalizer, 0)
        p2,p1 = p[:,0::2],p[:,1::2]
        m = (p1+p2)/2
        js[:,:,start:end] = .5*(torch.special.xlogy(p1,p1.clamp_min(1e-30)/m.clamp_min(1e-30))
                              +torch.special.xlogy(p2,p2.clamp_min(1e-30)/m.clamp_min(1e-30))).sum(-1)
    result.update(attention_entropy_q1=mean_valid(entropy[:,1::2].transpose(1,2)),
                  attention_entropy_q2=mean_valid(entropy[:,0::2].transpose(1,2)),
                  attention_js=mean_valid(js.transpose(1,2)))
    return result


@contextmanager
def observe(model, rows):
    if model.training or model.config._attn_implementation != "sdpa":
        raise ValueError("passive probe requires eval SDPA")
    saved = []
    def accumulate(index, measurements):
        target = rows.setdefault(index,{})
        for key,value in measurements.items():
            if value is None:
                target[key] = None
            else:
                target.setdefault(key,[]).append(value.cpu().double())
    try:
        for i,layer in enumerate(model.model.layers):
            saved.append((getattr(layer,"_experiment23_route_probe",None),
                          getattr(layer.self_attn,"_experiment23_attention_probe",None)))
            layer._experiment23_route_probe = lambda l,blocks,partial,a,b,index=i: accumulate(index,
                route_metrics(l,blocks,partial,a,b))
            layer.self_attn._experiment23_attention_probe = lambda module,q,k,mask,out,index=i: accumulate(index,
                head_metrics(module,q,k,mask,out))
        yield
    finally:
        for layer,(a,b) in zip(model.model.layers,saved):
            layer._experiment23_route_probe = a
            layer.self_attn._experiment23_attention_probe = b


def evaluate_probe(*, variant, checkpoint, artifact, root, device):
    parent = training_id(variant)
    info = validate_checkpoint(checkpoint,parent)
    payload,data = artifact_identity(artifact)
    run_id = "exp23-probe-" + variant
    path = Path(root)/"data/raw/probe"/run_id/"result.json"
    if path.exists():
        raise FileExistsError(path)
    from src.experiments.experiment23_reuse import load_model
    net = load_model(checkpoint, variant, dtype=torch.bfloat16, device=device).eval()
    splits = {}
    for split in ("discovery","confirmation"):
        rows = {}
        tokens = payload[f"{split}_input_ids"][:32]
        # Batch one is intentional so each recorded batch receives equal weight.
        with torch.no_grad(), observe(net,rows):
            for batch in tokens.split(1):
                x = batch.to(device)
                logits = net(x,use_cache=False).logits
                # Disable callbacks for an exact same-backend comparison.
                callbacks = [(l._experiment23_route_probe,l.self_attn._experiment23_attention_probe)
                             for l in net.model.layers]
                try:
                    for l in net.model.layers:
                        l._experiment23_route_probe = None
                        l.self_attn._experiment23_attention_probe = None
                    plain = net(x,use_cache=False).logits
                    if not torch.equal(logits,plain):
                        raise RuntimeError("passive instrumentation changed logits")
                finally:
                    for l,(a,b) in zip(net.model.layers,callbacks):
                        l._experiment23_route_probe = a; l.self_attn._experiment23_attention_probe = b
        if len(rows) != len(net.model.layers):
            raise RuntimeError("incomplete layer probe")
        keys = rows[0].keys()
        splits[split] = {key: [None if rows[i][key] is None else torch.stack(rows[i][key]).mean(0).tolist()
                               for i in range(len(rows))] for key in keys}
    run = wandb_job(run_id,"passive-probe",info)
    try:
        result = {"id":run_id,"parent":parent,**info,"artifact_identity":data,
            "protocol_sha256":sha256_file(PROTOCOL),"passive_logits_bit_identical":True,
            "sequences_per_split":32,"splits":splits,"wandb":{"run_id":run.id,"run_url":run.url}}
        write_new(path,result)
        run.log({"probe/complete":1})
        return result
    finally:
        run.finish()
