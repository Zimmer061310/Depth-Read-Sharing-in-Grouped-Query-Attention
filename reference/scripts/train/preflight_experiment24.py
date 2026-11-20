"""Disposable CUDA readiness gates; explicit invocation only, never training launch."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

from scripts.train.preflight_experiment23 import (
    invocation, tensor_digest, environment as inherited_environment,
    compare_tree, verify_worker_restoration,
)
from src.training import train_scratch as training
from src.experiments.experiment24_workflow import (
    ROOT, PROTOCOL, binding_identity, train_arguments, identity_for_arguments,
    validate_training_data_identity, spec, RUNS, write_new,
)
from src.experiments.experiment24_three_route import router_backend, aaab_value_interpolation
from src.experiments.experiment24_evaluation import (
    artifact_identity, sequence_metrics, validate_checkpoint, load_model, import_references,
    result_path, evaluate_condition,
)
from src.experiments.experiment1_partition_compatibility import sha256_file


def environment():
    env = inherited_environment()
    env.update(router_backend="tiled-r2-exact",router_backward="canonical_bf16_vjp_three_real_routes")
    return env


def report(output, gate, **values):
    write_new(Path(output)/(gate+".json"),{"gate":gate,"passed":True,
              **binding_identity(),"environment":environment(),**values})


def model(variant, data_files):
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    with invocation(train_arguments(variant,output="/unused",data_files=data_files,disposable=True)) as args:
        return training.build_model(args,torch.device("cuda")).eval()


def parity(output, artifact, data_files):
    payload,data = artifact_identity(artifact)
    rows,common,rng = [],None,None
    for variant in RUNS:
        net=model(variant,data_files)
        hashes={n:tensor_digest(p) for n,p in net.named_parameters()
                if not n.endswith(("local_attn_res_proj.weight","third_attn_res_proj.weight"))}
        current_rng=tensor_digest(torch.get_rng_state())
        if common is None: common,rng=hashes,current_rng
        if common!=hashes or rng!=current_rng:
            raise RuntimeError("common initializer/base RNG differs")
        for l in net.model.layers:
            if not torch.equal(l.attn_res_proj.weight,l.local_attn_res_proj.weight) or not torch.equal(l.attn_res_proj.weight,l.third_attn_res_proj.weight):
                raise RuntimeError("A/B/C initialization differs")
        torch.manual_seed(24042)
        with torch.no_grad():
            for l in net.model.layers:
                for p,std in zip((l.attn_res_proj,l.local_attn_res_proj,l.third_attn_res_proj),(.03,.04,.05)):
                    p.weight.normal_(0,std)
        with router_backend(net,"canonical"):
            reference=sequence_metrics(net,payload["discovery_input_ids"],batch_size=1,device="cuda")
        actual=sequence_metrics(net,payload["discovery_input_ids"],batch_size=1,device="cuda")
        maximum=max(abs(a-b) for a,b in zip(reference["sequence_losses"],actual["sequence_losses"]))
        if maximum>.0005: raise RuntimeError(f"{variant} BF16 parity: {maximum}")
        rows.append({"variant":variant,"sequences":512,"max_per_sequence_loss_difference":maximum,
                     "unequal_routers":True,"projection_reference":"same packed/half-output GEMM shapes"})
        del net;gc.collect();torch.cuda.empty_cache()
    report(output,"full_discovery_router_parity",rows=rows,artifact_identity=data,
           common_initializer_hashes=common,construction_cpu_rng_sha256=rng)


def backward(output):
    from tests.test_experiment24_graph import make
    rows=[]
    for variant in RUNS:
        for checkpointing in (False,True):
            snapshots=[]
            for backend in ("canonical","tiled-r2-exact"):
                net=make(variant,backend).to(device="cuda",dtype=torch.bfloat16).train()
                with torch.no_grad():
                    for l in net.model.layers:
                        for p in (l.attn_res_proj,l.local_attn_res_proj,l.third_attn_res_proj): p.weight.normal_(0,.04)
                if checkpointing: net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
                x=torch.arange(16,device="cuda").reshape(2,8)
                opt=torch.optim.AdamW(net.parameters(),lr=5e-4,betas=(.9,.95),eps=1e-8,weight_decay=.1)
                loss=net(x,labels=x).loss;loss.backward()
                gradients={n:None if p.grad is None else p.grad.detach().cpu() for n,p in net.named_parameters()}
                norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
                if not torch.isfinite(loss) or not torch.isfinite(norm): raise RuntimeError("non-finite VJP")
                opt.step()
                snapshots.append((float(loss),gradients,{n:p.detach().cpu() for n,p in net.named_parameters()}))
                del net,opt;gc.collect();torch.cuda.empty_cache()
            if abs(snapshots[0][0]-snapshots[1][0])>.0005: raise RuntimeError("checkpointed Loss mismatch")
            for index in (1,2):
                for name,a in snapshots[0][index].items():
                    b=snapshots[1][index][name]
                    if a is None or b is None:
                        if a is not b: raise RuntimeError("gradient coverage mismatch")
                    elif not torch.allclose(a,b,atol=1e-6,rtol=1e-5): raise RuntimeError(f"BF16 VJP/update mismatch: {name}")
            rows.append({"variant":variant,"checkpointing":checkpointing,"gradient_atol":1e-6,"rtol":1e-5})
    report(output,"checkpointed_bf16_vjp",rows=rows)


def reliability(output, artifact, checkpoint):
    from src.experiments.experiment21_evaluation import read_json
    from src.experiments.experiment23_route_topology import execution_backend
    parent=spec()["analysis"]["primary_reference"]
    info=validate_checkpoint(checkpoint,parent)
    payload,data=artifact_identity(artifact)
    net=load_model(checkpoint,parent,dtype=torch.bfloat16,device="cuda").eval()
    endpoint_rows=[]
    for coefficient,condition in ((0,parent),(1,parent+"-collapse-b")):
        with aaab_value_interpolation(net,coefficient):
            metrics=sequence_metrics(net,payload["discovery_input_ids"],batch_size=1,device="cuda")
        old=read_json(result_path(ROOT/"results/experiment23",condition,"discovery"))
        difference=max(abs(a-b) for a,b in zip(metrics["sequence_losses"],old["metrics"]["sequence_losses"]))
        if difference>.0005: raise RuntimeError(f"AAAB endpoint reproduction failed: {difference}")
        endpoint_rows.append({"coefficient":coefficient,"sequences":512,"max_per_sequence_loss_difference":difference})
    rows=[]
    telemetry=[]
    for coefficient in (0,.25,.5,.75,1):
        with aaab_value_interpolation(net,coefficient):
            for split in ("discovery","confirmation"):
                actual=sequence_metrics(net,payload[f"{split}_input_ids"],batch_size=1,device="cuda")
                if split=="discovery":
                    with execution_backend(net,router="shared-eager",projection="eager-matched"):
                        reference=sequence_metrics(net,payload["discovery_input_ids"][:32],batch_size=1,device="cuda")
                    maximum=max(abs(a-b) for a,b in zip(actual["sequence_losses"][:32],reference["sequence_losses"]))
                    if maximum>.0005: raise RuntimeError("AAAB interpolation shape-matched parity failed")
                rows.append({"coefficient":coefficient,"split":split,"loss":actual["loss"],"finite":True})
                if coefficient in (0,1):
                    from src.experiments.experiment24_probe import probe_split
                    telemetry.append({"coefficient":coefficient,"split":split,
                                      "probe":probe_split(net,payload[f"{split}_input_ids"][:32],torch.device("cuda"))})
    del net;gc.collect();torch.cuda.empty_cache()
    # Produce the six genuine W&B-recorded interior conditions, not extra jobs.
    publication=Path(output)/"reliability-publication"
    import_references(publication)
    for r in spec()["evaluation"]["aaab_value_interpolation"]["new_conditions"]:
        for split in ("discovery","confirmation"):
            evaluate_condition(condition=r["id"],checkpoint=checkpoint,artifact=artifact,root=publication,
                               split=split,device=torch.device("cuda"))
    report(output,"aaab_reliability",**info,artifact_identity=data,endpoints=endpoint_rows,
           finite_curve=rows,endpoint_telemetry=telemetry,
           interior_records_root=str(publication),monotonicity_required=False)


def smoke(output,data_files):
    """Full ABBC trainer, exact loaded bytes and two independent resumes.

    Leaves disposable states/logs intact; no broad recursive cleanup and no
    production optimizer update. Continuous comparison is diagnostic only.
    """
    root=Path(output)/"disposable-smoke"
    root.mkdir(parents=True,exist_ok=False)
    if shutil.disk_usage(root).free<65*1024**3: raise RuntimeError("need 65 GiB scratch for atomic disposable checkpoints")
    env=dict(os.environ,WANDB_MODE="disabled",CUBLAS_WORKSPACE_CONFIG=":4096:8",
             PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",HF_HUB_OFFLINE="1")
    started=time.monotonic();rows=[]
    for name,stop,resume in (("saved",1,None),("resume-a",2,"saved"),("resume-b",2,"saved"),("continuous",2,None)):
        directory=root/name
        if resume:
            shutil.copytree(root/resume/"step-1",directory/"step-1")
            if training.sha256_tree(directory/"step-1")!=training.sha256_tree(root/resume/"step-1"):
                raise RuntimeError("resume copy changed bytes")
        args=train_arguments("global-abbc",output=directory,data_files=data_files,disposable=True,
                             resume=directory/"step-1" if resume else None)
        for key,value in {"--stop_after_step":stop,"--save_every":1,"--keep_steps":"1,2",
                          "--eval_every":1,"--eval_steps":1,"--log_every":1}.items(): args[args.index(key)+1]=str(value)
        with (root/(name+".log")).open("x") as log:
            subprocess.run([sys.executable,"-m","torch.distributed.run","--nproc_per_node=1",
                "--master_port",os.environ.get("MHAR_PREFLIGHT_PORT","30434"),
                "--module","scripts.train.preflight_experiment24","--worker",*args],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        details=[json.loads(l.split("EXP24_WORKER ",1)[1]) for l in (root/(name+".log")).read_text().splitlines() if l.startswith("EXP24_WORKER ")]
        if len(details)!=1: raise RuntimeError("missing worker integrity/memory evidence")
        rows.append(details[0])
    from safetensors import safe_open
    def weights(path):
        out={}
        for p in path.glob("*.safetensors"):
            with safe_open(p,framework="pt",device="cpu") as f:
                out.update({n:tensor_digest(f.get_tensor(n)) for n in f.keys()})
        return out
    def state(name):
        return torch.load(root/name/"step-2/training_state.pt",map_location="cpu",weights_only=False,mmap=True)
    a,b,c=state("resume-a"),state("resume-b"),state("continuous")
    for key in ("run_identity","optimizer","scheduler","cpu_rng_state","cuda_rng_state_all","global_step","chunks_consumed"):
        compare_tree(a[key],b[key])
    if weights(root/"resume-a/step-2")!=weights(root/"resume-b/step-2"): raise RuntimeError("independent resumes differ")
    saved=torch.load(root/"saved/step-1/training_state.pt",map_location="cpu",weights_only=False,mmap=True)
    count=len(weights(root/"saved/step-1"))
    if (saved["global_step"]!=1 or saved["chunks_consumed"]!=32 or a["global_step"]!=2
        or a["chunks_consumed"]!=64 or len(saved["optimizer"]["state"])!=count):
        raise RuntimeError("full model optimizer coverage/packed position incomplete")
    for row in rows[1:3]:
        integrity=row["restored"]
        if integrity.get("model_tensors_exact")!=count or any(integrity.get(k) is not True for k in
            ("optimizer_exact","scheduler_exact","cpu_rng_exact","cuda_rng_exact")):
            raise RuntimeError("actual pre-update restore verification failed")
    if rows[1]["next_batch_hashes"]!=rows[2]["next_batch_hashes"]: raise RuntimeError("resumed next batches differ")
    report(output,"full_optimizer_and_repeatable_resume",seconds=time.monotonic()-started,
        worker_evidence=rows,checkpoint_tensor_count=count,reproducible_resumes=True,
        parameter_dtype_policy="native BF16; checked in worker",optimizer_dtype_policy="native BF16 moments; checked in worker",
        uninterrupted_weights_identical_diagnostic=weights(root/"resume-a/step-2")==weights(root/"continuous/step-2"),
        uninterrupted_optimizer_first_moment_max_abs_diagnostic=max(float((a["optimizer"]["state"][k]["exp_avg"].float()-c["optimizer"]["state"][k]["exp_avg"].float()).abs().max()) for k in a["optimizer"]["state"]))


def certificate(output, artifact, data_files):
    artifact_identity(artifact)
    args=train_arguments("global-aabc",output="/unused",data_files=data_files,disposable=True)
    identity=identity_for_arguments(args)
    validate_training_data_identity(identity["data_files"])
    initial=json.loads((Path(output)/"full_discovery_router_parity.json").read_text())
    subprocess.run([sys.executable,"-m","unittest","tests.test_experiment24_graph","-q"],cwd=ROOT,check=True)
    streams={}
    with invocation(args) as parsed:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(parsed.tokenizer,revision=parsed.tokenizer_revision,local_files_only=True)
        for label,seed,count in (("training",42,64000),("monitoring",10041,1000)):
            it=training.token_stream(parsed.dataset,parsed.dataset_name,tok,1024,0,1,seed,
                data_files=data_files,dataset_revision=parsed.dataset_revision)
            h=hashlib.sha256();first=[]
            for index in range(count):
                x=next(it)
                if tuple(x.shape)!=(1025,): raise RuntimeError("packed stream shape differs")
                h.update(x.contiguous().numpy().tobytes())
                if index<33: first.append(tensor_digest(x))
            streams[label]={"seed":seed,"chunks":count,"sha256":h.hexdigest(),"first33_chunk_hashes":first}
    report(output,"initializer_stream_alias",streams=streams,training_data=identity["data_files"],
           initializer=initial["common_initializer_hashes"],alias_abac_abbc_unequal_router_test=True)


def bind(output,artifact,data_files):
    artifact_identity(artifact)
    bindings=binding_identity();gates={};env=environment()
    for name in json.loads(PROTOCOL.read_text())["cuda_gates"]:
        p=Path(output)/(name+".json");row=json.loads(p.read_text())
        if row.get("passed") is not True or any(row.get(k)!=v for k,v in bindings.items()): raise ValueError("stale/failed gate")
        if any(row["environment"].get(k)!=v for k,v in env.items() if k not in ("device_uuid","visible_devices")):
            raise ValueError("gate environment changed")
        gates[name]={"path":str(p.resolve()),"sha256":sha256_file(p)}
    cert=json.loads((Path(output)/"initializer_stream_alias.json").read_text())
    for variant in RUNS:
        write_new(Path(output)/"execution-locks"/(variant+".json"),{**bindings,"variant":variant,"environment":env,
            "training_identity":identity_for_arguments(train_arguments(variant,output="/unused",data_files=data_files)),
            "initializer_and_stream_certificate":cert,"readiness_gate_hashes":gates})


def worker():
    from unittest.mock import patch
    sys.argv=["preflight-worker",*sys.argv[sys.argv.index("--worker")+1:]]
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.cuda.reset_peak_memory_stats()
    original=training.token_stream;next_batches=[]
    def stream(*args,**kwargs):
        for x in original(*args,**kwargs):
            if args[6]==42 and len(next_batches)<32: next_batches.append(tensor_digest(x))
            yield x
    with verify_worker_restoration() as restored,patch.object(training,"token_stream",stream):
        training.main()
    # Validate actual state dtype, not a policy string.
    args=training.parse_args();checkpoint=Path(args.out_dir)/f"step-{args.stop_after_step}"
    state=torch.load(checkpoint/"training_state.pt",map_location="cpu",weights_only=False,mmap=True)
    if any(v[k].dtype!=torch.bfloat16 for v in state["optimizer"]["state"].values() for k in ("exp_avg","exp_avg_sq")):
        raise RuntimeError("native optimizer moment policy changed")
    from safetensors import safe_open
    for shard in checkpoint.glob("*.safetensors"):
        with safe_open(shard,framework="pt",device="cpu") as tensors:
            if any(tensors.get_slice(n).get_dtype()!="BF16" for n in tensors.keys()):
                raise RuntimeError("native parameter dtype policy changed")
    print("EXP24_WORKER "+json.dumps({"restored":restored,"next_batch_hashes":next_batches,
          "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
          "peak_reserved_bytes":torch.cuda.max_memory_reserved()}),flush=True)


def main():
    if "--worker" in sys.argv:
        worker();return
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("parity","backward","reliability","smoke","certificate","bind"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--artifact",type=Path,required=True)
    p.add_argument("--data-files",required=True)
    p.add_argument("--aaab-checkpoint",type=Path)
    a=p.parse_args();environment();binding_identity()
    if not os.environ.get("HF_HOME"): raise ValueError("export verified HF_HOME cache")
    a.output.mkdir(parents=True,exist_ok=True)
    if a.stage=="parity": parity(a.output,a.artifact,a.data_files)
    elif a.stage=="backward": backward(a.output)
    elif a.stage=="reliability":
        if a.aaab_checkpoint is None: p.error("reliability requires --aaab-checkpoint")
        reliability(a.output,a.artifact,a.aaab_checkpoint)
    elif a.stage=="smoke": smoke(a.output,a.data_files)
    elif a.stage=="certificate": certificate(a.output,a.artifact,a.data_files)
    else: bind(a.output,a.artifact,a.data_files)


if __name__=="__main__": main()
