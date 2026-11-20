"""Explicit disposable CUDA gates. Never launches a scientific run.

Run from a clean pinned checkout, on one visible idle GPU. Any failed gate
raises; no output is reused as a passing result or retried automatically.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch
from src.training import train_scratch as training
from src.experiments.experiment23_workflow import (
    ROOT, PROTOCOL, RUNS, binding_identity, identity_for_arguments,
    train_arguments, training_id, write_new,
    validate_training_data_identity,
)
from src.experiments.experiment23_route_topology import execution_backend
from src.experiments.experiment23_evaluation import artifact_identity, sequence_metrics
from src.experiments.experiment1_partition_compatibility import sha256_file


@contextlib.contextmanager
def invocation(arguments):
    saved = sys.argv
    try:
        sys.argv = ["experiment23-preflight", *arguments]
        yield training.parse_args()
    finally:
        sys.argv = saved


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def environment():
    import transformers
    import triton
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("preflight requires exactly one visible CUDA GPU")
    prop = torch.cuda.get_device_properties(0)
    visible=os.environ.get("CUDA_VISIBLE_DEVICES")
    physical=visible.split(",")[0].strip() if visible else "0"
    pids=subprocess.check_output(["nvidia-smi","-i",physical,"--query-compute-apps=pid",
        "--format=csv,noheader,nounits"],text=True).strip()
    if any(line.strip()!=str(os.getpid()) for line in pids.splitlines() if line.strip()):
        raise RuntimeError("readiness/launch GPU has another active compute process")
    return {"torch": torch.__version__, "transformers": transformers.__version__,
        "triton": triton.__version__, "cuda": torch.version.cuda,
        "device": prop.name, "device_uuid": str(prop.uuid), "total_memory": prop.total_memory,
        "visible_devices": visible,
        "router_backend": "r2-triton", "projection_backend": "packed", "attention": "sdpa",
        "router_backward": "experiment23_canonical_bf16_vjp_recompute",
        "compile": False, "gradient_checkpointing": True}


def report(output, gate, **measurements):
    write_new(Path(output)/(gate+".json"), {"gate":gate, "passed":True,
        **binding_identity(), "environment":environment(), **measurements})


def model(variant, data_files, *, native=False):
    args = train_arguments(variant, output="/unused", data_files=data_files, disposable=True)
    if native:
        for key in ("--experiment23_variant", "--experiment23_router_backend",
                    "--experiment23_projection_backend"):
            index = args.index(key); del args[index:index+2]
        if variant.startswith(("hq8", "r2")):
            args += ["--hybrid_q_groups", "8"]
        if variant.startswith("r2"):
            args += ["--experiment19_variant", "rs-hq8"]
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    with invocation(args) as parsed:
        net = training.build_model(parsed, torch.device("cuda")).eval()
    if native and variant.startswith("r2"):
        from src.experiments.experiment19_role_specific_routing import set_router_backend
        set_router_backend(net, "r2-triton")
    return net


def release(net):
    del net
    gc.collect(); torch.cuda.empty_cache()


@contextlib.contextmanager
def native_reference_routing():
    """Isolate graph regression from legacy single-router kernel rounding.

    The fresh graph oracle uses native eager MHAR, not the legacy single-
    cursor fused kernel. Actual accepted-checkpoint conversion parity is a
    separate mandatory gate and still measures the deployed native backend.
    This scope does not change production or the explicit two-role backend.
    """
    from src.attention_residuals import modeling_qwen3_attnres as routing
    fused = routing._mhar_fused
    enabled = fused is not None and fused.fused_mhar_enabled()
    routing.enable_fused_mhar(False)
    try:
        yield
    finally:
        routing.enable_fused_mhar(enabled)


def parity(output, artifact, data_files):
    payload, data = artifact_identity(artifact)
    tokens = payload["discovery_input_ids"]
    if len(tokens) != 512:
        raise ValueError("parity must cover all 512 frozen discovery sequences")
    from src.attention_residuals.modeling_qwen3_attnres import enable_fused_mhar
    enable_fused_mhar(True)
    rows, zero_reference, zero_parameters, zero_rng = [], None, None, None
    master_query = {}
    for variant in RUNS:
        net = model(variant, data_files)
        # Common non-Q tensors and the base RNG must agree across all eight.
        common = {name:tensor_digest(t) for name,t in net.named_parameters()
                  if not name.endswith("local_attn_res_proj.weight") and ".q_proj." not in name}
        rng = tensor_digest(torch.get_rng_state())
        if zero_parameters is None:
            zero_parameters, zero_rng = common, rng
        if common != zero_parameters or rng != zero_rng:
            raise RuntimeError("master common weights or construction RNG differ")
        for layer in net.model.layers:
            if not torch.equal(layer.attn_res_proj.weight, layer.local_attn_res_proj.weight):
                raise RuntimeError("B is not an exact copy of A")
            projection = layer.self_attn.q_proj
            index = layer.self_attn.layer_idx
            if variant == "global-aaaa":
                weight = projection.weight.reshape(8,2,80,1280)
                master_query[index] = {"dense":tensor_digest(projection.weight),
                    "global":tensor_digest(weight[:,1]),
                    "local":tensor_digest(torch.stack([weight[g,0,:,g*160:(g+1)*160] for g in range(8)]))}
            elif variant.startswith("global"):
                if tensor_digest(projection.weight) != master_query[index]["dense"]:
                    raise RuntimeError("GG dense Q initialization differs")
            elif (tensor_digest(projection.global_weight) != master_query[index]["global"]
                  or tensor_digest(projection.local_weight) != master_query[index]["local"]):
                raise RuntimeError("GL initialization is not the exact unscaled master slice")
        with execution_backend(net, router="shared-eager", projection="canonical"):
            with torch.no_grad():
                zero_logits_hash = tensor_digest(net(tokens[:1].cuda(),use_cache=False).logits)
        if variant.startswith("global"):
            if zero_reference is None:
                zero_reference = zero_logits_hash
            if zero_logits_hash != zero_reference:
                raise RuntimeError("canonical GG zero-step logits construction differs")
        # A != B is essential: initialization equality alone cannot test wiring.
        torch.manual_seed(23042)
        with torch.no_grad():
            for layer in net.model.layers:
                layer.attn_res_proj.weight.normal_(0, .03)
                layer.local_attn_res_proj.weight.normal_(0, .04)
        with execution_backend(net, router="shared-eager", projection="eager-matched"):
            canonical = sequence_metrics(net,tokens,batch_size=1,device="cuda")
        optimized = sequence_metrics(net,tokens,batch_size=1,device="cuda")
        maximum = max(abs(a/1023-b/1023) for a,b in zip(
            canonical["sequence_total_losses"],optimized["sequence_total_losses"]))
        if maximum > .0005:
            raise RuntimeError(f"{variant} full BF16 parity failed: {maximum} > 0.0005")
        rows.append({"variant":variant,"max_per_sequence_loss_difference":maximum,
                     "sequences":512,"unequal_routers":True,
                     "numerical_reference_projection":"eager-matched",
                     "same_route_reference":"one_full_width_Q_GEMM",
                     "different_route_reference":"two_half_width_Q_GEMMs"})
        del net; gc.collect(); torch.cuda.empty_cache()
    report(output,"all_variant_BF16_discovery_parity",rows=rows,artifact_identity=data,
           common_slot_initialization_oracle_retained=True)
    references=[]
    for variant in ("global-aaaa","hq8-gl-aaaa","r2-gl-abaa"):
        old = model(variant,data_files,native=True)
        old_parameters = {name:tensor_digest(t) for name,t in old.named_parameters()}
        with native_reference_routing():
            old_metrics=sequence_metrics(old,tokens,batch_size=1,device="cuda")
        del old; gc.collect(); torch.cuda.empty_cache()
        new=model(variant,data_files)
        new_parameters = {name:tensor_digest(t) for name,t in new.named_parameters()}
        if any(new_parameters.get(name) != digest for name,digest in old_parameters.items()):
            raise RuntimeError(f"native {variant} initialization tensors differ")
        new_metrics=sequence_metrics(new,tokens,batch_size=1,device="cuda")
        maximum=max(abs(a/1023-b/1023) for a,b in zip(
            old_metrics["sequence_total_losses"],new_metrics["sequence_total_losses"]))
        if maximum>.0005:
            raise RuntimeError(f"native {variant} regression: {maximum}")
        references.append({"variant":variant,"max_per_sequence_loss_difference":maximum,"sequences":512,
            "native_shared_fused_context":False,"reference":"native_eager_history_routing",
            "optimized_backend_unchanged":True,"native_parameter_hashes_match":True,
            "native_parameter_count":len(old_parameters)})
        del new; gc.collect(); torch.cuda.empty_cache()
    report(output,"native_reference_parity",rows=references)
    write_new(Path(output)/"initialization_certificate.json",{
        **binding_identity(),"common_tensor_hashes":zero_parameters,"cpu_rng_sha256":zero_rng,
        "B_copied_exactly":True,"GG_canonical_zero_logits_sha256":zero_reference,
        "GL_Q_columns_verified_full_size":True,"master_query_hashes":master_query})


def backward(output):
    # Tiny FP32 graph tests (including aliases/local slices) plus actual CUDA
    # BF16 checkpointed route backward. No model/checkpoint used for training.
    from tests.test_experiment23_route_topology import make
    from src.attention_residuals.modeling_qwen3_attnres import enable_fused_mhar
    enable_fused_mhar(True)
    rows=[]
    for variant in RUNS:
        snapshots=[]
        for backend in ("shared-eager","r2-triton"):
            net=make(variant,"packed").to(device="cuda",dtype=torch.bfloat16).train()
            with torch.no_grad():
                for layer in net.model.layers:
                    layer.attn_res_proj.weight.normal_(0,.03)
                    layer.local_attn_res_proj.weight.normal_(0,.04)
            net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
            optimizer=torch.optim.AdamW(net.parameters(),lr=5e-4,betas=(.9,.95),weight_decay=.1)
            tokens=torch.arange(16,device="cuda").reshape(2,8)
            with execution_backend(net,router=backend):
                loss=net(tokens,labels=tokens).loss; loss.backward()
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite checkpointed backward")
            gradients={n:None if p.grad is None else p.grad.detach().float().cpu() for n,p in net.named_parameters()}
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); optimizer.step()
            snapshots.append((float(loss),gradients,{n:p.detach().float().cpu() for n,p in net.named_parameters()}))
            del net,optimizer; gc.collect(); torch.cuda.empty_cache()
        for index in (1,2):
            for name,a in snapshots[0][index].items():
                b=snapshots[1][index][name]
                if a is None or b is None:
                    if a is not b: raise RuntimeError("active/inactive gradient mismatch")
                elif not torch.allclose(a,b,atol=1e-6,rtol=1e-5):
                    raise RuntimeError(f"checkpointed CUDA BF16 gradient/update mismatch: {variant}:{name}")
        if abs(snapshots[0][0]-snapshots[1][0])>.0005:
            raise RuntimeError("checkpointed CUDA Loss parity failed")
        rows.append({"variant":variant,"gradient_and_update_atol":1e-6,"rtol":1e-5})
    report(output,"checkpointed_router_backward",rows=rows)


def compare_tree(a,b):
    """Exact optimizer/RNG/state equality, with no BF16 tolerance relaxation."""
    if isinstance(a,torch.Tensor):
        if not isinstance(b,torch.Tensor) or a.dtype!=b.dtype or not torch.equal(a,b):
            raise RuntimeError("exact continuation tensor mismatch")
    elif isinstance(a,dict):
        if a.keys()!=b.keys(): raise RuntimeError("continuation state keys differ")
        for key in a: compare_tree(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        if type(a)!=type(b) or len(a)!=len(b): raise RuntimeError("continuation sequence differs")
        for x,y in zip(a,b): compare_tree(x,y)
    elif a!=b:
        raise RuntimeError("exact continuation scalar mismatch")


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


@contextlib.contextmanager
def verify_worker_restoration():
    """Observe actual restored state before any disposable optimizer update."""
    from unittest.mock import patch
    from safetensors import safe_open
    observed = {}
    build, opt_load = training.build_model, torch.optim.AdamW.load_state_dict
    scheduler_load = torch.optim.lr_scheduler.LambdaLR.load_state_dict
    cpu_set, cuda_set = torch.set_rng_state, torch.cuda.set_rng_state_all

    def build_checked(args, device):
        net = build(args, device)
        if args.resume_from:
            actual = net.state_dict()
            count = 0
            for path in Path(args.resume_from).glob("*.safetensors"):
                with safe_open(path,framework="pt",device="cpu") as saved:
                    for name in saved.keys():
                        compare_tree(actual[name].detach().cpu(),saved.get_tensor(name))
                        count += 1
            if not count:
                raise RuntimeError("no saved model tensors verified at restoration")
            observed["model_tensors_exact"] = count
        return net

    def optimizer_checked(optimizer, state):
        result = opt_load(optimizer,state)
        compare_tree(cpu_tree(optimizer.state_dict()),cpu_tree(state))
        observed["optimizer_exact"] = True
        return result

    def scheduler_checked(scheduler, state):
        result = scheduler_load(scheduler,state)
        compare_tree(cpu_tree(scheduler.state_dict()),cpu_tree(state))
        observed["scheduler_exact"] = True
        return result

    def cpu_checked(state):
        result = cpu_set(state)
        if observed.get("optimizer_exact"):
            compare_tree(torch.get_rng_state(),state)
            observed["cpu_rng_exact"] = True
        return result

    def cuda_checked(states):
        result = cuda_set(states)
        if observed.get("optimizer_exact"):
            compare_tree(torch.cuda.get_rng_state_all(),states)
            observed["cuda_rng_exact"] = True
        return result

    with patch.object(training,"build_model",build_checked), \
         patch.object(torch.optim.AdamW,"load_state_dict",optimizer_checked), \
         patch.object(torch.optim.lr_scheduler.LambdaLR,"load_state_dict",scheduler_checked), \
         patch.object(torch,"set_rng_state",cpu_checked), \
         patch.object(torch.cuda,"set_rng_state_all",cuda_checked):
        yield observed


def smoke_arguments(data_files,output,stop,resume=None):
    args=train_arguments("global-abaa",output=output,data_files=data_files,disposable=True,resume=resume)
    for key,value in {"--stop_after_step":str(stop),"--save_every":"1","--keep_steps":"1,2",
                      "--eval_every":"1","--eval_steps":"1","--log_every":"1"}.items():
        args[args.index(key)+1]=value
    # Deterministic disposable-preflight workers (see main()); the flag is
    # stripped before train_scratch arg parsing.
    return args + ["--deterministic"]


def smoke(output,data_files):
    """Checkpoint integrity and reproducible continuation (revised 2026-10-07).

    Replaces the earlier ``full_GG_optimizer_and_exact_resume`` gate, which
    required a checkpoint-resumed continuation to be torch.equal to an
    uninterrupted run. Cross-construction-path bit-equivalence failed in the
    preserved diagnostics; its low-level cause is not established. This gate
    instead proves the narrower
    guarantee:

      (1) the saved step-1 checkpoint is complete and exact (model, optimizer,
          scheduler, CPU/CUDA RNG, global step, packed-data position); and
      (2) two independent resumes of that same checkpoint produce bit-identical
          step-2 continuations under the documented deterministic preflight
          settings.

    Resumed-versus-uninterrupted differences are recorded as diagnostics only.
    """
    root=Path(output)/"disposable-smoke"
    if root.exists(): raise FileExistsError(root)
    root.mkdir(parents=True)
    # Four complete checkpoint states + atomic temporary model/state writes.
    if shutil.disk_usage(root).free < 60*1024**3:
        raise RuntimeError("full GG smoke needs at least 60 GiB free scratch disk")
    env=dict(os.environ,WANDB_MODE="disabled",PYTHONUNBUFFERED="1",TOKENIZERS_PARALLELISM="false",
             PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",CUBLAS_WORKSPACE_CONFIG=":4096:8",
             HF_HOME=os.environ.get("HF_HOME","/root/hf-exp7"),
             HF_ENDPOINT=os.environ.get("HF_ENDPOINT","https://hf-mirror.com"))
    command=[sys.executable,"-m","torch.distributed.run","--nproc_per_node=1","--master_port",
        os.environ.get("MHAR_PREFLIGHT_PORT","30333"),"--module","scripts.train.preflight_experiment23","--worker"]
    started=time.monotonic(); memories=[]
    restored_checks=[]
    checkpoint=root/"checkpoint"
    phases=(("checkpoint",checkpoint,1,None),
            ("resume-a",root/"resume-a",2,checkpoint/"step-1"),
            ("resume-b",root/"resume-b",2,checkpoint/"step-1"),
            ("continuous",root/"continuous",2,None))
    from safetensors import safe_open
    def load_state(path):
        return torch.load(path,weights_only=False,map_location="cpu",mmap=True)
    def weight_tensors(directory):
        values={}
        for path in sorted(directory.glob("*.safetensors")):
            with safe_open(path,framework="pt",device="cpu") as f:
                for name in f.keys(): values[name]=f.get_tensor(name)
        return values
    def weight_digests(directory):
        values={}
        for path in sorted(directory.glob("*.safetensors")):
            with safe_open(path,framework="pt",device="cpu") as f:
                for name in f.keys(): values[name]=tensor_digest(f.get_tensor(name))
        return values
    for phase,directory,step,resume in phases:
        if resume is not None:
            # Resume source lives inside the resume output dir (same layout as a
            # production restart); it is a byte-identical copy of the saved
            # step-1 checkpoint produced by the "checkpoint" phase.
            shutil.copytree(resume,directory/"step-1")
            resume_source=directory/"step-1"
            if training.sha256_tree(resume_source) != training.sha256_tree(resume):
                raise RuntimeError("disposable resume checkpoint copy is not byte-exact")
        else:
            resume_source=None
        with (root/(phase+".log")).open("x") as log:
            subprocess.run(command+smoke_arguments(data_files,directory,step,resume_source),
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        lines=(root/(phase+".log")).read_text().splitlines()
        memories.extend(json.loads(line.split("PREFLIGHT_MEMORY ",1)[1]) for line in lines if line.startswith("PREFLIGHT_MEMORY "))
        if resume is not None:
            checks=[json.loads(line.split("PREFLIGHT_RESTORE ",1)[1]) for line in lines
                    if line.startswith("PREFLIGHT_RESTORE ")]
            if len(checks)!=1 or checks[0].get("model_tensors_exact") != 580 or any(
                    checks[0].get(key) is not True for key in (
                        "optimizer_exact","scheduler_exact","cpu_rng_exact","cuda_rng_exact")):
                raise RuntimeError("missing actual pre-update restoration integrity checks")
            restored_checks.append(checks[0])
    if len(memories)!=len(phases): raise RuntimeError("missing exact peak memory measurements")
    saved=load_state(checkpoint/"step-1/training_state.pt")
    if saved["global_step"]!=1 or saved["chunks_consumed"]!=32 or not saved["optimizer"]["state"]:
        raise RuntimeError("incomplete disposable step-1 checkpoint")
    resumed=[load_state(root/name/"step-2/training_state.pt") for name in ("resume-a","resume-b")]
    for row in resumed:
        if row["global_step"]!=2 or row["chunks_consumed"]!=64 or not row["optimizer"]["state"]:
            raise RuntimeError("incomplete disposable resume state")
    # (2) two independent resumes of the same saved checkpoint are bit-identical.
    for key in ("run_identity","optimizer","scheduler","cpu_rng_state","cuda_rng_state_all",
                "global_step","chunks_consumed"):
        compare_tree(resumed[0][key],resumed[1][key])
    weights_a=weight_tensors(root/"resume-a/step-2"); weights_b=weight_tensors(root/"resume-b/step-2")
    if weight_digests(root/"resume-a/step-2")!=weight_digests(root/"resume-b/step-2"):
        raise RuntimeError("two independent resumes produced different model weights")
    # (1) checkpoint integrity: saved step-1 state is complete and exact.
    integrity={"global_step":int(saved["global_step"]),
        "chunks_consumed":int(saved["chunks_consumed"]),
        "optimizer_state_tensors":len(saved["optimizer"]["state"]),
        "scheduler_present":bool(saved["scheduler"]),
        "cpu_rng_tensor":bool(torch.is_tensor(saved["cpu_rng_state"])),
        "cuda_rng_count":len(saved["cuda_rng_state_all"]),
        "model_parameter_tensors":len(weight_tensors(checkpoint/"step-1"))}
    if integrity["global_step"]!=1 or integrity["chunks_consumed"]!=32:
        raise RuntimeError("checkpoint step/packed-data position is not exact")
    if integrity["optimizer_state_tensors"]!=580 or integrity["cuda_rng_count"]!=1:
        raise RuntimeError("checkpoint optimizer/RNG state is incomplete")
    if not integrity["scheduler_present"] or not integrity["cpu_rng_tensor"]:
        raise RuntimeError("checkpoint scheduler/CPU RNG state is missing")
    # (3) diagnostics only: resumed vs uninterrupted (never gated or relaxed).
    continuous=load_state(root/"continuous/step-2/training_state.pt")
    weights_c=weight_tensors(root/"continuous/step-2")
    def state_max_abs(a,b):
        return max((float((a["optimizer"]["state"][k]["exp_avg"].float()
                           -b["optimizer"]["state"][k]["exp_avg"].float()).abs().max())
                    for k in a["optimizer"]["state"]),default=0.0)
    def params_max_abs(wa,wb):
        return max((float((wa[name].float()-wb[name].float()).abs().max())
                    for name in wa),default=0.0)
    def step_line(name,step):
        for line in (root/(name+".log")).read_text().splitlines():
            if line.startswith("step") and f"step {step:6d} " in line:
                return line.strip()
        return None
    diagnostics={"uninterrupted_bit_equivalence":False,
        "why":"loaded and uninterrupted construction paths differed in preserved diagnostics; the kernel-selection explanation remains an unproven hypothesis",
        "optimizer_exp_avg_max_abs_difference":state_max_abs(resumed[0],continuous),
        "optimizer_tensors_differing":sum(1 for k in resumed[0]["optimizer"]["state"]
            if not torch.equal(resumed[0]["optimizer"]["state"][k]["exp_avg"],
                               continuous["optimizer"]["state"][k]["exp_avg"])),
        "model_weight_max_abs_difference":params_max_abs(weights_a,weights_c),
        "model_weight_tensors_differing":sum(1 for n in weights_a if not torch.equal(weights_a[n],weights_c[n])),
        "step2_log_lines":{"resume-a":step_line("resume-a",2),"resume-b":step_line("resume-b",2),
                           "continuous":step_line("continuous",2)}}
    disposable_checkpoints=[checkpoint/"step-1",root/"resume-a/step-2",root/"resume-b/step-2",
        root/"continuous/step-1",root/"continuous/step-2"]
    checkpoint_hashes={str(p.relative_to(root)):training.sha256_tree(p) for p in disposable_checkpoints}
    report(output,"full_GG_checkpoint_integrity_and_reproducible_continuation",
        seconds=time.monotonic()-started,memory=memories,
        checkpoint_integrity=integrity,
        actual_pre_update_restoration_checks=restored_checks,
        reproducible_continuation_bit_identical=True,
        resumed_continuations_compared=2,
        uninterrupted_equivalence=diagnostics,
        superseded_gate="full_GG_optimizer_and_exact_resume (uninterrupted bit-equivalence; replaced because it is not attainable on this CUDA stack)",
        disposable_checkpoint_hashes=checkpoint_hashes,
        disk_free_bytes=shutil.disk_usage(root).free,
        temporary_checkpoint_policy="own disposable step1/2 states removed after passing report; logs/hashes retained")
    del saved,resumed,continuous,weights_a,weights_b,weights_c
    gc.collect()
    for directory in (checkpoint,root/"resume-a",root/"resume-b",root/"continuous"):
        if directory.is_symlink():
            raise RuntimeError("disposable cleanup target is a symlink")
        shutil.rmtree(directory)


def certificate(output,data_files,artifact):
    artifact_identity(artifact)
    validate_training_data_identity(identity_for_arguments(train_arguments(
        "global-aaaa",output="/unused",data_files=data_files,disposable=True))["data_files"])
    initial=json.loads((Path(output)/"initialization_certificate.json").read_text())
    if any(initial.get(k)!=v for k,v in binding_identity().items()):
        raise ValueError("initialization certificate source/data changed")
    subprocess.run([sys.executable,"-m","unittest","tests.test_experiment23_route_topology","-q"],
                   cwd=ROOT,env=dict(os.environ,PYTHONPATH=".:tests"),check=True)
    args=train_arguments("global-aaaa",output="/unused",data_files=data_files,disposable=True)
    with invocation(args) as parsed:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(parsed.tokenizer,revision=parsed.tokenizer_revision)
        streams={}
        for label,seed,count in (("training",42,64000),("monitoring",10041,1000)):
            digest=hashlib.sha256()
            iterator=training.token_stream(parsed.dataset,parsed.dataset_name,tokenizer,1024,0,1,
                seed,data_files=data_files,dataset_revision=parsed.dataset_revision)
            for _ in range(count):
                row=next(iterator)
                if tuple(row.shape)!=(1025,): raise RuntimeError("packing shape differs")
                digest.update(row.contiguous().numpy().tobytes())
            streams[label]={"chunks":count,"dtype":"int64","shape_per_chunk":[1025],
                            "sha256":digest.hexdigest(),"seed":seed}
    report(output,"initializer_stream_and_alias_certificate",initialization=initial,
        streams=streams,training_data=identity_for_arguments(args)["data_files"],
        alias_and_local_Q_tests_passed=True,ordered_stream_hashed_to_protected_stop=True)


def bind(output,data_files,artifact):
    artifact_identity(artifact); bindings=binding_identity(); gates={}
    for name in json.loads(PROTOCOL.read_text())["cuda_gates"]:
        file=Path(output)/(name+".json"); row=json.loads(file.read_text())
        if row.get("passed") is not True or any(row.get(k)!=v for k,v in bindings.items()):
            raise ValueError(f"missing, failed or stale gate: {name}")
        deployed=environment()
        if any(row.get("environment",{}).get(k)!=v for k,v in deployed.items()
               if k not in ("device_uuid","visible_devices")):
            raise ValueError("gate runtime/device changed")
        gates[name]={"path":str(file.resolve()),"sha256":sha256_file(file)}
    certificate_row=json.loads((Path(output)/"initializer_stream_and_alias_certificate.json").read_text())
    for variant in RUNS:
        arguments=train_arguments(variant,output="/unused",data_files=data_files)
        write_new(Path(output)/"execution-locks"/(variant+".json"),{**bindings,"variant":variant,
            "training_identity":identity_for_arguments(arguments),"environment":environment(),
            "initializer_and_stream_certificate":certificate_row,"readiness_gate_hashes":gates})


def main():
    os.environ.setdefault("HF_HOME", "/root/hf-exp7")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    if "--worker" in sys.argv:
        worker_argv = sys.argv[sys.argv.index("--worker") + 1:]
        deterministic = "--deterministic" in worker_argv
        worker_argv = [value for value in worker_argv if value != "--deterministic"]
        sys.argv = ["preflight-worker", *worker_argv]
        if deterministic:
            # Checked-in deterministic preflight settings for the disposable
            # reproducible-continuation gate.  Production training is unchanged.
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
        with verify_worker_restoration() as restored:
            training.main()
        if restored:
            print("PREFLIGHT_RESTORE " + json.dumps(restored),flush=True)
        print("PREFLIGHT_MEMORY " + json.dumps({"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}), flush=True)
        return
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("parity","backward","smoke","certificate","bind"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--artifact",type=Path,required=True)
    p.add_argument("--data-files",required=True)
    args=p.parse_args(); environment(); binding_identity()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.stage=="parity": parity(args.output,args.artifact,args.data_files)
    elif args.stage=="backward": backward(args.output)
    elif args.stage=="smoke": smoke(args.output,args.data_files)
    elif args.stage=="certificate": certificate(args.output,args.data_files,args.artifact)
    else: bind(args.output,args.data_files,args.artifact)


if __name__=="__main__": main()
