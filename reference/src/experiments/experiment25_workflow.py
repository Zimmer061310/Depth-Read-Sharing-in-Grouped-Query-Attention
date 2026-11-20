"""Frozen recipe, immutable evidence and mandatory launch fences for Exp25."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from src.experiments.experiment1_partition_compatibility import sha256_file
from src.experiments.experiment21_evaluation import require_clean_commit
from src.experiments.experiment23_workflow import identity_for_arguments, validate_training_data_identity
from src.experiments.experiment25_global_local import VARIANT, RUN_ID
from src.experiments.experiment25_budget import require_work_allowed

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT/"configs/experiment25/protocol.json"
MATRIX = ROOT/"configs/experiment25/run_matrix.json"
LOCK = ROOT/"results/experiment24/provenance/data_lock.json"


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    """Exclusive creation: accepted evidence is never replaced under recovery."""
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode="w",encoding="utf-8",dir=path.parent,
                                         prefix=".pending-",delete=False) as handle:
            temporary=Path(handle.name)
            json.dump(value,handle,sort_keys=True,indent=2,allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary,path)  # Atomic no-clobber publication on this filesystem.
        descriptor=os.open(path.parent,os.O_RDONLY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def copy_identical(source,target):
    source,target=Path(source),Path(target)
    if target.exists():
        if target.read_bytes()!=source.read_bytes():raise ValueError("accepted copied bytes changed")
        return
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode="wb",dir=target.parent,prefix=".pending-",delete=False) as handle:
            temporary=Path(handle.name);handle.write(source.read_bytes());handle.flush();os.fsync(handle.fileno())
        os.link(temporary,target)
        descriptor=os.open(target.parent,os.O_RDONLY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


def train_arguments(*, output, data_files, resume=None):
    from src.experiments.experiment24_workflow import train_arguments as inherited
    args = inherited("global-abbc",output=output,data_files=data_files,resume=resume)
    for flag in ("--experiment24_variant","--experiment24_router_backend"):
        i=args.index(flag); del args[i:i+2]
    args += ["--experiment25_variant",VARIANT,"--experiment25_router_backend","tiled-r2-exact"]
    for flag,value in (("--run_name",RUN_ID),("--wandb_project","MHAR stuff"),
                       ("--wandb_group",read(PROTOCOL)["publication"]["wandb_group"])):
        args[args.index(flag)+1]=value
    return args


def binding_identity():
    p = read(PROTOCOL)
    if sha256_file(LOCK) != p["evaluation"]["data_lock_sha256"]:
        raise ValueError("frozen data lock changed")
    return {"source_commit":require_clean_commit(),"protocol_sha256":sha256_file(PROTOCOL),
            "matrix_sha256":sha256_file(MATRIX),"data_lock_sha256":sha256_file(LOCK),
            "artifact_sha256":p["evaluation"]["artifact_sha256"]}


def launch_environment():
    # Device-property queries can create a CUDA context. Probe in a process
    # that exits before launching the worker, otherwise the waiting launcher
    # itself violates the otherwise-idle GPU check in the child trainer.
    command=[sys.executable,"-c","import json; from src.experiments.experiment25_compute import environment; print(json.dumps(environment()))"]
    return json.loads(subprocess.check_output(command,cwd=ROOT,text=True))


def verify_execution_lock(path, arguments):
    lock, binding, protocol = read(path), binding_identity(), read(PROTOCOL)
    if any(lock.get(k)!=v for k,v in binding.items()):
        raise ValueError("stale source/protocol/matrix/data binding")
    if lock.get("training_identity") != identity_for_arguments(arguments):
        raise ValueError("changed recipe or ordered stream")
    validate_training_data_identity(lock["training_identity"]["data_files"])
    required = protocol["readiness"]["local_gates"]+protocol["readiness"]["cuda_gates"]
    gates = lock.get("gates",{})
    if set(gates)!=set(required): raise ValueError("all local and CUDA gate receipts required")
    for name,row in gates.items():
        report=read(row["path"])
        if (sha256_file(row["path"])!=row["sha256"] or report.get("gate")!=name
            or report.get("passed") is not True or any(report.get(k)!=v for k,v in binding.items())):
            raise ValueError(f"failed, modified or unbound readiness gate: {name}")
    actual=launch_environment()
    if actual != lock.get("environment"):
        raise ValueError("runtime/GPU changed since preflight")
    for name in protocol["readiness"]["cuda_gates"]:
        if read(gates[name]["path"]).get("environment")!=actual:
            raise ValueError(f"CUDA gate runtime/GPU differs: {name}")
    expected=protocol["readiness"]["dependency_reference"]
    if any(actual.get(k)!=v for k,v in expected.items()):
        raise ValueError("runtime differs from frozen dependency reference")
    if lock.get("separate_paid_execution_authorized") is not True:
        raise ValueError("implementation is not paid execution authorization")
    return lock


def verify_trainer_invocation(args):
    if not all((args.experiment25_execution_lock,args.experiment25_budget_lock,args.experiment25_budget_heartbeat)):
        raise ValueError("Experiment 25 requires execution and independent budget locks")
    arguments=train_arguments(output=args.out_dir,data_files=args.data_files,resume=args.resume_from)
    from src.training.train_scratch import training_identity
    if training_identity(args,1)!=identity_for_arguments(arguments):
        raise ValueError("trainer invocation differs from frozen scientific recipe")
    # Operational flags are not all in the scientific identity. Explicitly fence
    # saving, stop, validation, precision/backend and W&B behavior as well.
    saved=sys.argv
    try:
        from src.training.train_scratch import parse_args
        sys.argv=["exp25",*arguments]; expected=vars(parse_args())
    finally: sys.argv=saved
    allowed={"experiment25_execution_lock","experiment25_budget_lock","experiment25_budget_heartbeat"}
    if any(vars(args).get(k)!=v for k,v in expected.items() if k not in allowed):
        raise ValueError("trainer operational flags differ from frozen recipe")
    execution=verify_execution_lock(args.experiment25_execution_lock,arguments)
    if execution.get("budget_lock_sha256")!=sha256_file(args.experiment25_budget_lock):
        raise ValueError("budget lock not bound to execution")
    require_work_allowed(args.experiment25_budget_lock,args.experiment25_budget_heartbeat)


def verify_resources(output):
    directory=Path(output).resolve().parent
    while not directory.exists(): directory=directory.parent
    if shutil.disk_usage(directory).free < 85*1024**3:
        raise RuntimeError("85 GiB free after inputs required")
    if not os.environ.get("CUDA_VISIBLE_DEVICES","").isdigit():
        raise RuntimeError("exactly one selected CUDA card required")
    rows=subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total,memory.used",
        "--format=csv,noheader,nounits"],text=True).strip().splitlines()
    if len(rows)!=1 or "5090" not in rows[0]:
        raise RuntimeError("one-card 5090 allocation required; do not rent two billed cards")
    _,total,used=rows[0].rsplit(",",2)
    if int(total)<30000 or int(used)>1024: raise RuntimeError("wrong VRAM or occupied GPU")


def require_disk_floor(output):
    directory=Path(output).resolve()
    while not directory.exists():directory=directory.parent
    if shutil.disk_usage(directory).free < read(PROTOCOL)["readiness"]["disk_free_floor_gib"]*1024**3:
        raise RuntimeError("Experiment 25 disk reserve below20 GiB; preserve atomic state and stop work")


def launch(*, output, data_files, execution_lock, budget_lock, budget_heartbeat, resume=None):
    arguments=train_arguments(output=output,data_files=data_files,resume=resume)
    verify_execution_lock(execution_lock,arguments)
    require_work_allowed(budget_lock,budget_heartbeat)
    verify_resources(output)
    if resume is None and Path(output).exists() and any(Path(output).iterdir()):
        raise FileExistsError("fresh run output must be empty; no automatic restart")
    if resume is not None and Path(resume).resolve().parent!=Path(output).resolve():
        raise ValueError("resume must be this run's own exact atomic checkpoint")
    arguments += ["--experiment25_execution_lock",str(execution_lock),
                  "--experiment25_budget_lock",str(budget_lock),
                  "--experiment25_budget_heartbeat",str(budget_heartbeat)]
    env=dict(os.environ,WANDB_MODE="online",PYTHONUNBUFFERED="1",HF_HUB_OFFLINE="1")
    if not env.get("HF_HOME") or not Path(env["HF_HOME"]).is_dir():
        raise ValueError("verified offline cache required")
    subprocess.run([sys.executable,"-m","torch.distributed.run","--nproc_per_node=1",
        "--master_port",env.get("MHAR_MASTER_PORT","30525"),"--module",
        "src.training.train_scratch",*arguments],cwd=ROOT,env=env,check=True)
