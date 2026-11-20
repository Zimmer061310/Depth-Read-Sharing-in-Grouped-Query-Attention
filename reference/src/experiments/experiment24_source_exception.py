"""Explicit postprocessing-only source exceptions, checked against Git ASTs.

This cannot authorize a model/trainer/protocol/data/threshold change. It is not
an automatic retry permission; create only for a diagnosed bookkeeping repair.
"""
import ast
import subprocess

from src.experiments.experiment24_workflow import ROOT, write_new

ALLOWED = {
    "src/experiments/experiment24_evaluation.py",
    "src/experiments/experiment24_compute.py",
    "src/experiments/experiment24_probe.py",
    "src/experiments/experiment24_publication.py",
    "src/experiments/experiment24_source_exception.py",
    "scripts/evaluate/run_experiment24_evaluation.py",
    "scripts/evaluate/run_experiment24_postprocess.py",
    "figures/gen_fig_experiment24.py",
    "tests/test_experiment24_workflow.py",
    "docs/runbooks/experiment24-three-route.md",
}
PROTECTED = {
    "src/experiments/experiment24_evaluation.py": ("evaluate_condition", "apply_condition"),
    "src/experiments/experiment24_compute.py": ("analytic_ledger", "timing_block", "iteration"),
    "src/experiments/experiment24_probe.py": ("route_metrics", "content_metrics", "observe", "probe_split"),
}


def git(*args):
    return subprocess.check_output(["git",*args],cwd=ROOT)


def source_diff(original, corrected):
    import hashlib
    for s in (original,corrected):
        if len(s)!=40 or set(s)-set("0123456789abcdef"):
            raise ValueError("invalid source commit")
    subprocess.run(["git","merge-base","--is-ancestor",original,corrected],cwd=ROOT,check=True)
    paths = git("diff","--no-renames","--name-only",original,corrected).decode().splitlines()
    if not paths or set(paths)-ALLOWED:
        raise ValueError("source exception changes scientific/out-of-scope files")
    for file,names in PROTECTED.items():
        def functions(commit):
            tree = ast.parse(git("show",f"{commit}:{file}"))
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(tree)
                    if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names}
        a,b = functions(original),functions(corrected)
        if set(a)!=set(names) or a!=b:
            raise ValueError("measurement arithmetic changed")
    return {"changed_paths":sorted(paths),"diff_sha256":hashlib.sha256(
        git("diff","--no-renames","--binary",original,corrected)).hexdigest()}


def create_exception(original, corrected, reason, output):
    from src.experiments.experiment24_evaluation import digest
    if not reason.strip():
        raise ValueError("diagnosis required")
    body = {"training_source":original,"evaluation_source":corrected,"reason":reason,
            "scientific_changes":False,**source_diff(original,corrected)}
    value = {**body,"content_sha256":digest(body)}
    write_new(output,value)
    return value


def verify_exception(value, original, corrected):
    from src.experiments.experiment24_evaluation import digest
    if not isinstance(value,dict):
        raise ValueError("documented source exception required")
    body = {k:v for k,v in value.items() if k!="content_sha256"}
    if (value.get("content_sha256")!=digest(body) or body.get("training_source")!=original
        or body.get("evaluation_source")!=corrected or body.get("scientific_changes") is not False
        or not body.get("reason") or any(body.get(k)!=v for k,v in source_diff(original,corrected).items())):
        raise ValueError("invalid/unprotected source exception")
