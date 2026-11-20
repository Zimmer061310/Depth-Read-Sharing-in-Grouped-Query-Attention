"""Narrow metadata-only postprocessing exception; never authorizes training.

An immutable runtime manifest binds exact old/new commits and the Git diff.
It is copied into publication provenance, not substituted for training source.
Model, trainer, protocol, data, and measurement arithmetic changes are forbidden.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess

from src.experiments.experiment23_workflow import ROOT, PROTOCOL, MATRIX, binding_identity, write_new
from src.experiments.experiment21_evaluation import require_clean_commit, read_json
from src.experiments.experiment1_partition_compatibility import sha256_file

EXECUTION_SOURCE = "239ba5981c043d5ea1643996efec24035e8cd298"
REASON = "experiment23-paired-recipe-provenance-bookkeeping-recovery"
ALLOWED_PATHS = frozenset({
    "src/experiments/experiment23_evaluation.py",
    "src/experiments/experiment23_reuse.py",
    "src/experiments/experiment23_source_exception.py",
    "src/experiments/experiment23_compute.py",
    "scripts/evaluate/run_experiment23_postprocess.py",
    "scripts/evaluate/run_experiment23_benchmark_recovery.py",
    "tests/test_experiment23_recovery.py",
    "docs/runbooks/experiment23-route-topology.md",
})
PROTECTED_FUNCTIONS = {
    "src/experiments/experiment23_evaluation.py": ("sequence_metrics", "evaluate_condition", "bootstrap_contrast"),
    "src/experiments/experiment23_reuse.py": ("native_class", "adapt_control", "load_model", "certify_control"),
    "src/experiments/experiment23_compute.py": ("analytic_ledger", "timing_block", "matched_timing"),
}


def protected_function_ast(node):
    """Allow only the known label-index dtype correction, outside timing.

    Everything else in matched_timing, and all of timing_block/analytic_ledger,
    must remain AST-identical to the accepted execution source.
    """
    node = copy.deepcopy(node)
    if node.name == "matched_timing":
        assignments = [item for item in node.body if isinstance(item,ast.Assign)
                       and len(item.targets) == 1 and isinstance(item.targets[0],ast.Name)
                       and item.targets[0].id == "tokens"]
        before = ast.parse('payload["discovery_input_ids"][:4].to(device)',mode="eval").body
        after = ast.parse('payload["discovery_input_ids"][:4].to(device=device, dtype=torch.long)',mode="eval").body
        allowed = (ast.dump(before,include_attributes=False),ast.dump(after,include_attributes=False))
        if len(assignments) != 1 or ast.dump(assignments[0].value,include_attributes=False) not in allowed:
            raise ValueError("benchmark token preparation differs beyond the long-index correction")
        assignments[0].value = before
    return ast.dump(node,include_attributes=False)


def git(*arguments):
    return subprocess.check_output(["git",*arguments],cwd=ROOT)


def source_diff(original, corrected):
    if original != EXECUTION_SOURCE or len(corrected) != 40 or any(c not in "0123456789abcdef" for c in corrected):
        raise ValueError("unapproved source pair")
    subprocess.run(["git","merge-base","--is-ancestor",original,corrected],cwd=ROOT,check=True)
    paths = git("diff","--no-renames","--name-only",original,corrected).decode().splitlines()
    if not paths or set(paths) - ALLOWED_PATHS:
        raise ValueError("source exception includes changes outside bookkeeping scope")
    for file,names in PROTECTED_FUNCTIONS.items():
        def functions(commit):
            tree = ast.parse(git("show",f"{commit}:{file}").decode())
            return {node.name:protected_function_ast(node) for node in tree.body
                    if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names}
        before,after = functions(original),functions(corrected)
        if set(before) != set(names) or before != after:
            raise ValueError("measurement/model conversion arithmetic changed")
    return {"changed_paths":sorted(paths), "git_diff_sha256":hashlib.sha256(
        git("diff","--no-renames","--binary",original,corrected)).hexdigest()}


def digest(body):
    return hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()


def create_exception(output):
    corrected = require_clean_commit()
    body = {"format_version":1,"reason":REASON,
        "original_execution_source_commit":EXECUTION_SOURCE,
        "postprocess_source_commit":corrected, **binding_identity(),
        **source_diff(EXECUTION_SOURCE,corrected), "training_changes":False}
    write_new(output,{**body,"content_sha256":digest(body)})


def verify_exception(record, original, corrected):
    body = dict(record or {})
    observed = body.pop("content_sha256",None)
    if (observed != digest(body) or body.get("format_version") != 1 or body.get("reason") != REASON
        or body.get("original_execution_source_commit") != original
        or body.get("postprocess_source_commit") != corrected or body.get("source_commit") != corrected
        or body.get("training_changes") is not False
        or body.get("protocol_sha256") != sha256_file(PROTOCOL)
        or body.get("matrix_sha256") != sha256_file(MATRIX)):
        raise ValueError("missing, stale or tampered postprocess source exception")
    current = binding_identity_without_source()
    if any(body.get(key) != value for key,value in current.items()):
        raise ValueError("source exception scientific inputs differ")
    changes = source_diff(original,corrected)
    if any(body.get(key) != value for key,value in changes.items()):
        raise ValueError("source exception Git diff differs")
    return record


def binding_identity_without_source():
    from src.experiments.experiment23_workflow import LOCK, spec
    return {"artifact_sha256":spec()["evaluation"]["artifact_sha256"],"data_lock_sha256":sha256_file(LOCK)}


def live_exception(original):
    corrected = require_clean_commit()
    if original == corrected:
        return None
    file = os.environ.get("MHAR_EXP23_SOURCE_EXCEPTION")
    if not file:
        raise ValueError("changed postprocess source requires MHAR_EXP23_SOURCE_EXCEPTION")
    return verify_exception(read_json(file),original,corrected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    create_exception(parser.parse_args().output)
