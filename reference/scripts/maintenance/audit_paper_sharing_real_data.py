#!/usr/bin/env python3
"""Retokenize/repack a frozen local bundle and test real stream-offset resume.

No model, primary scoring, GPU, network or trainer. This independent packing
loop checks source-to-token/provenance correspondence, not just self-hashes.
"""
from array import array
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.experiments import paper_sharing_data as data
from src.experiments.paper_sharing_protocol import SEEDS, require


def _eligible(policy, excluded):
    for _, document in data.iter_documents(policy):
        key, _, text = data.identity(document, policy)
        if text and key not in excluded:
            yield {"document_id": key, "text": text}


def _compare_repack(directory, documents, tokenizer, policy, count):
    """Separate packing oracle, tracking offsets in the unchunked token stream."""
    length = policy["sequence_length"]
    pending, intervals, emitted, consumed, total = [], [], 0, 0, 0
    seen = set()
    chunks = data._raw_chunks(directory / "tokens.u32le", length, count)
    with (directory / "provenance.jsonl").open() as provenance:
        for document in documents:
            encoded = tokenizer.encode(document["text"]) + [policy["tokenizer"]["eos_token_id"]]
            intervals.append((total, total + len(encoded), document["document_id"]))
            total += len(encoded)
            pending.extend(encoded)
            position = 0
            while len(pending) - position >= length + 1 and emitted < count:
                raw = pending[position:position + length + 1]
                require(raw == next(chunks), "independent source-to-token replay mismatch")
                expected = []
                for start, end, key in intervals:
                    left, right = max(start, consumed), min(end, consumed + length + 1)
                    if left < right:
                        expected.append({"document_id": key, "raw_token_start": left - consumed,
                            "raw_token_end": right - consumed, "document_token_start": left - start,
                            "document_token_end": right - start})
                        seen.add(key)
                require(json.loads(next(provenance)) == {"sequence": emitted, "spans": expected},
                        "independent source-to-provenance replay mismatch")
                emitted += 1
                consumed += length
                position += length
                intervals = [item for item in intervals if item[1] > consumed]
            if emitted == count:
                break
            pending = pending[position:]
        require(emitted == count and not provenance.readline(), "incomplete replay or extra provenance")
    return seen


def _resume_audit(root, seed, manifest_sha, count, length):
    offsets = sorted({0, 1, min(32, count), count // 2, count - 1, count})
    reports = []
    for offset in offsets:
        # Exercise the public reader, including its external identity and bundle
        # checks, rather than only seeking the raw file. Hash the entire suffix.
        observed = hashlib.sha256()
        rows = 0
        for example in data.iter_training_examples(root, seed, expected_manifest_sha256=manifest_sha,
                                                    skip_chunks=offset):
            require(example["chunk_index"] == offset + rows
                    and len(example["input_ids"]) == length
                    and example["input_ids"][1:] == example["targets"][:-1]
                    and example["valid_target_mask"] == [True] * length, "resume target/order/mask mismatch")
            raw = array("I", example["input_ids"] + [example["targets"][-1]])
            if sys.byteorder != "little":
                raw.byteswap()
            observed.update(raw.tobytes())
            rows += 1
        expected = hashlib.sha256()
        with (root / f"training-seed{seed}" / "tokens.u32le").open("rb") as handle:
            handle.seek(offset * (length + 1) * 4)
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                expected.update(chunk)
        require(rows == count - offset and observed.hexdigest() == expected.hexdigest(), "full resume suffix mismatch")
        reports.append({"offset": offset, "suffix_examples": rows, "suffix_sha256": observed.hexdigest()})
    return reports


def audit_bundle(root, expected_manifest_sha256, frozen_policy, *, tokenizer=None, progress=None):
    root = Path(root)
    data.verify_bundle(root, expected_manifest_sha256=expected_manifest_sha256)
    policy = json.loads((root / "policy.json").read_text())
    require(data.digest(policy) == data.digest(frozen_policy), "external frozen policy mismatch")
    data.validate_policy(policy)  # Independently rehash original source bytes.
    if policy["synthetic_fixture"]:
        require(tokenizer is not None and getattr(tokenizer, "fixture_only", False), "fixture tokenizer required")
    else:
        require(tokenizer is None, "production audit cannot inject a tokenizer")
        tokenizer = data.LocalTokenizer(policy["tokenizer"])
    manifest = json.loads((root / "manifest.json").read_text())
    exclusions = json.loads((root / "exclusions.json").read_text())["document_ids"]
    excluded = set(exclusions)
    connection = sqlite3.connect(f"{(root / 'document-index.sqlite').resolve().as_uri()}?mode=ro", uri=True)
    try:
        source_ids = set()
        source_rows = eligible = 0
        for _, document in data.iter_documents(policy):
            source_rows += 1
            key, text_hash, text = data.identity(document, policy)
            if not text:
                continue
            eligible += 1
            source_ids.add(key)
            row = connection.execute("SELECT text_hash,rank,selected FROM documents WHERE id=?", (key,)).fetchone()
            rank = data.digest(["primary-selection-v1", policy["selection_salt"], key])
            require(row == (text_hash, rank, int(key in excluded)), "raw source/index/selection mismatch")
            if key in excluded:
                stored = connection.execute("SELECT text FROM selected_payload WHERE id=?", (key,)).fetchone()
                require(stored == (text,), "selected payload differs from raw source")
        require(len(source_ids) == manifest["unique_eligible_documents"]
                == connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
                and source_rows == manifest["source_rows"] and eligible == manifest["eligible_occurrences"], "source pool inventory mismatch")
        ranked = [item[0] for item in connection.execute("SELECT id FROM documents ORDER BY rank,id LIMIT ?",
                                                       (policy["selection_document_count"],))]
        require(len(exclusions) == policy["selection_document_count"] and sorted(ranked) == exclusions,
                "held-out selection is not the frozen complete-pool rank selection")
        if progress:
            progress("complete source/index/selection audit passed")
        selected = ({"document_id": key, "text": text} for key, text in connection.execute(
            "SELECT d.id,p.text FROM documents d JOIN selected_payload p ON d.id=p.id ORDER BY d.rank,d.id"))
        seen = _compare_repack(root / "primary-reserved", selected, tokenizer, policy, policy["primary_sequences"])
        require(seen <= excluded and sorted(seen) == manifest["primary"]["contributing_document_ids"], "primary source inventory mismatch")
        if progress:
            progress("primary source-to-token/provenance replay passed; no model scoring")
        from datasets import IterableDataset
        streams = {}
        for seed in SEEDS:
            documents = IterableDataset.from_generator(_eligible, gen_kwargs={"policy": policy, "excluded": excluded})
            documents = documents.shuffle(seed=policy["shuffle_seeds"][str(seed)], buffer_size=policy["shuffle"]["buffer_size"])
            iterator = iter(documents)
            try:
                seen = _compare_repack(root / f"training-seed{seed}", iterator, tokenizer, policy, policy["training_chunks_per_seed"])
            finally:
                if hasattr(iterator, "close"):
                    iterator.close()
            require(not seen.intersection(excluded)
                    and sorted(seen) == manifest["training_streams"][str(seed)]["contributing_document_ids"], "real training document leakage")
            if progress:
                progress(f"seed {seed}: complete independent token/provenance replay passed")
            streams[str(seed)] = {"train_eval_intersection_count": 0,
                "ordered_tokens_sha256": manifest["training_streams"][str(seed)]["files"]["tokens.u32le"],
                "all_three_run_aliases_verified": True,
                "resume": _resume_audit(root, seed, expected_manifest_sha256, policy["training_chunks_per_seed"], policy["sequence_length"])}
            if progress:
                progress(f"seed {seed}: six full-suffix stream-offset resume checks passed")
    finally:
        connection.close()
    data.validate_policy(policy)
    if not policy["synthetic_fixture"]:
        tokenizer.verify()
    data.verify_bundle(root, expected_manifest_sha256=expected_manifest_sha256)
    return {"format_version": 1, "study_id": policy["study_id"], "synthetic_fixture": policy["synthetic_fixture"],
        "status": "fixture_audit_passed" if policy["synthetic_fixture"] else "real_data_audit_passed",
        "manifest_sha256": expected_manifest_sha256, "policy_sha256": data.digest(policy),
        "source_pool_rehashed_and_replayed": True, "selected_documents": len(excluded),
        "primary_contributing_documents": len(manifest["primary"]["contributing_document_ids"]),
        "primary": manifest["primary"], "exclusion_sha256": manifest["exclusion_sha256"],
        "training_stream_audits": streams,
        "real_data_gate_passed": not policy["synthetic_fixture"],
        "trainer_ddp_accumulation_optimizer_rng_resume_validated": False,
        "primary_model_scoring_performed": False, "real_model_gate_passed": False,
        "paid_execution_authorized": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--policy-lock", type=Path, required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    require(not args.receipt.exists(), "refusing to overwrite an audit receipt")
    require(data.file_sha256(args.policy_lock) == args.policy_sha256, "external policy file identity mismatch")
    for key, value in {"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_MODE": "disabled"}.items():
        os.environ[key] = value
    started = time.monotonic()
    scope = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "audit_source_sha256": data.file_sha256(__file__), "bundle": str(args.bundle.resolve()),
        "policy_file_sha256": args.policy_sha256,
        "data_module_sha256": data.file_sha256(ROOT / "src/experiments/paper_sharing_data.py"),
        "python": sys.version.split()[0],
        "dependencies": {name: importlib.metadata.version(name) for name in ("datasets", "numpy", "pyarrow", "transformers", "tokenizers")}}
    try:
        report = audit_bundle(args.bundle, args.manifest_sha256, json.loads(args.policy_lock.read_text()),
                              progress=lambda message: print(message, flush=True))
    except Exception as error:
        data._exclusive_json(args.receipt, {**scope, "status": "failed", "error_type": type(error).__name__,
            "error": str(error), "elapsed_seconds": time.monotonic() - started, "real_data_gate_passed": False})
        raise
    report.update(scope)
    report["elapsed_seconds"] = time.monotonic() - started
    data._exclusive_json(args.receipt, report)
    print(json.dumps({key: value for key, value in report.items() if key != "primary"}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
