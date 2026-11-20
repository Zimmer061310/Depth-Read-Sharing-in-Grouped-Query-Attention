"""Offline Gate B builder: document exclusion, packing, provenance and hashes.

No model imports/scoring, network fetch, trainer or paid execution. SQLite keeps
the corpus identity/rank index on disk; token/provenance writers are streaming.
Policy/source review and full Gate B approval are separate from building bytes.
"""
from __future__ import annotations

from array import array
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile

from src.experiments.paper_sharing_protocol import SEEDS, active_seeds, expected_rows, require, sha


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def validate_policy(policy):
    require(policy.get("format_version") == 1 and type(policy.get("format_version")) is int
            and policy.get("study_id") == "paper-sharing-fresh-2k-v1", "unknown data policy")
    require(policy.get("status") in ("frozen_fixture", "frozen_local_build")
            and type(policy.get("synthetic_fixture")) is bool, "source/selection policy is not frozen")
    fixture = policy["synthetic_fixture"]
    require((policy["status"] == "frozen_fixture") == fixture, "fixture/status mismatch")
    require(isinstance(policy["dataset_revision"], str) and policy["dataset_revision"], "dataset revision required")
    for field in ("text_field", "identity_namespace", "selection_salt"):
        require(isinstance(policy[field], str) and policy[field], f"explicit {field} required")
    require(policy["identity_mode"] in ("native_id", "exact_text_sha256"), "unknown document identity rule")
    if policy["identity_mode"] == "native_id":
        require(isinstance(policy["id_field"], str) and policy["id_field"], "native ID field required")
    for field in ("selection_document_count", "primary_sequences", "sequence_length", "training_chunks_per_seed", "global_batch_size"):
        require(type(policy[field]) is int and policy[field] > 0, f"invalid {field}")
    require(set(policy["shuffle_seeds"]) == {str(seed) for seed in active_seeds()}
            and all(type(value) is int and value >= 0 for value in policy["shuffle_seeds"].values()), "all seed mappings required")
    require(policy["packing"] == "causal_t_plus_one_stride_t_eos_all_targets_v1", "changed packing/mask semantics")
    require(policy["shuffle"]["algorithm"] == "datasets_iterable_buffer_v1"
            and type(policy["shuffle"]["buffer_size"]) is int and policy["shuffle"]["buffer_size"] > 0,
            "explicit buffered shuffle policy required")
    require(policy["shuffle"]["datasets_version"] == importlib.metadata.version("datasets"), "datasets version mismatch")
    tokenizer = policy["tokenizer"]
    require(isinstance(tokenizer["revision"], str) and tokenizer["revision"], "tokenizer revision required")
    require(type(tokenizer["vocab_size"]) is int and tokenizer["vocab_size"] > 0
            and type(tokenizer["eos_token_id"]) is int
            and 0 <= tokenizer["eos_token_id"] < tokenizer["vocab_size"] <= 2**32, "invalid tokenizer range/EOS")
    require(tokenizer["kind"] == ("fixture_byte_tokenizer" if fixture else "local_hf_snapshot"), "tokenizer/fixture mismatch")
    if not fixture:
        for revision in (policy["dataset_revision"], tokenizer["revision"]):
            require(len(revision) == 40 and all(character in "0123456789abcdef" for character in revision),
                    "production dataset/tokenizer revision must be an immutable lowercase commit")
        require(policy["primary_sequences"] == 1024 and policy["sequence_length"] == 1024,
                "production primary must be exactly 1024 by 1024")
        require(policy["training_chunks_per_seed"] == 2000 * policy["global_batch_size"], "incomplete 2K training stream length")
        sha(policy["scientific_recipe_sha256"])
        require(isinstance(tokenizer["files"], dict) and tokenizer["files"], "tokenizer snapshot identities required")
    sources = policy["sources"]
    require(isinstance(sources, list) and sources, "ordered local source pool required")
    paths = []
    for source in sources:
        require(source["format"] in ("jsonl", "parquet"), "unsupported local source format")
        path = Path(source["path"]).resolve(strict=True)
        require(path.is_file(), "local source file missing")
        paths.append(str(path))
        sha(source["sha256"])
        require(file_sha256(path) == source["sha256"], "source file hash mismatch")
    require(len(paths) == len(set(paths)), "duplicate source shard path")
    return policy


def identity(document, policy):
    text = document[policy["text_field"]]
    require(isinstance(text, str), "document text must be a string")
    text_hash = hashlib.sha256(text.encode("utf-8", errors="strict")).hexdigest()
    if policy["identity_mode"] == "native_id":
        value = document[policy["id_field"]]
        require(isinstance(value, str) and value, "stable native ID must be a nonempty string")
        document_id = digest(["native_id_v1", policy["identity_namespace"], value])
    else:
        document_id = digest(["exact_text_utf8_v1", text_hash])
    return document_id, text_hash, text


def iter_documents(policy):
    """Yield the same pinned source/row order on every pass; never fetch data."""
    index = 0
    for source in policy["sources"]:
        if source["format"] == "jsonl":
            with Path(source["path"]).open(encoding="utf-8") as handle:
                for line in handle:
                    require(bool(line.strip()), "blank JSONL record is not a document")
                    yield index, json.loads(line)
                    index += 1
        else:
            import pyarrow.parquet as parquet
            columns = [policy["text_field"]]
            if policy["identity_mode"] == "native_id" and policy["id_field"] not in columns:
                columns.append(policy["id_field"])
            for batch in parquet.ParquetFile(source["path"]).iter_batches(batch_size=4096, columns=columns):
                for row in batch.to_pylist():
                    yield index, row
                    index += 1


class LocalTokenizer:
    """Verified offline tokenizer snapshot, never a model or Hub download."""
    def __init__(self, specification):
        self.specification = specification
        self.verify()
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(specification["directory"], local_files_only=True, trust_remote_code=False)
        require(len(self.tokenizer) == specification["vocab_size"]
                and self.tokenizer.eos_token_id == specification["eos_token_id"], "tokenizer vocabulary/EOS mismatch")

    def verify(self):
        root = Path(self.specification["directory"]).resolve(strict=True)
        files = self.specification["files"]
        require(set(files) == {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()},
                "tokenizer snapshot file inventory mismatch")
        for name, expected in files.items():
            require(not Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe tokenizer file name")
            require(file_sha256(root / name) == sha(expected), "tokenizer snapshot hash mismatch")

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)


def _checked_tokens(tokenizer, text, specification):
    tokens = list(tokenizer.encode(text))
    require(all(type(token) is int and 0 <= token < specification["vocab_size"] for token in tokens), "invalid token ID")
    return tokens + [specification["eos_token_id"]]


def _filtered_documents(policy, database):
    connection = sqlite3.connect(f"{Path(database).resolve().as_uri()}?mode=ro", uri=True)
    try:
        for _, document in iter_documents(policy):
            document_id, text_hash, text = identity(document, policy)
            if not text:
                continue
            row = connection.execute("SELECT text_hash, selected FROM documents WHERE id=?", (document_id,)).fetchone()
            require(row is not None and row[0] == text_hash, "source identity/content changed between passes")
            if not row[1]:
                yield {"document_id": document_id, "text": text}
    finally:
        connection.close()


def _shuffled_documents(policy, database, seed):
    from datasets import IterableDataset
    # Do not move this shuffle before _filtered_documents: exclusion precedes
    # shuffle and tokenization, including all duplicate occurrences of an ID.
    stream = IterableDataset.from_generator(_filtered_documents, gen_kwargs={"policy": policy, "database": str(database)})
    yield from stream.shuffle(seed=seed, buffer_size=policy["shuffle"]["buffer_size"])


def _write_packed(path, documents, tokenizer, policy, count):
    length = policy["sequence_length"]
    buffer, spans, base, position, emitted = [], [], 0, 0, 0
    used = set()
    with (path / "tokens.u32le").open("xb") as tokens_file, (path / "valid-target-mask.u8").open("xb") as masks_file, \
            (path / "provenance.jsonl").open("xb") as provenance:
        for document in documents:
            tokens = _checked_tokens(tokenizer, document["text"], policy["tokenizer"])
            spans.append((position, position + len(tokens), document["document_id"]))
            position += len(tokens)
            buffer.extend(tokens)
            offset = 0
            while len(buffer) - offset >= length + 1 and emitted < count:
                window = array("I", buffer[offset:offset + length + 1])
                require(window.itemsize == 4, "host unsigned-int width is not 32 bits")
                if sys.byteorder != "little":
                    window.byteswap()
                tokens_file.write(window.tobytes())
                masks_file.write(b"\x01" * length)
                contributors = []
                for start, end, document_id in spans:
                    left, right = max(start, base), min(end, base + length + 1)
                    if left < right:
                        contributors.append({"document_id": document_id, "raw_token_start": left - base,
                            "raw_token_end": right - base, "document_token_start": left - start,
                            "document_token_end": right - start})
                        used.add(document_id)
                provenance.write(canonical_bytes({"sequence": emitted, "spans": contributors}) + b"\n")
                emitted += 1
                offset += length
                base += length
                spans = [span for span in spans if span[1] > base]
            if emitted == count:
                break
            buffer = buffer[offset:]
        require(emitted == count, "insufficient selected or filtered tokens; do not adapt the frozen pool/selection")
        for handle in (tokens_file, masks_file, provenance):
            handle.flush()
            os.fsync(handle.fileno())
    return {"sequences": count, "sequence_length": length, "raw_tokens_per_sequence": length + 1,
            "input_ids": "raw[:-1]", "targets": "raw[1:]", "valid_targets": count * length,
            "contributing_document_ids": sorted(used), "contributing_document_ids_sha256": digest(sorted(used)),
            "files": {name: file_sha256(path / name) for name in ("tokens.u32le", "valid-target-mask.u8", "provenance.jsonl")}}


def _exclusive_json(path, value):
    with Path(path).open("xb") as handle:
        handle.write(canonical_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_bundle(policy, output, *, tokenizer=None):
    """Materialize a new local bundle. Failures preserve partial output, never retry.

    A successful build verifies separation within the supplied policy; it does
    not approve that source-selection lock or grant final Gate B/launch status.
    """
    validate_policy(policy)
    if policy["synthetic_fixture"]:
        require(tokenizer is not None and getattr(tokenizer, "fixture_only", False) is True, "explicit fixture tokenizer required")
    else:
        require(tokenizer is None, "production cannot inject an unverified tokenizer")
        tokenizer = LocalTokenizer(policy["tokenizer"])
    output = Path(output).resolve()
    output.mkdir(exist_ok=False)  # Reserve a fresh directory; no overwrites/deletion.
    _exclusive_json(output / "policy.json", policy)
    database = output / "document-index.sqlite"
    connection = sqlite3.connect(database)
    try:
        connection.execute("CREATE TABLE documents (id TEXT PRIMARY KEY, text_hash TEXT NOT NULL, rank TEXT NOT NULL, first_index INTEGER NOT NULL, selected INTEGER NOT NULL DEFAULT 0)")
        connection.execute("CREATE INDEX document_rank ON documents(rank, id)")
        connection.execute("CREATE TABLE selected_payload (id TEXT PRIMARY KEY, text TEXT NOT NULL)")
        raw_count = eligible = 0
        for index, document in iter_documents(policy):
            raw_count += 1
            document_id, text_hash, text = identity(document, policy)
            if not text:
                continue
            eligible += 1
            row = connection.execute("SELECT text_hash FROM documents WHERE id=?", (document_id,)).fetchone()
            require(row is None or row[0] == text_hash, "one native identity has inconsistent document content")
            if row is None:
                rank = digest(["primary-selection-v1", policy["selection_salt"], document_id])
                connection.execute("INSERT INTO documents(id,text_hash,rank,first_index) VALUES(?,?,?,?)", (document_id, text_hash, rank, index))
        rows = connection.execute("SELECT id,first_index FROM documents ORDER BY rank,id LIMIT ?", (policy["selection_document_count"],)).fetchall()
        require(len(rows) == policy["selection_document_count"], "selection document count exceeds eligible unique pool")
        excluded = sorted(row[0] for row in rows)
        selected_indices = {row[1]: row[0] for row in rows}
        connection.executemany("UPDATE documents SET selected=1 WHERE id=?", ((value,) for value in excluded))
        for index, document in iter_documents(policy):
            if index in selected_indices:
                document_id, text_hash, text = identity(document, policy)
                expected = connection.execute("SELECT text_hash FROM documents WHERE id=?", (selected_indices[index],)).fetchone()
                require(document_id == selected_indices[index] and expected[0] == text_hash, "selected source changed between passes")
                connection.execute("INSERT INTO selected_payload VALUES(?,?)", (document_id, text))
        connection.commit()
        _exclusive_json(output / "exclusions.json", {"document_ids": excluded, "sha256": digest(excluded)})
        selected = connection.execute("SELECT d.id,p.text FROM documents d JOIN selected_payload p ON d.id=p.id ORDER BY d.rank,d.id")
        primary_documents = ({"document_id": row[0], "text": row[1]} for row in selected)
        primary_path = output / "primary-reserved"
        primary_path.mkdir()
        primary = _write_packed(primary_path, primary_documents, tokenizer, policy, policy["primary_sequences"])
        require(set(primary["contributing_document_ids"]) <= set(excluded), "primary provenance not fully excluded")
        streams = {}
        for seed in active_seeds():
            directory = output / f"training-seed{seed}"
            directory.mkdir()
            stream = _shuffled_documents(policy, database, policy["shuffle_seeds"][str(seed)])
            try:
                streams[str(seed)] = _write_packed(directory, stream, tokenizer, policy, policy["training_chunks_per_seed"])
            finally:
                stream.close()
            require(not set(streams[str(seed)]["contributing_document_ids"]).intersection(excluded), "held-out identity leaked into training")
        manifest = {"format_version": 1, "study_id": policy["study_id"], "policy_sha256": digest(policy),
            "synthetic_fixture": policy["synthetic_fixture"], "source_rows": raw_count, "eligible_occurrences": eligible,
            "unique_eligible_documents": connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
            "exclusion_sha256": digest(excluded), "primary": primary, "training_streams": streams,
            "run_streams": {row["id"]: {"seed": row["seed"], "directory": f"training-seed{row['seed']}",
                "tokens_sha256": streams[str(row["seed"])]["files"]["tokens.u32le"]} for row in expected_rows()},
            "primary_scoring_reserved": True, "separation_checks_passed": True,
            "real_data_gate_passed": False, "requires_source_lock_review": True, "execution_authorized": False}
        validate_policy(policy)  # Rehash all source files; reject mutation during construction.
        if not policy["synthetic_fixture"]:
            tokenizer.verify()
        connection.close()
        connection = None
        manifest["document_index_sha256"] = file_sha256(database)
        _exclusive_json(output / "manifest.json", manifest)
        # Atomic, exclusive completion marker. A crash/exception leaves partial
        # output for audit; readers must reject missing/invalid completion.
        fd, temporary = tempfile.mkstemp(prefix=".complete-", dir=output)
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_bytes({"manifest_sha256": file_sha256(output / "manifest.json")}) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output / "COMPLETE.json")
        os.unlink(temporary)
        return manifest
    finally:
        if connection is not None:
            connection.close()


def verify_bundle(output, *, expected_manifest_sha256=None):
    """Structural/hash/provenance verification only; never run a model."""
    root = Path(output).resolve(strict=True)
    completion = json.loads((root / "COMPLETE.json").read_text())
    require(file_sha256(root / "manifest.json") == sha(completion["manifest_sha256"]), "manifest/completion mismatch")
    if expected_manifest_sha256 is not None:
        require(completion["manifest_sha256"] == sha(expected_manifest_sha256), "externally frozen manifest identity mismatch")
    manifest = json.loads((root / "manifest.json").read_text())
    policy = json.loads((root / "policy.json").read_text())
    require(manifest["format_version"] == 1 and manifest["study_id"] == policy["study_id"] == "paper-sharing-fresh-2k-v1"
            and manifest["synthetic_fixture"] is policy["synthetic_fixture"], "bundle identity/scope mismatch")
    require(set(manifest["training_streams"]) == {str(seed) for seed in active_seeds()}, "missing/extra training seed stream")
    require(digest(policy) == manifest["policy_sha256"], "policy hash mismatch")
    require(file_sha256(root / "document-index.sqlite") == manifest["document_index_sha256"], "document index hash mismatch")
    exclusions = json.loads((root / "exclusions.json").read_text())
    excluded = set(exclusions["document_ids"])
    require(digest(sorted(excluded)) == exclusions["sha256"] == manifest["exclusion_sha256"], "exclusion identity mismatch")
    for name, stream in [("primary-reserved", manifest["primary"]), *[(f"training-seed{seed}", manifest["training_streams"][str(seed)]) for seed in active_seeds()]]:
        directory = root / name
        length, count = stream["sequence_length"], stream["sequences"]
        expected_count = policy["primary_sequences"] if name == "primary-reserved" else policy["training_chunks_per_seed"]
        require(type(length) is int and length == policy["sequence_length"]
                and type(count) is int and count == expected_count
                and stream["raw_tokens_per_sequence"] == length + 1 and stream["valid_targets"] == count * length
                and stream["input_ids"] == "raw[:-1]" and stream["targets"] == "raw[1:]", "stream dimensions or causal-target semantics mismatch")
        require(set(stream["files"]) == {"tokens.u32le", "valid-target-mask.u8", "provenance.jsonl"}, "incomplete stream file inventory")
        for file, expected in stream["files"].items():
            require(file in ("tokens.u32le", "valid-target-mask.u8", "provenance.jsonl"), "unexpected stream file")
            require(file_sha256(directory / file) == sha(expected), "packed stream file hash mismatch")
        require((directory / "tokens.u32le").stat().st_size == count * (length + 1) * 4, "packed token byte count mismatch")
        require((directory / "valid-target-mask.u8").read_bytes() == b"\x01" * (count * length), "invalid target mask")
        previous = None
        for chunk in _raw_chunks(directory / "tokens.u32le", length, count):
            require(all(token < policy["tokenizer"]["vocab_size"] for token in chunk), "out-of-vocabulary packed token")
            require(previous is None or chunk[0] == previous, "broken stride-T lookahead continuity")
            previous = chunk[-1]
        seen, rows = set(), 0
        with (directory / "provenance.jsonl").open() as handle:
            for line in handle:
                record = json.loads(line)
                require(record["sequence"] == rows, "sequence order mismatch")
                position = 0
                for span in record["spans"]:
                    require(span["raw_token_start"] == position and position < span["raw_token_end"] <= length + 1, "incomplete or overlapping token provenance")
                    require(type(span["document_token_start"]) is int and span["document_token_start"] >= 0
                            and span["document_token_end"] - span["document_token_start"] == span["raw_token_end"] - position,
                            "invalid document token offsets")
                    position = span["raw_token_end"]
                    seen.add(sha(span["document_id"]))
                require(position == length + 1, "lookahead target lacks source provenance")
                rows += 1
        require(rows == count and sorted(seen) == stream["contributing_document_ids"]
                and digest(sorted(seen)) == stream["contributing_document_ids_sha256"], "document/sequence inventory mismatch")
        require(seen <= excluded if name == "primary-reserved" else not seen.intersection(excluded), "document leakage")
    expected_mapping = {row["id"]: {"seed": row["seed"], "directory": f"training-seed{row['seed']}",
        "tokens_sha256": manifest["training_streams"][str(row["seed"])]["files"]["tokens.u32le"]} for row in expected_rows()}
    require(manifest["run_streams"] == expected_mapping and len(expected_mapping) == 9, "missing or mismatched run-stream identity")
    require(manifest["primary_scoring_reserved"] is True and manifest["real_data_gate_passed"] is False
            and manifest["execution_authorized"] is False, "bundle cannot grant scoring/launch authority")
    return {"bundle_hashes_and_provenance_verified": True, "synthetic_fixture": manifest["synthetic_fixture"],
            "externally_bound_identity_verified": expected_manifest_sha256 is not None,
            "real_data_gate_passed": False, "primary_model_scoring_performed": False, "execution_authorized": False}


def _raw_chunks(path, length, count, skip=0):
    with Path(path).open("rb") as handle:
        handle.seek(skip * (length + 1) * 4)
        for _ in range(skip, count):
            payload = handle.read((length + 1) * 4)
            require(len(payload) == (length + 1) * 4, "truncated packed stream")
            values = array("I")
            values.frombytes(payload)
            if sys.byteorder != "little":
                values.byteswap()
            yield list(values)


def iter_training_examples(output, seed, *, expected_manifest_sha256, skip_chunks=0):
    """Replay only a training stream, including exact chunk-offset resume.

    No primary-scoring reader is exposed. Trainer/RNG/checkpoint binding remains
    a separate integration task; this reader never starts or advances a model.
    """
    require(type(seed) is int and seed in active_seeds(), "unknown training seed")
    require(type(skip_chunks) is int and skip_chunks >= 0, "invalid chunk resume offset")
    sha(expected_manifest_sha256)
    verify_bundle(output, expected_manifest_sha256=expected_manifest_sha256)
    root = Path(output)
    manifest = json.loads((root / "manifest.json").read_text())
    stream = manifest["training_streams"][str(seed)]
    require(skip_chunks <= stream["sequences"], "resume offset beyond frozen stream")
    for index, raw in enumerate(_raw_chunks(root / f"training-seed{seed}" / "tokens.u32le",
                                stream["sequence_length"], stream["sequences"], skip_chunks), start=skip_chunks):
        yield {"chunk_index": index, "input_ids": raw[:-1], "targets": raw[1:],
               "valid_target_mask": [True] * stream["sequence_length"]}
