"""Disposable local corpus/tokenizer fixtures, not production Gate B evidence."""
from array import array
import copy
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.experiments import paper_sharing_data as data
from src.experiments.paper_sharing_protocol import ProtocolError, SEEDS, expected_rows


ROOT = Path(__file__).resolve().parents[1]


class ByteTokenizer:
    fixture_only = True

    def __init__(self):
        self.calls = []

    def encode(self, text):
        self.calls.append(text)
        return list(text.encode("utf-8"))


def write_json(path, value):
    path.write_bytes(data.canonical_bytes(value) + b"\n")


def corpus_fixture(root, documents=None):
    documents = documents or [{"id": f"doc-{index}", "text": f"document-{index:02d} " + chr(65 + index % 26) * 48} for index in range(40)]
    path = root / "corpus.jsonl"
    path.write_bytes(b"".join(data.canonical_bytes(document) + b"\n" for document in documents))
    policy = {
        "format_version": 1, "study_id": "paper-sharing-fresh-2k-v1", "status": "frozen_fixture",
        "synthetic_fixture": True, "dataset_revision": "fixture-source-v1", "text_field": "text",
        "identity_mode": "native_id", "id_field": "id", "identity_namespace": "fixture-source-v1",
        "selection_salt": "fixture-selection-v1", "selection_document_count": 4,
        "primary_sequences": 4, "sequence_length": 8, "training_chunks_per_seed": 12,
        "global_batch_size": 2, "shuffle_seeds": {str(seed): seed for seed in SEEDS},
        "packing": "causal_t_plus_one_stride_t_eos_all_targets_v1",
        "shuffle": {"algorithm": "datasets_iterable_buffer_v1", "buffer_size": 7,
                    "datasets_version": importlib.metadata.version("datasets")},
        "tokenizer": {"kind": "fixture_byte_tokenizer", "revision": "fixture-byte-v1", "vocab_size": 257, "eos_token_id": 256},
        "sources": [{"path": str(path), "format": "jsonl", "sha256": data.file_sha256(path)}],
    }
    return policy, documents


def build_fixture(root, **overrides):
    policy, documents = corpus_fixture(root)
    policy.update(overrides)
    tokenizer = ByteTokenizer()
    output = root / "bundle"
    manifest = data.build_bundle(policy, output, tokenizer=tokenizer)
    return policy, documents, tokenizer, output, manifest


def reseal(root, manifest):
    write_json(root / "manifest.json", manifest)
    write_json(root / "COMPLETE.json", {"manifest_sha256": data.file_sha256(root / "manifest.json")})


def fixture_training_examples(output, seed, *, skip_chunks=0):
    # Test helper only: fixtures have just been built in this test. Production
    # callers must supply the separately frozen launch-lock identity.
    return data.iter_training_examples(output, seed, skip_chunks=skip_chunks,
                                      expected_manifest_sha256=data.file_sha256(output / "manifest.json"))


class DataPipelineTests(unittest.TestCase):
    def test_complete_bundle_real_bytes_and_nine_run_aliases(self):
        with tempfile.TemporaryDirectory(prefix="mhar-gate-b-") as directory:
            _, _, _, output, manifest = build_fixture(Path(directory))
            report = data.verify_bundle(output)
            self.assertTrue(report["bundle_hashes_and_provenance_verified"])
            self.assertTrue(report["synthetic_fixture"])
            self.assertFalse(report["real_data_gate_passed"])
            self.assertEqual(set(manifest["run_streams"]), {row["id"] for row in expected_rows()})
            for seed in SEEDS:
                paths = {row["directory"] for row in manifest["run_streams"].values() if row["seed"] == seed}
                self.assertEqual(paths, {f"training-seed{seed}"})
            self.assertEqual(len(list(output.glob("training-seed*"))), 3)

    def test_deterministic_rebuild_matches_manifests_and_packed_bytes(self):
        with tempfile.TemporaryDirectory(prefix="mhar-gate-b-determinism-") as directory:
            root = Path(directory)
            policy, _ = corpus_fixture(root)
            first = data.build_bundle(policy, root / "first", tokenizer=ByteTokenizer())
            second = data.build_bundle(policy, root / "second", tokenizer=ByteTokenizer())
            self.assertEqual(first, second)
            self.assertEqual((root / "first/manifest.json").read_bytes(), (root / "second/manifest.json").read_bytes())

    def test_exclusion_before_training_tokenization_including_unused_selected_tail(self):
        with tempfile.TemporaryDirectory(prefix="mhar-gate-b-exclusion-") as directory:
            policy, documents, tokenizer, output, manifest = build_fixture(Path(directory))
            excluded = set(json.loads((output / "exclusions.json").read_text())["document_ids"])
            used = set(manifest["primary"]["contributing_document_ids"])
            self.assertLess(len(used), len(excluded))
            primary_calls = len(used)
            lookup = {row["text"]: data.identity(row, policy)[0] for row in documents}
            self.assertTrue(all(lookup[text] not in excluded for text in tokenizer.calls[primary_calls:]))
            for stream in manifest["training_streams"].values():
                self.assertFalse(set(stream["contributing_document_ids"]) & excluded)

    def test_exact_hf_buffer_shuffle_and_legacy_stride_t_token_packing(self):
        from datasets import IterableDataset
        with tempfile.TemporaryDirectory(prefix="mhar-gate-b-shuffle-") as directory:
            policy, documents, _, output, _ = build_fixture(Path(directory))
            excluded = set(json.loads((output / "exclusions.json").read_text())["document_ids"])
            eligible = [row for row in documents if data.identity(row, policy)[0] not in excluded]
            for seed in SEEDS:
                stream = IterableDataset.from_generator(lambda: iter(eligible)).shuffle(seed=seed, buffer_size=7)
                raw = []
                for row in stream:
                    raw.extend(list(row["text"].encode()) + [256])
                expected = [raw[index * 8:index * 8 + 9] for index in range(12)]
                observed = list(fixture_training_examples(output, seed))
                self.assertEqual([row["input_ids"] + row["targets"][-1:] for row in observed], expected)

    def test_training_resume_is_exact_suffix_and_masks_are_valid(self):
        with tempfile.TemporaryDirectory(prefix="mhar-gate-b-resume-") as directory:
            _, _, _, output, _ = build_fixture(Path(directory))
            whole = list(fixture_training_examples(output, 123))
            self.assertEqual(list(fixture_training_examples(output, 123, skip_chunks=5)), whole[5:])
            self.assertEqual(list(fixture_training_examples(output, 123, skip_chunks=12)), [])
            self.assertTrue(all(row["valid_target_mask"] == [True] * 8 for row in whole))
            self.assertEqual(whole[0]["targets"][-1], whole[1]["input_ids"][0])
            for seed, skip in ((42, 0), (123, -1), (123, 13), (True, 0), (123, True)):
                with self.subTest(seed=seed, skip=skip), self.assertRaises(ProtocolError):
                    list(fixture_training_examples(output, seed, skip_chunks=skip))

    def test_training_reader_requires_external_identity_and_rejects_resealed_bundle(self):
        with tempfile.TemporaryDirectory(prefix="mhar-gate-b-external-lock-") as directory:
            _, _, _, output, manifest = build_fixture(Path(directory))
            frozen = data.file_sha256(output / "manifest.json")
            self.assertTrue(data.verify_bundle(output, expected_manifest_sha256=frozen)["externally_bound_identity_verified"])
            with self.assertRaises(TypeError):
                list(data.iter_training_examples(output, 123))
            with self.assertRaises(ProtocolError):
                list(data.iter_training_examples(output, 123, expected_manifest_sha256="short"))
            modified = copy.deepcopy(manifest)
            modified["source_rows"] += 1
            reseal(output, modified)
            with self.assertRaisesRegex(ProtocolError, "externally frozen manifest"):
                list(data.iter_training_examples(output, 123, expected_manifest_sha256=frozen))

    def test_source_change_during_construction_cannot_publish_completion(self):
        with tempfile.TemporaryDirectory(prefix="mhar-gate-b-source-race-") as directory:
            root = Path(directory)
            policy, _ = corpus_fixture(root)
            class Mutating(ByteTokenizer):
                def encode(self, text):
                    if not self.calls:
                        with (root / "corpus.jsonl").open("ab") as handle:
                            handle.write(data.canonical_bytes({"id": "new", "text": "changed source"}) + b"\n")
                    return super().encode(text)
            with self.assertRaises(ProtocolError):
                data.build_bundle(policy, root / "partial", tokenizer=Mutating())
            self.assertFalse((root / "partial/COMPLETE.json").exists())

    def test_rehashed_out_of_vocabulary_token_and_broken_lookahead_rejected(self):
        for mode in ("out_of_vocab", "broken_lookahead"):
            with tempfile.TemporaryDirectory(prefix="mhar-gate-b-invalid-token-bytes-") as directory:
                _, _, _, output, manifest = build_fixture(Path(directory))
                path = output / "training-seed123/tokens.u32le"
                tokens = array("I")
                tokens.frombytes(path.read_bytes())
                tokens[0 if mode == "out_of_vocab" else 9] = 1000 if mode == "out_of_vocab" else (tokens[8] + 1) % 257
                path.write_bytes(tokens.tobytes())
                modified = copy.deepcopy(manifest)
                modified["training_streams"]["123"]["files"][path.name] = data.file_sha256(path)
                for mapping in modified["run_streams"].values():
                    if mapping["seed"] == 123:
                        mapping["tokens_sha256"] = modified["training_streams"]["123"]["files"][path.name]
                reseal(output, modified)
                with self.subTest(mode=mode), self.assertRaises(ProtocolError):
                    data.verify_bundle(output)

    def test_target_only_contributor_is_in_provenance(self):
        with tempfile.TemporaryDirectory(prefix="mhar-gate-b-lookahead-") as directory:
            root = Path(directory)
            policy, _ = corpus_fixture(root)
            policy["sequence_length"] = 4
            result = data._write_packed(root, [{"document_id": "a" * 64, "text": "abc"},
                {"document_id": "b" * 64, "text": "DEF"}], ByteTokenizer(), policy, 1)
            record = json.loads((root / "provenance.jsonl").read_text())
            self.assertEqual(set(result["contributing_document_ids"]), {"a" * 64, "b" * 64})
            self.assertEqual(record["spans"][-1], {"document_id": "b" * 64, "raw_token_start": 4,
                "raw_token_end": 5, "document_token_start": 0, "document_token_end": 1})
            raw = next(data._raw_chunks(root / "tokens.u32le", 4, 1))
            self.assertEqual(raw, [97, 98, 99, 256, 68])

    def test_real_1024_by_1024_file_dimensions_on_disposable_byte_fixture(self):
        with tempfile.TemporaryDirectory(prefix="mhar-gate-b-dimensions-") as directory:
            root = Path(directory)
            documents = [{"id": str(index), "text": chr(65 + index) * 600000} for index in range(4)]
            policy, _ = corpus_fixture(root, documents)
            policy.update(selection_document_count=2, primary_sequences=1024, sequence_length=1024, training_chunks_per_seed=2)
            output = root / "bundle"
            manifest = data.build_bundle(policy, output, tokenizer=ByteTokenizer())
            self.assertEqual((output / "primary-reserved/tokens.u32le").stat().st_size, 1024 * 1025 * 4)
            self.assertEqual((output / "primary-reserved/valid-target-mask.u8").stat().st_size, 1024 * 1024)
            self.assertEqual(manifest["primary"]["valid_targets"], 1024 * 1024)
            self.assertFalse(data.verify_bundle(output)["real_data_gate_passed"])

    def test_exact_text_identity_has_no_implicit_normalization(self):
        with tempfile.TemporaryDirectory(prefix="mhar-gate-b-identity-") as directory:
            policy, _ = corpus_fixture(Path(directory))
            policy["identity_mode"] = "exact_text_sha256"
            variants = ["x", " x", "x\n", "x\r\n", "é", "e\u0301"]
            ids = [data.identity({"text": text}, policy)[0] for text in variants]
            self.assertEqual(len(set(ids)), len(variants))
            self.assertEqual(data.identity({"text": "x", "id": "first"}, policy)[0],
                             data.identity({"text": "x", "id": "second"}, policy)[0])

    def test_native_identity_namespace_and_invalid_ids(self):
        with tempfile.TemporaryDirectory(prefix="mhar-gate-b-native-") as directory:
            policy, _ = corpus_fixture(Path(directory))
            first = data.identity({"id": "stable", "text": "same"}, policy)[0]
            changed = copy.deepcopy(policy)
            changed["identity_namespace"] = "other-source"
            self.assertNotEqual(first, data.identity({"id": "stable", "text": "same"}, changed)[0])
            for value in (1, True, "", None):
                with self.subTest(value=value), self.assertRaises(ProtocolError):
                    data.identity({"id": value, "text": "same"}, policy)

    def test_duplicate_native_id_with_different_content_rejected_and_partial_retained(self):
        with tempfile.TemporaryDirectory(prefix="mhar-gate-b-conflict-") as directory:
            root = Path(directory)
            policy, documents = corpus_fixture(root)
            documents.append({"id": documents[0]["id"], "text": "conflicting bytes"})
            path = root / "corpus.jsonl"
            path.write_bytes(b"".join(data.canonical_bytes(row) + b"\n" for row in documents))
            policy["sources"][0]["sha256"] = data.file_sha256(path)
            with self.assertRaisesRegex(ProtocolError, "inconsistent document content"):
                data.build_bundle(policy, root / "partial", tokenizer=ByteTokenizer())
            self.assertTrue((root / "partial/policy.json").exists())
            self.assertFalse((root / "partial/COMPLETE.json").exists())

    def test_identical_duplicate_occurrences_cannot_escape_exclusion(self):
        with tempfile.TemporaryDirectory(prefix="mhar-gate-b-duplicates-") as directory:
            root = Path(directory)
            policy, documents = corpus_fixture(root)
            repeated = documents + documents
            path = root / "corpus.jsonl"
            path.write_bytes(b"".join(data.canonical_bytes(row) + b"\n" for row in repeated))
            policy["sources"][0]["sha256"] = data.file_sha256(path)
            manifest = data.build_bundle(policy, root / "bundle", tokenizer=ByteTokenizer())
            self.assertEqual(manifest["eligible_occurrences"], 80)
            self.assertEqual(manifest["unique_eligible_documents"], 40)
            data.verify_bundle(root / "bundle")

    def test_source_hash_duplicate_shard_and_shuffle_version_rejected(self):
        with tempfile.TemporaryDirectory(prefix="mhar-gate-b-source-") as directory:
            policy, _ = corpus_fixture(Path(directory))
            variants = []
            mismatch = copy.deepcopy(policy)
            mismatch["sources"][0]["sha256"] = "0" * 64
            variants.append(mismatch)
            duplicate = copy.deepcopy(policy)
            duplicate["sources"].append(copy.deepcopy(duplicate["sources"][0]))
            variants.append(duplicate)
            version = copy.deepcopy(policy)
            version["shuffle"]["datasets_version"] = "wrong"
            variants.append(version)
            for variant in variants:
                with self.assertRaises(ProtocolError):
                    data.validate_policy(variant)

    def test_pending_and_malformed_policies_fail_before_output_creation(self):
        with tempfile.TemporaryDirectory(prefix="mhar-gate-b-pending-") as directory:
            root = Path(directory)
            policy, _ = corpus_fixture(root)
            for key, value in (("status", "pending"), ("primary_sequences", True), ("sequence_length", 0),
                               ("packing", "pad_or_mask_boundaries"), ("shuffle_seeds", {"123": 123})):
                invalid = copy.deepcopy(policy)
                invalid[key] = value
                with self.subTest(key=key), self.assertRaises(ProtocolError):
                    data.build_bundle(invalid, root / "invalid", tokenizer=ByteTokenizer())
                self.assertFalse((root / "invalid").exists())

    def test_no_overwrite_insufficient_selection_or_training(self):
        with tempfile.TemporaryDirectory(prefix="mhar-gate-b-capacity-") as directory:
            root = Path(directory)
            policy, _ = corpus_fixture(root)
            output = root / "exists"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                data.build_bundle(policy, output, tokenizer=ByteTokenizer())
            variants = [{"selection_document_count": 100}, {"primary_sequences": 100000}, {"training_chunks_per_seed": 100000}]
            for index, changes in enumerate(variants):
                modified = copy.deepcopy(policy)
                modified.update(changes)
                path = root / f"insufficient-{index}"
                with self.subTest(changes=changes), self.assertRaises(ProtocolError):
                    data.build_bundle(modified, path, tokenizer=ByteTokenizer())
                self.assertFalse((path / "COMPLETE.json").exists())

    def test_production_policy_rejects_fixture_tokenizer_and_wrong_dimensions(self):
        with tempfile.TemporaryDirectory(prefix="mhar-gate-b-production-") as directory:
            policy, _ = corpus_fixture(Path(directory))
            policy.update(status="frozen_local_build", synthetic_fixture=False)
            with self.assertRaises(ProtocolError):
                data.validate_policy(policy)
            policy["tokenizer"]["kind"] = "local_hf_snapshot"
            policy["dataset_revision"] = "a" * 40
            policy["tokenizer"]["revision"] = "b" * 40
            with self.assertRaisesRegex(ProtocolError, "1024 by 1024"):
                data.validate_policy(policy)

    def test_complete_production_policy_shape_checks_without_materialization(self):
        with tempfile.TemporaryDirectory(prefix="mhar-gate-b-policy-shape-") as directory:
            root = Path(directory)
            policy, _ = corpus_fixture(root)
            policy.update(status="frozen_local_build", synthetic_fixture=False, dataset_revision="a" * 40,
                          primary_sequences=1024, sequence_length=1024, global_batch_size=32,
                          training_chunks_per_seed=64000, scientific_recipe_sha256="c" * 64)
            policy["tokenizer"].update(kind="local_hf_snapshot", revision="b" * 40,
                                       directory=str(root / "not_materialized"), files={"tokenizer.json": "d" * 64})
            self.assertIs(data.validate_policy(policy), policy)
            for key, value in (("dataset_revision", "main"), ("training_chunks_per_seed", 63999)):
                invalid = copy.deepcopy(policy)
                invalid[key] = value
                with self.subTest(key=key), self.assertRaises(ProtocolError):
                    data.validate_policy(invalid)
            with self.assertRaises(ProtocolError):
                data.build_bundle(policy, root / "forbidden", tokenizer=ByteTokenizer())
            self.assertFalse((root / "forbidden").exists())

    def test_wrong_token_ids_rejected_without_completion(self):
        class Bad(ByteTokenizer):
            def encode(self, text):
                return [True, -1, 1000]
        with tempfile.TemporaryDirectory(prefix="mhar-gate-b-token-range-") as directory:
            root = Path(directory)
            policy, _ = corpus_fixture(root)
            with self.assertRaisesRegex(ProtocolError, "invalid token ID"):
                data.build_bundle(policy, root / "bad", tokenizer=Bad())
            self.assertFalse((root / "bad/COMPLETE.json").exists())

    def test_missing_completion_and_tampered_policy_index_or_file_rejected(self):
        with tempfile.TemporaryDirectory(prefix="mhar-gate-b-tamper-") as directory:
            _, _, _, output, _ = build_fixture(Path(directory))
            for relative in ("policy.json", "document-index.sqlite", "training-seed123/tokens.u32le",
                             "primary-reserved/valid-target-mask.u8", "training-seed3407/provenance.jsonl"):
                path = output / relative
                original = path.read_bytes()
                path.write_bytes(original + b"x")
                with self.subTest(file=relative), self.assertRaises((ProtocolError, ValueError)):
                    data.verify_bundle(output)
                path.write_bytes(original)
            (output / "COMPLETE.json").unlink()
            with self.assertRaises(FileNotFoundError):
                data.verify_bundle(output)

    def test_resealed_wrong_run_alias_counts_mask_and_provenance_rejected(self):
        with tempfile.TemporaryDirectory(prefix="mhar-gate-b-reseal-") as directory:
            _, _, _, output, baseline = build_fixture(Path(directory))
            variant = copy.deepcopy(baseline)
            del variant["run_streams"][next(iter(variant["run_streams"]))]
            reseal(output, variant)
            with self.assertRaises(ProtocolError):
                data.verify_bundle(output)
            variant = copy.deepcopy(baseline)
            variant["primary"]["sequences"] = 1
            reseal(output, variant)
            with self.assertRaises(ProtocolError):
                data.verify_bundle(output)
            mask = output / "primary-reserved/valid-target-mask.u8"
            original = mask.read_bytes()
            mask.write_bytes(b"\x00" + original[1:])
            variant = copy.deepcopy(baseline)
            variant["primary"]["files"][mask.name] = data.file_sha256(mask)
            reseal(output, variant)
            with self.assertRaisesRegex(ProtocolError, "invalid target mask"):
                data.verify_bundle(output)
            mask.write_bytes(original)
            provenance = output / "primary-reserved/provenance.jsonl"
            rows = [json.loads(line) for line in provenance.read_text().splitlines()]
            rows[0]["spans"][-1]["raw_token_end"] -= 1
            provenance.write_bytes(b"".join(data.canonical_bytes(row) + b"\n" for row in rows))
            variant = copy.deepcopy(baseline)
            variant["primary"]["files"][provenance.name] = data.file_sha256(provenance)
            reseal(output, variant)
            with self.assertRaises(ProtocolError):
                data.verify_bundle(output)

    def test_parquet_local_reader_matches_jsonl_rows(self):
        import pyarrow as arrow
        import pyarrow.parquet as parquet
        with tempfile.TemporaryDirectory(prefix="mhar-gate-b-parquet-") as directory:
            root = Path(directory)
            policy, documents = corpus_fixture(root)
            path = root / "corpus.parquet"
            parquet.write_table(arrow.Table.from_pylist(documents), path)
            policy["sources"] = [{"path": str(path), "format": "parquet", "sha256": data.file_sha256(path)}]
            self.assertEqual([row for _, row in data.iter_documents(policy)], documents)
            data.build_bundle(policy, root / "bundle", tokenizer=ByteTokenizer())
            data.verify_bundle(root / "bundle")

    def test_cli_verify_only_and_pending_template_cannot_build(self):
        with tempfile.TemporaryDirectory(prefix="mhar-gate-b-cli-") as directory:
            root = Path(directory)
            _, _, _, output, _ = build_fixture(root)
            script = ROOT / "scripts/setup/build_paper_sharing_data.py"
            verified = subprocess.run([sys.executable, str(script), "--verify", str(output)], capture_output=True, text=True)
            self.assertEqual(verified.returncode, 0, verified.stderr)
            self.assertFalse(json.loads(verified.stdout)["real_data_gate_passed"])
            pending = ROOT / "configs/paper_sharing_validation_2k/gate-b-policy.pending.json"
            rejected = subprocess.run([sys.executable, str(script), "--policy", str(pending), "--output", str(root / "forbidden")], capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertFalse((root / "forbidden").exists())


class OfflineTokenizerTests(unittest.TestCase):
    def test_actual_local_tokenizer_snapshot_hashes_eos_and_no_model_download(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        with tempfile.TemporaryDirectory(prefix="mhar-gate-b-tokenizer-") as directory:
            root = Path(directory)
            core = Tokenizer(models.WordLevel({"[UNK]": 0, "[EOS]": 1, "a": 2, "b": 3}, unk_token="[UNK]"))
            core.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="[UNK]", eos_token="[EOS]")
            tokenizer.save_pretrained(root)
            spec = {"kind": "local_hf_snapshot", "directory": str(root), "revision": "local-fixture-tokenizer-v1",
                    "vocab_size": 4, "eos_token_id": 1,
                    "files": {str(path.relative_to(root)): data.file_sha256(path) for path in root.rglob("*") if path.is_file()}}
            adapter = data.LocalTokenizer(spec)
            self.assertEqual(adapter.encode("a b"), [2, 3])
            self.assertEqual(data._checked_tokens(adapter, "a b", spec), [2, 3, 1])
            (root / "tokenizer.json").write_bytes((root / "tokenizer.json").read_bytes() + b" ")
            with self.assertRaisesRegex(ProtocolError, "tokenizer snapshot hash mismatch"):
                adapter.verify()


if __name__ == "__main__":
    unittest.main()
