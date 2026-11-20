"""Disposable update mechanics; never score real data or count these as fresh runs."""
import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from src.experiments.paper_sharing_model import PaperSharingForCausalLM, configure, model_fingerprints
from src.experiments.paper_sharing_supervision import native_training_kwargs
from src.experiments.paper_sharing_training import (
    frozen_recipe, production_config, make_optimization, FrozenSchedule, UpdateProgress,
    validate_boundary, complete_update, validate_training_model,
    collate_training_window,
)
from src.training.train_scratch import cosine_with_warmup
from tests.test_paper_sharing_model import native_config, assert_tree_equal


def batch(size=1):
    raw = torch.arange(size * 1025).reshape(size, 1025) % 64
    return raw[:, :-1].clone(), raw[:, 1:].clone(), torch.ones(size, 1024, dtype=torch.bool)


def make(condition="ABBC"):
    torch.manual_seed(123)
    config = native_config(num_hidden_layers=1, max_position_embeddings=2048)
    return PaperSharingForCausalLM(configure(config, condition, test_fixture=True))


class TrainingAdapterCPUChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_frozen_byte_identity_rejects_recipe_change(self):
        self.assertEqual(frozen_recipe()["schedule_steps"], 20000)
        with tempfile.TemporaryDirectory(prefix="mhar-recipe-negative-") as directory:
            path = Path(directory) / "recipe.json"
            path.write_text('{}')
            with patch("src.experiments.paper_sharing_training.RECIPE", path), self.assertRaises(ValueError):
                frozen_recipe()

    def test_production_configs_are_full_size_cuda_intent_and_rng_neutral(self):
        rng = torch.get_rng_state().clone()
        for condition in ("AAAB", "AABC", "ABBC"):
            config = production_config(condition)
            self.assertEqual((config.num_hidden_layers, config.hidden_size, config.intermediate_size,
                              config.vocab_size, config.max_position_embeddings), (36, 1280, 5120, 151936, 2048))
            self.assertEqual(config.paper_sharing_router_backend, "cuda-exact")
            self.assertFalse(config.paper_sharing_test_fixture)
            with torch.device("meta"):
                model = PaperSharingForCausalLM(config)
            self.assertEqual(len(model.model.layers), 36)
            # Structure is possible offline; numerical execution is explicitly refused on meta/CPU.
            with self.assertRaises(ValueError):
                validate_training_model(model)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_actual_model_update_matches_manual_native_recipe_all_conditions(self):
        for condition in ("AAAB", "AABC", "ABBC"):
            with self.subTest(condition=condition):
                model = make(condition)
                manual = copy.deepcopy(model)
                opt, sch = make_optimization(model, local_fixture=True)
                ref_opt, ref_sch = make_optimization(manual, local_fixture=True)
                # Native update zero has LR=0, so also test update one, where parameters change.
                initial = model_fingerprints(model)["checkpoint_state_sha256"]
                progress = UpdateProgress(condition, 123, local_fixture=True)
                for _ in range(2):
                    batches = [batch(), batch()]
                    reference_loss = 0.
                    for x, y, mask in batches:
                        loss = manual(**native_training_kwargs(x, exposed_targets=y, data_validity_mask=mask)).loss / 2
                        loss.backward()
                        reference_loss += loss.detach().item()
                    norm = torch.nn.utils.clip_grad_norm_(manual.parameters(), 1.)
                    ref_opt.step(); ref_sch.step(); ref_opt.zero_grad(set_to_none=True)
                    result = complete_update(model, opt, sch, progress, batches, microbatch=1, accumulation=2)
                    self.assertEqual(result["loss"], reference_loss)
                    self.assertEqual(result["gradient_norm"], norm.item())
                    assert_tree_equal(self, model.state_dict(), manual.state_dict())
                    assert_tree_equal(self, opt.state_dict(), ref_opt.state_dict())
                    assert_tree_equal(self, sch.state_dict(), ref_sch.state_dict())
                self.assertEqual((progress.completed_updates, progress.chunks_consumed), (2, 4))
                self.assertEqual((result["input_positions"], result["supervised_positions"]), (4096, 4092))
                self.assertNotEqual(initial, model_fingerprints(model)["checkpoint_state_sha256"])
                self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_every_schedule_clock_matches_native_twenty_k_not_two_k(self):
        parameter = torch.nn.Parameter(torch.tensor(1.))
        opt = torch.optim.AdamW([parameter], lr=5e-4)
        sch = FrozenSchedule(opt)
        for step in range(2001):
            self.assertEqual(sch.last_epoch, step)
            self.assertEqual(opt.param_groups[0]["lr"], 5e-4 * cosine_with_warmup(step, 1000, 20000, .1))
            if step < 2000:
                opt.step(); sch.step()
        self.assertGreater(opt.param_groups[0]["lr"], .00049)

    def test_exact_eight_by_four_orchestration_and_terminal_stop(self):
        model = make()
        opt, sch = make_optimization(model, local_fixture=True)
        # Establish one disposable parameter state, then stage terminal clocks.
        weight = model.model.layers[0].self_attn.q_proj.weight
        weight.grad = torch.zeros_like(weight)
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        # Advance clocks without a model update: this is a terminal-guard fixture, not 1999 trained steps.
        for _ in range(1998):
            opt.step(); sch.step()
        opt.state[weight]["step"].fill_(1999)
        progress = UpdateProgress("ABBC", 123, completed_updates=1999, chunks_consumed=1999*32, local_fixture=True)
        def fake_forward(**kwargs):
            self.assertIs(kwargs["labels"], kwargs["input_ids"])
            self.assertEqual(kwargs["input_ids"].shape, (4, 1024))
            self.assertFalse(kwargs["use_cache"])
            return SimpleNamespace(loss=model.model.layers[0].self_attn.q_proj.weight.square().mean())
        with patch.object(model, "forward", side_effect=fake_forward) as forward:
            result = complete_update(model, opt, sch, progress, [batch(4) for _ in range(8)])
            self.assertEqual(forward.call_count, 8)
            with self.assertRaises(ValueError):
                complete_update(model, opt, sch, progress, [batch(4) for _ in range(8)])
            self.assertEqual(forward.call_count, 8)
        self.assertEqual((result["completed_updates"], result["chunks_consumed"]), (2000, 64000))
        self.assertEqual((result["input_positions"], result["supervised_positions"]), (65536000, 65472000))

    def test_invalid_identity_counts_partial_gradients_and_mode_rejected(self):
        model = make()
        opt, sch = make_optimization(model, local_fixture=True)
        original = UpdateProgress("ABBC", 123, local_fixture=True)
        for key, value in (("seed", 42), ("seed", True), ("condition", "AAAB"), ("completed_updates", -1),
                           ("completed_updates", 2001), ("chunks_consumed", 1), ("failed", True),
                           ("local_fixture", False)):
            progress = copy.deepcopy(original); setattr(progress, key, value)
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_boundary(model, opt, sch, progress, microbatch=4, accumulation=8)
        next(model.parameters()).grad = torch.zeros_like(next(model.parameters()))
        with self.assertRaises(ValueError):
            validate_boundary(model, opt, sch, original, microbatch=4, accumulation=8)
        opt.zero_grad(set_to_none=True)
        model.eval()
        with self.assertRaises(ValueError):
            validate_boundary(model, opt, sch, original, microbatch=4, accumulation=8)

    def test_optimizer_schedule_and_coverage_changes_rejected(self):
        for key, value in (("lr", .0005), ("betas", (.9, .99)), ("eps", 1e-7),
                           ("weight_decay", 0.), ("fused", True), ("foreach", True)):
            model = make(); opt, sch = make_optimization(model, local_fixture=True)
            opt.param_groups[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_boundary(model, opt, sch, UpdateProgress("ABBC", 123, local_fixture=True), microbatch=4, accumulation=8)
        model = make(); opt, sch = make_optimization(model, local_fixture=True)
        opt.param_groups[0]["params"].pop()
        with self.assertRaises(ValueError):
            validate_boundary(model, opt, sch, UpdateProgress("ABBC", 123, local_fixture=True), microbatch=4, accumulation=8)

    def test_incomplete_or_misaligned_batches_refused_before_forward(self):
        model = make(); opt, sch = make_optimization(model, local_fixture=True)
        progress = UpdateProgress("ABBC", 123, local_fixture=True)
        invalid = batch(); invalid[1][0, 0] = 63
        for batches in ([batch()], [batch(), invalid], [batch(4), batch(4)]):
            with patch.object(model, "forward") as forward, self.assertRaises(ValueError):
                complete_update(model, opt, sch, progress, batches, microbatch=1, accumulation=2)
            forward.assert_not_called()
            self.assertFalse(progress.failed)

    def test_nonfinite_loss_and_gradient_poison_without_optimizer_update(self):
        for kind in ("loss", "gradient"):
            model = make(); opt, sch = make_optimization(model, local_fixture=True)
            progress = UpdateProgress("ABBC", 123, local_fixture=True)
            weight = model.model.layers[0].self_attn.q_proj.weight
            def fake_forward(**kwargs):
                loss = weight.square().mean()
                return SimpleNamespace(loss=loss * float("nan") if kind == "loss" else loss)
            hook = weight.register_hook(lambda gradient: gradient * float("nan")) if kind == "gradient" else None
            try:
                with patch.object(model, "forward", side_effect=fake_forward), patch.object(opt, "step") as step:
                    with self.assertRaises(FloatingPointError):
                        complete_update(model, opt, sch, progress, [batch(), batch()], microbatch=1, accumulation=2)
                    step.assert_not_called()
                    self.assertTrue(progress.failed)
                    self.assertEqual((progress.completed_updates, progress.chunks_consumed, sch.last_epoch), (0, 0, 0))
                    with self.assertRaises(ValueError):
                        complete_update(model, opt, sch, progress, [batch(), batch()], microbatch=1, accumulation=2)
            finally:
                if hook is not None:
                    hook.remove()

    def test_optimizer_exception_cannot_be_silently_retried(self):
        model = make(); opt, sch = make_optimization(model, local_fixture=True)
        progress = UpdateProgress("ABBC", 123, local_fixture=True)
        loss = model.model.layers[0].self_attn.q_proj.weight.square().mean()
        with patch.object(model, "forward", return_value=SimpleNamespace(loss=loss)), patch.object(opt, "step", side_effect=RuntimeError("fixture failure")):
            with self.assertRaises(RuntimeError):
                complete_update(model, opt, sch, progress, [batch()], microbatch=1, accumulation=1)
        self.assertTrue(progress.failed)
        self.assertEqual(asdict(progress)["completed_updates"], 0)

    def test_interrupt_also_poisons_partial_computation(self):
        model = make(); opt, sch = make_optimization(model, local_fixture=True)
        progress = UpdateProgress("ABBC", 123, local_fixture=True)
        with patch.object(model, "forward", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                complete_update(model, opt, sch, progress, [batch()], microbatch=1, accumulation=1)
        self.assertTrue(progress.failed)

    def test_changed_schedule_function_and_optimizer_clocks_refused(self):
        model = make(); opt, sch = make_optimization(model, local_fixture=True)
        progress = UpdateProgress("ABBC", 123, local_fixture=True)
        sch.lr_lambdas = [lambda _: 1.]
        with self.assertRaises(ValueError):
            validate_boundary(model, opt, sch, progress, microbatch=1, accumulation=1)
        opt, sch = make_optimization(model, local_fixture=True)
        weight = model.model.layers[0].self_attn.q_proj.weight
        weight.grad = torch.zeros_like(weight)
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        progress.completed_updates = 1; progress.chunks_consumed = 1
        validate_boundary(model, opt, sch, progress, microbatch=1, accumulation=1)
        opt.state[weight]["step"].fill_(0)
        with self.assertRaises(ValueError):
            validate_boundary(model, opt, sch, progress, microbatch=1, accumulation=1)

    def test_window_rejects_missing_duplicate_bad_types_and_incomplete_stream(self):
        x, y, validity = batch()
        rows = [{"chunk_index": i, "input_ids": x[0].tolist(), "targets": y[0].tolist(),
                 "valid_target_mask": validity[0].tolist()} for i in range(33)]
        stream = iter(rows)
        self.assertEqual(len(collate_training_window(stream, expected_start=0)), 8)
        self.assertEqual(next(stream)["chunk_index"], 32)  # No prefetch.
        for field, value in (("chunk_index", 0), ("chunk_index", 2), ("chunk_index", True),
                             ("input_ids", [True]*1024), ("valid_target_mask", [1]*1024)):
            invalid = copy.deepcopy(rows[:2]); invalid[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                collate_training_window(iter(invalid), expected_start=0, microbatch=1, accumulation=2, local_fixture=True)
        with self.assertRaises(ValueError):
            collate_training_window(iter(rows[:31]), expected_start=0)
        with self.assertRaises(ValueError):
            collate_training_window(iter(rows), expected_start=0, microbatch=1)

    def test_disposable_data_reader_to_real_model_update_and_offset_continuation(self):
        from src.experiments import paper_sharing_data as data
        from tests.test_paper_sharing_data import corpus_fixture, ByteTokenizer
        class SmallVocabularyTokenizer(ByteTokenizer):
            def encode(self, text):
                return [value % 63 for value in text.encode()]
        with tempfile.TemporaryDirectory(prefix="mhar-update-stream-fixture-") as directory:
            root = Path(directory)
            policy, _ = corpus_fixture(root, [{"id": f"doc-{i}", "text": "fixture only "*500} for i in range(30)])
            policy.update(sequence_length=1024, primary_sequences=2, training_chunks_per_seed=4)
            policy["tokenizer"].update(vocab_size=64, eos_token_id=63)
            output = root / "bundle"
            data.build_bundle(policy, output, tokenizer=SmallVocabularyTokenizer())
            manifest_sha = data.file_sha256(output / "manifest.json")
            before = data.file_sha256(output / "training-seed123/tokens.u32le")
            model = make(); opt, sch = make_optimization(model, local_fixture=True)
            progress = UpdateProgress("ABBC", 123, local_fixture=True)
            stream = data.iter_training_examples(output, 123, expected_manifest_sha256=manifest_sha)
            batches = collate_training_window(stream, expected_start=progress.chunks_consumed,
                microbatch=1, accumulation=2, local_fixture=True)
            complete_update(model, opt, sch, progress, batches, microbatch=1, accumulation=2)
            expected = collate_training_window(stream, expected_start=2, microbatch=1, accumulation=2, local_fixture=True)
            resumed = data.iter_training_examples(output, 123, expected_manifest_sha256=manifest_sha, skip_chunks=2)
            actual = collate_training_window(resumed, expected_start=progress.chunks_consumed,
                microbatch=1, accumulation=2, local_fixture=True)
            assert_tree_equal(self, expected, actual)
            complete_update(model, opt, sch, progress, actual, microbatch=1, accumulation=2)
            self.assertEqual((progress.completed_updates, progress.chunks_consumed), (2, 4))
            self.assertEqual(before, data.file_sha256(output / "training-seed123/tokens.u32le"))


if __name__ == "__main__":
    unittest.main()
