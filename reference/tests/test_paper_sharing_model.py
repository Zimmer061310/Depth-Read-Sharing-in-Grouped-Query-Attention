"""Real classes/production Q width, reduced depth/FFN/vocab, CPU FP32 only.

Disposable random token IDs are debugging fixtures, NOT the fresh primary
artifact or a training/evaluation result. Optimizer steps here test mechanics.
"""
import copy
import json
from pathlib import Path
import tempfile
from types import MethodType
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from src.attention_residuals.modeling_qwen3_attnres import (
    Qwen3AttnResConfig, Qwen3AttnResForCausalLM, _supports_shared_fused_mhar_context,
)
from src.experiments.experiment23_route_topology import Experiment23Attention
from src.experiments.paper_sharing_model import (
    PaperSharingForCausalLM, PaperSharingAttention, configure, model_fingerprints,
    validate_config, validate_optimizer_coverage, load_local_model_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "configs/paper_sharing_validation_2k/gate-a-local-contract.json").read_text())
CONDITIONS = ("AAAB", "AABC", "ABBC")


def native_config(**overrides):
    settings = dict(vocab_size=64, hidden_size=1280, num_hidden_layers=2,
        num_attention_heads=16, num_key_value_heads=8, head_dim=80, intermediate_size=128,
        max_position_embeddings=64, tie_word_embeddings=True, rms_norm_eps=1e-6,
        attention_dropout=0.0, attnres_mode="full_mh", attnres_num_heads=8,
        attnres_gate_type="bias", use_cache=False, _attn_implementation="sdpa")
    settings.update(overrides)
    return Qwen3AttnResConfig(**settings)


def make(condition, seed=123):
    torch.manual_seed(seed)
    return PaperSharingForCausalLM(configure(native_config(), condition, test_fixture=True)).cpu()


def headwise_reference(attention, inputs):
    # Independent 16-head reference, not the implementation's two-GEMM reshape.
    physical = []
    q1, q2, k, v = attention._experiment23_routes
    for head in range(16):
        route = q1 if head % 2 else q2
        physical.append(F.linear(inputs[route], attention.q_proj.weight[head * 80:(head + 1) * 80]))
    query = torch.stack(physical, dim=-2).reshape(*inputs["A"].shape[:-1], 8, 2, 80)
    return query, F.linear(inputs[k], attention.k_proj.weight), F.linear(inputs[v], attention.v_proj.weight)


def assert_tree_equal(test, left, right):
    test.assertEqual(type(left), type(right))
    if isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            assert_tree_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_tree_equal(test, a, b)
    else:
        test.assertEqual(left, right)


class RealClassCPUChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.models = {condition: make(condition).eval() for condition in CONDITIONS}

    @classmethod
    def tearDownClass(cls):
        del cls.models
        torch.set_num_threads(cls.previous_threads)

    def test_explicit_fixture_and_config_guards(self):
        config = native_config()
        original = config.to_dict()
        prepared = configure(config, "ABBC", test_fixture=True)
        self.assertEqual(config.to_dict(), original)
        self.assertEqual(validate_config(prepared), ("ABBC", "canonical"))
        for key, value in (("paper_sharing_test_fixture", False), ("paper_sharing_query_execution", "packed"),
                           ("hidden_size", 32), ("head_dim", 2), ("experiment24_variant", "global-aabc"),
                           ("experiment19_role_router", True)):
            invalid = copy.deepcopy(prepared)
            setattr(invalid, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(invalid)
        with self.assertRaises(ValueError):
            configure(prepared, "AAAB", test_fixture=True)

    def test_full_36_layer_structure_on_meta_without_allocating_model_weights(self):
        for condition in CONDITIONS:
            config = native_config(num_hidden_layers=36, intermediate_size=5120, vocab_size=151936,
                                   max_position_embeddings=32768)
            with torch.device("meta"):
                model = PaperSharingForCausalLM(configure(config, condition))
            self.assertFalse(model.config.paper_sharing_test_fixture)
            self.assertEqual(len(model.model.layers), 36)
            self.assertTrue(all(parameter.is_meta for parameter in model.parameters()))
            for layer in model.model.layers:
                self.assertIs(type(layer.self_attn), PaperSharingAttention)
                self.assertEqual(layer.self_attn.q_proj.weight.shape, (1280, 1280))
                self.assertEqual(layer.self_attn.k_proj.weight.shape, (640, 1280))
                self.assertEqual(layer.self_attn._experiment23_routes, condition)
                self.assertEqual(layer.self_attn._experiment23_projection, "canonical")
                self.assertEqual(hasattr(layer, "third_attn_res_proj"), condition != "AAAB")

    def test_all_three_execute_exactly_two_production_shape_q_gemms(self):
        original = F.linear
        for condition, model in self.models.items():
            attention = model.model.layers[0].self_attn
            inputs = {route: torch.randn(1, 3, 1280) for route in set(condition)}
            with patch("torch.nn.functional.linear", wraps=original) as linear:
                query, key, value = attention.project_qkv(inputs)
            self.assertEqual([tuple(call.args[1].shape) for call in linear.call_args_list],
                             [(640, 1280)] * 4, condition)
            self.assertEqual(query.shape, (1, 3, 8, 2, 80))
            self.assertEqual(key.shape, (1, 3, 640))
            self.assertEqual(value.shape, key.shape)
            self.assertFalse(_supports_shared_fused_mhar_context(model.config))

    def test_headwise_forward_and_input_weight_backward_reference(self):
        for condition, model in self.models.items():
            actual = copy.deepcopy(model.model.layers[0].self_attn)
            reference = copy.deepcopy(actual)
            torch.manual_seed(17)
            inputs = {route: torch.randn(1, 3, 1280, requires_grad=True) for route in sorted(set(condition))}
            other = {name: tensor.detach().clone().requires_grad_(True) for name, tensor in inputs.items()}
            result = actual.project_qkv(inputs)
            expected = headwise_reference(reference, other)
            upstream = tuple(torch.randn_like(value) * .1 for value in result)
            for a, b in zip(result, expected):
                torch.testing.assert_close(a, b, **CONTRACT["fp32_projection_forward"])
            actual_targets = (*inputs.values(), actual.q_proj.weight, actual.k_proj.weight, actual.v_proj.weight)
            reference_targets = (*other.values(), reference.q_proj.weight, reference.k_proj.weight, reference.v_proj.weight)
            grads = torch.autograd.grad(result, actual_targets, upstream)
            ref_grads = torch.autograd.grad(expected, reference_targets, upstream)
            for a, b in zip(grads, ref_grads):
                torch.testing.assert_close(a, b, **CONTRACT["fp32_projection_backward"])

    def test_intended_and_forbidden_consumer_input_influence(self):
        for condition, model in self.models.items():
            attention = model.model.layers[0].self_attn
            inputs = {route: torch.randn(1, 2, 1280, requires_grad=True) for route in sorted(set(condition))}
            query, key, value = attention.project_qkv(inputs)
            for output, source in ((query[..., 1, :], condition[0]), (query[..., 0, :], condition[1]),
                                   (key, condition[2]), (value, condition[3])):
                grads = torch.autograd.grad(output.square().sum(), tuple(inputs.values()), retain_graph=True, allow_unused=True)
                for route, grad in zip(inputs, grads):
                    if route == source:
                        self.assertIsNotNone(grad)
                        self.assertGreater(float(grad.abs().sum()), 0)
                    else:
                        self.assertTrue(grad is None or torch.count_nonzero(grad) == 0)

    def test_equal_route_q_outputs_bit_identical(self):
        inputs = {"A": torch.randn(1, 3, 1280), "C": torch.randn(1, 3, 1280)}
        inputs["B"] = inputs["A"].clone()
        a = self.models["AABC"].model.layers[0].self_attn.project_qkv(inputs)[0]
        b = self.models["ABBC"].model.layers[0].self_attn.project_qkv(inputs)[0]
        torch.testing.assert_close(a, b, **CONTRACT["equal_route_forward"])

    def test_physical_head_order_with_distinguishable_routes(self):
        attention = copy.deepcopy(self.models["ABBC"].model.layers[0].self_attn)
        with torch.no_grad():
            attention.q_proj.weight.zero_()
            attention.q_proj.weight[:, 0] = torch.arange(1280, dtype=torch.float32) + 1
        inputs = {route: torch.zeros(1, 1, 1280) for route in "ABC"}
        inputs["A"][..., 0] = 2
        inputs["B"][..., 0] = 3
        query = attention.project_qkv(inputs)[0].reshape(16, 80)
        for head in range(16):
            expected = (torch.arange(head * 80, (head + 1) * 80) + 1) * (2 if head % 2 else 3)
            self.assertTrue(torch.equal(query[head], expected.float()))

    def test_common_initial_weights_routes_and_rng_match_three_fresh_seeds(self):
        for seed in (123, 2025, 3407):
            torch.manual_seed(seed)
            native = Qwen3AttnResForCausalLM(native_config())
            expected_rng = torch.get_rng_state().clone()
            common_state = native.state_dict()
            fingerprints = {}
            for condition in CONDITIONS:
                model = make(condition, seed)
                self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()), (seed, condition))
                for name, value in common_state.items():
                    self.assertTrue(torch.equal(value, model.state_dict()[name]), (seed, condition, name))
                fingerprints[condition] = model_fingerprints(model)
                self.assertFalse(any("dummy" in name for name, _ in model.named_parameters()))
                del model
            for field in ("common_parameters_sha256", "q_slot_initial_parameters_sha256"):
                self.assertEqual(fingerprints["AAAB"][field], fingerprints["AABC"][field])
                self.assertEqual(fingerprints["AABC"][field], fingerprints["ABBC"][field])
            self.assertEqual(fingerprints["AABC"]["attention_route_parameters_sha256"],
                             fingerprints["ABBC"]["attention_route_parameters_sha256"])
            self.assertEqual(fingerprints["AABC"]["nominal_parameters"] - fingerprints["AAAB"]["nominal_parameters"], 2 * 1280)
            self.assertNotEqual(fingerprints["AAAB"]["q_slot_initial_parameters_sha256"]["Q1"],
                                fingerprints["AAAB"]["q_slot_initial_parameters_sha256"]["Q2"])

    def test_norm_rope_and_full_logits_match_headwise_reference(self):
        tokens = torch.arange(4).reshape(1, 4)
        for condition, model in self.models.items():
            reference = copy.deepcopy(model)
            for layer in reference.model.layers:
                layer.self_attn.project_qkv = MethodType(headwise_reference, layer.self_attn)
            with torch.no_grad():
                torch.testing.assert_close(model(tokens).logits, reference(tokens).logits,
                                           **CONTRACT["fp32_projection_forward"])

    def test_full_model_route_gradients_checkpointing_and_q_row_coverage(self):
        tokens = torch.arange(4).reshape(1, 4)
        for condition, template in self.models.items():
            plain = copy.deepcopy(template).train()
            checked = copy.deepcopy(plain)
            checked.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            for model in (plain, checked):
                optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
                validate_optimizer_coverage(model, optimizer)
                model(tokens, labels=tokens).loss.backward()
                layer = model.model.layers[1]
                routes = [layer.attn_res_proj, layer.local_attn_res_proj]
                if condition != "AAAB":
                    routes.append(layer.third_attn_res_proj)
                for projection in routes:
                    self.assertGreater(float(projection.weight.grad.abs().sum()), 0)
                q_grad = layer.self_attn.q_proj.weight.grad.reshape(8, 2, 80, 1280)
                for slot in (0, 1):
                    self.assertGreater(float(q_grad[:, slot].abs().sum()), 0)
            for (name, left), (_, right) in zip(plain.named_parameters(), checked.named_parameters()):
                self.assertIsNotNone(left.grad, name)
                self.assertTrue(torch.equal(left.grad, right.grad), name)

    def test_optimizer_rejects_missing_and_duplicate_parameter_coverage(self):
        model = self.models["AAAB"]
        optimizer = torch.optim.AdamW(model.parameters())
        validate_optimizer_coverage(model, optimizer)
        parameter = optimizer.param_groups[0]["params"].pop()
        with self.assertRaises(ValueError):
            validate_optimizer_coverage(model, optimizer)
        optimizer.param_groups[0]["params"].extend([parameter, parameter])
        with self.assertRaises(ValueError):
            validate_optimizer_coverage(model, optimizer)

    def test_assignment_projection_and_dormant_c_tampering_rejected(self):
        for field, value in (("_experiment23_projection", "packed"), ("_experiment23_routes", "AAAB")):
            attention = copy.deepcopy(self.models["ABBC"].model.layers[0].self_attn)
            setattr(attention, field, value)
            with self.assertRaises(ValueError):
                attention.project_qkv({role: torch.zeros(1, 1, 1280) for role in "ABC"})
        model = copy.deepcopy(self.models["AAAB"])
        model.model.layers[0].third_attn_res_proj = copy.deepcopy(model.model.layers[0].attn_res_proj)
        with self.assertRaises(ValueError):
            model_fingerprints(model)

    def test_model_checkpoint_roundtrip_all_conditions(self):
        tokens = torch.arange(4).reshape(1, 4)
        for condition, model in self.models.items():
            with tempfile.TemporaryDirectory(prefix="mhar-paper-gate-a-") as directory:
                model.save_pretrained(directory)
                restored = load_local_model_checkpoint(directory, expected_condition=condition,
                    expected_state_sha256=model_fingerprints(model)["checkpoint_state_sha256"]).eval()
                self.assertEqual(restored.config.paper_sharing_condition, condition)
                self.assertEqual(model_fingerprints(model), model_fingerprints(restored))
                with torch.no_grad():
                    torch.testing.assert_close(model(tokens).logits, restored(tokens).logits, **CONTRACT["checkpoint_roundtrip"])

    def test_strict_local_reload_rejects_wrong_condition_and_state_identity(self):
        model = self.models["AAAB"]
        digest = model_fingerprints(model)["checkpoint_state_sha256"]
        with tempfile.TemporaryDirectory(prefix="mhar-paper-gate-a-reject-") as directory:
            model.save_pretrained(directory)
            for condition, expected in (("ABBC", digest), ("AAAB", "0" * 64), ("AAAB", "short")):
                with self.subTest(condition=condition, expected=expected), self.assertRaises(ValueError):
                    load_local_model_checkpoint(directory, expected_condition=condition, expected_state_sha256=expected)

    def test_strict_local_reload_rejects_actual_missing_weight(self):
        from safetensors.torch import load_file, save_file
        model = self.models["AAAB"]
        digest = model_fingerprints(model)["checkpoint_state_sha256"]
        with tempfile.TemporaryDirectory(prefix="mhar-paper-gate-a-incomplete-") as directory:
            model.save_pretrained(directory)
            path = Path(directory) / "model.safetensors"
            tensors = load_file(str(path))
            del tensors["model.layers.0.self_attn.q_proj.weight"]
            save_file(tensors, str(path), metadata={"format": "pt"})
            with self.assertRaisesRegex(ValueError, "incomplete or mismatched checkpoint"):
                load_local_model_checkpoint(directory, expected_condition="AAAB", expected_state_sha256=digest)

    def test_bf16_checkpoint_storage_dtype_preserved_without_bf16_execution(self):
        # Storage identity only: no BF16 forward/backward or CUDA gate is run.
        model = copy.deepcopy(self.models["ABBC"]).to(dtype=torch.bfloat16)
        expected = model_fingerprints(model)
        with tempfile.TemporaryDirectory(prefix="mhar-paper-gate-a-bf16-storage-") as directory:
            model.save_pretrained(directory)
            restored = load_local_model_checkpoint(directory, expected_condition="ABBC",
                expected_state_sha256=expected["checkpoint_state_sha256"])
            self.assertEqual(restored.model.layers[0].self_attn.q_proj.weight.dtype, torch.bfloat16)
            self.assertEqual(model_fingerprints(restored), expected)

    def test_runtime_backend_intervention_and_layer_weight_tying_rejected(self):
        model = copy.deepcopy(self.models["ABBC"])
        model.model.layers[0]._experiment24_intervention = "collapse-c-to-a"
        with self.assertRaises(ValueError):
            model_fingerprints(model)
        model.model.layers[0]._experiment24_intervention = None
        model.model.layers[0]._experiment24_backend = "tiled-r2-exact"
        with self.assertRaises(ValueError):
            model_fingerprints(model)
        model.model.layers[0]._experiment24_backend = "canonical"
        model.model.layers[1].self_attn.q_proj.weight = model.model.layers[0].self_attn.q_proj.weight
        with self.assertRaises(ValueError):
            model_fingerprints(model)

    def test_full_optimizer_scheduler_rng_and_next_update_roundtrip(self):
        from src.training.train_scratch import save_training_checkpoint, load_training_state
        from tests.test_train_resume import FakeTokenizer
        model = copy.deepcopy(self.models["ABBC"]).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(.9, .95), weight_decay=.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)

        def update(net, opt, sch):
            tokens = torch.randint(0, 64, (1, 4))
            net(tokens, labels=tokens).loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            opt.step()
            sch.step()
            opt.zero_grad(set_to_none=True)
            return tokens

        update(model, optimizer, scheduler)
        with tempfile.TemporaryDirectory(prefix="mhar-paper-gate-a-resume-") as directory:
            checkpoint = save_training_checkpoint(model=model, tokenizer=FakeTokenizer(), optimizer=optimizer,
                scheduler=scheduler, global_step=1, chunks_consumed=1, run_identity={"test_fixture": True},
                out_dir=directory, keep_last=1, wandb_run_id="cpu-test-only", elapsed_training_seconds=0.)
            expected_batch = update(model, optimizer, scheduler)
            expected_rng = torch.get_rng_state().clone()
            restored = PaperSharingForCausalLM.from_pretrained(checkpoint, attn_implementation="sdpa").train()
            state = load_training_state(checkpoint)
            opt = torch.optim.AdamW(restored.parameters(), lr=5e-4, betas=(.9, .95), weight_decay=.1)
            sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.)
            opt.load_state_dict(state["optimizer"])
            sch.load_state_dict(state["scheduler"])
            validate_optimizer_coverage(restored, opt)
            torch.set_rng_state(state["cpu_rng_state"])
            self.assertTrue(torch.equal(update(restored, opt, sch), expected_batch))
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
            assert_tree_equal(self, model.state_dict(), restored.state_dict())
            assert_tree_equal(self, optimizer.state_dict(), opt.state_dict())
            assert_tree_equal(self, scheduler.state_dict(), sch.state_dict())

    def test_archival_projection_class_and_packed_behavior_unchanged(self):
        attention = copy.deepcopy(self.models["AABC"].model.layers[0].self_attn)
        attention.__class__ = Experiment23Attention
        attention._experiment23_projection = "packed"
        with patch("torch.nn.functional.linear", wraps=F.linear) as linear:
            attention.project_qkv({role: torch.zeros(1, 1, 1280) for role in "ABC"})
        self.assertEqual([call.args[1].shape[0] for call in linear.call_args_list], [1280, 640, 640])
        self.assertIsNot(type(self.models["AABC"].model.layers[0].self_attn), Experiment23Attention)


if __name__ == "__main__":
    unittest.main()
