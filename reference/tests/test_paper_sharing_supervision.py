"""Disposable CPU-only loss/gradient parity; never score the real primary."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from transformers.loss.loss_utils import ForCausalLMLoss

from src.experiments import paper_sharing_data as data
from src.experiments.paper_sharing_protocol import ProtocolError
from src.experiments.paper_sharing_supervision import (
    load_contract, validate_contract, supervision_mask, native_training_kwargs, native_loss_statistics,
)


class NativeSupervisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.contract = load_contract()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def fixture(self):
        generator = torch.Generator().manual_seed(123)
        raw = torch.randint(17, (2, 1025), generator=generator)
        return raw[:, :-1], raw[:, 1:], torch.ones(2, 1024, dtype=torch.bool), torch.randn(2, 1024, 17, generator=generator)

    def test_author_contract_and_position_counts(self):
        self.assertEqual(self.contract['training_supervised_positions_per_run'], 64000 * 1023)
        self.assertEqual(self.contract['primary_supervised_positions'], 1024 * 1023)
        self.assertFalse(self.contract['primary_scoring_authorized'])

    def test_changed_semantics_identity_and_authority_rejected(self):
        for path, value in [(('supervised_targets_per_example',), 1024), (('sequence_length',), 1024.0),
                            (('training','model_labels'), 'exposed_targets'), (('training','internal_label_shift_count'), 2),
                            (('evaluation','targets'), 'raw[1:]'), (('primary_scoring_authorized',), True),
                            (('bundle_manifest_sha256',), 'a'*64), (('primary_supervised_positions',), 1048576)]:
            contract = copy.deepcopy(self.contract)
            target = contract
            for key in path[:-1]: target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ProtocolError): validate_contract(contract)

    def test_mask_is_derived_without_mutating_data_validity(self):
        _, _, validity, _ = self.fixture()
        result = supervision_mask(validity)
        self.assertTrue(validity.all())
        self.assertTrue(result[:, :-1].all())
        self.assertFalse(result[:, -1].any())
        self.assertEqual(result.sum().item(), 2046)

    def test_invalid_masks_rejected(self):
        for mask in [torch.ones(1,1023,dtype=torch.bool), torch.ones(1,1024), torch.ones(0,1024,dtype=torch.bool),
                     torch.zeros(1,1024,dtype=torch.bool)]:
            with self.assertRaises(ProtocolError): supervision_mask(mask)

    def test_training_labels_are_unshifted_native_input(self):
        inputs, targets, validity, _ = self.fixture()
        kwargs = native_training_kwargs(inputs, exposed_targets=targets, data_validity_mask=validity)
        self.assertIs(kwargs['labels'], inputs)
        self.assertIs(kwargs['input_ids'], inputs)
        self.assertEqual(set(kwargs), {'input_ids','labels','use_cache'})
        self.assertFalse(kwargs['use_cache'])
        self.assertFalse(torch.equal(kwargs['labels'], targets))

    def test_malformed_input_or_target_alignment_rejected(self):
        inputs, targets, validity, _ = self.fixture()
        broken = targets.clone(); broken[:, 0] = (broken[:, 0]+1) % 17
        negative = inputs.clone(); negative[:, 0] = -1
        for x, y in [(inputs.float(),targets), (negative,targets), (inputs,broken), (inputs,targets.float())]:
            with self.assertRaises(ProtocolError): native_training_kwargs(x, exposed_targets=y, data_validity_mask=validity)

    def test_native_loss_and_logit_gradient_parity(self):
        inputs, _, validity, original = self.fixture()
        native_logits = original.clone().requires_grad_()
        explicit_logits = original.clone().requires_grad_()
        native_loss = ForCausalLMLoss(native_logits, inputs, 17)
        stats = native_loss_statistics(explicit_logits, inputs, validity)
        tolerance = self.contract['local_fp32_parity']
        torch.testing.assert_close(stats['loss'],native_loss,atol=tolerance['loss_atol'],rtol=tolerance['loss_rtol'])
        native_loss.backward(); stats['loss'].backward()
        torch.testing.assert_close(explicit_logits.grad,native_logits.grad,
            atol=tolerance['logit_gradient_atol'],rtol=tolerance['logit_gradient_rtol'])
        self.assertEqual(torch.count_nonzero(explicit_logits.grad[:, -1]).item(),0)

    def test_ignored_final_logit_and_lookahead_cannot_change_loss(self):
        inputs, targets, validity, logits = self.fixture()
        base = native_loss_statistics(logits,inputs,validity)['loss']
        changed = logits.clone(); changed[:,-1] = torch.arange(17)*100
        self.assertTrue(torch.equal(base,native_loss_statistics(changed,inputs,validity)['loss']))
        targets = targets.clone(); targets[:,-1] = (targets[:,-1]+1)%17
        kwargs = native_training_kwargs(inputs,exposed_targets=targets,data_validity_mask=validity)
        self.assertTrue(torch.equal(kwargs['labels'], inputs))
        altered = inputs.clone(); altered[:,1] = (altered[:,1]+1)%17
        self.assertFalse(torch.equal(base,native_loss_statistics(logits,altered,validity)['loss']))

    def test_statistics_match_actual_historical_evaluation_function(self):
        from src.experiments.experiment23_evaluation import sequence_metrics
        inputs, _, validity, logits = self.fixture()
        class DisposableLogitOracle:
            offset = 0
            def __call__(self, tokens, **kwargs):
                start = self.offset; self.offset += len(tokens)
                return SimpleNamespace(logits=logits[start:self.offset])
        old = sequence_metrics(DisposableLogitOracle(),inputs,batch_size=1,device='cpu')
        new = native_loss_statistics(logits,inputs,validity)
        self.assertEqual(old['sequence_total_losses'],new['sequence_total_losses'].tolist())
        self.assertEqual(old['sequence_valid_tokens'],[1023,1023])
        self.assertEqual(old['loss'],new['token_weighted_loss'].item())

    def test_all_three_real_model_classes_use_native_loss_on_disposable_tokens(self):
        from tests.test_paper_sharing_model import native_config
        from src.experiments.paper_sharing_model import configure, PaperSharingForCausalLM
        inputs = torch.arange(1024).remainder(64).reshape(1,-1)
        targets = torch.cat([inputs[:,1:],torch.tensor([[7]])],dim=1)
        validity = torch.ones_like(inputs,dtype=torch.bool)
        for condition in ('AAAB','AABC','ABBC'):
            with self.subTest(condition=condition):
                torch.manual_seed(123)
                config = configure(native_config(num_hidden_layers=1,max_position_embeddings=2048),condition,test_fixture=True)
                model = PaperSharingForCausalLM(config).cpu().eval()
                output = model(**native_training_kwargs(inputs,exposed_targets=targets,data_validity_mask=validity))
                stats = native_loss_statistics(output.logits,inputs,validity)
                torch.testing.assert_close(output.loss,stats['loss'],atol=1e-6,rtol=1e-6)
                native_grad = torch.autograd.grad(output.loss,output.logits,retain_graph=True)[0]
                explicit_grad = torch.autograd.grad(stats['loss'],output.logits)[0]
                torch.testing.assert_close(native_grad,explicit_grad,atol=1e-7,rtol=1e-6)
                self.assertEqual(torch.count_nonzero(native_grad[:,-1]).item(),0)

    def test_existing_stream_reader_integrates_without_changing_stored_masks(self):
        from tests.test_paper_sharing_data import corpus_fixture, ByteTokenizer
        with tempfile.TemporaryDirectory(prefix='mhar-supervision-') as directory:
            root = Path(directory)
            documents = [{'id':f'doc-{i}','text':'hello world '*500} for i in range(30)]
            policy,_ = corpus_fixture(root,documents)
            policy.update(sequence_length=1024,primary_sequences=2,training_chunks_per_seed=3)
            output=root/'bundle'; data.build_bundle(policy,output,tokenizer=ByteTokenizer())
            before=data.file_sha256(output/'training-seed123/valid-target-mask.u8')
            rows=list(data.iter_training_examples(output,123,expected_manifest_sha256=data.file_sha256(output/'manifest.json')))
            x=torch.tensor([row['input_ids'] for row in rows]); y=torch.tensor([row['targets'] for row in rows])
            mask=torch.tensor([row['valid_target_mask'] for row in rows])
            kwargs=native_training_kwargs(x,exposed_targets=y,data_validity_mask=mask)
            self.assertEqual(supervision_mask(mask).sum().item(),3*1023)
            self.assertTrue(torch.equal(kwargs['labels'],x))
            self.assertEqual(before,data.file_sha256(output/'training-seed123/valid-target-mask.u8'))

    def test_nonfinite_logits_and_vocabulary_mismatch_rejected(self):
        inputs,_,validity,logits=self.fixture()
        logits[0,0,0]=float('nan')
        with self.assertRaises(ProtocolError): native_loss_statistics(logits,inputs,validity)
        with self.assertRaises(ProtocolError): native_loss_statistics(torch.zeros(2,1024,2),inputs,validity)


if __name__ == '__main__':
    unittest.main()
