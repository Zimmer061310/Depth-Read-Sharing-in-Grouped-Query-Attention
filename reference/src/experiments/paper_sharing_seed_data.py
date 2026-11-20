"""Append-only seed-data amendment; no edits to accepted numerical modules.

Build two genuinely new shuffled streams, copy seed123 and the reserved primary
byte-for-byte, independently replay the new streams, and audit offset resume.
This is data preparation, NOT a production launcher or CUDA acceptance waiver.
"""
from array import array
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

from src.experiments import paper_sharing_data as data
from src.experiments.paper_sharing_protocol import ROOT, MODELS, require, sha

SEEDS = (2026, 613, 123)
AMENDMENT = ROOT / 'configs/paper_sharing_validation_2k/seed-amendment-20261018.json'
BASE_POLICY = ROOT / 'configs/paper_sharing_validation_2k/gate-b-policy.frozen.json'
BASE_AUDIT = ROOT / 'docs/validation/paper-sharing-gate-b-production/production-audit-attempt-1.json'
AMENDED_POLICY = ROOT / 'configs/paper_sharing_validation_2k/gate-b-policy.seed-amendment-20261018.json'
AMENDED_DESIGN = ROOT / 'configs/paper_sharing_validation_2k/design.seed-amendment-20261018.json'
NAMES = ('tokens.u32le', 'valid-target-mask.u8', 'provenance.jsonl')


def read(path):
    path = Path(path)
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'regular nonsymlink input required')
    return json.loads(path.read_text())


def rows():
    return [{'id': f'paper-sharing-{m.lower()}-seed{s}-step2000', 'model': m, 'seed': s, 'updates': 2000}
            for s in SEEDS for m in MODELS]


def policy_for(parent_policy, amendment_sha256):
    value = copy.deepcopy(parent_policy)
    value['shuffle_seeds'] = {str(s): s for s in SEEDS}
    value['seed_amendment_sha256'] = sha(amendment_sha256)
    return value


def verify_profile(parent, policy, amendment, *, fixture=False):
    require(amendment['new_seeds'] == list(SEEDS) and amendment['old_seeds'] == [123, 2025, 3407]
            and amendment['models'] == list(MODELS) and amendment['updates_per_run'] == 2000
            and amendment['fresh_scientific_runs_started_at_amendment'] == 0
            and amendment['rescue_seeds_allowed'] is False
            and amendment['primary_and_all_exclusions_unchanged'] is True
            and amendment['recipe_model_optimizer_schedule_supervision_thresholds_unchanged'] is True
            and amendment['paid_execution_authorized'] is False
            and amendment['primary_scoring_authorized'] is False, 'changed amendment scope or authority')
    require(parent['synthetic_fixture'] is fixture and policy == policy_for(parent, data.file_sha256(AMENDMENT)),
            'only the author-selected seed mapping may change')
    if not fixture:
        require(amendment == read(AMENDMENT)
                and data.file_sha256(BASE_POLICY) == amendment['original_policy_sha256']
                and data.file_sha256(BASE_AUDIT) == amendment['original_audit_sha256']
                and data.file_sha256(ROOT / 'configs/paper_sharing_validation_2k/design.json') == amendment['original_design_sha256']
                and parent == read(BASE_POLICY), 'original freeze identity differs')
        expected_design = read(ROOT / 'configs/paper_sharing_validation_2k/design.json')
        expected_design['scientific_design']['fresh_seeds'] = list(SEEDS)
        expected_design['run_matrix'] = rows()
        expected_design['seed_amendment_sha256'] = data.file_sha256(AMENDMENT)
        require(read(AMENDED_DESIGN) == expected_design and read(AMENDED_POLICY) == policy,
                'amended policy/design must be frozen before construction; no other design changes')
    data.validate_policy(parent)  # Existing unchanged validator rehashes source bytes and versions.


def _copy_file(source, target, expected):
    require(not Path(source).is_symlink() and data.file_sha256(source) == sha(expected), 'parent file changed')
    require(not Path(target).exists(), 'refusing overwrite')
    shutil.copyfile(source, target)  # Independent copies, not mutable hard links.
    require(data.file_sha256(target) == expected, 'copy hash mismatch')


def build(parent, output, *, fixture_tokenizer=None, progress=print):
    parent, output = Path(parent), Path(output)
    original = read(parent / 'manifest.json')
    fixture = original['synthetic_fixture']
    require((fixture_tokenizer is not None) is fixture, 'fixture tokenizer must be explicit, never production injected')
    data.verify_bundle(parent, expected_manifest_sha256=data.file_sha256(parent / 'manifest.json'))
    amendment, original_policy = read(AMENDMENT), read(parent / 'policy.json')
    if not fixture:
        require(data.file_sha256(parent / 'manifest.json') == amendment['original_manifest_sha256'], 'wrong parent bundle')
    policy = policy_for(original_policy, data.file_sha256(AMENDMENT))
    verify_profile(original_policy, policy, amendment, fixture=fixture)
    tokenizer = fixture_tokenizer if fixture else data.LocalTokenizer(policy['tokenizer'])
    output.mkdir(exist_ok=False)
    data._exclusive_json(output / 'policy.json', policy)
    data._exclusive_json(output / 'SEED_AMENDMENT.json', amendment)
    _copy_file(parent / 'document-index.sqlite', output / 'document-index.sqlite', original['document_index_sha256'])
    _copy_file(parent / 'exclusions.json', output / 'exclusions.json', data.file_sha256(parent / 'exclusions.json'))
    for name, stream in [('primary-reserved', original['primary']), ('training-seed123', original['training_streams']['123'])]:
        (output / name).mkdir()
        for filename, expected in stream['files'].items():
            _copy_file(parent / name / filename, output / name / filename, expected)
    streams = {'123': copy.deepcopy(original['training_streams']['123'])}
    for seed in SEEDS:
        if seed == 123:
            continue
        progress(f'seed {seed}: generating actual shuffled token/provenance bytes', flush=True)
        directory = output / f'training-seed{seed}'
        directory.mkdir()
        documents = data._shuffled_documents(policy, output / 'document-index.sqlite', seed)
        try:
            streams[str(seed)] = data._write_packed(directory, documents, tokenizer, policy, policy['training_chunks_per_seed'])
        finally:
            documents.close()
        require(not set(streams[str(seed)]['contributing_document_ids']).intersection(
            read(output / 'exclusions.json')['document_ids']), 'excluded training document')
        progress(f'seed {seed}: packed complete', flush=True)
    manifest = copy.deepcopy(original)
    manifest.update(policy_sha256=data.digest(policy), training_streams=streams,
        seed_amendment_sha256=data.file_sha256(AMENDMENT),
        parent_manifest_sha256=data.file_sha256(parent / 'manifest.json'))
    manifest['run_streams'] = {r['id']: {'seed': r['seed'], 'directory': f"training-seed{r['seed']}",
        'tokens_sha256': streams[str(r['seed'])]['files']['tokens.u32le']} for r in rows()}
    verify_profile(original_policy, policy, amendment, fixture=fixture)
    if not fixture:
        tokenizer.verify()
    data._exclusive_json(output / 'manifest.json', manifest)
    verify(output, expected_manifest_sha256=data.file_sha256(output / 'manifest.json'), parent=parent, complete=False)
    data._exclusive_json(output / 'COMPLETE.json', {'manifest_sha256': data.file_sha256(output / 'manifest.json')})
    return manifest


def verify(root, *, expected_manifest_sha256, parent, complete=True):
    root, parent = Path(root), Path(parent)
    require(root.is_dir() and not root.is_symlink(), 'regular bundle directory required')
    require(data.file_sha256(root / 'manifest.json') == sha(expected_manifest_sha256), 'external manifest mismatch')
    if complete:
        require(read(root / 'COMPLETE.json') == {'manifest_sha256': expected_manifest_sha256}, 'incomplete bundle')
    manifest, original = read(root / 'manifest.json'), read(parent / 'manifest.json')
    policy, original_policy = read(root / 'policy.json'), read(parent / 'policy.json')
    require(read(root / 'SEED_AMENDMENT.json') == read(AMENDMENT), 'amendment bytes differ')
    require(manifest.get('synthetic_fixture') is original['synthetic_fixture'], 'fixture promotion refused')
    if not original['synthetic_fixture']:
        require(data.file_sha256(parent / 'manifest.json') == read(AMENDMENT)['original_manifest_sha256'], 'wrong original manifest')
    require(policy == policy_for(original_policy, data.file_sha256(AMENDMENT)), 'unapproved data policy change')
    expected = copy.deepcopy(original)
    expected.update(policy_sha256=data.digest(policy), training_streams=manifest['training_streams'],
        run_streams={r['id']: {'seed': r['seed'], 'directory': f"training-seed{r['seed']}",
            'tokens_sha256': manifest['training_streams'][str(r['seed'])]['files']['tokens.u32le']} for r in rows()},
        seed_amendment_sha256=data.file_sha256(AMENDMENT),
        parent_manifest_sha256=data.file_sha256(parent / 'manifest.json'))
    require(manifest == expected and set(manifest['training_streams']) == {str(s) for s in SEEDS},
            'unexpected manifest changes or extra/missing seed')
    for filename in ('document-index.sqlite', 'exclusions.json'):
        require(data.file_sha256(root / filename) == data.file_sha256(parent / filename), 'original selection/index changed')
    require(manifest['training_streams']['123'] == original['training_streams']['123'], 'seed123 relabeled or changed')
    excluded = set(read(root / 'exclusions.json')['document_ids'])
    require(data.digest(sorted(excluded)) == manifest['exclusion_sha256'], 'exclusion identity mismatch')
    for directory, stream in [('primary-reserved', manifest['primary']), *[
            (f'training-seed{s}', manifest['training_streams'][str(s)]) for s in SEEDS]]:
        path = root / directory
        require(path.is_dir() and not path.is_symlink(), 'symlink stream refused')
        count = policy['primary_sequences'] if directory == 'primary-reserved' else policy['training_chunks_per_seed']
        length = policy['sequence_length']
        require(stream['sequences'] == count and stream['sequence_length'] == length
                and stream['raw_tokens_per_sequence'] == length + 1 and stream['valid_targets'] == count * length
                and stream['input_ids'] == 'raw[:-1]' and stream['targets'] == 'raw[1:]'
                and set(stream['files']) == set(NAMES), 'invalid causal dimensions/inventory')
        for name, expected_sha in stream['files'].items():
            require(not (path / name).is_symlink() and data.file_sha256(path / name) == sha(expected_sha), 'stream file hash mismatch')
        require((path / 'tokens.u32le').stat().st_size == count * (length + 1) * 4
                and (path / 'valid-target-mask.u8').read_bytes() == b'\x01' * (count * length), 'token/mask size or validity mismatch')
        previous = None
        for raw in data._raw_chunks(path / 'tokens.u32le', length, count):
            require(all(t < policy['tokenizer']['vocab_size'] for t in raw)
                    and (previous is None or previous == raw[0]), 'invalid token or lookahead continuity')
            previous = raw[-1]
        seen, count_seen = set(), 0
        with (path / 'provenance.jsonl').open() as handle:
            for line in handle:
                record, position = json.loads(line), 0
                require(record['sequence'] == count_seen, 'sequence order mismatch')
                for span in record['spans']:
                    require(span['raw_token_start'] == position < span['raw_token_end'] <= length + 1
                        and type(span['document_token_start']) is int and span['document_token_start'] >= 0
                        and span['document_token_end'] - span['document_token_start'] == span['raw_token_end'] - position,
                        'invalid provenance coverage/offsets')
                    position = span['raw_token_end']
                    seen.add(sha(span['document_id']))
                require(position == length + 1, 'missing lookahead provenance')
                count_seen += 1
        require(count_seen == count and sorted(seen) == stream['contributing_document_ids']
                and data.digest(sorted(seen)) == stream['contributing_document_ids_sha256'], 'document provenance identity mismatch')
        require(seen <= excluded if directory == 'primary-reserved' else not seen.intersection(excluded), 'document leakage')
    return {'status': 'seed_amended_bundle_verified', 'fixture': original['synthetic_fixture'],
        'manifest_sha256': expected_manifest_sha256, 'seeds': list(SEEDS),
        'primary_and_exclusions_unchanged': True, 'seed123_bytes_unchanged': True,
        'execution_authorized': False, 'ready_for_training': False}


def training_examples(root, seed, *, expected_manifest_sha256, skip_chunks=0):
    """Training-only adapter for exact offset checks; does not authorize a trainer."""
    require(type(seed) is int and seed in SEEDS and type(skip_chunks) is int and skip_chunks >= 0, 'unknown seed/invalid offset')
    root = Path(root)
    require(data.file_sha256(root / 'manifest.json') == sha(expected_manifest_sha256)
        and read(root / 'COMPLETE.json') == {'manifest_sha256': expected_manifest_sha256}, 'unbound/incomplete stream')
    manifest = read(root / 'manifest.json')
    stream = manifest['training_streams'][str(seed)]
    require(skip_chunks <= stream['sequences'], 'resume past end')
    # Hash the selected training stream, not the reserved primary.
    for name, expected in stream['files'].items():
        require(name in NAMES and data.file_sha256(root / f'training-seed{seed}' / name) == sha(expected), 'changed selected stream')
    for index, raw in enumerate(data._raw_chunks(root / f'training-seed{seed}' / 'tokens.u32le',
            stream['sequence_length'], stream['sequences'], skip_chunks), start=skip_chunks):
        yield {'chunk_index': index, 'input_ids': raw[:-1], 'targets': raw[1:],
               'valid_target_mask': [True] * stream['sequence_length']}


def audit(root, *, parent, expected_manifest_sha256, fixture_tokenizer=None, progress=print):
    """Independent retokenization/packing oracle; byte-identical123 reuses its audit."""
    verify(root, expected_manifest_sha256=expected_manifest_sha256, parent=parent)
    root, parent = Path(root), Path(parent)
    manifest, original_policy = read(root / 'manifest.json'), read(parent / 'policy.json')
    policy, amendment = read(root / 'policy.json'), read(AMENDMENT)
    fixture = manifest['synthetic_fixture']
    require((fixture_tokenizer is not None) is fixture, 'explicit fixture tokenizer required')
    verify_profile(original_policy, policy, amendment, fixture=fixture)
    tokenizer = fixture_tokenizer if fixture else data.LocalTokenizer(policy['tokenizer'])
    from scripts.maintenance.audit_paper_sharing_real_data import _compare_repack, _eligible
    from datasets import IterableDataset
    excluded = set(read(root / 'exclusions.json')['document_ids'])
    reports = {}
    for seed in SEEDS:
        stream = manifest['training_streams'][str(seed)]
        if seed != 123 or fixture:
            documents = IterableDataset.from_generator(_eligible, gen_kwargs={'policy': policy, 'excluded': excluded})
            iterator = iter(documents.shuffle(seed=seed, buffer_size=policy['shuffle']['buffer_size']))
            try:
                seen = _compare_repack(root / f'training-seed{seed}', iterator, tokenizer, policy, policy['training_chunks_per_seed'])
            finally:
                if hasattr(iterator, 'close'):
                    iterator.close()
            require(not seen.intersection(excluded) and sorted(seen) == stream['contributing_document_ids'], 'independent replay leakage')
        else:
            old = read(BASE_AUDIT)
            require(old['manifest_sha256'] == manifest['parent_manifest_sha256']
                and old['status'] == 'real_data_audit_passed'
                and old['training_stream_audits']['123']['ordered_tokens_sha256'] == stream['files']['tokens.u32le']
                and old['training_stream_audits']['123']['train_eval_intersection_count'] == 0,
                'unchanged123 lacks bound original real-corpus audit')
        progress(f'seed {seed}: source replay accepted; checking six full resume suffixes', flush=True)
        count, length = stream['sequences'], stream['sequence_length']
        offsets = sorted({0, 1, min(32, count), count // 2, count - 1, count})
        suffixes = []
        for offset in offsets:
            observed, n = hashlib.sha256(), 0
            for example in training_examples(root, seed, expected_manifest_sha256=expected_manifest_sha256, skip_chunks=offset):
                require(example['chunk_index'] == offset + n and example['input_ids'][1:] == example['targets'][:-1]
                        and example['valid_target_mask'] == [True] * length, 'resume order/target/mask mismatch')
                raw = array('I', example['input_ids'] + [example['targets'][-1]])
                if sys.byteorder != 'little':
                    raw.byteswap()
                observed.update(raw.tobytes()); n += 1
            expected = hashlib.sha256()
            with (root / f'training-seed{seed}' / 'tokens.u32le').open('rb') as handle:
                handle.seek(offset * (length + 1) * 4)
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                    expected.update(block)
            require(n == count - offset and observed.hexdigest() == expected.hexdigest(), 'full suffix mismatch')
            suffixes.append({'offset': offset, 'suffix_examples': n, 'suffix_sha256': observed.hexdigest()})
        reports[str(seed)] = {'train_eval_intersection_count': 0, 'all_three_model_aliases_verified': True,
            'source_replay': 'new_independent_retokenization' if seed != 123 or fixture else 'bound_original_audit_byte_identical123',
            'tokens_sha256': stream['files']['tokens.u32le'], 'resume': suffixes}
    verify_profile(original_policy, policy, amendment, fixture=fixture)
    verify(root, expected_manifest_sha256=expected_manifest_sha256, parent=parent)
    return {'status': 'fixture_audit_passed' if fixture else 'real_amended_data_audit_passed',
        'fixture': fixture, 'manifest_sha256': expected_manifest_sha256, 'amendment_sha256': data.file_sha256(AMENDMENT),
        'seeds': list(SEEDS), 'run_matrix': rows(), 'training_stream_audits': reports,
        'primary_and_exclusions_unchanged': True, 'source_bytes_rehashed': True, 'seed123_bytes_unchanged': True,
        'primary_model_scoring_performed': False, 'new_scientific_updates': 0,
        'production_launcher_binding_pending': True, 'ready_for_training': False, 'execution_authorized': False}
