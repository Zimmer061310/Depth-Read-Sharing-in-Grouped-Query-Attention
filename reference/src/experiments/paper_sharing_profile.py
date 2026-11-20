"""Explicit, scoped seed profile. No environment switches or process-global patching.

Legacy calls keep the legacy matrix. CLI activation must match the externally
hashed execution lock; activation alone is neither paid nor scoring authority.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import copy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT/'configs/paper_sharing_validation_2k/seed-launch-profile.v1.json'
_ACTIVE = ContextVar('explicit_paper_seed_profile', default=None)


def _require(value, message):
    if not value:
        raise ValueError(message)


def file_sha(path):
    path = Path(path)
    _require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'regular profile input required')
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def current():
    return copy.deepcopy(_ACTIVE.get())


def profile_sha():
    value = _ACTIVE.get()
    return value['_sha256'] if value else None


def seeds(legacy):
    value = _ACTIVE.get()
    return tuple(value['seeds']) if value else legacy


def design_path(legacy):
    value = _ACTIVE.get()
    return ROOT/value['design'] if value else legacy


def manifest_sha(legacy):
    value = _ACTIVE.get()
    return value['manifest_sha256'] if value else legacy


def source_files():
    value = _ACTIVE.get()
    if value is None:
        return ()
    return (str(PROFILE.relative_to(ROOT)), value['amendment'], value['design'], value['policy'],
        'src/experiments/paper_sharing_profile.py', 'src/experiments/paper_sharing_source_scope.py',
        'src/experiments/paper_sharing_backup.py', 'scripts/maintenance/relay_paper_sharing_backup.py')


def validate_profile(expected_sha):
    _require(file_sha(PROFILE) == expected_sha, 'explicit external seed-profile SHA required')
    value = read(PROFILE)
    _require(value['scope']=='explicit_seed_amended_scientific_profile_v1'
        and value['seeds']==[2026,613,123] and all(type(x) is int for x in value['seeds'])
        and value['models']==['AAAB','AABC','ABBC'] and type(value['updates']) is int and value['updates']==2000
        and value['execution_authorized'] is False and value['primary_scoring_authorized'] is False,
        'profile is not the author seed-only amendment')
    for field in ('amendment','design','policy','supervision'):
        path = Path(value[field])
        _require(not path.is_absolute() and '..' not in path.parts
                 and file_sha(ROOT/path)==value[field+'_sha256'], 'frozen profile component differs: '+field)
    amendment=read(ROOT/value['amendment'])
    _require(amendment['new_seeds']==value['seeds'] and amendment['fresh_scientific_runs_started_at_amendment']==0
        and amendment['primary_and_all_exclusions_unchanged'] is True
        and amendment['recipe_model_optimizer_schedule_supervision_thresholds_unchanged'] is True,
        'seed amendment changed other scientific decisions')
    base=ROOT/'results/paper-sharing-validation/seed-amendment-20261018'
    ledger_name=str((base/'SHA256SUMS.json').relative_to(ROOT))
    blob=subprocess.check_output(['git','show',value['data_publication_commit']+':'+ledger_name],cwd=ROOT)
    _require(hashlib.sha256(blob).hexdigest()==value['data_publication_ledger_sha256'], 'original data ledger differs')
    ledger=json.loads(blob)
    for name in ('AUDIT_ATTEMPT_1.json','FREEZE.json','TRANSFER_REQUEST.json','BACKUP_ACK.json'):
        path=base/name
        _require(file_sha(path)==ledger[str(path.relative_to(ROOT))], 'audited raw data evidence differs: '+name)
    audit,freeze,request,ack=(read(base/name) for name in ('AUDIT_ATTEMPT_1.json','FREEZE.json','TRANSFER_REQUEST.json','BACKUP_ACK.json'))
    _require(file_sha(base/'AUDIT_ATTEMPT_1.json')==value['audit_sha256']
        and audit['status']=='real_amended_data_audit_passed' and audit['fixture'] is False
        and audit['manifest_sha256']==freeze['manifest_sha256']==value['manifest_sha256']
        and audit['seeds']==value['seeds'] and audit['primary_model_scoring_performed'] is False,
        'actual amended data audit missing')
    for seed in value['seeds']:
        row=audit['training_stream_audits'][str(seed)]
        _require(row['train_eval_intersection_count']==0 and row['all_three_model_aliases_verified'] is True
            and {x['offset'] for x in row['resume']}=={0,1,32,32000,63999,64000}, 'amended stream replay/resume incomplete')
    _require(ack['request_sha256']==file_sha(base/'TRANSFER_REQUEST.json')
        and ack['receiver_host']==request['backup_host']!=request['source_host']
        and ack['verified_files']==len(request['files'])==30
        and ack['large_checkpoints_backed_up_outside'] is False, 'data-only outside backup evidence differs')
    value['_sha256']=expected_sha
    return value


@contextmanager
def activate(expected_sha=None):
    value=validate_profile(expected_sha) if expected_sha is not None else None
    existing=_ACTIVE.get()
    _require(existing is None or value==existing, 'nested profile switch refused')
    token=_ACTIVE.set(value)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def require_lock(lock):
    _require(lock.get('seed_profile_sha256')==profile_sha(), 'lock/explicit seed-profile context mismatch')


def verify_bundle_binding(policy, manifest_sha256):
    value=_ACTIVE.get()
    _require(value is not None and manifest_sha256==value['manifest_sha256']
        and file_sha(ROOT/value['policy'])==value['policy_sha256']
        and policy==read(ROOT/value['policy']), 'unfrozen amended production data')


def same_runtime_on_clone(observed, tested):
    _require(set(observed)==set(tested), 'runtime field inventory differs')
    _require(all(observed[k]==tested[k] for k in tested if k!='devices'),
             'software/backend/driver differs; no silent evidence portability')
    _require(len(observed['devices'])==len(tested['devices'])==1, 'exactly one observed GPU required')
    a,b=observed['devices'][0],tested['devices'][0]
    _require(set(a)==set(b) and all(a[k]==b[k] for k in b if k not in ('uuid','uuid_native')),
        'tested5090 type/capability required')
    for device in (a,b):
        if 'uuid_native' in device:
            _require(device['uuid'].removeprefix('GPU-')==device['uuid_native'].removeprefix('GPU-'),
                'observed device UUID representations disagree')
