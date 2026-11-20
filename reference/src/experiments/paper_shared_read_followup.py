"""New follow-up contract and bounded arithmetic; no Torch, network or launcher.

The previous study is immutable. New IDs and fixed contrasts cannot become
rescue seeds, primary rescores, formal equivalence or automatic paid authority.
"""
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT/'configs/paper_shared_read_followup/design.v1.json'
DESIGN_SHA256 = '08ded0177f6bbc8b5f1fac135030fde8c0d3f6fa67c5ed5857b4fc9096e664bf'
STUDY = 'paper-shared-read-baseline-followup-v1'
SEEDS = (2026,613,123)
FAMILY = ('AAAB','ABAA','AABA','AABB','ABBA')
REUSED = ('AAAB','AABC','ABBC')
CONDITIONAL = ('ABAA','AABA','AABB','ABBA')


def require(condition, message):
    if not condition: raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def file_sha(path):
    path=Path(path)
    require(path.is_file() and not any(x.is_symlink() for x in (path,*path.parents)), 'regular nonsymlink input required')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def design(expected_sha256=None):
    require(file_sha(DESIGN)==DESIGN_SHA256,'frozen follow-up design bytes differ')
    if expected_sha256 is not None:require(DESIGN_SHA256==expected_sha256,'external follow-up design SHA differs')
    v=json.loads(DESIGN.read_text())
    require(v['study_id']==STUDY and v['version']==1 and v['seeds']==list(SEEDS) and v['updates']==2000
        and v['stage_a_new']==['AAAA'] and v['reused']==list(REUSED)
        and v['stage_b_conditional_new']==list(CONDITIONAL) and v['comparison_family']==list(FAMILY)
        and v['primary_contrast']==['AAAB','AAAA'] and v['practical_mean_delta_max']=='-0.0025'
        and v['all_seed_deltas_strictly_negative'] is True and v['stage_b_requires_stage_a_primary_pass'] is True
        and v['stage_b_separate_paid_binding_required'] is True and v['initial_new_training_runs']==3
        and v['maximum_new_training_runs']==15,'follow-up matrix/decisions changed')
    require(all(v[k] is False for k in ('paid_execution_authorized','primary_scoring_authorized',
        'production_ready','equivalence_claim_enabled','population_significance_claim_enabled')),
        'preparation contract cannot grant paid/scoring/equivalence authority')
    require((v['evaluation_sequences'],v['sequence_length'],v['supervised_targets_per_sequence'])==(1024,1024,1023)
        and v['new_baseline_attention_reads']==1 and v['new_baseline_dormant_router_parameters']==0,
        'evaluation or baseline changed')
    return v


def run_matrix():
    design()
    return [{'id':f'paper-baseline-{c.lower()}-seed{s}-step2000','condition':c,'seed':s,'updates':2000,
        'stage':'A' if c=='AAAA' else 'B_conditional','new_training':True}
        for c in ('AAAA',*CONDITIONAL) for s in SEEDS]


def reused_subjects():
    return [{'id':f'paper-sharing-{c.lower()}-seed{s}-step2000','condition':c,'seed':s,
        'new_training':False,'old_primary_claims_unchanged':True} for c in REUSED for s in SEEDS]


def contrast(records, candidate, *, evaluation_sha256):
    """All three paired seeds; explicit fresh artifact required. No sequence inference."""
    require(candidate in (*FAMILY,'AABC','ABBC'),'unknown candidate')
    require(isinstance(evaluation_sha256,str) and len(evaluation_sha256)==64
        and all(c in '0123456789abcdef' for c in evaluation_sha256),'evaluation SHA required')
    values={}
    for row in records:
        key=(row['condition'],row['seed'])
        require(key not in values and row['condition'] in ('AAAA',*FAMILY,'AABC','ABBC')
            and type(row['seed']) is int and row['seed'] in SEEDS,'duplicate or unknown subject/seed')
        require(type(row['updates']) is int and row['updates']==2000
            and row['evaluation_sha256']==evaluation_sha256 and row['old_primary_rewritten'] is False,
            'wrong horizon/artifact or rewritten old scoring')
        require(not isinstance(row['loss'],bool),'boolean loss refused')
        value=Decimal(str(row['loss']));require(value.is_finite(),'nonfinite loss')
        values[key]=value
    require(all((c,s) in values for c in ('AAAA',candidate) for s in SEEDS),'missing paired seed')
    deltas=[values[candidate,s]-values['AAAA',s] for s in SEEDS]
    mean=sum(deltas)/3
    return {'candidate':candidate,'baseline':'AAAA','per_seed':{str(s):str(x) for s,x in zip(SEEDS,deltas)},
        'mean':str(mean),'sample_sd':str(statistics.stdev(deltas)),
        'passes':all(x<0 for x in deltas) and mean<=Decimal('-0.0025'),
        'formal_equivalence_test':False,'population_significance_claim':False,'execution_authorized':False}


def broad_decision(records, *, evaluation_sha256):
    values={c:contrast(records,c,evaluation_sha256=evaluation_sha256) for c in FAMILY}
    return {'family':values,'all_five_tested_partitions_pass':all(x['passes'] for x in values.values()),
        'scope':'five tested two-read MHAR partitions; three fixed seeds; 2K horizon',
        'equivalence_established':False,'automatic_extension_authorized':False}
