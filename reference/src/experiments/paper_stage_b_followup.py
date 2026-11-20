"""Prospective Stage B: immutable Stage-A reference scores, no new authority."""
from src.experiments.paper_shared_read_followup import *
from src.experiments import paper_shared_read_followup as parent
STUDY=parent.STUDY+'-stage-b-v1'
STAGE_A=ROOT/'results/paper-sharing-validation/shared-read-stage-a-20261029/raw'
REFERENCE_HASHES={'paper-sharing-aaab-seed613-step2000': '3918ad88253bdf8a7b3301cc692e934573dbb9c2f9247a878a0c99d9d167db1f', 'paper-sharing-aaab-seed123-step2000': '39224b4176fcf585070c0cd2426999fe3bc0d8e4e6a772c35ea22e0cf41dab04', 'paper-sharing-aaab-seed2026-step2000': 'da5b32969d41a32542b904c474fb182579017b82a0991f058ab2b0942eafbd1e', 'paper-baseline-aaaa-seed613-step2000': '5c46791584b8d19f6a0932abb2e9e7d6e9199a8c5d5e944a36cd2fac4771eff9', 'paper-baseline-aaaa-seed2026-step2000': 'de30c32f8e43dc820521e634859fca4f9442900bb62da78a0025e0feb9c38572', 'paper-baseline-aaaa-seed123-step2000': '06f0b4281f057e7011a7f61e8e96488de8f24c9974bf54e0abd2c5f6a831e6f6'}

def design(expected_sha256=None):
    v=parent.design(expected_sha256)
    require(file_sha(STAGE_A/'ANALYSIS.json')=='e8bd66075b9efd5f97c99d8fdaa54485027b45953e3cc78221fa2046ee8f27f6','Stage-A gate bytes changed')
    require(json.loads((STAGE_A/'ANALYSIS.json').read_text())['stage_b_scientific_gate_passed'] is True,'Stage-A primary gate failed')
    reference_records()
    return v

def reference_records():
    rows=[]
    for identifier,h in REFERENCE_HASHES.items():
        p=STAGE_A/f'SCORE_RESULT-{identifier}.json'
        require(file_sha(p)==h,'immutable Stage-A score changed')
        score=json.loads(p.read_text());require(score['run_id']==identifier and not score['old_primary_rewritten'],'foreign reference')
        rows.append({k:score[k] for k in ('condition','seed','updates','evaluation_sha256','old_primary_rewritten','loss')})
    return rows

def reused_subjects():
    return [{'id':identifier,'score_sha256':h,'new_training':False} for identifier,h in sorted(REFERENCE_HASHES.items())]
