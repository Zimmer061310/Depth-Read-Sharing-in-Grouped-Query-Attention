#!/usr/bin/env python3
"""Verify saved-result bytes and independently reproduce all 33 saved scores."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

ROOT=Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/name).read_text())
def close(a,b):assert math.isclose(a,b,rel_tol=0,abs_tol=2e-12),(a,b)


def main():
    manifest=read('results/checksums.json')
    for name,spec in manifest['files'].items():
        p=ROOT/name;raw=p.read_bytes()
        assert len(raw)==spec['bytes'] and hashlib.sha256(raw).hexdigest()==spec['sha256'],name
    rows=list(csv.DictReader((ROOT/'results/scores.csv').open()))
    values={};checkpoints={};evaluation_ids={}
    assert len(rows)==33
    for row in rows:
        artifact,c,s=row['artifact'],row['condition'],int(row['seed'])
        v=read(f'results/scores/{artifact}/{c}-seed{s}.json')
        assert v['updates']==2000 and len(v['per_sequence'])==1024
        assert [x['sequence'] for x in v['per_sequence']]==list(range(1024))
        assert all(x['tokens']==1023 and math.isfinite(x['loss_sum']) for x in v['per_sequence'])
        loss=sum(x['loss_sum'] for x in v['per_sequence'])/sum(x['tokens'] for x in v['per_sequence'])
        close(loss,v['loss']);close(loss,float(row['loss']))
        key=artifact,c,s;assert key not in values;values[key]=loss
        old=checkpoints.setdefault((c,s),v['checkpoint_manifest_sha256'])
        assert old==v['checkpoint_manifest_sha256']
        assert evaluation_ids.setdefault(artifact,v['evaluation_sha256'])==v['evaluation_sha256']
    assert len(checkpoints)==24 and len(set(evaluation_ids.values()))==2
    expected=read('results/expected.json')
    for artifact,key in (('followup','followup_contrasts'),('original-primary','original_primary_contrasts')):
        for r in expected[key]:
            ds=[values[artifact,r['candidate'],s]-values[artifact,r['reference'],s] for s in (2026,613,123)]
            close(statistics.mean(ds),r['mean']);close(statistics.stdev(ds),r['sample_sd'])
            for s,d in zip((2026,613,123),ds):close(d,r['per_seed'][str(s)])
            assert all(d<0 for d in ds)==r['all_negative']
            if r['role']!='descriptive':
                assert (all(d<0 for d in ds) and statistics.mean(ds)<=r['margin'])==r['numerical_gate']
    assert all(not r['numerical_gate'] for r in expected['original_primary_contrasts'])
    print('PASS: saved-result hashes; 24 checkpoints / 33 sequence-aggregated scores; separate artifacts; paired means/SDs and unchanged decisions.')
    print('Saved numerical evidence only; no training, scoring or seed-significance claim.')


if __name__=='__main__':main()
