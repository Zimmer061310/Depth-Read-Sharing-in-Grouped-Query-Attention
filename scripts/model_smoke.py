#!/usr/bin/env python3
"""Reduced CPU forward checks. No updates, corpus access or paper scoring."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from depth_read_sharing import PATTERNS,build_model,strict_runtime

torch.set_num_threads(1)
with strict_runtime(fixture=True):
    for pattern in PATTERNS:
        torch.manual_seed(0)
        model=build_model(pattern,fixture=True).cpu().eval()
        with torch.no_grad():
            logits=model(input_ids=torch.arange(16).reshape(1,16)).logits
        assert logits.shape==(1,16,64)
        assert torch.isfinite(logits).all()
        print('PASS:',pattern,'reduced CPU forward',tuple(logits.shape))
        del model,logits
print('Engineering fixture only; no training or scientific evaluation performed.')
