#!/usr/bin/env python3
"""Reduced CPU model checks. No updates, corpus access or paper scoring."""
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from depth_read_sharing import PATTERNS,build_model,strict_runtime

torch.set_num_threads(1)
with strict_runtime(fixture=True):
    for pattern in PATTERNS:
        torch.manual_seed(0)
        model=build_model(pattern,fixture=True).cpu().eval()
        generator=torch.Generator().manual_seed(4827)
        with torch.no_grad():
            for name,parameter in model.named_parameters():
                if 'attn_res_proj.weight' in name:
                    parameter.copy_(torch.randn(parameter.shape,generator=generator)*0.1)
        tokens=torch.arange(16).reshape(1,16)
        result=model(input_ids=tokens,labels=tokens)
        logits=result.logits
        assert logits.shape==(1,16,64)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(result.loss)
        result.loss.backward()
        gradients=[p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(p.grad is not None and torch.count_nonzero(p.grad)>0
                   for name,p in model.named_parameters() if 'attn_res_proj.weight' in name)
        print('PASS:',pattern,'reduced CPU forward/backward',tuple(logits.shape))
        with TemporaryDirectory() as path:
            model.save_pretrained(path)
            restored=type(model).from_pretrained(path,attn_implementation='sdpa').cpu().eval()
            with torch.no_grad():
                restored_logits=restored(input_ids=tokens).logits
            assert torch.equal(logits.detach(),restored_logits)
            assert restored.get_input_embeddings().weight.data_ptr()==restored.get_output_embeddings().weight.data_ptr()
            print('PASS:',pattern,'explicit-SDPA save/reload and tied embeddings')
            del restored,restored_logits
        del model,result,logits,gradients
print('Engineering fixture only; no training or scientific evaluation performed.')
