"""Single-read MHAR control with matched two-slot query execution."""
import copy
from pathlib import Path

import torch
from torch import nn

from src.attention_residuals import modeling_qwen3_attnres as native
from src.experiments.experiment23_route_topology import Experiment23MHARForCausalLM,Experiment23Attention
from src.experiments.paper_sharing_execution_policy import POLICY_ID,POLICY_SHA256,canonical_policy
from src.experiments.paper_sharing_determinism import RUNTIME_POLICY_ID,RUNTIME_POLICY_SHA256,runtime_binding
from src.experiments.paper_sharing_model import _state_sha256,runtime_buffer_fingerprints
from src.experiments.paper_sharing_spec import require


def configure(config, *, fixture=False):
    require(type(fixture) is bool,'explicit boolean fixture identity required')
    require(not any(k.startswith('experiment') and v is not None and v is not False
        for k,v in config.to_dict().items()),'start from a clean native config')
    require(not getattr(config,'paper_sharing_validation',False)
        and not getattr(config,'paper_shared_read_baseline',False),'nested paper/follow-up config refused')
    canonical_policy()
    result=copy.deepcopy(config)
    result.paper_shared_read_baseline=True
    result.paper_shared_read_condition='AAAA'
    result.paper_sharing_condition='AAAA'  # Honest identity; legacy three-model guards MUST reject it.
    result.paper_sharing_test_fixture=fixture
    result.paper_sharing_router_backend='canonical'
    result.paper_sharing_query_execution='two-slot-v1'
    result.paper_sharing_execution_policy=POLICY_ID
    result.paper_sharing_execution_policy_sha256=POLICY_SHA256
    result.paper_sharing_runtime_policy=RUNTIME_POLICY_ID
    result.paper_sharing_runtime_policy_sha256=RUNTIME_POLICY_SHA256
    result.experiment23_route_topology=True
    result.experiment23_variant='global-aaaa'
    result.experiment23_projection_backend='canonical'
    result.experiment23_router_backend='shared-eager'
    validate_config(result)
    return result


def validate_config(config):
    require(getattr(config,'paper_shared_read_baseline',False) is True
        and getattr(config,'paper_shared_read_condition',None)=='AAAA'
        and getattr(config,'paper_sharing_condition',None)=='AAAA'
        and getattr(config,'paper_sharing_validation',False) is False
        and type(getattr(config,'paper_sharing_test_fixture',None)) is bool
        and getattr(config,'paper_sharing_query_execution',None)=='two-slot-v1'
        and getattr(config,'paper_sharing_router_backend',None)=='canonical','explicit AAAA identity required')
    expected={'hidden_size':1280,'head_dim':80,'num_attention_heads':16,'num_key_value_heads':8,
              'attnres_num_heads':8}
    require(all(type(getattr(config,k,None)) is int and getattr(config,k)==v for k,v in expected.items()),
            'fixed two-Q/one-KV GQA and H8 feature routing required')
    require(config.attnres_mode=='full_mh' and config.attnres_gate_type=='bias'
        and config.attention_dropout==0 and not config.attention_bias and not config.attnres_use_null_source
        and not config.use_cache,'native MHAR/no dropout/bias/null source/cache training required')
    require(config.experiment23_route_topology is True and config.experiment23_variant=='global-aaaa'
        and config.experiment23_projection_backend=='canonical' and config.experiment23_router_backend=='shared-eager',
        'single read and matched canonical projections required')
    allowed={'experiment23_route_topology','experiment23_variant','experiment23_projection_backend','experiment23_router_backend'}
    require(not any(k.startswith('experiment') and k not in allowed and v is not None and v is not False
        for k,v in config.to_dict().items()),'other experiment dispatch refused')
    if not config.paper_sharing_test_fixture:
        require((config.num_hidden_layers,config.intermediate_size,config.vocab_size)==(36,5120,151936),
                'reduced model only in explicit fixtures')
    runtime_binding(config)


class SharedReadBaselineForCausalLM(Experiment23MHARForCausalLM):
    def __init__(self, config):
        validate_config(config)
        super().__init__(config)
        for layer in self.model.layers:
            # The archival generic wrapper registers an inactive B. Remove it
            # in this distinct class; never optimize/count a dummy route.
            del layer.local_attn_res_proj

    def validate_execution_dispatch(self):
        validate_config(self.config)
        runtime_binding(self.config,check_runtime=True)
        fused=native._mhar_fused
        require(native._compiled_mh_block_kernel is None
            and (fused is None or (not fused._ENABLED and fused.current_context() is None)),
            'compiled/fused single-read path refused')
        for layer in self.model.layers:
            att=layer.self_attn
            require(layer._experiment23_route_topology is True and layer._experiment23_active_b is False
                and layer._experiment23_collapse is False and layer._experiment23_router_backend=='shared-eager'
                and not getattr(layer,'_experiment24_three_route',False)
                and not hasattr(layer,'local_attn_res_proj') and not hasattr(layer,'third_attn_res_proj')
                and type(att) is Experiment23Attention and att._experiment23_routes=='AAAA'
                and att._experiment23_projection=='canonical' and type(att.q_proj) is nn.Linear
                and att.q_proj.weight.shape==(1280,1280) and att.q_proj.bias is None,
                'baseline route/projection dispatch or active parameter set changed')
            require(all(getattr(layer,key) is None for key in
                ('_mhar_partition','_mhar_mixed_partition','_mhar_contiguous_partition')),
                'native full-width/H8 source access required')

    def forward(self,*args,**kwargs):
        self.validate_execution_dispatch()
        return super().forward(*args,**kwargs)


def state_fingerprint(model):
    validate_config(model.config)
    return _state_sha256(model.state_dict().items())


def model_fingerprints(model):
    model.validate_execution_dispatch()
    return {'condition': 'AAAA', 'checkpoint_state_sha256': state_fingerprint(model),
            'attention_route_count': 1, 'test_fixture': model.config.paper_sharing_test_fixture,
            'nominal_parameters': sum(p.numel() for p in model.parameters())}


def load_checkpoint_model(path, *, expected_condition, expected_state_sha256):
    require(expected_condition == 'AAAA', 'foreign checkpoint condition')
    # Runtime-buffer binding is independently checked after device restoration
    # by the exact-resume adapter.
    import json
    metadata = json.loads((Path(path)/'checkpoint-manifest.json').read_text())
    return load_local_checkpoint(path, expected_state_sha256=expected_state_sha256,
                                 expected_runtime_buffers=metadata['runtime_buffers'])


def initialization_bridge(model):
    """Comparable STEP-ZERO hashes; virtual B is never a model/optimizer parameter.

    Legacy common hashes include B and exclude C. B was a zero-RNG copy of A at
    initialization. Reconstruct only that hash inventory for pairing; do not
    treat virtual tensors as a saved/trained AAAB model or use after an update.
    Caller must separately bind actual completed subjects' recorded step zero.
    """
    validate_config(model.config)
    require(all(p.grad is None for p in model.parameters()),'initialization requires clean gradients')
    require(all(bool(torch.equal(layer.attn_res_proj.weight,torch.zeros_like(layer.attn_res_proj.weight)))
        for layer in model.model.layers),'bridge only accepts native zero-initialized attention routers')
    common=list(model.state_dict().items())
    common.extend((f'model.layers.{i}.local_attn_res_proj.weight',layer.attn_res_proj.weight)
        for i,layer in enumerate(model.model.layers))
    slots={name:[] for name in ('Q1','Q2')}
    for i,layer in enumerate(model.model.layers):
        weight=layer.self_attn.q_proj.weight.reshape(8,2,80,1280)
        slots['Q1'].append((str(i),weight[:,1]));slots['Q2'].append((str(i),weight[:,0]))
    return {'condition':'AAAA','comparison_scope':'step_zero_common_parameters_only_virtual_B_not_registered',
        'legacy_common_parameters_sha256':_state_sha256(common),
        'q_slot_initial_parameters_sha256':{k:_state_sha256(v) for k,v in slots.items()},
        'baseline_state_sha256':state_fingerprint(model),'full_scientific_pairing_verified':False}


def optimizer_coverage(model,optimizer):
    validate_config(model.config)
    expected={id(p) for p in model.parameters() if p.requires_grad}
    actual=[id(p) for group in optimizer.param_groups for p in group['params']]
    require(len(actual)==len(set(actual)) and set(actual)==expected,'missing/duplicate/foreign optimizer parameter')
    require(not any('local_attn_res_proj' in name or 'third_attn_res_proj' in name
        for name,_ in model.named_parameters()),'dormant route parameter refused')
    return {'active_parameter_objects':len(expected),'dormant_router_parameters':0,'fixture':model.config.paper_sharing_test_fixture}


def costs(model):
    validate_config(model.config)
    depth,width=model.config.num_hidden_layers,model.config.hidden_size
    return {'nominal_parameters':sum(p.numel() for p in model.parameters()),'active_attention_reads':1,
        'dormant_router_parameters':0,'extra_weights_per_additional_attention_read':depth*width,
        'additional_read_ideal_score_mix_macs_per_token':2*width*sum(2*i+1 for i in range(depth)),
        'query_gemm_output_widths':[640,640],'kv_heads':8,'head_dim':80,
        'packed_kv_payload_multiplier':1,'fresh_runtime_measured':False}


def load_local_checkpoint(path, *, expected_state_sha256, expected_runtime_buffers):
    """Strict MODEL-only reload, not scientific optimizer/data/RNG resume."""
    root=Path(path).resolve(strict=True)
    cfg=SharedReadBaselineForCausalLM.config_class.from_pretrained(root,local_files_only=True)
    cfg._attn_implementation='sdpa';validate_config(cfg)
    model,info=SharedReadBaselineForCausalLM.from_pretrained(root,config=cfg,local_files_only=True,
        output_loading_info=True,attn_implementation='sdpa',dtype='auto')
    require(not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),
            'missing/foreign checkpoint weights refused')
    require(state_fingerprint(model)==expected_state_sha256,'saved model state differs')
    model.to(dtype=next(model.parameters()).dtype)  # Replay native whole-module cast for nonpersistent RoPE.
    require(state_fingerprint(model)==expected_state_sha256,'buffer cast changed model weights')
    require(runtime_buffer_fingerprints(model)==expected_runtime_buffers,'restored execution buffers differ')
    return model
