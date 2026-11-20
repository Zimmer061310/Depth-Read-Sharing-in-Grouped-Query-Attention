"""Entry points to the unchanged evaluated reference classes."""
from pathlib import Path
import json
import sys

REFERENCE = Path(__file__).resolve().parents[1] / 'reference'
PATTERNS = ('AAAA','AAAB','ABAA','AABA','AABB','ABBA','AABC','ABBC')


def _imports():
    if str(REFERENCE) not in sys.path:
        sys.path.insert(0,str(REFERENCE))


def build_config(pattern, *, fixture=False):
    if pattern not in PATTERNS or type(fixture) is not bool:
        raise ValueError('select an evaluated pattern and explicit boolean fixture flag')
    _imports()
    from src.attention_residuals.modeling_qwen3_attnres import Qwen3AttnResConfig
    from src.experiments.paper_sharing_execution_policy import POLICY_ID
    from src.experiments.paper_sharing_determinism import RUNTIME_POLICY_ID
    recipe=json.loads((REFERENCE/'configs/paper_sharing_validation_2k/training-recipe.frozen.json').read_text())
    cfg=Qwen3AttnResConfig(vocab_size=64 if fixture else recipe['model_vocab_size'],
        hidden_size=1280,num_hidden_layers=2 if fixture else 36,
        intermediate_size=128 if fixture else 5120,num_attention_heads=16,
        num_key_value_heads=8,head_dim=80,max_position_embeddings=recipe['max_position_embeddings'],
        tie_word_embeddings=True,rms_norm_eps=1e-6,attention_dropout=0.,
        attnres_mode='full_mh',attnres_num_heads=8,attnres_gate_type='bias',
        attnres_use_null_source=False,use_cache=False,_attn_implementation='sdpa')
    if pattern=='AAAA':
        from src.experiments.paper_shared_read_baseline import configure
        return configure(cfg,fixture=fixture)
    if pattern in ('ABAA','AABA','AABB','ABBA'):
        from src.experiments.paper_stage_b_baseline import configure
        return configure(cfg,pattern,fixture=fixture)
    from src.experiments.paper_sharing_model import configure
    return configure(cfg,pattern,router_backend='canonical',test_fixture=fixture,
                     execution_policy=POLICY_ID,runtime_policy=RUNTIME_POLICY_ID)


def build_model(pattern, *, fixture=False):
    config=build_config(pattern,fixture=fixture)
    if pattern=='AAAA':
        from src.experiments.paper_shared_read_baseline import SharedReadBaselineForCausalLM as Model
    elif pattern in ('ABAA','AABA','AABB','ABBA'):
        from src.experiments.paper_stage_b_baseline import StageBForCausalLM as Model
    else:
        from src.experiments.paper_sharing_model import PaperSharingForCausalLM as Model
    return Model(config)


def strict_runtime(*, fixture=False):
    _imports()
    from src.experiments.paper_sharing_determinism import strict_runtime as context
    return context(fixture=fixture)
