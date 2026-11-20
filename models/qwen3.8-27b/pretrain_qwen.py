"""Qwen text-model callbacks for Megatron's standard pretrain() entry.

Random initialization, full vocabulary and all parameters trainable. Megatron
owns the training loop and optimizer; the launchers default to CPU AdamW offload.
"""
import os
import sys
import types
import json
import importlib.util
from pathlib import Path
from functools import partial
from dataclasses import fields

TEXT_PARAMETERS = 26_895_998_464

# Host 183 PCIe NUMA topology, verified from /sys/bus/pci/devices/*/numa_node.
# Bind before importing torch so CPU pools and first-touch allocations inherit it.
numa_mode = os.getenv('NUMA_BIND', 'none')
expected_affinity = None
if numa_mode != 'none':
    local_rank = int(os.environ['LOCAL_RANK'])
    node_by_rank = list(map(int, os.environ['NPU_NUMA_NODES'].split(',')))
    node = node_by_rank[local_rank]
    peers = [i for i, n in enumerate(node_by_rank) if n == node]
    slot = peers.index(local_rank)
    if numa_mode == 'local':
        expected_affinity = set(range(node * 24 + slot * 12, node * 24 + (slot + 1) * 12))
    elif numa_mode == 'socket':
        expected_affinity = set(range((node + slot) * 24, (node + slot + 1) * 24))
    else:
        raise ValueError(numa_mode)
    os.sched_setaffinity(0, expected_affinity)

import mindspeed.megatron_adaptor
mm = types.ModuleType('mindspeed_mm')
# Import only the text-model namespace, without optional multimodal pipelines.
mm.__path__ = list(importlib.util.find_spec('mindspeed_mm').submodule_search_locations)
sys.modules['mindspeed_mm'] = mm

import torch
import torch.distributed as dist
from megatron.training import get_args, pretrain
from pretrain_gpt import loss_func
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core import parallel_state as ps
from megatron.core.enums import ModelType
from mindspeed_mm.mcore.models.qwen3_5.qwen35_config import Qwen3_5ModelConfig
from mindspeed_mm.mcore.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
from mindspeed_mm.mcore.models.qwen3_5.spec import get_qwen3_5_text_block_spec
from mindspeed_mm.mcore.process_group_configs import ProcessGroupCollection


class TextModel(Qwen3_5TextModel):
    # The data callback shifts labels once; reuse Megatron CE without MM label shifting.
    compute_language_model_loss = LanguageModule.compute_language_model_loss


def model_provider(pre_process=True, post_process=True, **kwargs):
    a = get_args()
    ref = json.loads(Path(a.model_config).read_text())['text_config']
    for arg, key in [('num_layers', 'num_hidden_layers'), ('hidden_size', 'hidden_size'),
                     ('ffn_hidden_size', 'intermediate_size'), ('num_attention_heads', 'num_attention_heads'),
                     ('num_query_groups', 'num_key_value_heads'), ('kv_channels', 'head_dim'),
                     ('vocab_size', 'vocab_size')]:
        assert getattr(a, arg) == ref[key], f'{arg} must match --model-config'
    layout_arg = os.getenv('PIPELINE_LAYERS')
    layout = list(map(int, layout_arg.split(','))) if layout_arg else [a.num_layers // a.pipeline_model_parallel_size] * a.pipeline_model_parallel_size
    assert len(layout) == a.pipeline_model_parallel_size and sum(layout) == a.num_layers
    assert all(n > 0 for n in layout) and len(set(layout[1:-1])) <= 1
    stage = ps.get_pipeline_model_parallel_rank()
    chunk_size = int(os.getenv('CE_CHUNK_TOKENS', '0'))
    cfg = Qwen3_5ModelConfig(
        num_layers_in_first_pipeline_stage=layout[0] if layout_arg and len(layout) > 1 else None,
        num_layers_in_last_pipeline_stage=layout[-1] if layout_arg and len(layout) > 2 else None,
        num_layers=a.num_layers, hidden_size=ref['hidden_size'],
        num_attention_heads=ref['num_attention_heads'], num_query_groups=ref['num_key_value_heads'],
        ffn_hidden_size=ref['intermediate_size'], kv_channels=ref['head_dim'],
        tensor_model_parallel_size=a.tensor_model_parallel_size,
        pipeline_model_parallel_size=a.pipeline_model_parallel_size,
        sequence_parallel=a.sequence_parallel, params_dtype=torch.bfloat16, bf16=True,
        pipeline_dtype=torch.bfloat16, vocab_size=ref['vocab_size'], seq_length=a.seq_length,
        max_position_embeddings=ref['max_position_embeddings'],
        linear_attention_freq=4, linear_num_key_heads=ref['linear_num_key_heads'],
        linear_num_value_heads=ref['linear_num_value_heads'],
        linear_key_head_dim=ref['linear_key_head_dim'], linear_value_head_dim=ref['linear_value_head_dim'],
        linear_conv_kernel_dim=ref['linear_conv_kernel_dim'],
        causal_conv1d_implementation=os.getenv('CONV_IMPL', 'triton_with_transpose'),
        gdn_implementation=os.getenv('GDN_IMPL', 'ascendc'),
        recompute_granularity=a.recompute_granularity,
        recompute_method=a.recompute_method, recompute_num_layers=a.recompute_num_layers,
        gradient_accumulation_fusion=False,
        bias_activation_fusion=os.getenv('QWEN_FUSED_SWIGLU', '0') == '1',
        bias_dropout_fusion=False, use_cpu_initialization=False,
        init_method_std=ref['initializer_range'], layernorm_epsilon=ref['rms_norm_eps'],
    )
    if os.getenv('QWEN_UNBATCH_P2P', '0') == '1':
        cfg.batch_p2p_comm = False
    if cfg.bias_activation_fusion:
        from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl, SwiGLUFunction
        probe = torch.linspace(-2, 2, 256, device='npu', dtype=torch.bfloat16).view(4, 64).requires_grad_()
        lhs, rhs = probe.chunk(2, dim=-1)
        reference = torch.nn.functional.silu(lhs) * rhs
        actual = bias_swiglu_impl(probe, None)
        upstream = torch.ones_like(actual)
        ga, = torch.autograd.grad(actual, probe, upstream)
        gr, = torch.autograd.grad(reference, probe, upstream)
        torch.testing.assert_close(actual, reference, rtol=.02, atol=.016)
        torch.testing.assert_close(ga, gr, rtol=.02, atol=.016)
        print(f'BENCH_SWIGLU rank={dist.get_rank()} owner={SwiGLUFunction.__module__} bf16_forward_backward=pass', flush=True)
    stage_depths = os.getenv('STAGE_RECOMPUTE')
    if stage_depths:
        depths = [int(v) for v in stage_depths.split(',')]
        assert len(depths) == a.pipeline_model_parallel_size
        depth = depths[ps.get_pipeline_model_parallel_rank()]
        assert 0 <= depth <= layout[stage]
        cfg.recompute_granularity = 'full' if depth else None
        cfg.recompute_method = 'block' if depth else None
        cfg.recompute_num_layers = depth if depth else None
    spec = get_qwen3_5_text_block_spec(cfg)
    # Megatron 0.12 has no embedding group on intermediate pipeline stages.
    pg = ProcessGroupCollection.use_mpu_process_groups(
        required_pgs=[f.name for f in fields(ProcessGroupCollection) if f.name != 'embd'])
    pg.embd = ps.get_embedding_group() if ps.is_rank_in_embedding_group() else None
    model = TextModel(
        config=cfg, transformer_layer_spec=spec, vocab_size=ref['vocab_size'],
        max_sequence_length=a.seq_length, pre_process=pre_process, post_process=post_process,
        position_embedding_type='mrope', rotary_percent=0.25, rotary_base=10000000,
        share_embeddings_and_output_weights=False, parallel_output=True,
        scatter_embedding_sequence_parallel=a.sequence_parallel,
        pg_collection=pg,
    )
    if layout_arg:
        first = sum(layout[:stage]) + 1
        actual_layers = [(layer.layer_number, type(layer.self_attention).__name__) for layer in model.decoder.layers]
        expected_layers = [(n, 'Qwen3_5SelfAttention' if n % 4 == 0 else 'GatedDeltaNet')
                           for n in range(first, first + layout[stage])]
        assert actual_layers == expected_layers, (stage, actual_layers, expected_layers)
        print(f'BENCH_LAYOUT stage={stage} layers={actual_layers}', flush=True)
    if chunk_size and post_process:
        from chunked_ce import compute_chunk_loss, verify_chunked_ce
        verify_chunked_ce()
        cfg.use_chunk_loss = True
        cfg.chunk_size = chunk_size
        model.compute_chunk_loss = types.MethodType(compute_chunk_loss, model)
    else:
        assert not getattr(cfg, "use_chunk_loss", False)
    if os.getenv('MC2_FP32_ACCUM') == '1':
        assert a.use_ascend_mc2 and a.main_grads_dtype == torch.float32
        from mindspeed.core.tensor_parallel.mc2_feature.adaptor import (
            MindSpeedMC2ColumnParallelLinear, MindSpeedMC2RowParallelLinear)
        fused_layers = 0
        for module in model.modules():
            if isinstance(module, (MindSpeedMC2ColumnParallelLinear, MindSpeedMC2RowParallelLinear)):
                if module.weight is not None and module.weight.requires_grad:
                    module.gradient_accumulation_fusion = True
                    fused_layers += 1
        assert fused_layers
        print(f'BENCH_MC2 rank={dist.get_rank()} enabled={fused_layers}', flush=True)
    params = list(model.parameters())
    assert all(p.requires_grad for p in params), 'Frozen parameters are forbidden'
    unique_numel = sum(p.numel() for p in params
                       if getattr(p, 'tensor_model_parallel', False) or ps.get_tensor_model_parallel_rank() == 0)
    total = torch.tensor(unique_numel, dtype=torch.int64, device='npu')
    dist.all_reduce(total)
    a.benchmark_parameters = total.item() // ps.get_data_parallel_world_size()
    if a.num_layers == 64:
        assert a.benchmark_parameters == TEXT_PARAMETERS
    # Release initialization temporaries before upstream DDP/optimizer allocation.
    torch.npu.empty_cache()
    return model


def synthetic_iterator(a):
    gen = torch.Generator(device='cpu').manual_seed(a.seed + ps.get_data_parallel_rank())
    while True:
        # All TP/PP ranks see the same token sequence; DP ranks see independent data.
        yield torch.randint(0, a.vocab_size, (a.micro_batch_size, a.seq_length + 1), generator=gen)


def train_valid_test_datasets_provider(num_samples):
    # Megatron's external loader accepts a ready iterator on every TP/PP rank.
    a = get_args()
    assert a.dataloader_type == 'external' and a.eval_iters == 0
    assert a.context_parallel_size == 1 and a.virtual_pipeline_model_parallel_size is None
    assert not a.load, 'This synthetic performance example starts from random initialization'
    # Called after upstream DDP/optimizer setup: release old parameter-storage
    # caches so CANN/HCCL workspace allocations have room on 32 GiB devices.
    torch.npu.empty_cache()
    return synthetic_iterator(a), None, None


def forward_step(iterator, model):
    a = get_args()
    batch = next(iterator).to('npu', non_blocking=True)
    tokens, labels = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
    pos = torch.arange(a.seq_length, device='npu').view(1, 1, -1).expand(3, a.micro_batch_size, -1)
    output = model(input_ids=tokens, position_ids=pos, attention_mask=None, labels=labels)
    loss_mask = torch.ones_like(tokens, dtype=torch.float32)
    return output, partial(loss_func, loss_mask)


def extra_args_provider(parser):
    parser.add_argument('--model-config', default=str(Path(__file__).parent / 'model-reference/config.json'))
    return parser


if __name__ == '__main__':
    torch.set_num_threads(int(os.getenv('BENCH_CPU_THREADS', '24')))
    if expected_affinity is not None:
        assert set(os.sched_getaffinity(0)) == expected_affinity
    train_valid_test_datasets_provider.is_distributed = True
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=extra_args_provider,
        args_defaults={'tokenizer_type': 'NullTokenizer', 'dataloader_type': 'external', 'eval_iters': 0},
    )
    # Completion metadata only; iteration timing and optimization are upstream-owned.
    a = get_args()
    dist.barrier()
    print('BENCH_COMPLETE ' + json.dumps(dict(
        rank=dist.get_rank(), world=dist.get_world_size(), tp=a.tensor_model_parallel_size,
        pp=a.pipeline_model_parallel_size, seq=a.seq_length, global_batch=a.global_batch_size,
        micro_batch=a.micro_batch_size, steps=a.train_iters, parameters=a.benchmark_parameters,
    )), flush=True)
    dist.destroy_process_group()
