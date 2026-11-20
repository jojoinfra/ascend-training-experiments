"""Chunk the vocabulary-parallel LM projection and CE across token rows."""
import torch
import torch.nn.functional as F
from megatron.core import parallel_state as ps, tensor_parallel as tp


class ChunkedProjectionCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, chunk_size):
        flat = hidden.reshape(-1, hidden.shape[-1])
        targets = targets.reshape(-1)
        assert flat.shape[0] == targets.numel()
        dlogits = hidden.new_empty((targets.numel(), weight.shape[0]))
        total = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, targets.numel(), chunk_size):
            rows = slice(start, start + chunk_size)
            logits = F.linear(flat[rows], weight)
            # Use Megatron's actual TP collectives and gradient, releasing the
            # full-precision softmax as soon as each chunk is consumed.
            with torch.enable_grad():
                logits.requires_grad_()
                losses = tp.vocab_parallel_cross_entropy(logits, targets[rows])
                part = losses.float().sum() / targets.numel()
                derivative, = torch.autograd.grad(part, logits)
            dlogits[rows].copy_(derivative)
            total.add_(part.detach())
            del logits, losses, part, derivative
        ctx.save_for_backward(hidden, weight, dlogits)
        return total

    @staticmethod
    def backward(ctx, upstream):
        hidden, weight, dlogits = ctx.saved_tensors
        scaled = dlogits * upstream.to(dlogits.dtype)
        dx = scaled.matmul(weight).view_as(hidden)
        dw = scaled.t().matmul(hidden.reshape(-1, hidden.shape[-1]))
        return dx, dw, None, None


def loss_from_shards(hidden, weight, labels, chunk_size):
    gathered = tp.gather_from_sequence_parallel_region(hidden, group=ps.get_tensor_model_parallel_group())
    return ChunkedProjectionCE.apply(gathered, weight, labels.transpose(0, 1).contiguous(), chunk_size)


def compute_chunk_loss(self, hidden_states, lm_head, labels, chunk_size):
    assert lm_head.bias is None and self.config.sequence_parallel and not self.config.use_mup
    return loss_from_shards(hidden_states, lm_head.weight, labels, chunk_size)


def verify_chunked_ce():
    # Probe first/last labels, each TP vocab shard, SP gather/reduce-scatter,
    # a non-divisible final chunk, and the pipeline's upstream loss scaling.
    rank = ps.get_tensor_model_parallel_rank()
    world = ps.get_tensor_model_parallel_world_size()
    dtype = torch.bfloat16
    device = 'npu'
    hidden = ((torch.arange(32*64, device=device).float()+31*rank).cos()*.25).to(dtype).reshape(32,1,64).requires_grad_()
    weight = ((torch.arange(128*64, device=device).float()+17*rank).sin()*.25).to(dtype).reshape(128,64).requires_grad_()
    labels = (torch.arange(world*32, device=device)*131 % (world*128)).reshape(1,-1)
    gathered = tp.gather_from_sequence_parallel_region(hidden, group=ps.get_tensor_model_parallel_group())
    expected = tp.vocab_parallel_cross_entropy(F.linear(gathered, weight), labels.t().contiguous()).float().mean()
    actual = loss_from_shards(hidden, weight, labels, 48)
    scaler = 1/64
    ref_grads = torch.autograd.grad(expected * scaler, (hidden, weight))
    got_grads = torch.autograd.grad(actual * scaler, (hidden, weight))
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    for got, ref in zip(got_grads, ref_grads):
        torch.testing.assert_close(got.float(), ref.float(), atol=2e-6, rtol=.02)
    print(f'BENCH_CHUNK_PARITY tp_rank={rank} loss={actual.item()} reference={expected.item()} dx_dw=pass', flush=True)
