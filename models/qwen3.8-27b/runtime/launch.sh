#!/usr/bin/env bash
# Shared Megatron-style torchrun command. Use a deployment's train.sh.
set -eo pipefail
cd "$(dirname "$0")/.."
source /usr/local/Ascend/cann/set_env.sh || { echo 'CANN environment setup failed.' >&2; exit 1; }
set -u
source runtime/env.sh

: "${DEPLOYMENT_DIR:?Use a deployment train.sh}" "${NNODES:?}" "${NODE_RANK:?}" "${MASTER_ADDR:?}"
if (( NNODES == 2 )); then
    : "${HCCL_IF_IP:?Set this host's communication IP}"
    : "${GLOO_SOCKET_IFNAME:?Set this host's communication interface}"
fi
export TP=${TP:-4} PP=${PP:-4} GBS=${GBS:-64} MBS=${MBS:-1} STEPS=${STEPS:-5}
export RECOMPUTE=${RECOMPUTE:-full} RECOMPUTE_LAYERS=${RECOMPUTE_LAYERS:-1}
export MC2=${MC2:-0} QWEN_FUSED_SWIGLU=${QWEN_FUSED_SWIGLU:-0}
export PIPELINE_LAYERS=${PIPELINE_LAYERS:-} STAGE_RECOMPUTE=${STAGE_RECOMPUTE:-}
export CE_CHUNK_TOKENS=${CE_CHUNK_TOKENS:-0}
export OFFLOAD_FRACTION=${OFFLOAD_FRACTION:-1.0}
export MC2_FP32_ACCUM=${MC2_FP32_ACCUM:-0} QWEN_UNBATCH_P2P=${QWEN_UNBATCH_P2P:-0}
for removed in HOST_VERIFY PIN_CPU_MASTER PROFILE_STAGES OBSERVE_MICROBATCHES OBSERVE_FROM_STEP; do
    [[ -z ${!removed:-} ]] || { echo "$removed belongs to the retired custom training loop; unset it." >&2; exit 1; }
done
(( NNODES >= 1 && NNODES <= 2 && NODE_RANK >= 0 && NODE_RANK < NNODES )) || {
    echo 'Invalid node count or rank.' >&2; exit 1;
}
(( (TP == 1 || TP == 2 || TP == 4) && PP > 0 && 64 % PP == 0 && (8 * NNODES) % (TP * PP) == 0 )) || {
    echo 'Invalid TP/PP configuration.' >&2; exit 1;
}
(( MBS > 0 && GBS > 0 && GBS % (MBS * (8 * NNODES / (TP * PP))) == 0 && STEPS >= 5 )) || {
    echo 'Invalid batch size or timing window.' >&2; exit 1;
}
(( MC2_FP32_ACCUM == 0 || MC2 == 1 )) || {
    echo 'FP32 accumulation fusion requires MC2=1.' >&2; exit 1;
}
(( CE_CHUNK_TOKENS >= 0 && CE_CHUNK_TOKENS <= 4096 )) || {
    echo 'CE_CHUNK_TOKENS must be between 0 and 4096.' >&2; exit 1;
}

DISTRIBUTED_ARGS=(
    --nproc_per_node 8 --nnodes "$NNODES" --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR" --master_port "${MASTER_PORT:-29871}"
)
MODEL_ARGS=(
    --model-config /model-reference/config.json
    --num-layers 64 --hidden-size 5120 --ffn-hidden-size 17408
    --num-attention-heads 24 --group-query-attention --num-query-groups 4 --kv-channels 256
    --seq-length 4096 --max-position-embeddings 262144
    --normalization RMSNorm --swiglu --disable-bias-linear --untie-embeddings-and-output-weights
    --attention-dropout 0 --hidden-dropout 0
    --tokenizer-type NullTokenizer --vocab-size 248320 --make-vocab-size-divisible-by 1
)
TRAINING_ARGS=(
    --micro-batch-size "$MBS" --global-batch-size "$GBS" --train-iters "$STEPS"
    --lr 0.0001 --lr-decay-style constant --clip-grad 0
    --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 --weight-decay 0.1
    --bf16 --no-gradient-accumulation-fusion --distributed-backend nccl --seed 1234
    --log-interval 1 --dataloader-type external --eval-iters 0
)
OPTIMIZER_ARGS=(
    --use-distributed-optimizer --use-precision-aware-optimizer --optimizer-cpu-offload
    --optimizer-offload-fraction "$OFFLOAD_FRACTION" --overlap-cpu-optimizer-d2h-h2d
)
MODEL_PARALLEL_ARGS=(--tensor-model-parallel-size "$TP" --pipeline-model-parallel-size "$PP")
if (( TP > 1 )); then MODEL_PARALLEL_ARGS+=(--sequence-parallel); fi
case "$RECOMPUTE" in
    full) MODEL_PARALLEL_ARGS+=(--recompute-granularity full --recompute-method uniform --recompute-num-layers "$RECOMPUTE_LAYERS") ;;
    block) MODEL_PARALLEL_ARGS+=(--recompute-granularity full --recompute-method block --recompute-num-layers "$RECOMPUTE_LAYERS") ;;
    none) ;;
    *) echo 'RECOMPUTE must be full, block or none.' >&2; exit 1 ;;
esac
FUSION_ARGS=(
    --use-flash-attn --use-fused-rmsnorm --use-fused-swiglu
    --no-masked-softmax-fusion --no-bias-dropout-fusion --no-bias-swiglu-fusion
)
if (( MC2 )); then FUSION_ARGS+=(--use-ascend-mc2); fi

# Extra Megatron flags, e.g. --overlap-grad-reduce, can be passed to either deployment script.
export RUN_NAME=${RUN_NAME:-run-$(date +%Y%m%d-%H%M%S)}
[[ $RUN_NAME =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid RUN_NAME.' >&2; exit 1; }
export RESULT_DIR="$DEPLOYMENT_DIR/results/$RUN_NAME"
exec 9>.training.lock
flock -n 9 || { echo 'Another training run holds .training.lock.' >&2; exit 1; }
mkdir -p "$DEPLOYMENT_DIR/results"
mkdir "$RESULT_DIR"
torchrun "${DISTRIBUTED_ARGS[@]}" pretrain_qwen.py \
    "${MODEL_ARGS[@]}" "${TRAINING_ARGS[@]}" "${MODEL_PARALLEL_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" "${FUSION_ARGS[@]}" "$@" 2>&1 | tee "$RESULT_DIR/node-$NODE_RANK.log"
if (( NNODES == 1 )); then
    MEASURE_ARGS=(--world 8)
    if [[ -n ${LAST_TIMED_STEP:-} ]]; then
        MEASURE_ARGS+=(--last-timed-step "$LAST_TIMED_STEP")
    fi
    python3 measure.py "$RESULT_DIR" "${MEASURE_ARGS[@]}" > "$RESULT_DIR/summary.json"
    cat "$RESULT_DIR/summary.json"
fi
