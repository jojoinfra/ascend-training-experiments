# Host and runtime settings. Override these variables when invoking the launcher.
# This NUMA mapping is from the measured 192-thread 910B4 hosts.
export BENCH_CPU_THREADS=${BENCH_CPU_THREADS:-24}
export NUMA_BIND=${NUMA_BIND:-socket}
export NPU_NUMA_NODES=${NPU_NUMA_NODES:-6,0,6,0,4,2,4,2}
export OMP_NUM_THREADS=$BENCH_CPU_THREADS
export OPENBLAS_NUM_THREADS=1
FRAMEWORK_ROOT=${FRAMEWORK_ROOT:-/work}
export PYTHONPATH=$PWD:$FRAMEWORK_ROOT/MindSpeed:$FRAMEWORK_ROOT/Megatron-LM:$FRAMEWORK_ROOT/MindSpeed-MM
export FLA_NPU_SOURCE=${FLA_NPU_SOURCE:-$FRAMEWORK_ROOT/flash-linear-attention-npu}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TASK_QUEUE_ENABLE=2
export HCCL_CONNECT_TIMEOUT=180
export HCCL_EXEC_TIMEOUT=600
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
if [[ $NUMA_BIND == none ]]; then
    export CPU_AFFINITY_CONF=1
else
    unset CPU_AFFINITY_CONF
fi
