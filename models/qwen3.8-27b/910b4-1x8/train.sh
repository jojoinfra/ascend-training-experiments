#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export DEPLOYMENT_DIR=$PWD

# One host, 8 devices, TP2 × PP4, global batch 32.
export NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1
export TP=${TP:-2} PP=${PP:-4}
export GBS=${GBS:-32} MBS=${MBS:-1} STEPS=${STEPS:-5}
export RECOMPUTE=${RECOMPUTE:-full} RECOMPUTE_LAYERS=${RECOMPUTE_LAYERS:-1}

exec bash "$DEPLOYMENT_DIR/../runtime/launch.sh" "$@"
