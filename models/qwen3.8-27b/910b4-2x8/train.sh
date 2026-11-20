#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export DEPLOYMENT_DIR=$PWD

# Run on both hosts with NODE_RANK=0/1 and the same MASTER_ADDR and MASTER_PORT.
export NNODES=2
export TP=${TP:-4} PP=${PP:-4}
export GBS=${GBS:-64} MBS=${MBS:-1} STEPS=${STEPS:-5}
export RECOMPUTE=${RECOMPUTE:-full} RECOMPUTE_LAYERS=${RECOMPUTE_LAYERS:-1}

exec bash "$DEPLOYMENT_DIR/../runtime/launch.sh" "$@"
