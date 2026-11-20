#!/usr/bin/env bash
# Run on each eight-card training host.
set -euo pipefail
TRAINING_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONTAINER_NAME=${CONTAINER_NAME:-qwen38-benchmark}
TRAINING_IMAGE=${TRAINING_IMAGE:-qwen38-training:910b4-20261114}
MODEL_REFERENCE=${MODEL_REFERENCE:-$TRAINING_DIR/model-reference}
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    echo "Container $CONTAINER_NAME already exists; use docker start if stopped." >&2
    exit 1
fi
if ! docker image inspect "$TRAINING_IMAGE" >/dev/null; then
    echo "Docker must be able to read $TRAINING_IMAGE on this host." >&2
    echo "Load the supplied image archive with docker load -i, or set TRAINING_IMAGE to a prepared base image (see the model README.md)." >&2
    exit 1
fi
test -f "$MODEL_REFERENCE/config.json"
dependency_dir=$(python3 "$TRAINING_DIR/../../tools/prepare_dependencies.py" "$TRAINING_DIR/dependencies.lock.json")
dependency_mounts=()
for framework in Megatron-LM MindSpeed MindSpeed-MM flash-linear-attention-npu; do
    dependency_mounts+=(-v "$dependency_dir/$framework:/work/$framework:ro")
done
devices=()
for card in {0..7}; do devices+=(--device "/dev/davinci$card"); done
for device in davinci_manager devmm_svm hisi_hdc; do devices+=(--device "/dev/$device"); done
docker run -d --name "$CONTAINER_NAME" --user 0 --network host --shm-size 32g \
    "${devices[@]}" \
    -v "$TRAINING_DIR:/work" \
    "${dependency_mounts[@]}" \
    -v "$MODEL_REFERENCE:/model-reference:ro" \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /etc/hccn.conf:/etc/hccn.conf:ro \
    --entrypoint /bin/bash "$TRAINING_IMAGE" -c 'exec sleep infinity'
