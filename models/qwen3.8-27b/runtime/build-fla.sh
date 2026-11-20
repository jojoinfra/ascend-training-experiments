#!/usr/bin/env bash
source /usr/local/Ascend/cann/set_env.sh
set -euo pipefail
# The locked source mount is read-only. FLA generates files in its build tree.
build_dir=$(mktemp -d)
trap 'rm -rf -- "$build_dir"' EXIT
cp -a /work/flash-linear-attention-npu/. "$build_dir/"
cd "$build_dir"
mkdir -p /work/wheels
python3 -m pip install --no-cache-dir 'torch-npu==2.10.0.post2' expecttest cmake
python3 scripts/check_npu_env.py --build-only
export FLA_NPU_SOC=ascend910b
export MAX_JOBS=16
export CMAKE_BUILD_PARALLEL_LEVEL=16
export OPS_CPU_NUMBER=16
python3 -m pip wheel --no-build-isolation --no-deps . -w /work/wheels
python3 -m pip install --no-deps /work/wheels/flash_linear_attention_npu-*.whl
python3 scripts/check_packaged_wheel_api.py
