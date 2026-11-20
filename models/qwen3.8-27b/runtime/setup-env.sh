#!/usr/bin/env bash
source /usr/local/Ascend/cann/set_env.sh
set -euo pipefail
cd /work
mkdir -p logs wheels
apt-get update
apt-get install -y gawk libssl-dev libgmp-dev
python3 -m pip install --disable-pip-version-check \
  pybind11 ninja wheel packaging six regex einops sentencepiece protobuf \
  'transformers==5.2.0' 'numpy==1.26.4' 'scipy==1.13.1' pandas scikit-learn gpytorch SQLAlchemy \
  'triton-ascend==3.2.1' --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple
# MindSpeed and Megatron are loaded from pinned source directories by PYTHONPATH.
# Avoid the optional editable-package build, which traverses all bundled sources.
python3 -m pip list --format=json > /work/logs/packages.json
