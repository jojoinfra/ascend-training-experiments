"""Use the pinned FLA-NPU release's validated 910B autograd entry point.

MindSpeed-MM's current generic wrapper references unavailable 950 composite
operators and undefined A2 helper functions. Its GatedDeltaNet model uses the
same layout and call signature as this upstream, numerically validated entry.
"""
import importlib.util
import os
import sys
from pathlib import Path

_path = Path(os.getenv('FLA_NPU_SOURCE', '/work/flash-linear-attention-npu')) / 'examples/flash_gated_delta_rule.py'
_spec = importlib.util.spec_from_file_location('qwen38_upstream_fla', _path)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
flash_gated_delta_rule = _module.flash_gated_delta_rule
