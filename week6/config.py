# -*- coding: utf-8 -*-
"""Week 6 config — imports from week5, adds IVND repair parameters."""

import os, sys

_W6 = os.path.dirname(os.path.abspath(__file__))
_W5 = os.path.join(_W6, '..', 'week5')
_W4 = os.path.join(_W6, '..', 'week4')

# Week 5 must come first for config resolution
if _W5 in sys.path: sys.path.remove(_W5)
sys.path.insert(0, _W5)
if _W4 in sys.path: sys.path.remove(_W4)
sys.path.insert(1, _W4)

# Import week5 config
import importlib.util
spec = importlib.util.spec_from_file_location("week5_config", os.path.join(_W5, "config.py"))
w5cfg = importlib.util.module_from_spec(spec)
sys.modules['week5_config'] = w5cfg
spec.loader.exec_module(w5cfg)

for _name in dir(w5cfg):
    if not _name.startswith('_') and _name.isupper():
        globals()[_name] = getattr(w5cfg, _name)

# ── Week 6: IVND Repair Parameters ───────────────────────────────────

IVND_REPAIR = {
    'max_iterations': 500,        # Focused repair (vs 5000+ for full IVND)
    'tabu_tenure': 15,
    'temperature': 0.5,           # Lower temp → greedier acceptance
    'cooling_rate': 0.95,
    'improvement_threshold': 0.0, # Accept any tardiness reduction
}

# Pipeline comparison variants
PIPELINE_VARIANTS = [
    'w5_baseline',          # W5 hybrid_drone without repair
    'w5_plus_repair',       # W5 hybrid_drone + IVND repair
]

# Results directory
RESULTS_DIR = os.path.join(_W6, 'results')
