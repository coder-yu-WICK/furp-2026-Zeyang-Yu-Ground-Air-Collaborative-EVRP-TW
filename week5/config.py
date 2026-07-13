# -*- coding: utf-8 -*-
"""Week 5 config — imports from week4, adds Week 5 specific parameters."""

import os, sys

_W5 = os.path.dirname(os.path.abspath(__file__))
_W4 = os.path.join(_W5, '..', 'week4')

# Import week4's config using importlib (same pattern as week4→week3)
import importlib.util
spec = importlib.util.spec_from_file_location("week4_config", os.path.join(_W4, "config.py"))
w4cfg = importlib.util.module_from_spec(spec)
sys.modules['week4_config'] = w4cfg
spec.loader.exec_module(w4cfg)

# Re-export all constants from week4
for _name in dir(w4cfg):
    if not _name.startswith('_') and _name.isupper():
        globals()[_name] = getattr(w4cfg, _name)

# ── Week 5 Specific Parameters ────────────────────────────────────────

# Direction 1: TW-Aware Clustering
TW_AWARE = {
    'alpha': 1.0,        # spatial weight
    'beta': 0.5,         # temporal weight (0 = spatial-only, 1 = equal, >1 = TW-dominant)
    'tw_horizon_rc1': 120.0,
    'tw_horizon_rc2': 240.0,
}

# Direction 3: Drone Post-Processing
DRONE_PP = {
    'drone_speed': 50.0,       # km/h
    'drone_endurance': 4.0,    # km (medium), 6.0 for high
    'drone_capacity': 40.0,
    'drone_cost_rate': 1.0,    # per km
    'drone_fixed_cost': 0.0,
}

# Ablation study variants (original 4)
ABLATION_VARIANTS = [
    'baseline',          # spatial-only clustering + no drones (same as week4 POMO-MT)
    'tw_aware',          # TW-aware clustering + no drones
    'drone_only',        # spatial-only clustering + drone post-processing
    'tw_aware_drone',    # TW-aware clustering + drone post-processing
]

# Extended variants (Week 5.5)
EXTENDED_VARIANTS = [
    'adaptive_tw',         # Adaptive TW-aware clustering + no drones
    'adaptive_tw_drone',   # Adaptive TW-aware + drone post-processing + re-opt
    'angle',               # Angle-based petal clustering + no drones
    'angle_drone',         # Angle-based petal clustering + drone + re-opt
    'hybrid',              # Auto-select clustering + no drones
    'hybrid_drone',        # Hybrid + drone + re-opt (best overall)
    'hybrid_drone_no_reopt', # Hybrid + drone WITHOUT re-opt (ablation)
]

# Parameter sweep config
PARAM_SWEEP = {
    'max_gap_ratios': [0.2, 0.3, 0.4, 0.5, 0.6],
    'fleet_sizes': [0, 1, 2, 3],  # drones per truck
    'n_repeats': 5,                # runs per config for statistical significance
}

# Results directory
RESULTS_DIR = os.path.join(_W5, 'results')
