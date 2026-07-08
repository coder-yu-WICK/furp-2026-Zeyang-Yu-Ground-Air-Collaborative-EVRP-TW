# -*- coding: utf-8 -*-
"""Week 4 config — re-exports week3 constants."""

import os, sys

# Ensure we import from week3, not ourselves
_W4 = os.path.dirname(os.path.abspath(__file__))
_W3 = os.path.join(_W4, '..', 'week3')

# Use importlib to avoid self-import issues
import importlib.util
spec = importlib.util.spec_from_file_location("week3_config", os.path.join(_W3, "config.py"))
w3cfg = importlib.util.module_from_spec(spec)
sys.modules['week3_config'] = w3cfg
spec.loader.exec_module(w3cfg)

# Re-export all the constants needed by week4 modules
for _name in dir(w3cfg):
    if not _name.startswith('_') and _name.isupper():
        globals()[_name] = getattr(w3cfg, _name)

# Also export POMO-related (kept from original week4 config)
POMO = {
    'embedding_dim': 128,
    'encoder_layers': 6,
    'heads': 8,
    'qkv_dim': 16,
    'ff_hidden': 512,
    'logit_clipping': 10.0,
    'training_epochs': 80,
    'episodes_per_epoch': 400,
    'batch_size': 64,
    'lr': 1e-4,
    'weight_decay': 1e-6,
    'use_augmentation': True,
}
