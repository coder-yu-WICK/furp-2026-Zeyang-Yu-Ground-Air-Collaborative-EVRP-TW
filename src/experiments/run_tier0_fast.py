#!/usr/bin/env python3
"""Tier 0 Experiments — Optimized for speed."""

import sys, os, time, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.data_loader import load_instance_from_disk
from src.experiments.run_sota_expanded import run_one_method, METHOD_REGISTRY

REPRESENTATIVES = {
    'RC1': 'RC101', 'RC2': 'RC201',
    'R1': 'R101', 'R2': 'R201',
    'C1': 'C101', 'C2': 'C201',
}
SIZES = [50, 100]
BASE_SEED = 42

# Methods with per-method repeat counts
# Fast methods: 5 reps. Slow metaheuristics: 2 reps (they're deterministic-ish anyway)
METHODS = [
    ('ours_full', 5), ('ours_1drone', 5), ('ours_no_drone', 5),
    ('ours_no_edd', 5), ('ours_partial_edd', 5),
    ('nsga2', 2), ('paco', 1), ('ivnd', 2),
    ('sweep_nn', 5), ('cw_savings', 5), ('kmeans_nn', 5),
    ('kmeans_2opt', 5), ('sweep_pomo', 5), ('cw_pomo', 5),
]

configs = []
for tw_type, src_inst in REPRESENTATIVES.items():
    for nc in SIZES:
        ik = f'{src_inst}_{nc}c'
        try:
            load_instance_from_disk(ik)
        except FileNotFoundError:
            continue
        configs.append({
            'instance_key': ik, 'source_instance': src_inst,
            'n_customers': nc, 'tw_type': tw_type,
            'n_trucks': 4 if nc <= 50 else 6,
            'repair_mode': 'partial' if nc <= 50 else 'full',
        })

total_runs = sum(nr for _, nr in METHODS) * len(configs)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(outdir, exist_ok=True)

print(f'Tier 0 (fast): {len(configs)} configs x {len(METHODS)} methods = {total_runs} total runs')
print(f'Started: {timestamp}')
# Estimate: fast methods ~2s, slow ~40-200s
per_instance_est = sum(nr * (2 if not m.startswith(('nsga','paco','ivnd')) else (40 if m=='nsga2' else (216 if m=='paco' else 5)))
                       for m, nr in METHODS)
est_min = per_instance_est * len(configs) / 60
print(f'Estimated: ~{est_min:.0f} min')
print()

all_results = []
for ci, cfg in enumerate(configs):
    inst = load_instance_from_disk(cfg['instance_key'])
    print(f'[{ci+1}/{len(configs)}] {cfg["instance_key"]} ({cfg["tw_type"]}, {cfg["n_customers"]}c)')
    sys.stdout.flush()

    methods = {}
    for mi, (mkey, n_reps) in enumerate(METHODS):
        info = METHOD_REGISTRY[mkey]
        t0 = time.time()
        methods[mkey] = run_one_method(inst, cfg, mkey, n_repeats=n_reps, base_seed=BASE_SEED)
        elapsed = time.time() - t0
        m = methods[mkey]
        star = ' *' if m['feasibility_rate'] >= 0.99 and m['mean_tardiness'] < 1.0 else ''
        print(f'  [{mi+1}/{len(METHODS)}] {info["short"]:<20s} cost={m["mean_cost"]:>10.1f} tard={m["mean_tardiness"]:>8.1f} feas={m["feasibility_rate"]*100:>5.0f}% t={elapsed:>5.1f}s{star}')
        sys.stdout.flush()

    all_results.append({
        'instance_key': cfg['instance_key'],
        'source_instance': cfg['source_instance'],
        'n_customers': cfg['n_customers'],
        'tw_type': cfg['tw_type'],
        'n_trucks': cfg['n_trucks'],
        'methods': methods,
    })

    interim_path = os.path.join(outdir, f'week7_tier0_fast_{timestamp}.json')
    with open(interim_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

final_path = os.path.join(outdir, f'week7_tier0_fast_{timestamp}.json')
with open(final_path, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print(f'\nDone. Results: {final_path}')
