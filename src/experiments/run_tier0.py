#!/usr/bin/env python3
"""Tier 0 Experiments: 50c/100c, 6 Solomon types, 14 methods, 5 reps."""

import sys, os, time, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.data_loader import load_instance_from_disk
from src.experiments.run_sota_expanded import run_one_method, METHOD_REGISTRY, TIER_METHODS

# ── Config ──
REPRESENTATIVES = {
    'RC1': 'RC101', 'RC2': 'RC201',
    'R1': 'R101', 'R2': 'R201',
    'C1': 'C101', 'C2': 'C201',
}
SIZES = [50, 100]
METHODS = TIER_METHODS[0]  # 14 methods
N_REPEATS = 5
BASE_SEED = 42

# ── Build configs ──
configs = []
for tw_type, src_inst in REPRESENTATIVES.items():
    for nc in SIZES:
        instance_key = f'{src_inst}_{nc}c'
        try:
            load_instance_from_disk(instance_key)
        except FileNotFoundError:
            continue
        n_trucks = 4 if nc <= 50 else 6
        repair_mode = 'partial' if nc <= 50 else 'full'
        configs.append({
            'instance_key': instance_key,
            'source_instance': src_inst,
            'n_customers': nc,
            'tw_type': tw_type,
            'n_trucks': n_trucks,
            'repair_mode': repair_mode,
        })

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(outdir, exist_ok=True)

total_runs = len(configs) * len(METHODS) * N_REPEATS
print(f'Tier 0: {len(configs)} configs x {len(METHODS)} methods x {N_REPEATS} reps = {total_runs} runs')
print(f'Started: {timestamp}\n')

all_results = []
for ci, cfg in enumerate(configs):
    inst = load_instance_from_disk(cfg['instance_key'])
    print(f'[{ci+1}/{len(configs)}] {cfg["instance_key"]} ({cfg["tw_type"]}, {cfg["n_customers"]}c, {cfg["n_trucks"]} trucks)')
    sys.stdout.flush()

    methods = {}
    for mi, mkey in enumerate(METHODS):
        info = METHOD_REGISTRY[mkey]
        t0 = time.time()
        methods[mkey] = run_one_method(inst, cfg, mkey, n_repeats=N_REPEATS, base_seed=BASE_SEED)
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

    # Interim save
    interim_path = os.path.join(outdir, f'week7_tier0_interim_{timestamp}.json')
    with open(interim_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

# ── Final save ──
final_path = os.path.join(outdir, f'week7_tier0_{timestamp}.json')
with open(final_path, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

# ── Summary ──
print('\n' + '=' * 90)
print('TIER 0 SUMMARY')
print('=' * 90)

for r in all_results:
    print(f'\n{r["instance_key"]} ({r["tw_type"]}, {r["n_customers"]}c):')
    header = f'  {"Method":<22s} {"Cost":>10s} {"Tard":>10s} {"Feas":>7s} {"Time":>8s}'
    print(header)
    print('  ' + '-' * 65)
    for mkey in METHODS:
        m = r['methods'][mkey]
        info = METHOD_REGISTRY[mkey]
        star = ' *' if m['feasibility_rate'] >= 0.99 and m['mean_tardiness'] < 1.0 else ''
        print(f'  {info["short"]:<22s} {m["mean_cost"]:>10.1f} {m["mean_tardiness"]:>10.1f} {m["feasibility_rate"]*100:>6.0f}% {m["mean_runtime"]:>7.1f}s{star}')

# ── Statistical summary ──
print('\n' + '=' * 90)
print('FEASIBILITY & TARDINESS SUMMARY')
print('=' * 90)
for mkey in METHODS:
    info = METHOD_REGISTRY[mkey]
    feas_rates = []
    tards = []
    costs = []
    for r in all_results:
        m = r['methods'][mkey]
        if m['mean_cost'] < 1e8:
            feas_rates.append(m['feasibility_rate'])
            tards.append(m['mean_tardiness'])
            costs.append(m['mean_cost'])
    avg_feas = sum(feas_rates)/len(feas_rates)*100 if feas_rates else 0
    avg_tard = sum(tards)/len(tards) if tards else 0
    avg_cost = sum(costs)/len(costs) if costs else 0
    perfect = sum(1 for f, t in zip(feas_rates, tards) if f >= 0.99 and t < 1.0)
    print(f'  {info["short"]:<22s} avg_cost={avg_cost:>8.1f} avg_tard={avg_tard:>8.1f} avg_feas={avg_feas:>5.1f}% perfect={perfect}/{len(feas_rates)}')

print(f'\nDone. Results saved to: {final_path}')
