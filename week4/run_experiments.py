#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
POMO Multi-Truck Experiment Runner.

Runs POMO-MT (cluster-first, route-second) on all 48 configurations.
No-Drone baseline loaded from Week 3 results JSON (no re-running needed).

Usage:
    python run_experiments.py              # Full 48-config run
    python run_experiments.py --quick      # 25c only, 1 repeat
    python run_experiments.py --test       # Single config smoke test
"""

import json, os, sys, time
from datetime import datetime

_W4 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _W4)

# Import week3 constants without path conflict
from config import (
    RESULTS_DIR, RC1_INSTANCES, RC2_INSTANCES,
    CUSTOMER_SIZES, VEHICLE_CONFIGS, DRONE_ENDURANCE, N_REPEATS,
)

# Add week3 path for data loading (NOT for algorithms imports)
_W3 = os.path.join(_W4, '..', 'week3')
# Put week3 AFTER week4 so week4's algorithms is found first
if _W3 in sys.path:
    sys.path.remove(_W3)
sys.path.insert(1, _W3)  # after week4

from utils.data_loader import load_instance_from_disk, build_all_instances
from utils.problem_model import evaluate_solution_batch, extract_pareto_front
from pomo_multi_truck import run_pomo_multitruck

# Load week3 No-Drone results for comparison
_W3_RESULTS = os.path.join(_W3, 'results', 'results_20260702_152443_hv_fixed.json')
_w3_data = {}
if os.path.exists(_W3_RESULTS):
    with open(_W3_RESULTS) as f:
        for exp in json.load(f):
            _w3_data[exp['label']] = exp['methods'].get('No-Drone', None)
    print(f'Loaded {len(_w3_data)} No-Drone baselines from Week 3')


def build_matrix():
    """Build 48-config experiment matrix (same as week3)."""
    matrix = []
    for src_inst in RC1_INSTANCES + RC2_INSTANCES:
        for n_cust in CUSTOMER_SIZES:
            instance_key = f'{src_inst}_{n_cust}c'
            try:
                load_instance_from_disk(instance_key)
            except FileNotFoundError:
                print(f'  SKIP: Instance {instance_key} not found')
                continue

            for n_t, n_d in VEHICLE_CONFIGS[n_cust]:
                for end_name, end_val in DRONE_ENDURANCE.items():
                    tw_type = 'RC1' if src_inst.startswith('RC1') else 'RC2'
                    label = f'{n_cust}c_{tw_type}_{end_name}_{n_t}T+{n_d}D'
                    matrix.append({
                        'instance_key': instance_key,
                        'source_instance': src_inst,
                        'n_customers': n_cust,
                        'tw_type': tw_type,
                        'n_trucks': n_t,
                        'n_drones': n_d,
                        'endurance': end_val,
                        'endurance_name': end_name,
                        'label': label,
                    })
    return matrix


def run_config(exp, n_repeats):
    """Run POMO-MT on a single config. No-Drone from stored data."""
    instance = load_instance_from_disk(exp['instance_key'])
    n_t = exp['n_trucks']
    results = {}

    # POMO Multi-Truck
    t0 = time.time()
    try:
        r = run_pomo_multitruck(instance, n_runs=n_repeats, n_trucks=n_t, seed=42)
        elapsed = time.time() - t0
        m = evaluate_solution_batch(r['solutions'])
        m['mean_runtime'] = r['mean_runtime']
        m['std_runtime'] = r['std_runtime']
        # Convert Pareto front objects to serializable tuples
        if 'pareto_front' in m:
            m['pareto_points'] = [(s.cost, s.tardiness) for s in m['pareto_front']]
            del m['pareto_front']
        results['POMO-MT'] = m
        print(f'    POMO-MT: cost={m["mean_cost"]:.0f} tard={m["mean_tardiness"]:.0f} '
              f'feas={m["feasibility_rate"]*100:.0f}% time={elapsed:.1f}s')
    except Exception as e:
        print(f'    POMO-MT ERROR: {e}')
        import traceback; traceback.print_exc()

    # No-Drone from stored week3 data
    nd = _w3_data.get(exp['label'])
    if nd:
        results['No-Drone'] = {
            'mean_cost': nd['mean_cost'],
            'std_cost': nd.get('std_cost', 0),
            'mean_tardiness': nd['mean_tardiness'],
            'std_tardiness': nd.get('std_tardiness', 0),
            'feasibility_rate': nd['feasibility_rate'],
            'hypervolume': nd.get('hypervolume', 0),
            'pareto_points': nd.get('pareto_points', []),
            'drone_solution_pct': nd.get('drone_solution_pct', 0),
            'avg_drone_missions': nd.get('avg_drone_missions', 0),
            'pareto_size': nd.get('pareto_size', 0),
            'best_cost': nd.get('best_cost', nd['mean_cost']),
            'best_tardiness': nd.get('best_tardiness', nd['mean_tardiness']),
            'mean_runtime': nd.get('mean_runtime', 0),
            'std_runtime': nd.get('std_runtime', 0),
        }
    else:
        print('    No-Drone: not found in Week 3 results')

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='25c only, 1 repeat')
    parser.add_argument('--test', action='store_true', help='Single config smoke test')
    args = parser.parse_args()

    if args.test:
        print("=== SMOKE TEST ===\n")
        matrix = build_matrix()
        exp = matrix[0]
        print(f'Config: {exp["label"]}')
        results = run_config(exp, n_repeats=1)
        print('\nDone.')
        return

    if args.quick:
        import config as cfg
        cfg.CUSTOMER_SIZES = [25]
        n_repeats = 1
        print('QUICK MODE: 25c only, 1 repeat')
    else:
        n_repeats = N_REPEATS

    print('=' * 60)
    print('Building instances...')
    build_all_instances()
    print('=' * 60)

    matrix = build_matrix()
    print(f'Total configs: {len(matrix)}, Repeats: {n_repeats}')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_results = []

    for idx, exp in enumerate(matrix):
        print(f'\n{"="*60}')
        print(f'[{idx+1}/{len(matrix)}] {exp["label"]}')
        print(f'  Instance={exp["instance_key"]} Trucks={exp["n_trucks"]}')
        print(f'{"="*60}')

        method_results = run_config(exp, n_repeats)
        exp_result = {
            'label': exp['label'],
            'instance_key': exp['instance_key'],
            'source_instance': exp['source_instance'],
            'n_customers': exp['n_customers'],
            'tw_type': exp['tw_type'],
            'n_trucks': exp['n_trucks'],
            'n_drones': exp['n_drones'],
            'endurance': exp['endurance'],
            'endurance_name': exp['endurance_name'],
            'methods': method_results,
        }
        all_results.append(exp_result)

        os.makedirs(RESULTS_DIR, exist_ok=True)
        interim_path = os.path.join(RESULTS_DIR, f'interim_mt_{timestamp}.json')
        with open(interim_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    final_path = os.path.join(RESULTS_DIR, f'pomo_multitruck_{timestamp}.json')
    with open(final_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f'\n{"="*60}')
    print(f'Done! Results: {final_path}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
