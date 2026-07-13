#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 5.5 Parameter Sweep — max_gap_ratio tuning & drone fleet sizing.

Usage:
    python param_sweep.py --tune-gap        # Grid search max_gap_ratio
    python param_sweep.py --fleet-size      # Drone fleet sizing analysis
    python param_sweep.py --all             # Both sweeps
    python param_sweep.py --quick           # Single instance quick test
"""

import json, os, sys, time
from datetime import datetime
import numpy as np

_W5 = os.path.dirname(os.path.abspath(__file__))
_W4 = os.path.join(_W5, '..', 'week4')
sys.path.insert(0, _W5)
sys.path.insert(1, _W4)

from config import PARAM_SWEEP, RESULTS_DIR
from utils.data_loader import load_instance_from_disk
from utils.problem_model import evaluate_solution_batch
from adaptive_clustering import cluster_with_params
from pomo_mt_improved import run_pomo_improved


# ── Sweep 1: max_gap_ratio Grid Search ────────────────────────────────

def sweep_max_gap_ratio(instance_key='RC201_25c', n_trucks=2,
                         ratios=None, n_repeats=3):
    """
    Grid search over max_gap_ratio values.

    Measures tardiness and cost for each ratio on RC2 instances
    (where temporal splitting actually matters).
    """
    if ratios is None:
        ratios = PARAM_SWEEP['max_gap_ratios']

    instance = load_instance_from_disk(instance_key)
    tw_type = instance.get('tw_type', 'RC1')
    # Use right strategy: adaptive_tw for RC1, tw_aware for RC2
    variant = 'adaptive_tw' if tw_type == 'RC1' else 'tw_aware'
    results = []

    print(f'\n{"="*60}')
    print(f'Max Gap Ratio Sweep: {instance_key} ({tw_type}, variant={variant})')
    print(f'Ratios: {ratios}, Repeats: {n_repeats}')
    print(f'{"="*60}')

    for ratio in ratios:
        costs, tards, runtimes, n_clusters_list = [], [], [], []

        for rep in range(n_repeats):
            t0 = time.time()
            try:
                r = run_pomo_improved(
                    instance, n_runs=1, n_trucks=n_trucks,
                    endurance='medium', seed=42 + rep,
                    variant=variant, tw_beta=ratio,
                )
                elapsed = time.time() - t0
                m = evaluate_solution_batch(r['solutions'])
                costs.append(m['mean_cost'])
                tards.append(m['mean_tardiness'])
                runtimes.append(elapsed)
            except Exception as e:
                print(f'  ratio={ratio} rep={rep} ERROR: {e}')

        if costs:
            print(f'  ratio={ratio:.1f}: cost={np.mean(costs):.0f}±{np.std(costs):.0f} '
                  f'tard={np.mean(tards):.0f}±{np.std(tards):.0f} '
                  f'time={np.mean(runtimes):.1f}s')
            results.append({
                'ratio': ratio,
                'mean_cost': np.mean(costs), 'std_cost': np.std(costs),
                'mean_tardiness': np.mean(tards), 'std_tardiness': np.std(tards),
                'mean_runtime': np.mean(runtimes),
            })

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(RESULTS_DIR, f'gap_sweep_{instance_key}_{timestamp}.json')
    with open(path, 'w') as f:
        json.dump({'instance': instance_key, 'n_trucks': n_trucks,
                    'n_repeats': n_repeats, 'results': results}, f, indent=2)
    print(f'\nSaved: {path}')
    return results


# ── Sweep 2: Drone Fleet Sizing ───────────────────────────────────────

def sweep_fleet_size(instance_key='RC201_50c', n_trucks=4,
                      max_per_truck=3, n_repeats=3):
    """
    Measure marginal benefit of k drones per truck.

    Runs drone insertion with per-truck limits and measures
    cost/tardiness for each fleet size.
    """
    from drone_reopt import evaluate_drone_fleet_size
    from pomo_mt_improved import ImprovedPOMOSolver

    instance = load_instance_from_disk(instance_key)
    results = []

    print(f'\n{"="*60}')
    print(f'Drone Fleet Size Sweep: {instance_key}')
    print(f'Max drones/truck: {max_per_truck}, Repeats: {n_repeats}')
    print(f'{"="*60}')

    # First get base truck routes without drones
    _W4_DIR = os.path.join(_W5, '..', 'week4')
    model_path = os.path.join(_W4_DIR, 'algorithms', 'pomo', 'checkpoints', 'best_model.pt')

    for rep in range(n_repeats):
        seed = 42 + rep

        # Get TW-aware truck routes (best for drone insertion)
        r = run_pomo_improved(
            instance, n_runs=1, n_trucks=n_trucks,
            endurance='medium', seed=seed,
            variant='hybrid', tw_beta=0.4,
        )
        if not r['solutions']:
            continue

        best_sol = r['pareto_front'][0]
        truck_routes = best_sol.truck_routes

        # Evaluate fleet sizes
        fleet_results = evaluate_drone_fleet_size(
            truck_routes, instance, max_drones_per_truck=max_per_truck,
            drone_endurance=4.0)

        for n_drones, cost, tard, missions in fleet_results:
            results.append({
                'rep': rep, 'n_drones': n_drones,
                'cost': cost, 'tardiness': tard,
                'missions_found': missions,
            })

    # Aggregate
    summary = {}
    for r in results:
        k = r['n_drones']
        if k not in summary:
            summary[k] = {'cost': [], 'tard': [], 'missions': []}
        summary[k]['cost'].append(r['cost'])
        summary[k]['tard'].append(r['tardiness'])
        summary[k]['missions'].append(r['missions_found'])

    print(f'\n{"Drones":>8s} {"Cost":>10s} {"Tardiness":>12s} {"Missions":>10s}')
    print('-' * 45)
    for k in sorted(summary.keys()):
        c_mean = np.mean(summary[k]['cost'])
        c_std = np.std(summary[k]['cost'])
        t_mean = np.mean(summary[k]['tard'])
        m_mean = np.mean(summary[k]['missions'])
        print(f'  {k:4d}   {c_mean:8.0f}±{c_std:4.0f}  {t_mean:8.0f}  {m_mean:8.1f}')

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(RESULTS_DIR, f'fleet_sweep_{instance_key}_{timestamp}.json')
    with open(path, 'w') as f:
        json.dump({'instance': instance_key, 'n_trucks': n_trucks,
                    'results': results, 'summary': {
                        str(k): {sk: np.mean(sv) if sv else 0
                                for sk, sv in v.items()}
                        for k, v in summary.items()
                    }}, f, indent=2)
    print(f'\nSaved: {path}')
    return results


# ── Main ──────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune-gap', action='store_true',
                       help='Grid search max_gap_ratio')
    parser.add_argument('--fleet-size', action='store_true',
                       help='Drone fleet sizing sweep')
    parser.add_argument('--all', action='store_true',
                       help='Both sweeps')
    parser.add_argument('--quick', action='store_true',
                       help='Single instance quick test')
    parser.add_argument('--instance', type=str, default=None,
                       help='Instance key (e.g. RC201_25c)')
    args = parser.parse_args()

    if args.quick:
        print('=== QUICK SWEEP TEST ===')
        sweep_max_gap_ratio('RC201_25c', n_trucks=2,
                           ratios=[0.3, 0.4, 0.5], n_repeats=1)
        return

    if args.tune_gap or args.all:
        for inst in ['RC201_25c', 'RC201_50c', 'RC202_25c', 'RC202_50c']:
            n_t = 2 if '25c' in inst else 4
            sweep_max_gap_ratio(inst, n_trucks=n_t,
                               ratios=PARAM_SWEEP['max_gap_ratios'],
                               n_repeats=PARAM_SWEEP['n_repeats'])

    if args.fleet_size or args.all:
        for inst in ['RC201_25c', 'RC201_50c', 'RC201_100c']:
            nc = int(inst.split('_')[1].replace('c', ''))
            n_t = {25: 2, 50: 4, 100: 4}[nc]
            sweep_fleet_size(inst, n_trucks=n_t,
                            max_per_truck=3,
                            n_repeats=PARAM_SWEEP['n_repeats'])

    if not (args.tune_gap or args.fleet_size or args.all or args.quick):
        parser.print_help()


if __name__ == '__main__':
    main()
