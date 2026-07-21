#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 5 Experiment Runner — Ablation Study.

Compares 4 original + 6 extended variants of POMO-MT:
  1. baseline          — spatial-only clustering, no drones (Week 4 baseline)
  2. tw_aware          — TW-aware clustering, no drones
  3. drone_only        — spatial-only clustering, drone post-processing
  4. tw_aware_drone    — TW-aware clustering, drone post-processing
  5. adaptive_tw       — Adaptive TW-aware clustering, no drones
  6. adaptive_tw_drone — Adaptive TW-aware + drone + re-opt
  7. angle             — Angle-based petal clustering, no drones
  8. angle_drone       — Angle-based petal clustering + drone + re-opt
  9. hybrid            — Auto-select clustering, no drones
  10. hybrid_drone     — Hybrid + drone + re-opt

Also compares against Week 3 methods (No-Drone, P-ACO, NSGA-II, IVND).

Usage:
    python run_experiments.py                      # Full 48-config × 4 variants
    python run_experiments.py --extended           # Full 48-config × 10 variants
    python run_experiments.py --quick              # 25c only, 1 repeat
    python run_experiments.py --test               # Single config smoke test
    python run_experiments.py --variants baseline,hybrid_drone  # Specific variants
"""

import json, os, sys, time
from datetime import datetime

_W5 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _W5)

from config import (
    RESULTS_DIR, RC1_INSTANCES, RC2_INSTANCES,
    CUSTOMER_SIZES, VEHICLE_CONFIGS, DRONE_ENDURANCE, N_REPEATS,
    ABLATION_VARIANTS, EXTENDED_VARIANTS, PARAM_SWEEP,
)

# Add week4 for data loading
_W4 = os.path.join(_W5, '..', 'week4')
if _W4 in sys.path:
    sys.path.remove(_W4)
sys.path.insert(1, _W4)

# Import week4 utils
from utils.data_loader import load_instance_from_disk, build_all_instances
from utils.problem_model import evaluate_solution_batch, extract_pareto_front
from pomo_mt_improved import run_pomo_improved

# Load Week 3 baseline data
_W3 = os.path.join(_W5, '..', 'week3')
_W3_RESULTS = os.path.join(_W3, 'results', 'results_20260702_152443_hv_fixed.json')
_w3_data = {}
if os.path.exists(_W3_RESULTS):
    with open(_W3_RESULTS) as f:
        for exp in json.load(f):
            _w3_data[exp['label']] = exp['methods']
    print(f'Loaded {len(_w3_data)} Week 3 baselines')

# Load Week 4 POMO-MT baseline
_W4_RESULTS = os.path.join(_W4, 'results', 'pomo_multitruck_20260707_152155.json')
_w4_data = {}
if os.path.exists(_W4_RESULTS):
    with open(_W4_RESULTS) as f:
        for exp in json.load(f):
            _w4_data[exp['label']] = exp['methods'].get('POMO-MT', None)
    print(f'Loaded {len(_w4_data)} Week 4 POMO-MT baselines')


def build_matrix():
    """Build 48-config experiment matrix."""
    matrix = []
    for src_inst in RC1_INSTANCES + RC2_INSTANCES:
        for n_cust in CUSTOMER_SIZES:
            instance_key = f'{src_inst}_{n_cust}c'
            try:
                load_instance_from_disk(instance_key)
            except FileNotFoundError:
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


def run_config(exp, n_repeats, variants):
    """Run all variants on a single config."""
    instance = load_instance_from_disk(exp['instance_key'])
    n_t = exp['n_trucks']
    results = {}

    for variant in variants:
        t0 = time.time()
        try:
            r = run_pomo_improved(
                instance, n_runs=n_repeats, n_trucks=n_t,
                endurance=exp['endurance_name'], seed=42,
                variant=variant, tw_beta=0.5,
            )
            elapsed = time.time() - t0
            m = evaluate_solution_batch(r['solutions'])
            m['mean_runtime'] = r['mean_runtime']
            m['std_runtime'] = r['std_runtime']
            if 'pareto_front' in m:
                m['pareto_points'] = [(s.cost, s.tardiness) for s in m['pareto_front']]
                m['pareto_size'] = len(m['pareto_front'])
                del m['pareto_front']
            results[variant] = m

            n_drone = m.get('avg_drone_missions', 0)
            print(f'    [{variant:20s}] cost={m["mean_cost"]:.0f} tard={m["mean_tardiness"]:.0f} '
                  f'feas={m["feasibility_rate"]*100:.0f}% drones={n_drone:.1f} '
                  f'pareto={m.get("pareto_size",0)}pts time={elapsed:.1f}s')
        except Exception as e:
            print(f'    [{variant:20s}] ERROR: {e}')
            import traceback; traceback.print_exc()

    # Add Week 3 methods from stored data
    w3_methods = _w3_data.get(exp['label'], {})
    for method_name in ['No-Drone', 'P-ACO', 'NSGA-II', 'IVND']:
        md = w3_methods.get(method_name)
        if md:
            results[method_name] = {
                'mean_cost': md['mean_cost'],
                'std_cost': md.get('std_cost', 0),
                'mean_tardiness': md['mean_tardiness'],
                'std_tardiness': md.get('std_tardiness', 0),
                'feasibility_rate': md['feasibility_rate'],
                'hypervolume': md.get('hypervolume', 0),
                'pareto_points': md.get('pareto_points', []),
                'drone_solution_pct': md.get('drone_solution_pct', 0),
                'avg_drone_missions': md.get('avg_drone_missions', 0),
                'pareto_size': md.get('pareto_size', 0),
                'best_cost': md.get('best_cost', md['mean_cost']),
                'best_tardiness': md.get('best_tardiness', md['mean_tardiness']),
                'mean_runtime': md.get('mean_runtime', 0),
                'std_runtime': md.get('std_runtime', 0),
            }

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='25c only, 1 repeat')
    parser.add_argument('--test', action='store_true', help='Single config smoke test')
    parser.add_argument('--extended', action='store_true',
                        help='Include all 10 extended variants')
    parser.add_argument('--repeats', type=int, default=None,
                        help=f'Repeats per config (default from config: {PARAM_SWEEP["n_repeats"]})')
    parser.add_argument('--variants', type=str,
                        default='baseline,tw_aware,drone_only,tw_aware_drone',
                        help='Comma-separated variant names')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='TW-aware beta parameter')
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(',')]

    # Add extended variants if requested
    if args.extended:
        variants = [v for v in variants if v not in EXTENDED_VARIANTS]  # dedup
        variants = ABLATION_VARIANTS + EXTENDED_VARIANTS

    if args.test:
        print("=== SMOKE TEST ===\n")
        matrix = build_matrix()
        exp = matrix[0]
        print(f'Config: {exp["label"]} (instance={exp["instance_key"]})')
        print(f'Variants: {variants}')
        results = run_config(exp, n_repeats=1, variants=variants)
        print('\nSummary:')
        for var, m in results.items():
            print(f'  {var:20s}: cost={m.get("mean_cost","?"):.0f} '
                  f'tard={m.get("mean_tardiness","?"):.0f} '
                  f'feas={m.get("feasibility_rate",0)*100:.0f}%')
        return

    if args.quick:
        import config as cfg
        cfg.CUSTOMER_SIZES[:] = [25]  # mutate in-place so imported ref sees it
        n_repeats = args.repeats if args.repeats is not None else 1
        print(f'QUICK MODE: 25c only, {n_repeats} repeats')
    else:
        n_repeats = args.repeats if args.repeats is not None else PARAM_SWEEP['n_repeats']

    print('=' * 60)
    print(f'Week 5 Ablation Study: {len(variants)} variants × 48 configs')
    print(f'Variants: {variants}')
    print(f'Repeats: {n_repeats}')
    print('=' * 60)

    build_all_instances()
    matrix = build_matrix()
    print(f'Total configs: {len(matrix)}')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_results = []

    for idx, exp in enumerate(matrix):
        print(f'\n{"="*60}')
        print(f'[{idx+1}/{len(matrix)}] {exp["label"]}')
        print(f'  Instance={exp["instance_key"]} Trucks={exp["n_trucks"]}')
        print(f'{"="*60}')

        method_results = run_config(exp, n_repeats, variants)
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
        interim_path = os.path.join(RESULTS_DIR, f'interim_w5_{timestamp}.json')
        with open(interim_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    final_path = os.path.join(RESULTS_DIR, f'week5_ablation_{timestamp}.json')
    with open(final_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f'\n{"="*60}')
    print(f'Done! Results: {final_path}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
