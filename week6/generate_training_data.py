#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 6 P1: Generate Training Data for Meta-Learner.

Runs ALL 10 W5 variants on ALL 12 instances (single repeat each)
to produce labeled training data: (features, best_variant) pairs.

Usage:
    python generate_training_data.py              # Full run (~30-60 min)
    python generate_training_data.py --quick      # 25c + 50c only (~15 min)
    python generate_training_data.py --test       # Single instance smoke test
"""

import json, os, sys, time
from datetime import datetime

_W6 = os.path.dirname(os.path.abspath(__file__))
_W5 = os.path.join(_W6, '..', 'week5')
_W4 = os.path.join(_W6, '..', 'week4')

sys.path.insert(0, _W5)
sys.path.insert(1, _W4)

from config import RC1_INSTANCES, RC2_INSTANCES, CUSTOMER_SIZES
from utils.data_loader import load_instance_from_disk, build_all_instances
from utils.problem_model import evaluate_solution_batch
from pomo_mt_improved import run_pomo_improved
from meta_learner import (
    extract_features, label_best_variant, ALL_VARIANTS, FEATURE_NAMES
)


def build_all_configs(sizes=None):
    """Build all instance configs for training data generation."""
    if sizes is None:
        sizes = [25, 50, 100]

    configs = []
    for src_inst in RC1_INSTANCES + RC2_INSTANCES:
        for nc in sizes:
            instance_key = f'{src_inst}_{nc}c'
            try:
                load_instance_from_disk(instance_key)
            except FileNotFoundError:
                continue

            n_trucks = 2 if nc <= 25 else 4
            tw_type = 'RC1' if src_inst.startswith('RC1') else 'RC2'
            configs.append({
                'instance_key': instance_key,
                'n_customers': nc,
                'tw_type': tw_type,
                'n_trucks': n_trucks,
                'label': f'{nc}c_{tw_type}',
            })
    return configs


def run_variant(instance, n_trucks, variant, seed=42):
    """Run a single W5 variant and return metrics."""
    try:
        r = run_pomo_improved(
            instance, n_runs=1, n_trucks=n_trucks,
            endurance='medium', seed=seed,
            variant=variant, tw_beta=0.4,
        )
        m = evaluate_solution_batch(r['solutions'])
        return {
            'mean_cost': m['mean_cost'],
            'mean_tardiness': m['mean_tardiness'],
            'feasibility_rate': m['feasibility_rate'],
            'avg_drone_missions': m.get('avg_drone_missions', 0),
            'mean_runtime': r.get('mean_runtime', 0),
        }
    except Exception as e:
        print(f'      ERROR [{variant}]: {e}')
        return {
            'mean_cost': 1e9,
            'mean_tardiness': 1e9,
            'feasibility_rate': 0.0,
            'avg_drone_missions': 0,
            'mean_runtime': 0,
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='25c + 50c only')
    parser.add_argument('--test', action='store_true', help='Single config smoke test')
    parser.add_argument('--variants', type=str, default=None,
                       help='Comma-separated variant list (default: all 10)')
    args = parser.parse_args()

    build_all_instances()

    if args.test:
        print("=== SMOKE TEST ===\n")
        inst = load_instance_from_disk('RC101_25c')
        feats = extract_features(inst)
        print('Features:')
        for name, val in zip(FEATURE_NAMES, feats):
            print(f'  {name:25s} = {val:.4f}')

        print('\nRunning 3 variants...')
        for v in ['baseline', 'hybrid', 'hybrid_drone']:
            t0 = time.time()
            m = run_variant(inst, n_trucks=2, variant=v, seed=42)
            dt = time.time() - t0
            print(f'  {v:20s}: cost={m["mean_cost"]:.0f}  tard={m["mean_tardiness"]:.0f}  '
                  f'feas={m["feasibility_rate"]*100:.0f}%  drones={m["avg_drone_missions"]:.1f}  '
                  f'time={dt:.1f}s')
        return

    # Build configs
    if args.quick:
        configs = build_all_configs(sizes=[25, 50])
        print('QUICK MODE: 25c + 50c')
    else:
        configs = build_all_configs()
        print('FULL MODE: 25c + 50c + 100c')

    variants = args.variants.split(',') if args.variants else ALL_VARIANTS
    print(f'Configs: {len(configs)}, Variants: {len(variants)}')
    print(f'Total runs: {len(configs)} × {len(variants)} = {len(configs) * len(variants)}')
    print('=' * 60)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_data = []

    for idx, cfg in enumerate(configs):
        inst = load_instance_from_disk(cfg['instance_key'])
        feats = extract_features(inst)
        t0_cfg = time.time()

        print(f'\n[{idx+1}/{len(configs)}] {cfg["label"]} ({cfg["instance_key"]})')

        variant_results = {}
        for vi, variant in enumerate(variants):
            t0 = time.time()
            m = run_variant(inst, cfg['n_trucks'], variant, seed=42)
            dt = time.time() - t0
            variant_results[variant] = m
            status = '✓' if m['feasibility_rate'] > 0 else '✗'
            print(f'  {vi+1:2d}/{len(variants)} {variant:22s} {status} '
                  f'cost={m["mean_cost"]:8.0f}  tard={m["mean_tardiness"]:8.0f}  '
                  f'feas={m["feasibility_rate"]*100:3.0f}%  '
                  f'drones={m["avg_drone_missions"]:5.1f}  {dt:.0f}s')

        best_variant = label_best_variant(variant_results, prefer='tardiness')
        # Also find best by cost (among those with acceptable tardiness)
        best_by_cost = label_best_variant(variant_results, prefer='cost')

        # What hybrid_rule would pick
        from meta_learner import hybrid_rule
        hybrid_pick = hybrid_rule(inst)

        cfg_time = time.time() - t0_cfg
        print(f'  → Best (tardiness): {best_variant}  |  Best (cost): {best_by_cost}  '
              f'|  Hybrid rule: {hybrid_pick}  |  cfg time: {cfg_time:.0f}s')

        all_data.append({
            'instance_key': cfg['instance_key'],
            'n_customers': cfg['n_customers'],
            'tw_type': cfg['tw_type'],
            'n_trucks': cfg['n_trucks'],
            'features': feats.tolist(),
            'feature_names': FEATURE_NAMES,
            'variant_results': variant_results,
            'best_variant_tardiness': best_variant,
            'best_variant_cost': best_by_cost,
            'hybrid_rule_pick': hybrid_pick,
        })

        # Interim save
        os.makedirs(os.path.join(_W6, 'results'), exist_ok=True)
        interim_path = os.path.join(_W6, 'results', f'meta_training_{timestamp}.json')
        with open(interim_path, 'w') as f:
            json.dump(all_data, f, indent=2)

    # Final save
    final_path = os.path.join(_W6, 'results', f'meta_training_{timestamp}.json')
    with open(final_path, 'w') as f:
        json.dump(all_data, f, indent=2)

    # Summary
    print(f'\n{"="*60}')
    print('SUMMARY')
    print(f'{"="*60}')
    from collections import Counter
    best_counts = Counter(d['best_variant_tardiness'] for d in all_data)
    print('Best variant (by tardiness) distribution:')
    for v, c in best_counts.most_common():
        print(f'  {v:25s}: {c}/{len(all_data)} instances')
    print(f'\nResults saved: {final_path}')

    # Print feature matrix summary
    print(f'\nFeature matrix: {len(all_data)} instances × {len(FEATURE_NAMES)} features')
    X = np.array([d['features'] for d in all_data])
    print(f'Feature ranges:')
    for fi, name in enumerate(FEATURE_NAMES):
        print(f'  {name:25s}: [{X[:, fi].min():.4f}, {X[:, fi].max():.4f}]')


if __name__ == '__main__':
    import numpy as np
    main()
