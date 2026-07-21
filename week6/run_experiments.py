#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 6 Experiment Runner — W5 Pipeline vs W5 + IVND Repair.

Tests whether IVND repair reduces tardiness on the POMO + drone solution,
especially on the 100c instances where W5 degrades.

Usage:
    python run_experiments.py                    # Full comparison
    python run_experiments.py --quick            # 25c + 50c only
    python run_experiments.py --test             # Single config smoke test
    python run_experiments.py --focus 100c       # Only 100c (where repair matters most)
"""

import json, os, sys, time
from datetime import datetime

_W6 = os.path.dirname(os.path.abspath(__file__))
_W5 = os.path.join(_W6, '..', 'week5')
_W4 = os.path.join(_W6, '..', 'week4')

sys.path.insert(0, _W5)
sys.path.insert(1, _W4)
sys.path.insert(0, _W6)

from config import RESULTS_DIR, RC1_INSTANCES, RC2_INSTANCES, CUSTOMER_SIZES
from utils.data_loader import load_instance_from_disk, build_all_instances
from utils.problem_model import evaluate_solution_batch
from pipeline import run_pipeline
from pomo_mt_improved import run_pomo_improved


def build_configs(sizes=None):
    """Build experiment configs. Filters to 25c/50c/100c with medium endurance."""
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='25c + 50c only')
    parser.add_argument('--test', action='store_true', help='Single config smoke test')
    parser.add_argument('--focus', type=str, default=None,
                       help='Only specific scale (25c, 50c, 100c)')
    parser.add_argument('--repeats', type=int, default=3,
                       help='Runs per config (default: 3)')
    args = parser.parse_args()

    if args.test:
        print("=== SMOKE TEST ===\n")
        inst = load_instance_from_disk('RC201_100c')
        print(f'Config: RC201_100c, 4 trucks')

        # W5 baseline
        t0 = time.time()
        r = run_pomo_improved(inst, n_runs=1, n_trucks=4, endurance='medium',
                             seed=42, variant='hybrid_drone', tw_beta=0.4)
        m = evaluate_solution_batch(r['solutions'])
        t_w5 = time.time() - t0

        # W5 + repair
        t0 = time.time()
        r2 = run_pipeline(inst, n_trucks=4, variant='hybrid_drone',
                          use_repair=True, n_runs=1, seed=42)
        m2 = evaluate_solution_batch(r2['solutions'])
        t_r = time.time() - t0

        print(f'  W5 baseline:    cost={m["mean_cost"]:.0f}  tard={m["mean_tardiness"]:.0f}  '
              f'feas={m["feasibility_rate"]*100:.0f}%  drones={m.get("avg_drone_missions",0):.1f}  time={t_w5:.1f}s')
        print(f'  W5 + repair:    cost={m2["mean_cost"]:.0f}  tard={m2["mean_tardiness"]:.0f}  '
              f'feas={m2["feasibility_rate"]*100:.0f}%  drones={m2.get("avg_drone_missions",0):.1f}  time={t_r:.1f}s')
        if r2.get('repair_stats'):
            rs = r2['repair_stats']
            print(f'  Repair stats: tard_before={rs["avg_tardiness_before"]:.0f} → '
                  f'tard_after={rs["avg_tardiness_after"]:.0f} '
                  f'(-{rs["avg_tardiness_reduction"]:.0f}), moves={rs["avg_moves_accepted"]:.0f}')
        return

    # Build config list
    if args.focus:
        scale = int(args.focus.replace('c', ''))
        configs = build_configs(sizes=[scale])
        print(f'FOCUS MODE: {scale}c only')
    elif args.quick:
        configs = build_configs(sizes=[25, 50])
        print('QUICK MODE: 25c + 50c')
    else:
        configs = build_configs()
        print('FULL MODE: 25c + 50c + 100c')

    n_repeats = args.repeats

    print(f'Configs: {len(configs)}, Repeats: {n_repeats}')
    print(f'Comparing: W5 baseline vs W5 + IVND repair')
    print('=' * 60)

    build_all_instances()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_results = []

    for idx, cfg in enumerate(configs):
        inst = load_instance_from_disk(cfg['instance_key'])
        print(f'\n[{idx+1}/{len(configs)}] {cfg["label"]} (inst={cfg["instance_key"]})')

        methods = {}

        # Variant A: W5 baseline
        costs, tards, drones, feas, rts = [], [], [], [], []
        for rep in range(n_repeats):
            t0 = time.time()
            r = run_pomo_improved(inst, n_runs=1, n_trucks=cfg['n_trucks'],
                                 endurance='medium', seed=42+rep,
                                 variant='hybrid_drone', tw_beta=0.4)
            m = evaluate_solution_batch(r['solutions'])
            costs.append(m['mean_cost']); tards.append(m['mean_tardiness'])
            drones.append(m.get('avg_drone_missions', 0))
            feas.append(m['feasibility_rate']); rts.append(time.time() - t0)

        methods['w5_baseline'] = {
            'mean_cost': sum(costs)/len(costs), 'std_cost': (sum((c-sum(costs)/len(costs))**2 for c in costs)/len(costs))**0.5 if len(costs)>1 else 0,
            'mean_tardiness': sum(tards)/len(tards), 'std_tardiness': (sum((t-sum(tards)/len(tards))**2 for t in tards)/len(tards))**0.5 if len(tards)>1 else 0,
            'avg_drone_missions': sum(drones)/len(drones),
            'feasibility_rate': sum(feas)/len(feas),
            'mean_runtime': sum(rts)/len(rts),
        }

        # Variant B: W5 + repair
        costs, tards, drones, feas, rts = [], [], [], [], []
        treps = []
        for rep in range(n_repeats):
            t0 = time.time()
            r = run_pipeline(inst, n_trucks=cfg['n_trucks'], variant='hybrid',
                            use_repair=True, n_runs=1, seed=42+rep)
            m = evaluate_solution_batch(r['solutions'])
            costs.append(m['mean_cost']); tards.append(m['mean_tardiness'])
            drones.append(m.get('avg_drone_missions', 0))
            feas.append(m['feasibility_rate']); rts.append(time.time() - t0)
            if r.get('repair_stats'):
                treps.append(r['repair_stats']['avg_tardiness_reduction'])

        methods['w5_plus_repair'] = {
            'mean_cost': sum(costs)/len(costs), 'std_cost': (sum((c-sum(costs)/len(costs))**2 for c in costs)/len(costs))**0.5 if len(costs)>1 else 0,
            'mean_tardiness': sum(tards)/len(tards), 'std_tardiness': (sum((t-sum(tards)/len(tards))**2 for t in tards)/len(tards))**0.5 if len(tards)>1 else 0,
            'avg_drone_missions': sum(drones)/len(drones),
            'feasibility_rate': sum(feas)/len(feas),
            'mean_runtime': sum(rts)/len(rts),
            'avg_tardiness_repaired': sum(treps)/len(treps) if treps else 0,
        }

        # Print comparison
        b = methods['w5_baseline']
        rp = methods['w5_plus_repair']
        t_red = (b['mean_tardiness'] - rp['mean_tardiness']) / max(b['mean_tardiness'], 1) * 100
        print(f'  W5 baseline:   cost={b["mean_cost"]:.0f}±{b["std_cost"]:.0f}  '
              f'tard={b["mean_tardiness"]:.0f}±{b["std_tardiness"]:.0f}  '
              f'feas={b["feasibility_rate"]*100:.0f}%  drones={b["avg_drone_missions"]:.1f}')
        print(f'  W5 + repair:   cost={rp["mean_cost"]:.0f}±{rp["std_cost"]:.0f}  '
              f'tard={rp["mean_tardiness"]:.0f}±{rp["std_tardiness"]:.0f}  '
              f'feas={rp["feasibility_rate"]*100:.0f}%  drones={rp["avg_drone_missions"]:.1f}')
        print(f'  Improvement:   tard {t_red:+.0f}%,  '
              f'repaired={rp.get("avg_tardiness_repaired",0):.0f} units/run')

        all_results.append({
            'label': cfg['label'],
            'instance_key': cfg['instance_key'],
            'n_customers': cfg['n_customers'],
            'tw_type': cfg['tw_type'],
            'n_trucks': cfg['n_trucks'],
            'methods': methods,
        })

        # Interim save
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(os.path.join(RESULTS_DIR, f'w6_interim_{timestamp}.json'), 'w') as f:
            json.dump(all_results, f, indent=2)

    # Final save
    final_path = os.path.join(RESULTS_DIR, f'week6_pipeline_{timestamp}.json')
    with open(final_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f'\n{"="*60}')
    print('SUMMARY')
    print(f'{"="*60}')
    for twt in ['RC1', 'RC2']:
        for nc in sorted(set(e['n_customers'] for e in all_results)):
            exps = [e for e in all_results if e['tw_type'] == twt and e['n_customers'] == nc]
            if not exps: continue
            b_tard = sum(e['methods']['w5_baseline']['mean_tardiness'] for e in exps) / len(exps)
            r_tard = sum(e['methods']['w5_plus_repair']['mean_tardiness'] for e in exps) / len(exps)
            b_cost = sum(e['methods']['w5_baseline']['mean_cost'] for e in exps) / len(exps)
            r_cost = sum(e['methods']['w5_plus_repair']['mean_cost'] for e in exps) / len(exps)
            red = (b_tard - r_tard) / max(b_tard, 1) * 100
            print(f'  {twt} {nc}c: tard {b_tard:.0f}→{r_tard:.0f} ({red:+.0f}%),  '
                  f'cost {b_cost:.0f}→{r_cost:.0f} ({r_cost-b_cost:+.0f})')

    print(f'\nResults: {final_path}')


if __name__ == '__main__':
    main()
