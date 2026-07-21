#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 6-7: ALNS Baseline — Experiment Runner.

Compares ALNS against W5 pipeline + EDD repair on the same 12 instances.

Usage:
    python week6/run_alns_baseline.py                     # Full comparison
    python week6/run_alns_baseline.py --test              # Smoke test
    python week6/run_alns_baseline.py --quick             # 25c + 50c only
    python week6/run_alns_baseline.py --focus RC202_100c  # Single instance
"""

import json, os, sys, time
from datetime import datetime

_W6 = os.path.dirname(os.path.abspath(__file__))
_W5 = os.path.join(_W6, '..', 'week5')
_W4 = os.path.join(_W6, '..', 'week4')

sys.path.insert(0, _W5)
sys.path.insert(1, _W4)
sys.path.insert(0, _W6)

from config import RC1_INSTANCES, RC2_INSTANCES
from utils.data_loader import load_instance_from_disk, build_all_instances
from utils.problem_model import evaluate_solution_batch
from pipeline import run_pipeline
from pomo_mt_improved import run_pomo_improved
from alns_baseline import run_alns


def build_configs(sizes=None):
    """Build experiment configs."""
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
    parser.add_argument('--test', action='store_true', help='Smoke test on RC201_100c')
    parser.add_argument('--quick', action='store_true', help='25c + 50c only')
    parser.add_argument('--focus', type=str, default=None, help='Single instance key')
    parser.add_argument('--alns-iters', type=int, default=2000,
                       help='ALNS iterations (default: 2000)')
    parser.add_argument('--output-dir', type=str,
                       default=os.path.join(_W6, 'results'))
    args = parser.parse_args()

    # ── Smoke test ──
    if args.test:
        print("=== ALNS SMOKE TEST ===\n")
        build_all_instances()
        inst = load_instance_from_disk('RC201_100c')

        # Get W5 baseline as starting point
        r_w5_init = run_pomo_improved(inst, n_runs=1, n_trucks=4, endurance='medium',
                                      seed=42, variant='hybrid_drone', tw_beta=0.4)
        init_sol = min(r_w5_init['pareto_front'], key=lambda s: s.tardiness)
        print(f"Starting from W5: cost={init_sol.cost:.0f}, tard={init_sol.tardiness:.0f}")

        print(f"Running ALNS ({args.alns_iters} iterations)...")
        t0 = time.time()
        from alns_baseline import ALNSSolver
        solver = ALNSSolver(inst, n_trucks=4, seed=42)
        r_alns = solver.solve(max_iter=args.alns_iters, initial_solution=init_sol)
        # Wrap in expected format
        r_alns['solutions'] = r_alns.get('solutions', [r_alns.get('best_solution', init_sol)])
        m_alns = evaluate_solution_batch(r_alns['solutions'])
        elapsed = time.time() - t0

        print(f"\n  ALNS:        cost={m_alns['mean_cost']:.0f}  "
              f"tard={m_alns['mean_tardiness']:.0f}  "
              f"drones={m_alns.get('avg_drone_missions', 0):.1f}  "
              f"feas={m_alns['feasibility_rate']*100:.0f}%  "
              f"time={elapsed:.1f}s")

        # Compare with W5+EDD
        print("Running W5+full EDD for comparison...")
        r_w5 = run_pipeline(inst, n_trucks=4, variant='hybrid',
                           use_repair=True, repair_mode='full',
                           n_runs=1, seed=42)
        m_w5 = evaluate_solution_batch(r_w5['solutions'])
        print(f"  W5+full EDD:  cost={m_w5['mean_cost']:.0f}  "
              f"tard={m_w5['mean_tardiness']:.0f}  "
              f"drones={m_w5.get('avg_drone_missions', 0):.1f}  "
              f"feas={m_w5['feasibility_rate']*100:.0f}%")

        return

    # ── Build configs ──
    build_all_instances()

    if args.focus:
        inst_key = args.focus
        try:
            load_instance_from_disk(inst_key)
        except FileNotFoundError:
            print(f"ERROR: Instance not found: {inst_key}")
            return
        nc = int(inst_key.split('_')[1].replace('c', ''))
        src = inst_key.split('_')[0]
        tw_type = 'RC1' if src.startswith('RC1') else 'RC2'
        configs = [{
            'instance_key': inst_key, 'n_customers': nc,
            'tw_type': tw_type,
            'n_trucks': 2 if nc <= 25 else 4,
            'label': f'{nc}c_{tw_type}',
        }]
        print(f'FOCUS MODE: {inst_key}')
    elif args.quick:
        configs = build_configs(sizes=[25, 50])
        print('QUICK MODE: 25c + 50c')
    else:
        configs = build_configs()
        print('FULL MODE: 25c + 50c + 100c')

    print(f'Configs: {len(configs)}')
    print(f'ALNS iterations: {args.alns_iters}')
    print(f'Comparing: W5 baseline | W5 + full EDD | ALNS')
    print('=' * 60)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_results = []

    for idx, cfg in enumerate(configs):
        inst = load_instance_from_disk(cfg['instance_key'])
        print(f'\n[{idx+1}/{len(configs)}] {cfg["label"]} (inst={cfg["instance_key"]})')

        methods = {}

        # W5 baseline
        t0 = time.time()
        r_baseline = run_pomo_improved(inst, n_runs=1, n_trucks=cfg['n_trucks'],
                                       endurance='medium', seed=42,
                                       variant='hybrid_drone', tw_beta=0.4)
        methods['w5_baseline'] = evaluate_solution_batch(r_baseline['solutions'])
        methods['w5_baseline']['mean_runtime'] = time.time() - t0

        # W5 + full EDD
        t0 = time.time()
        r_full = run_pipeline(inst, n_trucks=cfg['n_trucks'], variant='hybrid',
                             use_repair=True, repair_mode='full',
                             n_runs=1, seed=100)
        methods['w5_full_edd'] = evaluate_solution_batch(r_full['solutions'])
        methods['w5_full_edd']['mean_runtime'] = time.time() - t0

        # ALNS — starts from W5 baseline solution
        t0 = time.time()
        init_sol = min(r_baseline.get('pareto_front', r_baseline['solutions']),
                      key=lambda s: s.tardiness)
        from alns_baseline import ALNSSolver
        alns_solver = ALNSSolver(inst, n_trucks=cfg['n_trucks'], seed=42)
        r_alns = alns_solver.solve(max_iter=args.alns_iters, initial_solution=init_sol)
        r_alns['solutions'] = r_alns.get('solutions', [r_alns.get('best_solution', init_sol)])
        methods['alns'] = evaluate_solution_batch(r_alns['solutions'])
        methods['alns']['mean_runtime'] = time.time() - t0

        # Print comparison
        print(f"  {'Method':<15s} {'Cost':>10s} {'Tard':>10s} {'Drones':>8s} {'Feas':>7s} {'Time':>8s}")
        print(f"  {'-'*60}")
        for name, label in [('w5_baseline', 'W5 baseline'),
                           ('w5_full_edd', 'W5 + full EDD'),
                           ('alns', 'ALNS')]:
            m = methods[name]
            print(f"  {label:<15s} {m['mean_cost']:>10.0f} {m['mean_tardiness']:>10.0f} "
                  f"{m.get('avg_drone_missions', 0):>8.1f} {m['feasibility_rate']*100:>6.0f}% "
                  f"{m.get('mean_runtime', 0):>7.1f}s")

        all_results.append({
            'label': cfg['label'],
            'instance_key': cfg['instance_key'],
            'n_customers': cfg['n_customers'],
            'tw_type': cfg['tw_type'],
            'n_trucks': cfg['n_trucks'],
            'methods': {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)}
                       for k, v in methods.items()},
        })

        # Interim save
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f'alns_interim_{timestamp}.json'), 'w') as f:
            json.dump(all_results, f, indent=2)

    # ── Summary ──
    print(f'\n{"="*80}')
    print('ALNS COMPARISON SUMMARY')
    print(f'{"="*80}')

    for twt in ['RC1', 'RC2']:
        for nc in sorted(set(e['n_customers'] for e in all_results)):
            exps = [e for e in all_results if e['tw_type'] == twt and e['n_customers'] == nc]
            if not exps:
                continue

            b_cost = sum(e['methods']['w5_baseline']['mean_cost'] for e in exps) / len(exps)
            fe_cost = sum(e['methods']['w5_full_edd']['mean_cost'] for e in exps) / len(exps)
            alns_cost = sum(e['methods']['alns']['mean_cost'] for e in exps) / len(exps)
            b_tard = sum(e['methods']['w5_baseline']['mean_tardiness'] for e in exps) / len(exps)
            fe_tard = sum(e['methods']['w5_full_edd']['mean_tardiness'] for e in exps) / len(exps)
            alns_tard = sum(e['methods']['alns']['mean_tardiness'] for e in exps) / len(exps)

            print(f'\n  {twt} {nc}c:')
            print(f'    W5 baseline:   cost={b_cost:.0f}  tard={b_tard:.0f}')
            print(f'    W5 + full EDD: cost={fe_cost:.0f}  tard={fe_tard:.0f}')
            print(f'    ALNS:          cost={alns_cost:.0f}  tard={alns_tard:.0f}')

            # ALNS vs W5+EDD
            alns_vs_w5 = (alns_cost - fe_cost) / max(fe_cost, 1) * 100
            print(f'    ALNS vs W5+EDD: cost {alns_vs_w5:+.1f}%')

    final_path = os.path.join(args.output_dir, f'week6_alns_{timestamp}.json')
    with open(final_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved: {final_path}')


if __name__ == '__main__':
    main()
