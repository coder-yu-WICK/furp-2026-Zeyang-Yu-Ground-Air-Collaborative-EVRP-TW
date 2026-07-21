#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 6 P3: Smarter Partial Repair — Experiment Runner.

Compares 3 variants:
  1. W5 baseline (no repair)
  2. W5 + full EDD repair (current P2)
  3. W5 + partial EDD repair (P3: reorder only tardy segments)

Hypothesis: Partial repair preserves more of POMO's distance optimization
than full EDD, while still eliminating 100% of tardiness.

Usage:
    python week6/run_p3_experiments.py                    # Full comparison
    python week6/run_p3_experiments.py --quick            # 25c + 50c only
    python week6/run_p3_experiments.py --test             # Single config smoke test
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


def run_one_variant(inst, cfg, variant_label, repair_mode, n_repeats, base_seed):
    """Run one variant with timing and evaluation."""
    costs, tards, drones, feas, rts = [], [], [], [], []
    extra = {}

    for rep in range(n_repeats):
        t0 = time.time()

        if variant_label == 'baseline':
            r = run_pomo_improved(inst, n_runs=1, n_trucks=cfg['n_trucks'],
                                 endurance='medium', seed=base_seed + rep,
                                 variant='hybrid_drone', tw_beta=0.4)
        else:
            r = run_pipeline(inst, n_trucks=cfg['n_trucks'], variant='hybrid',
                           use_repair=True, repair_mode=repair_mode,
                           n_runs=1, seed=base_seed + rep)

        m = evaluate_solution_batch(r['solutions'])
        costs.append(m['mean_cost'])
        tards.append(m['mean_tardiness'])
        drones.append(m.get('avg_drone_missions', 0))
        feas.append(m['feasibility_rate'])
        rts.append(time.time() - t0)

        if r.get('repair_stats'):
            rs = r['repair_stats']
            extra.setdefault('tardiness_before', []).append(rs.get('tardiness_before', 0))
            extra.setdefault('tardiness_after', []).append(rs.get('tardiness_after', 0))
            extra.setdefault('segments_repaired', []).append(rs.get('segments_repaired', 0))
            extra.setdefault('fallback_count', []).append(rs.get('fallback_count', 0))
            extra.setdefault('partial_success', []).append(rs.get('partial_success', False))

    n = len(costs)
    result = {
        'mean_cost': sum(costs) / n,
        'std_cost': (sum((c - sum(costs)/n)**2 for c in costs) / n)**0.5 if n > 1 else 0,
        'mean_tardiness': sum(tards) / n,
        'std_tardiness': (sum((t - sum(tards)/n)**2 for t in tards) / n)**0.5 if n > 1 else 0,
        'avg_drone_missions': sum(drones) / n,
        'feasibility_rate': sum(feas) / n,
        'mean_runtime': sum(rts) / n,
    }

    if extra:
        for k, v in extra.items():
            if k == 'partial_success':
                result['partial_success_rate'] = sum(1 for x in v if x) / len(v)
            else:
                result[f'avg_{k}'] = sum(v) / len(v)

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='25c + 50c only')
    parser.add_argument('--test', action='store_true', help='Single config smoke test')
    parser.add_argument('--focus', type=str, default=None,
                       help='Only specific scale (25c, 50c, 100c) or instance (e.g. RC201_100c)')
    parser.add_argument('--repeats', type=int, default=3,
                       help='Runs per config (default: 3)')
    parser.add_argument('--output-dir', type=str,
                       default=os.path.join(_W6, 'results'))
    args = parser.parse_args()

    # ── Smoke test ──
    if args.test:
        print("=== P3 SMOKE TEST ===\n")
        inst = load_instance_from_disk('RC201_100c')
        cfg = {'instance_key': 'RC201_100c', 'n_customers': 100,
               'tw_type': 'RC2', 'n_trucks': 4, 'label': '100c_RC2'}

        for label, mode in [('baseline', None), ('full EDD', 'full'), ('partial EDD (P3)', 'partial')]:
            r = run_one_variant(inst, cfg, label, mode, n_repeats=1, base_seed=42)
            print(f"  {label:20s}: cost={r['mean_cost']:.0f}  tard={r['mean_tardiness']:.0f}  "
                  f"drones={r['avg_drone_missions']:.1f}  feas={r['feasibility_rate']*100:.0f}%  "
                  f"time={r['mean_runtime']:.1f}s")
            if 'avg_segments_repaired' in r:
                print(f"    segments_repaired={r['avg_segments_repaired']:.0f}  "
                      f"fallback={r.get('avg_fallback_count', 0):.0f}  "
                      f"partial_success={r.get('partial_success_rate', 0)*100:.0f}%")
        return

    # ── Build configs ──
    if args.focus:
        if 'c' in args.focus and '_' in args.focus:
            # Specific instance
            inst_key = args.focus
            try:
                load_instance_from_disk(inst_key)
            except FileNotFoundError:
                print(f"ERROR: Instance not found: {inst_key}")
                return
            nc = int(args.focus.split('_')[1].replace('c', ''))
            src = args.focus.split('_')[0]
            tw_type = 'RC1' if src.startswith('RC1') else 'RC2'
            configs = [{
                'instance_key': inst_key, 'n_customers': nc,
                'tw_type': tw_type,
                'n_trucks': 2 if nc <= 25 else 4,
                'label': f'{nc}c_{tw_type}',
            }]
            print(f'FOCUS MODE: {inst_key}')
        else:
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
    print(f'Comparing: W5 baseline | W5 + full EDD | W5 + partial EDD (P3)')
    print('=' * 60)

    build_all_instances()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_results = []

    for idx, cfg in enumerate(configs):
        inst = load_instance_from_disk(cfg['instance_key'])
        print(f'\n[{idx+1}/{len(configs)}] {cfg["label"]} (inst={cfg["instance_key"]})')

        methods = {}

        # Variant A: W5 baseline (no repair)
        methods['w5_baseline'] = run_one_variant(
            inst, cfg, 'baseline', None, n_repeats, base_seed=42)

        # Variant B: W5 + full EDD repair
        methods['full_edd'] = run_one_variant(
            inst, cfg, 'full_edd', 'full', n_repeats, base_seed=100)

        # Variant C: W5 + partial EDD repair (P3)
        methods['partial_edd'] = run_one_variant(
            inst, cfg, 'partial_edd', 'partial', n_repeats, base_seed=200)

        # Print comparison table
        b = methods['w5_baseline']
        fe = methods['full_edd']
        pe = methods['partial_edd']

        print(f"  {'Variant':<20s} {'Cost':>10s} {'Tard':>10s} {'Drones':>8s} {'Feas':>7s} {'Time':>8s}")
        print(f"  {'-'*60}")
        print(f"  {'W5 baseline':<20s} {b['mean_cost']:>10.0f} {b['mean_tardiness']:>10.0f} "
              f"{b['avg_drone_missions']:>8.1f} {b['feasibility_rate']*100:>6.0f}% {b['mean_runtime']:>7.1f}s")
        print(f"  {'W5 + full EDD':<20s} {fe['mean_cost']:>10.0f} {fe['mean_tardiness']:>10.0f} "
              f"{fe['avg_drone_missions']:>8.1f} {fe['feasibility_rate']*100:>6.0f}% {fe['mean_runtime']:>7.1f}s")
        print(f"  {'W5 + partial EDD':<20s} {pe['mean_cost']:>10.0f} {pe['mean_tardiness']:>10.0f} "
              f"{pe['avg_drone_missions']:>8.1f} {pe['feasibility_rate']*100:>6.0f}% {pe['mean_runtime']:>7.1f}s")

        # Comparison: partial vs full
        cost_delta = (pe['mean_cost'] - fe['mean_cost']) / max(fe['mean_cost'], 1) * 100
        tard_delta = pe['mean_tardiness'] - fe['mean_tardiness']
        drone_delta = pe['avg_drone_missions'] - fe['avg_drone_missions']
        print(f"  Partial vs Full:   cost {cost_delta:+.1f}%  tard {tard_delta:+.0f}  "
              f"drones {drone_delta:+.1f}")

        if 'avg_segments_repaired' in pe:
            print(f"  P3 details: segments={pe.get('avg_segments_repaired', 0):.1f}  "
                  f"fallbacks={pe.get('avg_fallback_count', 0):.1f}  "
                  f"success_rate={pe.get('partial_success_rate', 0)*100:.0f}%")

        all_results.append({
            'label': cfg['label'],
            'instance_key': cfg['instance_key'],
            'n_customers': cfg['n_customers'],
            'tw_type': cfg['tw_type'],
            'n_trucks': cfg['n_trucks'],
            'methods': methods,
        })

        # Interim save
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f'p3_interim_{timestamp}.json'), 'w') as f:
            json.dump(all_results, f, indent=2)

    # ── Final save ──
    final_path = os.path.join(args.output_dir, f'week6_p3_{timestamp}.json')
    with open(final_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # ── Summary ──
    print(f'\n{"="*80}')
    print('P3 SUMMARY: Baseline vs Full EDD vs Partial EDD')
    print(f'{"="*80}')

    for twt in ['RC1', 'RC2']:
        for nc in sorted(set(e['n_customers'] for e in all_results)):
            exps = [e for e in all_results if e['tw_type'] == twt and e['n_customers'] == nc]
            if not exps:
                continue

            b_cost = sum(e['methods']['w5_baseline']['mean_cost'] for e in exps) / len(exps)
            b_tard = sum(e['methods']['w5_baseline']['mean_tardiness'] for e in exps) / len(exps)

            fe_cost = sum(e['methods']['full_edd']['mean_cost'] for e in exps) / len(exps)
            fe_tard = sum(e['methods']['full_edd']['mean_tardiness'] for e in exps) / len(exps)
            fe_drones = sum(e['methods']['full_edd']['avg_drone_missions'] for e in exps) / len(exps)

            pe_cost = sum(e['methods']['partial_edd']['mean_cost'] for e in exps) / len(exps)
            pe_tard = sum(e['methods']['partial_edd']['mean_tardiness'] for e in exps) / len(exps)
            pe_drones = sum(e['methods']['partial_edd']['avg_drone_missions'] for e in exps) / len(exps)

            # Improvements
            fe_vs_b = (fe_cost - b_cost) / max(b_cost, 1) * 100
            pe_vs_b = (pe_cost - b_cost) / max(b_cost, 1) * 100
            pe_vs_fe = (pe_cost - fe_cost) / max(fe_cost, 1) * 100

            print(f'\n  {twt} {nc}c:')
            print(f'    Baseline:        cost={b_cost:.0f}  tard={b_tard:.0f}')
            print(f'    Full EDD:        cost={fe_cost:.0f}  tard={fe_tard:.0f}  '
                  f'drones={fe_drones:.1f}  (cost {fe_vs_b:+.1f}% vs baseline)')
            print(f'    Partial EDD:     cost={pe_cost:.0f}  tard={pe_tard:.0f}  '
                  f'drones={pe_drones:.1f}  (cost {pe_vs_b:+.1f}% vs baseline, {pe_vs_fe:+.1f}% vs full)')

            if pe_tard == 0 and fe_tard == 0:
                if pe_cost < fe_cost:
                    print(f'    ✓ Partial EDD WINS: same tardiness (0), {abs(pe_vs_fe):.1f}% lower cost')
                elif pe_cost > fe_cost:
                    print(f'    ✗ Full EDD wins: same tardiness (0), {abs(pe_vs_fe):.1f}% lower cost')
                else:
                    print(f'    = Tie: identical cost and tardiness')

    # ── Overall stats ──
    all_b_cost = sum(e['methods']['w5_baseline']['mean_cost'] for e in all_results)
    all_fe_cost = sum(e['methods']['full_edd']['mean_cost'] for e in all_results)
    all_pe_cost = sum(e['methods']['partial_edd']['mean_cost'] for e in all_results)
    all_b_tard = sum(e['methods']['w5_baseline']['mean_tardiness'] for e in all_results)
    all_fe_tard = sum(e['methods']['full_edd']['mean_tardiness'] for e in all_results)
    all_pe_tard = sum(e['methods']['partial_edd']['mean_tardiness'] for e in all_results)

    print(f'\n{"="*80}')
    print(f'  OVERALL:')
    print(f'    Baseline:    cost={all_b_cost:.0f}  tard={all_b_tard:.0f}')
    print(f'    Full EDD:    cost={all_fe_cost:.0f}  tard={all_fe_tard:.0f}  '
          f'({(all_fe_cost-all_b_cost)/max(all_b_cost,1)*100:+.1f}% cost vs baseline)')
    print(f'    Partial EDD: cost={all_pe_cost:.0f}  tard={all_pe_tard:.0f}  '
          f'({(all_pe_cost-all_b_cost)/max(all_b_cost,1)*100:+.1f}% cost vs baseline, '
          f'{(all_pe_cost-all_fe_cost)/max(all_fe_cost,1)*100:+.1f}% vs full)')

    # Count wins
    n_partial_wins = 0
    n_full_wins = 0
    n_ties = 0
    for e in all_results:
        pe = e['methods']['partial_edd']
        fe = e['methods']['full_edd']
        if pe['mean_tardiness'] == 0 and fe['mean_tardiness'] == 0:
            if pe['mean_cost'] < fe['mean_cost'] * 0.99:
                n_partial_wins += 1
            elif fe['mean_cost'] < pe['mean_cost'] * 0.99:
                n_full_wins += 1
            else:
                n_ties += 1
        elif pe['mean_tardiness'] < fe['mean_tardiness']:
            n_partial_wins += 1
        elif fe['mean_tardiness'] < pe['mean_tardiness']:
            n_full_wins += 1
        else:
            n_ties += 1

    print(f'\n  Head-to-head (same tardiness → cost comparison):')
    print(f'    Partial EDD wins: {n_partial_wins}/{len(all_results)}')
    print(f'    Full EDD wins:    {n_full_wins}/{len(all_results)}')
    print(f'    Ties:             {n_ties}/{len(all_results)}')

    print(f'\nResults saved: {final_path}')


if __name__ == '__main__':
    main()
