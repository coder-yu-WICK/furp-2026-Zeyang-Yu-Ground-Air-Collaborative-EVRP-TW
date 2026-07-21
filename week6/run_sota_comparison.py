#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 7: SOTA Comparison Experiment Runner.

Compares 5 methods on the Truck-Drone EVRP-TW benchmark:
  1. w5_baseline   — POMO + hybrid_drone clustering (no repair)
  2. w5_edd        — POMO + hybrid_drone + full EDD repair (our best)
  3. nsga2         — NSGA-II classical metaheuristic
  4. paco          — P-ACO classical metaheuristic
  5. ivnd          — IVND classical metaheuristic

Usage:
    python week7/run_sota_comparison.py             # Full comparison (25c + 50c)
    python week7/run_sota_comparison.py --quick     # 25c only
    python week7/run_sota_comparison.py --test      # Single instance smoke test
"""

import json
import os
import sys
import time
import importlib.util
from datetime import datetime

# ── Path setup ──────────────────────────────────────────────────────────
_W6 = os.path.dirname(os.path.abspath(__file__))
_W5 = os.path.join(_W6, '..', 'week5')
_W4 = os.path.join(_W6, '..', 'week4')
_W3 = os.path.join(_W6, '..', 'week3')

sys.path.insert(0, _W5)
sys.path.insert(1, _W4)
sys.path.insert(0, _W6)

from config import RESULTS_DIR, RC1_INSTANCES, RC2_INSTANCES, CUSTOMER_SIZES
from utils.data_loader import load_instance_from_disk, build_all_instances
from utils.problem_model import evaluate_solution_batch, TruckDroneSolution
from pipeline import run_pipeline
from pomo_mt_improved import run_pomo_improved

# ── Week 3 algorithm imports (importlib for cross-package compatibility) ─
# Week 3 algorithms internally do `from config import ...` and
# `from utils.problem_model import ...`, so week3 must be on sys.path
# during their loading. We insert it temporarily then remove it so that
# subsequent imports from config/utils continue to resolve to week6/week4.
_W3_was_in_path = _W3 in sys.path
if not _W3_was_in_path:
    sys.path.insert(0, _W3)

try:
    _spec_nsga2 = importlib.util.spec_from_file_location(
        "nsga2", os.path.join(_W3, "algorithms", "nsga2.py"))
    _nsga2_mod = importlib.util.module_from_spec(_spec_nsga2)
    _spec_nsga2.loader.exec_module(_nsga2_mod)
    run_nsga2 = _nsga2_mod.run_nsga2

    _spec_paco = importlib.util.spec_from_file_location(
        "paco", os.path.join(_W3, "algorithms", "paco.py"))
    _paco_mod = importlib.util.module_from_spec(_spec_paco)
    _spec_paco.loader.exec_module(_paco_mod)
    run_paco = _paco_mod.run_paco

    _spec_ivnd = importlib.util.spec_from_file_location(
        "ivnd", os.path.join(_W3, "algorithms", "ivnd.py"))
    _ivnd_mod = importlib.util.module_from_spec(_spec_ivnd)
    _spec_ivnd.loader.exec_module(_ivnd_mod)
    run_ivnd = _ivnd_mod.run_ivnd
finally:
    if not _W3_was_in_path:
        sys.path.remove(_W3)


# ── Config builder ──────────────────────────────────────────────────────

def build_configs(sizes=None):
    """Build experiment configs from RC1 + RC2 source instances.

    Args:
        sizes: list of customer counts (default: [25, 50])

    Returns:
        list of dicts with instance_key, n_customers, tw_type, n_trucks, label
    """
    if sizes is None:
        sizes = [25, 50]

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


# ── Single-method runner ────────────────────────────────────────────────

def run_one_method(inst, cfg, method_name, n_repeats, base_seed):
    """Run one method and return aggregated evaluation metrics.

    Args:
        inst: problem instance dict
        cfg: experiment config dict (n_trucks, etc.)
        method_name: one of 'w5_baseline', 'w5_edd', 'nsga2', 'paco', 'ivnd'
        n_repeats: number of independent runs
        base_seed: base random seed

    Returns:
        dict with mean_cost, std_cost, mean_tardiness, std_tardiness,
             feasibility_rate, mean_runtime, std_runtime
    """
    costs, tards, feas, rts = [], [], [], []

    for rep in range(n_repeats):
        t0 = time.time()
        run_seed = base_seed + rep

        try:
            if method_name == 'w5_baseline':
                r = run_pomo_improved(
                    inst, n_runs=1, n_trucks=cfg['n_trucks'],
                    endurance='medium', seed=run_seed,
                    variant='hybrid_drone', tw_beta=0.4,
                )
            elif method_name == 'w5_edd':
                r = run_pipeline(
                    inst, n_trucks=cfg['n_trucks'], variant='hybrid',
                    use_repair=True, repair_mode='full',
                    n_runs=1, seed=run_seed,
                )
            elif method_name == 'nsga2':
                r = run_nsga2(
                    inst, n_trucks=cfg['n_trucks'], n_drones=2,
                    endurance=4.0, n_runs=1, seed=run_seed,
                )
            elif method_name == 'paco':
                r = run_paco(
                    inst, n_runs=1, endurance=4.0, seed=run_seed,
                )
            elif method_name == 'ivnd':
                r = run_ivnd(
                    inst, n_trucks=cfg['n_trucks'], n_drones=2,
                    endurance=4.0, n_runs=1, seed=run_seed,
                )
            else:
                raise ValueError(f"Unknown method: {method_name}")

        except Exception as e:
            print(f"      ERROR [{method_name}] rep {rep+1}: {e}")
            import traceback
            traceback.print_exc()
            # Record a failure placeholder
            costs.append(1e9)
            tards.append(1e9)
            feas.append(0.0)
            rts.append(time.time() - t0)
            continue

        # Evaluate solutions
        if r.get('solutions'):
            m = evaluate_solution_batch(r['solutions'])
            costs.append(m['mean_cost'])
            tards.append(m['mean_tardiness'])
            feas.append(m['feasibility_rate'])
        else:
            # No solutions produced
            costs.append(1e9)
            tards.append(1e9)
            feas.append(0.0)

        rts.append(time.time() - t0)

    n = len(costs)
    if n == 0:
        return {
            'mean_cost': 1e9, 'std_cost': 0,
            'mean_tardiness': 1e9, 'std_tardiness': 0,
            'feasibility_rate': 0.0,
            'mean_runtime': 0, 'std_runtime': 0,
        }

    mean_cost = sum(costs) / n
    mean_tard = sum(tards) / n
    mean_feas = sum(feas) / n
    mean_rt = sum(rts) / n

    return {
        'mean_cost': mean_cost,
        'std_cost': (sum((c - mean_cost) ** 2 for c in costs) / n) ** 0.5
                    if n > 1 else 0,
        'mean_tardiness': mean_tard,
        'std_tardiness': (sum((t - mean_tard) ** 2 for t in tards) / n) ** 0.5
                         if n > 1 else 0,
        'feasibility_rate': mean_feas,
        'mean_runtime': mean_rt,
        'std_runtime': (sum((t - mean_rt) ** 2 for t in rts) / n) ** 0.5
                       if n > 1 else 0,
    }


# ── Table printing helpers ──────────────────────────────────────────────

_METHOD_ORDER = ['w5_baseline', 'w5_edd', 'nsga2', 'paco', 'ivnd']
_METHOD_LABELS = {
    'w5_baseline': 'W5 Baseline',
    'w5_edd': 'W5 + EDD (Ours)',
    'nsga2': 'NSGA-II',
    'paco': 'P-ACO',
    'ivnd': 'IVND',
}


def print_comparison_table(methods, cfg):
    """Print a formatted comparison table for a single config."""
    label = cfg['label']
    print(f"\n{'=' * 90}")
    print(f"  Results: {label}  (inst={cfg['instance_key']}, "
          f"n_trucks={cfg['n_trucks']})")
    print(f"{'=' * 90}")

    header = (f"  {'Method':<22s} {'Cost':>10s} {'Tard':>10s} "
              f"{'Feas':>8s} {'Runtime':>10s}")
    sep = "  " + "-" * 62
    print(header)
    print(sep)

    for mname in _METHOD_ORDER:
        m = methods.get(mname)
        if m is None:
            continue
        lbl = _METHOD_LABELS.get(mname, mname)
        print(f"  {lbl:<22s} {m['mean_cost']:>10.1f} {m['mean_tardiness']:>10.1f} "
              f"{m['feasibility_rate']*100:>7.1f}% {m['mean_runtime']:>9.1f}s")


# ── Summary printer ─────────────────────────────────────────────────────

def print_summary(all_results):
    """Print summary by TW type and size."""
    print(f"\n{'=' * 90}")
    print("  SOTA COMPARISON SUMMARY")
    print(f"{'=' * 90}")

    for twt in ['RC1', 'RC2']:
        for nc in sorted(set(e['n_customers'] for e in all_results)):
            exps = [e for e in all_results
                    if e['tw_type'] == twt and e['n_customers'] == nc]
            if not exps:
                continue

            print(f"\n  {twt} {nc}c ({len(exps)} instances):")
            print(f"  {'Method':<22s} {'Avg Cost':>10s} {'Avg Tard':>10s} "
                  f"{'Avg Feas':>9s} {'Avg Time':>9s}")
            print(f"  {'-' * 62}")

            for mname in _METHOD_ORDER:
                if mname not in exps[0]['methods']:
                    continue
                avg_cost = sum(
                    e['methods'][mname]['mean_cost'] for e in exps) / len(exps)
                avg_tard = sum(
                    e['methods'][mname]['mean_tardiness'] for e in exps) / len(exps)
                avg_feas = sum(
                    e['methods'][mname]['feasibility_rate'] for e in exps) / len(exps)
                avg_rt = sum(
                    e['methods'][mname]['mean_runtime'] for e in exps) / len(exps)

                lbl = _METHOD_LABELS.get(mname, mname)
                print(f"  {lbl:<22s} {avg_cost:>10.1f} {avg_tard:>10.1f} "
                      f"{avg_feas*100:>8.1f}% {avg_rt:>9.1f}s")


def print_rankings(all_results):
    """Print method rankings based on Pareto (cost + tardiness)."""
    print(f"\n{'=' * 90}")
    print("  METHOD RANKINGS (lower cost = better, tardiness < 100 preferred)")
    print(f"{'=' * 90}")

    # Aggregate scores across all instances
    scores = {mname: {'total_cost': 0, 'total_tard': 0, 'count': 0}
              for mname in _METHOD_ORDER}

    for e in all_results:
        for mname in _METHOD_ORDER:
            if mname in e['methods']:
                m = e['methods'][mname]
                scores[mname]['total_cost'] += m['mean_cost']
                scores[mname]['total_tard'] += m['mean_tardiness']
                scores[mname]['count'] += 1

    # Print overall averages
    print(f"\n  Overall (averaged across {len(all_results)} configs):")
    print(f"  {'Rank':<6s} {'Method':<22s} {'Avg Cost':>10s} {'Avg Tard':>10s}")
    print(f"  {'-' * 50}")

    # Sort by cost (lower is better), but penalize high tardiness
    ranked = []
    for mname in _METHOD_ORDER:
        s = scores[mname]
        if s['count'] == 0:
            continue
        avg_c = s['total_cost'] / s['count']
        avg_t = s['total_tard'] / s['count']
        # Composite score: cost + tardiness penalty
        composite = avg_c + avg_t * 10
        ranked.append((mname, avg_c, avg_t, composite))

    ranked.sort(key=lambda x: x[3])

    for rank, (mname, avg_c, avg_t, composite) in enumerate(ranked, 1):
        lbl = _METHOD_LABELS.get(mname, mname)
        print(f"  {rank:<6d} {lbl:<22s} {avg_c:>10.1f} {avg_t:>10.1f}")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="SOTA Comparison Experiment Runner"
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='25c only (fast mode)')
    parser.add_argument(
        '--test', action='store_true',
        help='Single instance smoke test')
    parser.add_argument(
        '--repeats', type=int, default=3,
        help='Independent runs per method per config (default: 3)')
    parser.add_argument(
        '--output-dir', type=str,
        default=os.path.join(_W6, '..', 'week6', 'results'))
    args = parser.parse_args()

    # ── Clear __pycache__ in week3 to avoid stale bytecode ──
    _pycache = os.path.join(_W3, 'algorithms', '__pycache__')
    if os.path.exists(_pycache):
        import shutil
        shutil.rmtree(_pycache)
        print(f"[INFO] Cleared {_pycache}")

    # ── Smoke test ──
    if args.test:
        print("=== SOTA SMOKE TEST ===\n")
        inst = load_instance_from_disk('RC201_50c')
        cfg = {
            'instance_key': 'RC201_50c', 'n_customers': 50,
            'tw_type': 'RC2', 'n_trucks': 4, 'label': '50c_RC2',
        }

        methods = {}
        for mname in _METHOD_ORDER:
            lbl = _METHOD_LABELS.get(mname, mname)
            print(f"  Running {lbl}...", end=' ', flush=True)
            methods[mname] = run_one_method(
                inst, cfg, mname, n_repeats=1, base_seed=42)
            m = methods[mname]
            print(f"cost={m['mean_cost']:.1f}  tard={m['mean_tardiness']:.1f}  "
                  f"feas={m['feasibility_rate']*100:.0f}%  "
                  f"time={m['mean_runtime']:.1f}s")

        print_comparison_table(methods, cfg)
        return

    # ── Build configs ──
    if args.quick:
        configs = build_configs(sizes=[25])
        print('QUICK MODE: 25c only')
    else:
        configs = build_configs(sizes=[25, 50])
        print('FULL MODE: 25c + 50c')

    n_repeats = args.repeats
    print(f'Configs: {len(configs)}, Repeats per method: {n_repeats}')
    print(f'Methods: {", ".join(_METHOD_LABELS[m] for m in _METHOD_ORDER)}')
    print(f'{"=" * 90}')

    # Pre-build all instances
    build_all_instances()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_results = []

    for idx, cfg in enumerate(configs):
        inst = load_instance_from_disk(cfg['instance_key'])
        print(f'\n[{idx+1}/{len(configs)}] {cfg["label"]} '
              f'(inst={cfg["instance_key"]}, n_trucks={cfg["n_trucks"]})')

        methods = {}
        for mname in _METHOD_ORDER:
            lbl = _METHOD_LABELS.get(mname, mname)
            print(f'  [{lbl}] ', end='', flush=True)
            t_start = time.time()
            methods[mname] = run_one_method(
                inst, cfg, mname, n_repeats=n_repeats, base_seed=42)
            elapsed = time.time() - t_start
            m = methods[mname]
            print(f'cost={m["mean_cost"]:.1f}  tard={m["mean_tardiness"]:.1f}  '
                  f'feas={m["feasibility_rate"]*100:.0f}%  '
                  f'total={elapsed:.1f}s')

        print_comparison_table(methods, cfg)

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
        interim_path = os.path.join(
            args.output_dir, f'sota_interim_{timestamp}.json')
        with open(interim_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    # ── Final save ──
    final_path = os.path.join(
        args.output_dir, f'week7_sota_{timestamp}.json')
    with open(final_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # ── Summary ──
    print_summary(all_results)
    print_rankings(all_results)

    print(f'\nResults saved: {final_path}')
    print('Done.')


if __name__ == '__main__':
    main()
