# -*- coding: utf-8 -*-
"""
Week 7: Synchronization Study — Compare No-Sync vs Full-Sync.

FURP Model D: charging + launch-recovery synchronization.

Usage:
    python week7/run_sync_study.py              # Full comparison
    python week7/run_sync_study.py --test       # Smoke test
    python week7/run_sync_study.py --quick      # 25c only
"""

import json, os, sys, time
from datetime import datetime

_W6 = os.path.dirname(os.path.abspath(__file__))
_W5 = os.path.join(_W6, '..', 'week5')
_W4 = os.path.join(_W6, '..', 'week4')
_W3 = os.path.join(_W6, '..', 'week3')

for _p in [_W6, _W5, _W4, _W3]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import RC1_INSTANCES, RC2_INSTANCES, RESULTS_DIR
from utils.data_loader import load_instance_from_disk, build_all_instances
from utils.problem_model import TruckDroneSolution, evaluate_solution_batch
from pipeline import run_pipeline
from sync_constraints import (
    insert_cross_route_drones_sync, evaluate_with_sync,
    check_drone_sync, compute_route_timeline,
)


def run_sync_comparison(instance_key, n_trucks, n_repeats=3, base_seed=42):
    """Compare No-Sync vs Full-Sync on one instance."""
    inst = load_instance_from_disk(instance_key)

    results = {'no_sync': [], 'full_sync': []}

    for rep in range(n_repeats):
        seed = base_seed + rep

        # ── Get pipeline solution (use repair to ensure TW feasible routes) ──
        r = run_pipeline(inst, n_trucks=n_trucks, variant='hybrid',
                        use_repair=True, repair_mode='full',
                        n_runs=1, seed=seed)
        if not r['solutions']:
            continue
        sol = r['solutions'][0]

        # ── No Sync: Original drone insertion ──
        from drone_post_processing import insert_cross_route_drones
        routes_ns, missions_ns, saved_ns, n_drones_ns = insert_cross_route_drones(
            sol.truck_routes, inst, drone_endurance=4.0)
        sol_ns = TruckDroneSolution(routes_ns, missions_ns, inst)
        sync_eval_ns = evaluate_with_sync(sol_ns, inst)

        results['no_sync'].append({
            'cost': sol_ns.cost,
            'tardiness': sol_ns.tardiness,
            'feasible': sol_ns.feasible,
            'n_drones': n_drones_ns,
            'sync_violations': sync_eval_ns['total_sync_violation'],
            'n_sync_violations': sync_eval_ns['n_sync_violations'],
            'is_sync_feasible': sync_eval_ns['is_sync_feasible'],
        })

        # ── Full Sync: Sync-aware drone insertion ──
        routes_s, missions_s, saved_s, n_drones_s, sync_stats = \
            insert_cross_route_drones_sync(
                sol.truck_routes, inst, drone_endurance=4.0, require_sync=True)
        sol_s = TruckDroneSolution(routes_s, missions_s, inst)
        sync_eval_s = evaluate_with_sync(sol_s, inst)

        results['full_sync'].append({
            'cost': sol_s.cost,
            'tardiness': sol_s.tardiness,
            'feasible': sol_s.feasible,
            'n_drones': n_drones_s,
            'sync_violations': sync_eval_s['total_sync_violation'],
            'n_sync_violations': sync_eval_s['n_sync_violations'],
            'is_sync_feasible': sync_eval_s['is_sync_feasible'],
            'rejected_by_sync': sync_stats.get('rejected_by_sync', 0),
            'total_checked': sync_stats.get('checked', 0),
        })

    # Aggregate
    agg = {}
    for model, entries in results.items():
        if not entries:
            continue
        n = len(entries)
        agg[model] = {
            'mean_cost': sum(e['cost'] for e in entries) / n,
            'mean_tardiness': sum(e['tardiness'] for e in entries) / n,
            'feasibility': sum(1 for e in entries if e['feasible']) / n,
            'mean_n_drones': sum(e['n_drones'] for e in entries) / n,
            'mean_sync_violations': sum(e['sync_violations'] for e in entries) / n,
            'sync_feasibility': sum(1 for e in entries if e['is_sync_feasible']) / n,
            'n_runs': n,
        }
        if 'rejected_by_sync' in entries[0]:
            agg[model]['mean_rejected'] = sum(e.get('rejected_by_sync', 0) for e in entries) / n

    return agg


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Smoke test on one instance')
    parser.add_argument('--quick', action='store_true', help='25c only')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()

    build_all_instances()

    if args.test:
        print("=== SYNC STUDY SMOKE TEST ===\n")
        agg = run_sync_comparison('RC201_50c', n_trucks=4, n_repeats=args.repeats)
        for model, m in agg.items():
            print(f"  {model}: cost={m['mean_cost']:.0f} tard={m['mean_tardiness']:.0f} "
                  f"feas={m['feasibility']*100:.0f}% drones={m['mean_n_drones']:.1f} "
                  f"sync_viol={m['mean_sync_violations']:.1f} sync_feas={m['sync_feasibility']*100:.0f}%")
        return

    sizes = [25, 50] if args.quick else [25, 50]
    configs = []
    for src in RC1_INSTANCES + RC2_INSTANCES:
        for nc in sizes:
            key = f'{src}_{nc}c'
            try:
                load_instance_from_disk(key)
            except FileNotFoundError:
                continue
            configs.append({
                'instance_key': key,
                'n_trucks': 2 if nc <= 25 else 4,
                'label': f'{nc}c_{"RC1" if src.startswith("RC1") else "RC2"}',
            })

    print(f"SYNC STUDY: {len(configs)} instances, {args.repeats} repeats each")
    print(f"Comparing: No Sync (original) vs Full Sync (sync-aware)")
    print('=' * 80)

    all_results = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for idx, cfg in enumerate(configs):
        print(f"\n[{idx+1}/{len(configs)}] {cfg['label']} ({cfg['instance_key']})")
        agg = run_sync_comparison(cfg['instance_key'], cfg['n_trucks'],
                                   n_repeats=args.repeats)

        print(f"  {'Model':<15s} {'Cost':>10s} {'Tard':>10s} {'Feas':>7s} "
              f"{'Drones':>8s} {'SyncViol':>10s} {'SyncFeas':>9s}")
        print(f"  {'-'*72}")
        for model, label in [('no_sync', 'No Sync'), ('full_sync', 'Full Sync')]:
            if model in agg:
                m = agg[model]
                print(f"  {label:<15s} {m['mean_cost']:>10.0f} {m['mean_tardiness']:>10.0f} "
                      f"{m['feasibility']*100:>6.0f}% {m['mean_n_drones']:>7.1f} "
                      f"{m['mean_sync_violations']:>9.1f} {m['sync_feasibility']*100:>8.0f}%")

        all_results.append({'config': cfg, 'results': agg})

        os.makedirs(RESULTS_DIR, exist_ok=True)
        interim_path = os.path.join(RESULTS_DIR, f'sync_study_{timestamp}.json')
        with open(interim_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    # Summary
    print(f'\n{"="*80}')
    print('SYNC STUDY SUMMARY')
    for nc in sorted(set(c['config']['n_trucks'] for c in all_results)):
        size_exps = [e for e in all_results if e['config']['n_trucks'] == nc]
        n_inst = len(size_exps)
        print(f'\n  {nc} trucks ({n_inst} instances):')
        for model, label in [('no_sync', 'No Sync'), ('full_sync', 'Full Sync')]:
            if model not in size_exps[0]['results']:
                continue
            avg_cost = sum(e['results'][model]['mean_cost'] for e in size_exps) / n_inst
            avg_drones = sum(e['results'][model]['mean_n_drones'] for e in size_exps) / n_inst
            avg_sync_v = sum(e['results'][model]['mean_sync_violations'] for e in size_exps) / n_inst
            avg_sync_f = sum(e['results'][model]['sync_feasibility'] for e in size_exps) / n_inst
            print(f'    {label}: cost={avg_cost:.0f}  drones={avg_drones:.1f}  '
                  f'sync_viol={avg_sync_v:.1f}  sync_feas={avg_sync_f*100:.0f}%')

    print(f'\nResults saved to: {interim_path}')


if __name__ == '__main__':
    main()
