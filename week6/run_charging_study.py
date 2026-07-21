# -*- coding: utf-8 -*-
"""
Week 7: Charging Study — Compare Models A, B, C.

FURP Ablation:
  Model A: Baseline (no charging constraints)
  Model B: Linear charging
  Model C: Non-linear charging

Usage:
    python week7/run_charging_study.py              # Full comparison
    python week7/run_charging_study.py --test       # Smoke test
    python week7/run_charging_study.py --quick      # 25c only
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

from config import RC1_INSTANCES, RC2_INSTANCES, CUSTOMER_SIZES, RESULTS_DIR
from utils.data_loader import load_instance_from_disk, build_all_instances
from utils.problem_model import TruckDroneSolution, evaluate_solution_batch
from ev_problem_model import (
    EVTruckDroneSolution, compare_charging_models,
    insert_charging_stops, is_charging_station,
    BATTERY_CAPACITY, ENERGY_PER_KM,
)
from pipeline import run_pipeline


def run_charging_comparison(instance_key, n_trucks, n_repeats=3, base_seed=42):
    """Compare Models A, B, C on one instance."""
    inst = load_instance_from_disk(instance_key)
    n_customers = inst['n_customers']

    results = {'A_baseline': [], 'B_linear': [], 'C_nonlinear': []}

    for rep in range(n_repeats):
        seed = base_seed + rep

        # ── Get W5 solution (same pipeline for all models) ──
        r = run_pipeline(inst, n_trucks=n_trucks, variant='hybrid',
                        use_repair=True, repair_mode='full',
                        n_runs=1, seed=seed)
        if not r['solutions']:
            continue
        sol = r['solutions'][0]

        # ── Model A: Baseline (no charging) ──
        sol_a = TruckDroneSolution(sol.truck_routes, sol.drone_missions, inst)
        results['A_baseline'].append({
            'cost': sol_a.cost, 'tardiness': sol_a.tardiness,
            'feasible': sol_a.feasible,
        })

        # Strip any charging station nodes from routes
        clean_routes = [[n for n in r if not is_charging_station(n, n_customers)]
                       for r in sol.truck_routes]

        # ── Model B: Linear Charging ──
        routes_b, stats_b = insert_charging_stops(
            clean_routes, inst, charging_model='linear')
        sol_b = EVTruckDroneSolution(routes_b, sol.drone_missions, inst,
                                      charging_model='linear')
        results['B_linear'].append({
            'cost': sol_b.cost, 'tardiness': sol_b.tardiness,
            'feasible': sol_b.feasible,
            'battery_violations': sol_b.battery_violations,
            'charging_time': sol_b.charging_time,
            'n_charges': sol_b.n_charges,
            'energy_consumed': sol_b.energy_consumed,
            'charging_stops': sum(len(s) for s in stats_b),
        })

        # ── Model C: Non-linear Charging ──
        routes_c, stats_c = insert_charging_stops(
            clean_routes, inst, charging_model='nonlinear')
        sol_c = EVTruckDroneSolution(routes_c, sol.drone_missions, inst,
                                      charging_model='nonlinear')
        results['C_nonlinear'].append({
            'cost': sol_c.cost, 'tardiness': sol_c.tardiness,
            'feasible': sol_c.feasible,
            'battery_violations': sol_c.battery_violations,
            'charging_time': sol_c.charging_time,
            'n_charges': sol_c.n_charges,
            'energy_consumed': sol_c.energy_consumed,
            'charging_stops': sum(len(s) for s in stats_c),
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
            'mean_charging_time': sum(e.get('charging_time', 0) for e in entries) / n,
            'mean_n_charges': sum(e.get('n_charges', 0) for e in entries) / n,
            'mean_energy': sum(e.get('energy_consumed', 0) for e in entries) / n,
            'mean_battery_violations': sum(e.get('battery_violations', 0) for e in entries) / n,
            'n_runs': n,
        }
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
        print("=== CHARGING STUDY SMOKE TEST ===\n")
        agg = run_charging_comparison('RC201_50c', n_trucks=4, n_repeats=args.repeats)
        for model, m in agg.items():
            print(f"  {model}: cost={m['mean_cost']:.0f} tard={m['mean_tardiness']:.0f} "
                  f"feas={m['feasibility']*100:.0f}% charge_t={m['mean_charging_time']:.1f} "
                  f"n_charges={m['mean_n_charges']:.1f} battery_v={m['mean_battery_violations']:.1f} "
                  f"energy={m['mean_energy']:.1f}")
        return

    # Build configs
    sizes = [25, 50] if args.quick else [25, 50, 100]
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

    print(f"CHARGING STUDY: {len(configs)} instances, {args.repeats} repeats each")
    print(f"Comparing: Model A (baseline) | Model B (linear) | Model C (non-linear)")
    print(f"Battery: {BATTERY_CAPACITY} kWh, Energy: {ENERGY_PER_KM} kWh/km")
    print('=' * 80)

    all_results = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for idx, cfg in enumerate(configs):
        print(f"\n[{idx+1}/{len(configs)}] {cfg['label']} ({cfg['instance_key']})")
        agg = run_charging_comparison(cfg['instance_key'], cfg['n_trucks'],
                                       n_repeats=args.repeats)

        print(f"  {'Model':<20s} {'Cost':>10s} {'Tard':>10s} {'Feas':>7s} "
              f"{'ChargeT':>9s} {'#Chg':>6s} {'BattV':>8s}")
        print(f"  {'-'*75}")
        for model, label in [('A_baseline', 'A: Baseline'),
                            ('B_linear', 'B: Linear Chg'),
                            ('C_nonlinear', 'C: Non-linear Chg')]:
            if model in agg:
                m = agg[model]
                print(f"  {label:<20s} {m['mean_cost']:>10.0f} {m['mean_tardiness']:>10.0f} "
                      f"{m['feasibility']*100:>6.0f}% "
                      f"{m['mean_charging_time']:>8.1f}m {m['mean_n_charges']:>5.1f} "
                      f"{m['mean_battery_violations']:>7.1f}")

        all_results.append({'config': cfg, 'results': agg})

        # Interim save
        os.makedirs(RESULTS_DIR, exist_ok=True)
        interim_path = os.path.join(RESULTS_DIR, f'charging_study_{timestamp}.json')
        with open(interim_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    # Summary
    print(f'\n{"="*80}')
    print('CHARGING STUDY SUMMARY')
    for nc in sorted(set(c['config']['n_trucks'] for c in all_results), reverse=True):
        size_exps = [e for e in all_results if e['config']['n_trucks'] == nc]
        if not size_exps:
            continue
        n_inst = len(size_exps)
        print(f'\n  {nc} trucks ({n_inst} instances):')
        for model, label in [('A_baseline', 'A'), ('B_linear', 'B'), ('C_nonlinear', 'C')]:
            avg_cost = sum(e['results'][model]['mean_cost'] for e in size_exps if model in e['results']) / max(n_inst, 1)
            avg_feas = sum(e['results'][model]['feasibility'] for e in size_exps if model in e['results']) / max(n_inst, 1)
            avg_cht = sum(e['results'][model].get('mean_charging_time', 0) for e in size_exps if model in e['results']) / max(n_inst, 1)
            print(f'    {label}: cost={avg_cost:.0f}  feas={avg_feas*100:.0f}%  charge_t={avg_cht:.1f}m')

    print(f'\nResults saved to: {interim_path}')


if __name__ == '__main__':
    main()
