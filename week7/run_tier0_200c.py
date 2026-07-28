#!/usr/bin/env python3
"""
Tier 0 — 200c Experiments with EV Ablation (Models A/B/C).

Model A (Baseline):         Standard pipeline, no EV constraints
Model B (+Linear Charging):  EV evaluation with linear charging stops
Model C (+Non-linear):       EV evaluation with non-linear (piecewise) charging

12 configs: 6 Solomon types × 200c × 14 methods + EV ablation
"""

import sys, os, time, json, math
from datetime import datetime

# Path setup
_W7 = os.path.dirname(os.path.abspath(__file__))
# IMPORTANT: Insert in REVERSE order because insert(0) puts last-first.
# Week7 must take priority, then week6, then week4, then week5, then week3.
for p in [os.path.join(_W7, '..', 'week3'),
          os.path.join(_W7, '..', 'week5'),
          os.path.join(_W7, '..', 'week4'),
          os.path.join(_W7, '..', 'week6'),
          _W7]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.data_loader import load_instance_from_disk
from run_sota_expanded import (
    run_one_method, METHOD_REGISTRY, run_ours, _extract_best, _std,
)
from utils.problem_model import TruckDroneSolution
from ev_problem_model import (
    EVTruckDroneSolution, insert_charging_stops, simulate_route_ev,
    get_charging_station_coords,
    BATTERY_CAPACITY, ENERGY_CONSUMPTION_RATE, CHARGING_RATE,
)

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

REPRESENTATIVES = {
    'RC1': 'RC101', 'RC2': 'RC201',
    'R1': 'R101',   'R2': 'R201',
    'C1': 'C101',   'C2': 'C201',
}
SIZE = 200
N_TRUCKS = 8   # 8 trucks for 200 customers (4:1 ratio)
REPAIR_MODE = 'full'
BASE_SEED = 42

# Methods with per-method repeat counts
# P-ACO at 200c is extremely slow (~600s/run). NSGA-II also slow (~80s).
METHODS = [
    # Our methods — 3 reps each
    ('ours_full', 3),
    ('ours_1drone', 3),
    ('ours_no_drone', 3),
    ('ours_no_edd', 3),
    ('ours_partial_edd', 3),
    # IVND only (fast: ~0.5s). NSGA-II and P-ACO skipped at 200c —
    # they don't scale (NSGA-II hung overnight, P-ACO ~600s/run)
    # and already fail TW constraints at 50c/100c anyway.
    ('ivnd', 1),
    # Clustering-first baselines — 3 reps
    ('sweep_nn', 3),
    ('cw_savings', 3),
    ('kmeans_nn', 3),
    ('kmeans_2opt', 3),
    ('sweep_pomo', 3),
    ('cw_pomo', 3),
]

# EV Ablation models (run on our best pipeline results)
EV_MODELS = [
    ('ev_linear', 'linear', 'Model B: +Linear Charging'),
    ('ev_nonlinear', 'nonlinear', 'Model C: +Non-linear Charging'),
]

# ═══════════════════════════════════════════════════════════════════════════
# EV Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_ev_pipeline(instance, n_trucks, seed, charging_model='linear',
                    n_drones_per_truck=2):
    """
    Run our pipeline with EV constraints.

    1. Run standard pipeline to get base solution
    2. Wrap as EVTruckDroneSolution
    3. Insert charging stops where needed
    4. Evaluate with EV battery tracking
    """
    import random
    random.seed(seed)

    # Step 1: Standard pipeline (same as ours_full without EV)
    base_sol = run_ours(instance, n_trucks, seed, use_repair=True,
                        repair_mode=REPAIR_MODE,
                        n_drones_per_truck=n_drones_per_truck)

    if base_sol is None:
        return None

    # Step 2: Insert charging stops into truck routes
    customers = instance['customers']
    dist_matrix = instance['distance_matrix']
    n_cust = len(customers)

    routes_with_cs, cs_stats = insert_charging_stops(
        base_sol.truck_routes, customers, dist_matrix, instance,
        battery_capacity=BATTERY_CAPACITY,
        energy_rate=ENERGY_CONSUMPTION_RATE,
    )

    # Step 3: Create EV solution with charging stations
    ev_sol = EVTruckDroneSolution(
        routes_with_cs, base_sol.drone_missions, instance,
        max_drones_per_truck=n_drones_per_truck,
        charging_model=charging_model,
        battery_capacity=BATTERY_CAPACITY,
        energy_rate=ENERGY_CONSUMPTION_RATE,
    )

    # Step 4: Force EV evaluation
    ev_sol._evaluate_ev()

    # Step 5: Adjust cost for charging time (charging adds to route time)
    # Base cost + charging_time penalty (time = money)
    base_cost = ev_sol.cost
    charging_time_cost = ev_sol.total_charge_time * TARDINESS_COST_RATE
    ev_sol._cost = base_cost + charging_time_cost

    return ev_sol, cs_stats


# ═══════════════════════════════════════════════════════════════════════════
# EV Method Runner
# ═══════════════════════════════════════════════════════════════════════════

TARDINESS_COST_RATE = 1.0  # from config


def run_ev_method(inst, cfg, charging_model, n_repeats, base_seed):
    """Run EV pipeline n_repeats times and return aggregated metrics."""
    n_trucks = cfg['n_trucks']
    costs, tards, feas, rts = [], [], [], []
    ev_feas_list, energy_violations, n_charges_list = [], [], []
    charge_energies, charge_times = [], []
    per_run = []

    for rep in range(n_repeats):
        t0 = time.time()
        run_seed = base_seed + rep

        try:
            result = run_ev_pipeline(
                inst, n_trucks, run_seed,
                charging_model=charging_model,
                n_drones_per_truck=2,
            )

            if result is not None:
                ev_sol, cs_stats = result
                rt = time.time() - t0
                rts.append(rt)
                costs.append(ev_sol.cost)
                tards.append(ev_sol.tardiness)
                feas.append(1.0 if ev_sol.feasible else 0.0)
                ev_feas_list.append(1.0 if ev_sol.ev_feasible else 0.0)
                energy_violations.append(ev_sol.energy_violation)
                n_charges_list.append(ev_sol.n_charges)
                charge_energies.append(ev_sol.total_charge_energy)
                charge_times.append(ev_sol.total_charge_time)
                per_run.append({
                    'cost': ev_sol.cost,
                    'tardiness': ev_sol.tardiness,
                    'feasible': ev_sol.feasible,
                    'ev_feasible': ev_sol.ev_feasible,
                    'n_drones': len(getattr(ev_sol, 'drone_missions', [])),
                    'energy_violation': ev_sol.energy_violation,
                    'n_charges': ev_sol.n_charges,
                    'total_charge_energy': ev_sol.total_charge_energy,
                    'total_charge_time': ev_sol.total_charge_time,
                    'runtime': rt,
                    'cs_insertions': cs_stats['n_insertions'],
                })
            else:
                rt = time.time() - t0
                rts.append(rt)
                costs.append(1e9)
                tards.append(1e9)
                feas.append(0.0)
                ev_feas_list.append(0.0)
                energy_violations.append(1e9)
                n_charges_list.append(0)
                charge_energies.append(0)
                charge_times.append(0)
                per_run.append({'cost': 1e9, 'tardiness': 1e9, 'feasible': False,
                              'ev_feasible': False, 'runtime': rt})

        except Exception as e:
            rt = time.time() - t0
            rts.append(rt)
            costs.append(1e9)
            tards.append(1e9)
            feas.append(0.0)
            ev_feas_list.append(0.0)
            energy_violations.append(1e9)
            n_charges_list.append(0)
            charge_energies.append(0)
            charge_times.append(0)
            per_run.append({'cost': 1e9, 'tardiness': 1e9, 'feasible': False,
                          'ev_feasible': False, 'runtime': rt, 'error': str(e)})

    n = len(costs)
    if n == 0:
        return _empty_ev_result()

    mc = sum(costs) / n
    mt = sum(tards) / n

    return {
        'mean_cost': mc,
        'std_cost': _std(costs, mc) if n > 1 else 0,
        'mean_tardiness': mt,
        'std_tardiness': _std(tards, mt) if n > 1 else 0,
        'feasibility_rate': sum(feas) / n,
        'ev_feasibility_rate': sum(ev_feas_list) / n,
        'mean_energy_violation': sum(v for v in energy_violations if v < 1e8) / max(1, sum(1 for v in energy_violations if v < 1e8)),
        'mean_n_charges': sum(n_charges_list) / n,
        'mean_charge_energy': sum(charge_energies) / n,
        'mean_charge_time': sum(charge_times) / n,
        'mean_runtime': sum(rts) / n,
        'std_runtime': _std(rts, sum(rts)/n) if n > 1 else 0,
        'best_cost': min(costs),
        'best_tardiness': min(tards),
        'n_successful': sum(1 for f in feas if f > 0),
        'per_run': per_run,
    }


def _empty_ev_result():
    return {
        'mean_cost': 1e9, 'std_cost': 0,
        'mean_tardiness': 1e9, 'std_tardiness': 0,
        'feasibility_rate': 0.0,
        'ev_feasibility_rate': 0.0,
        'mean_energy_violation': 1e9,
        'mean_n_charges': 0,
        'mean_charge_energy': 0,
        'mean_charge_time': 0,
        'mean_runtime': 0, 'std_runtime': 0,
        'best_cost': 1e9, 'best_tardiness': 1e9,
        'n_successful': 0, 'per_run': [],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    configs = []
    for tw_type, src_inst in REPRESENTATIVES.items():
        ik = f'{src_inst}_{SIZE}c'
        try:
            load_instance_from_disk(ik)
        except FileNotFoundError:
            print(f'WARNING: {ik} not found, skipping')
            continue
        configs.append({
            'instance_key': ik,
            'source_instance': src_inst,
            'n_customers': SIZE,
            'tw_type': tw_type,
            'n_trucks': N_TRUCKS,
            'repair_mode': REPAIR_MODE,
        })

    total_standard = sum(nr for _, nr in METHODS) * len(configs)
    total_ev = sum(3 for _ in EV_MODELS) * len(configs)
    total_runs = total_standard + total_ev

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(outdir, exist_ok=True)

    print(f'{"=" * 90}')
    print(f'  Tier 0 — 200c EXPERIMENTS WITH EV ABLATION')
    print(f'  {"=" * 90}')
    print(f'  Configs: {len(configs)} (6 Solomon types × 200c)')
    print(f'  Trucks: {N_TRUCKS}')
    print(f'  Standard methods: {len(METHODS)} × {len(configs)} instances = {total_standard} runs')
    print(f'  EV ablation: {len(EV_MODELS)} models × {len(configs)} instances = {total_ev} runs')
    print(f'  Total: {total_runs} runs')
    print(f'  Started: {timestamp}')

    # Estimate
    per_instance_est = sum(
        nr * (8 if m.startswith('ours') else (2 if m == 'ivnd' else 0.5))
        for m, nr in METHODS
    )
    per_instance_est += sum(3 * 12 for _ in EV_MODELS)  # EV pipeline ~12s each
    est_min = per_instance_est * len(configs) / 60
    print(f'  Estimated: ~{est_min:.0f} min')
    print()

    all_results = []

    for ci, cfg in enumerate(configs):
        inst = load_instance_from_disk(cfg['instance_key'])
        print(f'\n[{ci+1}/{len(configs)}] {cfg["instance_key"]} '
              f'({cfg["tw_type"]}, {cfg["n_customers"]}c, {cfg["n_trucks"]} trucks)')
        print(f'  {"─" * 80}')
        sys.stdout.flush()

        methods = {}

        # ── Standard methods ──
        for mi, (mkey, n_reps) in enumerate(METHODS):
            info = METHOD_REGISTRY[mkey]
            t0 = time.time()
            methods[mkey] = run_one_method(inst, cfg, mkey, n_repeats=n_reps, base_seed=BASE_SEED)
            elapsed = time.time() - t0
            m = methods[mkey]
            star = ' ⭐' if m['feasibility_rate'] >= 0.99 and m['mean_tardiness'] < 1.0 else ''
            print(f'  [{mi+1}/{len(METHODS)}] {info["short"]:<20s} '
                  f'cost={m["mean_cost"]:>10.1f} tard={m["mean_tardiness"]:>8.1f} '
                  f'feas={m["feasibility_rate"]*100:>5.0f}% t={elapsed:>6.1f}s{star}')
            sys.stdout.flush()

        # ── EV Ablation ──
        ev_methods = {}
        for ev_key, charging_model, ev_desc in EV_MODELS:
            t0 = time.time()
            ev_methods[ev_key] = run_ev_method(inst, cfg, charging_model,
                                                n_repeats=3, base_seed=BASE_SEED)
            elapsed = time.time() - t0
            m = ev_methods[ev_key]
            print(f'  [EV] {ev_desc:<35s} '
                  f'cost={m["mean_cost"]:>10.1f} tard={m["mean_tardiness"]:>8.1f} '
                  f'TW-feas={m["feasibility_rate"]*100:>5.0f}% '
                  f'EV-feas={m["ev_feasibility_rate"]*100:>5.0f}% '
                  f'E-vio={m["mean_energy_violation"]:>6.1f}kWh '
                  f'CS={m["mean_n_charges"]:>4.0f} t={elapsed:>5.1f}s')
            sys.stdout.flush()

        # ── EV impact summary ──
        if 'ours_full' in methods and 'ev_linear' in ev_methods:
            base_cost = methods['ours_full']['mean_cost']
            ev_cost = ev_methods['ev_linear']['mean_cost']
            ev_vio = ev_methods['ev_linear']['mean_energy_violation']
            ev_cs = ev_methods['ev_linear']['mean_n_charges']
            print(f'  {"─" * 80}')
            print(f'  EV Impact (linear): Δcost=+{ev_cost - base_cost:.1f} '
                  f'({(ev_cost/base_cost - 1)*100:+.1f}%), '
                  f'energy_vio={ev_vio:.1f}kWh, n_charges={ev_cs:.0f}')

        all_results.append({
            'instance_key': cfg['instance_key'],
            'source_instance': cfg['source_instance'],
            'n_customers': cfg['n_customers'],
            'tw_type': cfg['tw_type'],
            'n_trucks': cfg['n_trucks'],
            'methods': methods,
            'ev_methods': ev_methods,
        })

        # Interim save
        interim_path = os.path.join(outdir, f'week7_tier0_200c_{timestamp}.json')
        with open(interim_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    # ═══════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════
    print(f'\n{"=" * 90}')
    print(f'  200c SUMMARY')
    print(f'{"=" * 90}')

    for r in all_results:
        ik = r['instance_key']
        tw = r['tw_type']
        print(f'\n  {ik} ({tw}):')
        # Best methods by cost
        sorted_methods = sorted(
            [(k, v) for k, v in r['methods'].items() if v['mean_cost'] < 1e8],
            key=lambda x: x[1]['mean_cost']
        )
        print(f'    Best cost: {sorted_methods[0][0]} = {sorted_methods[0][1]["mean_cost"]:.1f}')
        # Our methods
        for mk in ['ours_full', 'ours_1drone', 'ours_no_drone']:
            if mk in r['methods']:
                m = r['methods'][mk]
                print(f'    {mk}: cost={m["mean_cost"]:.1f} tard={m["mean_tardiness"]:.1f} '
                      f'feas={m["feasibility_rate"]*100:.0f}%')
        # EV
        if 'ev_methods' in r:
            for ek, em in r['ev_methods'].items():
                print(f'    {ek}: cost={em["mean_cost"]:.1f} '
                      f'TW-feas={em["feasibility_rate"]*100:.0f}% '
                      f'EV-feas={em["ev_feasibility_rate"]*100:.0f}% '
                      f'E-vio={em["mean_energy_violation"]:.1f}kWh '
                      f'CS={em["mean_n_charges"]:.1f}')

    # EV Cross-type summary
    print(f'\n{"=" * 90}')
    print(f'  EV ABLATION CROSS-TYPE SUMMARY')
    print(f'{"=" * 90}')
    print(f'  {"Type":<8s} {"Model":<14s} {"ΔCost%":>8s} {"EV-Feas%":>9s} '
          f'{"E-Vio(kWh)":>11s} {"#CS":>5s} {"ChgTime":>8s}')
    print(f'  {"─" * 75}')

    for r in all_results:
        tw = r['tw_type']
        base_cost = r['methods'].get('ours_full', {}).get('mean_cost', 1)
        for ek, em in r.get('ev_methods', {}).items():
            model_name = 'Linear' if 'linear' in ek else 'Non-linear'
            dc_pct = ((em['mean_cost'] - base_cost) / base_cost * 100) if base_cost > 0 and base_cost < 1e8 else 0
            print(f'  {tw:<8s} {model_name:<14s} {dc_pct:>+7.1f}% '
                  f'{em["ev_feasibility_rate"]*100:>8.1f}% '
                  f'{em["mean_energy_violation"]:>10.1f} '
                  f'{em["mean_n_charges"]:>5.1f} '
                  f'{em["mean_charge_time"]:>7.1f}')

    final_path = os.path.join(outdir, f'week7_tier0_200c_{timestamp}.json')
    with open(final_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f'\nDone. Results: {final_path}')


if __name__ == '__main__':
    main()
