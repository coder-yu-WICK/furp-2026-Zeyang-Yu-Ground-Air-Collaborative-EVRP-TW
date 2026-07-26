#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 7 — EV Ablation Study: Models A, B, C with Binding Parameters.

Problem: At default parameters (100 kWh battery, 8 trucks, 200c), EV constraints
are uniformly non-binding. Model A = Model B = Model C, DeltaCost=0%, E-Vio=0,
n_charges=0. Experiment 2 (Charging Study) becomes meaningless.

Solution: Design EV stress-test scenarios at reduced battery capacity (30-50 kWh)
where constraints BIND, enabling meaningful differentiation between:
  - Model A (No EV): baseline, no battery tracking
  - Model B (+Linear Charging): constant-rate charging, CS insertion
  - Model C (+Non-linear Charging): piecewise charging curve (fast->normal->slow)

Differentiation mechanism:
  - Linear charging: constant 1.0 kWh/min rate regardless of SOC
  - Non-linear: 1.5x at 0-20% SOC, 1.0x at 20-80%, 0.5x at 80-100%
  - Charging from low SOC: non-linear is FASTER (1.5x rate) -> lower charge_time
  - Charging from high SOC: non-linear is SLOWER (0.5x rate) -> higher charge_time
  - Net effect: non-linear total charge_time != linear -> DeltaCost(B-C) != 0

Usage:
  python src/experiments/run_ev_ablation.py --test           # Single instance quick test
  python src/experiments/run_ev_ablation.py --quick           # 50c, 1 instance per type
  python src/experiments/run_ev_ablation.py                   # Full study
"""

import json
import os
import sys
import time
import math
import copy
import argparse
from datetime import datetime
from collections import defaultdict

# ── Path setup ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    RC1_INSTANCES, RC2_INSTANCES, R1_INSTANCES, R2_INSTANCES,
    C1_INSTANCES, C2_INSTANCES,
    BATTERY_CAPACITY, CHARGING_RATE, CHARGING_STATIONS, CHARGING_SEGMENTS,
    TRUCK_FIXED_COST, TRUCK_DIST_COST_RATE, TARDINESS_COST_RATE, DEPOT,
)
from src.core.data_loader import load_instance_from_disk, build_all_instances
from src.core.problem_model import TruckDroneSolution

from src.ev.ev_model import (
    EVTruckDroneSolution, insert_charging_stops, simulate_route_ev,
    get_charging_station_coords,
    ENERGY_CONSUMPTION_RATE,
)

from src.experiments.run_sota_expanded import run_ours


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

# Stress-test battery levels where EV constraints BIND.
# Binding analysis: at 100 kWh, no route exceeds battery. At 30 kWh with
# energy_rate=1.5, only ~20% of routes trigger CS. At 25 kWh with
# energy_rate=2.0, 50-100% of routes need CS (varies by instance type).
# We use 25 kWh as primary binding level + 30 kWh for comparison.
BATTERY_LEVELS = [25, 30, 40]       # kWh
ENERGY_RATES = [2.0]                # kWh/km (higher = more binding)

# Instance configs for EV stress test.
# Use CW-Savings constructor for ALL types to get compact routes that
# actually exceed battery capacity. POMO's fragmented routes (12-27 routes
# for 50 customers) are too short individually to trigger EV constraints.
INSTANCE_CONFIGS = [
    # (instance_key, n_trucks, tw_type)
    ('RC101_50c', 3, 'RC1'),
    ('RC201_50c', 3, 'RC2'),
    ('R101_50c', 3, 'R1'),
    ('C101_50c', 3, 'C1'),
    # 100c for scale: fewer trucks = longer routes = more binding
    ('RC101_100c', 3, 'RC1'),
    ('RC201_100c', 3, 'RC2'),
    ('R101_100c', 3, 'R1'),
]


# ═══════════════════════════════════════════════════════════════════════════
# Model Runners
# ═══════════════════════════════════════════════════════════════════════════

def run_model_a(instance, n_trucks, seed):
    """
    Model A: Baseline — standard pipeline (CW-Savings), no EV constraints.

    Uses CW-Savings constructor for consistent comparison with Models B/C.
    No drones — EV constraints are truck-only, drones add confounding variance.

    Returns dict with: cost, tardiness, feasible, truck_dist, n_truck_routes
    """
    from src.experiments.clustering_baselines import clarke_wright_savings
    import random
    random.seed(seed)

    # Use CW-Savings (not adaptive POMO) so Models A/B/C use the same routes
    base_sol = clarke_wright_savings(instance, n_trucks=n_trucks, seed=seed)

    if base_sol is None:
        return {'cost': 1e9, 'tardiness': 1e9, 'feasible': False,
                'error': 'CW-Savings returned None'}

    # Repair tardiness
    if base_sol.tardiness > 1e-6:
        from src.pipeline.repair import repair_tardiness_partial, repair_inter_route
        repaired, stats = repair_tardiness_partial(
            base_sol, instance, seed=seed + 1000, max_drones_per_truck=0)
        if repaired.tardiness > 1e-6:
            repaired, ir_stats = repair_inter_route(
                repaired, instance, max_iter=200, seed=seed + 2000,
                max_drones_per_truck=0)
        base_sol = repaired

    # Repair capacity
    if hasattr(base_sol, '_evaluated'):
        base_sol._evaluated = False
    base_sol._evaluate()

    from src.pipeline.repair import repair_capacity
    cap_sol, cap_stats = repair_capacity(
        base_sol, instance, max_iter=200, seed=seed + 500)
    if cap_stats.get('capacity_violation_before', 0) > 0.01:
        base_sol = cap_sol

    # Compute truck distance
    dist = instance['distance_matrix']
    truck_dist = 0.0
    for route in base_sol.truck_routes:
        if not route:
            continue
        prev = 0
        for cid in route:
            truck_dist += dist[prev][cid]
            prev = cid
        truck_dist += dist[prev][0]

    return {
        'cost': base_sol.cost,
        'tardiness': base_sol.tardiness,
        'feasible': base_sol.feasible,
        'truck_dist': truck_dist,
        'drone_dist': 0,
        'n_drones': 0,
        'n_truck_routes': len(base_sol.truck_routes),
        'solution': base_sol,
    }


def run_model_b(instance, n_trucks, seed, battery_kwh, energy_rate,
                n_drones_per_truck=0):
    """
    Model B: +Linear Charging — CS insertion + linear charging evaluation.

    Uses the same base solution as Model A, then:
    1. Insert charging stops with detour-minimizing heuristic
    2. Evaluate with linear charging model
    3. Add charging_time_cost to total cost

    Returns same dict as run_model_a plus EV-specific fields.
    """
    return _run_ev_model(instance, n_trucks, seed, battery_kwh, energy_rate,
                         charging_model='linear', n_drones_per_truck=n_drones_per_truck)


def run_model_c(instance, n_trucks, seed, battery_kwh, energy_rate,
                n_drones_per_truck=0):
    """
    Model C: +Non-linear Charging — CS insertion + piecewise charging evaluation.

    Same as Model B but with non-linear (piecewise) charging curve.
    The charging time differs due to SOC-dependent charging rates:
      - 0-20%: 1.5x -> faster
      - 20-80%: 1.0x -> same as linear
      - 80-100%: 0.5x -> slower

    Returns same dict as Model B.
    """
    return _run_ev_model(instance, n_trucks, seed, battery_kwh, energy_rate,
                         charging_model='nonlinear', n_drones_per_truck=n_drones_per_truck)


def _run_ev_model(instance, n_trucks, seed, battery_kwh, energy_rate,
                  charging_model='linear', n_drones_per_truck=0):
    """Shared EV pipeline for Models B and C. Uses CW-Savings constructor."""
    import random
    random.seed(seed)

    from src.experiments.clustering_baselines import clarke_wright_savings

    # Step 1: Base solution — CW-Savings (same as Model A for fair comparison)
    base_sol = clarke_wright_savings(instance, n_trucks=n_trucks, seed=seed)

    if base_sol is None:
        return {'cost': 1e9, 'tardiness': 1e9, 'feasible': False,
                'ev_feasible': False, 'error': 'CW-Savings returned None'}

    # Repair tardiness
    if base_sol.tardiness > 1e-6:
        from src.pipeline.repair import repair_tardiness_partial, repair_inter_route
        repaired, stats = repair_tardiness_partial(
            base_sol, instance, seed=seed + 1000, max_drones_per_truck=0)
        if repaired.tardiness > 1e-6:
            repaired, ir_stats = repair_inter_route(
                repaired, instance, max_iter=200, seed=seed + 2000,
                max_drones_per_truck=0)
        base_sol = repaired

    # Repair capacity
    if hasattr(base_sol, '_evaluated'):
        base_sol._evaluated = False
    base_sol._evaluate()

    from src.pipeline.repair import repair_capacity
    cap_sol, cap_stats = repair_capacity(
        base_sol, instance, max_iter=200, seed=seed + 500)
    if cap_stats.get('capacity_violation_before', 0) > 0.01:
        base_sol = cap_sol

    customers = instance['customers']
    dist_matrix = instance['distance_matrix']
    n_cust = len(customers)
    depot = instance.get('depot', DEPOT)
    if isinstance(depot, list):
        depot = tuple(depot)

    # Step 2: Insert charging stops with stress battery capacity
    routes_with_cs, cs_stats = insert_charging_stops(
        base_sol.truck_routes, customers, dist_matrix, instance,
        battery_capacity=battery_kwh,
        energy_rate=energy_rate,
    )

    # Step 3: Create EV solution (no drones for EV-only comparison)
    # NOTE: We compute costs manually rather than using ev_sol.cost because
    # TruckDroneSolution._evaluate() doesn't handle CS node IDs > n_customers.
    # EV cost = base_truck_cost + CS_detour_cost + charging_time_cost
    ev_sol = EVTruckDroneSolution(
        routes_with_cs, [], instance,
        max_drones_per_truck=0,
        charging_model=charging_model,
        battery_capacity=battery_kwh,
        energy_rate=energy_rate,
    )

    # Step 4: Force EV evaluation (battery tracking + charging simulation)
    ev_sol._evaluate_ev()

    # Get base cost from the pre-CS solution
    base_cost = base_sol.cost

    # EV-specific metrics from the EV evaluation
    n_charges = ev_sol.n_charges
    charge_time = ev_sol.total_charge_time
    charge_energy = ev_sol.total_charge_energy
    energy_violation = ev_sol.energy_violation
    total_energy = ev_sol.total_energy
    ev_feasible = ev_sol.ev_feasible

    # Compute truck distance with CS detours
    cs_coords = get_charging_station_coords(n_cust)
    truck_dist_with_cs = 0.0
    for route in routes_with_cs:
        if not route:
            continue
        prev = 0
        for node_id in route:
            if prev > n_cust or node_id > n_cust:
                from src.ev.ev_model import station_distance
                d = station_distance(prev, node_id, customers, cs_coords, depot)
            else:
                d = dist_matrix[prev][node_id]
            truck_dist_with_cs += d
            prev = node_id
        # Return to depot
        if prev > n_cust:
            from src.ev.ev_model import station_distance
            d = station_distance(prev, 0, customers, cs_coords, depot)
        else:
            d = dist_matrix[prev][0]
        truck_dist_with_cs += d

    # Base truck distance (without CS)
    base_truck_dist = 0.0
    for route in base_sol.truck_routes:
        if not route:
            continue
        prev = 0
        for cid in route:
            base_truck_dist += dist_matrix[prev][cid]
            prev = cid
        base_truck_dist += dist_matrix[prev][0]

    cs_detour = truck_dist_with_cs - base_truck_dist
    detour_cost = cs_detour * TRUCK_DIST_COST_RATE

    # Charging time cost (time = money, using tardiness rate)
    charge_time_cost = charge_time * TARDINESS_COST_RATE

    # Total EV-aware cost = base_truck_cost + detour_cost + charging_time_cost
    ev_total_cost = base_cost + detour_cost + charge_time_cost

    # Per-route EV breakdown
    route_breakdown = []
    for ri, detail in enumerate(ev_sol._route_ev_details):
        if detail is None:
            continue
        route_breakdown.append({
            'route_idx': ri,
            'total_energy': detail['total_energy'],
            'final_battery': detail['final_battery'],
            'energy_violation': detail['energy_violation'],
            'n_charges': detail['n_charges'],
            'total_charge_time': detail['total_charge_time'],
            'total_charge_energy': detail['total_charge_energy'],
            'feasible': detail['feasible'],
        })

    return {
        'cost': ev_total_cost,
        'base_cost': base_cost,
        'tardiness': base_sol.tardiness,
        'feasible': base_sol.feasible,
        'ev_feasible': ev_feasible,
        'n_charges': n_charges,
        'charge_time': charge_time,
        'charge_energy': charge_energy,
        'energy_violation': energy_violation,
        'total_energy': total_energy,
        'cs_detour': cs_detour,
        'detour_cost': detour_cost,
        'charge_time_cost': charge_time_cost,
        'cs_insertions': cs_stats['n_insertions'],
        'energy_violation_before_cs': cs_stats['energy_violations_before'],
        'battery_kwh': battery_kwh,
        'energy_rate': energy_rate,
        'charging_model': charging_model,
        'n_truck_routes': len(routes_with_cs),
        'n_drones': 0,
        'route_breakdown': route_breakdown,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════

def print_ev_table(results):
    """Print EV ablation comparison table."""
    print(f"\n{'=' * 140}")
    print(f"  EV ABLATION STUDY — Models A vs B vs C at STRESS PARAMETERS")
    print(f"{'=' * 140}")

    header = (f"  {'Instance':<16s} {'Bat':>5s} {'Er':>4s} "
              f"{'Cost(A)':>10s} {'Cost(B)':>10s} {'Cost(C)':>10s} "
              f"{'Delta(B-A)':>10s} {'Delta(C-B)':>10s} "
              f"{'CS(B)':>5s} {'CS(C)':>5s} "
              f"{'ChgT(B)':>8s} {'ChgT(C)':>8s} "
              f"{'EV-F(B)':>7s} {'EV-F(C)':>7s}")
    print(header)
    print(f"  {'-' * 135}")

    # Group by instance
    for r in results:
        a = r['model_a']
        b = r['model_b']
        c = r['model_c']

        ik = r['instance_key']
        bat = r['battery_kwh']
        er = r['energy_rate']

        # Skip if any model failed
        if a.get('cost', 1e9) >= 1e8:
            continue

        delta_ba = b['cost'] - a['cost']
        delta_cb = c['cost'] - b['cost']

        ev_f_b = "YES" if b.get('ev_feasible', False) else "NO"
        ev_f_c = "YES" if c.get('ev_feasible', False) else "NO"

        n_cs_b = b.get('n_charges', 0)
        n_cs_c = c.get('n_charges', 0)
        cht_b = b.get('charge_time', 0)
        cht_c = c.get('charge_time', 0)

        print(f"  {ik:<16s} {bat:>4.0f}kWh {er:>3.1f}x "
              f"{a['cost']:>10.1f} {b['cost']:>10.1f} {c['cost']:>10.1f} "
              f"{delta_ba:>+10.1f} {delta_cb:>+10.1f} "
              f"{n_cs_b:>5d} {n_cs_c:>5d} "
              f"{cht_b:>8.1f} {cht_c:>8.1f} "
              f"{ev_f_b:>7s} {ev_f_c:>7s}")


def print_charge_time_diff(results):
    """Print detailed charge time comparison between B (linear) and C (non-linear)."""
    print(f"\n{'=' * 120}")
    print(f"  CHARGING MODEL DIFFERENTIATION: Linear (B) vs Non-linear (C)")
    print(f"  Non-linear: 1.5x rate at 0-20% SOC, 1.0x at 20-80%, 0.5x at 80-100%")
    print(f"{'=' * 120}")

    print(f"  {'Instance':<16s} {'Bat':>5s} {'Er':>4s} "
          f"{'ChgT(B)':>10s} {'ChgT(C)':>10s} {'DeltaTime':>10s} {'Delta%':>8s} "
          f"{'n_CS':>5s} {'AvgChg(B)':>10s} {'AvgChg(C)':>10s} "
          f"{'DeltaCost':>10s}")
    print(f"  {'-' * 110}")

    for r in results:
        b = r['model_b']
        c = r['model_c']
        ik = r['instance_key']
        bat = r['battery_kwh']
        er = r['energy_rate']

        if b.get('cost', 1e9) >= 1e8 or c.get('cost', 1e9) >= 1e8:
            continue

        cht_b = b.get('charge_time', 0)
        cht_c = c.get('charge_time', 0)
        n_cs = b.get('n_charges', 0)

        if cht_b < 0.01 and cht_c < 0.01:
            continue

        delta_t = cht_c - cht_b
        delta_pct = (delta_t / cht_b * 100) if cht_b > 0.01 else 0
        avg_b = cht_b / n_cs if n_cs > 0 else 0
        avg_c = cht_c / n_cs if n_cs > 0 else 0
        delta_cost = c['cost'] - b['cost']

        print(f"  {ik:<16s} {bat:>4.0f}kWh {er:>3.1f}x "
              f"{cht_b:>10.1f} {cht_c:>10.1f} {delta_t:>+10.1f} {delta_pct:>+7.1f}% "
              f"{n_cs:>5d} {avg_b:>10.1f} {avg_c:>10.1f} "
              f"{delta_cost:>+10.1f}")


def print_summary_by_battery(results):
    """Aggregate results by battery level."""
    print(f"\n{'=' * 100}")
    print(f"  AGGREGATE SUMMARY BY BATTERY LEVEL")
    print(f"{'=' * 100}")

    groups = defaultdict(list)
    for r in results:
        groups[(r['battery_kwh'], r['energy_rate'])].append(r)

    for (bat, er), items in sorted(groups.items()):
        n = len(items)
        costs_a = [r['model_a']['cost'] for r in items if r['model_a'].get('cost', 1e9) < 1e8]
        costs_b = [r['model_b']['cost'] for r in items if r['model_b'].get('cost', 1e9) < 1e8]
        costs_c = [r['model_c']['cost'] for r in items if r['model_c'].get('cost', 1e9) < 1e8]

        n_cs_b = [r['model_b'].get('n_charges', 0) for r in items if r['model_b'].get('cost', 1e9) < 1e8]
        n_cs_c = [r['model_c'].get('n_charges', 0) for r in items if r['model_c'].get('cost', 1e9) < 1e8]
        cht_b = [r['model_b'].get('charge_time', 0) for r in items if r['model_b'].get('cost', 1e9) < 1e8]
        cht_c = [r['model_c'].get('charge_time', 0) for r in items if r['model_c'].get('cost', 1e9) < 1e8]
        ev_b = [r['model_b'].get('ev_feasible', False) for r in items]
        ev_c = [r['model_c'].get('ev_feasible', False) for r in items]

        if not costs_a:
            continue

        avg_a = sum(costs_a) / len(costs_a)
        avg_b = sum(costs_b) / len(costs_b)
        avg_c = sum(costs_c) / len(costs_c)
        avg_cs_b = sum(n_cs_b) / len(n_cs_b)
        avg_cs_c = sum(n_cs_c) / len(n_cs_c)
        avg_cht_b = sum(cht_b) / len(cht_b)
        avg_cht_c = sum(cht_c) / len(cht_c)
        ev_rate_b = sum(1 for v in ev_b if v) / len(ev_b) * 100
        ev_rate_c = sum(1 for v in ev_c if v) / len(ev_c) * 100

        delta_ba = avg_b - avg_a
        delta_cb = avg_c - avg_b
        delta_cht = avg_cht_c - avg_cht_b

        print(f"\n  Battery={bat:.0f}kWh, EnergyRate={er:.1f}x ({n} instances):")
        print(f"    Model A:         avg cost={avg_a:.1f}")
        print(f"    Model B (linear):    avg cost={avg_b:.1f}  "
              f"Delta(B-A)={delta_ba:+.1f}  CS={avg_cs_b:.1f}  "
              f"ChgTime={avg_cht_b:.1f}  EV-feas={ev_rate_b:.0f}%")
        print(f"    Model C (nonlin):    avg cost={avg_c:.1f}  "
              f"Delta(C-B)={delta_cb:+.1f}  CS={avg_cs_c:.1f}  "
              f"ChgTime={avg_cht_c:.1f}  EV-feas={ev_rate_c:.0f}%")
        print(f"    DeltaChgTime(C-B)={delta_cht:+.1f} min "
              f"({'C slower' if delta_cht > 0 else 'C faster'})")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="EV Ablation Study — Models A/B/C at Binding Parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--test', action='store_true',
                       help='Single instance quick test')
    parser.add_argument('--quick', action='store_true',
                       help='50c only, 1 battery level')
    parser.add_argument('--battery', type=float, nargs='+',
                       default=[30, 40, 50],
                       help='Battery capacities (kWh)')
    parser.add_argument('--energy-rate', type=float, nargs='+',
                       default=[1.5, 2.0],
                       help='Energy consumption rates (kWh/km)')
    parser.add_argument('--instance', type=str,
                       help='Run on a specific instance')
    parser.add_argument('--output-dir', type=str,
                       default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ── Build configs ──
    if args.test:
        configs = [
            ('R101_50c', 3, 'R1'),
        ]
        battery_levels = [25]
        energy_rates = [2.0]
    elif args.quick:
        configs = [c for c in INSTANCE_CONFIGS if '50c' in c[0]]
        battery_levels = [25]
        energy_rates = [2.0]
    elif args.instance:
        parts = args.instance.rsplit('_', 1)
        src = parts[0]
        nc = int(parts[1].replace('c', ''))
        for twt, inst_list in {
            'RC1': RC1_INSTANCES, 'RC2': RC2_INSTANCES,
            'R1': R1_INSTANCES, 'R2': R2_INSTANCES,
            'C1': C1_INSTANCES, 'C2': C2_INSTANCES,
        }.items():
            if src in inst_list:
                tw_type = twt
                break
        else:
            tw_type = 'RC2'
        n_trucks = 3 if nc <= 50 else 4
        configs = [(args.instance, n_trucks, tw_type)]
        battery_levels = args.battery
        energy_rates = args.energy_rate
    else:
        configs = INSTANCE_CONFIGS
        battery_levels = args.battery
        energy_rates = args.energy_rate

    # ── Pre-build instances ──
    build_all_instances()

    print(f"\n{'#' * 120}")
    print(f"  EV ABLATION STUDY — MODELS A, B, C AT STRESS PARAMETERS")
    print(f"  Instances: {len(configs)}")
    print(f"  Battery levels: {battery_levels} kWh")
    print(f"  Energy rates: {energy_rates} kWh/km")
    print(f"  Seed: {args.seed}")
    print(f"{'#' * 120}")

    print(f"\n  NOTE: Default params (100 kWh, 8 trucks) produce Model A = B = C.")
    print(f"  This study uses STRESS params (30-50 kWh, 3-4 trucks) to demonstrate")
    print(f"  EV constraint binding and linear vs non-linear charging differentiation.")

    # ── Run experiments ──
    all_results = []
    total_configs = len(configs) * len(battery_levels) * len(energy_rates)
    n_done = 0

    for ik, n_trucks, tw_type in configs:
        inst = load_instance_from_disk(ik)

        for battery_kwh in battery_levels:
            for energy_rate in energy_rates:
                n_done += 1
                print(f"\n[{n_done}/{total_configs}] {ik} "
                      f"({n_trucks}T, {battery_kwh:.0f}kWh, {energy_rate:.1f}x)")

                # Model A: Standard pipeline (no EV)
                t0 = time.time()
                result_a = run_model_a(inst, n_trucks, args.seed)
                t_a = time.time() - t0

                if result_a.get('cost', 1e9) >= 1e8:
                    print(f"  Model A: FAILED — skipping EV models")
                    continue

                print(f"  Model A (No EV):        cost={result_a['cost']:.1f}  "
                      f"tard={result_a['tardiness']:.1f}  "
                      f"feas={result_a['feasible']}  "
                      f"routes={result_a.get('n_truck_routes', '?')}  "
                      f"t={t_a:.1f}s")

                # Model B: Linear charging
                t0 = time.time()
                result_b = run_model_b(inst, n_trucks, args.seed,
                                      battery_kwh, energy_rate,
                                      n_drones_per_truck=0)
                t_b = time.time() - t0

                print(f"  Model B (Linear EV):    cost={result_b['cost']:.1f}  "
                      f"base={result_b.get('base_cost', 0):.1f}  "
                      f"detour={result_b.get('cs_detour', 0):.1f}km  "
                      f"chg_time={result_b.get('charge_time', 0):.1f}min  "
                      f"n_cs={result_b.get('n_charges', 0)}  "
                      f"EV-feas={result_b.get('ev_feasible', False)}  "
                      f"t={t_b:.1f}s")

                # Model C: Non-linear charging
                t0 = time.time()
                result_c = run_model_c(inst, n_trucks, args.seed,
                                      battery_kwh, energy_rate,
                                      n_drones_per_truck=0)
                t_c = time.time() - t0

                delta_cb = result_c['cost'] - result_b['cost']
                delta_cht = result_c.get('charge_time', 0) - result_b.get('charge_time', 0)

                print(f"  Model C (Nonlin EV):    cost={result_c['cost']:.1f}  "
                      f"base={result_c.get('base_cost', 0):.1f}  "
                      f"detour={result_c.get('cs_detour', 0):.1f}km  "
                      f"chg_time={result_c.get('charge_time', 0):.1f}min  "
                      f"n_cs={result_c.get('n_charges', 0)}  "
                      f"EV-feas={result_c.get('ev_feasible', False)}  "
                      f"t={t_c:.1f}s")

                print(f"  Delta(C-B): cost={delta_cb:+.1f}  "
                      f"chg_time={delta_cht:+.1f}min  "
                      f"{'! C!=B !' if abs(delta_cb) > 0.5 else 'C~=B (non-binding)'}")

                all_results.append({
                    'instance_key': ik,
                    'tw_type': tw_type,
                    'n_trucks': n_trucks,
                    'battery_kwh': battery_kwh,
                    'energy_rate': energy_rate,
                    'model_a': {k: v for k, v in result_a.items() if k != 'solution'},
                    'model_b': {k: v for k, v in result_b.items() if k != 'route_breakdown'},
                    'model_c': {k: v for k, v in result_c.items() if k != 'route_breakdown'},
                    'runtime_a': t_a,
                    'runtime_b': t_b,
                    'runtime_c': t_c,
                })

    # ── Print reports ──
    print_ev_table(all_results)
    print_charge_time_diff(all_results)
    print_summary_by_battery(all_results)

    # ── Key findings ──
    print(f"\n{'=' * 100}")
    print(f"  KEY FINDINGS")
    print(f"{'=' * 100}")

    valid = [r for r in all_results
             if r['model_a'].get('cost', 1e9) < 1e8]

    if valid:
        # 1. EV binding rate
        n_binding_b = sum(1 for r in valid if r['model_b'].get('n_charges', 0) > 0)
        n_binding_c = sum(1 for r in valid if r['model_c'].get('n_charges', 0) > 0)

        print(f"\n  1. EV CONSTRAINT BINDING:")
        print(f"     Configs with charging stops needed: {n_binding_b}/{len(valid)} (B), {n_binding_c}/{len(valid)} (C)")

        # 2. Cost of electrification
        costs_a = [r['model_a']['cost'] for r in valid]
        costs_b = [r['model_b']['cost'] for r in valid]
        avg_delta_ba = sum(b - a for a, b in zip(costs_a, costs_b)) / len(valid)
        avg_a = sum(costs_a) / len(costs_a)
        pct_ba = (avg_delta_ba / avg_a * 100) if avg_a > 0 else 0
        print(f"\n  2. COST OF ELECTRIFICATION (B vs A):")
        print(f"     Avg DeltaCost(B-A): {avg_delta_ba:+.1f} ({pct_ba:+.1f}% vs baseline)")

        # 3. Linear vs Non-linear differentiation
        delta_cb_list = [r['model_c']['cost'] - r['model_b']['cost'] for r in valid]
        delta_cht_list = [r['model_c'].get('charge_time', 0) - r['model_b'].get('charge_time', 0) for r in valid]
        n_differentiated = sum(1 for d in delta_cb_list if abs(d) > 0.5)

        print(f"\n  3. LINEAR vs NON-LINEAR DIFFERENTIATION:")
        print(f"     Configs with DeltaCost(C-B) > 0.5: {n_differentiated}/{len(valid)}")
        if delta_cht_list:
            avg_delta_cht = sum(delta_cht_list) / len(delta_cht_list)
            print(f"     Avg DeltaChargeTime(C-B): {avg_delta_cht:+.1f} min "
                  f"({'non-linear slower' if avg_delta_cht > 0 else 'non-linear faster'})")

        # 4. By battery level
        print(f"\n  4. BATTERY BINDING THRESHOLD:")
        for bat in sorted(set(r['battery_kwh'] for r in valid)):
            bat_results = [r for r in valid if r['battery_kwh'] == bat]
            n_cs = sum(1 for r in bat_results if r['model_b'].get('n_charges', 0) > 0)
            avg_cs = sum(r['model_b'].get('n_charges', 0) for r in bat_results) / len(bat_results)
            avg_ev = sum(1 for r in bat_results if r['model_b'].get('ev_feasible', False)) / len(bat_results) * 100
            print(f"     {bat:.0f} kWh: {n_cs}/{len(bat_results)} need CS, "
                  f"avg {avg_cs:.1f} CS/instance, {avg_ev:.0f}% EV-feasible")

    # ── Save results ──
    output_path = os.path.join(args.output_dir, f'ev_ablation_{timestamp}.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Results saved to: {output_path}")
    print(f"{'#' * 120}")


if __name__ == '__main__':
    main()
