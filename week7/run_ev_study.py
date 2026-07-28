#!/usr/bin/env python3
"""
EV Constraint Sensitivity Study.

Sweeps truck count and battery capacity to find where EV constraints
become binding (energy violations > 0, charging stops needed).

This addresses the issue that our default config (8 trucks, 100 kWh)
never triggers EV constraints — routes are too short.
"""

import sys, os, json, math
from datetime import datetime

_W7 = os.path.dirname(os.path.abspath(__file__))
for p in [os.path.join(_W7, '..', 'week3'),
          os.path.join(_W7, '..', 'week5'),
          os.path.join(_W7, '..', 'week4'),
          os.path.join(_W7, '..', 'week6'),
          _W7]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.data_loader import load_instance_from_disk
from run_sota_expanded import run_ours
from ev_problem_model import (
    EVTruckDroneSolution, insert_charging_stops,
    simulate_route_ev, get_charging_station_coords,
    BATTERY_CAPACITY, ENERGY_CONSUMPTION_RATE,
)
from config import DEPOT

OUTPUT_DIR = os.path.join(_W7, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate_ev_impact(instance, routes, battery_kwh, energy_rate, depot, cs_coords):
    """Evaluate EV metrics for a set of routes."""
    total_energy = 0.0
    total_violation = 0.0
    n_routes_violated = 0
    n_routes_need_cs = 0
    max_route_energy = 0.0

    customers = instance['customers']
    dist_matrix = instance['distance_matrix']

    for route in routes:
        if not route:
            continue
        sim = simulate_route_ev(
            route, customers, dist_matrix, cs_coords, depot,
            battery_capacity=battery_kwh,
            energy_rate=energy_rate,
        )
        total_energy += sim['total_energy']
        total_violation += sim['energy_violation']
        max_route_energy = max(max_route_energy, sim['total_energy'])
        if sim['energy_violation'] > 0.01:
            n_routes_violated += 1
        if sim['total_energy'] > battery_kwh:
            n_routes_need_cs += 1

    return {
        'total_energy': total_energy,
        'total_violation': total_violation,
        'max_route_energy': max_route_energy,
        'n_routes_violated': n_routes_violated,
        'n_routes_need_cs': n_routes_need_cs,
        'n_routes': len([r for r in routes if r]),
        'ev_binding': total_violation > 0.01 or n_routes_need_cs > 0,
    }


def main():
    print("EV CONSTRAINT SENSITIVITY STUDY")
    print("=" * 70)

    instances = ['RC101_50c', 'RC101_100c', 'RC201_50c', 'R101_50c', 'C101_50c']
    truck_configs = {
        'RC101_50c': [2, 4, 6],
        'RC101_100c': [2, 4, 6, 8],
        'RC201_50c': [2, 4, 6],
        'R101_50c': [2, 4, 6],
        'C101_50c': [2, 4, 6],
    }
    battery_levels = [30, 50, 75, 100, 150]  # kWh
    energy_rates = [1.0, 1.5, 2.0]  # kWh/km

    all_results = []

    for ik in instances:
        inst = load_instance_from_disk(ik)
        n_cust = inst['n_customers']
        customers = inst['customers']
        depot = inst.get('depot', DEPOT)
        if isinstance(depot, list):
            depot = tuple(depot)
        cs_coords = get_charging_station_coords(n_cust)

        print(f"\n{'─'*60}")
        print(f"  {ik} ({n_cust} customers)")
        print(f"{'─'*60}")

        for n_trucks in truck_configs.get(ik, [4]):
            for battery_kwh in battery_levels:
                for energy_rate in energy_rates:
                    try:
                        sol = run_ours(inst, n_trucks=n_trucks, seed=42,
                                      use_repair=True, repair_mode='full',
                                      n_drones_per_truck=0)
                    except Exception as e:
                        print(f"  {n_trucks}T/{battery_kwh}kWh/@{energy_rate}: ERROR {e}")
                        continue

                    ev = evaluate_ev_impact(
                        inst, sol.truck_routes, battery_kwh, energy_rate,
                        depot, cs_coords)

                    binding = '⚠ BINDING' if ev['ev_binding'] else '✓ ok'
                    viol = ev['total_violation']
                    max_e = ev['max_route_energy']
                    n_v = ev['n_routes_violated']
                    n_cs = ev['n_routes_need_cs']

                    print(f"  {n_trucks}T {battery_kwh:>3}kWh @{energy_rate}kW/km: "
                          f"max_route={max_e:.0f}kWh viol={viol:.0f}kWh "
                          f"routes_viol={n_v}/{ev['n_routes']} need_cs={n_cs} {binding}")

                    all_results.append({
                        'instance': ik,
                        'n_trucks': n_trucks,
                        'battery_kwh': battery_kwh,
                        'energy_rate': energy_rate,
                        **ev,
                    })

    # Summary: find binding threshold
    print(f"\n{'='*70}")
    print("BINDING THRESHOLD ANALYSIS")
    print("=" * 70)

    # Group by instance, find when EV becomes binding
    from collections import defaultdict
    by_instance = defaultdict(list)
    for r in all_results:
        by_instance[r['instance']].append(r)

    for ik, results in sorted(by_instance.items()):
        print(f"\n  {ik}:")
        # Find first binding config
        binding_configs = [r for r in results if r['ev_binding']]
        if binding_configs:
            # Sort by battery then trucks
            binding_configs.sort(key=lambda r: (r['battery_kwh'], -r['n_trucks']))
            first = binding_configs[0]
            print(f"    First binding: {first['n_trucks']} trucks, "
                  f"{first['battery_kwh']} kWh, @{first['energy_rate']} kW/km")
            print(f"    Max route energy: {first['max_route_energy']:.0f} kWh "
                  f"> battery {first['battery_kwh']} kWh")
            print(f"    Routes violated: {first['n_routes_violated']}/{first['n_routes']}")
        else:
            print(f"    No binding configs found — increase route length or reduce battery")

    # Save
    out_path = os.path.join(OUTPUT_DIR, 'ev_study_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == '__main__':
    main()
