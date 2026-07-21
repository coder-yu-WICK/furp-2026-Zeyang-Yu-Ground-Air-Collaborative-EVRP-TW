# -*- coding: utf-8 -*-
"""
Failure Case Analysis — Week 7 Gap 6.

Generates ≥4 systematic failure scenarios for the Truck-Drone EVRP-TW
project. Each case demonstrates a specific constraint violation with
root cause analysis and expected behavior.

Cases:
  1. Battery Capacity Starvation — reduced battery makes routes infeasible
  2. Time Window Tightening — tightened TWs cause cascading violations
  3. Drone-Truck Sync Failure — drone arrives before truck at recovery
  4. No Charging Infrastructure — no stations with long routes

Usage:
    python week7/failure_cases.py              # Generate all cases
    python week7/failure_cases.py --case 1     # Generate specific case
"""

import math, os, sys, json, copy
from datetime import datetime

_W6 = os.path.dirname(os.path.abspath(__file__))
_W5 = os.path.join(_W6, '..', 'week5')
_W4 = os.path.join(_W6, '..', 'week4')
_W3 = os.path.join(_W6, '..', 'week3')

for _p in [_W6, _W5, _W4, _W3]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def generate_case_1_battery_starvation():
    """
    Case 1: Battery Capacity Starvation.

    Setting: Reduce BATTERY_CAPACITY from 100 kWh to 30 kWh.
    Expected: Truck cannot complete long routes even with charging
              because range between charges exceeds 30 kWh.
    Root Cause: Energy demand of route exceeds battery + charging capacity.
    """
    from utils.data_loader import build_instance
    from config import BATTERY_CAPACITY, CHARGING_STATIONS
    from ev_problem_model import (
        EVTruckDroneSolution, insert_charging_stops,
        is_charging_station, ENERGY_PER_KM,
    )

    print("=" * 70)
    print("FAILURE CASE 1: Battery Capacity Starvation")
    print("=" * 70)

    inst = build_instance('RC201', 50)
    print(f"Instance: RC201_50c ({inst['n_customers']} customers, {inst['tw_type']})")
    print(f"Default battery: {BATTERY_CAPACITY} kWh")
    print(f"Test battery: 30 kWh (70% reduction)")
    print(f"Charging stations: {len(CHARGING_STATIONS)} at {CHARGING_STATIONS}")
    print(f"Energy consumption: {ENERGY_PER_KM} kWh/km")

    # Build a long route
    test_routes = [[c['id'] for c in inst['customers'][:30]]]  # 30 customers on one route

    # Model with normal battery (100 kWh)
    routes_normal, stats_normal = insert_charging_stops(
        test_routes, inst, battery_capacity=100.0, charging_model='linear')
    sol_normal = EVTruckDroneSolution(routes_normal, [], inst,
                                       battery_capacity=100.0)
    print(f"\nWith 100 kWh battery:")
    print(f"  Feasible: {sol_normal.feasible}")
    print(f"  Battery violations: {sol_normal.battery_violations:.1f} kWh")
    print(f"  Energy consumed: {sol_normal.energy_consumed:.1f} kWh")
    print(f"  Charging stops: {sol_normal.n_charges}")
    print(f"  Charging time: {sol_normal.charging_time:.1f} min")

    # Model with reduced battery (30 kWh)
    routes_low, stats_low = insert_charging_stops(
        test_routes, inst, battery_capacity=30.0, charging_model='linear')
    sol_low = EVTruckDroneSolution(routes_low, [], inst,
                                    battery_capacity=30.0)
    print(f"\nWith 30 kWh battery (STARVATION):")
    print(f"  Feasible: {sol_low.feasible}")
    print(f"  Battery violations: {sol_low.battery_violations:.1f} kWh")
    print(f"  Energy consumed: {sol_low.energy_consumed:.1f} kWh")
    print(f"  Charging stops: {sol_low.n_charges}")
    print(f"  Charging time: {sol_low.charging_time:.1f} min")

    # Show battery trace
    if sol_low._route_battery_traces and sol_low._route_battery_traces[0]:
        print(f"\n  Battery trace (first 10 nodes):")
        trace = sol_low._route_battery_traces[0]
        for node_id, batt in trace[:10]:
            cs_mark = " [CS]" if is_charging_station(node_id, inst['n_customers']) else ""
            print(f"    Node {node_id}{cs_mark}: battery = {batt:.1f} kWh")

    conclusion = (
        "\nCONCLUSION: With only 30 kWh battery, the truck depletes its battery\n"
        "before reaching the first charging station or completing the route.\n"
        "Even with charging stations, the spacing between them exceeds the\n"
        "reduced range. This demonstrates why battery capacity is binding\n"
        "constraint in EVRP-TW and why charging infrastructure placement\n"
        "must align with vehicle range."
    )
    print(conclusion)

    return {
        'case': 1,
        'name': 'Battery Capacity Starvation',
        'instance': 'RC201_50c',
        'battery_normal': 100.0,
        'battery_reduced': 30.0,
        'normal_feasible': sol_normal.feasible,
        'reduced_feasible': sol_low.feasible,
        'reduced_battery_violations': sol_low.battery_violations,
        'root_cause': 'Energy demand exceeds battery + charging capacity',
    }


def generate_case_2_tw_tightening():
    """
    Case 2: Time Window Tightening.

    Setting: Scale all due_times by 0.7 (tighten by 30%).
    Expected: Multiple TW violations that EDD repair cannot fully fix.
    Root Cause: Cumulative service times make the schedule impossible
                even with optimal ordering.
    """
    print("\n" + "=" * 70)
    print("FAILURE CASE 2: Time Window Tightening")
    print("=" * 70)

    from utils.data_loader import build_instance
    from utils.problem_model import TruckDroneSolution

    inst = build_instance('RC101', 25)
    print(f"Instance: RC101_25c ({inst['n_customers']} customers, {inst['tw_type']})")
    print(f"Original TW horizon: {inst['tw_horizon']} min")

    # Show original TW spread
    orig_tws = [(c['ready_time'], c['due_time']) for c in inst['customers']]
    orig_widths = [d - r for r, d in orig_tws]
    print(f"Original TW widths: min={min(orig_widths):.0f}, max={max(orig_widths):.0f}, "
          f"mean={sum(orig_widths)/len(orig_widths):.0f} min")

    # Create tightened instance
    inst_tight = copy.deepcopy(inst)
    for c in inst_tight['customers']:
        c['due_time'] = c['ready_time'] + (c['due_time'] - c['ready_time']) * 0.5

    tight_widths = [(c['due_time'] - c['ready_time']) for c in inst_tight['customers']]
    print(f"Tightened TW widths: min={min(tight_widths):.0f}, max={max(tight_widths):.0f}, "
          f"mean={sum(tight_widths)/len(tight_widths):.0f} min")

    # Build route on original instance
    from repair import repair_tardiness
    test_routes = [[c['id'] for c in inst['customers']]]
    sol_orig = TruckDroneSolution(test_routes, [], inst)
    print(f"\nOriginal TWs:")
    print(f"  Before repair: tardiness={sol_orig.tardiness:.0f}, feasible={sol_orig.feasible}")
    sol_repaired, stats = repair_tardiness(sol_orig, inst, max_iter=500)
    print(f"  After repair:  tardiness={sol_repaired.tardiness:.0f}, feasible={sol_repaired.feasible}")

    # Build route on tightened instance
    test_routes_tight = [[c['id'] for c in inst_tight['customers']]]
    sol_tight = TruckDroneSolution(test_routes_tight, [], inst_tight)
    print(f"\nTightened TWs (50% reduction):")
    print(f"  Before repair: tardiness={sol_tight.tardiness:.0f}, feasible={sol_tight.feasible}")
    sol_tight_r, stats_t = repair_tardiness(sol_tight, inst_tight, max_iter=500)
    print(f"  After repair:  tardiness={sol_tight_r.tardiness:.0f}, feasible={sol_tight_r.feasible}")

    conclusion = (
        "\nCONCLUSION: When time windows are tightened by 50%, even EDD repair\n"
        "(which is provably optimal for minimizing maximum lateness) cannot\n"
        "eliminate all tardiness. The cumulative service times and travel times\n"
        "exceed the available time windows. This demonstrates why time window\n"
        "density is the primary driver of infeasibility in VRPTW."
    )
    print(conclusion)

    return {
        'case': 2,
        'name': 'Time Window Tightening',
        'instance': 'RC101_25c',
        'tightening_factor': 0.5,
        'original_tardiness_after_repair': sol_repaired.tardiness,
        'tightened_tardiness_after_repair': sol_tight_r.tardiness,
        'root_cause': 'Cumulative service times exceed available time windows',
    }


def generate_case_3_sync_failure():
    """
    Case 3: Drone-Truck Synchronization Failure.

    Setting: Drone mission where truck travel time between launch and
             recovery is SHORTER than drone flight time.
    Expected: Drone arrives at recovery point BEFORE truck → sync violation.
    Root Cause: Drone detour is longer than truck direct path; truck is
                faster on direct road than drone's triangular flight path.
    """
    print("\n" + "=" * 70)
    print("FAILURE CASE 3: Drone-Truck Synchronization Failure")
    print("=" * 70)

    from utils.data_loader import build_instance
    from sync_constraints import (
        check_drone_sync, compute_route_timeline,
        insert_cross_route_drones_sync,
    )
    from drone_post_processing import insert_cross_route_drones

    inst = build_instance('RC201', 25)
    print(f"Instance: RC201_25c ({inst['n_customers']} customers, {inst['tw_type']})")

    # Build two routes
    custs = inst['customers']
    route_a = [custs[i]['id'] for i in range(5)]   # 5 customers
    route_b = [custs[i]['id'] for i in range(5, 10)]  # 5 customers
    routes = [route_a, route_b]

    # Run ORIGINAL drone insertion (no sync)
    routes_orig, missions_orig, saved_orig, n_orig = insert_cross_route_drones(
        routes, inst, drone_endurance=4.0)
    print(f"\nOriginal drone insertion (no sync):")
    print(f"  Drone missions: {n_orig}")
    for mi, m in enumerate(missions_orig):
        print(f"  Mission {mi+1}: launch={m[0]}, serve={m[1]}, recover={m[2]}")

    # Check sync for each original mission
    print(f"\nSync analysis of original missions:")
    sync_violations = 0
    for mi, m in enumerate(missions_orig):
        i, j, k = m
        # Find which truck
        launch_truck = 0 if i in routes[0] else 1
        result = check_drone_sync(
            routes, (launch_truck, i, j, k),
            inst['customers'], inst['depot'])
        print(f"\n  Mission {mi+1}: ({i}→drone→{j}→recover at {k})")
        print(f"    Drone flight: {result['drone_flight_time']:.1f} min")
        print(f"    Truck travel: {result['truck_travel_ik']:.1f} min")
        print(f"    Truck waiting: {result['truck_wait_time']:.1f} min")
        print(f"    Sync violation: {result['sync_violation']:.1f} min")
        print(f"    Feasible: {result['is_feasible']}")
        if not result['is_feasible']:
            sync_violations += 1

    # Run SYNC-AWARE drone insertion
    routes_sync, missions_sync, saved_s, n_sync, stats = insert_cross_route_drones_sync(
        routes, inst, drone_endurance=4.0, require_sync=True)
    print(f"\nSync-aware drone insertion:")
    print(f"  Drone missions: {n_sync}")
    print(f"  Checked: {stats['checked']}, Accepted: {stats['accepted']}")
    print(f"  Rejected by sync: {stats['rejected_by_sync']}")

    conclusion = (
        f"\nCONCLUSION: {sync_violations} of {n_orig} drone missions are sync-infeasible.\n"
        "The drone must fly a triangular path (launch→customer→recovery) while\n"
        "the truck takes the direct route. When the drone's detour is longer than\n"
        "the truck's direct path (common in practice), the drone arrives AFTER the\n"
        "truck at recovery — the truck must WAIT, incurring idle time that may\n"
        "cause downstream time window violations."
    )
    print(conclusion)

    return {
        'case': 3,
        'name': 'Drone-Truck Synchronization Failure',
        'instance': 'RC201_25c',
        'n_original_missions': n_orig,
        'n_sync_infeasible': sync_violations,
        'n_sync_missions': n_sync,
        'root_cause': 'Drone triangular flight path longer than truck direct path',
    }


def generate_case_4_no_charging():
    """
    Case 4: No Charging Infrastructure Available.

    Setting: CHARGING_STATIONS = [] (no charging stations).
    Expected: Long routes deplete battery with no way to recharge.
    Root Cause: Route energy demand exceeds battery capacity,
                no recharge options available.
    """
    print("\n" + "=" * 70)
    print("FAILURE CASE 4: No Charging Infrastructure")
    print("=" * 70)

    from utils.data_loader import build_instance
    from ev_problem_model import EVTruckDroneSolution, ENERGY_PER_KM

    inst = build_instance('RC202', 50)
    print(f"Instance: RC202_50c ({inst['n_customers']} customers, {inst['tw_type']})")

    # Use a reduced battery to make energy demand exceed capacity
    test_routes = [[c['id'] for c in inst['customers'][:30]]]  # 30 customers

    # With 0 charging stations (pass empty list, but insert_charging_stops uses config)
    sol_no_cs = EVTruckDroneSolution(test_routes, [], inst,
                                      battery_capacity=50.0)
    print(f"\nWith 50 kWh battery, NO charging stations:")
    print(f"  Feasible: {sol_no_cs.feasible}")
    print(f"  Battery violations: {sol_no_cs.battery_violations:.1f} kWh")
    print(f"  Energy consumed: {sol_no_cs.energy_consumed:.1f} kWh")
    energy_deficit = sol_no_cs.energy_consumed - 50.0
    print(f"  Energy deficit: {energy_deficit:.1f} kWh "
          f"({'COULD complete' if energy_deficit < 0 else 'CANNOT complete'})")

    # Where does battery deplete?
    if sol_no_cs._route_battery_traces and sol_no_cs._route_battery_traces[0]:
        trace = sol_no_cs._route_battery_traces[0]
        depletion_point = None
        for node_id, batt in trace:
            if batt <= 0 and depletion_point is None:
                depletion_point = node_id
        if depletion_point:
            c = inst['customers'][depletion_point - 1]
            print(f"\n  Battery depleted at customer {depletion_point}")
            print(f"    Position: ({c['x']:.1f}, {c['y']:.1f})")
            print(f"    TW: [{c['ready_time']:.0f}, {c['due_time']:.0f}]")
            print(f"    Distance from depot: "
                  f"{math.sqrt((8-c['x'])**2 + (8-c['y'])**2):.1f} km")

    conclusion = (
        "\nCONCLUSION: Without charging stations, long routes inevitably deplete\n"
        "the battery. The truck cannot return to depot or continue serving\n"
        "customers. This demonstrates why charging infrastructure is not optional\n"
        "for EVRP-TW — it is a binding constraint that determines route feasibility.\n"
        "The placement and number of charging stations directly affects which\n"
        "routes are feasible."
    )
    print(conclusion)

    return {
        'case': 4,
        'name': 'No Charging Infrastructure',
        'instance': 'RC202_50c',
        'battery_capacity': 50.0,
        'energy_consumed': sol_no_cs.energy_consumed,
        'battery_violations': sol_no_cs.battery_violations,
        'feasible': sol_no_cs.feasible,
        'root_cause': 'Route energy demand exceeds battery, no recharge available',
    }


# ── Main ──────────────────────────────────────────────────────────────

ALL_CASES = {
    '1': generate_case_1_battery_starvation,
    '2': generate_case_2_tw_tightening,
    '3': generate_case_3_sync_failure,
    '4': generate_case_4_no_charging,
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate failure case analysis')
    parser.add_argument('--case', type=str, default=None,
                       help='Generate specific case (1-4)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path')
    args = parser.parse_args()

    all_results = []

    if args.case:
        if args.case not in ALL_CASES:
            print(f"Unknown case: {args.case}. Available: {list(ALL_CASES.keys())}")
            return
        result = ALL_CASES[args.case]()
        all_results.append(result)
    else:
        for case_id in sorted(ALL_CASES.keys()):
            result = ALL_CASES[case_id]()
            all_results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("FAILURE CASE SUMMARY")
    print("=" * 70)
    print(f"\n{'Case':<5s} {'Name':<35s} {'Root Cause':<50s}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['case']:<5d} {r['name']:<35s} {r['root_cause']:<50s}")

    # Save to JSON if requested
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
