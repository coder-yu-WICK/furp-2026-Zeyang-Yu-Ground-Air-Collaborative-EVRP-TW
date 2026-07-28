#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FURP Requirement: 3+ Systematic Failure Case Analysis.

Each case constructs a controlled scenario to demonstrate a specific
failure mode, identifies root cause, and shows constraint manifestation.

Cases:
  1. Battery Capacity Starvation — EV battery too small for route length
  2. Time Window Tightening — Scaled due_times make schedule infeasible
  3. Fleet Capacity Exceeded — Route demand > truck capacity
  4. Drone-Truck Synchronization — Drone returns before truck arrives
  5. Charging Station Necessity — Long routes require charging
"""

import sys, os, math, json, copy, random

_W7 = os.path.dirname(os.path.abspath(__file__))
for p in [os.path.join(_W7, '..', 'week3'),
          os.path.join(_W7, '..', 'week5'),
          os.path.join(_W7, '..', 'week4'),
          os.path.join(_W7, '..', 'week6'),
          _W7]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import (
    TRUCK_SPEED, DRONE_SPEED, TRUCK_CAPACITY, DRONE_CAPACITY,
    BATTERY_CAPACITY, CHARGING_RATE, DEPOT,
)
from utils.data_loader import load_instance_from_disk
from utils.problem_model import TruckDroneSolution
from run_sota_expanded import run_ours
from ev_problem_model import (
    EVTruckDroneSolution, simulate_route_ev, get_charging_station_coords,
    BATTERY_CAPACITY as EV_BATTERY,
    ENERGY_CONSUMPTION_RATE as EV_ENERGY_RATE,
)

OUTPUT_DIR = os.path.join(_W7, 'results', 'failure_cases')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def print_header(title):
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════════════
# CASE 1: Battery Capacity Starvation
# ═══════════════════════════════════════════════════════════════════════

def case1_battery_starvation():
    """
    FAILURE MODE: Battery too small to complete a long route.

    CONSTRUCTED SCENARIO: Take all 50 RC101 customers in one long route.
    With EV_ENERGY_RATE=1.5 kWh/km, a 50-customer route is ~80-120 km
    needing 120-180 kWh, exceeding both 100 kWh and 30 kWh batteries.
    """
    print_header("CASE 1: Battery Capacity Starvation")

    inst = load_instance_from_disk('RC101_50c')
    customers = inst['customers']
    dist_matrix = inst['distance_matrix']
    depot = inst.get('depot', DEPOT)
    if isinstance(depot, list):
        depot = tuple(depot)
    n_cust = len(customers)
    cs_coords = get_charging_station_coords(n_cust)

    # Build ONE long route with all customers (extreme case)
    long_route = list(range(1, n_cust + 1))

    print(f"  Constructed route: {len(long_route)} customers (all customers in 1 route)")

    for label, battery_kwh, use_cs in [
        ('100 kWh battery, no CS', 100.0, False),
        ('30 kWh battery, no CS', 30.0, False),
        ('30 kWh battery, CS mid-route', 30.0, True),
    ]:
        test_route = list(long_route)
        cs_inserted = 0
        if use_cs:
            # Insert CS at midpoint
            mid = len(test_route) // 2
            test_route.insert(mid, n_cust + 1)  # Depot CS
            cs_inserted = 1

        sim = simulate_route_ev(
            test_route, customers, dist_matrix, cs_coords, depot,
            battery_capacity=battery_kwh,
            charging_model='linear',
            energy_rate=EV_ENERGY_RATE,
        )

        status = '✅ FEASIBLE' if sim['feasible'] else '❌ INFEASIBLE'
        print(f"\n  {label}: {status}")
        print(f"    Energy consumed: {sim['total_energy']:.1f} kWh "
              f"(battery: {battery_kwh} kWh)")
        print(f"    Distance: {sim['total_dist']:.1f} km")
        print(f"    Energy violation: {sim['energy_violation']:.1f} kWh")
        print(f"    CS visits: {sim['n_charges']}")
        if cs_inserted > 0:
            print(f"    Charge energy: {sim['total_charge_energy']:.1f} kWh")

    print(f"\n  ▶ ROOT CAUSE:")
    print(f"    Route energy demand (~{EV_ENERGY_RATE} kWh/km × route_length)")
    print(f"    exceeds battery capacity. Without charging stations,")
    print(f"    the EV cannot physically complete the route.")
    print(f"    This maps to FURP Model B/C — EV battery constraint.")

    return {'status': 'completed'}


# ═══════════════════════════════════════════════════════════════════════
# CASE 2: Time Window Tightening
# ═══════════════════════════════════════════════════════════════════════

def case2_time_window_tightening():
    """
    FAILURE MODE: Tightened time windows exceed what EDD can fix.

    Strategy: Use R101 (already tight TW, avg width ~10). Scale due_times
    toward ready_time to create infeasibility that EDD cannot resolve.
    Also test: run WITHOUT repair to show EDD's contribution.
    """
    print_header("CASE 2: Time Window Tightening (R101 — already tight)")

    inst = load_instance_from_disk('R101_50c')
    customers = inst['customers']
    orig_width = sum(c['due_time'] - c['ready_time'] for c in customers) / len(customers)
    print(f"  R101_50c: avg TW width = {orig_width:.0f} (already tight)")

    for scale, use_repair in [
        (1.0, False),   # Original, no repair → shows baseline tardiness
        (1.0, True),    # Original, with repair → EDD fixes it
        (0.7, True),    # 70% width, with repair → borderline
        (0.5, True),    # 50% width, with repair → should break
        (0.3, True),    # 30% width, with repair → severely broken
    ]:
        mod_inst = copy.deepcopy(inst)
        for c in mod_inst['customers']:
            window = c['due_time'] - c['ready_time']
            c['due_time'] = c['ready_time'] + window * scale

        try:
            sol = run_ours(mod_inst, n_trucks=4, seed=42,
                          use_repair=use_repair,
                          repair_mode='full',
                          n_drones_per_truck=0)

            tw_width = sum(c['due_time']-c['ready_time']
                         for c in mod_inst['customers'])/len(mod_inst['customers'])
            repair_lbl = '+EDD' if use_repair else 'no-EDD'
            status = '✅' if sol.tardiness < 1.0 else f'❌ tard={sol.tardiness:.0f}'
            print(f"  TW@{scale*100:.0f}% ({tw_width:.0f} width) {repair_lbl}: "
                  f"{status} cost={sol.cost:.0f} feas={sol.feasible}")
        except Exception as e:
            print(f"  TW@{scale*100:.0f}%: ERROR — {e}")

    print(f"\n  ▶ ROOT CAUSE:")
    print(f"    When time windows are tightened beyond what the physical")
    print(f"    travel+service times allow, no ordering (not even EDD which")
    print(f"    is optimal for Lmax per Jackson 1955) can create feasibility.")
    print(f"    This maps to VRPTW time window constraint.")

    return {'status': 'completed'}


# ═══════════════════════════════════════════════════════════════════════
# CASE 3: Fleet Capacity Exceeded
# ═══════════════════════════════════════════════════════════════════════

def case3_fleet_capacity():
    """
    FAILURE MODE: Route load exceeds TRUCK_CAPACITY.

    CONSTRUCTED SCENARIO: Build a single route with 30+ high-demand
    customers. The total demand exceeds TRUCK_CAPACITY=200.
    """
    print_header("CASE 3: Fleet Capacity Exceeded")

    inst = load_instance_from_disk('RC101_50c')
    customers = inst['customers']
    total_demand = sum(c['demand'] for c in customers)
    print(f"  TRUCK_CAPACITY: {TRUCK_CAPACITY}, Total demand: {total_demand:.0f}")

    # Sort customers by demand (highest first) and build an overloaded route
    sorted_custs = sorted(range(1, len(customers)+1),
                         key=lambda cid: customers[cid-1]['demand'], reverse=True)

    # Build routes with increasing load until we hit capacity
    route = []
    load = 0.0
    for cid in sorted_custs:
        demand = customers[cid-1]['demand']
        if load + demand <= TRUCK_CAPACITY:
            route.append(cid)
            load += demand
        else:
            break  # Stop at capacity limit

    print(f"  Feasible route: {len(route)} customers, load={load:.0f} ≤ {TRUCK_CAPACITY}")

    # Now build an overloaded route
    overloaded_route = []
    overload = 0.0
    for cid in sorted_custs:
        demand = customers[cid-1]['demand']
        overloaded_route.append(cid)
        overload += demand
        if overload > TRUCK_CAPACITY * 1.2:  # 20% over
            break

    print(f"  Overloaded route: {len(overloaded_route)} customers, "
          f"load={overload:.0f} > {TRUCK_CAPACITY} ({(overload/TRUCK_CAPACITY-1)*100:.0f}% over)")

    # Evaluate both
    for label, r in [('Feasible route', route), ('Overloaded route', overloaded_route)]:
        sol = TruckDroneSolution([r], [], inst)
        status = '✅ FEASIBLE' if sol.feasible else '❌ INFEASIBLE'
        v_cap = sol.violations.get('capacity', 0) if sol.violations else '?'
        print(f"\n  {label}: {status}")
        print(f"    Cost: {sol.cost:.1f}, Tardiness: {sol.tardiness:.1f}")
        print(f"    Capacity violation: {v_cap}")

    print(f"\n  ▶ ROOT CAUSE:")
    print(f"    When a route's cumulative demand exceeds TRUCK_CAPACITY")
    print(f"    ({TRUCK_CAPACITY}), the solution is infeasible regardless of")
    print(f"    routing quality. This is the core VRP capacity constraint.")

    return {'status': 'completed'}


# ═══════════════════════════════════════════════════════════════════════
# CASE 4: Drone-Truck Synchronization Analysis
# ═══════════════════════════════════════════════════════════════════════

def case4_drone_sync():
    """
    FAILURE MODE: Drone-truck sync — quantify wait/hover times.

    Analysis: For each drone mission, compare truck travel time (i→k
    including service at intermediate customers) vs drone flight time
    (i→j→k). When drone is faster, it must hover and wait.
    """
    print_header("CASE 4: Drone-Truck Synchronization Analysis")

    inst = load_instance_from_disk('RC101_50c')
    customers = inst['customers']
    dist = inst['distance_matrix']

    sol = run_ours(inst, n_trucks=4, seed=42, use_repair=True,
                   repair_mode='partial', n_drones_per_truck=2)

    n_routes = len(sol.truck_routes)
    n_drones = len(sol.drone_missions)
    print(f"  Solution: {n_routes} routes, {n_drones} drone missions")

    if not sol.drone_missions:
        print("  No drone missions — skipping")
        return {'status': 'skipped'}

    sync_ok = 0
    sync_wait = 0
    total_hover = 0.0

    print(f"\n  {'Mission':<20s} {'Truck(s)':>8s} {'Drone(s)':>8s} "
          f"{'Δ(s)':>8s} {'Hover?':>8s} {'Endurance':>10s}")
    print(f"  {'─'*65}")

    for mi, mission in enumerate(sol.drone_missions):
        i, j, k = mission[0], mission[1], mission[2]

        # Find truck route
        truck_route = None
        for route in sol.truck_routes:
            if (i == 0 or i in route) and (k == 0 or k in route):
                truck_route = route
                break
        if truck_route is None:
            continue

        i_pos = truck_route.index(i) if i > 0 else -1
        k_pos = truck_route.index(k) if k > 0 else len(truck_route)

        # Truck time from i to k (including service times)
        truck_time = 0.0
        prev_node = i if i > 0 else 0
        for pos in range(i_pos + 1, k_pos + 1):
            node = truck_route[pos] if pos < len(truck_route) else 0
            if prev_node == 0:
                seg_d = math.sqrt((DEPOT[0]-customers[node-1]['x'])**2 +
                                 (DEPOT[1]-customers[node-1]['y'])**2)
            elif node == 0:
                seg_d = math.sqrt((customers[prev_node-1]['x']-DEPOT[0])**2 +
                                 (customers[prev_node-1]['y']-DEPOT[1])**2)
            else:
                seg_d = dist[prev_node][node]
            truck_time += seg_d / TRUCK_SPEED
            if 0 < node <= len(customers):
                truck_time += customers[node-1]['service_time']
            prev_node = node

        # Drone flight time
        if i == 0:
            d_ij = math.sqrt((DEPOT[0]-customers[j-1]['x'])**2 +
                            (DEPOT[1]-customers[j-1]['y'])**2)
        else:
            d_ij = dist[i][j]
        if k == 0:
            d_jk = math.sqrt((customers[j-1]['x']-DEPOT[0])**2 +
                            (customers[j-1]['y']-DEPOT[1])**2)
        else:
            d_jk = dist[j][k]
        drone_time = (d_ij + d_jk) / DRONE_SPEED + customers[j-1]['service_time']
        drone_dist = d_ij + d_jk

        delta = drone_time - truck_time
        hover = max(0, delta)
        total_hover += hover
        if delta > 0.01:
            sync_wait += 1
        else:
            sync_ok += 1

        label = f'(i={i},j={j},k={k})'
        hover_str = f'{hover:.1f}s' if hover > 0.01 else 'OK'
        print(f"  {label:<20s} {truck_time:>7.1f}s {drone_time:>7.1f}s "
              f"{delta:>+7.1f}s {hover_str:>8s} {drone_dist:>9.1f}km")

    print(f"\n  Sync OK: {sync_ok}/{sync_ok+sync_wait}, "
          f"Drone waits: {sync_wait}/{sync_ok+sync_wait}")
    print(f"  Total hover time: {total_hover:.1f}s")

    print(f"\n  ▶ ROOT CAUSE:")
    print(f"    {sync_wait}/{sync_ok+sync_wait} drone missions require the drone to")
    print(f"    hover and wait for the truck at recovery point k.")
    print(f"    This happens because drone+service is faster than truck+service")
    print(f"    when the truck has intermediate customers between i and k.")
    print(f"    In our model, this is a SOFT constraint — hovering is permitted")
    print(f"    within drone endurance limits. Maps to FURP Model D.")

    return {'sync_ok': sync_ok, 'sync_wait': sync_wait, 'total_hover': total_hover}


# ═══════════════════════════════════════════════════════════════════════
# CASE 5: Charging Station Necessity
# ═══════════════════════════════════════════════════════════════════════

def case5_charging_necessity():
    """
    FAILURE MODE: Long routes require charging stations.

    CONSTRUCTED SCENARIO: Build increasingly long routes (more customers
    per route). Show the threshold where battery capacity is exceeded.
    """
    print_header("CASE 5: Charging Station Necessity")

    inst = load_instance_from_disk('RC201_50c')  # Wide TW — can fit all customers
    customers = inst['customers']
    dist_matrix = inst['distance_matrix']
    depot = inst.get('depot', DEPOT)
    if isinstance(depot, list):
        depot = tuple(depot)
    n_cust = len(customers)
    cs_coords = get_charging_station_coords(n_cust)

    # Sort customers by nearest-neighbor from depot for a natural route
    remaining = set(range(1, n_cust + 1))
    route = []
    prev = 0
    while remaining:
        nearest = min(remaining,
                     key=lambda cid: dist_matrix[prev][cid])
        route.append(nearest)
        remaining.remove(nearest)
        prev = nearest

    print(f"  Built nearest-neighbor route: {len(route)} customers")

    # Test progressive route lengths
    for n_cust_in_route in [10, 20, 30, 40, 50]:
        test_route = route[:n_cust_in_route]
        sim_no_cs = simulate_route_ev(
            test_route, customers, dist_matrix, cs_coords, depot,
            battery_capacity=EV_BATTERY,
            energy_rate=EV_ENERGY_RATE,
        )

        # With CS: insert at midpoint
        route_with_cs = list(test_route)
        route_with_cs.insert(len(route_with_cs)//2, n_cust + 1)
        sim_cs = simulate_route_ev(
            route_with_cs, customers, dist_matrix, cs_coords, depot,
            battery_capacity=EV_BATTERY,
            energy_rate=EV_ENERGY_RATE,
        )

        s1 = '✅' if sim_no_cs['feasible'] else '❌'
        s2 = '✅' if sim_cs['feasible'] else '❌'
        print(f"  {n_cust_in_route:>2} customers: "
              f"no-CS={s1} (e={sim_no_cs['total_energy']:.0f}kWh, "
              f"vio={sim_no_cs['energy_violation']:.0f}kWh) | "
              f"with-CS={s2} (chg={sim_cs['total_charge_energy']:.0f}kWh)")

    print(f"\n  ▶ ROOT CAUSE:")
    print(f"    As route length grows beyond {EV_BATTERY/EV_ENERGY_RATE:.0f} km "
          f"(={EV_BATTERY} kWh ÷ {EV_ENERGY_RATE} kWh/km),")
    print(f"    energy demand exceeds battery capacity. Charging stations")
    print(f"    are necessary infrastructure for EV fleet operation on long routes.")
    print(f"    Maps to FURP Model B/C.")

    return {'status': 'completed'}


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("FURP FAILURE CASE ANALYSIS — 5 Systematic Cases")
    print("=" * 70)
    print(f"Output: {OUTPUT_DIR}/")
    print()

    results = {}

    for name, func in [
        ('case1_battery', case1_battery_starvation),
        ('case2_tw_tightening', case2_time_window_tightening),
        ('case3_capacity', case3_fleet_capacity),
        ('case4_sync', case4_drone_sync),
        ('case5_charging', case5_charging_necessity),
    ]:
        try:
            results[name] = func()
            print(f"\n  ✓ {name} completed")
        except Exception as e:
            print(f"\n  ✗ {name} FAILED: {e}")
            import traceback; traceback.print_exc()
            results[name] = {'error': str(e)}

    # Save summary
    out_path = os.path.join(OUTPUT_DIR, 'failure_cases_summary.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"Summary saved to: {out_path}")
    print(f"Cases completed: {sum(1 for v in results.values() if 'error' not in v)}/5")


if __name__ == '__main__':
    main()
