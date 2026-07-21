# -*- coding: utf-8 -*-
"""
Truck-Drone Synchronization — Week 7 Gap 3.

Implements launch-recovery synchronization constraints for truck-drone
collaborative routing. Addresses the fundamental gap in the current
pipeline where drone insertion is purely distance-based and ignores
temporal feasibility.

Constraints:
  1. Drone can only launch AFTER truck arrives at launch point i
  2. Truck must arrive at recovery point k BEFORE or AT SAME TIME as drone
  3. If truck arrives before drone → truck WAITS (waiting time cost)
  4. If drone would arrive before truck → mission is SYNC-INFEASIBLE
     (unless we explicitly model truck waiting at recovery)

This supports FURP Model D: charging + synchronization.

Reference: Murray & Chu (2015) "The flying sidekick traveling salesman
problem" — the drone must be recovered at or before truck arrival.
"""

import math
import os, sys

_W6 = os.path.dirname(os.path.abspath(__file__))
_W5 = os.path.join(_W6, '..', 'week5')
_W4 = os.path.join(_W6, '..', 'week4')
_W3 = os.path.join(_W6, '..', 'week3')

for _p in [_W5, _W4, _W3]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import TRUCK_SPEED, DRONE_SPEED, DEPOT


# ── Route Timeline Computation ────────────────────────────────────────

def compute_route_timeline(route, customers, depot, truck_speed=None):
    """
    Forward-simulate a truck route to get arrival times at each node.

    Args:
        route: list of customer IDs (1-based), may include charging station IDs (>n)
        customers: list of customer dicts
        depot: (x, y) tuple
        truck_speed: km/h

    Returns:
        node_times: dict {node_id: arrival_time}
        total_time: float (makespan of route)
    """
    if truck_speed is None:
        truck_speed = TRUCK_SPEED

    node_times = {}
    current_time = 0.0
    prev_node = 0  # depot

    for cid in route:
        # Check if it's a charging station or customer
        n_cust = len(customers)
        if cid > n_cust:
            # Charging station — need to get its coordinates
            from config import CHARGING_STATIONS
            cs_idx = cid - n_cust - 1
            if 0 <= cs_idx < len(CHARGING_STATIONS):
                cs_coords = CHARGING_STATIONS[cs_idx]
            else:
                cs_coords = depot

            # Travel to charging station
            if prev_node == 0:
                d = math.sqrt((depot[0] - cs_coords[0])**2 + (depot[1] - cs_coords[1])**2)
            elif prev_node > n_cust:
                # Previous was also CS
                prev_cs_idx = prev_node - n_cust - 1
                if 0 <= prev_cs_idx < len(CHARGING_STATIONS):
                    prev_coords = CHARGING_STATIONS[prev_cs_idx]
                else:
                    prev_coords = depot
                d = math.sqrt((prev_coords[0] - cs_coords[0])**2 + (prev_coords[1] - cs_coords[1])**2)
            else:
                c_prev = customers[prev_node - 1]
                d = math.sqrt((c_prev['x'] - cs_coords[0])**2 + (c_prev['y'] - cs_coords[1])**2)

            current_time += d / truck_speed
            node_times[cid] = current_time
            # Assume some charging time (handled by EV model separately)
            prev_node = cid
            continue

        # Normal customer
        c = customers[cid - 1]

        # Travel time
        if prev_node == 0:
            d = math.sqrt((depot[0] - c['x'])**2 + (depot[1] - c['y'])**2)
        elif prev_node > n_cust:
            # Previous was charging station
            cs_idx = prev_node - n_cust - 1
            from config import CHARGING_STATIONS
            if 0 <= cs_idx < len(CHARGING_STATIONS):
                cs_coords = CHARGING_STATIONS[cs_idx]
            else:
                cs_coords = depot
            d = math.sqrt((cs_coords[0] - c['x'])**2 + (cs_coords[1] - c['y'])**2)
        else:
            c_prev = customers[prev_node - 1]
            d = math.sqrt((c_prev['x'] - c['x'])**2 + (c_prev['y'] - c['y'])**2)

        current_time += d / truck_speed

        # Wait if early
        if current_time < c['ready_time']:
            current_time = c['ready_time']

        node_times[cid] = current_time

        # Service time
        current_time += c['service_time']
        prev_node = cid

    return node_times, current_time


# ── Sync Feasibility Check ────────────────────────────────────────────

def check_drone_sync(truck_routes, drone_mission, customers, depot,
                     truck_speed=None, drone_speed=None):
    """
    Check if a drone mission (i, j, k) is temporally feasible.

    Args:
        truck_routes: list of all truck routes
        drone_mission: (launch_truck, i, j, k) where:
            launch_truck = which truck carries the drone
            i = launch node (customer ID where drone takes off)
            j = drone-served customer
            k = recovery node (customer ID where drone lands)
        customers: customer data
        depot: depot coordinates
        truck_speed, drone_speed: km/h

    Returns:
        dict with:
          - is_feasible: bool
          - drone_flight_time: float
          - truck_segment_time: float
          - truck_wait_time: float (time truck waits for drone at k)
          - sync_violation: float (0 if feasible, >0 if drone arrives after truck)
          - truck_arrival_i: float
          - drone_arrival_k: float
          - truck_arrival_k: float
    """
    if truck_speed is None:
        truck_speed = TRUCK_SPEED
    if drone_speed is None:
        drone_speed = DRONE_SPEED

    launch_truck, i, j, k = drone_mission

    # Get the launch truck's route
    route = truck_routes[launch_truck]
    n_cust = len(customers)

    # Compute timeline for this route
    node_times, _ = compute_route_timeline(route, customers, depot, truck_speed)

    # Truck arrival at launch point i
    truck_arrival_i = node_times.get(i, 0)

    # Truck arrival at recovery point k
    truck_arrival_k = node_times.get(k, float('inf'))

    # Drone flight distances
    def _dist(a, b):
        """Distance between two nodes (0=depot)."""
        if a == 0:
            return math.sqrt((depot[0] - customers[b-1]['x'])**2 + (depot[1] - customers[b-1]['y'])**2)
        if b == 0:
            return math.sqrt((depot[0] - customers[a-1]['x'])**2 + (depot[1] - customers[a-1]['y'])**2)
        ca = customers[a-1]
        cb = customers[b-1]
        return math.sqrt((ca['x'] - cb['x'])**2 + (ca['y'] - cb['y'])**2)

    d_ij = _dist(i, j)
    d_jk = _dist(j, k)

    # Drone flight time: travel + service at j
    drone_flight_time = (d_ij + d_jk) / drone_speed + customers[j-1]['service_time']

    # Truck departure from i: truck arrives at i, services i, then drone launches
    service_i = customers[i-1]['service_time'] if i > 0 else 0
    truck_depart_i = truck_arrival_i + service_i  # drone launches now

    # Truck travel time from i to k (direct path, no intermediate customers if k = i+1)
    d_ik = _dist(i, k) if k > 0 else _dist(i, 0)
    truck_travel_ik = d_ik / truck_speed

    # Truck arrival at k (from i departure + travel)
    truck_arrival_k_direct = truck_depart_i + truck_travel_ik

    # Drone arrival at k
    drone_arrival_k = truck_depart_i + drone_flight_time

    # Sync analysis:
    # - If truck arrives at k BEFORE drone: truck WAITS. Acceptable but adds idle time.
    # - If drone arrives at k BEFORE truck: SYNC VIOLATION (drone has nowhere to land).
    #   This is a hard infeasibility in the flying sidekick model.
    truck_wait_time = max(0.0, drone_arrival_k - truck_arrival_k_direct)
    sync_violation = max(0.0, truck_arrival_k_direct - drone_arrival_k)

    # Mission is feasible if drone doesn't arrive before truck (no sync violation)
    # Truck waiting is acceptable but adds cost
    is_feasible = sync_violation < 0.01  # small tolerance for floating point

    return {
        'is_feasible': is_feasible,
        'drone_flight_time': drone_flight_time,
        'truck_travel_ik': truck_travel_ik,
        'truck_wait_time': truck_wait_time,
        'sync_violation': sync_violation,
        'truck_arrival_i': truck_arrival_i,
        'truck_depart_i': truck_depart_i,
        'drone_arrival_k': drone_arrival_k,
        'truck_arrival_k': truck_arrival_k_direct,
    }


# ── Sync-aware Drone Insertion ────────────────────────────────────────

def insert_cross_route_drones_sync(truck_routes, instance, drone_endurance=4.0,
                                     drone_speed=None, truck_speed=None,
                                     drone_capacity=40.0, require_sync=True):
    """
    Cross-route drone insertion WITH synchronization constraints.

    Extends week5/drone_post_processing.insert_cross_route_drones() with
    temporal sync checks. Missions that violate synchronization are
    rejected even if distance-profitable.

    Args:
        truck_routes: list of lists of customer IDs
        instance: problem instance dict
        drone_endurance: max drone flight distance (km)
        drone_speed, truck_speed: km/h
        drone_capacity: max drone payload
        require_sync: if True, reject sync-infeasible missions

    Returns:
        (new_truck_routes, drone_missions, total_saved, n_drones, sync_stats)
        sync_stats: dict with rejected_by_sync, sync_violations_total
    """
    if drone_speed is None:
        drone_speed = DRONE_SPEED
    if truck_speed is None:
        truck_speed = TRUCK_SPEED

    customers = instance['customers']
    depot = instance['depot']
    dist = instance['distance_matrix']

    n_trucks = len(truck_routes)
    routes = [list(r) for r in truck_routes]
    drone_missions = []
    total_saved = 0.0
    n_drones = 0

    sync_stats = {
        'checked': 0,
        'rejected_by_sync': 0,
        'rejected_by_endurance': 0,
        'accepted': 0,
        'total_sync_violation': 0.0,
    }

    # Build customer → (truck_idx, position) map
    cust_to_truck = {}
    for ti, route in enumerate(routes):
        for pos, cid in enumerate(route):
            cust_to_truck[cid] = (ti, pos)

    # For each customer, try to serve by drone from another truck
    improved = True
    for _pass in range(3):
        if not improved:
            break
        improved = False

        for j_cid in list(cust_to_truck.keys()):
            src_truck, src_pos = cust_to_truck.get(j_cid, (None, None))
            if src_truck is None:
                continue

            cj = customers[j_cid - 1]
            if cj['demand'] > drone_capacity:
                continue

            best_saving = 0
            best_mission = None

            for launch_truck in range(n_trucks):
                if launch_truck == src_truck:
                    continue

                route = routes[launch_truck]
                for i_pos in range(len(route)):
                    i_cid = route[i_pos]
                    k_cid = route[i_pos + 1] if i_pos + 1 < len(route) else 0

                    # Distance feasibility
                    d_ij = _node_dist_fast(i_cid, j_cid, customers, depot)
                    d_jk = _node_dist_fast(j_cid, k_cid, customers, depot)
                    if d_ij + d_jk > drone_endurance:
                        sync_stats['rejected_by_endurance'] += 1
                        continue

                    sync_stats['checked'] += 1

                    # ── SYNC CHECK (NEW) ──
                    if require_sync:
                        sync_result = check_drone_sync(
                            routes, (launch_truck, i_cid, j_cid, k_cid),
                            customers, depot, truck_speed, drone_speed)

                        if not sync_result['is_feasible']:
                            sync_stats['rejected_by_sync'] += 1
                            sync_stats['total_sync_violation'] += sync_result['sync_violation']
                            continue

                        # Acceptable truck waiting threshold: max(30 min, 20% of drone flight)
                        max_wait = max(30.0, sync_result['drone_flight_time'] * 0.2)
                        if sync_result['truck_wait_time'] > max_wait:
                            sync_stats['rejected_by_sync'] += 1
                            continue

                    # Cost saving calculation (same as original)
                    prev_cid = routes[src_truck][src_pos - 1] if src_pos > 0 else 0
                    next_cid = routes[src_truck][src_pos + 1] if src_pos + 1 < len(routes[src_truck]) else 0

                    d_pj = _node_dist_fast(prev_cid, j_cid, customers, depot)
                    d_jn = _node_dist_fast(j_cid, next_cid, customers, depot)
                    d_pn = _node_dist_fast(prev_cid, next_cid, customers, depot)

                    saving = 2.0 * (d_pj + d_jn) - (2.0 * d_pn + 1.0 * (d_ij + d_jk))

                    if saving > best_saving:
                        best_saving = saving
                        best_mission = (launch_truck, i_cid, j_cid, k_cid, src_truck, src_pos)

            if best_mission is not None and best_saving > 0:
                launch_truck, i_cid, j_cid, k_cid, src_truck, src_pos = best_mission

                # Remove j from source truck route
                routes[src_truck].pop(src_pos)

                # Add drone mission
                drone_missions.append((i_cid, j_cid, k_cid))

                total_saved += best_saving
                n_drones += 1
                improved = True
                sync_stats['accepted'] += 1

                # Update customer map
                del cust_to_truck[j_cid]
                for new_pos, cid in enumerate(routes[src_truck]):
                    cust_to_truck[cid] = (src_truck, new_pos)

    # Cleanup
    new_routes = [r for r in routes if r]

    return new_routes, drone_missions, total_saved, n_drones, sync_stats


def _node_dist_fast(i, j, customers, depot):
    """Fast distance between two node indices."""
    if i == 0:
        return math.sqrt((depot[0] - customers[j-1]['x'])**2 +
                        (depot[1] - customers[j-1]['y'])**2)
    if j == 0:
        return math.sqrt((depot[0] - customers[i-1]['x'])**2 +
                        (depot[1] - customers[i-1]['y'])**2)
    ca = customers[i-1]
    cb = customers[j-1]
    return math.sqrt((ca['x'] - cb['x'])**2 + (ca['y'] - cb['y'])**2)


# ── Sync-aware Solution Evaluation ────────────────────────────────────

def evaluate_with_sync(solution, instance):
    """
    Re-evaluate a TruckDroneSolution with sync violation tracking.

    Augments the standard evaluation with per-mission sync checks.
    Uses check_drone_sync on each drone mission.

    Returns:
        dict with sync violations added
    """
    customers = instance['customers']
    depot = instance['depot']

    total_sync_violation = 0.0
    sync_details = []

    for mission in solution.drone_missions:
        i, j, k = mission

        # Find which truck route contains i and k
        launch_truck = None
        for ti, route in enumerate(solution.truck_routes):
            if i in route or i == 0:
                launch_truck = ti
                break

        if launch_truck is None:
            # Mission is orphaned (drone launch point not on any route)
            total_sync_violation += 10.0
            sync_details.append({'mission': mission, 'error': 'orphaned_launch'})
            continue

        result = check_drone_sync(
            solution.truck_routes, (launch_truck, i, j, k),
            customers, depot)

        if result['sync_violation'] > 0:
            total_sync_violation += result['sync_violation']
            sync_details.append({
                'mission': mission,
                'sync_violation': result['sync_violation'],
                'drone_flight_time': result['drone_flight_time'],
                'truck_arrival_k': result['truck_arrival_k'],
                'drone_arrival_k': result['drone_arrival_k'],
            })

    return {
        'total_sync_violation': total_sync_violation,
        'n_sync_violations': len(sync_details),
        'sync_details': sync_details,
        'is_sync_feasible': total_sync_violation == 0,
    }


# ── Self-Test ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== Sync Constraints Self-Test ===\n")

    from utils.data_loader import build_instance

    inst = build_instance('RC201', 50)
    print(f"Instance: {inst['name']}")

    # Build a test route
    test_routes = [
        inst['customers'][i]['id'] for i in range(10)  # First 10 customers
    ]
    test_routes = [test_routes[:5], test_routes[5:10]]  # Split into 2 routes
    print(f"Routes: {[len(r) for r in test_routes]}")

    # Compute timeline
    timeline, makespan = compute_route_timeline(
        test_routes[0], inst['customers'], inst['depot'])
    print(f"Route 0 makespan: {makespan:.1f} min")
    print(f"Node arrival times: {len(timeline)} nodes")

    # Test sync check on a hypothetical drone mission
    if len(test_routes[0]) >= 2 and len(test_routes[1]) >= 1:
        i = test_routes[0][0]  # launch from first customer of route 0
        j = test_routes[1][0]  # serve first customer of route 1
        k = test_routes[0][1]  # recover at second customer of route 0

        print(f"\nTesting drone mission: launch at {i}, serve {j}, recover at {k}")
        result = check_drone_sync(
            test_routes, (0, i, j, k),
            inst['customers'], inst['depot'])
        for key, val in result.items():
            print(f"  {key}: {val}")

    # Test sync-aware drone insertion
    print("\n--- Sync-Aware Drone Insertion ---")
    routes, missions, saved, n_drones, stats = insert_cross_route_drones_sync(
        test_routes, inst, drone_endurance=4.0, require_sync=True)
    print(f"Drone missions: {n_drones}")
    print(f"Sync stats: {stats}")
    print(f"Routes after: {[len(r) for r in routes]}")

    # Compare with original (no sync)
    from drone_post_processing import insert_cross_route_drones
    routes_orig, missions_orig, saved_orig, n_drones_orig = insert_cross_route_drones(
        test_routes, inst, drone_endurance=4.0)
    print(f"\nOriginal (no sync): {n_drones_orig} drones")
    print(f"Sync-aware: {n_drones} drones (diff: {n_drones_orig - n_drones})")

    print("\n=== Self-Test Complete ===")
