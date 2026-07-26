# -*- coding: utf-8 -*-
"""
Drone Integration for POMO-MT — Week 5.

Strategy: Cross-route truck-launched drone missions.

Depot-launched drones cannot reach any customers in the Solomon instances
(min round-trip distance from central depot = 11.5km > 4km endurance).
Drones must launch from trucks in the field.

This module implements inter-route drone insertion: for each customer j on
route B, check if a drone launched from truck A (at stop i) can serve j
and return to truck A (at stop k). If feasible and cost-saving, j is
removed from route B and served by drone from route A.

This is an "improvement exchange" operator — truck B takes a more direct
path, while truck A's drone handles the detour.
"""

import os, sys, math

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.problem_model import TruckDroneSolution


def insert_cross_route_drones(truck_routes, instance, drone_endurance=4.0,
                               drone_speed=50.0, truck_speed=35.0,
                               drone_capacity=40.0, max_drones_per_truck=2,
                               min_saving=0.5):
    """
    Insert cross-route drone missions with per-truck drone limits.

    For each customer j (on any route), check if a drone from a DIFFERENT
    truck can serve j more cheaply. If so, remove j from its original route
    and add a drone mission to the launching truck's route.

    Supports up to max_drones_per_truck drones per truck (default 2).
    Missions are assigned greedily by highest cost saving first.

    Drone mission format: (i, j, k, drone_id) 4-tuple where drone_id ∈ {0, 1}.
    Backward compatible: callers expecting 3-tuples can slice [:3].

    Args:
        truck_routes: list of lists of customer IDs
        instance: problem instance dict
        drone_endurance: max drone flight distance (km)
        drone_speed, truck_speed: km/h
        drone_capacity: max drone payload
        max_drones_per_truck: max drone missions per truck (default 2)
        min_saving: minimum cost saving to accept a mission (km equivalent)

    Returns:
        (new_truck_routes, drone_missions, total_saved, n_drones, drone_counts)
        drone_missions: flat list of (i, j, k, drone_id) 4-tuples
        drone_counts: per-truck drone mission counts
    """
    customers = instance['customers']
    depot = instance['depot']
    dist = instance['distance_matrix']

    n_trucks = len(truck_routes)
    routes = [list(r) for r in truck_routes]  # mutable copies
    drone_missions = [[] for _ in range(n_trucks)]  # per-truck drone missions
    drone_counts = [0] * n_trucks
    total_saved = 0.0
    n_drones = 0

    # Build customer → (truck_idx, position) map
    cust_to_truck = {}
    for ti, route in enumerate(routes):
        for pos, cid in enumerate(route):
            cust_to_truck[cid] = (ti, pos)

    # Track drone-served customers
    drone_served = set()

    # Multi-pass: collect all candidates, sort by saving, assign greedily
    for _pass in range(3):
        # Collect all feasible candidate missions
        candidates = []

        for j_cid in cust_to_truck:
            if j_cid in drone_served:
                continue
            src_truck, src_pos = cust_to_truck[j_cid]
            cj = customers[j_cid - 1]
            if cj['demand'] > drone_capacity:
                continue

            # Find best drone launch from another truck
            best_local_saving = 0
            best_local_mission = None

            for launch_truck in range(n_trucks):
                if launch_truck == src_truck:
                    continue
                if drone_counts[launch_truck] >= max_drones_per_truck:
                    continue

                route = routes[launch_truck]
                for i_pos in range(len(route)):
                    i_cid = route[i_pos]
                    k_cid = route[i_pos + 1] if i_pos + 1 < len(route) else 0

                    # Launch/recovery nodes must NOT be drone-served
                    if i_cid > 0 and i_cid in drone_served:
                        continue
                    if k_cid > 0 and k_cid in drone_served:
                        continue

                    # Feasibility: drone endurance
                    d_ij = _node_dist(i_cid, j_cid, dist, customers, depot)
                    d_jk = _node_dist(j_cid, k_cid, dist, customers, depot)
                    if d_ij + d_jk > drone_endurance:
                        continue

                    # Feasibility: sync — drone must land before truck departs k
                    # Compute actual truck travel time from i to k (including
                    # intermediate customers' service & waiting time)
                    truck_segment_time = _compute_truck_segment_time(
                        route, i_pos, k_cid, customers, dist, depot, truck_speed)
                    drone_flight_time = (d_ij + d_jk) / drone_speed
                    drone_mission_time = drone_flight_time + cj['service_time']

                    # Drone must arrive before truck departs recovery node.
                    # Truck departure = arrival + service at k. We allow the drone
                    # to land while truck is servicing k (grace period = service_time_k).
                    service_k = customers[k_cid-1]['service_time'] if k_cid > 0 else 30.0
                    if drone_mission_time > truck_segment_time + service_k:
                        continue  # drone would miss the recovery window

                    # Cost saving:
                    # Old: truck_src serves j → 2.0*(d_pj + d_jn)
                    # New: truck_src direct → 2.0*d_pn + drone 1.0*(d_ij+d_jk)
                    # Saving = 2.0*(d_pj+d_jn) - [2.0*d_pn + 1.0*(d_ij+d_jk)]

                    prev_cid = routes[src_truck][src_pos - 1] if src_pos > 0 else 0
                    next_cid = routes[src_truck][src_pos + 1] if src_pos + 1 < len(routes[src_truck]) else 0

                    d_pj = _node_dist(prev_cid, j_cid, dist, customers, depot)
                    d_jn = _node_dist(j_cid, next_cid, dist, customers, depot)
                    d_pn = _node_dist(prev_cid, next_cid, dist, customers, depot)

                    saving = 2.0 * (d_pj + d_jn) - (2.0 * d_pn + 1.0 * (d_ij + d_jk))

                    if saving > best_local_saving:
                        best_local_saving = saving
                        best_local_mission = (launch_truck, i_cid, j_cid, k_cid,
                                             src_truck, src_pos)

            if best_local_mission is not None and best_local_saving > min_saving:
                candidates.append((best_local_saving, best_local_mission, j_cid))

        if not candidates:
            break

        # Sort by saving (highest first) and apply greedily
        candidates.sort(key=lambda x: x[0], reverse=True)

        for saving, mission, j_cid in candidates:
            if j_cid in drone_served:
                continue

            launch_truck, i_cid, j_cid2, k_cid, src_truck, src_pos = mission

            # Re-check drone count
            if drone_counts[launch_truck] >= max_drones_per_truck:
                continue

            # Re-check: launch and recovery nodes must not be drone-served
            if i_cid > 0 and i_cid in drone_served:
                continue
            if k_cid > 0 and k_cid in drone_served:
                continue
            # Verify i_cid is still present in the launch truck's route
            if i_cid > 0 and i_cid not in routes[launch_truck]:
                continue
            if k_cid > 0 and k_cid not in routes[launch_truck]:
                continue

            # Re-check source position validity
            if j_cid not in cust_to_truck:
                continue
            curr_src_truck, curr_src_pos = cust_to_truck[j_cid]
            if curr_src_truck != src_truck:
                continue

            # Assign drone_id (0 or 1) based on current count for this truck
            drone_id = drone_counts[launch_truck]  # 0 for first, 1 for second

            # Apply the mission
            routes[src_truck].pop(curr_src_pos)
            drone_missions[launch_truck].append((i_cid, j_cid, k_cid, drone_id))
            drone_counts[launch_truck] += 1
            total_saved += saving
            n_drones += 1
            drone_served.add(j_cid)

            # Update customer map for source truck
            del cust_to_truck[j_cid]
            for new_pos, cid in enumerate(routes[src_truck]):
                cust_to_truck[cid] = (src_truck, new_pos)

        # ── Post-pass conflict resolution ──
        # Detect and fix missions where launch/recovery node is drone-served.
        # This happens when candidates are applied in saving order: a high-saving
        # mission uses node X as recovery, then a lower-saving mission serves X
        # by drone. We must undo the drone service and put X back in a route.
        max_conflict_iters = 10
        for _ in range(max_conflict_iters):
            fixed_any = False
            for ti in range(n_trucks):
                for mi, mission in enumerate(drone_missions[ti]):
                    i_cid, j_cid, k_cid = mission[0], mission[1], mission[2]
                    fixed_this = False
                    # Check launch node i
                    if i_cid > 0 and i_cid in drone_served:
                        drone_served.discard(i_cid)
                        # Remove the drone mission that serves i_cid
                        for t2 in range(n_trucks):
                            drone_missions[t2] = [m for m in drone_missions[t2] if m[1] != i_cid]
                            # Reduce drone count for that truck
                            old_count = drone_counts[t2]
                            new_count = len(drone_missions[t2])
                            if new_count < old_count:
                                drone_counts[t2] = new_count
                                n_drones -= (old_count - new_count)
                        # Put i_cid back in a route
                        routes[ti].append(i_cid)
                        cust_to_truck[i_cid] = (ti, len(routes[ti]) - 1)
                        fixed_this = True
                    # Check recovery node k
                    if k_cid > 0 and k_cid in drone_served:
                        drone_served.discard(k_cid)
                        for t2 in range(n_trucks):
                            drone_missions[t2] = [m for m in drone_missions[t2] if m[1] != k_cid]
                            old_count = drone_counts[t2]
                            new_count = len(drone_missions[t2])
                            if new_count < old_count:
                                drone_counts[t2] = new_count
                                n_drones -= (old_count - new_count)
                        routes[ti].append(k_cid)
                        cust_to_truck[k_cid] = (ti, len(routes[ti]) - 1)
                        fixed_this = True
                    if fixed_this:
                        fixed_any = True
                        break
                if fixed_any:
                    break
            if not fixed_any:
                break

    # Cleanup: filter empty routes
    new_routes = [r for r in routes if r]
    # Flatten drone missions (preserving 4-tuple format)
    all_drones = []
    for dm_list in drone_missions:
        all_drones.extend(dm_list)

    return new_routes, all_drones, total_saved, n_drones, drone_counts


def _compute_truck_segment_time(route, i_pos, k_cid, customers, dist_matrix,
                                depot, truck_speed=35.0):
    """
    Compute actual truck travel time from position i_pos to recovery node k_cid.

    Includes: travel distances + service times + waiting at intermediate customers.
    This is the actual time the truck needs to go from launching the drone at i
    to arriving at recovery node k (ready to pick up the drone).

    Args:
        route: truck route (list of customer IDs)
        i_pos: launch position in route (index)
        k_cid: recovery customer ID (0 = depot)
        customers, dist_matrix, depot: instance data
        truck_speed: km/h

    Returns:
        total_time: truck travel + service + wait time from i to k
    """
    total_time = 0.0
    current_time = 0.0

    # Start from position i_pos (launch node). The truck is at this node,
    # has already arrived and serviced it. We need travel from i to next nodes.
    prev_cid = route[i_pos]

    # Determine end position for recovery node k_cid
    if k_cid == 0:
        # Recovery at depot — iterate to end of route then add return to depot
        end_pos = len(route)
    else:
        # Find k_cid position in route
        try:
            end_pos = route.index(k_cid, i_pos + 1)
        except ValueError:
            # k not found after i in route — use direct distance
            d_ik = _node_dist(prev_cid, k_cid, dist_matrix, customers, depot)
            return d_ik / truck_speed

    for pos in range(i_pos + 1, end_pos + 1):
        if pos < len(route):
            curr_cid = route[pos]
        else:
            curr_cid = 0  # depot

        # Travel from prev to curr
        d = _node_dist(prev_cid, curr_cid, dist_matrix, customers, depot)
        current_time += d / truck_speed

        if curr_cid == 0:
            break  # arrived at depot

        # Time window handling at curr
        c = customers[curr_cid - 1]
        if current_time < c['ready_time']:
            current_time = c['ready_time']  # wait
        current_time += c['service_time']

        # Stop at recovery node (before servicing k — drone lands on arrival)
        if curr_cid == k_cid:
            # We've reached k. Drone can land while truck services k.
            # Return time UP TO arrival at k (service time is grace period)
            # Subtract service time since we added it above
            current_time -= c['service_time']
            break

        prev_cid = curr_cid

    return current_time


def _node_dist(i, j, dist_matrix, customers, depot):
    """Distance between two node indices (0=depot)."""
    if i == 0 and j == 0:
        return 0.0
    if i == 0:
        return math.sqrt((depot[0] - customers[j-1]['x'])**2 +
                        (depot[1] - customers[j-1]['y'])**2)
    if j == 0:
        return math.sqrt((depot[0] - customers[i-1]['x'])**2 +
                        (depot[1] - customers[i-1]['y'])**2)
    return dist_matrix[i][j]


def apply_drone_postprocessing(solution, instance, endurance='medium',
                               max_drones_per_truck=2, min_saving=0.5):
    """
    Apply cross-route drone insertion to a TruckDroneSolution.

    Args:
        solution: TruckDroneSolution
        instance: problem instance dict
        endurance: 'medium' (4km) or 'high' (6km)
        max_drones_per_truck: max drones per truck (0=skip, 1=single, 2=dual)
        min_saving: minimum cost saving to accept a mission

    Returns:
        (new_solution, cost_saved, n_drones, drone_counts)
    """
    if max_drones_per_truck == 0:
        return solution, 0.0, 0, [0]

    end_val = 4.0 if endurance == 'medium' else 6.0

    new_routes, drone_missions, saved, n_drones, drone_counts = insert_cross_route_drones(
        solution.truck_routes, instance,
        drone_endurance=end_val,
        max_drones_per_truck=max_drones_per_truck,
        min_saving=min_saving)

    new_sol = TruckDroneSolution(new_routes, drone_missions, instance,
                                  max_drones_per_truck=max_drones_per_truck)
    return new_sol, saved, n_drones, drone_counts


# Backward compatibility
def extract_drone_candidates(clusters, instance, drone_endurance=4.0,
                              drone_capacity=40.0, max_candidates=None):
    """
    No-op for pre-routing extraction (depot-launched drones don't work for
    these instances). Returns clusters unchanged.

    Real drone insertion happens via cross_route post-processing after routing.
    """
    return clusters, []
