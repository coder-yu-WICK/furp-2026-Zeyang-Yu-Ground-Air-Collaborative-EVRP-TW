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

import math


def insert_cross_route_drones(truck_routes, instance, drone_endurance=4.0,
                               drone_speed=50.0, truck_speed=35.0,
                               drone_capacity=40.0):
    """
    Insert cross-route drone missions.

    For each customer j (on any route), check if a drone from a DIFFERENT
    truck can serve j more cheaply. If so, remove j from its original route
    and add a drone mission to the launching truck's route.

    Args:
        truck_routes: list of lists of customer IDs
        instance: problem instance dict
        drone_endurance: max drone flight distance (km)
        drone_speed, truck_speed: km/h
        drone_capacity: max drone payload

    Returns:
        (new_truck_routes, drone_missions_by_truck, total_saved, n_drones)
        drone_missions_by_truck: list of lists of (i, j, k) missions per truck
    """
    customers = instance['customers']
    depot = instance['depot']
    dist = instance['distance_matrix']

    n_trucks = len(truck_routes)
    routes = [list(r) for r in truck_routes]  # mutable copies
    drone_missions = [[] for _ in range(n_trucks)]  # per-truck drone missions
    total_saved = 0.0
    n_drones = 0

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

            # Find best drone launch from another truck
            best_saving = 0
            best_mission = None

            for launch_truck in range(n_trucks):
                if launch_truck == src_truck:
                    continue

                route = routes[launch_truck]
                for i_pos in range(len(route)):
                    i_cid = route[i_pos]
                    k_cid = route[i_pos + 1] if i_pos + 1 < len(route) else 0

                    # Feasibility: drone endurance
                    d_ij = _node_dist(i_cid, j_cid, dist, customers, depot)
                    d_jk = _node_dist(j_cid, k_cid, dist, customers, depot)
                    if d_ij + d_jk > drone_endurance:
                        continue

                    # Cost saving:
                    # Old: truck_src serves j at position src_pos in its route
                    #   truck_src cost for prev→j→next segment = 2.0*(d_pj + d_jn)
                    # New: truck_src goes prev→next directly = 2.0*d_pn
                    #   Drone: 1.0*(d_ij + d_jk)
                    # Saving = 2.0*(d_pj+d_jn) - [2.0*d_pn + 1.0*(d_ij+d_jk)]

                    prev_cid = routes[src_truck][src_pos - 1] if src_pos > 0 else 0
                    next_cid = routes[src_truck][src_pos + 1] if src_pos + 1 < len(routes[src_truck]) else 0

                    d_pj = _node_dist(prev_cid, j_cid, dist, customers, depot)
                    d_jn = _node_dist(j_cid, next_cid, dist, customers, depot)
                    d_pn = _node_dist(prev_cid, next_cid, dist, customers, depot)

                    saving = 2.0 * (d_pj + d_jn) - (2.0 * d_pn + 1.0 * (d_ij + d_jk))

                    if saving > best_saving:
                        best_saving = saving
                        best_mission = (launch_truck, i_cid, j_cid, k_cid, src_truck, src_pos)

            if best_mission is not None and best_saving > 0:
                launch_truck, i_cid, j_cid, k_cid, src_truck, src_pos = best_mission

                # Remove j from source truck route
                routes[src_truck].pop(src_pos)

                # Add drone mission to launch truck
                drone_missions[launch_truck].append((i_cid, j_cid, k_cid))

                total_saved += best_saving
                n_drones += 1
                improved = True

                # Update customer map
                del cust_to_truck[j_cid]
                # Rebuild positions for source truck
                for new_pos, cid in enumerate(routes[src_truck]):
                    cust_to_truck[cid] = (src_truck, new_pos)

    # Cleanup: filter empty routes
    new_routes = [r for r in routes if r]
    # Flatten drone missions
    all_drones = []
    for dm_list in drone_missions:
        all_drones.extend(dm_list)

    return new_routes, all_drones, total_saved, n_drones


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


def apply_drone_postprocessing(solution, instance, endurance='medium'):
    """
    Apply cross-route drone insertion to a TruckDroneSolution.

    Args:
        solution: TruckDroneSolution
        instance: problem instance dict
        endurance: 'medium' (4km) or 'high' (6km)

    Returns:
        (new_solution, cost_saved, n_drones)
    """
    from utils.problem_model import TruckDroneSolution

    end_val = 4.0 if endurance == 'medium' else 6.0

    new_routes, drone_missions, saved, n_drones = insert_cross_route_drones(
        solution.truck_routes, instance, drone_endurance=end_val)

    new_sol = TruckDroneSolution(new_routes, drone_missions, instance)
    return new_sol, saved, n_drones


# Backward compatibility
def extract_drone_candidates(clusters, instance, drone_endurance=4.0,
                              drone_capacity=40.0, max_candidates=None):
    """
    No-op for pre-routing extraction (depot-launched drones don't work for
    these instances). Returns clusters unchanged.

    Real drone insertion happens via cross_route post-processing after routing.
    """
    return clusters, []
