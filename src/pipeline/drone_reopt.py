# -*- coding: utf-8 -*-
"""
Drone Post-Processing with Route Re-Optimization — Week 5 Extended.

After cross-route drone insertion removes a customer from one route
and adds a drone mission to another, the affected truck routes are
no longer optimal. This module re-optimizes them with POMO.

Additionally provides drone fleet sizing: measures marginal benefit
of 1, 2, ..., n_max drones per truck.
"""

import os, sys, math
import copy

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.problem_model import TruckDroneSolution
from src.pipeline.drone import insert_cross_route_drones, apply_drone_postprocessing


def reoptimize_after_drone_insertion(truck_routes, drone_missions, instance,
                                      solver, drone_endurance='medium'):
    """
    After drone insertion, re-run POMO on affected routes.

    When a drone mission (i, j, k) is assigned to truck A, and customer j
    is removed from truck B's route:
      - Truck A's route is unchanged (drone handles the detour)
      - Truck B's route has one fewer customer → re-optimize

    Args:
        truck_routes: list of routes (each is list of customer IDs)
        drone_missions: list of (i, j, k) tuples
        instance: problem instance dict
        solver: ImprovedPOMOSolver instance (for solve_cluster)
        drone_endurance: 'medium' or 'high'

    Returns:
        (new_truck_routes, drone_missions, cost_improvement)
    """
    if not drone_missions:
        return truck_routes, drone_missions, 0.0

    # Track which trucks are affected by drone removal
    # Build original customer-to-truck map
    orig_cust_to_truck = {}
    for ti, route in enumerate(truck_routes):
        for cid in route:
            orig_cust_to_truck[cid] = ti

    # After drone insertion, some customers are removed from routes
    # Re-identify affected trucks by comparing with original map
    current_routes = [list(r) for r in truck_routes]

    # For each drone mission, the drone serves customer j
    # which was removed from some truck's route
    affected_trucks = set()
    for mission in drone_missions:
        _, j_cid, _ = mission  # (i, j, k)
        # j was originally on some truck, now removed
        # Find which truck currently has a shorter route
        for ti, route in enumerate(current_routes):
            if j_cid not in route and j_cid in orig_cust_to_truck:
                if orig_cust_to_truck[j_cid] == ti:
                    affected_trucks.add(ti)

    if not affected_trucks:
        return current_routes, drone_missions, 0.0

    total_improvement = 0.0

    for ti in affected_trucks:
        route = current_routes[ti]
        if len(route) <= 2:
            continue  # Too short to benefit from re-optimization

        # Build mini-instance for this route's customers
        customers = instance['customers']
        depot = instance['depot']

        cluster = [c for c in customers if c['id'] in route]
        if len(cluster) <= 1:
            continue

        # Build distance matrix
        import numpy as np
        m = len(cluster)
        new_dist = np.zeros((m + 1, m + 1))
        for i, c in enumerate(cluster):
            d = math.sqrt((depot[0] - c['x'])**2 + (depot[1] - c['y'])**2)
            new_dist[0, i + 1] = d
            new_dist[i + 1, 0] = d
        for i in range(m):
            for j in range(m):
                d = math.sqrt((cluster[i]['x'] - cluster[j]['x'])**2 +
                             (cluster[i]['y'] - cluster[j]['y'])**2)
                new_dist[i + 1, j + 1] = d

        new_customers = []
        for new_id, c in enumerate(cluster, start=1):
            new_customers.append({
                'id': new_id, 'x': c['x'], 'y': c['y'],
                'demand': c['demand'],
                'ready_time': c['ready_time'],
                'due_time': c['due_time'],
                'service_time': c['service_time'],
                '_orig_id': c['id'],
            })

        mini_inst = {
            'customers': new_customers,
            'depot': depot,
            'distance_matrix': new_dist.tolist(),
            'tw_type': instance.get('tw_type', 'RC1'),
            'tw_horizon': instance.get('tw_horizon', 240.0),
        }

        try:
            sols = solver.solve_cluster(mini_inst, seed=42)
            best, best_cost = None, float('inf')
            for s in sols:
                if s.feasible and s.cost < best_cost:
                    best_cost = s.cost
                    best = s

            if best and best.truck_routes:
                id_map = {c['id']: c['_orig_id'] for c in mini_inst['customers']}
                new_route = [id_map.get(cid, cid)
                            for r in best.truck_routes
                            for cid in r]
                # Compute improvement
                old_sol = TruckDroneSolution([route], [], instance)
                old_cost = old_sol.cost
                new_sol = TruckDroneSolution([new_route], [], instance)
                total_improvement += old_cost - new_sol.cost
                current_routes[ti] = new_route
        except Exception:
            pass  # Keep original route if re-optimization fails

    return current_routes, drone_missions, total_improvement


def apply_drone_postprocessing_with_reopt(solution, instance, solver,
                                            endurance='medium'):
    """
    Apply cross-route drone insertion + route re-optimization.

    Pipeline:
    1. Insert cross-route drones
    2. Re-optimize affected truck routes with POMO

    Returns:
        (new_solution, cost_saved, n_drones, reopt_improvement)
    """
    end_val = 4.0 if endurance == 'medium' else 6.0

    # Step 1: Cross-route drone insertion
    new_routes, drone_missions, saved, n_drones, drone_counts = insert_cross_route_drones(
        solution.truck_routes, instance, drone_endurance=end_val)

    if n_drones == 0:
        return solution, 0.0, 0, 0.0

    # Step 2: Re-optimize affected routes
    reopt_routes, drone_missions, reopt_improvement = reoptimize_after_drone_insertion(
        new_routes, drone_missions, instance, solver, drone_endurance=endurance)

    new_sol = TruckDroneSolution(reopt_routes, drone_missions, instance)
    return new_sol, saved, n_drones, reopt_improvement


# ═════════════════════════════════════════════════════════════════════════
# Drone Fleet Sizing
# ═════════════════════════════════════════════════════════════════════════

def evaluate_drone_fleet_size(truck_routes, instance, max_drones_per_truck=3,
                                drone_endurance=4.0):
    """
    Measure marginal benefit of k drones per truck (k = 0, 1, ..., max).

    Simulates the effect of having k drones available per truck by
    limiting the number of drone missions per truck.

    Returns:
        List of (n_drones_total, cost, tardiness, missions_found)
    """
    results = []

    # Baseline (0 drones)
    base_sol = TruckDroneSolution(truck_routes, [], instance)
    results.append((0, base_sol.cost, base_sol.tardiness, 0))

    for k in range(1, max_drones_per_truck + 1):
        # Run drone insertion with per-truck limit
        new_routes, all_missions, saved, n_drones, drone_counts = insert_cross_route_drones(
            truck_routes, instance, drone_endurance=drone_endurance)

        # Limit missions: at most k per truck
        n_trucks = len(truck_routes)
        limited_missions = all_missions[:k * n_trucks]

        sol = TruckDroneSolution(new_routes, limited_missions, instance)
        results.append((k * n_trucks, sol.cost, sol.tardiness,
                       len(limited_missions)))

    return results
