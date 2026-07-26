# -*- coding: utf-8 -*-
"""
Clustering-First Baselines — Week 7.

Implements alternative cluster-first route-second methods for comparison
against our hybrid approach. Each method follows the same pattern:
  1. Cluster customers
  2. Route within each cluster
  3. (Optionally) insert drones

Methods implemented:
  1. Sweep Heuristic (Gillett & Miller, 1974) — polar angle sectoring
  2. Clarke-Wright Savings (Clarke & Wright, 1964) — savings-based routing
  3. K-means + Nearest Neighbor — basic cluster-first
  4. K-means + 2-opt — cluster-first with local search
  5. K-means + POMO (no drones) — our pipeline minus drones
  6. Sweep + POMO — sweep clustering + neural routing
  7. CW-Savings + POMO — savings clustering + neural routing
"""

import math
import random
import time
import os
import sys

# Path setup for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.problem_model import TruckDroneSolution


# ═══════════════════════════════════════════════════════════════════════════
# Baseline 1: Sweep Heuristic (Gillett & Miller, 1974)
# ═══════════════════════════════════════════════════════════════════════════

def sweep_clustering(instance, n_trucks, seed=42):
    """
    Partition customers by polar angle sweep from depot.

    Classic cluster-first method: sort all customers by polar angle
    around the depot, then sweep a ray to form sectors. Each sector
    becomes one truck's cluster. Capacity-balanced by splitting
    overloaded sectors.

    Args:
        instance: problem instance dict
        n_trucks: number of clusters to produce
        seed: random seed

    Returns:
        list of clusters (each cluster is list of customer dicts)
    """
    random.seed(seed)
    customers = instance['customers']
    depot = instance['depot']

    # Compute polar angle for each customer
    angled_customers = []
    for c in customers:
        dx = c['x'] - depot[0]
        dy = c['y'] - depot[1]
        angle = math.atan2(dy, dx)
        if angle < 0:
            angle += 2 * math.pi
        angled_customers.append((angle, c))

    # Sort by angle
    angled_customers.sort(key=lambda x: x[0])

    # Total demand for balancing
    total_demand = sum(c['demand'] for _, c in angled_customers)
    target_per_truck = total_demand / n_trucks

    # Sweep to form clusters
    clusters = []
    current_cluster = []
    current_demand = 0.0

    for angle, c in angled_customers:
        if current_demand + c['demand'] > target_per_truck * 1.3 and current_cluster:
            clusters.append(current_cluster)
            current_cluster = []
            current_demand = 0.0
        current_cluster.append(c)
        current_demand += c['demand']

    if current_cluster:
        clusters.append(current_cluster)

    # Merge/split to get exactly n_trucks clusters
    while len(clusters) > n_trucks:
        # Merge smallest adjacent pair
        min_size = float('inf')
        merge_idx = 0
        for i in range(len(clusters) - 1):
            combined = len(clusters[i]) + len(clusters[i+1])
            if combined < min_size:
                min_size = combined
                merge_idx = i
        clusters[merge_idx].extend(clusters[merge_idx + 1])
        clusters.pop(merge_idx + 1)

    while len(clusters) < n_trucks and len(clusters) > 0:
        # Split largest cluster
        largest_idx = max(range(len(clusters)), key=lambda i: len(clusters[i]))
        mid = len(clusters[largest_idx]) // 2
        new_cluster = clusters[largest_idx][mid:]
        clusters[largest_idx] = clusters[largest_idx][:mid]
        clusters.insert(largest_idx + 1, new_cluster)

    return clusters


# ═══════════════════════════════════════════════════════════════════════════
# Baseline 2: Clarke-Wright Savings Algorithm
# ═══════════════════════════════════════════════════════════════════════════

def clarke_wright_savings(instance, n_trucks, seed=42):
    """
    Clarke-Wright Savings algorithm for VRPTW.

    Start with each customer as a separate route (depot -> customer -> depot).
    Compute savings s(i,j) = d(0,i) + d(0,j) - d(i,j) for merging routes.
    Merge in descending order of savings, respecting capacity and TW constraints.

    Args:
        instance: problem instance dict
        n_trucks: target number of routes (may produce fewer)
        seed: random seed

    Returns:
        TruckDroneSolution (truck-only, no drones)
    """
    random.seed(seed)
    customers = instance['customers']
    depot = instance['depot']
    dist = instance['distance_matrix']
    n = len(customers)
    truck_capacity = 200.0
    truck_speed = 35.0

    # Compute savings for all pairs
    savings = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            d_0i = math.sqrt((depot[0] - customers[i-1]['x'])**2 +
                           (depot[1] - customers[i-1]['y'])**2)
            d_0j = math.sqrt((depot[0] - customers[j-1]['x'])**2 +
                           (depot[1] - customers[j-1]['y'])**2)
            s = d_0i + d_0j - dist[i][j]
            savings.append((s, i, j))

    savings.sort(reverse=True, key=lambda x: x[0])

    # Initialize: each customer is its own route
    routes = {i: [i] for i in range(1, n + 1)}
    route_demand = {i: customers[i-1]['demand'] for i in range(1, n + 1)}
    route_endpoints = {i: (i, i) for i in range(1, n + 1)}  # (start, end)

    # Merge routes
    for s, i, j in savings:
        if i not in routes or j not in routes:
            continue
        if routes[i] is routes[j]:
            continue  # already in same route

        ri = routes[i]
        rj = routes[j]

        # Check capacity
        if route_demand[i] + route_demand[j] > truck_capacity:
            continue

        # Check if i is at an endpoint of its route and j at an endpoint
        start_i, end_i = route_endpoints[i]
        start_j, end_j = route_endpoints[j]

        can_merge = False
        if end_i == i and start_j == j:
            # ri -> rj (i at end, j at start)
            merged = ri + rj
            can_merge = True
        elif end_j == j and start_i == i:
            # rj -> ri
            merged = rj + ri
            can_merge = True
        elif end_i == i and end_j == j:
            # ri -> reverse(rj)
            merged = ri + list(reversed(rj))
            can_merge = True
        elif start_i == i and start_j == j:
            # reverse(ri) -> rj
            merged = list(reversed(ri)) + rj
            can_merge = True
        else:
            continue

        if can_merge:
            # Check TW feasibility of merged route
            if _check_route_tw_feasible(merged, customers, depot, truck_speed):
                # Merge
                new_route = merged
                new_demand = route_demand[i] + route_demand[j]
                new_start = new_route[0]
                new_end = new_route[-1]

                # Update all customers in both original routes
                for cid in ri + rj:
                    routes[cid] = new_route
                    route_demand[cid] = new_demand
                    route_endpoints[cid] = (new_start, new_end)

    # Extract unique routes
    unique_routes = []
    seen = set()
    for cid in range(1, n + 1):
        route_tuple = tuple(routes[cid])
        if route_tuple not in seen:
            seen.add(route_tuple)
            unique_routes.append(list(route_tuple))

    return TruckDroneSolution(unique_routes, [], instance)


def _check_route_tw_feasible(route, customers, depot, truck_speed=35.0):
    """Check if a route satisfies time window constraints."""
    current_time = 0.0
    prev = 0  # depot

    for cid in route:
        c = customers[cid - 1]
        if prev == 0:
            d = math.sqrt((depot[0] - c['x'])**2 + (depot[1] - c['y'])**2)
        else:
            prev_c = customers[prev - 1]
            d = math.sqrt((prev_c['x'] - c['x'])**2 + (prev_c['y'] - c['y'])**2)

        current_time += d / truck_speed
        current_time = max(current_time, c['ready_time'])

        if current_time > c['due_time']:
            return False

        current_time += c['service_time']
        prev = cid

    return True


# ═══════════════════════════════════════════════════════════════════════════
# Baseline 3: K-means + Nearest Neighbor
# ═══════════════════════════════════════════════════════════════════════════

def kmeans_nearest_neighbor(instance, n_trucks, seed=42):
    """
    Standard cluster-first route-second: K-means clustering + NN routing.

    This is the simplest possible cluster-first baseline — spatial K-means
    followed by greedy nearest-neighbor within each cluster.

    Args:
        instance: problem instance dict
        n_trucks: number of clusters/routes
        seed: random seed

    Returns:
        TruckDroneSolution
    """
    random.seed(seed)
    customers = instance['customers']
    depot = instance['depot']
    n = len(customers)

    from src.algorithms.pomo_multi_truck import cluster_customers

    clusters = cluster_customers(instance, n_trucks)

    # Within each cluster, use nearest neighbor
    truck_routes = []
    for cluster in clusters:
        if not cluster:
            continue
        route = _nearest_neighbor_route(cluster, depot)
        truck_routes.append(route)

    return TruckDroneSolution(truck_routes, [], instance)


def _nearest_neighbor_route(cluster, depot):
    """Greedy nearest-neighbor route within a cluster."""
    if not cluster:
        return []

    unvisited = [c['id'] for c in cluster]
    coords = {c['id']: (c['x'], c['y']) for c in cluster}

    route = []
    current = (depot[0], depot[1])

    while unvisited:
        # Find nearest unvisited
        best_id = min(unvisited, key=lambda cid:
            (current[0] - coords[cid][0])**2 + (current[1] - coords[cid][1])**2)
        route.append(best_id)
        current = coords[best_id]
        unvisited.remove(best_id)

    return route


# ═══════════════════════════════════════════════════════════════════════════
# Baseline 4: K-means + 2-opt
# ═══════════════════════════════════════════════════════════════════════════

def kmeans_two_opt(instance, n_trucks, seed=42, max_iter=100):
    """
    K-means clustering followed by 2-opt local search within each route.

    Args:
        instance: problem instance dict
        n_trucks: number of clusters
        seed: random seed
        max_iter: max 2-opt improvement iterations

    Returns:
        TruckDroneSolution
    """
    random.seed(seed)
    customers = instance['customers']
    depot = instance['depot']
    dist = instance['distance_matrix']

    from src.algorithms.pomo_multi_truck import cluster_customers

    clusters = cluster_customers(instance, n_trucks)

    truck_routes = []
    for cluster in clusters:
        if not cluster:
            continue
        # Start with NN
        route = _nearest_neighbor_route(cluster, depot)
        # Improve with 2-opt
        route = _two_opt_improve(route, customers, depot, dist, max_iter)
        truck_routes.append(route)

    return TruckDroneSolution(truck_routes, [], instance)


def _two_opt_improve(route, customers, depot, dist, max_iter=100):
    """2-opt local search improvement."""
    if len(route) < 3:
        return route

    best = list(route)
    best_cost = _route_distance(best, customers, depot, dist)

    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1

        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                # 2-opt swap: reverse segment i+1..j
                candidate = best[:i+1] + list(reversed(best[i+1:j+1])) + best[j+1:]
                cand_cost = _route_distance(candidate, customers, depot, dist)

                if cand_cost < best_cost:
                    best = candidate
                    best_cost = cand_cost
                    improved = True
                    break
            if improved:
                break

    return best


def _route_distance(route, customers, depot, dist):
    """Total Euclidean distance of a route."""
    if not route:
        return 0.0

    total = 0.0
    prev = 0  # depot

    for cid in route:
        c = customers[cid - 1]
        if prev == 0:
            total += math.sqrt((depot[0] - c['x'])**2 + (depot[1] - c['y'])**2)
        else:
            total += dist[prev][cid]
        prev = cid

    # Return to depot
    if prev > 0:
        c = customers[prev - 1]
        total += math.sqrt((depot[0] - c['x'])**2 + (depot[1] - c['y'])**2)

    return total


# ═══════════════════════════════════════════════════════════════════════════
# Baseline 5: Sweep + POMO
# ═══════════════════════════════════════════════════════════════════════════

def sweep_pomo(instance, n_trucks, model_path=None, seed=42):
    """
    Sweep clustering + POMO neural routing (no drones, no repair).

    This is the closest comparison point to isolate the effect of
    our hybrid clustering vs traditional sweep clustering.

    Args:
        instance: problem instance dict
        n_trucks: number of clusters
        model_path: POMO checkpoint path
        seed: random seed

    Returns:
        TruckDroneSolution
    """
    from src.pipeline.pomo_solver import ImprovedPOMOSolver

    if model_path is None:
        _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(_PROJECT_ROOT, 'algorithms', 'pomo', 'checkpoints', 'best_model.pt')
        if not os.path.exists(model_path):
            model_path = os.path.join(_PROJECT_ROOT, 'algorithms', 'pomo', 'checkpoints', 'final_model.pt')

    # Step 1: Sweep clustering
    clusters = sweep_clustering(instance, n_trucks, seed=seed)

    # Step 2: POMO routing per cluster
    solver = ImprovedPOMOSolver(model_path, device='cpu')

    truck_routes = []
    for ci, cluster in enumerate(clusters):
        if len(cluster) == 0:
            continue
        if len(cluster) == 1:
            truck_routes.append([cluster[0]['id']])
            continue

        # Build mini-instance
        mini_inst = _build_cluster_instance(instance, cluster)

        try:
            sols = solver.solve_cluster(mini_inst, seed=seed + ci)
            best = None
            best_cost = float('inf')
            for s in sols:
                if s.feasible and s.cost < best_cost:
                    best_cost = s.cost
                    best = s
            if best is None and sols:
                best = sols[0]

            if best and best.truck_routes:
                id_map = {c['id']: c['_orig_id'] for c in mini_inst['customers']}
                for route in best.truck_routes:
                    mapped = [id_map.get(cid, cid) for cid in route]
                    truck_routes.append(mapped)
        except Exception:
            # Fallback to NN
            truck_routes.append(_nearest_neighbor_route(cluster, instance['depot']))

    return TruckDroneSolution(truck_routes, [], instance)


# ═══════════════════════════════════════════════════════════════════════════
# Baseline 6: CW-Savings + POMO (re-route savings clusters with POMO)
# ═══════════════════════════════════════════════════════════════════════════

def cw_savings_pomo(instance, n_trucks, model_path=None, seed=42):
    """
    CW-Savings to form clusters, then POMO to route within each cluster.

    Here we use CW-Savings as a clustering method: run CW to get routes,
    then treat each CW route as a cluster and re-optimize with POMO.

    Args:
        instance: problem instance dict
        n_trucks: target number of trucks
        model_path: POMO checkpoint path
        seed: random seed

    Returns:
        TruckDroneSolution
    """
    from src.pipeline.pomo_solver import ImprovedPOMOSolver

    if model_path is None:
        _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(_PROJECT_ROOT, 'algorithms', 'pomo', 'checkpoints', 'best_model.pt')
        if not os.path.exists(model_path):
            model_path = os.path.join(_PROJECT_ROOT, 'algorithms', 'pomo', 'checkpoints', 'final_model.pt')

    # Step 1: CW-Savings to get initial routes
    cw_sol = clarke_wright_savings(instance, n_trucks, seed=seed)

    # Step 2: Re-route each CW route with POMO
    solver = ImprovedPOMOSolver(model_path, device='cpu')
    customers = instance['customers']

    truck_routes = []
    for route in cw_sol.truck_routes:
        if len(route) == 0:
            continue
        if len(route) == 1:
            truck_routes.append(route)
            continue

        # Build mini-instance from this route's customers
        cluster = [customers[cid - 1] for cid in route]
        mini_inst = _build_cluster_instance(instance, cluster)

        try:
            sols = solver.solve_cluster(mini_inst, seed=seed)
            best = None
            best_cost = float('inf')
            for s in sols:
                if s.feasible and s.cost < best_cost:
                    best_cost = s.cost
                    best = s
            if best is None and sols:
                best = sols[0]

            if best and best.truck_routes:
                id_map = {c['id']: c['_orig_id'] for c in mini_inst['customers']}
                for r in best.truck_routes:
                    mapped = [id_map.get(cid, cid) for cid in r]
                    truck_routes.append(mapped)
        except Exception:
            truck_routes.append(route)

    return TruckDroneSolution(truck_routes, [], instance)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _build_cluster_instance(full_instance, cluster):
    """Create a mini-instance for a single cluster."""
    import numpy as np

    full_customers = full_instance['customers']
    depot = full_instance['depot']

    cluster_ids = [c['id'] for c in cluster]
    m = len(cluster_ids)

    new_dist = np.zeros((m + 1, m + 1))
    for i, cid in enumerate(cluster_ids):
        d = math.sqrt((depot[0] - cluster[i]['x'])**2 + (depot[1] - cluster[i]['y'])**2)
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
            'id': new_id,
            'x': c['x'], 'y': c['y'],
            'demand': c['demand'],
            'ready_time': c['ready_time'],
            'due_time': c['due_time'],
            'service_time': c['service_time'],
            '_orig_id': c['id'],
        })

    return {
        'customers': new_customers,
        'depot': depot,
        'distance_matrix': new_dist.tolist(),
        'tw_type': full_instance.get('tw_type', 'RC1'),
        'tw_horizon': full_instance.get('tw_horizon', 240.0),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Baseline Registry
# ═══════════════════════════════════════════════════════════════════════════

BASELINE_METHODS = {
    'sweep_nn': {
        'name': 'Sweep + NN',
        'description': 'Polar angle sweep clustering + nearest-neighbor routing (Gillett & Miller, 1974)',
        'type': 'cluster_first',
        'has_drones': False,
        'has_repair': False,
        'function': lambda inst, n_trucks, seed: sweep_pomo_nn(inst, n_trucks, seed),
    },
    'cw_savings': {
        'name': 'CW-Savings',
        'description': 'Clarke-Wright savings algorithm with TW feasibility check (Clarke & Wright, 1964)',
        'type': 'route_first',
        'has_drones': False,
        'has_repair': False,
        'function': clarke_wright_savings,
    },
    'kmeans_nn': {
        'name': 'K-means + NN',
        'description': 'Spatial K-means clustering + nearest-neighbor routing',
        'type': 'cluster_first',
        'has_drones': False,
        'has_repair': False,
        'function': kmeans_nearest_neighbor,
    },
    'kmeans_2opt': {
        'name': 'K-means + 2-opt',
        'description': 'Spatial K-means clustering + 2-opt local search',
        'type': 'cluster_first',
        'has_drones': False,
        'has_repair': False,
        'function': kmeans_two_opt,
    },
    'sweep_pomo': {
        'name': 'Sweep + POMO',
        'description': 'Polar angle sweep clustering + POMO neural routing (no drones, no repair)',
        'type': 'cluster_first_neural',
        'has_drones': False,
        'has_repair': False,
        'function': sweep_pomo,
    },
    'cw_pomo': {
        'name': 'CW + POMO',
        'description': 'CW-Savings clustering + POMO neural routing (no drones, no repair)',
        'type': 'cluster_first_neural',
        'has_drones': False,
        'has_repair': False,
        'function': cw_savings_pomo,
    },
}


def sweep_pomo_nn(instance, n_trucks, seed=42):
    """Sweep clustering + NN routing (no POMO dependency)."""
    clusters = sweep_clustering(instance, n_trucks, seed=seed)
    depot = instance['depot']
    truck_routes = []
    for cluster in clusters:
        if cluster:
            truck_routes.append(_nearest_neighbor_route(cluster, depot))
    return TruckDroneSolution(truck_routes, [], instance)


def get_baseline_names():
    """Return list of baseline method keys."""
    return list(BASELINE_METHODS.keys())


def get_baseline_info(key):
    """Return info dict for a baseline method."""
    return BASELINE_METHODS.get(key, {})


# ═══════════════════════════════════════════════════════════════════════════
# Ablation: All clustering-first variants for comparison table
# ═══════════════════════════════════════════════════════════════════════════

def run_all_clustering_baselines(instance, n_trucks, seed=42,
                                  model_path=None, include_pomo=True):
    """
    Run all clustering-first baselines on a single instance.

    Returns:
        dict mapping method_key -> (solution, runtime)
    """
    results = {}

    # Non-POMO baselines (fast)
    for key in ['sweep_nn', 'cw_savings', 'kmeans_nn', 'kmeans_2opt']:
        t0 = time.time()
        try:
            sol = BASELINE_METHODS[key]['function'](instance, n_trucks, seed)
            results[key] = {'solution': sol, 'runtime': time.time() - t0}
        except Exception as e:
            results[key] = {'solution': None, 'runtime': time.time() - t0, 'error': str(e)}

    # POMO-based baselines (slower, requires model)
    if include_pomo:
        for key in ['sweep_pomo', 'cw_pomo']:
            t0 = time.time()
            try:
                sol = BASELINE_METHODS[key]['function'](instance, n_trucks, model_path, seed)
                results[key] = {'solution': sol, 'runtime': time.time() - t0}
            except Exception as e:
                results[key] = {'solution': None, 'runtime': time.time() - t0, 'error': str(e)}

    return results
