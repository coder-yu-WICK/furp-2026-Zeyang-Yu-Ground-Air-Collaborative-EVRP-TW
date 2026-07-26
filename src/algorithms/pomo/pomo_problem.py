# -*- coding: utf-8 -*-
"""
Training instance generation and drone mission post-processing for POMO EVRP-TW.

1. Generates random EVRP-TW instances matching Solomon RC characteristics
2. Greedy drone mission insertion into truck routes (post-processing)
"""

import torch
import math
import random


# -- Training Instance Generation --------------------------------------

def generate_random_problems(batch_size, problem_size, tw_type='mixed',
                            coord_range=16.0, depot=(8.0, 8.0),
                            demand_min=5.0, demand_max=40.0,
                            tw_horizon=240.0, seed=None):
    """
    Generate random EVRP-TW instances for POMO training.

    Args:
        batch_size: number of instances
        problem_size: number of customers per instance
        tw_type: 'RC1' (tight), 'RC2' (wide), or 'mixed'
        coord_range: coordinate range [0, coord_range]
        depot: depot coordinates (x, y)
        demand_min, demand_max: demand range
        tw_horizon: time window horizon
        seed: random seed

    Returns:
        List of problem dicts, each with torch tensors:
            depot_xy, node_xy, node_demand, node_tw_start, node_tw_end, node_service
    """
    if seed is not None:
        torch.manual_seed(seed)

    problems = []

    for _ in range(batch_size):
        # Random coordinates
        node_xy = torch.rand(problem_size, 2) * coord_range
        depot_xy = torch.tensor(depot).float()

        # Random demands
        node_demand = torch.rand(problem_size) * (demand_max - demand_min) + demand_min

        # Compute distances from depot for TW generation
        dist_from_depot = torch.sqrt(
            (node_xy[:, 0] - depot[0])**2 + (node_xy[:, 1] - depot[1])**2
        ) / 35.0  # approximate travel time at truck speed

        # Generate time windows
        if tw_type == 'RC1' or (tw_type == 'mixed' and random.random() < 0.5):
            # Tight time windows
            tw_width = torch.rand(problem_size) * 30 + 20  # width 20-50
        else:
            # Wide time windows
            tw_width = torch.rand(problem_size) * 120 + 60  # width 60-180

        # Center TW around estimated arrival time
        center = dist_from_depot + torch.rand(problem_size) * tw_horizon * 0.5
        node_tw_start = torch.clamp(center - tw_width / 2, min=0)
        node_tw_end = torch.clamp(center + tw_width / 2, max=tw_horizon)

        # Service time (small fixed value like Solomon)
        node_service = torch.ones(problem_size) * 10.0  # 10 time units

        problems.append({
            'depot_xy': depot_xy,
            'node_xy': node_xy,
            'node_demand': node_demand,
            'node_tw_start': node_tw_start,
            'node_tw_end': node_tw_end,
            'node_service': node_service,
        })

    return problems


# -- Solomon-Realistic Problem Generation --------------------------------

def _generate_clustered_coords(n, coord_range=16.0, depot=(8.0, 8.0), seed=None):
    """
    Generate clustered customer coordinates (Solomon C-type pattern).

    Creates 2-4 cluster centers, then distributes customers around them
    with Gaussian noise.

    Returns: (coords, cluster_ids) -- coords shape (n, 2)
    """
    import numpy as np
    rng = np.random.RandomState(seed)

    n_clusters = rng.randint(2, 5)
    # Cluster centers randomly placed, but biased away from depot center
    cluster_centers = rng.uniform(1, coord_range - 1, size=(n_clusters, 2))
    # Push clusters away from depot
    for ci in range(n_clusters):
        cx, cy = cluster_centers[ci]
        dx, dy = cx - depot[0], cy - depot[1]
        dist = np.sqrt(dx*dx + dy*dy)
        if dist < 3.0:
            # Push outward
            cluster_centers[ci, 0] = depot[0] + dx / max(dist, 0.1) * rng.uniform(3, 8)
            cluster_centers[ci, 1] = depot[1] + dy / max(dist, 0.1) * rng.uniform(3, 8)

    # Assign customers to clusters (roughly equal)
    customers_per_cluster = []
    remaining = n
    for ci in range(n_clusters - 1):
        k = rng.randint(remaining // (n_clusters - ci + 1),
                       remaining // (n_clusters - ci) + 1)
        k = max(1, min(k, remaining - (n_clusters - ci - 1)))
        customers_per_cluster.append(k)
        remaining -= k
    customers_per_cluster.append(remaining)

    coords = np.zeros((n, 2))
    cluster_ids = np.zeros(n, dtype=int)
    idx = 0
    for ci, count in enumerate(customers_per_cluster):
        cx, cy = cluster_centers[ci]
        sigma = rng.uniform(1.0, 3.0)
        for _ in range(count):
            x = cx + rng.normal(0, sigma)
            y = cy + rng.normal(0, sigma)
            coords[idx] = [np.clip(x, 0, coord_range), np.clip(y, 0, coord_range)]
            cluster_ids[idx] = ci
            idx += 1

    return coords, cluster_ids


def _generate_tw_depot_correlated(coords, depot, tw_type, horizon,
                                   truck_speed=35.0, seed=None):
    """
    Generate time windows correlated with depot distance (like real Solomon).

    Customers closer to depot get earlier TWs on average.
    RC1: tight TW (width 20-50), small offset spread
    RC2: wide TW (width 60-180), larger offset spread
    """
    import numpy as np
    import random as py_random
    rng_py = py_random.Random(seed)
    rng_np = np.random.RandomState(seed if seed is not None else 0)

    n = len(coords)
    dx = coords[:, 0] - depot[0]
    dy = coords[:, 1] - depot[1]
    dists = np.sqrt(dx*dx + dy*dy)
    est_arrival = dists / truck_speed  # hours -> scale to horizon units

    if tw_type == 'RC1' or (tw_type == 'mixed' and rng_py.random() < 0.5):
        tw_widths = rng_np.uniform(20, 50, size=n)
        # Shift arrival times so they spread within first 60% of horizon
        offset = rng_np.uniform(0, horizon * 0.3, size=n)
    else:
        tw_widths = rng_np.uniform(60, 180, size=n)
        offset = rng_np.uniform(0, horizon * 0.6, size=n)

    # Center TW around estimated arrival + random offset
    # est_arrival is in hours (km / km/h), convert to minutes to match horizon
    est_arrival_minutes = est_arrival * 60.0
    centers = est_arrival_minutes + offset
    tw_start = np.clip(centers - tw_widths / 2, 0, horizon - 10)
    tw_end = np.clip(centers + tw_widths / 2, tw_start + 10, horizon)

    return tw_start, tw_end


def generate_solomon_problems(batch_size, problem_size, tw_type='mixed',
                               pattern='mixed', coord_range=16.0,
                               depot=(8.0, 8.0), demand_min=5.0, demand_max=40.0,
                               tw_horizon=240.0, seed=None):
    """
    Generate EVRP-TW instances matching Solomon RC/C/R characteristics.

    Three spatial patterns:
      - clustered: 2-4 clusters with Gaussian spread (C-type)
      - random: uniform random (R-type)
      - random-clustered: half-and-half (RC-type)

    TWs are correlated with depot distance -- closer customers get earlier TWs.
    This matches how real Solomon instances are structured.

    Args:
        batch_size: number of instances
        problem_size: number of customers per instance
        tw_type: 'RC1' (tight), 'RC2' (wide), or 'mixed'
        pattern: 'clustered', 'random', 'random-clustered', or 'mixed'
        coord_range, depot, demand_min, demand_max, tw_horizon: as original
        seed: random seed

    Returns:
        List of problem dicts (same format as generate_random_problems)
    """
    import random as py_random
    import numpy as np

    if seed is not None:
        torch.manual_seed(seed)
        py_random.seed(seed)
        np.random.seed(seed)

    problems = []
    rng_py = py_random.Random(seed)

    for bi in range(batch_size):
        # Determine pattern for this instance
        batch_seed = seed + bi * 1000 if seed is not None else None

        if pattern == 'mixed':
            pat = rng_py.choices(
                ['clustered', 'random', 'random-clustered'],
                weights=[0.3, 0.3, 0.4])[0]
        else:
            pat = pattern

        if pat == 'clustered':
            coords_np, _ = _generate_clustered_coords(
                problem_size, coord_range, depot, seed=batch_seed)
        elif pat == 'random-clustered':
            n_half = problem_size // 2
            c1, _ = _generate_clustered_coords(
                n_half, coord_range, depot, seed=batch_seed)
            c2 = np.random.RandomState(batch_seed).uniform(
                0, coord_range, size=(problem_size - n_half, 2))
            coords_np = np.concatenate([c1, c2], axis=0)
            np.random.RandomState(batch_seed + 1).shuffle(coords_np)
        else:  # random
            coords_np = np.random.RandomState(batch_seed).uniform(
                0, coord_range, size=(problem_size, 2))

        node_xy = torch.from_numpy(coords_np).float()
        depot_xy = torch.tensor(depot).float()

        # Demand
        demands = np.random.RandomState(batch_seed + 2 if batch_seed else None).uniform(
            demand_min, demand_max, size=problem_size)
        node_demand = torch.from_numpy(demands).float()

        # TW correlated with depot distance
        tw_start_np, tw_end_np = _generate_tw_depot_correlated(
            coords_np, depot, tw_type, tw_horizon,
            seed=batch_seed + 3 if batch_seed else None)
        node_tw_start = torch.from_numpy(tw_start_np).float()
        node_tw_end = torch.from_numpy(tw_end_np).float()

        # Service time
        node_service = torch.ones(problem_size) * 10.0

        problems.append({
            'depot_xy': depot_xy,
            'node_xy': node_xy,
            'node_demand': node_demand,
            'node_tw_start': node_tw_start,
            'node_tw_end': node_tw_end,
            'node_service': node_service,
        })

    return problems


# -- 8-fold Data Augmentation -------------------------------------------

def augment_xy_by_8_fold(depot_xy, node_xy):
    """
    Create 8 symmetric views of coordinates.

    Args:
        depot_xy: (batch, 1, 2) or (batch, 2)
        node_xy: (batch, N, 2)

    Returns:
        aug_depot_xy: (8*batch, 1, 2)
        aug_node_xy: (8*batch, N, 2)
    """
    if depot_xy.dim() == 2:
        depot_xy = depot_xy.unsqueeze(1)

    batch, _, _ = depot_xy.shape
    N = node_xy.shape[1]

    # Normalize to [0, 1]
    xy_all = torch.cat([depot_xy, node_xy], dim=1)  # (batch, N+1, 2)
    max_val = xy_all.max()
    xy_norm = xy_all / max_val

    # 8 augmentations
    augs = []
    for flip_x in [False, True]:
        for flip_y in [False, True]:
            for transpose in [False, True]:
                aug = xy_norm.clone()
                if flip_x:
                    aug[:, :, 0] = 1 - aug[:, :, 0]
                if flip_y:
                    aug[:, :, 1] = 1 - aug[:, :, 1]
                if transpose:
                    aug = torch.stack([aug[:, :, 1], aug[:, :, 0]], dim=-1)
                augs.append(aug * max_val)

    aug_xy = torch.cat(augs, dim=0)  # (8*batch, N+1, 2)
    aug_depot = aug_xy[:, :1, :]
    aug_nodes = aug_xy[:, 1:, :]

    return aug_depot, aug_nodes


# -- Drone Mission Insertion (Post-processing) ------------------------

def insert_drone_missions(truck_routes, instance, drone_endurance=4.0,
                          drone_speed=50.0, truck_speed=35.0,
                          drone_capacity=40.0):
    """
    Greedy post-processing: replace truck visits with drone missions.

    For each pair of consecutive truck nodes (i, k), check if any unvisited
    customer j can be served by drone (launch from i, serve j, return to k).

    Args:
        truck_routes: list of lists of customer indices (1-based)
        instance: problem instance dict with:
            - customers: list of dicts with x, y, demand, ready_time, due_time, service_time
            - distance_matrix: (N+1) x (N+1)
            - depot: (x, y)
        drone_endurance: max drone flight distance (km)
        drone_speed: km/h
        truck_speed: km/h
        drone_capacity: max payload

    Returns:
        (new_truck_routes, drone_missions)
        drone_missions: list of (launch_node, customer, recovery_node)
    """
    customers = instance['customers']
    dist = instance['distance_matrix']
    n = len(customers)

    # Collect all truck-served customers
    truck_served = set()
    for route in truck_routes:
        for cid in route:
            truck_served.add(cid)

    drone_missions = []
    drone_served = set()

    # For each route, try to replace truck visits with drone missions
    new_routes = []
    for route in truck_routes:
        new_route = []
        for idx, cid in enumerate(route):
            if cid in drone_served:
                continue

            # Check if this customer can be served by drone instead
            # Need launch node (previous or depot) and recovery node (next or depot)
            launch = new_route[-1] if new_route else 0
            recovery = route[idx + 1] if idx + 1 < len(route) else 0

            if cid > 0 and _drone_feasible(launch, cid, recovery, customers, dist,
                                           drone_endurance, drone_capacity,
                                           drone_speed, truck_speed):
                drone_missions.append((launch, cid, recovery))
                drone_served.add(cid)
            else:
                new_route.append(cid)

        if new_route:
            new_routes.append(new_route)

    return new_routes, drone_missions


def _drone_feasible(launch, customer, recovery, customers, dist,
                    endurance, capacity, drone_speed, truck_speed):
    """Check if a drone mission is feasible."""
    n = len(customers)
    if customer < 1 or customer > n:
        return False

    c = customers[customer - 1]

    # Capacity check
    if c['demand'] > capacity:
        return False

    # Distance check
    d_launch_to_cust = (dist[launch][customer] if launch > 0 else
                        math.sqrt((8.0 - c['x'])**2 + (8.0 - c['y'])**2))
    d_cust_to_recov = (dist[customer][recovery] if recovery > 0 else
                       math.sqrt((8.0 - c['x'])**2 + (8.0 - c['y'])**2))
    drone_dist = d_launch_to_cust + d_cust_to_recov

    if drone_dist > endurance:
        return False

    return True


# -- Feature Extraction for POMO ---------------------------------------

def instance_to_pomo_features(instance):
    """
    Convert a week3 instance dict to POMO format.

    Returns:
        problem dict, depot_xy: (2,) tensor, node_features: (N, 6) tensor
    """
    customers = instance['customers']
    n = len(customers)
    depot = instance['depot']

    depot_xy = torch.tensor([depot[0], depot[1]]).float()  # (2,)

    node_features = torch.zeros(n, 6)
    for i, c in enumerate(customers):
        node_features[i, 0] = c['x']
        node_features[i, 1] = c['y']
        node_features[i, 2] = c['demand']
        node_features[i, 3] = c['ready_time']
        node_features[i, 4] = c['due_time']
        node_features[i, 5] = c['service_time']

    problem = {
        'depot_xy': depot_xy,
        'node_xy': node_features[:, :2].clone(),
        'node_demand': node_features[:, 2].clone(),
        'node_tw_start': node_features[:, 3].clone(),
        'node_tw_end': node_features[:, 4].clone(),
        'node_service': node_features[:, 5].clone(),
    }
    return problem, depot_xy, node_features


# -- Route Decoding ----------------------------------------------------

def decode_routes(selected_list, problem_size):
    """
    Decode POMO output into truck routes.

    Args:
        selected_list: list of (batch, pomo) tensors from each step
        problem_size: number of customers

    Returns:
        routes: list of (batch, pomo) lists of node sequences
    """
    if not selected_list:
        return []

    batch, pomo = selected_list[0].shape
    routes = [[[] for _ in range(pomo)] for _ in range(batch)]

    for step_tensor in selected_list:
        for b in range(batch):
            for p in range(pomo):
                node = step_tensor[b, p].item()
                routes[b][p].append(node)

    return routes
