# -*- coding: utf-8 -*-
"""
Training instance generation and drone mission post-processing for POMO EVRP-TW.

1. Generates random EVRP-TW instances matching Solomon RC characteristics
2. Greedy drone mission insertion into truck routes (post-processing)
"""

import torch
import math
import random


# ── Training Instance Generation ──────────────────────────────────────

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


# ── 8-fold Data Augmentation ───────────────────────────────────────────

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


# ── Drone Mission Insertion (Post-processing) ────────────────────────

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


# ── Feature Extraction for POMO ───────────────────────────────────────

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


# ── Route Decoding ────────────────────────────────────────────────────

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
