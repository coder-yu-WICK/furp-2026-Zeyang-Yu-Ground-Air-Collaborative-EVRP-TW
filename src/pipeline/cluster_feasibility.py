# -*- coding: utf-8 -*-
"""
Temporal Feasibility-Aware Clustering — Week 6 Task A.

Ensures clusters are temporally feasible BEFORE POMO routing.
Addresses the "clustering-TW contradiction": spatial K-means ignores
time windows, and the temporal split threshold (0.4 × horizon) is too
coarse for large instances. Some clusters end up TW-infeasible — no
single-truck ordering can serve all assigned customers on time.

This module:
1. Checks each cluster for temporal feasibility (EDD as lower bound)
2. Splits infeasible clusters at natural TW gaps
3. Balances capacity after splits
"""

import os, sys, math
import numpy as np

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import TRUCK_CAPACITY


def check_cluster_tw_feasibility(cluster, instance):
    """
    Quick check: can a single truck serve all customers in this cluster
    on time? Uses EDD ordering as a lower-bound heuristic.

    Returns:
        (is_feasible, estimated_tardiness, temporal_spread_min)
    """
    if len(cluster) <= 1:
        return True, 0.0, 0.0

    customers = instance['customers']
    depot = instance['depot']
    truck_speed = 35.0

    # Sort by due_time (EDD) — optimal for minimizing max tardiness
    sorted_cluster = sorted(cluster, key=lambda c: c['due_time'])

    # Forward-simulate the EDD route
    current_time = 0.0
    total_tard = 0.0
    prev_x, prev_y = depot[0], depot[1]

    for c in sorted_cluster:
        # Travel from previous position
        dx = prev_x - c['x']
        dy = prev_y - c['y']
        travel = math.sqrt(dx*dx + dy*dy) / truck_speed

        current_time = max(current_time + travel, c['ready_time'])
        current_time += c['service_time']

        if current_time > c['due_time']:
            total_tard += current_time - c['due_time']

        prev_x, prev_y = c['x'], c['y']

    # Compute temporal spread
    midpoints = [(c['ready_time'] + c['due_time']) / 2 for c in cluster]
    tw_spread = max(midpoints) - min(midpoints)

    return total_tard == 0, total_tard, tw_spread


def _simulate_edd_route(cluster, depot, truck_speed=35.0):
    """
    Simulate EDD-ordered route and return total tardiness.

    Used internally by the split search — lighter than check_cluster_tw_feasibility
    since we already have the customer dicts sorted.
    """
    if len(cluster) <= 1:
        return 0.0

    sorted_cluster = sorted(cluster, key=lambda c: c['due_time'])

    current_time = 0.0
    total_tard = 0.0
    prev_x, prev_y = depot[0], depot[1]

    for c in sorted_cluster:
        dx = prev_x - c['x']
        dy = prev_y - c['y']
        travel = math.sqrt(dx*dx + dy*dy) / truck_speed

        current_time = max(current_time + travel, c['ready_time'])
        current_time += c['service_time']

        if current_time > c['due_time']:
            total_tard += current_time - c['due_time']

        prev_x, prev_y = c['x'], c['y']

    return total_tard


def split_temporally_infeasible_cluster(cluster, instance, max_sub_clusters=3):
    """
    Split a TW-infeasible cluster into temporally-compatible sub-clusters.

    Algorithm:
    1. Sort by TW midpoint (temporal order)
    2. For k = 2, 3, ..., max_sub_clusters:
       a. Try all possible split points (greedy for k>2)
       b. Evaluate: max EDD tardiness across sub-clusters
       c. Keep split with lowest worst-case tardiness
    3. Return the best split (minimum max sub-cluster tardiness)

    Args:
        cluster: list of customer dicts
        instance: problem instance dict
        max_sub_clusters: max number of sub-clusters to try

    Returns:
        list of sub-clusters (each is list of customer dicts)
    """
    if len(cluster) <= 2:
        return [cluster]

    depot = instance['depot']

    # Sort by TW midpoint — this is the temporal order
    sorted_cluster = sorted(cluster,
        key=lambda c: (c['ready_time'] + c['due_time']) / 2)
    n = len(sorted_cluster)

    best_split = [sorted_cluster]  # default: no split
    best_max_tard = _simulate_edd_route(sorted_cluster, depot)
    if best_max_tard == 0:
        return best_split

    # Try splitting into k sub-clusters
    for k in range(2, min(max_sub_clusters + 1, n)):
        best_k_split = None
        best_k_tard = float('inf')

        if k == 2:
            # Try all single-split points
            for split_at in range(1, n):
                sub1 = sorted_cluster[:split_at]
                sub2 = sorted_cluster[split_at:]
                tard1 = _simulate_edd_route(sub1, depot)
                tard2 = _simulate_edd_route(sub2, depot)
                max_tard = max(tard1, tard2)
                if max_tard < best_k_tard:
                    best_k_tard = max_tard
                    best_k_split = [sub1, sub2]
        else:
            # Greedy: find best first split, then recursively split worse sub-cluster
            for split_at in range(1, n - (k - 1) + 1):
                sub1 = sorted_cluster[:split_at]
                rest = sorted_cluster[split_at:]
                tard1 = _simulate_edd_route(sub1, depot)

                # Recursively split rest into k-1 sub-clusters
                # (simplified: try equal-ish splits for the rest)
                rest_n = len(rest)
                sub_splits = []
                sub_size = rest_n // (k - 1)
                for ki in range(k - 1):
                    start = ki * sub_size
                    end = rest_n if ki == k - 2 else (ki + 1) * sub_size
                    sub_splits.append(rest[start:end])

                tards = [tard1] + [_simulate_edd_route(s, depot) for s in sub_splits]
                max_tard = max(tards)
                if max_tard < best_k_tard:
                    best_k_tard = max_tard
                    best_k_split = [sub1] + sub_splits

        if best_k_split and best_k_tard < best_max_tard:
            best_max_tard = best_k_tard
            best_split = best_k_split

        # Early exit: found a zero-tardiness split
        if best_max_tard == 0:
            break

    # Filter empty sub-clusters
    result = [s for s in best_split if s]

    if len(result) <= 1:
        return [sorted_cluster]

    return result


def ensure_temporal_feasibility(clusters, instance, max_iter=3):
    """
    Main entry point. Ensures all clusters are temporally feasible.

    For each cluster:
    1. Check TW feasibility via EDD lower bound
    2. If infeasible, split into temporally-compatible sub-clusters
    3. Repeat up to max_iter times (splitting can cascade)

    After splitting, applies capacity balancing and filters empty clusters.

    Args:
        clusters: list of clusters (each is list of customer dicts)
        instance: problem instance dict
        max_iter: max splitting iterations per cluster

    Returns:
        list of temporally-feasible clusters
    """
    result = list(clusters)  # shallow copy for mutation

    for _ in range(max_iter):
        changed = False
        new_result = []

        for cluster in result:
            if not cluster:
                continue

            is_feasible, tard, spread = check_cluster_tw_feasibility(cluster, instance)

            if is_feasible or len(cluster) <= 2:
                new_result.append(cluster)
                continue

            # Split the infeasible cluster
            sub_clusters = split_temporally_infeasible_cluster(
                cluster, instance, max_sub_clusters=3)

            if len(sub_clusters) > 1:
                new_result.extend(sub_clusters)
                changed = True
            else:
                # Can't split further — accept as-is
                new_result.append(cluster)

        result = new_result
        if not changed:
            break

    # Filter empty and balance capacity
    result = [c for c in result if c]
    result = _balance_capacity(result, capacity=TRUCK_CAPACITY)

    return result


def _balance_capacity(clusters, capacity=200.0):
    """Move customers from overloaded to under-capacity clusters."""
    k = len(clusters)
    if k <= 1:
        return clusters

    total_demands = [sum(c['demand'] for c in cl) for cl in clusters]

    for _ in range(100):
        overloaded = [i for i in range(k) if total_demands[i] > capacity]
        if not overloaded:
            break

        src = overloaded[0]
        under = [i for i in range(k) if total_demands[i] < capacity]
        if not under:
            break

        dst = min(under, key=lambda i: total_demands[i])

        if clusters[src]:
            dst_centroid = np.mean([[c['x'], c['y']] for c in clusters[dst]], axis=0) if clusters[dst] else np.array([8.0, 8.0])
            dists = [(c['x'] - dst_centroid[0])**2 + (c['y'] - dst_centroid[1])**2 for c in clusters[src]]
            move_idx = min(range(len(dists)), key=lambda i: dists[i])

            c = clusters[src].pop(move_idx)
            clusters[dst].append(c)
            total_demands[src] -= c['demand']
            total_demands[dst] += c['demand']

    return clusters
