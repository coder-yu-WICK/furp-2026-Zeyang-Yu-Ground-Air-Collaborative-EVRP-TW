# -*- coding: utf-8 -*-
"""
TW-Aware Clustering for POMO-MT — Week 5.

Strategy: Spatial K-means first, then temporal split.

The fundamental limitation of weighted-distance K-means is that spatial
proximity dominates — two customers at the same location with time windows
4 hours apart will always end up in the same cluster because their spatial
distance to the shared centroid is near-zero.

This module uses a two-phase approach:
  1. Standard spatial K-means (same as Week 4 baseline)
  2. Temporal split: within each spatial cluster, sort by TW midpoint and
     split into morning/afternoon/evening sub-groups when the temporal gap
     exceeds a threshold.

This directly addresses the root cause identified in the Week 4 report:
customers at identical coordinates but with time windows hours apart are
now separated into different routes.
"""

import os, sys, math
import numpy as np

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from week8.config import TRUCK_CAPACITY


def _temporal_split_cluster(cluster, max_gap_ratio=0.4, horizon=240.0):
    """
    Split a cluster into temporal sub-groups.

    Sorts customers by TW midpoint (ready_time + due_time) / 2.
    Splits when the gap between consecutive customers exceeds
    max_gap_ratio * horizon.

    Args:
        cluster: list of customer dicts
        max_gap_ratio: split when temporal gap > this fraction of horizon
        horizon: TW horizon for normalization

    Returns:
        list of sub-clusters (each is list of customer dicts)
    """
    if len(cluster) <= 1:
        return [cluster] if cluster else []

    # Sort by TW midpoint
    sorted_cluster = sorted(cluster, key=lambda c: (c['ready_time'] + c['due_time']) / 2)

    # Find split points
    split_threshold = max_gap_ratio * horizon
    sub_clusters = []
    current = [sorted_cluster[0]]

    for i in range(1, len(sorted_cluster)):
        prev_mid = (sorted_cluster[i-1]['ready_time'] + sorted_cluster[i-1]['due_time']) / 2
        curr_mid = (sorted_cluster[i]['ready_time'] + sorted_cluster[i]['due_time']) / 2
        gap = curr_mid - prev_mid

        if gap > split_threshold:
            # Start new sub-cluster
            sub_clusters.append(current)
            current = [sorted_cluster[i]]
        else:
            current.append(sorted_cluster[i])

    sub_clusters.append(current)
    return sub_clusters


def tw_aware_cluster_customers(instance, n_trucks, max_gap_ratio=0.4, seed=42):
    """
    Partition customers using spatial K-means + temporal split.

    Phase 1: Standard spatial K-means (from week4)
    Phase 2: Within each spatial cluster, split temporally-distant customers
             into separate sub-groups.

    Args:
        instance: problem instance dict
        n_trucks: minimum number of trucks
        max_gap_ratio: temporal split threshold (fraction of TW horizon)
        seed: random seed

    Returns:
        list of clusters (each is list of customer dicts)
    """
    customers = instance['customers']
    total_demand = sum(c['demand'] for c in customers)
    min_clusters = max(1, int(math.ceil(total_demand / TRUCK_CAPACITY)))
    n_spatial = max(n_trucks, min_clusters)

    horizon = instance.get('tw_horizon', 240.0)

    # Phase 1: Spatial K-means
    from week8.pipeline.pomo_multitruck import _kmeans_cluster
    spatial_clusters = _kmeans_cluster(customers, n_spatial, seed=seed)

    # Phase 2: Temporal split within each spatial cluster
    all_clusters = []
    for sc in spatial_clusters:
        if not sc:
            continue
        sub = _temporal_split_cluster(sc, max_gap_ratio=max_gap_ratio, horizon=horizon)
        all_clusters.extend(sub)

    # Phase 3: Capacity balancing
    all_clusters = _balance_capacity(all_clusters, capacity=TRUCK_CAPACITY)

    # Filter empty clusters
    all_clusters = [c for c in all_clusters if c]
    return all_clusters


def _balance_capacity(clusters, capacity=200.0):
    """Simple capacity balancing — move from overloaded to under-capacity."""
    k = len(clusters)
    total_demands = [sum(c['demand'] for c in cl) for cl in clusters]

    for _ in range(100):
        overloaded = [i for i in range(k) if total_demands[i] > capacity]
        if not overloaded:
            break

        src = overloaded[0]
        under = [i for i in range(k) if total_demands[i] < capacity]
        if not under:
            break

        # Move nearest customer from src to nearest under-capacity cluster
        dst = min(under, key=lambda i: total_demands[i])  # least full

        if clusters[src]:
            # Move the customer closest to dst centroid
            dst_centroid = np.mean([[c['x'], c['y']] for c in clusters[dst]], axis=0) if clusters[dst] else np.array([8.0, 8.0])
            dists = [(c['x'] - dst_centroid[0])**2 + (c['y'] - dst_centroid[1])**2 for c in clusters[src]]
            move_idx = min(range(len(dists)), key=lambda i: dists[i])

            c = clusters[src].pop(move_idx)
            clusters[dst].append(c)
            total_demands[src] -= c['demand']
            total_demands[dst] += c['demand']

    return clusters


# Backward-compatible alias
def cluster_customers_tw_aware(instance, n_trucks, beta=0.5, seed=42):
    """
    Backward-compatible interface.
    beta is re-interpreted as max_gap_ratio (0 = no split, 1 = aggressive split).
    """
    return tw_aware_cluster_customers(instance, n_trucks, max_gap_ratio=beta, seed=seed)


# ═══════════════════════════════════════════════════════════════════════════
# Budget-Aware Clustering — Designed for Partial EDD Compatibility
# ═══════════════════════════════════════════════════════════════════════════

def _compute_edd_route_time(cluster, depot, truck_speed=35.0):
    """
    Compute total time for an EDD-ordered route through a cluster.

    This is the LOWER BOUND on route time — no ordering can be faster
    than EDD for satisfying time windows. If even EDD takes too long,
    the cluster is fundamentally too large and must be split.

    Returns: (total_time, tardiness, n_tardy_customers)
    """
    if len(cluster) <= 1:
        c = cluster[0]
        d = math.sqrt((depot[0]-c['x'])**2 + (depot[1]-c['y'])**2)
        travel = d / truck_speed
        arr = max(travel, c['ready_time'])
        tard = max(0, arr - c['due_time'])
        return arr + c['service_time'] + d/truck_speed, tard, 1 if tard > 0 else 0

    sorted_by_due = sorted(cluster, key=lambda c: c['due_time'])
    current_time = 0.0
    total_tard = 0.0
    n_tardy = 0
    prev_x, prev_y = depot[0], depot[1]

    for c in sorted_by_due:
        dx = prev_x - c['x']
        dy = prev_y - c['y']
        travel = math.sqrt(dx*dx + dy*dy) / truck_speed
        current_time = max(current_time + travel, c['ready_time'])
        current_time += c['service_time']
        if current_time > c['due_time']:
            total_tard += current_time - c['due_time']
            n_tardy += 1
        prev_x, prev_y = c['x'], c['y']

    # Return to depot
    return_dist = math.sqrt((prev_x-depot[0])**2 + (prev_y-depot[1])**2)
    total_time = current_time + return_dist / truck_speed
    return total_time, total_tard, n_tardy


def _find_best_split_point(cluster, depot, truck_speed=35.0):
    """
    Find the best temporal split point that minimizes
    the maximum EDD route time of the resulting sub-clusters.

    Unlike the fixed-threshold approach, this directly optimizes
    the split for EDD route time reduction — the metric that matters.

    Returns: (best_split_index, best_max_time)
    """
    if len(cluster) <= 2:
        return -1, float('inf')

    sorted_cluster = sorted(cluster, key=lambda c: (c['ready_time'] + c['due_time']) / 2)
    n = len(sorted_cluster)
    best_split = -1
    best_max_time = float('inf')

    # Try each possible split point
    for split_at in range(1, n):
        sub1 = sorted_cluster[:split_at]
        sub2 = sorted_cluster[split_at:]
        if len(sub1) == 0 or len(sub2) == 0:
            continue
        t1, _, _ = _compute_edd_route_time(sub1, depot, truck_speed)
        t2, _, _ = _compute_edd_route_time(sub2, depot, truck_speed)
        max_time = max(t1, t2)
        if max_time < best_max_time:
            best_max_time = max_time
            best_split = split_at

    return best_split, best_max_time


def budget_aware_cluster(instance, n_trucks, time_budget_ratio=0.75, seed=42):
    """
    Budget-aware clustering designed for Partial EDD compatibility.

    PHILOSOPHY (different from previous approaches):
      - Start with PURE SPATIAL K-means (preserve POMO's distance optimization)
      - Only split when the EDD route time exceeds the time budget
      - Split at the point that MOST reduces the worst sub-cluster's route time
      - This creates MINIMAL, TARGETED splits — leaving most clusters intact

    Why this helps Partial EDD:
      Partial EDD fails when the upstream (non-tardy) part of a route is too long.
      By ensuring each cluster's EDD route fits within budget_ratio × horizon,
      we leave SLACK for POMO's sub-optimal ordering. Then when Partial EDD
      reorders a tardy segment, there's enough time buffer upstream.

    Args:
        instance: problem instance dict
        n_trucks: minimum number of trucks
        time_budget_ratio: fraction of TW horizon that a cluster's EDD route
                          must fit within. 0.75 = EDD route ≤ 75% of horizon.
                          Lower = tighter budget = more splits = easier for Partial EDD.
        seed: random seed

    Returns:
        list of clusters (each is list of customer dicts)
    """
    from week8.pipeline.pomo_multitruck import _kmeans_cluster

    customers = instance['customers']
    total_demand = sum(c['demand'] for c in customers)
    min_clusters = max(1, int(math.ceil(total_demand / TRUCK_CAPACITY)))
    n_spatial = max(n_trucks, min_clusters)
    horizon = instance.get('tw_horizon', 240.0)
    depot = instance['depot']
    time_budget = horizon * time_budget_ratio

    # ── Phase 1: Pure spatial K-means ──
    clusters = _kmeans_cluster(customers, n_spatial, seed=seed)

    # ── Phase 2: Targeted splitting (only where needed) ──
    # Iteratively split the worst cluster until all fit within the time budget
    MAX_TOTAL_CLUSTERS = n_spatial * 3  # Safety: never more than 3x original

    for _ in range(MAX_TOTAL_CLUSTERS - len(clusters)):
        # Find the cluster with the worst (longest) EDD route time
        worst_idx = -1
        worst_time = 0

        for i, cluster in enumerate(clusters):
            if len(cluster) <= 1:
                continue
            route_time, tard, n_tardy = _compute_edd_route_time(cluster, depot)
            if route_time > worst_time:
                worst_time = route_time
                worst_idx = i

        # If all clusters fit within budget, we're done
        if worst_time <= time_budget:
            break

        # Split the worst cluster at the optimal temporal point
        worst_cluster = clusters[worst_idx]
        split_idx, split_max_time = _find_best_split_point(worst_cluster, depot)

        if split_idx < 0 or split_max_time >= worst_time:
            # Can't improve by splitting — accept as-is
            break

        sorted_cluster = sorted(worst_cluster,
                                key=lambda c: (c['ready_time'] + c['due_time']) / 2)
        clusters[worst_idx] = sorted_cluster[:split_idx]
        clusters.append(sorted_cluster[split_idx:])

    # ── Phase 3: Capacity balancing ──
    clusters = _balance_capacity(clusters, capacity=TRUCK_CAPACITY)
    clusters = [c for c in clusters if c]

    return clusters
