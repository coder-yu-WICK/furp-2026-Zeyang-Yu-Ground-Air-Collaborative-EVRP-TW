# -*- coding: utf-8 -*-
"""
Advanced Clustering Strategies for POMO-MT — Week 5 Extended.

This module provides three clustering strategies that address the
limitations of the basic TW-aware approach:

1. Adaptive TW-Aware: auto-tunes max_gap_ratio per cluster based on
   internal TW density, avoiding both over-splitting (tight clusters)
   and under-splitting (wide clusters).

2. Angle-Based (Petal): for RC1 tight-TW instances, clusters customers
   by polar angle from depot, creating geographically compact sectors
   that minimize intra-route travel time.

3. Hybrid: uses angle-based for RC1 instances and adaptive TW-aware
   for RC2 instances, automatically selected by TW type.
"""

import os, sys, math
import numpy as np

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from week8.config import TRUCK_CAPACITY


# ═════════════════════════════════════════════════════════════════════════
# Strategy 1: Adaptive TW-Aware Clustering
# ═════════════════════════════════════════════════════════════════════════

def _adaptive_gap_threshold(cluster, base_ratio=0.4, horizon=240.0):
    """
    Compute adaptive gap threshold for a cluster based on its internal
    TW density.

    For clusters with tight, uniform TWs → higher threshold (fewer splits).
    For clusters with wide, bimodal TWs → lower threshold (more splits).

    Returns: adaptive gap threshold in minutes
    """
    if len(cluster) <= 1:
        return float('inf')

    # Sort midpoints and compute gaps
    midpoints = sorted((c['ready_time'] + c['due_time']) / 2 for c in cluster)
    gaps = [midpoints[i+1] - midpoints[i] for i in range(len(midpoints)-1)]

    if not gaps:
        return float('inf')

    median_gap = np.median(gaps)
    tw_spread = midpoints[-1] - midpoints[0]
    tw_density = tw_spread / horizon  # 0=tight, 1=spread across full horizon

    # Adaptive formula:
    # - If TW spread is small (<25% of horizon): threshold = large (don't split)
    # - If TW spread is large (>50% of horizon): threshold = tighter (split more)
    # - Base threshold uses median_gap to find natural breaks
    if tw_density < 0.25:
        # Tight cluster: only split on very large gaps
        adaptive = max(base_ratio * horizon, 5.0 * median_gap)
    elif tw_density < 0.5:
        # Moderate spread: split on 3x median gap
        adaptive = max(base_ratio * horizon * 0.8, 3.0 * median_gap)
    else:
        # Wide spread: aggressive splitting
        adaptive = max(base_ratio * horizon * 0.5, 2.0 * median_gap)

    return adaptive


def _temporal_split_adaptive(cluster, base_ratio=0.4, horizon=240.0):
    """
    Split a cluster using adaptive gap threshold.
    """
    if len(cluster) <= 1:
        return [cluster] if cluster else []

    threshold = _adaptive_gap_threshold(cluster, base_ratio, horizon)
    sorted_cluster = sorted(cluster, key=lambda c: (c['ready_time'] + c['due_time']) / 2)

    sub_clusters = []
    current = [sorted_cluster[0]]

    for i in range(1, len(sorted_cluster)):
        prev_mid = (sorted_cluster[i-1]['ready_time'] + sorted_cluster[i-1]['due_time']) / 2
        curr_mid = (sorted_cluster[i]['ready_time'] + sorted_cluster[i]['due_time']) / 2
        gap = curr_mid - prev_mid

        if gap > threshold:
            sub_clusters.append(current)
            current = [sorted_cluster[i]]
        else:
            current.append(sorted_cluster[i])

    sub_clusters.append(current)
    return sub_clusters


def adaptive_tw_aware_cluster(instance, n_trucks, base_ratio=0.4, seed=42):
    """
    TW-aware clustering with adaptive threshold per cluster.

    Uses the same spatial K-means + temporal split structure as the
    basic version, but the split threshold adapts to each cluster's
    internal TW density.
    """
    from week8.pipeline.pomo_multitruck import _kmeans_cluster

    customers = instance['customers']
    total_demand = sum(c['demand'] for c in customers)
    min_clusters = max(1, int(math.ceil(total_demand / TRUCK_CAPACITY)))
    n_spatial = max(n_trucks, min_clusters)

    horizon = instance.get('tw_horizon', 240.0)

    # Phase 1: Spatial K-means
    spatial_clusters = _kmeans_cluster(customers, n_spatial, seed=seed)

    # Phase 2: Adaptive temporal split
    all_clusters = []
    for sc in spatial_clusters:
        if not sc:
            continue
        sub = _temporal_split_adaptive(sc, base_ratio=base_ratio, horizon=horizon)
        all_clusters.extend(sub)

    # Phase 3: Capacity balancing
    all_clusters = _balance_capacity(all_clusters, capacity=TRUCK_CAPACITY)

    all_clusters = [c for c in all_clusters if c]
    return all_clusters


# ═════════════════════════════════════════════════════════════════════════
# Strategy 2: Angle-Based (Petal) Clustering for RC1
# ═════════════════════════════════════════════════════════════════════════

def angle_based_cluster(instance, n_trucks, seed=42):
    """
    Cluster customers by polar angle from depot (petal pattern).

    For RC1 tight-TW instances, temporal splitting doesn't help because
    all TWs are within a 120-min window. Instead, creating angle-based
    sectors minimizes the travel distance within each cluster, allowing
    trucks to serve more customers before their deadlines.

    The algorithm:
    1. Sort customers by polar angle from depot
    2. Sweep and assign to n_trucks sectors
    3. Balance capacity between sectors
    """
    customers = instance['customers']
    depot = instance['depot']

    if len(customers) == 0:
        return []

    # Compute polar angles
    angles = []
    for c in customers:
        dx = c['x'] - depot[0]
        dy = c['y'] - depot[1]
        angle = math.atan2(dy, dx)  # [-pi, pi]
        angles.append(angle)

    # Sort customers by angle
    sorted_pairs = sorted(zip(angles, customers), key=lambda p: p[0])

    # Split into n_trucks sectors
    total = len(customers)
    base_size = total // n_trucks
    remainder = total % n_trucks

    clusters = []
    idx = 0
    for t in range(n_trucks):
        size = base_size + (1 if t < remainder else 0)
        cluster = [c for _, c in sorted_pairs[idx:idx + size]]
        clusters.append(cluster)
        idx += size

    # Capacity balancing
    clusters = _balance_capacity(clusters, capacity=TRUCK_CAPACITY)

    return [c for c in clusters if c]


# ═════════════════════════════════════════════════════════════════════════
# Strategy 3: Hybrid — Auto-Select by TW Type
# ═════════════════════════════════════════════════════════════════════════

def hybrid_cluster(instance, n_trucks, base_ratio=0.4, seed=42):
    """
    Hybrid clustering: budget-aware spatial + targeted temporal splits.

    Strategy:
      - Start with PURE spatial K-means (preserves POMO distance structure)
      - Only split clusters whose EDD route time exceeds 75% of TW horizon
      - Split at the optimal temporal point (minimizes worst sub-cluster time)
      - This is MINIMAL intervention — most clusters stay intact

    Why this design:
      Partial EDD fails when the upstream route is too long — the accumulated
      travel time means the tardy segment is already late before it starts.
      By ensuring each cluster's EDD route fits within 75% of the horizon,
      we leave slack for POMO's sub-optimal ordering AND for Partial EDD's
      segment-level fix to actually work.
    """
    from week8.pipeline.clustering import budget_aware_cluster
    return budget_aware_cluster(
        instance, n_trucks, time_budget_ratio=0.75, seed=seed)


# ═════════════════════════════════════════════════════════════════════════
# Shared: Capacity Balancing
# ═════════════════════════════════════════════════════════════════════════

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


# ═════════════════════════════════════════════════════════════════════════
# Parameter Sweep Helpers
# ═════════════════════════════════════════════════════════════════════════

def cluster_with_params(instance, n_trucks, strategy='adaptive_tw',
                        base_ratio=0.4, seed=42):
    """
    Unified clustering interface for parameter sweeps.

    Args:
        strategy: 'adaptive_tw' | 'angle' | 'hybrid' | 'spatial' | 'tw_aware'
        base_ratio: max_gap_ratio for TW-aware strategies
    """
    if strategy == 'adaptive_tw':
        return adaptive_tw_aware_cluster(instance, n_trucks, base_ratio=base_ratio, seed=seed)
    elif strategy == 'angle':
        return angle_based_cluster(instance, n_trucks, seed=seed)
    elif strategy == 'hybrid':
        return hybrid_cluster(instance, n_trucks, base_ratio=base_ratio, seed=seed)
    elif strategy == 'spatial':
        from week8.pipeline.pomo_multitruck import cluster_customers
        return cluster_customers(instance, n_trucks)
    elif strategy == 'tw_aware':
        from week8.pipeline.clustering import cluster_customers_tw_aware
        return cluster_customers_tw_aware(instance, n_trucks, beta=base_ratio, seed=seed)
    elif strategy == 'spatiotemporal':
        from week8.pipeline.pomo_multitruck import cluster_customers_spatiotemporal
        return cluster_customers_spatiotemporal(
            instance, n_trucks, tw_weight=base_ratio if base_ratio else 0.3, seed=seed)
    elif strategy == 'budget_aware':
        from week8.pipeline.clustering import budget_aware_cluster
        return budget_aware_cluster(
            instance, n_trucks,
            time_budget_ratio=base_ratio if base_ratio else 0.75, seed=seed)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
