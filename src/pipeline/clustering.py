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

from src.config import TRUCK_CAPACITY


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
    from src.pipeline.pomo_multitruck import _kmeans_cluster
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
