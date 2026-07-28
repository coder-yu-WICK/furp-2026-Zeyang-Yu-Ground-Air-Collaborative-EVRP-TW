# -*- coding: utf-8 -*-
"""
Dual-Drone Support — Week 7.

Thin compatibility wrapper around the canonical week5/drone_post_processing.py
which now natively supports multi-drone-per-truck.

These functions delegate to the canonical implementation. New code should
import directly from drone_post_processing.
"""

import os, sys

# Ensure week5 is importable
_W7 = os.path.dirname(os.path.abspath(__file__))
_W5 = os.path.join(_W7, '..', 'week5')
if _W5 not in sys.path:
    sys.path.insert(0, _W5)

from drone_post_processing import (
    insert_cross_route_drones,
    apply_drone_postprocessing,
    _node_dist,
)


def insert_cross_route_drones_dual(truck_routes, instance,
                                     drone_endurance=4.0,
                                     drone_speed=50.0,
                                     truck_speed=35.0,
                                     drone_capacity=40.0,
                                     max_drones_per_truck=2,
                                     min_saving=0.5):
    """
    Compatibility wrapper — delegates to week5.drone_post_processing.insert_cross_route_drones.
    """
    return insert_cross_route_drones(
        truck_routes, instance,
        drone_endurance=drone_endurance,
        drone_speed=drone_speed,
        truck_speed=truck_speed,
        drone_capacity=drone_capacity,
        max_drones_per_truck=max_drones_per_truck,
        min_saving=min_saving)


def apply_drone_dual(solution, instance, endurance='medium',
                     max_drones_per_truck=2, min_saving=0.5):
    """
    Compatibility wrapper — delegates to week5.drone_post_processing.apply_drone_postprocessing.
    """
    return apply_drone_postprocessing(
        solution, instance,
        endurance=endurance,
        max_drones_per_truck=max_drones_per_truck,
        min_saving=min_saving)


# ── Ablation helper: compare 0 vs 1 vs 2 drones ──────────────────────────

def compare_drone_configs(solution, instance, endurance='medium'):
    """
    Compare 0, 1, and 2 drones per truck on the same solution.

    Returns:
        dict mapping config name → (solution, cost_saved, n_drones)
    """
    results = {}

    # 0 drones (truck-only)
    results['0_drones'] = (solution, 0.0, 0)

    # 1 drone per truck
    sol_1, saved_1, n_1, counts_1 = apply_drone_dual(
        solution, instance, endurance, max_drones_per_truck=1)
    results['1_drone'] = (sol_1, saved_1, n_1)

    # 2 drones per truck
    sol_2, saved_2, n_2, counts_2 = apply_drone_dual(
        solution, instance, endurance, max_drones_per_truck=2)
    results['2_drones'] = (sol_2, saved_2, n_2)

    return results
