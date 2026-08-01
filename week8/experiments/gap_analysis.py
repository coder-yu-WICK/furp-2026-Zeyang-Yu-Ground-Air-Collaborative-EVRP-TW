#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FURP: Optimality Gap Analysis.

Computes optimal/benchmark solutions for small instances (10-15 customers)
and compares POMO results against them to quantify the optimality gap.

Approach:
  1. Generate 10c/15c subsets from Solomon instances
  2. Try OR-Tools exact solver (CP-SAT or Routing)
  3. Fallback: Exhaustive enumeration for <=10c; best-known for 15c
  4. Compare POMO (without drones, without repair) vs optimal

References:
  - Solomon (1987) best-known solutions (BKS)
  - OR-Tools VRPTW example (Google)
"""

import sys, os, math, time, json, random
from itertools import permutations, combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from week8.config import (TRUCK_SPEED, TRUCK_CAPACITY, DEPOT,
                     TRUCK_DIST_COST_RATE, TRUCK_FIXED_COST)
from week8.core.data_loader import load_instance_from_disk, build_instance
from week8.core.problem_model import TruckSolution


def make_small_instance(source_inst, n_cust):
    """Create a small instance with first n_cust customers."""
    inst = load_instance_from_disk(f'{source_inst}_50c')
    customers = inst['customers'][:n_cust]
    from week8.core.data_loader import compute_distance_matrix
    dist_matrix = compute_distance_matrix(customers, DEPOT)
    return {
        'name': f'{source_inst}_{n_cust}c_gap',
        'source_instance': source_inst,
        'depot': list(DEPOT),
        'n_customers': n_cust,
        'customers': customers,
        'distance_matrix': dist_matrix,
        'tw_type': inst['tw_type'],
        'tw_horizon': inst['tw_horizon'],
    }


def enumerate_all_routes(customers, dist_matrix, depot, capacity):
    """
    Enumerate all feasible single-truck routes for <=10 customers.
    Returns: (best_route, best_cost, best_tardiness)
    """
    n = len(customers)
    if n > 10:
        return None, float('inf'), float('inf')

    best_route = None
    best_cost = float('inf')
    best_tard = float('inf')

    # We'll try all permutations of customers
    for perm in permutations(range(1, n + 1)):
        route = list(perm)
        # Quick capacity check
        load = sum(customers[cid-1]['demand'] for cid in route)
        if load > capacity:
            continue

        # Simulate route
        current_time = 0.0
        prev = 0
        total_dist = 0.0
        total_tard = 0.0

        for cid in route:
            seg_dist = dist_matrix[prev][cid]
            total_dist += seg_dist
            current_time += seg_dist / TRUCK_SPEED

            c = customers[cid-1]
            if current_time < c['ready_time']:
                current_time = c['ready_time']
            if current_time > c['due_time']:
                total_tard += (current_time - c['due_time'])

            current_time += c['service_time']
            prev = cid

        # Return to depot
        total_dist += dist_matrix[prev][0]
        current_time += dist_matrix[prev][0] / TRUCK_SPEED

        cost = total_dist * TRUCK_DIST_COST_RATE  # distance-only (no fixed costs)

        # Minimize tardiness first, then cost
        if total_tard < best_tard - 0.01 or (abs(total_tard - best_tard) < 0.01 and cost < best_cost):
            best_tard = total_tard
            best_cost = cost
            best_route = route

    return best_route, best_cost, best_tard


def enumerate_multi_route(customers, dist_matrix, depot, capacity, n_trucks):
    """
    For multi-truck routing with <=10 customers and 2 trucks:
    Try all partitions of customers into routes, enumerate each route.
    """
    n = len(customers)
    if n > 10 or n_trucks > 2:
        return None, float('inf'), float('inf')

    best_overall = None
    best_cost = float('inf')
    best_tard = float('inf')

    all_ids = list(range(1, n + 1))

    # Try all ways to split customers into n_trucks routes
    for r in range(n + 1):
        for subset_combo in combinations(all_ids, r):
            route0_ids = list(subset_combo)
            route1_ids = [cid for cid in all_ids if cid not in route0_ids]

            # Enumerate both routes
            r0, c0, t0 = enumerate_all_routes(
                [customers[cid-1] for cid in route0_ids] if route0_ids else [],
                dist_matrix, depot, capacity) if route0_ids else ([], 0, 0)

            r1, c1, t1 = enumerate_all_routes(
                [customers[cid-1] for cid in route1_ids] if route1_ids else [],
                dist_matrix, depot, capacity) if route1_ids else ([], 0, 0)

            if r0 is None or r1 is None:
                continue

            # Remap route IDs to original customer IDs
            if route0_ids:
                r0_remapped = [route0_ids[i-1] if i <= len(route0_ids) else route0_ids[i]
                              for i in range(len(r0))]
            else:
                r0_remapped = []
            if route1_ids:
                r1_remapped = []
                for temp_id in r1:
                    r1_remapped.append(route1_ids[temp_id - 1])
            else:
                r1_remapped = []

            total_cost = c0 + c1
            total_tard = t0 + t1

            if total_tard < best_tard - 0.01 or (abs(total_tard - best_tard) < 0.01 and total_cost < best_cost):
                best_tard = total_tard
                best_cost = total_cost
                best_overall = [r0_remapped, r1_remapped]

    return best_overall, best_cost, best_tard


def run_pomo_on_instance(inst, n_trucks, seed=42):
    """Run POMO on a small instance and return best solution."""
    from week8.pipeline.pomo_solver import run_pomo_improved

    result = run_pomo_improved(
        inst, n_runs=1, n_trucks=n_trucks,
        endurance='medium', seed=seed,
        variant='hybrid', tw_beta=0.4,
        check_tw_feasibility=True,
    )

    if not result.get('solutions'):
        return None, float('inf'), float('inf')

    pareto = result.get('pareto_front', result['solutions'])
    best = min(pareto, key=lambda s: (s.tardiness, s.cost))
    return best.truck_routes, best.cost, best.tardiness


def main():
    print("OPTIMALITY GAP ANALYSIS")
    print("=" * 70)

    configs = [
        ('RC101', 8, 2),
        ('RC101', 10, 2),
        ('RC201', 8, 2),
        ('RC201', 10, 2),
        ('R101', 8, 2),
        ('C101', 8, 2),
    ]

    results = []

    for source_inst, n_cust, n_trucks in configs:
        print(f"\n{'-'*60}")
        print(f"  {source_inst}_{n_cust}c ({n_trucks} trucks)")
        print(f"{'-'*60}")

        inst = make_small_instance(source_inst, n_cust)
        customers = inst['customers']
        depot = inst['depot']
        if isinstance(depot, list):
            depot = tuple(depot)
        dist_matrix = inst['distance_matrix']

        # ── Optimal (exhaustive) ──
        t0 = time.time()
        if n_cust <= 8 and n_trucks == 1:
            opt_route, opt_cost, opt_tard = enumerate_all_routes(
                customers, dist_matrix, depot, TRUCK_CAPACITY)
        else:
            opt_route, opt_cost, opt_tard = enumerate_multi_route(
                customers, dist_matrix, depot, TRUCK_CAPACITY, n_trucks)
        opt_time = time.time() - t0

        if opt_route is not None:
            print(f"  Optimal (enumeration): cost={opt_cost:.1f}, tard={opt_tard:.1f}, time={opt_time:.1f}s")
        else:
            print(f"  Optimal: NOT FOUND (too large for enumeration)")
            opt_cost = float('inf')
            opt_tard = float('inf')

        # ── POMO ──
        try:
            t0 = time.time()
            pomo_route, pomo_cost, pomo_tard = run_pomo_on_instance(inst, n_trucks)
            pomo_time = time.time() - t0
            print(f"  POMO:                 cost={pomo_cost:.1f}, tard={pomo_tard:.1f}, time={pomo_time:.1f}s")

            if opt_cost < float('inf') and pomo_cost < float('inf'):
                gap = (pomo_cost - opt_cost) / opt_cost * 100
                print(f"  GAP: {gap:+.1f}% (POMO vs Optimal)")
            else:
                gap = None
        except Exception as e:
            print(f"  POMO: ERROR -- {e}")
            pomo_cost = float('inf')
            pomo_tard = float('inf')
            gap = None

        # ── POMO + EDD repair ──
        try:
            from week8.experiments.run_sota_expanded import run_ours
            t0 = time.time()
            sol = run_ours(inst, n_trucks=n_trucks, seed=42,
                          use_repair=True, repair_mode='full',
                          n_drones_per_truck=0)
            pomo_edd_time = time.time() - t0
            print(f"  POMO+EDD:             cost={sol.cost:.1f}, tard={sol.tardiness:.1f}, "
                  f"time={pomo_edd_time:.1f}s feas={sol.feasible}")

            if opt_cost < float('inf') and sol.cost < float('inf'):
                gap_edd = (sol.cost - opt_cost) / opt_cost * 100
                print(f"  GAP (with EDD): {gap_edd:+.1f}%")
        except Exception as e:
            print(f"  POMO+EDD: ERROR -- {e}")

        results.append({
            'instance': f'{source_inst}_{n_cust}c',
            'n_customers': n_cust,
            'n_trucks': n_trucks,
            'optimal_cost': opt_cost,
            'optimal_tardiness': opt_tard,
            'pomo_cost': pomo_cost,
            'pomo_tardiness': pomo_tard,
            'gap_pct': gap,
        })

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Instance':<16s} {'Optimal':>10s} {'POMO':>10s} {'POMO+EDD':>10s} {'GAP%':>8s}")
    print(f"  {'-'*60}")
    for r in results:
        gap_val = r['gap_pct']
        g = f'{gap_val:+.1f}%' if gap_val is not None else 'N/A'
        print(f"  {r['instance']:<16s} {r['optimal_cost']:>10.1f} {r['pomo_cost']:>10.1f} "
              f"{'N/A':>10s} {g:>8s}")

    valid_gaps = [r['gap_pct'] for r in results if r['gap_pct'] is not None]
    if valid_gaps:
        print(f"\n  Average gap: {sum(valid_gaps)/len(valid_gaps):.1f}%")
        print(f"  Range: {min(valid_gaps):.1f}% -- {max(valid_gaps):.1f}%")


if __name__ == '__main__':
    main()
