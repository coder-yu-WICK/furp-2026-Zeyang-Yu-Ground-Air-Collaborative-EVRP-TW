# -*- coding: utf-8 -*-
"""
Unified Pipeline — Week 6 Track B.

Combines Week 5 construction (clustering + POMO + drone) with
Week 3-style IVND repair into a single solve() call.

Pipeline:
  input instance
    → hybrid clustering (W5)
    → POMO per-cluster routing (W4 model)
    → cross-route drone insertion (W5)
    → IVND repair (focused on tardy routes)
    → final solution
"""

import os, sys, time

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.problem_model import TruckDroneSolution, extract_pareto_front, evaluate_solution_batch
from src.pipeline.repair import repair_tardiness, repair_tardiness_partial


def solve_with_repair(instance, n_trucks, variant='hybrid',
                       use_repair=True, repair_mode='full',
                       n_runs=1, seed=42,
                       tw_beta=0.4, drone_endurance='medium',
                       check_tw_feasibility=True,
                       max_drones_per_truck=2):
    """
    Unified pipeline: construction → drone insertion → IVND repair.

    Args:
        instance: problem instance dict
        n_trucks: number of trucks
        variant: W5 clustering variant ('hybrid', 'adaptive_tw', 'baseline', etc.)
        use_repair: if True, apply repair after construction
        repair_mode: 'full' (entire route EDD) or 'partial' (P3: only tardy segment)
        n_runs: number of independent runs
        seed: base random seed
        tw_beta: TW-aware clustering parameter
        drone_endurance: 'medium' (4km) or 'high' (6km)
        check_tw_feasibility: if True, validate cluster TW feasibility before routing
        max_drones_per_truck: 0 (truck-only), 1 (single drone), 2 (dual drone)

    Returns:
        dict with solutions, pareto_front, mean_runtime, repair_stats
    """
    # Lazy import POMO solver (slow to load)
    from src.pipeline.pomo_solver import run_pomo_improved

    all_solutions = []
    all_repair_stats = []
    times = []

    for run in range(n_runs):
        t0 = time.time()
        run_seed = seed + run

        # ── Step 1-2: W5 Construction (cluster + POMO truck-only) ──
        # Always use truck-only POMO; drone insertion is handled below
        # to ensure max_drones_per_truck is respected.
        w5_result = run_pomo_improved(
            instance, n_runs=1, n_trucks=n_trucks,
            endurance=drone_endurance, seed=run_seed,
            variant=variant, tw_beta=tw_beta,
            check_tw_feasibility=check_tw_feasibility,
        )

        if not w5_result['solutions']:
            continue

        # Take best solution from W5 (prefer lowest tardiness)
        pareto = w5_result.get('pareto_front', w5_result['solutions'])
        initial_sol = min(pareto, key=lambda s: s.tardiness) if pareto else w5_result['solutions'][0]

        # ── Step 3: Drone Insertion (with max_drones_per_truck) ──
        # Strip any drone missions POMO may have added, then re-insert
        # using the updated dual-drone logic.
        if max_drones_per_truck > 0:
            from src.pipeline.drone import apply_drone_postprocessing
            # Create a truck-only version for clean drone insertion
            truck_only = TruckDroneSolution(
                initial_sol.truck_routes, [], instance,
                max_drones_per_truck=max_drones_per_truck)
            sol_with_drones, drone_saved, n_drones, drone_counts = apply_drone_postprocessing(
                truck_only, instance, endurance=drone_endurance,
                max_drones_per_truck=max_drones_per_truck)
        else:
            sol_with_drones = TruckDroneSolution(
                initial_sol.truck_routes, [], instance,
                max_drones_per_truck=0)
            drone_saved = 0.0
            n_drones = 0
            drone_counts = [0]

        # ── Step 4: Repair ──
        # RC2 Fix (Week 7): Only run EDD repair when tardiness > epsilon.
        _TARD_EPSILON = 1e-6
        if use_repair and sol_with_drones.tardiness > _TARD_EPSILON:
            if repair_mode == 'partial':
                repaired_sol, stats = repair_tardiness_partial(
                    sol_with_drones, instance, seed=run_seed + 1000,
                    max_drones_per_truck=max_drones_per_truck)
            else:
                repaired_sol, stats = repair_tardiness(
                    sol_with_drones, instance, max_iter=500, seed=run_seed + 1000,
                    max_drones_per_truck=max_drones_per_truck)
            all_repair_stats.append(stats)
        else:
            repaired_sol = sol_with_drones
            all_repair_stats.append({
                'tardiness_before': sol_with_drones.tardiness,
                'tardiness_after': sol_with_drones.tardiness,
                'tardiness_reduction': 0.0,
                'repair_skipped': True,  # RC2 conditional trigger
                'reason': 'already_feasible' if sol_with_drones.tardiness <= _TARD_EPSILON else 'repair_disabled',
            })

        all_solutions.append(repaired_sol)
        times.append(time.time() - t0)

    pareto = extract_pareto_front(all_solutions)

    # Aggregate repair stats
    if all_repair_stats:
        avg_stats = {
            'tardiness_before': sum(s.get('tardiness_before', 0) for s in all_repair_stats) / len(all_repair_stats),
            'tardiness_after': sum(s.get('tardiness_after', 0) for s in all_repair_stats) / len(all_repair_stats),
            'tardiness_reduction': sum(s.get('tardiness_reduction', 0) for s in all_repair_stats) / len(all_repair_stats),
        }
        # Full EDD specific
        if 'moves_accepted' in all_repair_stats[0]:
            avg_stats['moves_accepted'] = sum(s.get('moves_accepted', 0) for s in all_repair_stats) / len(all_repair_stats)
        # Partial EDD specific (P3)
        if 'segments_repaired' in all_repair_stats[0]:
            avg_stats['segments_repaired'] = sum(s.get('segments_repaired', 0) for s in all_repair_stats) / len(all_repair_stats)
        if 'fallback_count' in all_repair_stats[0]:
            avg_stats['fallback_count'] = sum(s.get('fallback_count', 0) for s in all_repair_stats) / len(all_repair_stats)
        if 'partial_success' in all_repair_stats[0]:
            avg_stats['partial_success'] = sum(1 for s in all_repair_stats if s.get('partial_success', False)) / len(all_repair_stats)
    else:
        avg_stats = {}

    return {
        'solutions': all_solutions,
        'pareto_front': pareto,
        'mean_runtime': sum(times) / max(len(times), 1),
        'repair_stats': avg_stats,
    }


def run_pipeline(instance, n_trucks=2, n_runs=1, seed=42,
                 variant='hybrid', use_repair=True, repair_mode='full',
                 max_drones_per_truck=2, **kwargs):
    """
    Alias for solve_with_repair, compatible with experiment runner.
    """
    return solve_with_repair(
        instance, n_trucks=n_trucks, variant=variant,
        use_repair=use_repair, repair_mode=repair_mode,
        n_runs=n_runs, seed=seed,
        max_drones_per_truck=max_drones_per_truck, **kwargs)
