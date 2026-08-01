# -*- coding: utf-8 -*-
"""
EVRP-TW Pipeline — Week 8 (truck-only, no drones).

Pipeline: Clustering → POMO Neural Routing → EDD Repair → EV Evaluation.

Teacher guidance: Remove truck-drone collaboration to differentiate from
classmate's work. Focus on EDD repair as our unique contribution to EVRP-TW.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from week8.core.problem_model import TruckSolution, extract_pareto_front, evaluate_solution_batch
from week8.pipeline.repair import repair_tardiness_truck, repair_tardiness_partial_truck


def solve_evrptw(instance, n_trucks, variant='hybrid',
                 use_repair=True, repair_mode='full',
                 n_runs=1, seed=42, tw_beta=0.4,
                 check_tw_feasibility=True):
    """
    EVRP-TW pipeline: Clustering → POMO routing → EDD repair.

    Args:
        instance: problem instance dict
        n_trucks: number of trucks
        variant: clustering variant ('hybrid', 'adaptive_tw', 'baseline', etc.)
        use_repair: if True, apply EDD repair after construction
        repair_mode: 'full' (entire route EDD) or 'partial' (only tardy segments)
        n_runs: number of independent runs
        seed: base random seed
        tw_beta: TW-aware clustering parameter
        check_tw_feasibility: validate cluster TW feasibility before routing

    Returns:
        dict with solutions, pareto_front, mean_runtime, repair_stats
    """
    from week8.pipeline.pomo_solver import run_pomo_improved

    all_solutions = []
    all_repair_stats = []
    times = []

    for run in range(n_runs):
        t0 = time.time()
        run_seed = seed + run

        # ── Step 1-2: Clustering + POMO truck routing ─────────────
        w5_result = run_pomo_improved(
            instance, n_runs=1, n_trucks=n_trucks,
            seed=run_seed, variant=variant, tw_beta=tw_beta,
            check_tw_feasibility=check_tw_feasibility,
        )

        if not w5_result['solutions']:
            continue

        pareto = w5_result.get('pareto_front', w5_result['solutions'])
        truck_sol = min(pareto, key=lambda s: s.tardiness) if pareto else w5_result['solutions'][0]

        # Ensure it's a clean TruckSolution (no drone missions)
        if not isinstance(truck_sol, TruckSolution):
            truck_sol = TruckSolution(truck_sol.truck_routes, instance)

        # ── Step 3: EDD Repair ────────────────────────────────────
        _TARD_EPSILON = 1e-6
        if use_repair and truck_sol.tardiness > _TARD_EPSILON:
            if repair_mode == 'partial':
                repaired_sol, stats = repair_tardiness_partial_truck(
                    truck_sol, instance, seed=run_seed + 1000)
            else:
                repaired_sol, stats = repair_tardiness_truck(
                    truck_sol, instance, max_iter=500, seed=run_seed + 1000)
            all_repair_stats.append(stats)
        else:
            repaired_sol = truck_sol
            all_repair_stats.append({
                'tardiness_before': truck_sol.tardiness,
                'tardiness_after': truck_sol.tardiness,
                'tardiness_reduction': 0.0,
                'repair_skipped': True,
                'reason': 'already_feasible' if truck_sol.tardiness <= _TARD_EPSILON else 'repair_disabled',
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
        if 'moves_accepted' in all_repair_stats[0]:
            avg_stats['moves_accepted'] = sum(s.get('moves_accepted', 0) for s in all_repair_stats) / len(all_repair_stats)
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
                 variant='hybrid', use_repair=True, repair_mode='full', **kwargs):
    """Alias for solve_evrptw, compatible with experiment runners."""
    return solve_evrptw(
        instance, n_trucks=n_trucks, variant=variant,
        use_repair=use_repair, repair_mode=repair_mode,
        n_runs=n_runs, seed=seed, **kwargs)
