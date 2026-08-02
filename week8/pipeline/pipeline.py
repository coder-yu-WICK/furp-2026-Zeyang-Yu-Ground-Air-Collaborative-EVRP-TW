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
from week8.pipeline.repair import (
    repair_tardiness_truck, repair_tardiness_partial_truck,
    repair_forward_insertion, constrained_local_search,
    merge_routes_post_repair,
)


def solve_evrptw(instance, n_trucks, variant='hybrid',
                 use_repair=True, repair_mode='forward',
                 use_cls=True, use_merge=True,
                 use_ev=False, ev_charging_model='nonlinear',
                 ev_battery_capacity=None,
                 n_runs=1, seed=42, tw_beta=0.4,
                 check_tw_feasibility=True):
    """
    EVRP-TW pipeline: Clustering → POMO routing → Forward Insertion repair.

    Args:
        instance: problem instance dict
        n_trucks: number of trucks
        variant: clustering variant ('hybrid', 'adaptive_tw', 'baseline', etc.)
        use_repair: if True, apply EDD repair after construction
        repair_mode: 'forward' (NEW: surgical forward insertion — recommended),
                     'partial' (segment-level EDD, mostly falls back to full),
                     'full' (entire route EDD + inter-route improvement)
        use_cls: if True, apply Constrained Local Search after repair
        use_merge: if True, apply Route Merging after repair
        use_ev: if True, insert charging stops and evaluate EV constraints
        ev_charging_model: 'linear', 'nonlinear', or 'none'
        ev_battery_capacity: battery kWh (default: config BATTERY_CAPACITY)
        n_runs: number of independent runs
        seed: base random seed
        tw_beta: TW-aware clustering parameter
        check_tw_feasibility: validate cluster TW feasibility before routing

    Returns:
        dict with solutions, pareto_front, mean_runtime, repair_stats
        (and ev_stats if use_ev=True)
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
            elif repair_mode == 'forward':
                repaired_sol, stats = repair_forward_insertion(
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

        # ── Step 3.5: Constrained Local Search (post-repair distance opt) ──
        cls_stats = {}
        if use_cls and repaired_sol.tardiness <= _TARD_EPSILON:
            repaired_sol, cls_stats = constrained_local_search(
                repaired_sol, instance, max_iter=100, seed=run_seed + 2000)
            # Merge CLS stats into the last repair stats entry
            if all_repair_stats:
                all_repair_stats[-1].update({
                    'cls_2opt_moves': cls_stats.get('cls_2opt_moves', 0),
                    'cls_relocate_moves': cls_stats.get('cls_relocate_moves', 0),
                    'cls_total_moves': cls_stats.get('cls_total_moves', 0),
                    'cls_cost_before': cls_stats.get('cls_cost_before', 0),
                    'cls_cost_after': cls_stats.get('cls_cost_after', 0),
                    'cls_cost_reduction': cls_stats.get('cls_cost_reduction', 0),
                    'cls_cost_reduction_pct': cls_stats.get('cls_cost_reduction_pct', 0),
                })

        # ── Step 3.6: Route Merging (reduce multi-trip depot round-trips) ──
        merge_stats = {}
        if use_merge and repaired_sol.tardiness <= _TARD_EPSILON:
            repaired_sol, merge_stats = merge_routes_post_repair(
                repaired_sol, instance, seed=run_seed + 3000)
            if all_repair_stats:
                all_repair_stats[-1].update({
                    'merge_routes_before': merge_stats.get('merge_routes_before', 0),
                    'merge_routes_after': merge_stats.get('merge_routes_after', 0),
                    'merge_routes_reduced': merge_stats.get('merge_routes_reduced', 0),
                    'merge_cost_reduction': merge_stats.get('merge_cost_reduction', 0),
                    'merge_cost_reduction_pct': merge_stats.get('merge_cost_reduction_pct', 0),
                })

        # ── Step 3.7: EV Charging Integration (optional) ──
        ev_sol = None
        ev_stats = {}
        if use_ev and repaired_sol.tardiness <= _TARD_EPSILON:
            from week8.ev.ev_model import (
                EVTruckSolution, insert_charging_stops_lookahead,
                get_charging_station_coords,
            )
            from week8.config import BATTERY_CAPACITY, ENERGY_CONSUMPTION_RATE

            bat_cap = ev_battery_capacity if ev_battery_capacity is not None else BATTERY_CAPACITY

            # Insert charging stops with look-ahead + TW-aware
            routes_with_cs, cs_stats = insert_charging_stops_lookahead(
                repaired_sol.truck_routes,
                instance['customers'],
                instance['distance_matrix'],
                instance,
                battery_capacity=bat_cap,
                energy_rate=ENERGY_CONSUMPTION_RATE,
                charging_model=ev_charging_model,
                check_tw=True,
            )

            # Build EV solution
            ev_sol = EVTruckSolution(
                routes_with_cs, instance,
                charging_model=ev_charging_model,
                battery_capacity=bat_cap,
                energy_rate=ENERGY_CONSUMPTION_RATE,
            )

            ev_stats = {
                'ev_feasible': ev_sol.ev_feasible,
                'ev_energy_violation': round(ev_sol.energy_violation, 2),
                'ev_total_energy': round(ev_sol.total_energy, 2),
                'ev_n_charges': ev_sol.n_charges,
                'ev_total_charge_time': round(ev_sol.total_charge_time, 2),
                'ev_total_charge_energy': round(ev_sol.total_charge_energy, 2),
                'ev_charging_model': ev_charging_model,
                'ev_battery_capacity': bat_cap,
                'ev_cs_insertions': cs_stats.get('n_insertions', 0),
                'ev_cs_method': cs_stats.get('method', 'lookahead'),
                'ev_tw_rejections': cs_stats.get('tw_rejections', 0),
                'ev_energy_violation_before': cs_stats.get('energy_violations_before', 0),
            }

            if all_repair_stats:
                all_repair_stats[-1].update(ev_stats)

        # Use EV solution if available (includes charging station nodes),
        # otherwise use the repaired solution
        if use_ev and ev_sol is not None:
            all_solutions.append(ev_sol)
        else:
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
        if 'moves_attempted' in all_repair_stats[0]:
            avg_stats['moves_attempted'] = sum(s.get('moves_attempted', 0) for s in all_repair_stats) / len(all_repair_stats)
        if 'forward_insertion_success' in all_repair_stats[0]:
            avg_stats['forward_insertion_success'] = any(s.get('forward_insertion_success', False) for s in all_repair_stats)
        if 'partial_success' in all_repair_stats[0]:
            avg_stats['partial_success'] = sum(1 for s in all_repair_stats if s.get('partial_success', False)) / len(all_repair_stats)
        # CLS aggregate stats
        if 'cls_total_moves' in all_repair_stats[0]:
            avg_stats['cls_total_moves'] = sum(s.get('cls_total_moves', 0) for s in all_repair_stats) / len(all_repair_stats)
            avg_stats['cls_2opt_moves'] = sum(s.get('cls_2opt_moves', 0) for s in all_repair_stats) / len(all_repair_stats)
            avg_stats['cls_relocate_moves'] = sum(s.get('cls_relocate_moves', 0) for s in all_repair_stats) / len(all_repair_stats)
            avg_stats['cls_cost_reduction'] = sum(s.get('cls_cost_reduction', 0) for s in all_repair_stats) / len(all_repair_stats)
            avg_stats['cls_cost_reduction_pct'] = sum(s.get('cls_cost_reduction_pct', 0) for s in all_repair_stats) / len(all_repair_stats)
        # Merge aggregate stats
        if 'merge_routes_before' in all_repair_stats[0]:
            avg_stats['merge_routes_reduced'] = sum(s.get('merge_routes_reduced', 0) for s in all_repair_stats) / len(all_repair_stats)
            avg_stats['merge_cost_reduction'] = sum(s.get('merge_cost_reduction', 0) for s in all_repair_stats) / len(all_repair_stats)
    else:
        avg_stats = {}

    return {
        'solutions': all_solutions,
        'pareto_front': pareto,
        'mean_runtime': sum(times) / max(len(times), 1),
        'repair_stats': avg_stats,
        **({'ev_stats': ev_stats} if use_ev and ev_stats else {}),
    }


def run_pipeline(instance, n_trucks=2, n_runs=1, seed=42,
                 variant='hybrid', use_repair=True, repair_mode='full', **kwargs):
    """Alias for solve_evrptw, compatible with experiment runners."""
    return solve_evrptw(
        instance, n_trucks=n_trucks, variant=variant,
        use_repair=use_repair, repair_mode=repair_mode,
        n_runs=n_runs, seed=seed, **kwargs)


def solve_evrptw_iterative(instance, n_trucks, variant='budget_aware',
                            use_cls=True, n_restarts=5, min_restarts=3,
                            variance_threshold=0.05,
                            seed=42):
    """
    Multi-start POMO + FI + CLS: run N times with different seeds, pick best.

    PHILOSOPHY (Direction 5):
      POMO is a constructive neural heuristic using randomized trajectories.
      Different seeds produce different routes with different FI move counts,
      leading to different final costs.

      VARIANCE-AWARE EARLY STOPPING (NEW):
      After min_restarts, if the coefficient of variation (std/mean) of costs
      is below variance_threshold, stop early — additional restarts are
      unlikely to find a significantly better solution.

    Args:
        instance: problem instance dict
        n_trucks: number of trucks
        variant: clustering variant
        use_cls: whether to apply CLS after FI
        n_restarts: max number of independent restarts (default 5)
        min_restarts: minimum restarts before considering early stop (default 3)
        variance_threshold: CV threshold for early stop (default 0.05 = 5%)
        seed: base random seed

    Returns:
        dict with best_solution, all_results, best_run, cost_stats
    """
    all_results = []
    best_sol = None
    best_cost = float('inf')
    best_run = -1
    all_costs = []
    stopped_early = False

    for run in range(n_restarts):
        run_seed = seed + run * 100

        result = solve_evrptw(
            instance, n_trucks=n_trucks, variant=variant,
            use_repair=True, repair_mode='forward',
            use_cls=use_cls, use_merge=True, n_runs=1, seed=run_seed)

        if not result['solutions']:
            continue

        sol = result['solutions'][0]
        rs = result['repair_stats']

        if sol.tardiness > 1e-6:
            continue

        all_costs.append(sol.cost)

        run_result = {
            'run': run,
            'cost': sol.cost,
            'tardiness': sol.tardiness,
            'fi_moves': rs.get('moves_accepted', 0),
            'fi_fallback': rs.get('fallback_count', 0),
            'cls_moves': rs.get('cls_total_moves', 0),
        }
        all_results.append(run_result)

        if sol.cost < best_cost - 0.01:
            best_sol = sol
            best_cost = sol.cost
            best_run = run

        # ── Variance-aware early stopping ──
        if len(all_costs) >= min_restarts:
            mean_c = sum(all_costs) / len(all_costs)
            if mean_c > 0.01:
                variance = sum((c - mean_c)**2 for c in all_costs) / len(all_costs)
                cv = (variance ** 0.5) / mean_c  # coefficient of variation
                if cv < variance_threshold:
                    stopped_early = True
                    break

    if all_costs:
        mean_cost = sum(all_costs) / len(all_costs)
        min_cost = min(all_costs)
        max_cost = max(all_costs)
        cost_range = max_cost - min_cost
    else:
        mean_cost = min_cost = max_cost = cost_range = 0

    return {
        'best_solution': best_sol,
        'best_cost': best_cost,
        'best_run': best_run,
        'mean_cost': mean_cost,
        'min_cost': min_cost,
        'max_cost': max_cost,
        'cost_range': cost_range,
        'cost_range_pct': round(cost_range / mean_cost * 100, 1) if mean_cost > 0 else 0,
        'all_results': all_results,
        'n_successful_runs': len(all_costs),
        'stopped_early': stopped_early,
        'n_restarts_used': len(all_costs),
    }
