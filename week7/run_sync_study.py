#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 7 — Model D Synchronization Study.

Compares Model C (no sync) vs Model D (sync-aware) across all Solomon types.

Model C (No Sync):
  - Drone insertion: hard GO/NO-GO filter (reject if drone too slow)
  - Evaluation: sync is a soft 0.01-weighted hint, no truck waiting

Model D (With Sync):
  - Drone insertion: allows truck waiting up to max_wait_penalty
  - Evaluation: truck WAITS at recovery node, cascading delays tracked

Metrics compared:
  - sync_wait_time: total truck waiting time for drones
  - cascaded_tardiness: tardiness caused by sync waiting
  - n_drone_missions: number of feasible drone missions
  - cost: total cost (Model D may be higher due to waiting penalty)
  - makespan: max route completion time
  - drone_utilization: % customers served by drone

Usage:
  python week7/run_sync_study.py --test          # Single instance quick test
  python week7/run_sync_study.py --quick          # 50c only, 1 rep
  python week7/run_sync_study.py                  # Full study (50c/100c/200c)
  python week7/run_sync_study.py --instance RC101_100c
"""

import json
import os
import sys
import time
import math
import copy
import argparse
from datetime import datetime
from collections import defaultdict

# ── Path setup ──
_W7 = os.path.dirname(os.path.abspath(__file__))
_W6 = os.path.join(_W7, '..', 'week6')
_W5 = os.path.join(_W7, '..', 'week5')
_W4 = os.path.join(_W7, '..', 'week4')

for p in [_W5, _W4, _W6, _W7]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import (
    RC1_INSTANCES, RC2_INSTANCES, R1_INSTANCES, R2_INSTANCES,
    C1_INSTANCES, C2_INSTANCES, CUSTOMER_SIZES, TW_TYPES,
)
from utils.data_loader import load_instance_from_disk, build_all_instances
from utils.problem_model import TruckDroneSolution

# Week 5 drone insertion (original, with hard GO/NO-GO filter)
from drone_post_processing import insert_cross_route_drones

# Week 7 sync module
from sync_evaluator import (
    evaluate_sync_aware,
    evaluate_no_sync,
    insert_drones_sync_aware,
    compare_sync_vs_nosync,
)

# Pipeline
from pipeline import run_pipeline
from pomo_mt_improved import run_pomo_improved


# ═══════════════════════════════════════════════════════════════════════════
# Config Builder
# ═══════════════════════════════════════════════════════════════════════════

def build_sync_configs(sizes=None, types=None):
    """Build configs for sync study."""
    if sizes is None:
        sizes = [50, 100, 200]
    if types is None:
        types = ['RC1', 'RC2', 'R1', 'R2', 'C1', 'C2']

    all_instance_lists = {
        'RC1': RC1_INSTANCES, 'RC2': RC2_INSTANCES,
        'R1': R1_INSTANCES, 'R2': R2_INSTANCES,
        'C1': C1_INSTANCES, 'C2': C2_INSTANCES,
    }

    configs = []
    for tw_type in types:
        for src_inst in all_instance_lists.get(tw_type, []):
            for nc in sizes:
                instance_key = f'{src_inst}_{nc}c'
                try:
                    load_instance_from_disk(instance_key)
                except FileNotFoundError:
                    continue

                if nc <= 25:
                    n_trucks = 2
                elif nc <= 50:
                    n_trucks = 4
                else:
                    n_trucks = 6

                repair_mode = 'partial' if nc <= 50 else 'full'

                configs.append({
                    'instance_key': instance_key,
                    'source_instance': src_inst,
                    'n_customers': nc,
                    'tw_type': tw_type,
                    'n_trucks': n_trucks,
                    'repair_mode': repair_mode,
                })
    return configs


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline Runners
# ═══════════════════════════════════════════════════════════════════════════

def run_model_c(instance, n_trucks, seed, n_drones_per_truck=2):
    """
    Model C: Current pipeline with hard GO/NO-GO sync filter.

    This is the existing pipeline: POMO → repair → drones (standard insertion)
    → standard evaluation (soft sync).
    """
    import random
    random.seed(seed)

    tw_type = instance.get('tw_type', '')
    use_cw_savings = tw_type.startswith('C') or tw_type.startswith('R1')

    # Step 1: Construction
    if use_cw_savings:
        from clustering_baselines import clarke_wright_savings
        cw_sol = clarke_wright_savings(instance, n_trucks, seed=seed)
        w5_result = {'solutions': [cw_sol], 'pareto_front': [cw_sol]}
    else:
        w5_result = run_pomo_improved(
            instance, n_runs=1, n_trucks=n_trucks,
            endurance='medium', seed=seed,
            variant='hybrid', tw_beta=0.4,
            check_tw_feasibility=True,
        )

    if not w5_result.get('solutions'):
        return _empty_sync_result(instance, n_trucks, 'model_c')

    pareto = w5_result.get('pareto_front', w5_result['solutions'])
    initial_sol = min(pareto, key=lambda s: (s.tardiness, s.cost))

    # Step 2: Capacity repair
    working_sol = initial_sol
    from repair import repair_capacity
    cap_sol, cap_stats = repair_capacity(
        initial_sol, instance, max_iter=200, seed=seed + 500)
    if cap_stats['capacity_violation_before'] > 0.01:
        working_sol = cap_sol

    # Step 3: EDD Repair
    if working_sol.tardiness > 1e-6:
        from repair import (repair_tardiness, repair_tardiness_partial,
                           repair_inter_route)
        if instance['n_customers'] <= 50:
            repaired_sol, stats = repair_tardiness_partial(
                working_sol, instance, seed=seed + 1000, max_drones_per_truck=0)
        else:
            repaired_sol, stats = repair_tardiness(
                working_sol, instance, max_iter=500, seed=seed + 1000,
                max_drones_per_truck=0)
        if repaired_sol.tardiness > 1e-6:
            repaired_sol, ir_stats = repair_inter_route(
                repaired_sol, instance, max_iter=200, seed=seed + 2000,
                max_drones_per_truck=0)
        working_sol = repaired_sol

    # Step 4: Standard drone insertion (hard GO/NO-GO filter)
    if n_drones_per_truck > 0:
        pre_drone_sol = copy.deepcopy(working_sol)

        new_routes, drone_missions, saved, n_drones, drone_counts = \
            insert_cross_route_drones(
                working_sol.truck_routes, instance,
                drone_endurance=4.0,
                max_drones_per_truck=n_drones_per_truck,
                min_saving=0.5)

        sol = TruckDroneSolution(new_routes, drone_missions, instance,
                                 max_drones_per_truck=n_drones_per_truck)

        # Post-drone EDD reordering
        custs = instance['customers']
        new_routes_sorted = []
        for route in sol.truck_routes:
            if len(route) <= 1:
                new_routes_sorted.append(route)
            else:
                new_routes_sorted.append(
                    sorted(route, key=lambda cid: custs[cid - 1]['due_time']))
        sol = TruckDroneSolution(new_routes_sorted, sol.drone_missions,
                                 instance, max_drones_per_truck=n_drones_per_truck)

        # Fallback check
        if sol.tardiness > 0 and pre_drone_sol.feasible:
            if (sol.cost + sol.tardiness) > (pre_drone_sol.cost + pre_drone_sol.tardiness):
                sol = pre_drone_sol
        if not sol.feasible and pre_drone_sol.feasible:
            sol = pre_drone_sol
    else:
        sol = working_sol

    # Evaluate with no-sync evaluator
    result = evaluate_no_sync(sol, instance)
    result['model'] = 'C (No Sync)'
    result['n_drone_missions'] = len(sol.drone_missions)
    return result


def run_model_d(instance, n_trucks, seed, n_drones_per_truck=2,
                max_wait_penalty=60.0):
    """
    Model D: Sync-aware pipeline with truck waiting allowed.

    Pipeline: POMO → repair → sync-aware drone insertion
    → sync-aware evaluation (truck waiting, cascading delays).
    """
    import random
    random.seed(seed)

    tw_type = instance.get('tw_type', '')
    use_cw_savings = tw_type.startswith('C') or tw_type.startswith('R1')

    # Step 1: Construction (same as Model C)
    if use_cw_savings:
        from clustering_baselines import clarke_wright_savings
        cw_sol = clarke_wright_savings(instance, n_trucks, seed=seed)
        w5_result = {'solutions': [cw_sol], 'pareto_front': [cw_sol]}
    else:
        w5_result = run_pomo_improved(
            instance, n_runs=1, n_trucks=n_trucks,
            endurance='medium', seed=seed,
            variant='hybrid', tw_beta=0.4,
            check_tw_feasibility=True,
        )

    if not w5_result.get('solutions'):
        return _empty_sync_result(instance, n_trucks, 'model_d')

    pareto = w5_result.get('pareto_front', w5_result['solutions'])
    initial_sol = min(pareto, key=lambda s: (s.tardiness, s.cost))

    # Step 2: Capacity repair
    working_sol = initial_sol
    from repair import repair_capacity
    cap_sol, cap_stats = repair_capacity(
        initial_sol, instance, max_iter=200, seed=seed + 500)
    if cap_stats['capacity_violation_before'] > 0.01:
        working_sol = cap_sol

    # Step 3: EDD Repair (same as Model C)
    if working_sol.tardiness > 1e-6:
        from repair import (repair_tardiness, repair_tardiness_partial,
                           repair_inter_route)
        if instance['n_customers'] <= 50:
            repaired_sol, stats = repair_tardiness_partial(
                working_sol, instance, seed=seed + 1000, max_drones_per_truck=0)
        else:
            repaired_sol, stats = repair_tardiness(
                working_sol, instance, max_iter=500, seed=seed + 1000,
                max_drones_per_truck=0)
        if repaired_sol.tardiness > 1e-6:
            repaired_sol, ir_stats = repair_inter_route(
                repaired_sol, instance, max_iter=200, seed=seed + 2000,
                max_drones_per_truck=0)
        working_sol = repaired_sol

    # Step 4: Sync-aware drone insertion (allows truck waiting)
    if n_drones_per_truck > 0:
        pre_drone_sol = copy.deepcopy(working_sol)

        new_routes, drone_missions, saved, n_drones, drone_counts, sync_stats = \
            insert_drones_sync_aware(
                working_sol.truck_routes, instance,
                drone_endurance=4.0,
                max_drones_per_truck=n_drones_per_truck,
                min_saving=0.5,
                max_wait_penalty=max_wait_penalty)

        sol = TruckDroneSolution(new_routes, drone_missions, instance,
                                 max_drones_per_truck=n_drones_per_truck)

        # Post-drone EDD reordering — preserve drone launch/recovery node
        # adjacency. EDD reordering can separate i and k, making the truck
        # segment time artificially long (drone flies direct, truck goes
        # through many intermediate nodes).
        custs = instance['customers']
        # Collect drone launch and recovery nodes
        drone_nodes = set()
        for m in drone_missions:
            i, k = m[0], m[2]
            if i > 0:
                drone_nodes.add(i)
            if k > 0:
                drone_nodes.add(k)

        new_routes_sorted = []
        for route in sol.truck_routes:
            if len(route) <= 1:
                new_routes_sorted.append(list(route))
            elif any(n in drone_nodes for n in route):
                # Route contains drone launch/recovery nodes — don't reorder
                new_routes_sorted.append(list(route))
            else:
                new_routes_sorted.append(
                    sorted(route, key=lambda cid: custs[cid - 1]['due_time']))
        sol = TruckDroneSolution(new_routes_sorted, sol.drone_missions,
                                 instance, max_drones_per_truck=n_drones_per_truck)

        # Fallback check using sync-aware evaluation
        sync_eval = evaluate_sync_aware(sol, instance)
        pre_eval = evaluate_sync_aware(pre_drone_sol, instance)

        if sync_eval['feasible'] and pre_eval['feasible']:
            if sync_eval['cost'] > pre_eval['cost']:
                sol = pre_drone_sol
        elif not sync_eval['feasible'] and pre_eval['feasible']:
            sol = pre_drone_sol
    else:
        sol = working_sol

    # Evaluate with sync-aware evaluator
    result = evaluate_sync_aware(sol, instance)
    result['model'] = 'D (Sync-Aware)'
    result['n_drone_missions'] = len(sol.drone_missions)

    # Also compute no-sync evaluation for comparison
    nosync_eval = evaluate_no_sync(sol, instance)
    result['nosync_cost'] = nosync_eval['cost']
    result['nosync_tardiness'] = nosync_eval['tardiness']
    result['nosync_feasible'] = nosync_eval['feasible']

    return result


def _empty_sync_result(instance, n_trucks, model):
    """Return empty result for failed runs."""
    return {
        'cost': 1e9,
        'tardiness': 1e9,
        'feasible': False,
        'violations': {},
        'drone_util': {'n_drones_used': 0, 'n_drone_customers': 0},
        'sync_wait_time': 0.0,
        'cascaded_tardiness': 0.0,
        'makespan': 0.0,
        'per_route_sync_wait': [],
        'sync_details': [],
        'model': model,
        'n_drone_missions': 0,
        'total_truck_dist': 0.0,
        'total_drone_dist': 0.0,
        'n_drones_used': 0,
        'n_trucks': n_trucks,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════

def print_comparison_table(results_by_instance):
    """Print a side-by-side comparison table of Model C vs Model D."""
    print(f"\n{'=' * 120}")
    print(f"  MODEL C vs MODEL D — SYNCHRONIZATION ABLATION STUDY")
    print(f"{'=' * 120}")

    print(f"\n  {'Instance':<18s} "
          f"{'Model':<18s} "
          f"{'Cost':>10s} {'Tard':>10s} {'Feas':>7s} "
          f"{'SyncWait':>10s} {'#Drones':>8s} {'Makespan':>10s} "
          f"{'ΔCost':>10s}")
    print(f"  {'─' * 110}")

    for r in results_by_instance:
        c = r['model_c']
        d = r['model_d']
        ikey = r['instance_key']

        feas_c = "✓" if c['feasible'] else "✗"
        feas_d = "✓" if d['feasible'] else "✗"

        print(f"  {ikey:<18s} "
              f"{'C (No Sync)':<18s} "
              f"{c['cost']:>10.1f} {c['tardiness']:>10.1f} {feas_c:>7s} "
              f"{c['sync_wait_time']:>10.1f} {c.get('n_drone_missions', 0):>8d} "
              f"{c['makespan']:>10.1f} {'—':>10s}")

        delta_cost = d['cost'] - c['cost']
        delta_str = f"{delta_cost:+.1f}"

        print(f"  {'':18s} "
              f"{'D (Sync-Aware)':<18s} "
              f"{d['cost']:>10.1f} {d['tardiness']:>10.1f} {feas_d:>7s} "
              f"{d['sync_wait_time']:>10.1f} {d.get('n_drone_missions', 0):>8d} "
              f"{d['makespan']:>10.1f} {delta_str:>10s}")
        print(f"  {'─' * 110}")


def print_sync_details(results_by_instance):
    """Print per-instance sync details for Model D."""
    print(f"\n{'=' * 100}")
    print(f"  MODEL D — SYNC DETAILS (missions requiring truck waiting)")
    print(f"{'=' * 100}")

    n_with_wait = 0
    total_wait = 0.0

    for r in results_by_instance:
        d = r['model_d']
        wait = d['sync_wait_time']
        cascaded = d.get('cascaded_tardiness', 0.0)
        n_missions = d.get('n_drone_missions', 0)

        if wait > 0.01 or cascaded > 0.01:
            n_with_wait += 1
            total_wait += wait
            print(f"  {r['instance_key']:<20s} "
                  f"wait={wait:.1f}min  cascaded_tard={cascaded:.1f}  "
                  f"n_drones={n_missions}")

    print(f"\n  Summary: {n_with_wait}/{len(results_by_instance)} instances "
          f"have non-zero sync wait time. Total wait: {total_wait:.1f} min.")


def print_aggregate_summary(results_by_instance):
    """Print aggregate summary by type and size."""
    print(f"\n{'=' * 100}")
    print(f"  AGGREGATE SUMMARY BY TYPE AND SIZE")
    print(f"{'=' * 100}")

    groups = defaultdict(list)
    for r in results_by_instance:
        key = (r['tw_type'], r['n_customers'])
        groups[key].append(r)

    for (tw_type, nc), results in sorted(groups.items()):
        n = len(results)

        c_costs = [r['model_c']['cost'] for r in results if r['model_c']['cost'] < 1e8]
        d_costs = [r['model_d']['cost'] for r in results if r['model_d']['cost'] < 1e8]
        c_tards = [r['model_c']['tardiness'] for r in results]
        d_tards = [r['model_d']['tardiness'] for r in results]
        d_waits = [r['model_d']['sync_wait_time'] for r in results]
        d_drones = [r['model_d'].get('n_drone_missions', 0) for r in results]
        c_drones = [r['model_c'].get('n_drone_missions', 0) for r in results]

        avg_c_cost = sum(c_costs) / len(c_costs) if c_costs else 0
        avg_d_cost = sum(d_costs) / len(d_costs) if d_costs else 0
        avg_d_wait = sum(d_waits) / len(d_waits) if d_waits else 0
        avg_c_drones = sum(c_drones) / len(c_drones) if c_drones else 0
        avg_d_drones = sum(d_drones) / len(d_drones) if d_drones else 0

        print(f"\n  {tw_type} {nc}c ({n} instances):")
        print(f"    Model C:  avg cost={avg_c_cost:.1f}  "
              f"avg tard={sum(c_tards)/len(c_tards):.1f}  "
              f"avg drones={avg_c_drones:.1f}")
        print(f"    Model D:  avg cost={avg_d_cost:.1f}  "
              f"avg tard={sum(d_tards)/len(d_tards):.1f}  "
              f"avg drones={avg_d_drones:.1f}  "
              f"avg sync_wait={avg_d_wait:.1f}")
        print(f"    Δ:        cost={avg_d_cost - avg_c_cost:+.1f}  "
              f"drones={avg_d_drones - avg_c_drones:+.1f}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Model D Synchronization Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--test', action='store_true',
                       help='Single instance quick test')
    parser.add_argument('--quick', action='store_true',
                       help='50c only, 1 instance per type')
    parser.add_argument('--instance', type=str,
                       help='Run on a specific instance')
    parser.add_argument('--types', type=str, nargs='+',
                       choices=['RC1', 'RC2', 'R1', 'R2', 'C1', 'C2'],
                       help='Solomon types to run')
    parser.add_argument('--sizes', type=int, nargs='+',
                       default=[50, 100, 200],
                       help='Customer sizes')
    parser.add_argument('--max-wait', type=float, default=60.0,
                       help='Max truck wait penalty (minutes) for Model D')
    parser.add_argument('--output-dir', type=str,
                       default=os.path.join(_W7, 'results'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ── Build configs ──
    if args.test:
        configs = [{
            'instance_key': 'RC201_50c', 'source_instance': 'RC201',
            'n_customers': 50, 'tw_type': 'RC2', 'n_trucks': 4,
            'repair_mode': 'partial',
        }]
    elif args.quick:
        configs = build_sync_configs(
            sizes=[50],
            types=args.types or ['RC1', 'RC2', 'R1', 'R2', 'C1', 'C2'])
        # Only first source instance per type for quick run
        seen_types = set()
        filtered = []
        for c in configs:
            if c['tw_type'] not in seen_types:
                seen_types.add(c['tw_type'])
                filtered.append(c)
        configs = filtered
    elif args.instance:
        parts = args.instance.rsplit('_', 1)
        src = parts[0]
        nc = int(parts[1].replace('c', ''))
        for twt, inst_list in {
            'RC1': RC1_INSTANCES, 'RC2': RC2_INSTANCES,
            'R1': R1_INSTANCES, 'R2': R2_INSTANCES,
            'C1': C1_INSTANCES, 'C2': C2_INSTANCES,
        }.items():
            if src in inst_list:
                tw_type = twt
                break
        else:
            tw_type = 'RC2'
        configs = [{
            'instance_key': args.instance, 'source_instance': src,
            'n_customers': nc, 'tw_type': tw_type,
            'n_trucks': 2 if nc <= 25 else (4 if nc <= 50 else 6),
            'repair_mode': 'partial' if nc <= 50 else 'full',
        }]
    else:
        configs = build_sync_configs(
            sizes=args.sizes, types=args.types)

    # ── Pre-build instances ──
    build_all_instances()

    print(f"\n{'█' * 100}")
    print(f"  MODEL D — SYNCHRONIZATION ABLATION STUDY")
    print(f"  Comparing: Model C (No Sync) vs Model D (Sync-Aware)")
    print(f"  Configs: {len(configs)}")
    print(f"  Max wait penalty: {args.max_wait} min")
    print(f"{'█' * 100}")

    # ── Run comparison ──
    results_by_instance = []

    for idx, cfg in enumerate(configs):
        inst = load_instance_from_disk(cfg['instance_key'])
        print(f"\n[{idx + 1}/{len(configs)}] {cfg['instance_key']} "
              f"({cfg['tw_type']}, {cfg['n_customers']}c, {cfg['n_trucks']} trucks)")

        # Model C
        t0 = time.time()
        result_c = run_model_c(inst, cfg['n_trucks'], args.seed,
                               n_drones_per_truck=2)
        t_c = time.time() - t0
        print(f"  Model C: cost={result_c['cost']:.1f}  "
              f"tard={result_c['tardiness']:.1f}  "
              f"feas={result_c['feasible']}  "
              f"drones={result_c.get('n_drone_missions', 0)}  "
              f"t={t_c:.1f}s")

        # Model D
        t0 = time.time()
        result_d = run_model_d(inst, cfg['n_trucks'], args.seed,
                               n_drones_per_truck=2,
                               max_wait_penalty=args.max_wait)
        t_d = time.time() - t0
        print(f"  Model D: cost={result_d['cost']:.1f}  "
              f"tard={result_d['tardiness']:.1f}  "
              f"feas={result_d['feasible']}  "
              f"drones={result_d.get('n_drone_missions', 0)}  "
              f"sync_wait={result_d['sync_wait_time']:.1f}  "
              f"t={t_d:.1f}s")

        # Delta
        delta_cost = result_d['cost'] - result_c['cost']
        delta_drones = result_d.get('n_drone_missions', 0) - result_c.get('n_drone_missions', 0)
        print(f"  Δ: cost={delta_cost:+.1f}  drones={delta_drones:+d}  "
              f"sync_wait={result_d['sync_wait_time']:.1f}")

        results_by_instance.append({
            'instance_key': cfg['instance_key'],
            'source_instance': cfg['source_instance'],
            'n_customers': cfg['n_customers'],
            'tw_type': cfg['tw_type'],
            'n_trucks': cfg['n_trucks'],
            'model_c': result_c,
            'model_d': result_d,
            'runtime_c': t_c,
            'runtime_d': t_d,
        })

    # ── Print reports ──
    print_comparison_table(results_by_instance)
    print_sync_details(results_by_instance)
    print_aggregate_summary(results_by_instance)

    # ── Generate summary statistics ──
    print(f"\n{'=' * 100}")
    print(f"  KEY FINDINGS")
    print(f"{'=' * 100}")

    valid = [r for r in results_by_instance
             if r['model_c']['cost'] < 1e8 and r['model_d']['cost'] < 1e8]

    if valid:
        n_with_wait = sum(1 for r in valid if r['model_d']['sync_wait_time'] > 0.01)
        n_more_drones = sum(1 for r in valid
                          if r['model_d'].get('n_drone_missions', 0) >
                          r['model_c'].get('n_drone_missions', 0))
        n_fewer_drones = sum(1 for r in valid
                           if r['model_d'].get('n_drone_missions', 0) <
                           r['model_c'].get('n_drone_missions', 0))
        avg_wait = sum(r['model_d']['sync_wait_time'] for r in valid) / len(valid)
        avg_cascaded = sum(r['model_d'].get('cascaded_tardiness', 0)
                          for r in valid) / len(valid)

        print(f"  Instances with sync waiting:          {n_with_wait}/{len(valid)}")
        print(f"  Avg sync wait time (Model D):         {avg_wait:.2f} min")
        print(f"  Avg cascaded tardiness (Model D):     {avg_cascaded:.2f}")
        print(f"  Instances with MORE drones in D:      {n_more_drones}")
        print(f"  Instances with FEWER drones in D:     {n_fewer_drones}")
        print(f"  Instances with SAME drones:           {len(valid) - n_more_drones - n_fewer_drones}")

        # By type analysis
        print(f"\n  Sync wait by type:")
        by_type = defaultdict(list)
        for r in valid:
            by_type[r['tw_type']].append(r['model_d']['sync_wait_time'])
        for twt in sorted(by_type):
            waits = by_type[twt]
            print(f"    {twt}: mean={sum(waits)/len(waits):.2f}  "
                  f"max={max(waits):.2f}  n>0={sum(1 for w in waits if w > 0.01)}/{len(waits)}")

    # ── Save results ──
    output_path = os.path.join(args.output_dir, f'sync_study_{timestamp}.json')
    with open(output_path, 'w') as f:
        json.dump(results_by_instance, f, indent=2, default=str)

    print(f"\n  Results saved to: {output_path}")
    print(f"{'█' * 100}")


if __name__ == '__main__':
    main()
