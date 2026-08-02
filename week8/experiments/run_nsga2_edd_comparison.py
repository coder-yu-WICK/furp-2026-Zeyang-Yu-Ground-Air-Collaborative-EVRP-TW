#!/usr/bin/env python3
"""P0 Experiment: NSGA-II + Full EDD vs POMO + Forward Insertion.

Answers the reviewer's key question: "If NSGA-II + EDD can also achieve
TW feasibility, why do you need POMO?"

Compares three approaches on all 224 instances:
  A: NSGA-II only
  B: NSGA-II + Full EDD repair
  C: POMO + Forward Insertion (our method)
"""
import os, sys, json, time, math, traceback
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from week8.config import (
    TRUCK_FLEET_CONFIGS, RC1_INSTANCES, RC2_INSTANCES,
    R1_INSTANCES, R2_INSTANCES, C1_INSTANCES, C2_INSTANCES,
)
from week8.core.data_loader import load_instance_from_disk
from week8.core.problem_model import TruckSolution

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

ALL_INSTANCES = {
    'RC1': RC1_INSTANCES, 'RC2': RC2_INSTANCES,
    'R1': R1_INSTANCES, 'R2': R2_INSTANCES,
    'C1': C1_INSTANCES, 'C2': C2_INSTANCES,
}
SCALES = [25, 50, 100, 200]
BASE_SEED = 42


def get_n_trucks(scale):
    configs = TRUCK_FLEET_CONFIGS.get(scale, [2])
    return configs[len(configs)//2]


def build_instance_list():
    instances = []
    for tw_type, inst_names in ALL_INSTANCES.items():
        for inst_name in inst_names:
            for scale in SCALES:
                key = f"{inst_name}_{scale}c"
                instances.append({
                    'key': key, 'tw_type': tw_type, 'inst_name': inst_name,
                    'scale': scale, 'n_trucks': get_n_trucks(scale),
                })
    return instances


def nsga2_best_solution(instance, n_trucks, seed=42):
    """Run NSGA-II and return the best solution (with routes)."""
    from week8.algorithms.nsga2 import run_nsga2
    result = run_nsga2(instance, n_trucks=n_trucks, n_runs=1, seed=seed,
                       pop_size=80, n_generations=80)
    solutions = result.get('pareto_front', result.get('solutions', []))
    if not solutions:
        return None
    return min(solutions, key=lambda s: (s.tardiness, s.cost))


def apply_full_edd(solution, instance):
    """Apply Full EDD repair to a solution, return repaired solution."""
    from week8.pipeline.repair import repair_tardiness_truck
    repaired, stats = repair_tardiness_truck(solution, instance, max_iter=200, seed=42)
    return repaired, stats


def main():
    all_instances = build_instance_list()
    total = len(all_instances)

    # Load existing checkpoints
    ckpt_path = os.path.join(RESULTS_DIR, 'exp_nsga2_edd_comparison.json')
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            results = json.load(f)
    else:
        results = {}

    # Load our FI results for comparison
    fi_path = os.path.join(RESULTS_DIR, 'sweep_forward_insertion.json')
    with open(fi_path) as f:
        fi_results = json.load(f)

    print(f"{'='*70}")
    print(f"NSGA-II + Full EDD vs POMO + Forward Insertion — 224 instances")
    print(f"{'='*70}")

    for idx, inst_info in enumerate(all_instances):
        key = inst_info['key']
        if key in results:
            continue

        try:
            instance = load_instance_from_disk(key)
        except FileNotFoundError:
            results[key] = {'error': 'instance_not_found'}
            continue

        n_trucks = inst_info['n_trucks']
        print(f"  [{idx+1}/{total}] {key} ({n_trucks}t) ...", end=' ', flush=True)

        try:
            t0 = time.time()

            # A: NSGA-II only
            nsga2_sol = nsga2_best_solution(instance, n_trucks, seed=BASE_SEED)
            if nsga2_sol is None:
                results[key] = {'error': 'nsga2_no_solution'}
                print("NSGA2 FAIL")
                continue

            # B: NSGA-II + Full EDD repair
            nsga2_edd_sol, edd_stats = apply_full_edd(nsga2_sol, instance)

            # C: Our method (from sweep)
            our_data = fi_results.get(key, {})
            our_cost = our_data.get('new_cost')
            our_tw = our_data.get('new_tw_feasible')

            nsga2_time = time.time() - t0

            results[key] = {
                'tw_type': inst_info['tw_type'],
                'scale': inst_info['scale'],
                'n_trucks': n_trucks,
                # NSGA-II raw
                'nsga2_cost': round(nsga2_sol.cost, 2),
                'nsga2_tardiness': round(nsga2_sol.tardiness, 2),
                'nsga2_tw_feasible': nsga2_sol.tardiness <= 1e-6,
                'nsga2_n_routes': len(nsga2_sol.truck_routes),
                # NSGA-II + Full EDD
                'nsga2_edd_cost': round(nsga2_edd_sol.cost, 2),
                'nsga2_edd_tardiness': round(nsga2_edd_sol.tardiness, 2),
                'nsga2_edd_tw_feasible': nsga2_edd_sol.tardiness <= 1e-6,
                'nsga2_edd_tard_reduction': round(edd_stats.get('tardiness_reduction', 0), 2),
                # POMO + Forward Insertion
                'pomo_fi_cost': our_cost,
                'pomo_fi_tw_feasible': our_tw,
                # Meta
                'runtime': round(nsga2_time, 2),
            }

            nsga2_tw = '✓' if results[key]['nsga2_tw_feasible'] else '✗'
            nsga2_edd_tw = '✓' if results[key]['nsga2_edd_tw_feasible'] else '✗'
            our_tw_str = '✓' if our_tw else '✗'
            print(f"NSGA2: cost={nsga2_sol.cost:.0f} TW={nsga2_tw} | "
                  f"+EDD: cost={nsga2_edd_sol.cost:.0f} TW={nsga2_edd_tw} | "
                  f"Ours: cost={our_cost} TW={our_tw_str} | "
                  f"{nsga2_time:.1f}s")

        except Exception as e:
            print(f"ERROR: {e}")
            results[key] = {'error': str(e), 'traceback': traceback.format_exc()}

        # Checkpoint every 10
        if (idx + 1) % 10 == 0:
            with open(ckpt_path + '.tmp', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            os.replace(ckpt_path + '.tmp', ckpt_path)
            print(f"    [checkpoint: {idx+1}/{total}]")

    # Final save
    with open(ckpt_path + '.tmp', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    os.replace(ckpt_path + '.tmp', ckpt_path)

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"SUMMARY: NSGA-II + EDD vs POMO + Forward Insertion")
    print(f"{'='*70}")

    nsga2_tw = sum(1 for v in results.values() if v.get('nsga2_tw_feasible', False))
    nsga2_edd_tw = sum(1 for v in results.values() if v.get('nsga2_edd_tw_feasible', False))
    pomo_fi_tw = sum(1 for v in results.values() if v.get('pomo_fi_tw_feasible', False))
    total_ok = sum(1 for v in results.values() if 'error' not in v)

    print(f"  Valid instances: {total_ok}")
    print(f"  NSGA-II only:            {nsga2_tw}/{total_ok} TW feasible ({nsga2_tw/max(total_ok,1)*100:.0f}%)")
    print(f"  NSGA-II + Full EDD:      {nsga2_edd_tw}/{total_ok} TW feasible ({nsga2_edd_tw/max(total_ok,1)*100:.0f}%)")
    print(f"  POMO + Forward Insertion: {pomo_fi_tw}/{total_ok} TW feasible ({pomo_fi_tw/max(total_ok,1)*100:.0f}%)")

    # Cost comparison
    nsga2_costs = [v['nsga2_cost'] for v in results.values() if 'error' not in v]
    nsga2_edd_costs = [v['nsga2_edd_cost'] for v in results.values() if 'error' not in v]
    pomo_fi_costs = [v['pomo_fi_cost'] for v in results.values()
                     if 'error' not in v and v.get('pomo_fi_cost') is not None]

    if nsga2_costs:
        print(f"\n  Avg Cost:")
        print(f"    NSGA-II only:            {sum(nsga2_costs)/len(nsga2_costs):.0f}")
        print(f"    NSGA-II + Full EDD:      {sum(nsga2_edd_costs)/len(nsga2_edd_costs):.0f}")
        if pomo_fi_costs:
            print(f"    POMO + Forward Insertion: {sum(pomo_fi_costs)/len(pomo_fi_costs):.0f}")

    print(f"\nResults saved to: {ckpt_path}")


if __name__ == '__main__':
    main()
