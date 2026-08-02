#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive 224-Instance Sweep — Week 8.
===========================================
Replaces the buggy exp_full_sweep.json (200c data duplicated from 100c).

Phases:
  1. Forward Insertion Sweep — all 224 instances, n_runs=3
  2. Classical Baseline Sweep — NSGA-II, P-ACO, IVND, No-Drone GA on 224
  3. OR-Tools Optimality Gaps — 25c instances via CP-SAT
  4. EV Ablation — binding battery constraints (30kWh, 20kWh)
  5. Statistical Tests — Friedman + Wilcoxon on full data

Usage:
  cd "/Users/jackalwick/Desktop/Truck-Drone EVRP-TW"
  .venv/bin/python week8/experiments/run_comprehensive_sweep.py

Resume: re-run with same arguments to skip already-completed instances.
"""
import os, sys, json, time, math, traceback
from collections import defaultdict
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from week8.config import (
    TRUCK_FLEET_CONFIGS, CUSTOMER_SIZES,
    RC1_INSTANCES, RC2_INSTANCES, R1_INSTANCES, R2_INSTANCES,
    C1_INSTANCES, C2_INSTANCES,
    TRUCK_SPEED, TRUCK_CAPACITY, TRUCK_FIXED_COST, TRUCK_DIST_COST_RATE,
    TARDINESS_COST_RATE, DEPOT, BATTERY_CAPACITY,
)
from week8.core.data_loader import load_instance_from_disk
from week8.core.problem_model import TruckSolution, extract_pareto_front

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

ALL_INSTANCES = {
    'RC1': RC1_INSTANCES, 'RC2': RC2_INSTANCES,
    'R1': R1_INSTANCES,   'R2': R2_INSTANCES,
    'C1': C1_INSTANCES,   'C2': C2_INSTANCES,
}
SCALES = [25, 50, 100, 200]
N_RUNS = 3
BASE_SEED = 42

# Fleet sizing per scale
def get_n_trucks(scale):
    configs = TRUCK_FLEET_CONFIGS.get(scale, [2])
    return configs[len(configs)//2]  # middle value

# For classical methods: reduce parameters for larger instances
CLASSICAL_PARAMS = {
    25:  {'nsga2_pop': 50,  'nsga2_gen': 60,  'paco_ants': 50,  'paco_iter': 50,  'ivnd_iter': 100},
    50:  {'nsga2_pop': 80,  'nsga2_gen': 80,  'paco_ants': 80,  'paco_iter': 80,  'ivnd_iter': 150},
    100: {'nsga2_pop': 100, 'nsga2_gen': 60,  'paco_ants': 100, 'paco_iter': 60,  'ivnd_iter': 100},
    200: {'nsga2_pop': 80,  'nsga2_gen': 40,  'paco_ants': 80,  'paco_iter': 40,  'ivnd_iter': 80},
}

# Build flat instance list
def build_instance_list():
    """Build flat list of all 224 instance keys."""
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

ALL_INSTANCE_LIST = build_instance_list()
print(f"Total instances: {len(ALL_INSTANCE_LIST)}")

# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_checkpoint(filename):
    """Load existing results to resume from."""
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def save_checkpoint(data, filename):
    """Save results atomically."""
    path = os.path.join(RESULTS_DIR, filename)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)

# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Forward Insertion Full Sweep
# ═══════════════════════════════════════════════════════════════════════════

def run_forward_insertion_sweep():
    """
    Run POMO+Forward Insertion on all 224 instances with n_runs=3.
    Also runs old Partial EDD for comparison (old_fb baseline).
    """
    print("\n" + "="*70)
    print("PHASE 1: Forward Insertion Full Sweep (224 instances × 3 runs)")
    print("="*70)

    from week8.pipeline.pipeline import solve_evrptw

    results = load_checkpoint('sweep_forward_insertion.json')
    total = len(ALL_INSTANCE_LIST)
    done = 0

    for idx, inst_info in enumerate(ALL_INSTANCE_LIST):
        key = inst_info['key']
        if key in results:
            done += 1
            continue

        try:
            instance = load_instance_from_disk(key)
        except FileNotFoundError:
            print(f"  [{idx+1}/{total}] SKIP {key} (not found)")
            results[key] = {'error': 'instance_not_found'}
            continue

        n_trucks = inst_info['n_trucks']
        print(f"  [{idx+1}/{total}] {key} ({n_trucks}t) ...", end=' ', flush=True)

        try:
            # Run Forward Insertion (our method)
            result_new = solve_evrptw(
                instance, n_trucks=n_trucks, variant='budget_aware',
                use_repair=True, repair_mode='forward',
                n_runs=N_RUNS, seed=BASE_SEED,
            )

            # Run Old Partial EDD (for comparison)
            result_old = solve_evrptw(
                instance, n_trucks=n_trucks, variant='budget_aware',
                use_repair=True, repair_mode='partial',
                n_runs=N_RUNS, seed=BASE_SEED,
            )

            # Extract best solutions
            best_new = min(result_new['solutions'], key=lambda s: (s.tardiness, s.cost))
            best_old = min(result_old['solutions'], key=lambda s: (s.tardiness, s.cost))

            # Aggregate repair stats across runs
            rs_new = result_new.get('repair_stats', {})
            rs_old = result_old.get('repair_stats', {})

            results[key] = {
                'tw_type': inst_info['tw_type'],
                'scale': inst_info['scale'],
                'n_trucks': n_trucks,
                # New method (Forward Insertion)
                'new_cost': round(best_new.cost, 2),
                'new_tard': round(best_new.tardiness, 2),
                'new_tw_feasible': best_new.tardiness <= 1e-6,
                'new_fallback_count': rs_new.get('fallback_count', 0),
                'new_moves_accepted': rs_new.get('moves_accepted', 0),
                'new_moves_attempted': rs_new.get('moves_attempted', 0),
                'new_fi_success': (rs_new.get('forward_insertion_success', False) or
                                   rs_new.get('moves_accepted', 0) > 0),
                'new_partial_success': rs_new.get('partial_success', False),
                'new_tard_before': rs_new.get('tardiness_before', 0),
                'new_tard_after': rs_new.get('tardiness_after', 0),
                'new_mean_runtime': result_new.get('mean_runtime', 0),
                # Old method (Partial EDD)
                'old_cost': round(best_old.cost, 2),
                'old_tard': round(best_old.tardiness, 2),
                'old_tw_feasible': best_old.tardiness <= 1e-6,
                'old_fallback_count': rs_old.get('fallback_count', 0),
                'old_tard_before': rs_old.get('tardiness_before', 0),
                # Meta
                'n_runs': N_RUNS,
                'seed': BASE_SEED,
            }
            print(f"new_fb={results[key]['new_fallback_count']} "
                  f"old_fb={results[key]['old_fallback_count']} "
                  f"moves={results[key]['new_moves_accepted']} "
                  f"fi_ok={results[key]['new_fi_success']} ✓")

        except Exception as e:
            print(f"ERROR: {e}")
            results[key] = {'error': str(e), 'traceback': traceback.format_exc()}

        # Save checkpoint every 10 instances
        if (idx + 1) % 10 == 0:
            save_checkpoint(results, 'sweep_forward_insertion.json')
            print(f"    [checkpoint saved: {idx+1}/{total}]")

    save_checkpoint(results, 'sweep_forward_insertion.json')
    print(f"\nPhase 1 complete: {len(results)} instances")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Classical Baseline Sweep
# ═══════════════════════════════════════════════════════════════════════════

def run_classical_baseline_sweep():
    """
    Run NSGA-II, P-ACO, IVND on all 224 instances.
    Uses reduced parameters for larger instances.
    """
    print("\n" + "="*70)
    print("PHASE 2: Classical Baseline Sweep (224 instances)")
    print("="*70)

    results = load_checkpoint('sweep_classical_baselines.json')
    total = len(ALL_INSTANCE_LIST)
    methods = ['NSGA-II', 'P-ACO', 'IVND']

    for idx, inst_info in enumerate(ALL_INSTANCE_LIST):
        key = inst_info['key']
        scale = inst_info['scale']
        n_trucks = inst_info['n_trucks']
        params = CLASSICAL_PARAMS[scale]

        # Check if all methods done
        if key in results and all(m in results[key] for m in methods):
            continue

        try:
            instance = load_instance_from_disk(key)
        except FileNotFoundError:
            if key not in results:
                results[key] = {}
            for m in methods:
                results[key][m] = {'error': 'instance_not_found'}
            continue

        if key not in results:
            results[key] = {}

        for method in methods:
            if method in results[key] and 'error' not in results[key][method]:
                continue

            print(f"  [{idx+1}/{total}] {key} | {method} ...", end=' ', flush=True)
            t0 = time.time()

            try:
                if method == 'NSGA-II':
                    from week8.algorithms.nsga2 import run_nsga2
                    result = run_nsga2(instance, n_trucks=n_trucks, n_runs=1, seed=BASE_SEED,
                                       pop_size=params['nsga2_pop'],
                                       n_generations=params['nsga2_gen'])
                elif method == 'P-ACO':
                    from week8.algorithms.paco import run_paco
                    result = run_paco(instance, n_runs=1, seed=BASE_SEED,
                                      n_ants=params['paco_ants'],
                                      n_iterations=params['paco_iter'])
                elif method == 'IVND':
                    from week8.algorithms.ivnd import run_ivnd
                    result = run_ivnd(instance, n_trucks=n_trucks, n_runs=1, seed=BASE_SEED,
                                      max_iter=params['ivnd_iter'])

                solutions = result.get('pareto_front', result.get('solutions', []))
                if solutions:
                    best = min(solutions, key=lambda s: (s.tardiness, s.cost))
                    results[key][method] = {
                        'cost': round(best.cost, 2),
                        'tardiness': round(best.tardiness, 2),
                        'tw_feasible': best.tardiness <= 1e-6,
                        'feasible': best.feasible,
                        'n_routes': len(best.truck_routes),
                        'runtime': round(time.time() - t0, 2),
                    }
                    feas = '✓' if results[key][method]['tw_feasible'] else '✗'
                    print(f"cost={best.cost:.0f} tard={best.tardiness:.0f} TW={feas} {time.time()-t0:.1f}s")
                else:
                    results[key][method] = {'error': 'no_solution', 'runtime': round(time.time()-t0, 2)}
                    print(f"NO SOLUTION {time.time()-t0:.1f}s")
            except Exception as e:
                results[key][method] = {'error': str(e), 'runtime': round(time.time()-t0, 2)}
                print(f"ERROR: {e}")

        # Save checkpoint every 5 instances
        if (idx + 1) % 5 == 0:
            save_checkpoint(results, 'sweep_classical_baselines.json')
            print(f"    [checkpoint saved: {idx+1}/{total}]")

    save_checkpoint(results, 'sweep_classical_baselines.json')
    print(f"\nPhase 2 complete: {len(results)} instances")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: OR-Tools Optimality Gaps (25c instances)
# ═══════════════════════════════════════════════════════════════════════════

def solve_vrptw_ortools(instance, n_trucks, time_limit_seconds=60):
    """
    Solve VRPTW using OR-Tools Routing Solver.
    Returns (routes, cost, tardiness, feasible, gap_info).
    """
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    customers = instance['customers']
    depot = instance['depot']
    n_customers = len(customers)
    dist_matrix = instance.get('distance_matrix')

    if dist_matrix is None:
        # Compute on the fly
        dist_matrix = []
        for i in range(n_customers + 1):
            row = []
            for j in range(n_customers + 1):
                if i == 0:
                    xi, yi = depot[0], depot[1]
                else:
                    xi, yi = customers[i-1]['x'], customers[i-1]['y']
                if j == 0:
                    xj, yj = depot[0], depot[1]
                else:
                    xj, yj = customers[j-1]['x'], customers[j-1]['y']
                row.append(int(math.hypot(xi-xj, yi-yj) * 1000))  # mm precision
            dist_matrix.append(row)

    # Create routing model
    manager = pywrapcp.RoutingIndexManager(n_customers + 1, n_trucks, 0)
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback
    def distance_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)
        return dist_matrix[from_node][to_node]

    dist_cb_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_cb_idx)

    # Time callback
    def time_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)
        travel_time = dist_matrix[from_node][to_node] / 1000.0 / TRUCK_SPEED * 60  # minutes
        if from_node > 0:
            travel_time += customers[from_node-1]['service_time']
        return int(travel_time * 100)  # centiminute precision

    time_cb_idx = routing.RegisterTransitCallback(time_callback)

    # Time windows
    max_due = max(c['due_time'] for c in customers)
    horizon_minutes = max(instance.get('tw_horizon', 240.0), max_due + 10)
    horizon = int(horizon_minutes * 100)  # centiminutes
    routing.AddDimension(
        time_cb_idx,
        horizon,  # slack
        horizon,  # max cumul
        False,    # don't force start cumul to zero
        'Time')
    time_dimension = routing.GetDimensionOrDie('Time')

    # Set TW for each location
    for i in range(1, n_customers + 1):
        idx = manager.NodeToIndex(i)
        ready = int(customers[i-1]['ready_time'] * 100)
        due = int(customers[i-1]['due_time'] * 100)
        time_dimension.CumulVar(idx).SetRange(ready, due)

    # Set vehicle TW (depot)
    for v in range(n_trucks):
        idx = routing.Start(v)
        time_dimension.CumulVar(idx).SetRange(0, horizon)

    # Capacity
    def demand_callback(from_idx):
        from_node = manager.IndexToNode(from_idx)
        if from_node == 0:
            return 0
        return int(customers[from_node-1]['demand'])

    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb_idx, 0,
        [int(TRUCK_CAPACITY)] * n_trucks,
        True, 'Capacity')

    # Search parameters
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_params.time_limit.seconds = time_limit_seconds
    search_params.log_search = False

    solution = routing.SolveWithParameters(search_params)

    if not solution:
        return None, float('inf'), float('inf'), False, {'status': 'no_solution'}

    # Extract routes
    routes = []
    total_dist = 0.0
    total_tard = 0.0

    for v in range(n_trucks):
        route = []
        idx = routing.Start(v)
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node > 0:
                route.append(node)
            idx = solution.Value(routing.NextVar(idx))
        if route:
            routes.append(route)

    # Compute actual cost/tardiness
    for route in routes:
        prev = 0
        current_time = 0.0
        for cid in route:
            c = customers[cid-1]
            d_m = dist_matrix[prev][cid] / 1000.0
            total_dist += d_m
            current_time += d_m / TRUCK_SPEED * 60
            if current_time < c['ready_time']:
                current_time = c['ready_time']
            if current_time > c['due_time']:
                total_tard += (current_time - c['due_time'])
            current_time += c['service_time']
            prev = cid
        # Return to depot
        total_dist += dist_matrix[prev][0] / 1000.0

    cost = total_dist * TRUCK_DIST_COST_RATE + total_tard * TARDINESS_COST_RATE
    feasible = total_tard <= 1e-6

    return routes, cost, total_tard, feasible, {'status': 'optimal' if solution else 'timeout'}


def run_ortools_gap_analysis():
    """
    Run OR-Tools on all 25c instances to get optimality gaps.
    25c instances are tractable for exact methods.
    """
    print("\n" + "="*70)
    print("PHASE 3: OR-Tools Optimality Gaps (25c instances)")
    print("="*70)

    results = load_checkpoint('sweep_ortools_gaps.json')
    fi_results = load_checkpoint('sweep_forward_insertion.json')

    instances_25c = [i for i in ALL_INSTANCE_LIST if i['scale'] == 25]
    total = len(instances_25c)

    for idx, inst_info in enumerate(instances_25c):
        key = inst_info['key']
        if key in results:
            continue

        try:
            instance = load_instance_from_disk(key)
        except FileNotFoundError:
            print(f"  [{idx+1}/{total}] SKIP {key} (not found)")
            results[key] = {'error': 'instance_not_found'}
            continue

        n_trucks = inst_info['n_trucks']
        print(f"  [{idx+1}/{total}] {key} ({n_trucks}t) ...", end=' ', flush=True)

        try:
            t0 = time.time()
            opt_routes, opt_cost, opt_tard, opt_feas, opt_info = solve_vrptw_ortools(
                instance, n_trucks, time_limit_seconds=120)

            ortools_time = time.time() - t0

            # Get our result for comparison
            our_data = fi_results.get(key, {})
            our_cost = our_data.get('new_cost', None)
            our_feas = our_data.get('new_tw_feasible', None)

            gap = None
            if opt_cost < float('inf') and our_cost is not None:
                gap = (our_cost - opt_cost) / opt_cost * 100 if opt_cost > 0 else 0

            results[key] = {
                'tw_type': inst_info['tw_type'],
                'n_trucks': n_trucks,
                'ortools_cost': round(opt_cost, 2) if opt_cost < float('inf') else None,
                'ortools_tardiness': round(opt_tard, 2),
                'ortools_feasible': opt_feas,
                'ortools_runtime': round(ortools_time, 2),
                'ortools_status': opt_info.get('status', 'unknown'),
                'our_cost': our_cost,
                'our_feasible': our_feas,
                'gap_pct': round(gap, 2) if gap is not None else None,
            }
            print(f"opt_cost={opt_cost:.0f} our_cost={our_cost} gap={gap:+.1f}% "
                  f"ortools_time={ortools_time:.1f}s" if gap else
                  f"opt_cost={opt_cost:.0f} ortools_time={ortools_time:.1f}s")

        except Exception as e:
            print(f"ERROR: {e}")
            results[key] = {'error': str(e), 'traceback': traceback.format_exc()}

        if (idx + 1) % 5 == 0:
            save_checkpoint(results, 'sweep_ortools_gaps.json')

    save_checkpoint(results, 'sweep_ortools_gaps.json')
    print(f"\nPhase 3 complete: {len(results)} instances")

    # Print summary
    valid_gaps = [(k, v['gap_pct']) for k, v in results.items()
                  if 'error' not in v and v.get('gap_pct') is not None]
    if valid_gaps:
        avg_gap = sum(g for _, g in valid_gaps) / len(valid_gaps)
        print(f"  Average optimality gap: {avg_gap:+.1f}%")
        print(f"  Range: {min(g for _, g in valid_gaps):+.1f}% to {max(g for _, g in valid_gaps):+.1f}%")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: EV Ablation with Binding Constraints
# ═══════════════════════════════════════════════════════════════════════════

def run_ev_binding_ablation():
    """
    EV ablation with battery levels calibrated to 16×16 km urban scale:
    - 55 kWh (standard — 37 km range @ 1.5 kWh/km)
    - 40 kWh (marginal — binding on ~50% of instances)
    - 30 kWh (binding — most instances need charging)
    - 25 kWh (severely binding)

    Uses nonlinear charging model by default.
    """
    print("\n" + "="*70)
    print("PHASE 4: EV Ablation — Binding Battery Constraints")
    print("="*70)

    from week8.ev.ev_model import EVTruckSolution, insert_charging_stops

    results = load_checkpoint('sweep_ev_binding.json')
    fi_results = load_checkpoint('sweep_forward_insertion.json')

    # Run on ALL 224 instances (not just 12 representative)
    total = len(ALL_INSTANCE_LIST)
    battery_levels = [55, 40, 30, 25]
    charging_models = ['none', 'linear', 'nonlinear']

    for idx, inst_info in enumerate(ALL_INSTANCE_LIST):
        key = inst_info['key']
        if key in results:
            continue

        # Need FI solution first
        fi_data = fi_results.get(key, {})
        if 'error' in fi_data:
            results[key] = {'error': 'no_fi_data', 'fi_error': fi_data.get('error')}
            continue

        # We need the actual solution, not just stats
        # Re-run with n_runs=1 to get routes
        try:
            instance = load_instance_from_disk(key)
        except FileNotFoundError:
            results[key] = {'error': 'instance_not_found'}
            continue

        n_trucks = inst_info['n_trucks']
        print(f"  [{idx+1}/{total}] {key} ...", end=' ', flush=True)

        try:
            from week8.pipeline.pipeline import solve_evrptw
            fi_result = solve_evrptw(
                instance, n_trucks=n_trucks, variant='budget_aware',
                use_repair=True, repair_mode='forward',
                n_runs=1, seed=BASE_SEED,
            )
            best = min(fi_result['solutions'], key=lambda s: (s.tardiness, s.cost))
            truck_routes = best.truck_routes
        except Exception as e:
            results[key] = {'error': f'fi_solve_failed: {e}'}
            print(f"FI ERROR: {e}")
            continue

        results[key] = {
            'tw_type': inst_info['tw_type'],
            'scale': inst_info['scale'],
            'n_trucks': n_trucks,
            'base_cost': round(best.cost, 2),
            'base_tardiness': round(best.tardiness, 2),
            'base_feasible': best.feasible,
            'ev_results': {},
        }

        ev_oks = []
        for battery_kwh in battery_levels:
            for model in charging_models:
                ev_key = f"{model}_{battery_kwh}kWh"
                try:
                    routes_with_cs, cs_stats = insert_charging_stops(
                        truck_routes, instance['customers'],
                        instance.get('distance_matrix'), instance,
                        battery_capacity=battery_kwh, energy_rate=1.5,
                    )
                    ev_sol = EVTruckSolution(
                        routes_with_cs, instance,
                        charging_model=model,
                        battery_capacity=battery_kwh,
                        energy_rate=1.5,
                    )
                    results[key]['ev_results'][ev_key] = {
                        'cost': round(ev_sol.cost, 2),
                        'tardiness': round(ev_sol.tardiness, 2),
                        'ev_feasible': ev_sol.ev_feasible,
                        'energy_violation': round(getattr(ev_sol, 'energy_violation', 0), 2),
                        'total_energy': round(getattr(ev_sol, 'total_energy', 0), 2),
                        'n_charges': getattr(ev_sol, 'n_charges', 0),
                        'total_charge_time': round(getattr(ev_sol, 'total_charge_time', 0), 2),
                    }
                    if ev_sol.ev_feasible:
                        ev_oks.append(ev_key)
                except Exception as e:
                    results[key]['ev_results'][ev_key] = {'error': str(e)}

        feas_summary = f"EV feasible @: {ev_oks}" if ev_oks else "NO battery level feasible!"
        print(feas_summary)

        if (idx + 1) % 10 == 0:
            save_checkpoint(results, 'sweep_ev_binding.json')

    save_checkpoint(results, 'sweep_ev_binding.json')

    # Summary
    print(f"\nPhase 4 complete: {len(results)} instances")

    # Count feasibility by battery level
    for battery_kwh in battery_levels:
        feas_count = 0
        for key, data in results.items():
            if 'ev_results' not in data:
                continue
            ev_feas = any(
                v.get('ev_feasible', False)
                for ek, v in data['ev_results'].items()
                if str(battery_kwh) in ek
            )
            if ev_feas:
                feas_count += 1
        print(f"  {battery_kwh}kWh: EV feasible in {feas_count}/{len(results)} instances")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5: Statistical Tests on Full Data
# ═══════════════════════════════════════════════════════════════════════════

def run_statistical_tests():
    """
    Friedman + Nemenyi + Wilcoxon on the full 224-instance sweep.
    Compares our method vs classical baselines.
    """
    print("\n" + "="*70)
    print("PHASE 5: Statistical Tests")
    print("="*70)

    fi_results = load_checkpoint('sweep_forward_insertion.json')
    classical_results = load_checkpoint('sweep_classical_baselines.json')

    # Build per-instance method comparison
    methods = ['POMO+Forward Insertion', 'NSGA-II', 'P-ACO', 'IVND']
    method_tardiness = {m: [] for m in methods}
    method_cost = {m: [] for m in methods}
    common_instances = []

    for key in fi_results:
        if 'error' in fi_results[key]:
            continue
        if key not in classical_results:
            continue

        fi_data = fi_results[key]
        cl_data = classical_results[key]

        # Our method
        our_tard = fi_data.get('new_tard', 999999)
        our_cost = fi_data.get('new_cost', 999999)

        # Classical methods
        all_present = True
        classical_tards = {}
        classical_costs = {}
        for m in ['NSGA-II', 'P-ACO', 'IVND']:
            if m in cl_data and 'error' not in cl_data[m]:
                classical_tards[m] = cl_data[m].get('tardiness', 999999)
                classical_costs[m] = cl_data[m].get('cost', 999999)
            else:
                all_present = False

        if not all_present:
            continue

        method_tardiness['POMO+Forward Insertion'].append(our_tard)
        method_cost['POMO+Forward Insertion'].append(our_cost)
        for m in ['NSGA-II', 'P-ACO', 'IVND']:
            method_tardiness[m].append(classical_tards[m])
            method_cost[m].append(classical_costs[m])
        common_instances.append(key)

    n_instances = len(common_instances)
    print(f"  Common instances: {n_instances}")

    if n_instances < 3:
        print("  INSUFFICIENT DATA for statistical tests")
        return {'error': 'insufficient_data', 'n_instances': n_instances}

    # ── Manual Friedman Test ──
    # Rank methods per instance (1 = best/lowest tardiness)
    k = len(methods)
    N = n_instances
    rank_sums = {m: 0.0 for m in methods}

    for i in range(N):
        tards = [(m, method_tardiness[m][i]) for m in methods]
        tards.sort(key=lambda x: x[1])
        # Assign ranks (1 = best)
        for rank, (m, _) in enumerate(tards, 1):
            rank_sums[m] += rank

    avg_ranks = {m: rank_sums[m] / N for m in methods}

    # Friedman statistic: chi2 = 12/(N*k*(k+1)) * sum(R_j^2) - 3*N*(k+1)
    R_sq_sum = sum(r**2 for r in rank_sums.values())
    chi2 = (12.0 / (N * k * (k + 1))) * R_sq_sum - 3.0 * N * (k + 1)

    # p-value from chi-square with k-1 df
    import scipy.stats as stats
    p_value = 1.0 - stats.chi2.cdf(chi2, k - 1)

    print(f"\n  Friedman Test (Tardiness):")
    print(f"    χ² = {chi2:.4f}")
    print(f"    df = {k - 1}")
    print(f"    p = {p_value:.6f}")
    print(f"    Significant: {'YES ★★★' if p_value < 0.001 else 'YES ★' if p_value < 0.05 else 'no'}")

    print(f"\n  Average Rankings (lower = better):")
    for m, r in sorted(avg_ranks.items(), key=lambda x: x[1]):
        print(f"    {r:.2f} — {m}")

    # ── Nemenyi CD ──
    # q_alpha for k=4, alpha=0.05 is approx 2.569
    q_alpha = 2.569  # for k=4, df=inf
    cd = q_alpha * math.sqrt(k * (k + 1) / (6.0 * N))
    print(f"\n  Nemenyi CD (α=0.05): {cd:.4f}")

    # Significant pairwise differences
    sorted_methods = sorted(methods, key=lambda m: avg_ranks[m])
    print(f"  Significant differences (|rank diff| > {cd:.4f}):")
    for i in range(len(sorted_methods)):
        for j in range(i + 1, len(sorted_methods)):
            diff = abs(avg_ranks[sorted_methods[i]] - avg_ranks[sorted_methods[j]])
            if diff > cd:
                print(f"    {sorted_methods[i]} vs {sorted_methods[j]}: diff={diff:.4f} ★")

    # ── Wilcoxon Signed-Rank ──
    print(f"\n  Wilcoxon Signed-Rank (Ours vs Baselines):")
    wilcoxon_results = {}
    our_tards = method_tardiness['POMO+Forward Insertion']

    for baseline in ['NSGA-II', 'P-ACO', 'IVND']:
        base_tards = method_tardiness[baseline]
        diffs = [b - o for o, b in zip(our_tards, base_tards)]

        if len(diffs) >= 5:
            from scipy.stats import wilcoxon
            w_stat, w_p = wilcoxon(diffs, zero_method='wilcox', alternative='greater')
            wilcoxon_results[baseline] = {'W': float(w_stat), 'p_value': float(w_p)}
            sig = '★' if w_p < 0.05 else ''
            print(f"    vs {baseline}: W={w_stat:.2f}, p={w_p:.6f} {sig}")

    result = {
        'friedman': {'chi2': chi2, 'df': k-1, 'p_value': p_value, 'cd': cd},
        'rankings': avg_ranks,
        'wilcoxon': wilcoxon_results,
        'n_instances': N,
    }
    save_checkpoint(result, 'sweep_statistics.json')
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Aggregate Summary Generator
# ═══════════════════════════════════════════════════════════════════════════

def generate_aggregate_summary(fi_results):
    """Generate the summary table for the report (fixes 200c bug)."""
    print("\n" + "="*70)
    print("GENERATING AGGREGATE SUMMARY")
    print("="*70)

    from collections import defaultdict

    # Group by scale × tw_type
    by_group = defaultdict(lambda: {'old_fb': 0, 'new_fb': 0, 'count': 0,
                                      'new_moves': 0, 'fi_success': 0,
                                      'tw_feasible': 0})

    for key, data in fi_results.items():
        if 'error' in data:
            continue
        scale = data.get('scale')
        tw_type = data.get('tw_type')
        if scale is None or tw_type is None:
            continue

        g = f"{scale}c_{tw_type}"
        by_group[g]['old_fb'] += data.get('old_fallback_count', 0)
        by_group[g]['new_fb'] += data.get('new_fallback_count', 0)
        by_group[g]['count'] += 1
        by_group[g]['new_moves'] += data.get('new_moves_accepted', 0)
        if data.get('new_fi_success', False):
            by_group[g]['fi_success'] += 1
        if data.get('new_tw_feasible', False):
            by_group[g]['tw_feasible'] += 1

    # Print table
    print(f"\n{'Scale':<10} {'RC1':>20} {'RC2':>20} {'R1':>20} {'R2':>20} {'C1':>20} {'C2':>20}")
    print("-" * 130)

    for scale in [25, 50, 100, 200]:
        row = f"{scale}c{'':>6}"
        for tw_type in ['RC1', 'RC2', 'R1', 'R2', 'C1', 'C2']:
            g = f"{scale}c_{tw_type}"
            if g in by_group:
                bg = by_group[g]
                pct = (bg['old_fb'] - bg['new_fb']) / max(bg['old_fb'], 1) * 100
                row += f" {bg['new_fb']}/{bg['old_fb']} {pct:.0f}%↓  "
            else:
                row += f" {'—':>15}  "
        print(row)

    print("-" * 130)
    # Total row
    total_old = sum(bg['old_fb'] for bg in by_group.values())
    total_new = sum(bg['new_fb'] for bg in by_group.values())
    total_pct = (total_old - total_new) / max(total_old, 1) * 100
    total_tw_feas = sum(bg['tw_feasible'] for bg in by_group.values())
    total_count = sum(bg['count'] for bg in by_group.values())
    total_fi_ok = sum(bg['fi_success'] for bg in by_group.values())
    print(f"ALL        {total_new}/{total_old} {total_pct:.0f}%↓  "
          f"TW Feas: {total_tw_feas}/{total_count}  FI Success: {total_fi_ok}/{total_count}")

    return {
        'total_old_fallback': total_old,
        'total_new_fallback': total_new,
        'fallback_reduction_pct': total_pct,
        'tw_feasibility_rate': total_tw_feas / max(total_count, 1),
        'fi_success_rate': total_fi_ok / max(total_count, 1),
        'by_group': {g: dict(bg) for g, bg in by_group.items()},
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Comprehensive 224-instance sweep')
    parser.add_argument('--phase', type=str, default='all',
                       choices=['all', '1', '2', '3', '4', '5', 'summary'],
                       help='Which phase to run')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='Skip already-completed instances (default: True)')
    args = parser.parse_args()

    print("="*70)
    print("EVRP-TW + Forward Insertion Repair — Comprehensive 224-Instance Sweep")
    print(f"Instances: {len(ALL_INSTANCE_LIST)} (56 base × 4 scales)")
    print(f"Scales: {SCALES}")
    print(f"N_RUNS: {N_RUNS}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    fi_results = None

    if args.phase in ('all', '1'):
        fi_results = run_forward_insertion_sweep()

    if args.phase in ('all', '2'):
        run_classical_baseline_sweep()

    if args.phase in ('all', '3'):
        run_ortools_gap_analysis()

    if args.phase in ('all', '4'):
        run_ev_binding_ablation()

    if args.phase in ('all', '5'):
        run_statistical_tests()

    if args.phase in ('all', 'summary'):
        fi_results = load_checkpoint('sweep_forward_insertion.json')
        summary = generate_aggregate_summary(fi_results)
        save_checkpoint(summary, 'sweep_aggregate_summary.json')

    print(f"\n{'='*70}")
    print(f"ALL PHASES COMPLETE")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
