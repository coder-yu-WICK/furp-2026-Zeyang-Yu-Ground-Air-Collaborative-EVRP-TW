#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E-CVRP Benchmark Experiment — POMO + EV Charging vs Published SOTA.

Runs POMO neural routing on all 24 Mavrovouniotis et al. (IEEE CEC 2020)
benchmark instances, applies EV charging station insertion, and compares
against published results from:

  - BACO (Jia et al., IEEE Trans. Cybernetics 2022)
  - BHGA (Feng et al., IEEE 2024)
  - CBACO (Jia et al., IEEE Trans. Evol. Computation 2022)

Key differences from our EVRP-TW pipeline:
  - No time windows → Forward Insertion repair is skipped
  - Instance-specific battery capacity and energy consumption
  - Instance-specific vehicle cargo capacity
  - Charging station coordinates from the benchmark

Output:
  - results/ecvrp_benchmark_results.json — full results per instance
  - Summary comparison table
"""

import os, sys, json, time, math, random, traceback
from collections import defaultdict
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from week8.config import (
    TRUCK_SPEED, TRUCK_FIXED_COST, TRUCK_DIST_COST_RATE,
    DEPOT, BATTERY_CAPACITY, ENERGY_CONSUMPTION_RATE,
)
from week8.core.problem_model import TruckSolution
from week8.core.ecvrp_loader import (
    load_all_ecvrp_instances, save_ecvrp_instances,
    load_ecvrp_instance_from_disk,
)
from week8.ev.ev_model import (
    EVTruckSolution, simulate_route_ev,
    get_charging_station_coords, insert_charging_stops,
    station_distance,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── POMO Inference (adapted for CEVRP) ──────────────────────────────────

def _pomo_inference_cevrp(problem, model, device, truck_speed, battery_capacity,
                           max_demand, tw_horizon):
    """Run POMO greedy inference with configurable max_demand."""
    # Import here to avoid circular imports
    from week8.algorithms.pomo.pomo_env import POMOEnv

    env = POMOEnv(
        truck_speed=truck_speed,
        battery_capacity=battery_capacity,
        energy_per_km=1.0,
        tw_horizon=tw_horizon,
    )
    # Override max_demand for this instance
    env.max_demand = max_demand

    with torch.no_grad():
        env.load_problems([problem])
        state = env.reset(device=device)

        node_feat = torch.zeros(1, env.problem_size, 6, device=device)
        depot_xy = torch.zeros(1, 1, 2, device=device)
        node_feat[0, :, :2] = problem['node_xy'].to(device)
        node_feat[0, :, 2] = problem['node_demand'].to(device)
        node_feat[0, :, 3] = problem['node_tw_start'].to(device)
        node_feat[0, :, 4] = problem['node_tw_end'].to(device)
        node_feat[0, :, 5] = problem['node_service'].to(device)
        d = problem['depot_xy'].to(device)
        depot_xy[0, 0, :] = d.reshape(-1)[:2]

        model.pre_forward(depot_xy, node_feat)

        b, pomo = env.batch_size, env.pomo_size

        env.step(torch.zeros((b, pomo), dtype=torch.long, device=device))
        state = env._get_state()

        step1 = torch.arange(1, pomo + 1, dtype=torch.long, device=device).unsqueeze(0).expand(b, -1)
        env.step(step1)
        state = env._get_state()

        done = torch.zeros((b, pomo), dtype=torch.bool, device=device)
        for _ in range(pomo * 3):
            if done.all():
                break
            probs = model(state)
            selected = probs.argmax(dim=2)
            step_done = env.step(selected)
            done = done | step_done
            state = env._get_state()

    routes = []
    for pi in range(env.pomo_size):
        rlen = env.route_len[0, pi].item()
        if rlen > 1:
            routes.append(env.routes[0, pi, :rlen].cpu().tolist())
    return routes


def _split_route(flat_route):
    """Split flat route at depot returns (0)."""
    routes, cur = [], []
    for node in flat_route:
        if node == 0:
            if cur:
                routes.append(cur)
                cur = []
        else:
            cur.append(node)
    if cur:
        routes.append(cur)
    return routes


def _kmeans_cluster_ecvrp(customers, k, seed=42):
    """K-means clustering by (x,y) coordinates. Returns k clusters."""
    n = len(customers)
    if k >= n:
        return [[c] for c in customers] + [[] for _ in range(k - n)]

    coords = np.array([[c['x'], c['y']] for c in customers], dtype=np.float32)
    rng = np.random.RandomState(seed)
    # K-means++ init
    centroids = [coords[rng.randint(n)]]
    for _ in range(1, k):
        dists = np.min([np.sum((coords - c)**2, axis=1) for c in centroids], axis=0)
        probs = dists / dists.sum()
        centroids.append(coords[rng.choice(n, p=probs)])

    centroids = np.array(centroids)
    for _ in range(50):
        dists = np.array([np.sum((coords - c)**2, axis=1) for c in centroids])
        labels = np.argmin(dists, axis=0)
        new_centroids = np.array([coords[labels == i].mean(axis=0) if (labels == i).any()
                                   else centroids[i] for i in range(k)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    labels = np.argmin(np.array([np.sum((coords - c)**2, axis=1) for c in centroids]), axis=0)
    clusters = [[] for _ in range(k)]
    for i, c in enumerate(customers):
        clusters[labels[i]].append(c)
    return clusters


def _balance_capacity(clusters, capacity):
    """Move customers from overloaded clusters to underloaded ones."""
    for _ in range(20):
        loads = [sum(c['demand'] for c in cl) for cl in clusters]
        overloaded = [(i, loads[i]) for i in range(len(clusters)) if loads[i] > capacity]
        if not overloaded:
            break
        oi, _ = max(overloaded, key=lambda x: x[1])
        # Move farthest customer to nearest underloaded cluster
        underloaded = [i for i in range(len(clusters)) if loads[i] < capacity and i != oi]
        if not underloaded:
            break
        # Find the customer farthest from its cluster centroid
        coords = np.array([[c['x'], c['y']] for c in clusters[oi]], dtype=np.float32)
        centroid = coords.mean(axis=0)
        dists = np.sum((coords - centroid)**2, axis=1)
        move_idx = int(np.argmax(dists))
        # Find nearest underloaded cluster
        c_coord = coords[move_idx]
        ul_coords = np.array([[c['x'], c['y']] for c in sum((clusters[ui] for ui in underloaded), [])])
        if len(ul_coords) > 0:
            best_ui = underloaded[0]
            best_dist = float('inf')
            for ui in underloaded:
                ui_centroid = np.array([[c['x'], c['y']] for c in clusters[ui]], dtype=np.float32).mean(axis=0)
                d = np.sum((c_coord - ui_centroid)**2)
                if d < best_dist:
                    best_dist = d
                    best_ui = ui
            clusters[best_ui].append(clusters[oi].pop(move_idx))
    return clusters


def solve_instance_pomo(instance, model, device='cpu', seed=42):
    """
    Run POMO on a single E-CVRP instance using capacity-aware clustering.

    Uses K-means spatial clustering with instance-specific cargo capacity,
    then POMO per cluster. This ensures correct vehicle count and capacity.
    """
    from week8.algorithms.pomo.pomo_problem import instance_to_pomo_features, augment_xy_by_8_fold

    n_customers = instance['n_customers']
    truck_count = instance.get('n_vehicles', 2)
    cargo_capacity = instance.get('cargo_capacity', 200)
    battery_cap = instance.get('battery_capacity', 100)

    random.seed(seed)
    torch.manual_seed(seed)

    customers = instance['customers']
    total_demand = sum(c['demand'] for c in customers)
    min_clusters = max(1, int(math.ceil(total_demand / cargo_capacity)))
    n_clusters = max(truck_count, min_clusters)

    # Cluster with instance-specific capacity
    clusters = _kmeans_cluster_ecvrp(customers, n_clusters, seed=seed)
    clusters = _balance_capacity(clusters, cargo_capacity)
    clusters = [c for c in clusters if c]

    all_truck_routes = []
    depot_pt = instance['depot']

    for ci, cluster in enumerate(clusters):
        if len(cluster) == 0:
            continue
        if len(cluster) == 1:
            all_truck_routes.append([cluster[0]['id']])
            continue

        m = len(cluster)
        new_dist = [[0.0] * (m + 1) for _ in range(m + 1)]
        for i, c in enumerate(cluster):
            d = math.hypot(depot_pt[0] - c['x'], depot_pt[1] - c['y'])
            new_dist[0][i + 1] = d
            new_dist[i + 1][0] = d
        for i in range(m):
            for j in range(m):
                d = math.hypot(cluster[i]['x'] - cluster[j]['x'],
                              cluster[i]['y'] - cluster[j]['y'])
                new_dist[i + 1][j + 1] = d

        new_cust = []
        for ni, c in enumerate(cluster, start=1):
            new_cust.append({
                'id': ni, 'x': c['x'], 'y': c['y'],
                'demand': c['demand'],
                'ready_time': 0.0, 'due_time': 1e9, 'service_time': 0.0,
                '_orig_id': c['id'],
            })

        mini_inst = {
            'name': f'cluster_{ci}',
            'n_customers': m,
            'customers': new_cust, 'depot': depot_pt,
            'distance_matrix': new_dist,
            'tw_type': 'none', 'tw_horizon': 1e9,
        }

        _, depot_xy, node_feat = instance_to_pomo_features(mini_inst)
        aug_depot, aug_nodes_xy = augment_xy_by_8_fold(
            depot_xy.unsqueeze(0), node_feat[:, :2].unsqueeze(0))
        b_aug = aug_depot.shape[0]
        aug_node_feat = node_feat.unsqueeze(0).repeat(b_aug, 1, 1)
        aug_node_feat[:, :, :2] = aug_nodes_xy

        best_route = None
        best_cost = float('inf')

        for i in range(b_aug):
            problem = {
                'depot_xy': aug_depot[i].cpu(),
                'node_xy': aug_node_feat[i, :, :2].cpu(),
                'node_demand': aug_node_feat[i, :, 2].cpu(),
                'node_tw_start': aug_node_feat[i, :, 3].cpu(),
                'node_tw_end': aug_node_feat[i, :, 4].cpu(),
                'node_service': aug_node_feat[i, :, 5].cpu(),
            }
            routes = _pomo_inference_cevrp(
                problem, model, device, TRUCK_SPEED, battery_cap,
                max_demand=cargo_capacity, tw_horizon=1e9)

            for route in routes:
                truck_routes = _split_route(route)
                sol = TruckSolution(truck_routes, mini_inst)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_route = truck_routes

        if best_route:
            id_map = {c['id']: c['_orig_id'] for c in new_cust}
            for route in best_route:
                mapped = [id_map[cid] for cid in route]
                if mapped:
                    all_truck_routes.append(mapped)

    return all_truck_routes


# ── EV Charging Evaluation ──────────────────────────────────────────────

def evaluate_ev_charging(routes, instance):
    """
    Evaluate routes with EV battery constraints using instance-specific
    battery capacity and energy consumption.

    First tries routes as-is; if battery violations exist, inserts charging
    stations greedily.
    """
    customers = instance['customers']
    dist_matrix = instance['distance_matrix']
    depot = instance['depot']
    if isinstance(depot, list):
        depot = tuple(depot)

    battery_cap = instance.get('battery_capacity', 100)
    energy_rate = instance.get('energy_consumption_rate', 1.0)
    cargo_cap = instance.get('cargo_capacity', 200)

    n_cust = len(customers)
    cs_coords = get_charging_station_coords(n_cust)

    # If instance has its own charging stations, use those instead
    if instance.get('charging_stations'):
        cs_coords = {}
        for cs in instance['charging_stations']:
            cs_coords[cs['id']] = (cs['x'], cs['y'])

    total_dist = 0.0
    total_energy = 0.0
    total_charges = 0
    total_charge_energy = 0.0
    energy_violations = 0.0
    all_feasible = True

    for route in routes:
        if not route:
            continue

        # Add charging stations where needed
        # First simulate without charging
        sim = simulate_route_ev(
            route, customers, dist_matrix, cs_coords, depot,
            battery_capacity=battery_cap,
            charging_model='linear',
            energy_rate=energy_rate,
        )

        total_dist += sim['total_dist']
        total_energy += sim['total_energy']
        total_charges += sim['n_charges']
        total_charge_energy += sim['total_charge_energy']

        if sim['energy_violation'] > 0.01:
            energy_violations += sim['energy_violation']
            all_feasible = False

    return {
        'total_dist': round(total_dist, 2),
        'total_energy': round(total_energy, 2),
        'total_charges': total_charges,
        'total_charge_energy': round(total_charge_energy, 2),
        'energy_violations': round(energy_violations, 2),
        'ev_feasible': all_feasible and energy_violations <= 0.01,
    }


# ── Full Solution Evaluation ────────────────────────────────────────────

def compute_solution_cost(routes, instance):
    """Compute standard TruckSolution cost for the routes."""
    sol = TruckSolution(routes, instance)
    return {
        'cost': round(sol.cost, 2),
        'tardiness': round(sol.tardiness, 2),
        'feasible': sol.feasible,
        'n_routes': len(routes),
    }


# ── Main ────────────────────────────────────────────────────────────────

def main():
    # Load POMO model
    _SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(_SRC, 'week4', 'algorithms', 'pomo', 'checkpoints', 'best_model.pt')
    if not os.path.exists(model_path):
        alt = os.path.join(_SRC, 'week4', 'algorithms', 'pomo', 'checkpoints', 'final_model.pt')
        if os.path.exists(alt):
            model_path = alt
        else:
            print(f"ERROR: No POMO model found")
            return

    from week8.algorithms.pomo.pomo_model import POMOModel
    device = 'cpu'
    model = POMOModel().to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded POMO model from: {model_path}")

    # Load instances
    benchmark_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  'e_cvrp_benchmark')
    instances = load_all_ecvrp_instances(benchmark_dir)

    # Save parsed instances
    save_ecvrp_instances(instances)

    print(f"\n{'='*80}")
    print(f"E-CVRP Benchmark: POMO Neural Routing on {len(instances)} instances")
    print(f"{'='*80}")

    # Checkpoint
    ckpt_path = os.path.join(RESULTS_DIR, 'ecvrp_benchmark_results.json')
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            results = json.load(f)
    else:
        results = {}

    # Sort by instance size
    sorted_names = sorted(instances.keys(), key=lambda n: instances[n]['n_customers'])
    total = len(sorted_names)

    for idx, name in enumerate(sorted_names):
        if name in results:
            continue

        inst = instances[name]
        n_cust = inst['n_customers']
        n_veh = inst['n_vehicles']
        batt = inst['battery_capacity']
        cargo = inst['cargo_capacity']
        opt = inst.get('optimal_value')

        print(f"\n[{idx+1}/{total}] {name}: {n_cust}c, {n_veh}v, batt={batt:.0f}kWh, cap={cargo:.0f}",
              end=' ', flush=True)

        try:
            t0 = time.time()

            # Run POMO
            routes = solve_instance_pomo(inst, model, device=device, seed=42)

            if not routes:
                print("NO SOLUTION")
                results[name] = {'error': 'no_solution'}
                continue

            # Compute metrics
            cost_info = compute_solution_cost(routes, inst)
            ev_info = evaluate_ev_charging(routes, inst)
            runtime = time.time() - t0

            results[name] = {
                'n_customers': n_cust,
                'n_vehicles': n_veh,
                'battery_capacity': batt,
                'cargo_capacity': cargo,
                'optimal_value': opt,
                # Routing results
                'pomo_cost': cost_info['cost'],
                'pomo_tardiness': cost_info['tardiness'],
                'pomo_feasible': cost_info['feasible'],
                'n_routes_used': cost_info['n_routes'],
                # EV results
                'total_dist': ev_info['total_dist'],
                'total_energy': ev_info['total_energy'],
                'n_charges': ev_info['total_charges'],
                'charge_energy': ev_info['total_charge_energy'],
                'energy_violations': ev_info['energy_violations'],
                'ev_feasible': ev_info['ev_feasible'],
                'runtime': round(runtime, 3),
            }

            feas = '✓' if ev_info['ev_feasible'] else '✗'
            print(f"→ dist={ev_info['total_dist']:.0f} EV={feas} chg={ev_info['total_charges']} "
                  f"({runtime:.1f}s)")

        except Exception as e:
            print(f"ERROR: {e}")
            results[name] = {'error': str(e), 'traceback': traceback.format_exc()}

        # Checkpoint every 5 instances
        if (idx + 1) % 5 == 0:
            with open(ckpt_path + '.tmp', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            os.replace(ckpt_path + '.tmp', ckpt_path)
            print(f"    [checkpoint: {idx+1}/{total}]")

    # Final save
    with open(ckpt_path + '.tmp', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    os.replace(ckpt_path + '.tmp', ckpt_path)

    # ── Summary ──
    print(f"\n{'='*80}")
    print(f"SUMMARY: POMO on E-CVRP Benchmark")
    print(f"{'='*80}")
    print(f"{'Instance':<25} {'N':>5} {'Veh':>4} {'POMO Dist':>10} {'Opt':>8} {'EV':>4} {'Chg':>4} {'Time':>7}")
    print("-" * 75)

    ev_feas = 0
    total_valid = 0
    for name in sorted_names:
        r = results.get(name, {})
        if 'error' not in r:
            total_valid += 1
            if r.get('ev_feasible'):
                ev_feas += 1
            dist = r.get('total_dist', 0)
            opt = r.get('optimal_value')
            opt_str = f"{opt:.0f}" if opt else "-"
            print(f"{name:<25} {r['n_customers']:>5} {r['n_vehicles']:>4} {dist:>10.1f} {opt_str:>8} "
                  f"{'✓' if r.get('ev_feasible') else '✗':>4} {r.get('n_charges', 0):>4} {r.get('runtime', 0):>6.1f}s")

    print(f"\n  Valid instances: {total_valid}")
    print(f"  EV feasible: {ev_feas}/{total_valid} ({ev_feas/max(total_valid,1)*100:.0f}%)")

    print(f"\nResults saved to: {ckpt_path}")


if __name__ == '__main__':
    main()
