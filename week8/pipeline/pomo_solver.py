# -*- coding: utf-8 -*-
"""
Improved POMO Multi-Truck Solver — Week 5.

Combines two improvements over the Week 4 baseline:
  1. TW-Aware Clustering: spatio-temporal distance for K-means assignment
  2. Drone Post-Processing: greedy drone mission insertion after routing

Supports ablation study: each improvement can be independently toggled.
"""

import os, sys, math, random, time
import torch
import numpy as np

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from week8.config import (
    TRUCK_SPEED, TRUCK_CAPACITY, TRUCK_FIXED_COST,
    TRUCK_DIST_COST_RATE, TARDINESS_COST_RATE,
    DEPOT, BATTERY_CAPACITY,
)
from week8.core.problem_model import TruckSolution, extract_pareto_front

from week8.algorithms.pomo.pomo_model import POMOModel
from week8.algorithms.pomo.pomo_env import POMOEnv
from week8.algorithms.pomo.pomo_problem import augment_xy_by_8_fold

from week8.pipeline.clustering import cluster_customers_tw_aware
from week8.pipeline.adaptive_clustering import cluster_with_params
from week8.pipeline.cluster_feasibility import ensure_temporal_feasibility


# ── Re-use week4's mini-instance builder and helpers ─────────────────

def _build_cluster_instance(full_instance, cluster):
    """Create a mini-instance for a single cluster (same as week4)."""
    full_customers = full_instance['customers']
    depot = full_instance['depot']

    cluster_ids = [c['id'] for c in cluster]
    m = len(cluster_ids)

    new_dist = np.zeros((m + 1, m + 1))
    for i, cid in enumerate(cluster_ids):
        d = math.sqrt((depot[0] - cluster[i]['x'])**2 + (depot[1] - cluster[i]['y'])**2)
        new_dist[0, i + 1] = d
        new_dist[i + 1, 0] = d
    for i in range(m):
        for j in range(m):
            d = math.sqrt((cluster[i]['x'] - cluster[j]['x'])**2 +
                         (cluster[i]['y'] - cluster[j]['y'])**2)
            new_dist[i + 1, j + 1] = d

    new_customers = []
    for new_id, c in enumerate(cluster, start=1):
        new_customers.append({
            'id': new_id,
            'x': c['x'], 'y': c['y'],
            'demand': c['demand'],
            'ready_time': c['ready_time'],
            'due_time': c['due_time'],
            'service_time': c['service_time'],
            '_orig_id': c['id'],
        })

    return {
        'customers': new_customers,
        'depot': depot,
        'distance_matrix': new_dist.tolist(),
        'tw_type': full_instance.get('tw_type', 'RC1'),
        'tw_horizon': full_instance.get('tw_horizon', 240.0),
    }


def _nearest_neighbor(cluster, depot):
    """Fallback nearest-neighbor route."""
    if not cluster:
        return []
    ids = [c['id'] for c in cluster]
    coords = {c['id']: (c['x'], c['y']) for c in cluster}
    route, unvisited = [], set(ids)
    cur = (depot[0], depot[1])
    while unvisited:
        best_id = min(unvisited, key=lambda cid:
            (cur[0] - coords[cid][0])**2 + (cur[1] - coords[cid][1])**2)
        route.append(best_id)
        cur = coords[best_id]
        unvisited.remove(best_id)
    return route


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


# ── POMO Single-Cluster Inference ───────────────────────────────────────

def _pomo_inference_cluster(problem, model, device, truck_speed, battery_capacity, tw_horizon):
    """Run POMO greedy inference on a single cluster (same as week4)."""
    env = POMOEnv(
        truck_speed=truck_speed,
        battery_capacity=battery_capacity,
        energy_per_km=1.0,
        tw_horizon=tw_horizon,
    )

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


# ── Improved Multi-Truck Solver ─────────────────────────────────────────

class ImprovedPOMOSolver:
    """
    POMO-MT with configurable improvements.

    Clustering variants:
      - baseline:          spatial-only clustering
      - tw_aware:          TW-aware clustering
      - adaptive_tw:       Adaptive TW-aware clustering
      - angle:             Angle-based petal clustering
      - hybrid:            Auto-select clustering (angle for RC1, adaptive TW for RC2)
    """

    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model_path = model_path
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = POMOModel().to(self.device)
            ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
            self._model.load_state_dict(ckpt['model_state_dict'])
            self._model.eval()
        return self._model

    def solve_cluster(self, cluster_instance, seed=42):
        """Run POMO with 8-fold augmentation on a single cluster."""
        from week8.algorithms.pomo.pomo_problem import instance_to_pomo_features

        random.seed(seed)
        torch.manual_seed(seed)

        _, depot_xy, node_feat = instance_to_pomo_features(cluster_instance)
        tw_horizon = cluster_instance.get('tw_horizon', 240.0)

        node_xy_2d = node_feat[:, :2]
        aug_depot, aug_nodes_xy = augment_xy_by_8_fold(
            depot_xy.unsqueeze(0), node_xy_2d.unsqueeze(0))
        b_aug = aug_depot.shape[0]
        aug_node_feat = node_feat.unsqueeze(0).repeat(b_aug, 1, 1)
        aug_node_feat[:, :, :2] = aug_nodes_xy

        all_routes = []
        for i in range(b_aug):
            problem = {
                'depot_xy': aug_depot[i].cpu(),
                'node_xy': aug_node_feat[i, :, :2].cpu(),
                'node_demand': aug_node_feat[i, :, 2].cpu(),
                'node_tw_start': aug_node_feat[i, :, 3].cpu(),
                'node_tw_end': aug_node_feat[i, :, 4].cpu(),
                'node_service': aug_node_feat[i, :, 5].cpu(),
            }
            routes = _pomo_inference_cluster(
                problem, self.model, self.device,
                TRUCK_SPEED, BATTERY_CAPACITY, tw_horizon)
            all_routes.extend(routes)

        solutions = []
        for route in all_routes:
            truck_routes = _split_route(route)
            solutions.append(TruckSolution(truck_routes, cluster_instance))

        return solutions

    def _get_cluster_strategy(self, variant):
        """Map variant name to clustering strategy."""
        if variant in ('baseline',):
            return 'spatial', None
        elif variant in ('tw_aware',):
            return 'tw_aware', 0.5
        elif variant in ('adaptive_tw',):
            return 'adaptive_tw', 0.4
        elif variant in ('angle',):
            return 'angle', None
        elif variant in ('hybrid',):
            return 'hybrid', 0.4
        else:
            return 'spatial', None

    def solve(self, instance, n_trucks, variant='baseline', n_runs=1,
              seed=42, tw_beta=0.5,
              check_tw_feasibility=True):
        """
        Solve full instance.

        Args:
            check_tw_feasibility: if True, validate clusters for temporal
                feasibility after clustering and split any infeasible ones.
                Fixes the clustering-TW contradiction at the source.
        """
        strategy, base_ratio = self._get_cluster_strategy(variant)

        all_solutions = []

        for run in range(n_runs):
            run_seed = seed + run

            # ── Step 1: Cluster customers ──
            if strategy == 'spatial':
                from week8.pipeline.pomo_multitruck import cluster_customers as spatial_cluster
                clusters = spatial_cluster(instance, n_trucks)
            elif strategy == 'tw_aware':
                clusters = cluster_customers_tw_aware(
                    instance, n_trucks, beta=tw_beta, seed=run_seed)
            else:
                clusters = cluster_with_params(
                    instance, n_trucks, strategy=strategy,
                    base_ratio=(base_ratio or tw_beta), seed=run_seed)

            # ── Step 1.5: Temporal feasibility check ──
            if check_tw_feasibility:
                n_before = len(clusters)
                clusters = ensure_temporal_feasibility(clusters, instance)
                n_after = len(clusters)
                if n_after != n_before:
                    pass  # splitting occurred; silently applied

            # ── Step 2: Solve each cluster with POMO ──
            truck_routes = []
            for ci, cluster in enumerate(clusters):
                if len(cluster) == 0:
                    continue
                if len(cluster) == 1:
                    truck_routes.append([cluster[0]['id']])
                    continue

                mini_inst = _build_cluster_instance(instance, cluster)
                try:
                    sols = self.solve_cluster(mini_inst, seed=run_seed + ci)

                    best, best_cost = None, float('inf')
                    for s in sols:
                        if s.feasible and s.cost < best_cost:
                            best_cost = s.cost
                            best = s
                    if best is None and sols:
                        best = sols[0]

                    if best and best.truck_routes:
                        id_map = {c['id']: c['_orig_id'] for c in mini_inst['customers']}
                        for route in best.truck_routes:
                            mapped = [id_map.get(cid, cid) for cid in route]
                            truck_routes.append(mapped)
                except Exception as e:
                    truck_routes.append(_nearest_neighbor(cluster, instance['depot']))

            # ── Step 3: Build solution ──
            sol = TruckSolution(truck_routes, instance)

            all_solutions.append(sol)

        return extract_pareto_front(all_solutions)


# ── Week3-compatible Interface ──────────────────────────────────────────

def run_pomo_improved(instance, n_runs=1, n_trucks=2,
                       seed=42, variant='baseline',
                       tw_beta=0.5, model_path=None, check_tw_feasibility=True):
    """
    Interface compatible with week3 experiment runner.

    Args:
        variant: 'baseline' | 'tw_aware' | 'adaptive_tw' | 'angle' | 'hybrid'
        tw_beta: temporal penalty weight (only for tw_aware variants)
        model_path: optional POMO checkpoint path (default: week4 best_model.pt)
    """
    if model_path is None:
        _SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(_SRC_DIR, 'week4', 'algorithms', 'pomo', 'checkpoints', 'best_model.pt')
        if not os.path.exists(model_path):
            alt = os.path.join(_SRC_DIR, 'week4', 'algorithms', 'pomo', 'checkpoints', 'final_model.pt')
            if os.path.exists(alt):
                model_path = alt
            else:
                raise FileNotFoundError(f"No POMO model found at {model_path}")

    device = 'cpu'
    solver = ImprovedPOMOSolver(model_path, device=device)

    all_solutions, times = [], []
    for run in range(n_runs):
        t0 = time.time()
        try:
            pareto = solver.solve(
                instance, n_trucks=n_trucks,
                variant=variant, n_runs=1,
                seed=seed + run, tw_beta=tw_beta,
                check_tw_feasibility=check_tw_feasibility,
            )
            all_solutions.extend(pareto)
        except Exception as e:
            print(f"    POMO-Improved [{variant}] run {run+1} error: {e}")
            import traceback; traceback.print_exc()
        times.append(time.time() - t0)

    pareto = extract_pareto_front(all_solutions)
    return {
        'solutions': all_solutions,
        'pareto_front': pareto,
        'mean_runtime': sum(times) / max(len(times), 1),
        'std_runtime': (sum((t - sum(times)/len(times))**2 for t in times) / max(len(times)-1, 1))**0.5
                        if len(times) > 1 else 0,
    }
