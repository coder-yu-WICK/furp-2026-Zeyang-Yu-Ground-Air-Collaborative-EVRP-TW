# -*- coding: utf-8 -*-
"""
POMO Multi-Truck Solver — Cluster-first, Route-second.

Partitions customers among K trucks via K-means clustering, then each truck
independently runs POMO (neural single-vehicle router) on its cluster.

No drones — clean comparison against No-Drone baseline from Week 3.
"""

import os, sys, math, random, time
import torch
import numpy as np

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    TRUCK_SPEED, TRUCK_CAPACITY, TRUCK_FIXED_COST,
    TRUCK_DIST_COST_RATE, TARDINESS_COST_RATE,
    DEPOT, BATTERY_CAPACITY,
)
from src.core.problem_model import TruckDroneSolution, extract_pareto_front
from src.algorithms.pomo.pomo_model import POMOModel
from src.algorithms.pomo.pomo_env import POMOEnv
from src.algorithms.pomo.pomo_problem import augment_xy_by_8_fold


# ── K-Means Clustering ──────────────────────────────────────────────

def _kmeans_cluster(customers, k, max_iter=50, seed=42):
    """
    Cluster customers into K groups by (x, y) coordinates.
    Manual Lloyd's algorithm — no sklearn dependency.

    Returns: list of K lists of customer dicts
    """
    n = len(customers)
    if k >= n:
        # Each customer gets its own cluster; remaining clusters are empty
        clusters = [[c] for c in customers] + [[] for _ in range(k - n)]
        return clusters

    coords = np.array([[c['x'], c['y']] for c in customers], dtype=np.float32)

    # Initialize centroids with k-means++
    rng = np.random.RandomState(seed)
    centroids = [coords[rng.randint(n)]]
    for _ in range(1, k):
        dists = np.min([np.sum((coords - c)**2, axis=1) for c in centroids], axis=0)
        probs = dists / dists.sum()
        centroids.append(coords[rng.choice(n, p=probs)])
    centroids = np.array(centroids)

    for _ in range(max_iter):
        # Assign
        dists = np.sum((coords[:, None, :] - centroids[None, :, :])**2, axis=2)
        labels = np.argmin(dists, axis=1)

        # Update
        new_centroids = np.array([
            coords[labels == ki].mean(axis=0) if (labels == ki).sum() > 0
            else coords[rng.randint(n)]
            for ki in range(k)
        ])

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    # Build clusters
    clusters = [[] for _ in range(k)]
    for i, c in enumerate(customers):
        clusters[labels[i]].append(c)
    return clusters


def _balance_capacity(clusters, capacity=200.0):
    """
    Reassign customers from overloaded clusters to nearby under-capacity clusters.
    Simple greedy: move the farthest customer from the overloaded cluster
    to the nearest under-capacity cluster.
    """
    k = len(clusters)
    total_demands = [sum(c['demand'] for c in cl) for cl in clusters]

    for _ in range(100):  # max iterations
        overloaded = [i for i in range(k) if total_demands[i] > capacity]
        if not overloaded:
            break

        src = overloaded[0]
        # Find target: under-capacity cluster nearest to src's centroid
        src_xy = np.mean([[c['x'], c['y']] for c in clusters[src]], axis=0) if clusters[src] else np.zeros(2)
        under = [(i, total_demands[i]) for i in range(k) if total_demands[i] < capacity]
        if not under:
            break  # all full, can't fix

        # Pick target: nearest to src centroid with most remaining capacity
        under.sort(key=lambda x: total_demands[x[0]])  # least full first
        dst = under[0][0]

        # Move one customer: farthest from dst centroid in src cluster
        dst_xy = np.mean([[c['x'], c['y']] for c in clusters[dst]], axis=0) if clusters[dst] else np.array([8.0, 8.0])
        if clusters[src]:
            dist_to_dst = [((c['x'] - dst_xy[0])**2 + (c['y'] - dst_xy[1])**2, i)
                          for i, c in enumerate(clusters[src])]
            dist_to_dst.sort()
            _, move_idx = dist_to_dst[0]  # nearest to dst = easiest to reassign
            c = clusters[src].pop(move_idx)
            clusters[dst].append(c)
            total_demands[src] -= c['demand']
            total_demands[dst] += c['demand']

    return clusters


def cluster_customers(instance, n_trucks):
    """Partition customers into clusters. Uses at least n_trucks,
    but may use more if capacity requires it."""
    customers = instance['customers']
    total_demand = sum(c['demand'] for c in customers)
    min_clusters = max(1, int(math.ceil(total_demand / TRUCK_CAPACITY)))
    n_clusters = max(n_trucks, min_clusters)
    clusters = _kmeans_cluster(customers, n_clusters)
    clusters = _balance_capacity(clusters, capacity=TRUCK_CAPACITY)
    # Filter empty clusters
    clusters = [c for c in clusters if c]
    return clusters


# ── Mini-Instance Builder ───────────────────────────────────────────

def build_cluster_instance(full_instance, cluster):
    """
    Create a mini-instance for a single cluster of customers.
    Preserves original customer IDs for route reconstruction.
    """
    full_customers = full_instance['customers']
    depot = full_instance['depot']

    # Build new distance matrix: depot (idx 0) + cluster customers (idx 1..m)
    cluster_ids = [c['id'] for c in cluster]
    m = len(cluster_ids)

    new_dist = np.zeros((m + 1, m + 1))
    # Depot row/col
    for i, cid in enumerate(cluster_ids):
        d = math.sqrt((depot[0] - cluster[i]['x'])**2 + (depot[1] - cluster[i]['y'])**2)
        new_dist[0, i + 1] = d
        new_dist[i + 1, 0] = d
    # Customer-to-customer
    for i in range(m):
        for j in range(m):
            d = math.sqrt((cluster[i]['x'] - cluster[j]['x'])**2 +
                         (cluster[i]['y'] - cluster[j]['y'])**2)
            new_dist[i + 1, j + 1] = d

    # Re-map customer IDs to 1..m
    new_customers = []
    for new_id, c in enumerate(cluster, start=1):
        new_customers.append({
            'id': new_id,
            'x': c['x'],
            'y': c['y'],
            'demand': c['demand'],
            'ready_time': c['ready_time'],
            'due_time': c['due_time'],
            'service_time': c['service_time'],
            '_orig_id': c['id'],  # keep original ID for reconstruction
        })

    return {
        'customers': new_customers,
        'depot': depot,
        'distance_matrix': new_dist.tolist(),
        'tw_type': full_instance.get('tw_type', 'RC1'),
        'tw_horizon': full_instance.get('tw_horizon', 240.0),
    }


# ── POMO Single-Cluster Inference ───────────────────────────────────

def _pomo_inference_cluster(problem, model, device, truck_speed, battery_capacity, tw_horizon):
    """
    Run POMO greedy inference on a single cluster problem.
    Returns list of route candidates (list of node indices, 0 = depot).
    """
    env = POMOEnv(
        truck_speed=truck_speed,
        battery_capacity=battery_capacity,
        energy_per_km=1.0,
        tw_horizon=tw_horizon,
    )

    with torch.no_grad():
        env.load_problems([problem])
        state = env.reset(device=device)

        # Build features
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

        # Step 0: depot
        env.step(torch.zeros((b, pomo), dtype=torch.long, device=device))
        state = env._get_state()

        # Step 1: POMO init (each trajectory → different first customer)
        step1 = torch.arange(1, pomo + 1, dtype=torch.long, device=device).unsqueeze(0).expand(b, -1)
        env.step(step1)
        state = env._get_state()

        # Greedy decoding
        done = torch.zeros((b, pomo), dtype=torch.bool, device=device)
        for _ in range(pomo * 3):
            if done.all():
                break
            probs = model(state)
            selected = probs.argmax(dim=2)
            step_done = env.step(selected)
            done = done | step_done
            state = env._get_state()

    # Decode routes
    routes = []
    for pi in range(env.pomo_size):
        rlen = env.route_len[0, pi].item()
        if rlen > 1:
            routes.append(env.routes[0, pi, :rlen].cpu().tolist())
    return routes


# ── Multi-Truck Solver ──────────────────────────────────────────────

class POMOMultiTruckSolver:
    """Cluster-first, route-second: K-means + per-cluster POMO."""

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
        """
        Run POMO (with 8-fold augmentation) on a single cluster.
        Returns list of TruckDroneSolution candidates.
        """
        from src.algorithms.pomo.pomo_problem import instance_to_pomo_features

        random.seed(seed)
        torch.manual_seed(seed)

        _, depot_xy, node_feat = instance_to_pomo_features(cluster_instance)
        n = len(cluster_instance['customers'])
        tw_horizon = cluster_instance.get('tw_horizon', 240.0)

        # 8-fold augmentation
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

        # Convert to solutions (single truck, no drones)
        solutions = []
        for route in all_routes:
            # route is a flat list like [0, 3, 7, 0, ...] — split at depot
            truck_routes = _split_route(route)
            solutions.append(TruckDroneSolution(
                truck_routes, [], cluster_instance))

        return solutions

    def solve(self, instance, n_trucks, n_runs=1, seed=42):
        """
        Solve full instance with n_trucks trucks.

        1. Cluster customers
        2. Per-cluster POMO inference
        3. Combine routes
        4. Return Pareto front
        """
        all_solutions = []
        n_cust = len(instance['customers'])

        for run in range(n_runs):
            run_seed = seed + run

            # Cluster customers
            clusters = cluster_customers(instance, n_trucks)
            actual_trucks = len(clusters)

            # Solve each cluster independently
            truck_routes = []
            for ci, cluster in enumerate(clusters):
                if len(cluster) <= 1:
                    # Trivial: single customer → direct depot→customer→depot
                    if cluster:
                        truck_routes.append([cluster[0]['id']])
                    continue

                mini_inst = build_cluster_instance(instance, cluster)
                try:
                    sols = self.solve_cluster(mini_inst, seed=run_seed + ci)

                    # Pick best feasible solution for this cluster
                    best, best_cost = None, float('inf')
                    for s in sols:
                        if s.feasible and s.cost < best_cost:
                            best_cost = s.cost
                            best = s
                    if best is None and sols:
                        best = sols[0]  # fallback to infeasible

                    if best and best.truck_routes:
                        # Map mini-instance IDs back to original IDs
                        id_map = {c['id']: c['_orig_id'] for c in mini_inst['customers']}
                        for route in best.truck_routes:
                            mapped = [id_map.get(cid, cid) for cid in route]
                            truck_routes.append(mapped)
                except Exception as e:
                    # Fallback: nearest-neighbor heuristic for this cluster
                    truck_routes.append(_nearest_neighbor(cluster, instance['depot']))

            # Build full solution
            sol = TruckDroneSolution(truck_routes, [], instance)
            all_solutions.append(sol)

        return extract_pareto_front(all_solutions)


# ── Helpers ─────────────────────────────────────────────────────────

def _split_route(flat_route):
    """Split flat route into sub-routes at depot returns (0)."""
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


def _nearest_neighbor(cluster, depot):
    """Fallback: nearest-neighbor route for a single cluster."""
    if not cluster:
        return []
    ids = [c['id'] for c in cluster]
    coords = {c['id']: (c['x'], c['y']) for c in cluster}

    route = []
    unvisited = set(ids)
    cur = (depot[0], depot[1])

    while unvisited:
        best_id = min(unvisited, key=lambda cid:
            (cur[0] - coords[cid][0])**2 + (cur[1] - coords[cid][1])**2)
        route.append(best_id)
        cur = coords[best_id]
        unvisited.remove(best_id)

    return route


# ── Week3-compatible Interface ──────────────────────────────────────

def run_pomo_multitruck(instance, n_runs=1, n_trucks=2, n_drones=0,
                        endurance=None, seed=42):
    """
    Interface compatible with week3 experiment runner.
    n_drones and endurance are ignored (truck-only comparison).
    """
    _here = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(_here, '..', 'algorithms', 'pomo', 'checkpoints', 'best_model.pt')
    if not os.path.exists(model_path):
        alt = os.path.join(_here, '..', 'algorithms', 'pomo', 'checkpoints', 'final_model.pt')
        if os.path.exists(alt):
            model_path = alt
        else:
            raise FileNotFoundError(f"No POMO model found. Train first: python -m algorithms.pomo.train")

    device = 'cpu'
    solver = POMOMultiTruckSolver(model_path, device=device)

    all_solutions, times = [], []
    for run in range(n_runs):
        t0 = time.time()
        try:
            pareto = solver.solve(instance, n_trucks=n_trucks,
                                  n_runs=1, seed=seed + run)
            all_solutions.extend(pareto)
        except Exception as e:
            print(f"    POMO-MT run {run+1} error: {e}")
        times.append(time.time() - t0)

    pareto = extract_pareto_front(all_solutions)
    return {
        'solutions': all_solutions,
        'pareto_front': pareto,
        'mean_runtime': sum(times) / max(len(times), 1),
        'std_runtime': (sum((t - sum(times)/len(times))**2 for t in times) / max(len(times)-1, 1))**0.5
                        if len(times) > 1 else 0,
    }
