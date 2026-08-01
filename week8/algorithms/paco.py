# -*- coding: utf-8 -*-
"""
P-ACO: Collaborative Pareto Ant Colony Optimization for Truck-Only Routing.
Based on Das et al. (2020), "Synchronized Truck and Drone Routing in Package
Delivery Logistics", IEEE Trans. Intelligent Transportation Systems, 22(9),
5772-5782. DOI: 10.1109/TITS.2020.2992549
Adapted for truck-only routing (no drone missions).

Optimized version: sparse 2D pheromone, precomputed distances, greedy construction.
"""

import random
import math
import time
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from week8.config import (
    TRUCK_SPEED,
    TRUCK_CAPACITY,
    TRUCK_DIST_COST_RATE,
    TARDINESS_COST_RATE,
    DEPOT,
    PACO,
)
from week8.core.problem_model import TruckSolution, extract_pareto_front


class PACOSolver:
    """P-ACO solver for truck-only routing (optimized)."""

    def __init__(self, instance, seed=42):
        self.instance = instance
        self.customers = instance['customers']
        self.dist = instance['distance_matrix']
        self.n = instance['n_customers']
        random.seed(seed)

        cfg = PACO
        self.n_ants = cfg['ants_25c'] if self.n <= 25 else (
            cfg['ants_50c'] if self.n <= 50 else cfg['ants_100c'])
        self.max_iter = cfg['iterations']
        self.alpha = cfg['alpha']
        self.beta = cfg['beta']
        self.rho = cfg['rho']
        self.q0 = cfg['q0']
        self.Q_cost = cfg['Q_cost']
        self.Q_tard = cfg['Q_tard']

        # Pre-compute depot distances
        self.depot_dist = [0.0] * (self.n + 1)
        for cid in range(1, self.n + 1):
            c = self.customers[cid - 1]
            self.depot_dist[cid] = math.sqrt(
                (DEPOT[0] - c['x'])**2 + (DEPOT[1] - c['y'])**2)

        # 2D truck pheromone (cost and tardiness objectives)
        n_nodes = self.n + 1
        self.tau_truck_cost = [[1.0] * n_nodes for _ in range(n_nodes)]
        self.tau_truck_tard = [[1.0] * n_nodes for _ in range(n_nodes)]

    def _construct_solution(self):
        """Greedy randomized construction with P-ACO transition rules."""
        n_trucks = {25: 2, 50: 4, 100: 4}.get(self.n, 4)

        unserved = set(range(1, self.n + 1))
        truck_routes = [[] for _ in range(n_trucks)]
        truck_pos = [0] * n_trucks
        truck_load = [0.0] * n_trucks
        truck_time = [0.0] * n_trucks

        while unserved:
            made_progress = False
            for t in range(n_trucks):
                if not unserved:
                    break

                i = truck_pos[t]
                ti = truck_time[t]
                load = truck_load[t]

                # -- Build truck candidates --
                truck_opts = []
                for j in list(unserved):
                    cj = self.customers[j - 1]
                    if load + cj['demand'] > TRUCK_CAPACITY:
                        continue
                    d = self.dist[i][j] if i > 0 else self.depot_dist[j]
                    arr = ti + d / TRUCK_SPEED
                    if arr < cj['ready_time']:
                        arr = cj['ready_time']
                    ret = cj['service_time'] + self.depot_dist[j] / TRUCK_SPEED
                    if arr + ret <= self.instance['tw_horizon'] + 120:
                        heur = 1.0 / max(d, 0.001)
                        pher = self.tau_truck_cost[i][j] ** self.alpha * heur ** self.beta
                        truck_opts.append((j, heur, pher))

                if not truck_opts:
                    # Reset truck to depot
                    truck_pos[t] = 0
                    truck_load[t] = 0.0
                    truck_time[t] = 0.0
                    continue

                made_progress = True

                # -- Pseudo-random proportional selection --
                q = random.random()
                if q < self.q0:
                    # Exploitation: best truck move
                    best = max(truck_opts, key=lambda x: x[2])
                    self._do_truck(t, best[0], truck_routes, truck_pos, truck_load, truck_time, unserved)
                else:
                    # Probabilistic truck
                    total_p = sum(p for _, _, p in truck_opts) or 1.0
                    r = random.random() * total_p
                    cum = 0
                    for j, _, p in truck_opts:
                        cum += p
                        if cum >= r:
                            self._do_truck(t, j, truck_routes, truck_pos, truck_load, truck_time, unserved)
                            break

            if not made_progress and unserved:
                # Force-feed remaining to trucks
                for j in list(unserved):
                    for t in range(n_trucks):
                        if truck_load[t] + self.customers[j-1]['demand'] <= TRUCK_CAPACITY:
                            self._do_truck(t, j, truck_routes, truck_pos, truck_load, truck_time, unserved)
                            break

        return TruckSolution(truck_routes, self.instance)

    def _do_truck(self, t, cid, routes, pos, load, time_, unserved):
        """Serve customer by truck."""
        i = pos[t]
        c = self.customers[cid - 1]
        d = self.dist[i][cid] if i > 0 else self.depot_dist[cid]
        arr = time_[t] + d / TRUCK_SPEED
        if arr < c['ready_time']:
            arr = c['ready_time']
        routes[t].append(cid)
        unserved.discard(cid)
        pos[t] = cid
        load[t] += c['demand']
        time_[t] = arr + c['service_time']

    def _update_pheromone(self, solutions):
        """Evaporate + deposit on best-cost and best-tardiness solutions."""
        rho = self.rho
        for i in range(self.n + 1):
            for j in range(self.n + 1):
                self.tau_truck_cost[i][j] *= (1 - rho)
                self.tau_truck_tard[i][j] *= (1 - rho)

        if not solutions:
            return

        best_c = min(solutions, key=lambda s: s.cost)
        best_t = min(solutions, key=lambda s: s.tardiness)

        dc = self.Q_cost / max(best_c.cost, 1.0)
        for rt in best_c.truck_routes:
            prev = 0
            for cid in rt:
                self.tau_truck_cost[prev][cid] += dc
                prev = cid

        dt = self.Q_tard / max(best_t.tardiness, 1.0)
        for rt in best_t.truck_routes:
            prev = 0
            for cid in rt:
                self.tau_truck_tard[prev][cid] += dt
                prev = cid

    def solve(self):
        """Run P-ACO and return all solutions + Pareto front."""
        all_solutions = []

        for _ in range(self.max_iter):
            iter_sols = [self._construct_solution() for _ in range(self.n_ants)]
            self._update_pheromone(iter_sols)
            all_solutions.extend(iter_sols)

        pareto = extract_pareto_front(all_solutions)
        return all_solutions, pareto


def run_paco(instance, n_runs=10, seed=42):
    """Run P-ACO multiple times."""
    all_solutions = []
    times = []

    for run in range(n_runs):
        s = seed + run
        t0 = time.time()
        solver = PACOSolver(instance, seed=s)
        sols, pareto = solver.solve()
        elapsed = time.time() - t0
        times.append(elapsed)
        all_solutions.extend(sols)

    final_pareto = extract_pareto_front(all_solutions)
    return {
        'solutions': all_solutions,
        'pareto_front': final_pareto,
        'mean_runtime': sum(times) / max(len(times), 1),
        'std_runtime': (sum((t - sum(times)/len(times))**2 for t in times) / max(len(times)-1, 1))**0.5 if len(times) > 1 else 0,
    }
