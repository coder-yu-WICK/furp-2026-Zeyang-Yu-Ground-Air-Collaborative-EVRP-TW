# -*- coding: utf-8 -*-
"""
IVND: Improved Variable Neighborhood Descent for Truck-Drone Routing.
Based on DOI: 10.1109/TITS.2022.3181282

Key features:
  - K-means clustering + Nearest Neighbor initial solution
  - 7 neighborhood structures for truck and drone routes
  - Metropolis acceptance criterion (Simulated Annealing)
  - Tabu list to avoid cycling
  - Systematic neighborhood search with shaking
"""

import random
import math
import time
import copy

from config import (
    TRUCK_SPEED, DRONE_SPEED,
    TRUCK_CAPACITY, DRONE_CAPACITY,
    TRUCK_FIXED_COST, DRONE_FIXED_COST,
    TRUCK_DIST_COST_RATE, DRONE_DIST_COST_RATE,
    TARDINESS_COST_RATE, DEPOT,
    IVND,
)
from utils.problem_model import TruckDroneSolution, extract_pareto_front


class IVNDSolver:
    """IVND solver for truck-drone collaborative routing."""

    def __init__(self, instance, n_trucks=2, n_drones=2, endurance=4.0, seed=42):
        self.instance = instance
        self.customers = instance['customers']
        self.dist = instance['distance_matrix']
        self.n = instance['n_customers']
        self.n_trucks = n_trucks
        self.n_drones = n_drones
        self.endurance = endurance
        self.seed = seed
        random.seed(seed)

        cfg = IVND
        self.max_iter = cfg['max_iterations']
        self.tabu_tenure = cfg['tabu_tenure']
        self.shaking_k_max = cfg['shaking_k_max']
        self.temp_init = cfg['temperature_initial']
        self.cooling = cfg['cooling_rate']
        self.neighborhoods = cfg['neighborhood_structures']
        self.tabu_list = []
        self.tabu_iter = {}

    def _dist_depot(self, cid):
        c = self.customers[cid - 1]
        return math.sqrt((DEPOT[0] - c['x'])**2 + (DEPOT[1] - c['y'])**2)

    def _initial_solution(self):
        """
        K-means clustering initialization: cluster customers, then
        nearest-neighbor within each cluster for truck routes.
        Insert drone missions greedily.
        """
        # Simple clustering: divide customers by angular position around depot
        cust_angles = []
        for i in range(1, self.n + 1):
            c = self.customers[i - 1]
            angle = math.atan2(c['y'] - DEPOT[1], c['x'] - DEPOT[0])
            cust_angles.append((i, angle))
        cust_angles.sort(key=lambda x: x[1])

        # Divide into n_trucks clusters
        cluster_size = max(1, self.n // self.n_trucks)
        clusters = [[] for _ in range(self.n_trucks)]
        for idx, (cid, _) in enumerate(cust_angles):
            cluster_idx = min(idx // cluster_size, self.n_trucks - 1)
            clusters[cluster_idx].append(cid)

        # Nearest-neighbor within each cluster
        truck_routes = []
        for cluster in clusters:
            if not cluster:
                truck_routes.append([])
                continue
            route = []
            unvisited = set(cluster)
            current = 0  # depot
            while unvisited:
                best = min(unvisited, key=lambda cid: (
                    self.dist[current][cid] if current > 0 else self._dist_depot(cid)))
                route.append(best)
                unvisited.remove(best)
                current = best
            truck_routes.append(route)

        # Greedy drone mission insertion
        drone_missions = []
        for t, route in enumerate(truck_routes):
            new_route = []
            i = 0
            while i < len(route):
                new_route.append(route[i])
                if i < len(route) - 1:
                    # Try inserting drone between route[i] and route[i+1]
                    i_node = route[i]
                    k_node = route[i + 1]
                    # Find best drone customer between them
                    best_j = None
                    best_saving = 0
                    for j in range(1, self.n + 1):
                        if j in route or j in [m[1] for m in drone_missions]:
                            continue
                        cj = self.customers[j - 1]
                        if cj['demand'] > DRONE_CAPACITY:
                            continue
                        d_ij = self.dist[i_node][j]
                        d_jk = self.dist[j][k_node]
                        d_ik = self.dist[i_node][k_node]
                        drone_leg = d_ij + d_jk
                        if drone_leg <= self.endurance:
                            # Saving = truck detour avoided - drone cost
                            saving = (d_ik * TRUCK_DIST_COST_RATE -
                                      drone_leg * DRONE_DIST_COST_RATE)
                            if saving > best_saving:
                                best_saving = saving
                                best_j = j

                    if best_j and best_saving > 0:
                        drone_missions.append((i_node, best_j, k_node))
                        # Don't add k_node yet, will be added next iteration

                i += 1

        return truck_routes, drone_missions

    def _neighbor_relocate_truck(self, routes, missions):
        """Relocate one customer within truck routes."""
        r = copy.deepcopy(routes)
        if not r or all(len(rt) == 0 for rt in r):
            return r, missions
        # Pick random non-empty route
        valid_routes = [i for i, rt in enumerate(r) if len(rt) > 0]
        if not valid_routes:
            return r, missions
        t1 = random.choice(valid_routes)
        if len(r[t1]) == 0:
            return r, missions
        pos = random.randrange(len(r[t1]))
        cust = r[t1].pop(pos)
        # Insert into random position in same or another route
        t2 = random.randrange(len(r))
        insert_pos = random.randrange(len(r[t2]) + 1) if r[t2] else 0
        r[t2].insert(insert_pos, cust)
        return r, missions

    def _neighbor_relocate_drone(self, routes, missions):
        """Change a drone mission's launch or recovery node."""
        if not missions:
            return routes, missions
        m = copy.deepcopy(missions)
        idx = random.randrange(len(m))
        i, j, k = m[idx]
        # Change recovery node
        all_nodes = [0] + [c for rt in routes for c in rt]
        if all_nodes:
            new_k = random.choice([n for n in all_nodes if n != j and n != i])
            m[idx] = (i, j, new_k)
        return routes, m

    def _neighbor_swap_truck(self, routes, missions):
        """Swap two customers in truck routes."""
        r = copy.deepcopy(routes)
        all_custs = [c for rt in r for c in rt]
        if len(all_custs) < 2:
            return r, missions
        c1, c2 = random.sample(all_custs, 2)
        # Find and swap
        for rt in r:
            for idx in range(len(rt)):
                if rt[idx] == c1:
                    rt[idx] = c2
                elif rt[idx] == c2:
                    rt[idx] = c1
        return r, missions

    def _neighbor_swap_drone(self, routes, missions):
        """Swap drone mission customer with a truck customer."""
        if not missions:
            return routes, missions
        r = copy.deepcopy(routes)
        m = copy.deepcopy(missions)
        # Pick a drone mission
        idx = random.randrange(len(m))
        drone_j = m[idx][1]
        # Pick a random truck customer
        all_truck = [c for rt in r for c in rt]
        if not all_truck:
            return r, m
        truck_c = random.choice(all_truck)
        # Swap
        m[idx] = (m[idx][0], truck_c, m[idx][2])
        for rt in r:
            for i in range(len(rt)):
                if rt[i] == truck_c:
                    rt[i] = drone_j
        return r, m

    def _neighbor_two_opt_truck(self, routes, missions):
        """2-opt within a single truck route."""
        r = copy.deepcopy(routes)
        valid = [i for i, rt in enumerate(r) if len(rt) >= 2]
        if not valid:
            return r, missions
        t = random.choice(valid)
        i, j = sorted(random.sample(range(len(r[t])), 2))
        r[t][i:j+1] = reversed(r[t][i:j+1])
        return r, missions

    def _neighbor_drone_to_truck(self, routes, missions):
        """Convert a drone-served customer to truck service."""
        if not missions:
            return routes, missions
        r = copy.deepcopy(routes)
        m = copy.deepcopy(missions)
        idx = random.randrange(len(m))
        _, j, _ = m.pop(idx)
        # Add j to a random truck route
        if r:
            t = random.randrange(len(r))
            r[t].append(j)
        return r, m

    def _neighbor_truck_to_drone(self, routes, missions):
        """Convert a truck-served customer to drone mission."""
        r = copy.deepcopy(routes)
        m = copy.deepcopy(missions)
        all_truck = [(t, i) for t, rt in enumerate(r)
                     for i, c in enumerate(rt)]
        if not all_truck:
            return r, m
        t, pos = random.choice(all_truck)
        j = r[t].pop(pos)
        # Create drone mission: depot→j→depot
        if self._dist_depot(j) * 2 <= self.endurance:
            m.append((0, j, 0))
        else:
            r[t].insert(pos, j)  # revert if infeasible
        return r, m

    def _apply_neighborhood(self, name, routes, missions):
        """Apply a specific neighborhood structure."""
        methods = {
            'relocate_truck': self._neighbor_relocate_truck,
            'relocate_drone': self._neighbor_relocate_drone,
            'swap_truck': self._neighbor_swap_truck,
            'swap_drone': self._neighbor_swap_drone,
            'two_opt_truck': self._neighbor_two_opt_truck,
            'drone_to_truck': self._neighbor_drone_to_truck,
            'truck_to_drone': self._neighbor_truck_to_drone,
        }
        if name in methods:
            return methods[name](routes, missions)
        return routes, missions

    def _solution_hash(self, routes, missions):
        """Create a hashable representation for tabu list."""
        route_tuple = tuple(tuple(rt) for rt in routes)
        mission_tuple = tuple(sorted(missions))
        return hash((route_tuple, mission_tuple))

    def _is_tabu(self, routes, missions):
        """Check if a solution is tabu."""
        h = self._solution_hash(routes, missions)
        return h in self.tabu_list

    def _add_tabu(self, routes, missions):
        """Add solution to tabu list with tenure."""
        h = self._solution_hash(routes, missions)
        self.tabu_list.append(h)
        self.tabu_iter[h] = self.iteration
        # Remove expired tabu
        while len(self.tabu_list) > self.tabu_tenure * 2:
            old = self.tabu_list.pop(0)
            self.tabu_iter.pop(old, None)

    def _acceptance_probability(self, new_cost, current_cost, temp):
        """Metropolis acceptance criterion."""
        if new_cost <= current_cost:
            return 1.0
        return math.exp(-(new_cost - current_cost) / max(temp, 0.001))

    def solve(self):
        """Run IVND and return Pareto front solutions."""
        all_solutions = []
        pareto_archive = []

        for restart in range(3):  # Multiple restarts for diversity
            routes, missions = self._initial_solution()
            current_sol = TruckDroneSolution(routes, missions, self.instance)
            best_sol = current_sol
            temperature = self.temp_init

            for self.iteration in range(self.max_iter // 3):
                improved = False

                # Shaking phase
                k = 1
                while k <= self.shaking_k_max and not improved:
                    # Apply k random neighborhoods
                    r_shake, m_shake = copy.deepcopy(routes), copy.deepcopy(missions)
                    for _ in range(k):
                        neigh = random.choice(self.neighborhoods)
                        r_shake, m_shake = self._apply_neighborhood(neigh, r_shake, m_shake)

                    if self._is_tabu(r_shake, m_shake):
                        k += 1
                        continue

                    # Evaluate shaken solution
                    new_sol = TruckDroneSolution(r_shake, m_shake, self.instance)
                    current_obj = current_sol.cost + current_sol.tardiness
                    new_obj = new_sol.cost + new_sol.tardiness

                    # Metropolis acceptance
                    if (new_sol.feasible and
                        (self._acceptance_probability(new_obj, current_obj, temperature) >
                         random.random())):
                        routes, missions = r_shake, m_shake
                        current_sol = new_sol
                        self._add_tabu(routes, missions)
                        improved = True

                        if new_obj < best_sol.cost + best_sol.tardiness:
                            best_sol = new_sol

                    k += 1

                if not improved:
                    # Local search on best neighborhoods
                    for neigh in self.neighborhoods[:5]:
                        r_ls, m_ls = self._apply_neighborhood(neigh, routes, missions)
                        if not self._is_tabu(r_ls, m_ls):
                            ls_sol = TruckDroneSolution(r_ls, m_ls, self.instance)
                            ls_obj = ls_sol.cost + ls_sol.tardiness
                            if ls_sol.feasible and ls_obj < (current_sol.cost + current_sol.tardiness):
                                routes, missions = r_ls, m_ls
                                current_sol = ls_sol
                                self._add_tabu(routes, missions)

                # Cool down
                temperature *= self.cooling

            # Add to archive
            all_solutions.append(current_sol)
            pareto_archive.append(best_sol)

        all_solutions.extend(pareto_archive)
        pareto = extract_pareto_front(all_solutions)
        return all_solutions, pareto


def run_ivnd(instance, n_trucks=2, n_drones=2, endurance=4.0, n_runs=10, seed=42):
    """Run IVND multiple times."""
    all_solutions = []
    times = []

    for run in range(n_runs):
        s = seed + run
        t0 = time.time()
        solver = IVNDSolver(instance, n_trucks=n_trucks, n_drones=n_drones,
                            endurance=endurance, seed=s)
        sols, pareto = solver.solve()
        elapsed = time.time() - t0
        times.append(elapsed)
        all_solutions.extend(sols)

    final_pareto = extract_pareto_front(all_solutions)
    return {
        'solutions': all_solutions,
        'pareto_front': final_pareto,
        'mean_runtime': sum(times) / len(times),
        'std_runtime': (sum((t - sum(times)/len(times))**2 for t in times) / max(len(times)-1, 1))**0.5 if len(times) > 1 else 0,
    }
