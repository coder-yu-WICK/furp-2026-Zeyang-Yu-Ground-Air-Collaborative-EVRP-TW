# -*- coding: utf-8 -*-
"""
NSGA-II: Non-dominated Sorting Genetic Algorithm II for Truck-Only Routing.
Based on Deb et al. (2002), adapted for truck-only routing.

Key features:
  - Non-dominated sorting for Pareto-based selection
  - Crowding distance for diversity preservation
  - SBX crossover + polynomial mutation
  - Chromosome encodes truck routes (permutation only, no drones)
"""

import random
import math
import time
import copy
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from week8.config import (
    TRUCK_SPEED,
    TRUCK_CAPACITY,
    TRUCK_FIXED_COST,
    TRUCK_DIST_COST_RATE,
    TARDINESS_COST_RATE,
    DEPOT,
    NSGA2,
)
from week8.core.problem_model import TruckSolution, extract_pareto_front


class NSGA2Solver:
    """NSGA-II solver for truck-only routing."""

    def __init__(self, instance, n_trucks=2, seed=42):
        self.instance = instance
        self.customers = instance['customers']
        self.dist = instance['distance_matrix']
        self.n = instance['n_customers']
        self.n_trucks = n_trucks
        self.seed = seed
        random.seed(seed)

        cfg = NSGA2
        self.pop_size = (cfg['pop_25c'] if self.n <= 25 else
                         (cfg['pop_50c'] if self.n <= 50 else cfg['pop_100c']))
        self.max_gen = cfg['generations']
        self.cx_pb = cfg['crossover_pb']
        self.mut_pb = cfg['mutation_pb']
        self.cx_eta = cfg['crossover_eta']
        self.mut_eta = cfg['mutation_eta']

    def _random_chromosome(self):
        """
        Generate random chromosome.
        Format: [permutation_of_customers]
        Chromosome is just the permutation -- no drone flags.
        """
        perm = list(range(1, self.n + 1))
        random.shuffle(perm)
        return perm

    def _decode(self, chrom):
        """
        Decode chromosome to (truck_routes, []).
        All customers are served by truck.
        """
        truck_customers = list(chrom)
        truck_routes = self._build_truck_routes(truck_customers)
        return truck_routes, []

    def _dist_depot(self, cid):
        c = self.customers[cid - 1]
        return math.sqrt((DEPOT[0] - c['x'])**2 + (DEPOT[1] - c['y'])**2)

    def _build_truck_routes(self, customers_list):
        """Split customer list into feasible routes for n_trucks."""
        if not customers_list:
            return [[] for _ in range(self.n_trucks)]

        routes = [[] for _ in range(self.n_trucks)]
        loads = [0.0] * self.n_trucks
        times = [0.0] * self.n_trucks
        positions = [0] * self.n_trucks

        for cid in customers_list:
            c = self.customers[cid - 1]
            # Find best truck to assign
            best_t = 0
            best_extra = float('inf')
            for t in range(self.n_trucks):
                if loads[t] + c['demand'] > TRUCK_CAPACITY:
                    continue
                d = self.dist[positions[t]][cid] if positions[t] > 0 else self._dist_depot(cid)
                if d < best_extra:
                    best_extra = d
                    best_t = t

            routes[best_t].append(cid)
            loads[best_t] += c['demand']
            positions[best_t] = cid

        return routes

    def _sbx_crossover(self, p1, p2):
        """Simulated Binary Crossover on the permutation chromosome."""
        n = self.n
        c1, c2 = p1[:], p2[:]

        for i in range(n):
            if random.random() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-6:
                    u = random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (self.cx_eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (self.cx_eta + 1))

                    c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                    c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])

        # Repair: ensure valid permutation
        perm1 = self._repair_permutation(c1[:n])
        perm2 = self._repair_permutation(c2[:n])

        return perm1, perm2

    def _repair_permutation(self, vals):
        """Repair to valid permutation of 1..n using order-based mapping."""
        n = self.n
        indexed = [(vals[i], i) for i in range(n)]
        indexed.sort(key=lambda x: x[0])
        result = [0] * n
        for rank, (_, orig_idx) in enumerate(indexed):
            result[orig_idx] = rank + 1
        return result

    def _mutate(self, chrom):
        """Swap mutation for permutation chromosome."""
        n = self.n
        mut = chrom[:]

        # Swap mutation
        if random.random() < self.mut_pb:
            i, j = random.sample(range(n), 2)
            mut[i], mut[j] = mut[j], mut[i]

        return mut

    def _non_dominated_sort(self, solutions):
        """Non-dominated sorting: returns list of fronts."""
        fronts = []
        dominated_by = {i: 0 for i in range(len(solutions))}
        dominates = {i: [] for i in range(len(solutions))}

        for i in range(len(solutions)):
            for j in range(len(solutions)):
                if i == j:
                    continue
                if solutions[i].dominates(solutions[j]):
                    dominates[i].append(j)
                elif solutions[j].dominates(solutions[i]):
                    dominated_by[i] += 1

        current_front = [i for i in range(len(solutions)) if dominated_by[i] == 0]

        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in dominates[i]:
                    dominated_by[j] -= 1
                    if dominated_by[j] == 0:
                        next_front.append(j)
            current_front = next_front

        return fronts

    def _crowding_distance(self, solutions, front_indices):
        """Compute crowding distance for solutions in a front."""
        if len(front_indices) <= 2:
            return {i: float('inf') for i in front_indices}

        distances = {i: 0.0 for i in front_indices}
        front_sols = [solutions[i] for i in front_indices]

        for obj_idx in [0, 1]:  # cost, tardiness
            sorted_pairs = sorted(
                zip(front_indices, front_sols),
                key=lambda x: x[1].objectives[obj_idx]
            )
            obj_min = sorted_pairs[0][1].objectives[obj_idx]
            obj_max = sorted_pairs[-1][1].objectives[obj_idx]
            obj_range = obj_max - obj_min if obj_max > obj_min else 1.0

            distances[sorted_pairs[0][0]] = float('inf')
            distances[sorted_pairs[-1][0]] = float('inf')

            for k in range(1, len(sorted_pairs) - 1):
                prev_obj = sorted_pairs[k - 1][1].objectives[obj_idx]
                next_obj = sorted_pairs[k + 1][1].objectives[obj_idx]
                distances[sorted_pairs[k][0]] += (next_obj - prev_obj) / obj_range

        return distances

    def solve(self):
        """Run NSGA-II and return all solutions + Pareto front."""
        # Initialize population
        population = [self._random_chromosome() for _ in range(self.pop_size)]
        all_solutions = []

        for gen in range(self.max_gen):
            # Decode and evaluate
            decoded = [self._decode(chrom) for chrom in population]
            solutions = [TruckSolution(r, self.instance)
                        for r, _ in decoded]
            all_solutions.extend(solutions)

            # Non-dominated sorting
            fronts = self._non_dominated_sort(solutions)

            # Build next generation
            new_pop = []
            for front in fronts:
                if len(new_pop) + len(front) <= self.pop_size:
                    new_pop.extend([population[i] for i in front])
                else:
                    # Crowding distance for remaining slots
                    distances = self._crowding_distance(solutions, front)
                    sorted_front = sorted(front, key=lambda i: distances[i], reverse=True)
                    remaining = self.pop_size - len(new_pop)
                    new_pop.extend([population[i] for i in sorted_front[:remaining]])
                    break

            # Selection, crossover, mutation
            offspring = []
            while len(offspring) < self.pop_size:
                # Tournament selection
                t1, t2 = random.sample(range(len(new_pop)), 2)
                parent1 = new_pop[t1]
                parent2 = new_pop[t2]

                if random.random() < self.cx_pb:
                    child1, child2 = self._sbx_crossover(parent1, parent2)
                    offspring.append(self._mutate(child1))
                    offspring.append(self._mutate(child2))
                else:
                    offspring.append(parent1[:])
                    offspring.append(parent2[:])

            population = offspring[:self.pop_size]

        # Final evaluation
        final_decoded = [self._decode(chrom) for chrom in population]
        final_solutions = [TruckSolution(r, self.instance)
                          for r, _ in final_decoded]
        all_solutions.extend(final_solutions)

        pareto = extract_pareto_front(all_solutions)
        return all_solutions, pareto


def run_nsga2(instance, n_trucks=2, n_runs=10, seed=42):
    """Run NSGA-II multiple times."""
    all_solutions = []
    times = []

    for run in range(n_runs):
        s = seed + run
        t0 = time.time()
        solver = NSGA2Solver(instance, n_trucks=n_trucks, seed=s)
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
