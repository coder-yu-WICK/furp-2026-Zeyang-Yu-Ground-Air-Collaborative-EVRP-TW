# -*- coding: utf-8 -*-
"""
ALNS (Adaptive Large Neighborhood Search) Baseline for Truck-Drone VRPTW.

Implements the ALNS framework of Ropke & Pisinger (2006) adapted for
truck-drone collaborative routing with time windows.

Destroy operators:
  - random_removal: remove q random customers
  - worst_removal: remove q customers with highest tardiness contribution
  - shaw_removal: remove q related customers (similar TW + location)
  - route_removal: remove an entire route

Repair operators:
  - greedy_insertion: insert each removed customer at best position
  - regret_k_insertion: insert considering regret (k=2,3)

Adaptive weights via roulette wheel selection based on operator scores.
Simulated annealing acceptance criterion.
"""

import random
import math
import copy
import time
import numpy as np

from utils.problem_model import TruckDroneSolution, extract_pareto_front


# ── Destroy Operators ──────────────────────────────────────────────────

def _random_removal(routes, q, customers, depot, rng):
    """Remove q random customers from routes."""
    all_positions = []
    for ri, route in enumerate(routes):
        for pi in range(len(route)):
            all_positions.append((ri, pi))

    if len(all_positions) <= q:
        # Remove everything
        removed = []
        for ri, route in enumerate(routes):
            for cid in route:
                removed.append((ri, cid))
            routes[ri] = []
        return removed

    selected = rng.sample(all_positions, q)
    selected.sort(key=lambda x: (x[0], -x[1]))  # sort by route, reverse position

    removed = []
    for ri, pi in selected:
        cid = routes[ri].pop(pi)
        removed.append((ri, cid))

    return removed


def _worst_removal(routes, q, customers, depot, rng):
    """Remove q customers with highest tardiness contribution."""
    depot_coords = depot
    truck_speed = 35.0

    # Compute per-customer tardiness
    scored = []
    for ri, route in enumerate(routes):
        if len(route) <= 1:
            continue
        current_time = 0.0
        prev_x, prev_y = depot_coords[0], depot_coords[1]
        for pi, cid in enumerate(route):
            c = customers[cid - 1]
            dx = prev_x - c['x']
            dy = prev_y - c['y']
            travel = math.sqrt(dx*dx + dy*dy) / truck_speed
            current_time = max(current_time + travel, c['ready_time'])
            current_time += c['service_time']
            tard = max(0.0, current_time - c['due_time'])
            scored.append((tard, ri, pi))
            prev_x, prev_y = c['x'], c['y']

    scored.sort(key=lambda x: -x[0])  # highest tardiness first
    to_remove = scored[:min(q, len(scored))]
    to_remove.sort(key=lambda x: (x[1], -x[2]))  # sort by route, reverse position

    removed = []
    for _, ri, pi in to_remove:
        if pi < len(routes[ri]):
            cid = routes[ri].pop(pi)
            removed.append((ri, cid))

    return removed


def _shaw_removal(routes, q, customers, depot, rng):
    """Remove q related customers (similar TW + spatial proximity)."""
    all_positions = []
    for ri, route in enumerate(routes):
        for pi, cid in enumerate(route):
            all_positions.append((ri, pi, cid))

    if len(all_positions) <= q:
        removed = []
        for ri, route in enumerate(routes):
            for cid in route:
                removed.append((ri, cid))
            routes[ri] = []
        return removed

    # Pick a random seed customer
    seed_ri, seed_pi, seed_cid = rng.choice(all_positions)
    seed_c = customers[seed_cid - 1]

    # Score all other customers by relatedness
    relatedness = []
    for ri, pi, cid in all_positions:
        if ri == seed_ri and pi == seed_pi:
            continue
        c = customers[cid - 1]
        # Spatial distance
        d_spatial = math.sqrt((seed_c['x'] - c['x'])**2 + (seed_c['y'] - c['y'])**2)
        # TW distance
        seed_mid = (seed_c['ready_time'] + seed_c['due_time']) / 2
        c_mid = (c['ready_time'] + c['due_time']) / 2
        d_tw = abs(seed_mid - c_mid) / 240.0
        # Combined relatedness (lower = more related)
        rel = d_spatial / 16.0 + d_tw
        relatedness.append((rel, ri, pi))

    relatedness.sort(key=lambda x: x[0])  # most related first
    to_remove = [(seed_ri, seed_pi)]
    for _, ri, pi in relatedness[:q-1]:
        to_remove.append((ri, pi))

    to_remove.sort(key=lambda x: (x[0], -x[1]))

    removed = []
    for ri, pi in to_remove:
        if pi < len(routes[ri]):
            cid = routes[ri].pop(pi)
            removed.append((ri, cid))

    return removed


def _route_removal(routes, q, customers, depot, rng):
    """Remove an entire route (all its customers)."""
    valid = [ri for ri, route in enumerate(routes) if route]
    if not valid:
        return []

    ri = rng.choice(valid)
    removed = [(ri, cid) for cid in routes[ri]]
    routes[ri] = []
    return removed


# ── Repair Operators ───────────────────────────────────────────────────

def _greedy_insertion(routes, removed, instance, rng):
    """
    Greedy insertion: for each removed customer, find the position
    that minimizes total cost increase.
    """
    customers = instance['customers']
    depot = instance['depot']

    # Shuffle removed for order randomness
    removed_shuffled = list(removed)
    rng.shuffle(removed_shuffled)

    for _, cid in removed_shuffled:
        best_route = None
        best_pos = None
        best_cost = float('inf')

        for ri in range(len(routes)):
            if not _can_insert(routes[ri], cid, customers):
                continue
            for pos in range(len(routes[ri]) + 1):
                test_route = list(routes[ri])
                test_route.insert(pos, cid)
                cost = _route_cost_truck([test_route], customers, depot)
                if cost < best_cost:
                    best_cost = cost
                    best_route = ri
                    best_pos = pos

        # Always have option to create a new route
        if best_route is not None:
            routes[best_route].insert(best_pos, cid)
        else:
            routes.append([cid])

    return routes


def _regret_k_insertion(routes, removed, instance, rng, k=2):
    """
    Regret-k insertion: insert the customer with highest regret first.
    Regret = (2nd best cost - best cost).
    """
    customers = instance['customers']
    depot = instance['depot']

    remaining = list(removed)  # list of (orig_ri, cid)

    while remaining:
        best_regret = -float('inf')
        best_to_insert = None
        best_insertion = None

        for rem_idx, (_, cid) in enumerate(remaining):
            # Find k best insertion positions
            insertion_costs = []
            for ri in range(len(routes)):
                if not _can_insert(routes[ri], cid, customers):
                    continue
                for pos in range(len(routes[ri]) + 1):
                    test_route = list(routes[ri])
                    test_route.insert(pos, cid)
                    cost = _route_cost_truck([test_route], customers, depot)
                    insertion_costs.append((cost, ri, pos))

            # Also try new route
            cost = _route_cost_truck([[cid]], customers, depot)
            insertion_costs.append((cost, len(routes), 0))

            insertion_costs.sort(key=lambda x: x[0])

            if len(insertion_costs) >= k:
                regret = insertion_costs[k-1][0] - insertion_costs[0][0]
            else:
                regret = 0

            if regret > best_regret:
                best_regret = regret
                best_to_insert = rem_idx
                best_insertion = insertion_costs[0]

        if best_to_insert is not None:
            _, cid = remaining.pop(best_to_insert)
            _, ri, pos = best_insertion
            if ri == len(routes):
                routes.append([cid])
            else:
                routes[ri].insert(pos, cid)

    return routes


# ── Cost Helpers ────────────────────────────────────────────────────────

def _route_cost_truck(routes, customers, depot, capacity=200.0):
    """Truck cost: distance*2.0 + tardiness*5.0 + unserved*10000 + cap_violation*1000."""
    truck_speed = 35.0
    total_dist = 0.0
    total_tard = 0.0
    total_cap_violation = 0.0

    for route in routes:
        if not route:
            continue
        current_time = 0.0
        prev_x, prev_y = depot[0], depot[1]
        route_load = 0.0

        for cid in route:
            c = customers[cid - 1]
            dx = prev_x - c['x']
            dy = prev_y - c['y']
            dist = math.sqrt(dx*dx + dy*dy)
            total_dist += dist

            current_time += dist / truck_speed
            current_time = max(current_time, c['ready_time'])
            current_time += c['service_time']
            if current_time > c['due_time']:
                total_tard += current_time - c['due_time']

            route_load += c['demand']
            prev_x, prev_y = c['x'], c['y']

        if route_load > capacity:
            total_cap_violation += route_load - capacity

        # Return to depot
        dx = prev_x - depot[0]
        dy = prev_y - depot[1]
        total_dist += math.sqrt(dx*dx + dy*dy)

    return total_dist * 2.0 + total_tard * 5.0 + total_cap_violation * 1000.0


def _can_insert(route, cid, customers, capacity=200.0):
    """Check if customer can be inserted into route without exceeding capacity."""
    route_load = sum(customers[rcid - 1]['demand'] for rcid in route)
    c_demand = customers[cid - 1]['demand']
    return route_load + c_demand <= capacity


# ── Initial Solution ───────────────────────────────────────────────────

def _initial_solution(instance, n_trucks, seed=42):
    """
    Generate initial solution using nearest-neighbor heuristic + EDD ordering.
    """
    customers = instance['customers']
    depot = instance['depot']
    n = len(customers)

    # Split customers evenly among trucks
    rng = random.Random(seed)
    all_ids = list(range(1, n + 1))
    rng.shuffle(all_ids)

    # Simple: assign to n_trucks via round-robin
    routes = [[] for _ in range(n_trucks)]
    for i, cid in enumerate(all_ids):
        routes[i % n_trucks].append(cid)

    # Apply EDD ordering to each route
    for ri in range(n_trucks):
        routes[ri] = sorted(routes[ri],
            key=lambda cid: customers[cid-1]['due_time'])

    # Insert drones
    try:
        from drone_post_processing import insert_cross_route_drones
        final_routes, new_drones, _, n_drones = insert_cross_route_drones(
            routes, instance, drone_endurance=4.0)
        sol = TruckDroneSolution(final_routes, new_drones, instance)
    except Exception:
        sol = TruckDroneSolution(routes, [], instance)

    return sol


# ── ALNS Solver ─────────────────────────────────────────────────────────

class ALNSSolver:
    """
    Adaptive Large Neighborhood Search for Truck-Drone VRPTW.

    Uses adaptive weight adjustment based on operator performance,
    simulated annealing acceptance, and drone post-processing.
    """

    def __init__(self, instance, n_trucks, seed=42):
        self.instance = instance
        self.n_trucks = n_trucks
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        self.customers = instance['customers']
        self.depot = instance['depot']

        # Destroy operators
        self.destroy_ops = [
            ('random_removal', _random_removal),
            ('worst_removal', _worst_removal),
            ('shaw_removal', _shaw_removal),
            ('route_removal', _route_removal),
        ]

        # Repair operators
        self.repair_ops = [
            ('greedy_insertion', _greedy_insertion),
            ('regret2', lambda r, rem, inst, rng: _regret_k_insertion(r, rem, inst, rng, k=2)),
            ('regret3', lambda r, rem, inst, rng: _regret_k_insertion(r, rem, inst, rng, k=3)),
        ]

        # Operator weights and scores
        self.destroy_weights = np.ones(len(self.destroy_ops))
        self.repair_weights = np.ones(len(self.repair_ops))
        self.destroy_scores = np.zeros(len(self.destroy_ops))
        self.repair_scores = np.zeros(len(self.repair_ops))
        self.destroy_counts = np.zeros(len(self.destroy_ops))
        self.repair_counts = np.zeros(len(self.repair_ops))

        # Score increments
        self.SCORE_NEW_BEST = 30
        self.SCORE_BETTER = 10
        self.SCORE_ACCEPTED = 5

        # Weight update parameter
        self.react_factor = 0.1

    def _select_operator(self, weights, rng):
        """Roulette wheel selection."""
        total = np.sum(weights)
        if total == 0:
            return rng.randint(0, len(weights) - 1)
        probs = weights / total
        return rng.choices(range(len(weights)), weights=probs)[0]

    def _update_weights(self):
        """Update operator weights based on accumulated scores."""
        for i in range(len(self.destroy_ops)):
            if self.destroy_counts[i] > 0:
                self.destroy_weights[i] = (
                    self.destroy_weights[i] * (1 - self.react_factor)
                    + self.react_factor * self.destroy_scores[i] / self.destroy_counts[i]
                )
        self.destroy_scores.fill(0)
        self.destroy_counts.fill(0)

        for i in range(len(self.repair_ops)):
            if self.repair_counts[i] > 0:
                self.repair_weights[i] = (
                    self.repair_weights[i] * (1 - self.react_factor)
                    + self.react_factor * self.repair_scores[i] / self.repair_counts[i]
                )
        self.repair_scores.fill(0)
        self.repair_counts.fill(0)

    def _removal_quantity(self, n_customers):
        """Number of customers to remove: ~5-15% of total."""
        min_q = max(1, n_customers // 20)
        max_q = max(min_q + 1, n_customers // 8)
        return self.rng.randint(min_q, max_q + 1)

    def solve(self, max_iter=2000, T_start=100.0, T_end=0.01, cooling=0.995,
              segment_size=100, initial_solution=None):
        """
        Main ALNS loop.

        Args:
            max_iter: maximum iterations
            T_start: initial temperature
            T_end: final temperature
            cooling: temperature decay per iteration
            segment_size: iterations between weight updates
            initial_solution: optional TruckDroneSolution to use as starting point.
                              If None, generates via nearest-neighbor + EDD.

        Returns:
            dict with solutions, pareto_front, stats
        """
        t0_total = time.time()

        # Generate or use provided initial solution
        if initial_solution is not None:
            current = initial_solution
        else:
            current = _initial_solution(self.instance, self.n_trucks, seed=self.rng.randint(0, 99999))
        best = current
        best_cost = best.cost
        best_tard = best.tardiness

        T = T_start
        all_solutions = [current]
        stats = {
            'iterations': [],
            'cost': [],
            'tardiness': [],
        }

        for iteration in range(max_iter):
            # Select destroy + repair operators
            d_idx = self._select_operator(self.destroy_weights, self.rng)
            r_idx = self._select_operator(self.repair_weights, self.rng)

            d_name, d_fn = self.destroy_ops[d_idx]
            r_name, r_fn = self.repair_ops[r_idx]

            # Copy current solution
            routes = [list(r) for r in current.truck_routes]
            q = self._removal_quantity(len(self.customers))

            # Apply destroy
            removed = d_fn(routes, q, self.customers, self.depot, self.rng)

            if not removed:
                continue

            # Apply repair
            routes = r_fn(routes, removed, self.instance, self.rng)

            # Filter empty routes
            routes = [r for r in routes if r]

            # Evaluate truck-only (drone insertion is expensive, add only at end)
            candidate = TruckDroneSolution(routes, [], self.instance)
            candidate_cost = candidate.cost
            candidate_tard = candidate.tardiness

            # Acceptance decision
            accept = False
            if candidate_tard < current.tardiness:
                # Always accept if tardiness improves
                accept = True
                score = self.SCORE_BETTER
            elif candidate_tard == current.tardiness and candidate_cost < current.cost:
                accept = True
                score = self.SCORE_BETTER
            else:
                # Simulated annealing
                delta = (candidate.cost - current.cost) / max(1.0, current.cost)
                if self.rng.random() < math.exp(-delta / max(T, 1e-8)):
                    accept = True
                    score = self.SCORE_ACCEPTED
                else:
                    score = 0

            if accept:
                current = candidate
                self.destroy_scores[d_idx] += score
                self.repair_scores[r_idx] += score
                self.destroy_counts[d_idx] += 1
                self.repair_counts[r_idx] += 1

                # Update best
                if candidate_tard < best_tard or \
                   (candidate_tard == best_tard and candidate_cost < best_cost):
                    best = candidate
                    best_cost = candidate_cost
                    best_tard = candidate_tard
                    self.destroy_scores[d_idx] += self.SCORE_NEW_BEST
                    self.repair_scores[r_idx] += self.SCORE_NEW_BEST

            all_solutions.append(candidate)

            # Update temperature
            T = max(T_end, T * cooling)

            # Update weights every segment
            if (iteration + 1) % segment_size == 0:
                self._update_weights()

            # Log stats
            if iteration % 200 == 0 or iteration == max_iter - 1:
                stats['iterations'].append(iteration)
                stats['cost'].append(best_cost)
                stats['tardiness'].append(best_tard)

        pareto = extract_pareto_front(all_solutions)

        # Apply drone insertion to best solution (only at the end)
        best_with_drones = best
        try:
            from drone_post_processing import insert_cross_route_drones
            final_routes, new_drones, _, n_drones = insert_cross_route_drones(
                [list(r) for r in best.truck_routes], self.instance, drone_endurance=4.0)
            best_with_drones = TruckDroneSolution(final_routes, new_drones, self.instance)
        except Exception:
            pass

        return {
            'solutions': all_solutions + [best_with_drones],
            'pareto_front': pareto,
            'best_solution': best_with_drones,
            'mean_runtime': time.time() - t0_total,
            'stats': stats,
            'final_weights': {
                'destroy': dict(zip([d[0] for d in self.destroy_ops],
                                   self.destroy_weights.tolist())),
                'repair': dict(zip([r[0] for r in self.repair_ops],
                                  self.repair_weights.tolist())),
            },
        }


# ── Week3-compatible Interface ─────────────────────────────────────────

def run_alns(instance, n_runs=1, n_trucks=2, n_drones=0,
             endurance=None, seed=42, max_iter=2000, **kwargs):
    """
    Interface compatible with week3 experiment runner.

    Returns dict with solutions, pareto_front, mean_runtime.
    """
    all_solutions = []
    times = []

    for run in range(n_runs):
        t0 = time.time()
        solver = ALNSSolver(instance, n_trucks=n_trucks, seed=seed + run)
        result = solver.solve(max_iter=max_iter)
        all_solutions.extend(result['solutions'])
        times.append(result['mean_runtime'])

    pareto = extract_pareto_front(all_solutions)

    return {
        'solutions': all_solutions,
        'pareto_front': pareto,
        'mean_runtime': sum(times) / max(len(times), 1),
        'std_runtime': (sum((t - sum(times)/len(times))**2 for t in times) / max(len(times)-1, 1))**0.5
                        if len(times) > 1 else 0,
    }
