# -*- coding: utf-8 -*-
"""
No-Drone Baseline: Pure truck delivery using GA (from py-ga-VRPTW adapted to multi-objective).
Solves standard VRPTW - trucks only, no drones.
"""

import random
import math
import time
import copy

from src.config import (
    TRUCK_SPEED, TRUCK_CAPACITY, TRUCK_FIXED_COST, TRUCK_DIST_COST_RATE,
    TARDINESS_COST_RATE, DEPOT,
)
from week8.core.problem_model import TruckSolution, extract_pareto_front


def _make_solution(truck_routes, instance):
    """Create a TruckSolution with no drone missions."""
    return TruckSolution(truck_routes, instance)


def _greedy_init(instance, customers):
    """Nearest-neighbor greedy initialization for truck routes."""
    n = instance['n_customers']
    unserved = set(range(1, n + 1))
    routes = []

    while unserved:
        route = []
        load = 0.0
        current_time = 0.0
        prev = 0  # depot

        while unserved:
            # Find nearest feasible unserved customer
            best = None
            best_dist = float('inf')
            for cid in unserved:
                c = customers[cid - 1]
                # Check capacity
                if load + c['demand'] > TRUCK_CAPACITY:
                    continue
                # Distance from current position
                if prev == 0:
                    d = math.sqrt((DEPOT[0] - c['x'])**2 + (DEPOT[1] - c['y'])**2)
                else:
                    d = instance['distance_matrix'][prev][cid]
                if d < best_dist:
                    best_dist = d
                    best = cid

            if best is None:
                break

            c = customers[best - 1]
            if prev == 0:
                travel_time = math.sqrt((DEPOT[0] - c['x'])**2 + (DEPOT[1] - c['y'])**2) / TRUCK_SPEED
            else:
                travel_time = instance['distance_matrix'][prev][best] / TRUCK_SPEED

            arrival = current_time + travel_time
            if arrival < c['ready_time']:
                arrival = c['ready_time']

            # Check time window feasibility for return
            return_time = (math.sqrt((DEPOT[0] - c['x'])**2 + (DEPOT[1] - c['y'])**2)
                           / TRUCK_SPEED if prev != 0 else travel_time)

            route.append(best)
            unserved.remove(best)
            load += c['demand']
            current_time = arrival + c['service_time']
            prev = best

        if route:
            routes.append(route)

    return routes


def _crossover(parent1, parent2):
    """PMX crossover for route sequences (flattened)."""
    # Flatten routes
    flat1 = [c for r in parent1 for c in r]
    flat2 = [c for r in parent2 for c in r]

    if len(flat1) < 2 or len(flat2) < 2:
        return parent1, parent2

    size = min(len(flat1), len(flat2))
    cx1, cx2 = sorted(random.sample(range(size), 2))

    child1_flat = flat1[:]
    child2_flat = flat2[:]

    # Swap segments
    mapping1 = {}
    mapping2 = {}
    for i in range(cx1, cx2 + 1):
        mapping1[flat2[i]] = flat1[i]
        mapping2[flat1[i]] = flat2[i]
        child1_flat[i] = flat2[i]
        child2_flat[i] = flat1[i]

    # Repair
    for i in list(range(cx1)) + list(range(cx2 + 1, size)):
        while child1_flat[i] in mapping1:
            child1_flat[i] = mapping1[child1_flat[i]]
        while child2_flat[i] in mapping2:
            child2_flat[i] = mapping2[child2_flat[i]]

    return child1_flat, child2_flat


def _mutate(flat_route):
    """Inversion mutation."""
    if len(flat_route) < 2:
        return flat_route
    i, j = sorted(random.sample(range(len(flat_route)), 2))
    flat_route[i:j+1] = reversed(flat_route[i:j+1])
    return flat_route


def _split_to_routes(flat_route, instance):
    """Split flat customer sequence into feasible routes (capacity + TW aware)."""
    customers = instance['customers']
    dist = instance['distance_matrix']
    routes = []

    route = []
    load = 0.0
    current_time = 0.0
    prev = 0  # depot

    for cid in flat_route:
        c = customers[cid - 1]
        d = dist[prev][cid]
        travel_time = d / TRUCK_SPEED
        arrival = current_time + travel_time
        if arrival < c['ready_time']:
            arrival = c['ready_time']

        return_to_depot = dist[cid][0] / TRUCK_SPEED

        # Check if feasible to add
        if (load + c['demand'] <= TRUCK_CAPACITY and
                arrival + c['service_time'] + return_to_depot <= instance['tw_horizon'] + 60):
            route.append(cid)
            load += c['demand']
            current_time = arrival + c['service_time']
            prev = cid
        else:
            if route:
                routes.append(route)
            route = [cid]
            load = c['demand']
            d0 = dist[0][cid] / TRUCK_SPEED
            current_time = d0 + c['service_time']
            prev = cid

    if route:
        routes.append(route)

    return routes


def _evaluate_truck_only(truck_routes, instance):
    """Evaluate truck-only solution. Returns (cost, tardiness, feasible)."""
    customers = instance['customers']
    dist = instance['distance_matrix']

    total_cost = 0.0
    total_tardiness = 0.0
    feasible = True
    served = set()

    n_trucks = len(truck_routes)
    total_dist = 0.0

    for route in truck_routes:
        if not route:
            continue
        prev = 0
        load = 0.0
        current_time = 0.0
        route_dist = 0.0

        for cid in route:
            served.add(cid)
            c = customers[cid - 1]
            seg = dist[prev][cid]
            route_dist += seg
            current_time += seg / TRUCK_SPEED

            if current_time < c['ready_time']:
                current_time = c['ready_time']
            if current_time > c['due_time']:
                total_tardiness += (current_time - c['due_time']) * TARDINESS_COST_RATE

            load += c['demand']
            if load > TRUCK_CAPACITY:
                feasible = False

            current_time += c['service_time']
            prev = cid

        route_dist += dist[prev][0]
        total_dist += route_dist

    vehicle_cost = n_trucks * TRUCK_FIXED_COST
    dist_cost = total_dist * TRUCK_DIST_COST_RATE
    total_cost = vehicle_cost + dist_cost

    all_customers = set(range(1, instance['n_customers'] + 1))
    if all_customers - served:
        feasible = False
        total_cost += len(all_customers - served) * 10000.0

    return total_cost, total_tardiness, feasible


def solve_no_drone(instance, pop_size=80, generations=120,
                   crossover_pb=0.85, mutation_pb=0.1, seed=42):
    """
    Solve VRPTW using GA (truck only, no drones).
    Returns list of TruckSolution (Pareto front).
    """
    random.seed(seed)
    n = instance['n_customers']

    # Initialize population using greedy + random
    population = []
    # One greedy solution
    greedy_routes = _greedy_init(instance, instance['customers'])
    population.append(greedy_routes)

    # Random solutions
    for _ in range(pop_size - 1):
        flat = list(range(1, n + 1))
        random.shuffle(flat)
        routes = _split_to_routes(flat, instance)
        population.append(routes)

    for gen in range(generations):
        # Evaluate all
        fitnesses = [_evaluate_truck_only(r, instance) for r in population]

        # Non-dominated sorting for selection
        # Simple: tournament selection based on cost
        new_pop = []
        while len(new_pop) < pop_size:
            # Binary tournament
            i1, i2 = random.sample(range(len(population)), 2)
            c1, t1, f1 = fitnesses[i1]
            c2, t2, f2 = fitnesses[i2]

            if f1 and not f2:
                winner = population[i1]
            elif f2 and not f1:
                winner = population[i2]
            elif c1 + t1 < c2 + t2:
                winner = population[i1]
            else:
                winner = population[i2]

            new_pop.append(copy.deepcopy(winner))

        # Crossover
        for i in range(0, pop_size - 1, 2):
            if random.random() < crossover_pb:
                r1 = new_pop[i]
                r2 = new_pop[i + 1]
                flat1, flat2 = _crossover(r1, r2)
                new_pop[i] = _split_to_routes(flat1, instance)
                new_pop[i + 1] = _split_to_routes(flat2, instance)

        # Mutation
        for i in range(pop_size):
            if random.random() < mutation_pb:
                flat = [c for r in new_pop[i] for c in r]
                flat = _mutate(flat)
                new_pop[i] = _split_to_routes(flat, instance)

        population = new_pop

    # Build final solutions
    solutions = []
    for routes in population:
        sol = _make_solution(routes, instance)
        solutions.append(sol)

    return solutions


def run_no_drone(instance, n_runs=10, seed=42):
    """Run No-Drone baseline multiple times, return all solutions + Pareto front."""
    all_solutions = []
    times = []

    for run in range(n_runs):
        s = seed + run
        t0 = time.time()
        solutions = solve_no_drone(
            instance,
            pop_size=80 if instance['n_customers'] <= 50 else 100,
            generations=120,
            seed=s,
        )
        elapsed = time.time() - t0
        times.append(elapsed)
        all_solutions.extend(solutions)

    pareto = extract_pareto_front(all_solutions)
    return {
        'solutions': all_solutions,
        'pareto_front': pareto,
        'mean_runtime': sum(times) / len(times),
        'std_runtime': (sum((t - sum(times)/len(times))**2 for t in times) / max(len(times)-1, 1))**0.5 if len(times) > 1 else 0,
    }
