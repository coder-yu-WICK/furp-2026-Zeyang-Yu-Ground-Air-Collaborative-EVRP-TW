# -*- coding: utf-8 -*-
"""
Core problem model for Truck-Drone EVRP-TW.
Evaluates solutions for cost, tardiness, feasibility, and drone utilization.
"""

import math
import copy
from itertools import combinations

from config import (
    TRUCK_SPEED, DRONE_SPEED,
    TRUCK_CAPACITY, DRONE_CAPACITY,
    TRUCK_FIXED_COST, DRONE_FIXED_COST,
    TRUCK_DIST_COST_RATE, DRONE_DIST_COST_RATE,
    TARDINESS_COST_RATE,
    DEPOT,
)


class TruckDroneSolution:
    """
    Represents a solution to the truck-drone collaborative routing problem.

    A solution consists of truck routes and drone missions:
      - truck_routes: list of lists, each inner list = sequence of customer indices (0 = depot)
      - drone_missions: list of (launch_node, customer, recovery_node) tuples
    """

    def __init__(self, truck_routes, drone_missions, instance):
        self.truck_routes = truck_routes      # list of truck routes (each is list of customer IDs)
        self.drone_missions = drone_missions  # list of (i, j, k) where drone serves j, truck goes i→k
        self.instance = instance
        self._cost = None
        self._tardiness = None
        self._feasible = None
        self._violations = None
        self._drone_util = None

    def copy(self):
        return copy.deepcopy(self)

    @property
    def cost(self):
        if self._cost is None:
            self._evaluate()
        return self._cost

    @property
    def tardiness(self):
        if self._tardiness is None:
            self._evaluate()
        return self._tardiness

    @property
    def feasible(self):
        if self._feasible is None:
            self._evaluate()
        return self._feasible

    @property
    def violations(self):
        if self._violations is None:
            self._evaluate()
        return self._violations

    @property
    def drone_utilization(self):
        if self._drone_util is None:
            self._evaluate()
        return self._drone_util

    @property
    def objectives(self):
        return (self.cost, self.tardiness)

    def dominates(self, other):
        """Check if this solution Pareto-dominates other (minimization)."""
        c1, t1 = self.objectives
        c2, t2 = other.objectives
        return (c1 <= c2 and t1 <= t2) and (c1 < c2 or t1 < t2)

    def _evaluate(self):
        """Full solution evaluation: cost, tardiness, feasibility, drone utilization."""
        inst = self.instance
        customers = inst['customers']
        dist = inst['distance_matrix']

        total_cost = 0.0
        total_tardiness = 0.0
        feasible = True
        violations = {
            'capacity': 0,
            'time_window': 0,
            'drone_endurance': 0,
            'drone_capacity': 0,
            'sync': 0,
        }

        # Track which customers are served by drones
        drone_served_customers = set()
        for mission in self.drone_missions:
            _, j, _ = mission
            drone_served_customers.add(j)

        # Track all served customers (by truck or drone)
        served_customers = set(drone_served_customers)

        # Track per-node arrival times for sync checking
        node_arrival_times = {}  # node_id -> arrival_time

        # ── Evaluate truck routes ──────────────────────────────
        n_trucks = len(self.truck_routes)
        total_truck_dist = 0.0

        for route in self.truck_routes:
            if not route:
                continue

            prev = 0  # depot index in distance_matrix
            load = 0.0
            current_time = 0.0
            route_dist = 0.0

            for idx, cust_idx in enumerate(route):
                # Customer index in distance matrix = cust_idx (1-based, same as customer ID)
                c_data = customers[cust_idx - 1]
                served_customers.add(cust_idx)

                seg_dist = dist[prev][cust_idx]
                route_dist += seg_dist
                current_time += seg_dist / TRUCK_SPEED

                # Time window check
                ready = c_data['ready_time']
                due = c_data['due_time']
                if current_time < ready:
                    current_time = ready  # wait
                if current_time > due:
                    tardy = current_time - due
                    total_tardiness += tardy * TARDINESS_COST_RATE
                    violations['time_window'] += tardy

                # Record arrival time for sync checking
                node_arrival_times[cust_idx] = current_time

                # Capacity check
                load += c_data['demand']
                if load > TRUCK_CAPACITY:
                    violations['capacity'] += (load - TRUCK_CAPACITY)
                    feasible = False

                # Service time
                current_time += c_data['service_time']
                prev = cust_idx

            # Return to depot
            route_dist += dist[prev][0]
            current_time += dist[prev][0] / TRUCK_SPEED
            total_truck_dist += route_dist

        # ── Evaluate drone missions ────────────────────────────
        total_drone_dist = 0.0
        n_drones_used = 0
        n_drone_customers = 0

        for mission in self.drone_missions:
            i, j, k = mission
            # i, j, k are customer indices (1..n) or 0 for depot
            d_ij = dist[i][j] if i > 0 else self._depot_dist(j, customers, dist)
            d_jk = dist[j][k] if k > 0 else self._depot_dist(j, customers, dist)
            drone_leg = d_ij + d_jk

            total_drone_dist += drone_leg
            n_drones_used += 1
            n_drone_customers += 1

            # Endurance check
            if drone_leg > 6.0:  # high endurance ceiling
                violations['drone_endurance'] += (drone_leg - 6.0)
                feasible = False

            # Drone capacity check
            c_data = customers[j - 1]
            if c_data['demand'] > DRONE_CAPACITY:
                violations['drone_capacity'] += (c_data['demand'] - DRONE_CAPACITY)
                feasible = False

            # ── Sync check (NEW) ──
            # Drone flight time vs truck travel time between launch (i) and recovery (k)
            truck_arrival_i = node_arrival_times.get(i, 0.0) if i > 0 else 0.0
            truck_arrival_k = node_arrival_times.get(k, float('inf')) if k > 0 else float('inf')

            # Truck travel time from i to k (direct, no intermediate if consecutive)
            service_i = customers[i-1]['service_time'] if i > 0 else 0
            truck_depart_i = truck_arrival_i + service_i

            # Truck travel from i to k
            d_ik = dist[i][k] if (i > 0 and k > 0) else (
                self._depot_dist(k, customers, dist) if i == 0 and k > 0 else
                self._depot_dist(i, customers, dist) if i > 0 and k == 0 else 0.0)
            truck_travel_ik = d_ik / TRUCK_SPEED
            truck_arrival_k_expected = truck_depart_i + truck_travel_ik

            # Drone flight time
            drone_flight_time = drone_leg / DRONE_SPEED + c_data['service_time']

            # Sync: drone must not arrive before truck
            # If drone arrives first → sync violation (drone has nowhere to land)
            sync_violation = max(0.0, truck_arrival_k_expected - (
                truck_depart_i + drone_flight_time))
            if sync_violation > 0:
                violations['sync'] += sync_violation
                feasible = False

        # ── Calculate total cost ────────────────────────────────
        vehicle_fixed = n_trucks * TRUCK_FIXED_COST + n_drones_used * DRONE_FIXED_COST
        distance_cost = total_truck_dist * TRUCK_DIST_COST_RATE + total_drone_dist * DRONE_DIST_COST_RATE
        total_cost = vehicle_fixed + distance_cost

        # ── Check for unserved customers ───────────────────────
        all_customers = set(range(1, inst['n_customers'] + 1))
        unserved = all_customers - served_customers
        if unserved:
            feasible = False
            total_cost += len(unserved) * 1000.0  # large penalty

        # ── Drone utilization ──────────────────────────────────
        drone_util = {
            'n_drones_used': n_drones_used,
            'n_drone_customers': n_drone_customers,
            'total_drone_distance': total_drone_dist,
            'drone_served_set': drone_served_customers,
        }

        self._cost = total_cost
        self._tardiness = total_tardiness
        self._feasible = feasible
        self._violations = violations
        self._drone_util = drone_util

    def _depot_dist(self, customer_idx, customers, dist):
        """Distance from depot to customer (customer_idx is 1-based)."""
        return math.sqrt(
            (DEPOT[0] - customers[customer_idx - 1]['x'])**2 +
            (DEPOT[1] - customers[customer_idx - 1]['y'])**2
        )


def extract_pareto_front(solutions):
    """Extract non-dominated solutions from a list."""
    front = []
    for s in solutions:
        dominated = False
        for other in solutions:
            if other is s:
                continue
            if other.dominates(s):
                dominated = True
                break
        if not dominated:
            front.append(s)
    return front


def compute_hypervolume(pareto_front, ref_point):
    """
    Compute 2D Hypervolume for minimization (cost, tardiness).

    Reference point is auto-expanded to 1.2x max observed to ensure all
    Pareto points are covered (nadir point).

    Algorithm: sort by cost ascending, sweep tardiness descending.
      HV = Σ (ref_c - c_i) × (prev_t - t_i)  where prev_t is the
      tardiness of the previously-processed (better-cost) point.
    """
    if not pareto_front:
        return 0.0

    points = [(s.cost, s.tardiness) for s in pareto_front]
    return _hv_from_pairs(points, ref_point)


def _hv_from_pairs(pairs, ref_point):
    """Core HV computation from (cost, tardiness) pairs."""
    if not pairs:
        return 0.0

    # Auto-scale: reference must dominate all observed points
    max_c = max(c for c, _ in pairs)
    max_t = max(t for _, t in pairs)
    rc = max(ref_point[0], max_c * 1.2, 1.0)
    rt = max(ref_point[1], max_t * 1.2, 1.0)

    # Keep only points that are not worse than reference in both objectives
    pts = [(c, t) for c, t in pairs if c <= rc and t <= rt]
    if not pts:
        return 0.0

    # Build non-dominated subset within these points
    pts.sort(key=lambda p: p[0])
    nondom = []
    best_t = float('inf')
    for c, t in pts:
        if t < best_t:
            nondom.append((c, t))
            best_t = t

    # Sweep: for each point, area = width × height
    # width = (ref_c - cost_i), height = (prev_tardiness - tardiness_i)
    hv = 0.0
    prev_t = rt
    for c, t in nondom:
        if t < prev_t:
            hv += (rc - c) * (prev_t - t)
            prev_t = t

    return hv


def hypervolume_2d(points, ref_point):
    """
    Compute 2D Hypervolume. Points is list of (cost, tardiness).
    Both objectives are minimized. Ref point is (max_cost, max_tardiness).
    """
    if not points:
        return 0.0

    # Keep only points dominated by reference point
    pts = [(c, t) for c, t in points if c <= ref_point[0] and t <= ref_point[1]]
    if not pts:
        return 0.0

    # Sort by cost ascending
    pts.sort(key=lambda p: p[0])

    hv = 0.0
    prev_t = 0.0

    for c, t in pts:
        if t > prev_t:
            hv += (ref_point[0] - c) * (t - prev_t)
            prev_t = t

    return hv


def evaluate_solution_batch(solutions):
    """Evaluate a batch of solutions and return aggregate metrics."""
    costs = [s.cost for s in solutions]
    tards = [s.tardiness for s in solutions]
    feasible_count = sum(1 for s in solutions if s.feasible)

    pareto = extract_pareto_front(solutions)
    hv = compute_hypervolume(pareto, (5000.0, 20000.0))
    # Save serializable Pareto points for offline HV recalculation
    pareto_points = [(s.cost, s.tardiness) for s in pareto]

    drone_solutions = sum(1 for s in solutions
                          if s.drone_utilization['n_drone_customers'] > 0)
    avg_drone_missions = sum(s.drone_utilization['n_drone_customers']
                             for s in solutions) / max(len(solutions), 1)

    return {
        'mean_cost': sum(costs) / len(costs) if costs else 0,
        'std_cost': _std(costs),
        'mean_tardiness': sum(tards) / len(tards) if tards else 0,
        'std_tardiness': _std(tards),
        'feasibility_rate': feasible_count / len(solutions) if solutions else 0,
        'hypervolume': hv,
        'pareto_points': pareto_points,  # [(cost, tard), ...]  serializable
        'drone_solution_pct': drone_solutions / len(solutions) * 100 if solutions else 0,
        'avg_drone_missions': avg_drone_missions,
        'pareto_size': len(pareto),
        'best_cost': min(costs) if costs else float('inf'),
        'best_tardiness': min(tards) if tards else float('inf'),
        'pareto_front': pareto,
    }


def _std(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean)**2 for v in values) / (len(values) - 1))**0.5
