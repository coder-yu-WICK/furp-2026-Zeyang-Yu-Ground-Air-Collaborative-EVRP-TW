# -*- coding: utf-8 -*-
"""
EVRP-TW Problem Model — Week 8 (truck-only, no drones).

Evaluates truck-only solutions for cost, tardiness, feasibility.
Renamed from TruckSolution to TruckSolution to reflect the
teacher-guided decision to remove truck-drone collaboration.
"""

import math
import copy
from itertools import combinations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from week8.config import (
    TRUCK_SPEED,
    TRUCK_CAPACITY,
    TRUCK_FIXED_COST,
    TRUCK_DIST_COST_RATE,
    TARDINESS_COST_RATE,
    DEPOT,
)


class TruckSolution:
    """
    Represents a truck-only solution to the EVRP-TW.

    A solution consists of truck routes only:
      - truck_routes: list of lists, each inner list = sequence of customer indices
        (index 0 = depot, customer indices start at 1)

    No drone missions — removed per teacher guidance to differentiate
    from classmate's truck+drone work and focus on EDD repair as unique contribution.
    """

    def __init__(self, truck_routes, instance):
        self.truck_routes = truck_routes
        self.instance = instance
        self._cost = None
        self._tardiness = None
        self._feasible = None
        self._violations = None

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
    def objectives(self):
        return (self.cost, self.tardiness)

    def dominates(self, other):
        """Check if this solution Pareto-dominates other (minimization)."""
        c1, t1 = self.objectives
        c2, t2 = other.objectives
        return (c1 <= c2 and t1 <= t2) and (c1 < c2 or t1 < t2)

    def _depot_dist(self, cust_idx, customers, dist):
        if cust_idx == 0:
            return 0.0
        return math.sqrt(
            (DEPOT[0] - customers[cust_idx - 1]['x']) ** 2 +
            (DEPOT[1] - customers[cust_idx - 1]['y']) ** 2
        )

    def _evaluate(self):
        """
        Evaluate truck-only solution.
        Computes: cost, tardiness, feasible, violations.
        Only checks truck constraints: capacity, time windows.
        """
        inst = self.instance
        customers = inst['customers']
        dist = inst['distance_matrix']

        total_cost = 0.0
        total_tardiness = 0.0
        feasible = True
        violations = {
            'capacity': 0,
            'time_window': 0,
        }

        served_customers = set()
        n_trucks = len(self.truck_routes)
        total_truck_dist = 0.0

        for route in self.truck_routes:
            if not route:
                continue

            prev = 0
            load = 0.0
            current_time = 0.0
            route_dist = 0.0

            for cust_idx in route:
                c_data = customers[cust_idx - 1]
                served_customers.add(cust_idx)

                seg_dist = dist[prev][cust_idx]
                route_dist += seg_dist
                current_time += seg_dist / TRUCK_SPEED

                # Time window check
                ready = c_data['ready_time']
                due = c_data['due_time']
                if current_time < ready:
                    current_time = ready
                if current_time > due:
                    tardy = current_time - due
                    total_tardiness += tardy * TARDINESS_COST_RATE
                    violations['time_window'] += tardy

                # Capacity check (use instance-specific capacity if available)
                truck_cap = inst.get('truck_capacity', TRUCK_CAPACITY)
                load += c_data['demand']
                if load > truck_cap:
                    violations['capacity'] += (load - truck_cap)
                    feasible = False

                current_time += c_data['service_time']
                prev = cust_idx

            # Return to depot
            route_dist += dist[prev][0]
            current_time += dist[prev][0] / TRUCK_SPEED
            total_truck_dist += route_dist

        # Cost: fixed + distance (truck-only, no drone components)
        vehicle_fixed = n_trucks * TRUCK_FIXED_COST
        distance_cost = total_truck_dist * TRUCK_DIST_COST_RATE
        total_cost = vehicle_fixed + distance_cost

        # Unserved customers penalty
        all_customers = set(range(1, inst['n_customers'] + 1))
        unserved = all_customers - served_customers
        if unserved:
            feasible = False
            total_cost += len(unserved) * 1000.0

        self._cost = total_cost
        self._tardiness = total_tardiness
        self._feasible = feasible
        self._violations = violations


# ═══════════════════════════════════════════════════════════════════════════
# Pareto Utilities
# ═══════════════════════════════════════════════════════════════════════════

def extract_pareto_front(solutions):
    """Extract Pareto-optimal (non-dominated) solutions."""
    if not solutions:
        return []
    pareto = []
    for s in solutions:
        dominated = any(other is not s and other.dominates(s) for other in solutions)
        if not dominated:
            pareto.append(s)
    return pareto


def compute_hypervolume(pareto_front, ref_point):
    """Compute hypervolume of a Pareto front relative to reference point."""
    if not pareto_front:
        return 0.0
    points = sorted([(s.cost, s.tardiness) for s in pareto_front], key=lambda p: p[0])
    hv = 0.0
    prev_x = points[0][0]
    for i, (x, y) in enumerate(points):
        width = x - prev_x
        height = max(0, ref_point[1] - y)
        hv += width * height
        prev_x = x
    hv += max(0, ref_point[0] - prev_x) * ref_point[1]
    return hv


def evaluate_solution_batch(solutions):
    """Batch-evaluate a list of solutions, return summary statistics."""
    if not solutions:
        return {}
    costs = [s.cost for s in solutions]
    tards = [s.tardiness for s in solutions]
    feas = [s.feasible for s in solutions]
    return {
        'mean_cost': sum(costs) / len(costs),
        'mean_tardiness': sum(tards) / len(tards),
        'feasibility_rate': sum(1 for f in feas if f) / len(feas),
        'n_solutions': len(solutions),
        'min_cost': min(costs),
        'max_cost': max(costs),
    }
