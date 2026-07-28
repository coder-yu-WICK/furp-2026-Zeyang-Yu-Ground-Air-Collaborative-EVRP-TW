#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EV-Aware Problem Model for Truck-Drone EVRP-TW.

Extends TruckDroneSolution with battery state tracking, charging station
visits, and energy consumption modeling.

Literature basis for parameters:
  - Battery capacity: 100 kWh (typical Class 3-4 electric delivery truck)
    Source: Keskin & Çatay (2016), Schneider et al. (2014)
  - Energy consumption: 1.5 kWh/km (urban stop-and-go cycle)
    Source: Pelletier et al. (2017), Davis & Figliozzi (2013)
  - Charging rate: 1.0 kWh/min linear (≈60 kW DC fast charging)
    Source: Keskin & Çatay (2018), Froger et al. (2019)
  - Non-linear charging: piecewise (fast 0-20%, normal 20-80%, slow 80-100%)
    Source: Montoya et al. (2017), Pelletier et al. (2019)

Charging station nodes:
  - CS0 (node n+1): Depot area at (8.0, 8.0)
  - CS1 (node n+2): North-west at (4.0, 12.0)
  - CS2 (node n+3): South-east at (12.0, 4.0)
"""

import math, copy
from config import (
    TRUCK_SPEED, DRONE_SPEED,
    TRUCK_CAPACITY, DRONE_CAPACITY,
    TRUCK_FIXED_COST, DRONE_FIXED_COST,
    TRUCK_DIST_COST_RATE, DRONE_DIST_COST_RATE,
    TARDINESS_COST_RATE,
    BATTERY_CAPACITY, CHARGING_RATE, CHARGING_STATIONS,
    CHARGING_SEGMENTS,
    DEPOT, MAX_DRONES_PER_TRUCK, COORD_SCALE,
)
from utils.problem_model import TruckDroneSolution

# ── EV Literature Parameters ──────────────────────────────────────────
ENERGY_CONSUMPTION_RATE = 1.5   # kWh per km (urban delivery, stop-and-go)
BATTERY_SAFETY_MARGIN = 5.0     # kWh — minimum battery before must-charge
CHARGING_STATION_SERVICE_TIME = 0.0  # charging time is variable, not fixed


def get_charging_station_coords(n_customers):
    """Return dict of charging station node ID -> (x, y) in km."""
    cs_nodes = {}
    for idx, (cx, cy) in enumerate(CHARGING_STATIONS):
        cs_id = n_customers + 1 + idx
        cs_nodes[cs_id] = (cx, cy)
    return cs_nodes


def station_distance(from_node, to_node, customers, cs_coords, depot):
    """Compute distance between two nodes, where either could be a CS or depot."""
    def _coord(node_id):
        if node_id == 0:
            return depot
        if node_id > len(customers):
            return cs_coords.get(node_id, depot)
        c = customers[node_id - 1]
        return (c['x'], c['y'])

    p1 = _coord(from_node)
    p2 = _coord(to_node)
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


# ═══════════════════════════════════════════════════════════════════════════
# Charging Functions
# ═══════════════════════════════════════════════════════════════════════════

def charge_linear(charge_time, rate=None):
    """Linear charging: energy_added = rate * time."""
    if rate is None:
        rate = CHARGING_RATE
    return rate * charge_time


def charge_nonlinear(current_soc_pct, charge_time, base_rate=None):
    """
    Non-linear (piecewise) charging based on SOC at arrival.

    Uses CHARGING_SEGMENTS from config:
      (soc_low, soc_high, rate_multiplier)

    Args:
        current_soc_pct: battery SOC percentage (0-100) at arrival
        charge_time: time spent charging
        base_rate: base charging rate (default: CHARGING_RATE)

    Returns:
        energy_added (kWh)
    """
    if base_rate is None:
        base_rate = CHARGING_RATE

    remaining_time = charge_time
    total_energy = 0.0
    current_pct = current_soc_pct
    max_pct = 100.0

    for seg_low, seg_high, multiplier in CHARGING_SEGMENTS:
        if remaining_time <= 0 or current_pct >= max_pct:
            break
        seg_low_pct = seg_low * 100.0
        seg_high_pct = seg_high * 100.0

        if current_pct >= seg_high_pct:
            continue  # Already past this segment

        # How much SOC can we gain in this segment?
        start_pct = max(current_pct, seg_low_pct)
        end_pct = seg_high_pct

        # Time needed to fill this segment at this rate
        pct_range = end_pct - start_pct
        kwh_range = (pct_range / 100.0) * BATTERY_CAPACITY
        effective_rate = base_rate * multiplier
        time_needed = kwh_range / effective_rate if effective_rate > 0 else float('inf')

        actual_time = min(remaining_time, time_needed)
        energy = effective_rate * actual_time
        total_energy += energy
        remaining_time -= actual_time
        current_pct = start_pct + (energy / BATTERY_CAPACITY) * 100.0

    return total_energy


def compute_nonlinear_charge_time(soc_start_pct, energy_to_add_kwh,
                                   battery_capacity=None, base_rate=None,
                                   segments=None):
    """
    Compute EXACT time needed to add energy_to_add_kWh with non-linear charging.

    Unlike charge_nonlinear() which computes energy-gained-from-time, this
    inverts the piecewise curve: given target energy, compute required time.

    Walks SOC segments from start_pct upward, accumulating time until the
    target energy is reached. Handles partial segments correctly.

    Args:
        soc_start_pct: battery SOC percentage (0-100) at arrival
        energy_to_add_kwh: target energy to add
        battery_capacity: max battery kWh (default: BATTERY_CAPACITY)
        base_rate: base charging rate (default: CHARGING_RATE)
        segments: charging segments list (default: CHARGING_SEGMENTS)

    Returns:
        charge_time (minutes) needed to add the target energy
    """
    if battery_capacity is None:
        battery_capacity = BATTERY_CAPACITY
    if base_rate is None:
        base_rate = CHARGING_RATE
    if segments is None:
        segments = CHARGING_SEGMENTS

    remaining_energy = energy_to_add_kwh
    total_time = 0.0
    current_pct = soc_start_pct
    max_pct = 100.0

    for seg_low, seg_high, multiplier in segments:
        if remaining_energy <= 0.001:
            break
        if current_pct >= max_pct:
            break

        seg_low_pct = seg_low * 100.0
        seg_high_pct = seg_high * 100.0

        if current_pct >= seg_high_pct:
            continue

        start_pct = max(current_pct, seg_low_pct)
        end_pct = seg_high_pct
        pct_range = end_pct - start_pct
        kwh_range = (pct_range / 100.0) * battery_capacity
        effective_rate = base_rate * multiplier

        if effective_rate <= 0:
            continue

        # Energy we will add in this segment
        energy_in_segment = min(remaining_energy, kwh_range)
        time_in_segment = energy_in_segment / effective_rate

        total_time += time_in_segment
        remaining_energy -= energy_in_segment
        current_pct = end_pct

    return total_time


# ═══════════════════════════════════════════════════════════════════════════
# EV Route Simulation
# ═══════════════════════════════════════════════════════════════════════════

def simulate_route_ev(route, customers, dist_matrix, cs_coords, depot,
                       battery_capacity=None, charging_model='linear',
                       energy_rate=None):
    """
    Simulate a truck route with EV battery tracking.

    Charging station nodes have IDs > n_customers. They appear in the route
    like regular stops: the truck diverts to them, charges, and continues.

    Args:
        route: list of node IDs (customers + optional charging stations)
        customers: customer data list
        dist_matrix: pre-computed distance matrix (depot + customers only)
        cs_coords: dict of CS node ID -> (x, y)
        depot: (x, y) tuple
        battery_capacity: max battery kWh (default: BATTERY_CAPACITY)
        charging_model: 'linear' or 'nonlinear'
        energy_rate: kWh per km (default: ENERGY_CONSUMPTION_RATE)

    Returns:
        dict with:
          - total_dist, total_time, total_tardiness, total_energy
          - arrivals, departures, battery_levels
          - n_charges, total_charge_energy, total_charge_time
          - energy_violation (>0 if battery went below 0)
          - feasible (True if all constraints met)
    """
    if battery_capacity is None:
        battery_capacity = BATTERY_CAPACITY
    if energy_rate is None:
        energy_rate = ENERGY_CONSUMPTION_RATE

    n_cust = len(customers)
    total_dist = 0.0
    current_time = 0.0
    battery = battery_capacity
    total_tardiness = 0.0
    total_energy = 0.0
    energy_violation = 0.0
    n_charges = 0
    total_charge_energy = 0.0
    total_charge_time = 0.0
    load = 0.0
    capacity_violation = 0.0

    arrivals = []
    departures = []
    battery_levels = [battery]  # Starting SOC

    prev = 0  # depot

    for node_id in route:
        # Distance from previous node to this one
        if prev > n_cust or node_id > n_cust:
            travel_dist = station_distance(prev, node_id, customers, cs_coords, depot)
        else:
            travel_dist = dist_matrix[prev][node_id]

        total_dist += travel_dist

        # Energy consumption
        energy_used = travel_dist * energy_rate
        battery -= energy_used
        total_energy += energy_used

        if battery < 0:
            energy_violation += abs(battery)
            battery = 0.0  # clamp — this is a violation

        # Travel time
        current_time += travel_dist / TRUCK_SPEED

        if node_id > n_cust:
            # ── Charging station visit ──
            # We must decide how long to charge.
            # Strategy: charge enough to complete the route or to full.
            # For simulation, if a charge_amount was implied by the route
            # planner, we use it. Here we charge to reach a reasonable level.
            # Default: charge until 80% SOC or enough to finish route.

            # Compute remaining energy needed for rest of route
            energy_needed = 0.0
            remaining_prev = node_id
            for future_id in route[route.index(node_id)+1:]:
                if remaining_prev > n_cust or future_id > n_cust:
                    energy_needed += station_distance(remaining_prev, future_id,
                                                       customers, cs_coords, depot) * energy_rate
                else:
                    energy_needed += dist_matrix[remaining_prev][future_id] * energy_rate
                remaining_prev = future_id
            # Add return to depot
            if remaining_prev > n_cust:
                energy_needed += station_distance(remaining_prev, 0, customers, cs_coords, depot) * energy_rate
            else:
                energy_needed += dist_matrix[remaining_prev][0] * energy_rate

            target_energy = min(battery_capacity, max(
                battery + energy_needed + BATTERY_SAFETY_MARGIN,
                battery_capacity * 0.8  # charge to at least 80% by default
            ))
            energy_to_add = target_energy - battery
            energy_to_add = max(0, energy_to_add)

            if charging_model == 'nonlinear':
                soc_pct = (battery / battery_capacity) * 100.0
                # Direct computation: invert piecewise curve to find exact time
                charge_time = compute_nonlinear_charge_time(
                    soc_pct, energy_to_add, battery_capacity, CHARGING_RATE)
                energy_gained = energy_to_add  # exact
            else:
                charge_time = energy_to_add / CHARGING_RATE if CHARGING_RATE > 0 else 0
                energy_gained = charge_linear(charge_time)

            battery += energy_gained
            battery = min(battery, battery_capacity)
            current_time += charge_time
            n_charges += 1
            total_charge_energy += energy_gained
            total_charge_time += charge_time

            arrivals.append(current_time)
            departures.append(current_time)
            battery_levels.append(battery)

        else:
            # ── Customer visit ──
            c = customers[node_id - 1]

            # TW check (Solomon standard: at START of service)
            start_time = max(current_time, c['ready_time'])
            if start_time > c['due_time']:
                total_tardiness += (start_time - c['due_time'])

            arrivals.append(start_time)
            current_time = start_time + c['service_time']
            departures.append(current_time)

            # Capacity
            load += c['demand']
            if load > TRUCK_CAPACITY:
                capacity_violation += (load - TRUCK_CAPACITY)

            battery_levels.append(battery)

        prev = node_id

    # Return to depot
    if prev > n_cust:
        return_dist = station_distance(prev, 0, customers, cs_coords, depot)
    else:
        return_dist = dist_matrix[prev][0]

    total_dist += return_dist
    energy_used = return_dist * energy_rate
    battery -= energy_used
    total_energy += energy_used
    if battery < 0:
        energy_violation += abs(battery)
    current_time += return_dist / TRUCK_SPEED

    feasible = (energy_violation <= 0.01 and capacity_violation <= 0.01)

    return {
        'total_dist': total_dist,
        'total_time': current_time,
        'total_tardiness': total_tardiness,
        'total_energy': total_energy,
        'final_battery': battery,
        'energy_violation': energy_violation,
        'capacity_violation': capacity_violation,
        'n_charges': n_charges,
        'total_charge_energy': total_charge_energy,
        'total_charge_time': total_charge_time,
        'arrivals': arrivals,
        'departures': departures,
        'battery_levels': battery_levels,
        'feasible': feasible,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Charging Stop Insertion
# ═══════════════════════════════════════════════════════════════════════════

def find_nearest_cs(node_id, customers, cs_coords, depot):
    """Find the nearest charging station to a node."""
    best_cs = None
    best_dist = float('inf')
    for cs_id, cs_xy in cs_coords.items():
        d = station_distance(node_id, cs_id, customers, cs_coords, depot)
        if d < best_dist:
            best_dist = d
            best_cs = cs_id
    return best_cs, best_dist


def insert_charging_stops(routes, customers, dist_matrix, instance,
                           battery_capacity=None, energy_rate=None):
    """
    Insert charging station visits into truck routes where needed.

    Greedy strategy: simulate each route with EV constraints. When battery
    would drop below safety margin before reaching next node, insert the
    nearest charging station BEFORE that segment.

    Args:
        routes: list of truck routes (each is list of customer IDs)
        customers: customer data
        dist_matrix: distance matrix
        instance: full instance dict (for depot info)
        battery_capacity: max battery
        energy_rate: kWh/km

    Returns:
        (new_routes, stats) where stats has n_insertions, energy_violations_before, etc.
    """
    if battery_capacity is None:
        battery_capacity = BATTERY_CAPACITY
    if energy_rate is None:
        energy_rate = ENERGY_CONSUMPTION_RATE

    n_cust = len(customers)
    cs_coords = get_charging_station_coords(n_cust)
    depot = instance.get('depot', DEPOT)
    if isinstance(depot, list):
        depot = tuple(depot)

    new_routes = []
    total_insertions = 0
    violations_before = 0.0

    for route in routes:
        if not route:
            new_routes.append(route)
            continue

        new_route = []
        prev = 0  # depot
        battery = battery_capacity

        for node_id in route:
            # Check if we can reach this node
            travel_dist = dist_matrix[prev][node_id]
            energy_needed = travel_dist * energy_rate

            # Also check: can we reach the nearest CS after this node?
            nearest_cs, cs_dist = find_nearest_cs(node_id, customers, cs_coords, depot)
            energy_to_cs = travel_dist + cs_dist * energy_rate

            if battery - energy_needed < 0:
                # Cannot reach next customer — insert CS before it
                violations_before += (energy_needed - battery)
                # Find best CS to insert
                best_cs = None
                best_detour = float('inf')
                for cs_id in cs_coords:
                    d_to_cs = station_distance(prev, cs_id, customers, cs_coords, depot)
                    d_from_cs = station_distance(cs_id, node_id, customers, cs_coords, depot)
                    detour = d_to_cs + d_from_cs - dist_matrix[prev][node_id]
                    if detour < best_detour:
                        best_detour = detour
                        best_cs = cs_id

                if best_cs is not None:
                    new_route.append(best_cs)
                    # Charge at CS (reset battery)
                    d_to_cs = station_distance(prev, best_cs, customers, cs_coords, depot)
                    battery -= d_to_cs * energy_rate
                    battery = battery_capacity  # Full recharge
                    prev = best_cs
                    total_insertions += 1

                    # Recompute travel to node from CS
                    d_from_cs = station_distance(best_cs, node_id, customers, cs_coords, depot)
                    battery -= d_from_cs * energy_rate
                else:
                    # No viable CS — just try to reach node
                    battery -= energy_needed
            else:
                battery -= energy_needed

            if battery < BATTERY_SAFETY_MARGIN:
                # Proactive: insert CS after this customer if battery is low
                # Check if we can reach next customer
                pass  # Will be handled at the next iteration

            new_route.append(node_id)
            prev = node_id

        new_routes.append(new_route)

    return new_routes, {
        'n_insertions': total_insertions,
        'energy_violations_before': violations_before,
    }


# ═══════════════════════════════════════════════════════════════════════════
# EV Solution Class
# ═══════════════════════════════════════════════════════════════════════════

class EVTruckDroneSolution(TruckDroneSolution):
    """
    Extends TruckDroneSolution with EV battery-aware evaluation.

    Supports:
      - Battery state tracking per route
      - Charging station visits (nodes with ID > n_customers)
      - Linear and non-linear charging models
      - Energy violation tracking

    Additional properties:
      - ev_feasible: True if no energy violations
      - energy_violation: total kWh deficit
      - total_energy: total kWh consumed
      - n_charges: number of charging station visits
      - charging_model: 'linear' or 'nonlinear' or 'none'
    """

    def __init__(self, truck_routes, drone_missions, instance,
                 max_drones_per_truck=None, charging_model='linear',
                 battery_capacity=None, energy_rate=None):
        super().__init__(truck_routes, drone_missions, instance, max_drones_per_truck)
        self.charging_model = charging_model
        self.battery_capacity = battery_capacity if battery_capacity is not None else BATTERY_CAPACITY
        self.energy_rate = energy_rate if energy_rate is not None else ENERGY_CONSUMPTION_RATE
        self._ev_evaluated = False
        self._ev_feasible = None
        self._energy_violation = 0.0
        self._total_energy = 0.0
        self._n_charges = 0
        self._total_charge_energy = 0.0
        self._total_charge_time = 0.0
        self._route_ev_details = []

    @property
    def ev_feasible(self):
        if not self._ev_evaluated:
            self._evaluate_ev()
        return self._ev_feasible

    @property
    def energy_violation(self):
        if not self._ev_evaluated:
            self._evaluate_ev()
        return self._energy_violation

    @property
    def total_energy(self):
        if not self._ev_evaluated:
            self._evaluate_ev()
        return self._total_energy

    @property
    def n_charges(self):
        if not self._ev_evaluated:
            self._evaluate_ev()
        return self._n_charges

    @property
    def total_charge_energy(self):
        if not self._ev_evaluated:
            self._evaluate_ev()
        return self._total_charge_energy

    @property
    def total_charge_time(self):
        if not self._ev_evaluated:
            self._evaluate_ev()
        return self._total_charge_time

    @property
    def ev_objectives(self):
        """Three-objective: (cost, tardiness, energy_violation)."""
        return (self.cost, self.tardiness, self.energy_violation)

    def _evaluate_ev(self):
        """Evaluate with EV battery constraints on truck routes."""
        inst = self.instance
        customers = inst['customers']
        dist_matrix = inst['distance_matrix']
        depot = inst.get('depot', DEPOT)
        if isinstance(depot, list):
            depot = tuple(depot)
        n_cust = len(customers)
        cs_coords = get_charging_station_coords(n_cust)

        total_energy = 0.0
        total_energy_violation = 0.0
        total_n_charges = 0
        total_charge_energy = 0.0
        total_charge_time = 0.0
        total_ev_feasible = True
        route_details = []

        for route in self.truck_routes:
            if not route:
                route_details.append(None)
                continue

            sim = simulate_route_ev(
                route, customers, dist_matrix, cs_coords, depot,
                battery_capacity=self.battery_capacity,
                charging_model=self.charging_model,
                energy_rate=self.energy_rate,
            )

            total_energy += sim['total_energy']
            total_energy_violation += sim['energy_violation']
            total_n_charges += sim['n_charges']
            total_charge_energy += sim['total_charge_energy']
            total_charge_time += sim['total_charge_time']
            if not sim['feasible']:
                total_ev_feasible = False
            route_details.append(sim)

        self._ev_feasible = total_ev_feasible and total_energy_violation <= 0.01
        self._energy_violation = total_energy_violation
        self._total_energy = total_energy
        self._n_charges = total_n_charges
        self._total_charge_energy = total_charge_energy
        self._total_charge_time = total_charge_time
        self._route_ev_details = route_details
        self._ev_evaluated = True

    def copy(self):
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)
        # Deep copy mutable fields
        new.truck_routes = [list(r) for r in self.truck_routes]
        new.drone_missions = [list(m) for m in self.drone_missions]
        new._ev_evaluated = False  # Force re-evaluation
        return new


# ═══════════════════════════════════════════════════════════════════════════
# EV Solution Evaluation (drop-in replacement for evaluate_solution_batch)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_ev_solution(sol, charging_model='linear'):
    """
    Evaluate a TruckDroneSolution or EVTruckDroneSolution with EV constraints.

    Returns dict with EV-specific metrics added.
    """
    if isinstance(sol, EVTruckDroneSolution):
        sol.charging_model = charging_model
        sol._ev_evaluated = False
        sol._evaluate_ev()

    base = {
        'cost': sol.cost,
        'tardiness': sol.tardiness,
        'feasible': sol.feasible,
        'drone_utilization': sol.drone_utilization,
    }

    if isinstance(sol, EVTruckDroneSolution):
        base.update({
            'ev_feasible': sol.ev_feasible,
            'energy_violation': sol.energy_violation,
            'total_energy': sol.total_energy,
            'n_charges': sol.n_charges,
            'total_charge_energy': sol.total_charge_energy,
            'total_charge_time': sol.total_charge_time,
            'charging_model': sol.charging_model,
            'battery_capacity': sol.battery_capacity,
        })
    else:
        base.update({
            'ev_feasible': None,
            'energy_violation': 0.0,
            'total_energy': 0.0,
            'n_charges': 0,
        })

    return base


# ═══════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("EV Problem Model — Self Test")
    print(f"  Battery capacity: {BATTERY_CAPACITY} kWh")
    print(f"  Energy consumption: {ENERGY_CONSUMPTION_RATE} kWh/km")
    print(f"  Charging rate (linear): {CHARGING_RATE} kWh/time_unit")
    print(f"  Charging stations: {CHARGING_STATIONS}")
    print(f"  Non-linear segments: {CHARGING_SEGMENTS}")

    # Test linear charging
    e = charge_linear(30)
    print(f"\nLinear charge 30 min: {e:.2f} kWh")

    # Test non-linear charging
    e_nl = charge_nonlinear(10.0, 30)  # 10% SOC, 30 min
    print(f"Non-linear charge (10% SOC, 30 min): {e_nl:.2f} kWh")

    e_nl2 = charge_nonlinear(85.0, 30)  # 85% SOC, 30 min
    print(f"Non-linear charge (85% SOC, 30 min): {e_nl2:.2f} kWh")

    print("\nDone.")
