# -*- coding: utf-8 -*-
"""
Electric Vehicle Problem Model — Week 7 Gap 2.

Extends TruckDroneSolution with battery state tracking, charging station
visits, and linear/non-linear charging models.

Supports FURP Models:
  - Model A: baseline (no charging constraints) — existing TruckDroneSolution
  - Model B: + linear charging
  - Model C: + non-linear charging

Charging station representation:
  Charging stations are special route nodes with IDs > n_customers.
  - Station 1: n_customers + 1  (depot area)
  - Station 2: n_customers + 2
  - Station 3: n_customers + 3
"""

import math
import os, sys

# Ensure config is importable
_W6 = os.path.dirname(os.path.abspath(__file__))
_W4 = os.path.join(_W6, '..', 'week4')
_W3 = os.path.join(_W6, '..', 'week3')

if _W4 not in sys.path:
    sys.path.insert(0, _W4)
if _W3 not in sys.path:
    sys.path.insert(0, _W3)

from config import (
    TRUCK_SPEED, DRONE_SPEED,
    TRUCK_CAPACITY, DRONE_CAPACITY,
    TRUCK_FIXED_COST, DRONE_FIXED_COST,
    TRUCK_DIST_COST_RATE, DRONE_DIST_COST_RATE,
    TARDINESS_COST_RATE, DEPOT,
    CHARGING_STATIONS, CHARGING_RATE,
    BATTERY_CAPACITY, CHARGING_SEGMENTS,
)
from utils.problem_model import TruckDroneSolution


# ── Charging Model Utilities ──────────────────────────────────────────

ENERGY_PER_KM = 1.0  # kWh per km

# Charging station node IDs are offset from customer count
CS_OFFSET_NAMES = {
    1: 'Depot CS',
    2: 'NW Station',
    3: 'SE Station',
}


def get_charging_station_nodes(n_customers):
    """Return list of charging station node IDs (> n_customers)."""
    return [n_customers + i + 1 for i in range(len(CHARGING_STATIONS))]


def get_charging_station_coords(station_node_id, n_customers):
    """Get (x, y) coordinates for a charging station node ID."""
    idx = station_node_id - n_customers - 1
    if 0 <= idx < len(CHARGING_STATIONS):
        return CHARGING_STATIONS[idx]
    return DEPOT  # fallback


def is_charging_station(node_id, n_customers):
    """Check if a node ID represents a charging station."""
    return node_id > n_customers


def _node_dist(i, j, dist_matrix, customers, depot):
    """Distance between two nodes. 0 = depot, >n_customers = charging station."""
    if i == 0 and j == 0:
        return 0.0
    if i == 0:
        if j > len(customers):
            cs_coords = get_charging_station_coords(j, len(customers))
            return math.sqrt((depot[0] - cs_coords[0])**2 + (depot[1] - cs_coords[1])**2)
        return math.sqrt((depot[0] - customers[j-1]['x'])**2 + (depot[1] - customers[j-1]['y'])**2)
    if j == 0:
        if i > len(customers):
            cs_coords = get_charging_station_coords(i, len(customers))
            return math.sqrt((depot[0] - cs_coords[0])**2 + (depot[1] - cs_coords[1])**2)
        return math.sqrt((depot[0] - customers[i-1]['x'])**2 + (depot[1] - customers[i-1]['y'])**2)
    # Both are non-depot
    if i > len(customers):
        ci = get_charging_station_coords(i, len(customers))
    else:
        ci = (customers[i-1]['x'], customers[i-1]['y'])
    if j > len(customers):
        cj = get_charging_station_coords(j, len(customers))
    else:
        cj = (customers[j-1]['x'], customers[j-1]['y'])
    return math.sqrt((ci[0] - cj[0])**2 + (ci[1] - cj[1])**2)


def compute_charging_time_linear(energy_needed, charging_rate=None):
    """Linear charging: time = energy / rate."""
    if charging_rate is None:
        charging_rate = CHARGING_RATE
    return energy_needed / max(charging_rate, 0.01)


def compute_charging_time_nonlinear(current_soc, target_soc, battery_capacity=None,
                                      segments=None):
    """
    Non-linear charging time using piecewise segments.

    Charging rate varies by SOC:
      - 0-20%: fast (1.5 × CHARGING_RATE)
      - 20-80%: normal (1.0 × CHARGING_RATE)
      - 80-100%: slow (0.5 × CHARGING_RATE)

    Args:
        current_soc: current state of charge [0, 1]
        target_soc: target state of charge [0, 1]

    Returns: charging time in time units
    """
    if battery_capacity is None:
        battery_capacity = BATTERY_CAPACITY
    if segments is None:
        segments = CHARGING_SEGMENTS

    if target_soc <= current_soc:
        return 0.0

    total_time = 0.0
    remaining = (target_soc - current_soc) * battery_capacity  # kWh needed

    # Sort segments by SOC range
    for soc_low, soc_high, rate_mult in sorted(segments, key=lambda s: s[0]):
        if remaining <= 0:
            break

        # How much of this segment overlaps with [current_soc, target_soc]
        seg_energy = (min(target_soc, soc_high) - max(current_soc, soc_low)) * battery_capacity
        seg_energy = max(0, min(seg_energy, remaining))

        if seg_energy > 0:
            rate = CHARGING_RATE * rate_mult
            total_time += seg_energy / rate
            remaining -= seg_energy

    return total_time


# ── Charging Stop Insertion ───────────────────────────────────────────

def find_nearest_charging_station(position, n_customers):
    """Find the nearest charging station to a given (x, y) position."""
    best_id = None
    best_dist = float('inf')
    for idx, cs_coords in enumerate(CHARGING_STATIONS):
        d = math.sqrt((position[0] - cs_coords[0])**2 + (position[1] - cs_coords[1])**2)
        if d < best_dist:
            best_dist = d
            best_id = n_customers + idx + 1
    return best_id, best_dist


def insert_charging_stops(truck_routes, instance, battery_capacity=None,
                           charging_model='linear', safety_margin=0.1):
    """
    Insert charging station visits into truck routes where needed.

    Strategy: Forward-simulate each route. When battery would drop below
    safety threshold before reaching the next node (or returning to depot),
    insert the nearest charging station before that segment.

    Args:
        truck_routes: list of lists of customer IDs
        instance: problem instance dict
        battery_capacity: kWh (default from config)
        charging_model: 'linear' or 'nonlinear'
        safety_margin: fraction of battery to keep as safety buffer

    Returns:
        (new_routes, charging_stats)
        charging_stats: {route_idx: [(station_id, charge_time, energy_added), ...]}
    """
    if battery_capacity is None:
        battery_capacity = BATTERY_CAPACITY

    customers = instance['customers']
    depot = instance['depot']
    dist = instance['distance_matrix']
    n_customers = instance['n_customers']

    new_routes = []
    all_charging_stats = []

    for route in truck_routes:
        if not route:
            new_routes.append([])
            all_charging_stats.append([])
            continue

        new_route = []
        route_stats = []
        current_battery = battery_capacity
        prev_node = 0  # depot

        for idx, cid in enumerate(route):
            # Distance from prev to this customer
            seg_dist = _node_dist(prev_node, cid, dist, customers, depot)
            energy_needed = seg_dist * ENERGY_PER_KM

            # Distance from this customer to depot (for return check)
            return_dist = _node_dist(cid, 0, dist, customers, depot)
            return_energy = return_dist * ENERGY_PER_KM

            # Check if we need to charge before going to this customer
            if current_battery - energy_needed < safety_margin * battery_capacity:
                # Need to charge — insert charging station before this customer
                cs_id, cs_dist = find_nearest_charging_station(
                    (customers[prev_node - 1]['x'], customers[prev_node - 1]['y'])
                    if prev_node > 0 else depot, n_customers)

                # Travel to charging station
                cs_travel_dist = _node_dist(prev_node, cs_id, dist, customers, depot)
                cs_travel_energy = cs_travel_dist * ENERGY_PER_KM

                if current_battery - cs_travel_energy < 0:
                    # Can't even reach charging station — mark as infeasible
                    # Try a different station or skip (record violation)
                    pass
                else:
                    current_battery -= cs_travel_energy

                    # Charge at station
                    energy_to_add = battery_capacity - current_battery
                    if charging_model == 'nonlinear':
                        current_soc = current_battery / battery_capacity
                        target_soc = 1.0
                        charge_time = compute_charging_time_nonlinear(current_soc, target_soc)
                    else:
                        charge_time = compute_charging_time_linear(energy_to_add)

                    current_battery = battery_capacity
                    new_route.append(cs_id)
                    route_stats.append((cs_id, charge_time, energy_to_add))

            # Now travel to the customer
            if current_battery - energy_needed < 0:
                # Battery infeasible even after charging — record but continue
                pass

            current_battery -= energy_needed
            new_route.append(cid)
            prev_node = cid

        new_routes.append(new_route)
        all_charging_stats.append(route_stats)

    return new_routes, all_charging_stats


# ── EV Solution Class ─────────────────────────────────────────────────

class EVTruckDroneSolution(TruckDroneSolution):
    """
    Extends TruckDroneSolution with electric vehicle constraints.

    Additional tracking:
      - Battery state along each route
      - Charging station visit times and energy
      - Battery and charging violations
    """

    def __init__(self, truck_routes, drone_missions, instance,
                 charging_model='linear', battery_capacity=None,
                 energy_per_km=None):
        super().__init__(truck_routes, drone_missions, instance)
        self.charging_model = charging_model
        self.battery_capacity = battery_capacity or BATTERY_CAPACITY
        self.energy_per_km = energy_per_km or ENERGY_PER_KM

        # EV-specific state (populated during _evaluate)
        self._battery_violations = 0.0
        self._charging_time = 0.0
        self._n_charges = 0
        self._energy_consumed = 0.0
        self._route_battery_traces = []  # per-route battery level traces

    @property
    def battery_violations(self):
        if self._cost is None:
            self._evaluate()
        return self._battery_violations

    @property
    def charging_time(self):
        if self._cost is None:
            self._evaluate()
        return self._charging_time

    @property
    def n_charges(self):
        if self._cost is None:
            self._evaluate()
        return self._n_charges

    @property
    def energy_consumed(self):
        if self._cost is None:
            self._evaluate()
        return self._energy_consumed

    def _evaluate(self):
        """Override: add battery tracking to the standard evaluation."""
        inst = self.instance
        customers = inst['customers']
        dist = inst['distance_matrix']
        depot = inst['depot']
        n_customers = inst['n_customers']

        total_cost = 0.0
        total_tardiness = 0.0
        feasible = True
        violations = {
            'capacity': 0,
            'time_window': 0,
            'drone_endurance': 0,
            'drone_capacity': 0,
            'sync': 0,
            'battery': 0,
        }

        # Track customers served by drones
        drone_served_customers = set()
        for mission in self.drone_missions:
            _, j, _ = mission
            drone_served_customers.add(j)

        served_customers = set(drone_served_customers)

        # ── Evaluate truck routes WITH battery tracking ────────────
        n_trucks = len(self.truck_routes)
        total_truck_dist = 0.0
        total_energy = 0.0
        total_charge_time = 0.0
        total_charges = 0
        total_battery_violations = 0.0
        self._route_battery_traces = []

        for route in self.truck_routes:
            if not route:
                self._route_battery_traces.append([])
                continue

            prev = 0  # depot
            load = 0.0
            current_time = 0.0
            current_battery = self.battery_capacity
            route_dist = 0.0
            battery_trace = [(0, self.battery_capacity)]  # (node, battery) trace

            for idx, node_id in enumerate(route):
                if is_charging_station(node_id, n_customers):
                    # ── Charging station visit ──
                    seg_dist = _node_dist(prev, node_id, dist, customers, depot)
                    route_dist += seg_dist
                    energy_used = seg_dist * self.energy_per_km
                    total_energy += energy_used
                    current_battery -= energy_used

                    if current_battery < 0:
                        violations['battery'] += abs(current_battery)
                        total_battery_violations += abs(current_battery)
                        feasible = False
                        current_battery = 0

                    current_time += seg_dist / TRUCK_SPEED

                    # Charge battery
                    starting_soc = current_battery / self.battery_capacity
                    energy_to_add = self.battery_capacity - current_battery

                    if self.charging_model == 'nonlinear':
                        charge_time = compute_charging_time_nonlinear(
                            starting_soc, 1.0, self.battery_capacity)
                    else:
                        charge_time = compute_charging_time_linear(
                            energy_to_add, CHARGING_RATE)

                    current_time += charge_time
                    current_battery = self.battery_capacity
                    total_charge_time += charge_time
                    total_charges += 1

                    battery_trace.append((node_id, current_battery))
                    prev = node_id
                    continue

                # ── Normal customer node ──
                c_data = customers[node_id - 1]
                served_customers.add(node_id)

                seg_dist = _node_dist(prev, node_id, dist, customers, depot)
                route_dist += seg_dist
                energy_used = seg_dist * self.energy_per_km
                total_energy += energy_used
                current_battery -= energy_used

                if current_battery < 0:
                    violations['battery'] += abs(current_battery)
                    total_battery_violations += abs(current_battery)
                    feasible = False
                    current_battery = 0

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

                # Capacity check
                load += c_data['demand']
                if load > TRUCK_CAPACITY:
                    violations['capacity'] += (load - TRUCK_CAPACITY)
                    feasible = False

                # Service time
                current_time += c_data['service_time']
                prev = node_id
                battery_trace.append((node_id, current_battery))

            # Return to depot
            return_dist = _node_dist(prev, 0, dist, customers, depot)
            route_dist += return_dist
            energy_used = return_dist * self.energy_per_km
            total_energy += energy_used
            current_battery -= energy_used

            if current_battery < 0:
                violations['battery'] += abs(current_battery)
                total_battery_violations += abs(current_battery)
                feasible = False

            current_time += return_dist / TRUCK_SPEED
            total_truck_dist += route_dist
            self._route_battery_traces.append(battery_trace)

        # ── Evaluate drone missions (unchanged from base class) ────
        total_drone_dist = 0.0
        n_drones_used = 0
        n_drone_customers = 0

        for mission in self.drone_missions:
            i, j, k = mission
            d_ij = _node_dist(i if i > 0 else 0, j, dist, customers, depot)
            d_jk = _node_dist(j, k if k > 0 else 0, dist, customers, depot)
            drone_leg = d_ij + d_jk

            total_drone_dist += drone_leg
            n_drones_used += 1
            n_drone_customers += 1

            # Endurance check
            if drone_leg > 6.0:
                violations['drone_endurance'] += (drone_leg - 6.0)
                feasible = False

            # Drone capacity check
            c_data = customers[j - 1]
            if c_data['demand'] > DRONE_CAPACITY:
                violations['drone_capacity'] += (c_data['demand'] - DRONE_CAPACITY)
                feasible = False

        # ── Calculate total cost ────────────────────────────────
        vehicle_fixed = n_trucks * TRUCK_FIXED_COST + n_drones_used * DRONE_FIXED_COST
        distance_cost = total_truck_dist * TRUCK_DIST_COST_RATE + total_drone_dist * DRONE_DIST_COST_RATE
        # Charging time penalty: 0.5 × charging_time (cost of waiting to charge)
        charging_cost = total_charge_time * 0.5
        total_cost = vehicle_fixed + distance_cost + total_tardiness + charging_cost

        # ── Unserved customers check ─────────────────────────────
        all_customers = set(range(1, n_customers + 1))
        unserved = all_customers - served_customers
        if unserved:
            feasible = False
            total_cost += len(unserved) * 1000.0

        # ── Store results ────────────────────────────────────────
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
        self._battery_violations = total_battery_violations
        self._charging_time = total_charge_time
        self._n_charges = total_charges
        self._energy_consumed = total_energy


# ── EV-aware experiment helpers ───────────────────────────────────────

def compare_charging_models(solution, instance):
    """
    Evaluate the same solution under different charging assumptions.

    Returns dict with Model A, B, C evaluations.
    """
    results = {}

    # Model A: No charging constraints (standard TruckDroneSolution)
    from utils.problem_model import TruckDroneSolution
    sol_a = TruckDroneSolution(solution.truck_routes, solution.drone_missions, instance)
    results['A_baseline'] = {
        'cost': sol_a.cost, 'tardiness': sol_a.tardiness,
        'feasible': sol_a.feasible, 'violations': sol_a.violations,
    }

    # Model B: Linear charging
    # Strip charging station nodes for fair comparison (add them back)
    n_customers = instance['n_customers']
    clean_routes = [[n for n in r if not is_charging_station(n, n_customers)]
                    for r in solution.truck_routes]
    routes_with_cs, cs_stats = insert_charging_stops(
        clean_routes, instance, charging_model='linear')
    sol_b = EVTruckDroneSolution(routes_with_cs, solution.drone_missions,
                                  instance, charging_model='linear')
    results['B_linear_charging'] = {
        'cost': sol_b.cost, 'tardiness': sol_b.tardiness,
        'feasible': sol_b.feasible, 'violations': sol_b.violations,
        'battery_violations': sol_b.battery_violations,
        'charging_time': sol_b.charging_time,
        'n_charges': sol_b.n_charges,
        'energy_consumed': sol_b.energy_consumed,
    }

    # Model C: Non-linear charging
    routes_with_cs_nl, cs_stats_nl = insert_charging_stops(
        clean_routes, instance, charging_model='nonlinear')
    sol_c = EVTruckDroneSolution(routes_with_cs_nl, solution.drone_missions,
                                  instance, charging_model='nonlinear')
    results['C_nonlinear_charging'] = {
        'cost': sol_c.cost, 'tardiness': sol_c.tardiness,
        'feasible': sol_c.feasible, 'violations': sol_c.violations,
        'battery_violations': sol_c.battery_violations,
        'charging_time': sol_c.charging_time,
        'n_charges': sol_c.n_charges,
        'energy_consumed': sol_c.energy_consumed,
    }

    return results


# ── Self-test ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== EV Problem Model Self-Test ===\n")

    # Build a test instance
    from utils.data_loader import build_instance

    inst = build_instance('RC201', 50)
    print(f"Instance: {inst['name']}, {inst['n_customers']} customers")
    print(f"Battery capacity: {BATTERY_CAPACITY} kWh")
    print(f"Charging stations: {CHARGING_STATIONS}")
    print(f"Energy per km: {ENERGY_PER_KM} kWh/km")

    # Test charging time models
    print("\n--- Charging Time Tests ---")

    # Linear: charge from 20% to 100%
    t_linear = compute_charging_time_linear(0.8 * BATTERY_CAPACITY)
    print(f"Linear charge (20→100%): {t_linear:.1f} time units")

    # Non-linear: charge from 20% to 100%
    t_nonlinear = compute_charging_time_nonlinear(0.2, 1.0)
    print(f"Non-linear charge (20→100%): {t_nonlinear:.1f} time units")

    # Compare partial charges
    t_lin_20 = compute_charging_time_linear(0.2 * BATTERY_CAPACITY)
    t_nl_20 = compute_charging_time_nonlinear(0.0, 0.2)
    print(f"Linear (0→20%): {t_lin_20:.1f}, Non-linear (0→20%): {t_nl_20:.1f}")

    # Test charging stop insertion
    print("\n--- Charging Stop Insertion ---")
    test_routes = [[c['id'] for c in inst['customers'][:25]]]  # First 25 on one route
    new_routes, stats = insert_charging_stops(test_routes, inst, battery_capacity=50.0)
    print(f"Original route length: {len(test_routes[0])}")
    print(f"Route with charging stops: {len(new_routes[0])}")
    print(f"Charging stops added: {len(stats[0])}")
    for cs_id, charge_time, energy in stats[0]:
        print(f"  CS {cs_id}: charge {charge_time:.1f} time, add {energy:.1f} kWh")

    # Test EV solution evaluation
    print("\n--- EV Solution Evaluation ---")
    ev_sol = EVTruckDroneSolution(new_routes, [], inst, charging_model='linear')
    print(f"Cost: {ev_sol.cost:.1f}")
    print(f"Tardiness: {ev_sol.tardiness:.1f}")
    print(f"Feasible: {ev_sol.feasible}")
    print(f"Battery violations: {ev_sol.battery_violations:.1f}")
    print(f"Charging time: {ev_sol.charging_time:.1f}")
    print(f"Charges: {ev_sol.n_charges}")
    print(f"Energy consumed: {ev_sol.energy_consumed:.1f} kWh")

    # Compare with non-linear
    ev_sol_nl = EVTruckDroneSolution(new_routes, [], inst, charging_model='nonlinear')
    print(f"\nNon-linear — Cost: {ev_sol_nl.cost:.1f}, Charge time: {ev_sol_nl.charging_time:.1f}")

    print("\n=== Self-Test Complete ===")
