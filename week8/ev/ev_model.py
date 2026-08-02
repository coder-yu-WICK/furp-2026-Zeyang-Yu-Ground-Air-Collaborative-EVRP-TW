#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EV-Aware Problem Model for EVRP-TW (truck-only).

Extends TruckSolution with battery state tracking, charging station
visits, and energy consumption modeling.

Literature basis for parameters:
  - Battery capacity: 100 kWh (typical Class 3-4 electric delivery truck)
    Source: Keskin & Catay (2016), Schneider et al. (2014)
  - Energy consumption: 1.5 kWh/km (urban stop-and-go cycle)
    Source: Pelletier et al. (2017), Davis & Figliozzi (2013)
  - Charging rate: 1.0 kWh/min linear (approx 60 kW DC fast charging)
    Source: Keskin & Catay (2018), Froger et al. (2019)
  - Non-linear charging: piecewise (fast 0-20%, normal 20-80%, slow 80-100%)
    Source: Montoya et al. (2017), Pelletier et al. (2019)

Charging station nodes:
  - CS0 (node n+1): Depot area at (8.0, 8.0)
  - CS1 (node n+2): North-west at (4.0, 12.0)
  - CS2 (node n+3): South-east at (12.0, 4.0)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math, copy
from week8.config import (
    TRUCK_SPEED,
    TRUCK_CAPACITY,
    TRUCK_FIXED_COST,
    TRUCK_DIST_COST_RATE,
    TARDINESS_COST_RATE,
    BATTERY_CAPACITY, CHARGING_RATE, CHARGING_STATIONS,
    CHARGING_SEGMENTS, ENERGY_CONSUMPTION_RATE,
    DEPOT, COORD_SCALE,
)
from week8.core.problem_model import TruckSolution

# ── EV Literature Parameters ──────────────────────────────────────────
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

    # EV feasibility: energy constraint only (capacity is a routing concern,
    # already enforced by TruckSolution._evaluate() and clustering)
    ev_feasible = (energy_violation <= 0.01)
    # Full feasibility includes capacity check
    feasible = (ev_feasible and capacity_violation <= 0.01)

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
        'ev_feasible': ev_feasible,   # Energy-only feasibility
        'feasible': feasible,          # Full feasibility (energy + capacity)
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


def _node_distance(prev, node_id, customers, dist_matrix, cs_coords, depot):
    """Distance between two nodes, handling CS node IDs (> n_customers)."""
    n_cust = len(customers)
    if prev > n_cust or node_id > n_cust:
        return station_distance(prev, node_id, customers, cs_coords, depot)
    return dist_matrix[prev][node_id]


def _route_energy_profile(route, customers, dist_matrix, cs_coords, depot,
                           energy_rate, battery_capacity):
    """
    Forward-simulate a route (may contain CS nodes) and return the energy profile.

    When a CS node is encountered, battery resets to battery_capacity (full charge).

    Returns:
        dict with:
          - energy_at_node: list of battery level after reaching each node
          - energy_needed_to_node: list of kWh needed from depot to each node
          - energy_needed_from_node: list of kWh needed from each node to end+depot
          - total_energy: total kWh for entire route
          - shortfall_nodes: list of (index, node_id, deficit_kwh) where battery < 0
    """
    n_cust = len(customers)
    n = len(route)
    if n == 0:
        return {'total_energy': 0.0, 'shortfall_nodes': [], 'energy_at_node': [],
                'energy_needed_to_node': [], 'energy_needed_from_node': []}

    energy_at_node = []
    energy_needed_to_node = []
    battery = battery_capacity
    prev = 0  # depot
    cumulative_energy = 0.0
    shortfall_nodes = []

    for i, node_id in enumerate(route):
        travel_dist = _node_distance(prev, node_id, customers, dist_matrix,
                                      cs_coords, depot)
        energy_used = travel_dist * energy_rate
        cumulative_energy += energy_used
        battery -= energy_used

        # CS node: charge to full (energy reset)
        if node_id > n_cust:
            battery = battery_capacity
            # Recompute cumulative from this point (energy cost already counted)
            energy_needed_to_node.append(cumulative_energy)
            energy_at_node.append(battery)
            # Reset cumulative for "from this node" calculations
            # Actually, we keep cumulative_energy as total energy cost
            # but for CS nodes, battery is reset
        else:
            energy_needed_to_node.append(cumulative_energy)
            energy_at_node.append(battery)

            if battery < -0.01:
                shortfall_nodes.append((i, node_id, abs(battery)))

        prev = node_id

    # Return to depot
    if prev != 0:
        return_dist = _node_distance(prev, 0, customers, dist_matrix,
                                      cs_coords, depot)
        return_energy = return_dist * energy_rate
        cumulative_energy += return_energy
        battery -= return_energy
        # Check if return-to-depot causes shortfall
        # If so, flag the LAST node as the point before which a CS is needed
        if battery < -0.01 and len(route) > 0:
            last_idx = len(route) - 1
            last_id = route[last_idx]
            # Only add if not already in shortfall_nodes
            already = any(s[0] == last_idx for s in shortfall_nodes)
            if not already:
                shortfall_nodes.append((last_idx, last_id, abs(battery)))

    # Compute energy needed FROM each node to end
    energy_needed_from_node = []
    for i in range(n):
        if i == 0:
            energy_from_here = cumulative_energy
        else:
            energy_from_here = cumulative_energy - energy_needed_to_node[i - 1]
        energy_needed_from_node.append(energy_from_here)

    return {
        'energy_at_node': energy_at_node,
        'energy_needed_to_node': energy_needed_to_node,
        'energy_needed_from_node': energy_needed_from_node,
        'total_energy': cumulative_energy,
        'shortfall_nodes': shortfall_nodes,
    }


def _simulate_route_with_cs(route_with_cs, customers, dist_matrix, cs_coords,
                             depot, battery_capacity, energy_rate,
                             charging_model='nonlinear'):
    """
    Simulate a route that may contain charging station nodes.

    Returns dict with same keys as simulate_route_ev(), plus:
      - any_tardy: bool — True if any customer arrives after due_time
      - tardy_customers: list of (node_id, arrival, due_time) for tardy customers
    """
    n_cust = len(customers)
    current_time = 0.0
    battery = battery_capacity
    prev = 0
    any_tardy = False
    tardy_customers = []
    total_dist = 0.0
    total_energy = 0.0
    total_tardiness = 0.0
    n_charges = 0
    total_charge_time = 0.0
    energy_violation = 0.0

    for node_id in route_with_cs:
        # Distance (handles CS nodes properly)
        d = _node_distance(prev, node_id, customers, dist_matrix, cs_coords, depot)
        total_dist += d

        # Energy
        energy_used = d * energy_rate
        battery -= energy_used
        total_energy += energy_used
        if battery < -0.01:
            energy_violation += abs(battery)
            battery = 0.0

        # Time
        current_time += d / TRUCK_SPEED

        if node_id > n_cust:
            # ── Charging station ──
            # Compute energy needed for rest of route
            remaining_energy = 0.0
            rem_prev = node_id
            for future_id in route_with_cs[route_with_cs.index(node_id) + 1:]:
                remaining_energy += (_node_distance(rem_prev, future_id, customers,
                                                    dist_matrix, cs_coords, depot)
                                     * energy_rate)
                rem_prev = future_id
            remaining_energy += (_node_distance(rem_prev, 0, customers,
                                                dist_matrix, cs_coords, depot)
                                 * energy_rate)

            # Optimal charge amount: enough to finish + margin, cap at 100%
            # Smart charge: just enough to finish + margin
            target_energy = min(battery_capacity,
                                remaining_energy + BATTERY_SAFETY_MARGIN)
            energy_to_add = max(0.0, target_energy - battery)

            if charging_model == 'nonlinear':
                soc_pct = max(0.0, (battery / battery_capacity) * 100.0)
                charge_time = compute_nonlinear_charge_time(
                    soc_pct, energy_to_add, battery_capacity, CHARGING_RATE)
            else:
                charge_time = energy_to_add / CHARGING_RATE if CHARGING_RATE > 0 else 0

            battery += energy_to_add
            battery = min(battery, battery_capacity)
            current_time += charge_time
            n_charges += 1
            total_charge_time += charge_time
        else:
            # ── Customer ──
            c = customers[node_id - 1]
            start_time = max(current_time, c['ready_time'])
            if start_time > c['due_time'] + 0.01:
                any_tardy = True
                tardy_customers.append((node_id, start_time, c['due_time']))
                total_tardiness += (start_time - c['due_time'])
            current_time = start_time + c['service_time']

        prev = node_id

    # Return to depot
    return_dist = _node_distance(prev, 0, customers, dist_matrix, cs_coords, depot)
    total_dist += return_dist
    battery -= return_dist * energy_rate
    total_energy += return_dist * energy_rate
    if battery < -0.01:
        energy_violation += abs(battery)

    return {
        'total_dist': total_dist,
        'total_time': current_time,
        'total_tardiness': total_tardiness,
        'total_energy': total_energy,
        'energy_violation': energy_violation,
        'n_charges': n_charges,
        'total_charge_time': total_charge_time,
        'any_tardy': any_tardy,
        'tardy_customers': tardy_customers,
        'ev_feasible': energy_violation <= 0.01,  # Energy-only
        'feasible': energy_violation <= 0.01 and not any_tardy,  # Energy + TW
    }


def insert_charging_stops(routes, customers, dist_matrix, instance,
                           battery_capacity=None, energy_rate=None):
    """
    Insert charging station visits into truck routes where needed.
    (LEGACY — simple greedy, kept for backward compatibility.)

    For improved version with look-ahead and TW-awareness, use
    insert_charging_stops_lookahead().
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
        prev = 0
        battery = battery_capacity

        for node_id in route:
            travel_dist = dist_matrix[prev][node_id]
            energy_needed = travel_dist * energy_rate

            if battery - energy_needed < 0:
                violations_before += (energy_needed - battery)
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
                    d_to_cs = station_distance(prev, best_cs, customers, cs_coords, depot)
                    battery -= d_to_cs * energy_rate
                    battery = battery_capacity
                    prev = best_cs
                    total_insertions += 1

                    d_from_cs = station_distance(best_cs, node_id, customers, cs_coords, depot)
                    battery -= d_from_cs * energy_rate
                else:
                    battery -= energy_needed
            else:
                battery -= energy_needed

            new_route.append(node_id)
            prev = node_id

        new_routes.append(new_route)

    return new_routes, {
        'n_insertions': total_insertions,
        'energy_violations_before': violations_before,
    }


def insert_charging_stops_lookahead(routes, customers, dist_matrix, instance,
                                     battery_capacity=None, energy_rate=None,
                                     charging_model='nonlinear', check_tw=True):
    """
    Improved charging stop insertion with look-ahead planning.

    KEY IMPROVEMENTS over the legacy insert_charging_stops():
    1. LOOK-AHEAD: Pre-computes entire route energy profile, finds optimal
       charging positions BEFORE battery runs out (not after).
    2. SMART CHARGE AMOUNT: Charges only what's needed to finish the route
       (+ safety margin), not blindly to 100%. Uses non-linear charge time
       computation for accurate timing.
    3. TW-AWARE (check_tw=True): After inserting a CS, simulates the full
       route to verify no customer becomes tardy. Rejects insertions that
       would cause tardiness — tries alternative CS or position.
    4. OPTIMAL PLACEMENT: Tries inserting CS at every position between
       the last charge and the shortfall point, not just the segment
       immediately before failure.
    5. ALL-CS EVALUATION: Evaluates all charging stations at each candidate
       position, not just the one with minimum detour.

    NON-LINEAR CHARGING: When charging_model='nonlinear', the charge time
    is computed by inverting the piecewise charging curve (fast 0-20%,
    normal 20-80%, slow 80-100%). This means charging from 10%→50% takes
    much less time than 70%→100% for the same kWh — the algorithm exploits
    this by preferring partial charges in the fast-charging region.

    Args:
        routes: list of truck routes (each is list of customer IDs)
        customers: customer data list
        dist_matrix: distance matrix (depot+customers only)
        instance: full instance dict
        battery_capacity: max battery kWh (default: BATTERY_CAPACITY)
        energy_rate: kWh/km (default: ENERGY_CONSUMPTION_RATE)
        charging_model: 'linear' or 'nonlinear'
        check_tw: if True, reject CS insertions that cause tardiness

    Returns:
        (new_routes, stats) where stats has:
          - n_insertions: total CS nodes inserted
          - n_routes_modified: number of routes that got CS
          - energy_violations_before: total kWh shortfall before insertion
          - energy_violations_after: total kWh shortfall after insertion
          - total_charge_time: total minutes spent charging
          - tw_rejections: number of times a CS was rejected for causing tardiness
          - model: charging model used
    """
    if battery_capacity is None:
        battery_capacity = BATTERY_CAPACITY
    if energy_rate is None:
        energy_rate = ENERGY_CONSUMPTION_RATE

    n_cust = len(customers)
    cs_coords = get_charging_station_coords(n_cust)
    cs_ids = sorted(cs_coords.keys())
    depot = instance.get('depot', DEPOT)
    if isinstance(depot, list):
        depot = tuple(depot)

    new_routes = []
    total_insertions = 0
    routes_modified = 0
    violations_before_total = 0.0
    violations_after_total = 0.0
    total_charge_time_all = 0.0
    tw_rejections = 0

    for ri, route in enumerate(routes):
        if not route:
            new_routes.append([])
            continue
        if len(route) == 0:
            new_routes.append([])
            continue

        # ── Phase 1: Energy profile ──
        profile = _route_energy_profile(
            route, customers, dist_matrix, cs_coords, depot,
            energy_rate, battery_capacity)

        # If no shortfall, route is EV-feasible as-is
        if not profile['shortfall_nodes']:
            new_routes.append(list(route))
            continue

        violations_before_total += sum(s[2] for s in profile['shortfall_nodes'])

        # ── Phase 2: Iterative look-ahead repair ──
        working_route = list(route)
        route_insertions = 0
        MAX_CS_PER_ROUTE = 3  # Realistic for urban routes; more = diminishing returns

        for _ in range(MAX_CS_PER_ROUTE):
            # Recompute profile on current working route
            profile = _route_energy_profile(
                working_route, customers, dist_matrix, cs_coords, depot,
                energy_rate, battery_capacity)

            if not profile['shortfall_nodes']:
                break  # All fixed!

            # Take the FIRST shortfall (earliest in route)
            shortfall_idx, shortfall_node, deficit = profile['shortfall_nodes'][0]

            # ── Try all (position, CS) combinations ──
            best_solution = None  # (insert_after_idx, cs_id, route_with_cs, sim_result)
            best_tw_violating = None  # Best solution that violates TW (fallback)

            # Try inserting CS at each position from 0 (after depot) up to
            # the customer BEFORE the shortfall
            for insert_after in range(-1, shortfall_idx):
                for cs_id in cs_ids:
                    # Build candidate route with CS inserted
                    if insert_after == -1:
                        candidate = [cs_id] + working_route
                    else:
                        candidate = (working_route[:insert_after + 1] +
                                    [cs_id] +
                                    working_route[insert_after + 1:])

                    # Simulate this candidate
                    sim = _simulate_route_with_cs(
                        candidate, customers, dist_matrix, cs_coords, depot,
                        battery_capacity, energy_rate, charging_model)

                    if check_tw and sim['any_tardy']:
                        tw_rejections += 1
                        # Keep track of best TW-violating solution as fallback
                        if best_tw_violating is None or (
                                sim['energy_violation'] < best_tw_violating['sim']['energy_violation']
                        ):
                            best_tw_violating = {
                                'insert_after': insert_after,
                                'cs_id': cs_id,
                                'route': candidate,
                                'sim': sim,
                            }
                        continue

                    # TW-feasible: score by energy violation first, then distance
                    if sim['energy_violation'] <= 0.01:
                        score = sim['total_dist'] + sim['total_charge_time'] * 0.1
                        if best_solution is None or score < best_solution['score']:
                            best_solution = {
                                'insert_after': insert_after,
                                'cs_id': cs_id,
                                'route': candidate,
                                'sim': sim,
                                'score': score,
                            }
                    else:
                        if best_solution is None or (
                                sim['energy_violation'] < best_solution['sim']['energy_violation']
                        ):
                            best_solution = {
                                'insert_after': insert_after,
                                'cs_id': cs_id,
                                'route': candidate,
                                'sim': sim,
                                'score': sim['energy_violation'],
                            }

            # If no TW-feasible solution, use best TW-violating one (with warning)
            if best_solution is None and best_tw_violating is not None:
                best_solution = best_tw_violating

            if best_solution is None:
                break  # Cannot fix

            # Accept the best insertion
            working_route = best_solution['route']
            route_insertions += 1
            total_charge_time_all += best_solution['sim']['total_charge_time']

            if best_solution['sim']['energy_violation'] <= 0.01:
                break  # Route is now EV-feasible

        # ── Phase 3: Final verification ──
        final_sim = _simulate_route_with_cs(
            working_route, customers, dist_matrix, cs_coords, depot,
            battery_capacity, energy_rate, charging_model)

        violations_after_total += final_sim['energy_violation']
        total_insertions += route_insertions
        if route_insertions > 0:
            routes_modified += 1

        new_routes.append(working_route)

    return new_routes, {
        'n_insertions': total_insertions,
        'n_routes_modified': routes_modified,
        'energy_violations_before': round(violations_before_total, 2),
        'energy_violations_after': round(violations_after_total, 2),
        'total_charge_time': round(total_charge_time_all, 2),
        'tw_rejections': tw_rejections,
        'model': charging_model,
        'method': 'lookahead',
    }


# ═══════════════════════════════════════════════════════════════════════════
# EV Solution Class
# ═══════════════════════════════════════════════════════════════════════════

class EVTruckSolution(TruckSolution):
    """
    Extends TruckSolution with EV battery-aware evaluation.

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

    def __init__(self, truck_routes, instance,
                 charging_model='linear',
                 battery_capacity=None, energy_rate=None):
        super().__init__(truck_routes, instance)
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

    def _evaluate(self):
        """
        Override base _evaluate() to handle routes that may contain
        charging station nodes (IDs > n_customers).

        Uses simulate_route_ev() for distance/time/energy computation,
        which correctly handles CS detours, charge times, and their
        impact on downstream customer arrival times.
        """
        inst = self.instance
        customers = inst['customers']
        dist_matrix = inst['distance_matrix']
        depot = inst.get('depot', DEPOT)
        if isinstance(depot, list):
            depot = tuple(depot)
        n_cust = len(customers)
        cs_coords = get_charging_station_coords(n_cust)

        total_distance = 0.0
        total_tardiness = 0.0
        total_cost = 0.0
        feasible = True
        violations = {'capacity': 0.0, 'time_window': 0.0}
        served_customers = set()
        n_trucks_used = 0

        for route in self.truck_routes:
            if not route:
                continue
            n_trucks_used += 1

            sim = simulate_route_ev(
                route, customers, dist_matrix, cs_coords, depot,
                battery_capacity=self.battery_capacity,
                charging_model='none',  # Cost eval ignores charging time
                energy_rate=self.energy_rate,
            )

            total_distance += sim['total_dist']
            total_tardiness += sim['total_tardiness'] * TARDINESS_COST_RATE
            violations['capacity'] += sim.get('capacity_violation', 0.0)

            # Track served customers (skip CS nodes)
            for node_id in route:
                if 1 <= node_id <= n_cust:
                    served_customers.add(node_id)

        # Cost = fixed cost per truck + distance cost
        total_cost = (n_trucks_used * TRUCK_FIXED_COST +
                      total_distance * TRUCK_DIST_COST_RATE +
                      total_tardiness)
        feasible = (violations['capacity'] <= 0.01 and
                     total_tardiness <= TARDINESS_COST_RATE * 0.01)

        self._cost = total_cost
        self._tardiness = total_tardiness / TARDINESS_COST_RATE if TARDINESS_COST_RATE > 0 else total_tardiness
        self._feasible = feasible
        self._violations = violations

    def _evaluate_ev(self):
        """Evaluate EV-specific metrics: battery, charging, energy violations."""
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
            if not sim.get('ev_feasible', sim['feasible']):
                total_ev_feasible = False
            route_details.append(sim)

        # EV feasibility = energy-only (capacity is a routing concern)
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
        new._ev_evaluated = False  # Force re-evaluation
        return new


# ═══════════════════════════════════════════════════════════════════════════
# EV Solution Evaluation (drop-in replacement for evaluate_solution_batch)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_ev_solution(sol, charging_model='linear'):
    """
    Evaluate a TruckSolution or EVTruckSolution with EV constraints.

    Returns dict with EV-specific metrics added.
    """
    if isinstance(sol, EVTruckSolution):
        sol.charging_model = charging_model
        sol._ev_evaluated = False
        sol._evaluate_ev()

    base = {
        'cost': sol.cost,
        'tardiness': sol.tardiness,
        'feasible': sol.feasible,
    }

    if isinstance(sol, EVTruckSolution):
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
