# -*- coding: utf-8 -*-
"""
Sync-Aware Solution Evaluator — Model D.

Implements proper truck-drone synchronization with cascading truck waiting time.

Key difference from problem_model._evaluate():
  - Truck WAITS at recovery node k if drone hasn't arrived yet
  - Waiting time cascades to all subsequent nodes (may cause TW violations)
  - Sync violations are REAL constraints, not 0.01-weighted hints

Algorithm (two-pass to handle EDD-reordered routes where k may precede i):
  Pass 1: Compute base truck schedules (arrival/departure times) WITHOUT drone sync.
  Pass 2: For each drone mission, compute if truck must wait at recovery node k,
          then cascade delays forward through the route.
  Track: sync_wait_time, cascaded_tardiness, n_hard_sync_violations

Model D vs Model C (current):
  Model C: drone insertion uses hard GO/NO-GO filter; evaluation ignores sync
  Model D: drone insertion allows truck waiting; evaluation models cascading delays
"""

import math
from copy import deepcopy

# Constants (must match week3 config)
TRUCK_SPEED = 35.0       # km/h
DRONE_SPEED = 50.0       # km/h
DRONE_CAPACITY = 40.0
TRUCK_CAPACITY = 200.0
TRUCK_FIXED_COST = 100.0
DRONE_FIXED_COST = 0.0
TRUCK_DIST_COST_RATE = 2.0
DRONE_DIST_COST_RATE = 1.0
TARDINESS_COST_RATE = 1.0
DEPOT = (8.0, 8.0)


def evaluate_sync_aware(solution, instance):
    """
    Evaluate a TruckDroneSolution with full truck-drone synchronization.

    Two-pass algorithm (handles EDD-reordered routes where k may precede i):
      Pass 1: Compute base truck schedules WITHOUT drone sync.
              Record arrival/departure times at each node.
      Pass 2: For each drone mission, compute truck waiting at recovery k,
              then cascade delays forward through the route.

    Args:
        solution: TruckDroneSolution with truck_routes and drone_missions
        instance: problem instance dict

    Returns:
        dict with keys:
            cost, tardiness, feasible, violations, drone_util,
            sync_wait_time, cascaded_tardiness, makespan,
            per_route_sync_wait, sync_details
    """
    customers = instance['customers']
    dist = instance['distance_matrix']
    n_trucks = len(solution.truck_routes)
    max_drones = getattr(solution, 'max_drones_per_truck', 2)

    # ── Parse drone missions ──
    parsed_missions = []
    for mission in solution.drone_missions:
        if len(mission) >= 4:
            i, j, k, drone_id = mission[0], mission[1], mission[2], mission[3]
        else:
            i, j, k = mission[0], mission[1], mission[2]
            drone_id = 0
        parsed_missions.append({'i': i, 'j': j, 'k': k, 'drone_id': drone_id})

    drone_served_customers = set(m['j'] for m in parsed_missions)
    served_customers = set(drone_served_customers)

    # ── Track metrics ──
    total_tardiness = 0.0
    total_sync_wait = 0.0
    cascaded_tardiness = 0.0
    feasible = True
    makespan = 0.0

    violations = {
        'capacity': 0, 'time_window': 0,
        'drone_endurance': 0, 'drone_capacity': 0,
        'sync': 0, 'sync_wait': 0, 'max_drones_per_truck': 0,
    }

    total_truck_dist = 0.0
    total_drone_dist = 0.0
    n_drones_used = 0

    per_route_sync_wait = []
    sync_details = []
    truck_drone_ops = {ti: [] for ti in range(n_trucks)}

    # ═══════════════════════════════════════════════════════════════════
    # PASS 1: Compute base schedules (no drone sync waiting)
    # ═══════════════════════════════════════════════════════════════════
    # Per-route, per-position: (arrival_time, departure_time)
    route_schedules = []  # list of lists of (arrival, departure, cust_idx)

    for ti, route in enumerate(solution.truck_routes):
        if not route:
            route_schedules.append([])
            per_route_sync_wait.append(0.0)
            continue

        schedule = []
        prev = 0
        load = 0.0
        current_time = 0.0
        route_dist = 0.0

        for idx, cust_idx in enumerate(route):
            c_data = customers[cust_idx - 1]
            served_customers.add(cust_idx)

            seg_dist = dist[prev][cust_idx]
            route_dist += seg_dist
            current_time += seg_dist / TRUCK_SPEED

            ready = c_data['ready_time']
            due = c_data['due_time']
            if current_time < ready:
                current_time = ready

            arrival = current_time

            # Capacity check
            load += c_data['demand']
            if load > TRUCK_CAPACITY:
                violations['capacity'] += (load - TRUCK_CAPACITY)
                feasible = False

            current_time += c_data['service_time']
            departure = current_time

            schedule.append({
                'cust_idx': cust_idx,
                'arrival': arrival,
                'departure': departure,
                'due': due,
                'ready': ready,
            })

            prev = cust_idx

        # Return to depot
        route_dist += dist[prev][0]
        current_time += dist[prev][0] / TRUCK_SPEED

        total_truck_dist += route_dist
        route_schedules.append(schedule)

    # ═══════════════════════════════════════════════════════════════════
    # Find owning truck for each drone mission
    # ═══════════════════════════════════════════════════════════════════
    mission_assignments = []  # (ti, mission_dict, launch_pos, recovery_pos)
    orphan_missions = []

    for m in parsed_missions:
        i, j, k, drone_id = m['i'], m['j'], m['k'], m['drone_id']

        # Find owning truck (must have both i and k in same route)
        owner_ti = None
        launch_pos = None
        recovery_pos = None

        for ti, route in enumerate(solution.truck_routes):
            if i == 0 or i in route:
                if k == 0 or k in route:
                    owner_ti = ti
                    launch_pos = -1 if i == 0 else route.index(i)
                    recovery_pos = -1 if k == 0 else route.index(k)
                    break

        if owner_ti is None:
            orphan_missions.append(m)
            violations['sync'] += 100.0
            feasible = False
            sync_details.append({
                'error': f'Orphan mission: i={i}, j={j}, k={k} — no truck has both i and k'
            })
            continue

        m['owner_ti'] = owner_ti
        m['launch_pos'] = launch_pos
        m['recovery_pos'] = recovery_pos
        mission_assignments.append(m)

    # ═══════════════════════════════════════════════════════════════════
    # Compute drone flight parameters
    # ═══════════════════════════════════════════════════════════════════
    for m in mission_assignments:
        i, j, k = m['i'], m['j'], m['k']
        ti = m['owner_ti']

        cj = customers[j - 1]

        # Drone distance
        d_ij = _node_dist(i, j, dist, customers)
        d_jk = _node_dist(j, k, dist, customers)
        drone_leg = d_ij + d_jk
        total_drone_dist += drone_leg
        n_drones_used += 1

        m['drone_leg'] = drone_leg

        # Drone capacity
        if cj['demand'] > DRONE_CAPACITY:
            violations['drone_capacity'] += (cj['demand'] - DRONE_CAPACITY)
            feasible = False

        # Drone endurance
        if drone_leg > 6.0:
            violations['drone_endurance'] += (drone_leg - 6.0)
            feasible = False

        # Drone flight time
        drone_flight_time = drone_leg / DRONE_SPEED + cj['service_time']
        m['drone_flight_time'] = drone_flight_time

        # Get launch time from schedule
        if i == 0:
            launch_time = 0.0
        else:
            sched = route_schedules[ti]
            launch_entry = sched[m['launch_pos']]
            launch_time = launch_entry['departure']

        m['launch_time'] = launch_time
        m['drone_arrival_at_k'] = launch_time + drone_flight_time

    # ═══════════════════════════════════════════════════════════════════
    # PASS 2: Apply sync waiting and cascade delays
    # ═══════════════════════════════════════════════════════════════════
    # Group missions by (truck, recovery_node)
    recovery_waits = {}  # (ti, pos) -> max wait needed

    for m in mission_assignments:
        ti = m['owner_ti']
        k = m['k']
        drone_arrival = m['drone_arrival_at_k']

        if k == 0:
            # Recovery at depot — compute final route time
            sched = route_schedules[ti]
            if sched:
                # Truck arrival at depot = last departure + return travel
                last_cust = sched[-1]['cust_idx']
                depot_dist = dist[last_cust][0]
                truck_depot_arrival = sched[-1]['departure'] + depot_dist / TRUCK_SPEED
            else:
                truck_depot_arrival = 0.0

            if drone_arrival > truck_depot_arrival:
                wait = drone_arrival - truck_depot_arrival
                key = (ti, -1)  # -1 = depot
                recovery_waits[key] = max(recovery_waits.get(key, 0), wait)
                m['wait_needed'] = wait
            else:
                m['wait_needed'] = 0.0
            continue

        # Find recovery position in schedule
        recovery_pos = m['recovery_pos']
        sched = route_schedules[ti]
        recovery_entry = sched[recovery_pos]
        truck_arrival = recovery_entry['arrival']

        if drone_arrival > truck_arrival:
            wait = drone_arrival - truck_arrival
            key = (ti, recovery_pos)
            recovery_waits[key] = max(recovery_waits.get(key, 0), wait)
            m['wait_needed'] = wait
        else:
            m['wait_needed'] = 0.0

    # ── Apply waiting and cascade ──
    # For each route, apply accumulated waits at each position
    route_final_schedules = []

    for ti, sched in enumerate(route_schedules):
        if not sched:
            route_final_schedules.append([])
            makespan = max(makespan, 0.0)
            continue

        # Build cumulative delay array
        cum_delay = 0.0
        route_sync_wait = 0.0
        final = []

        for pos, entry in enumerate(sched):
            # Add wait at this position (from drone recovery)
            wait_key = (ti, pos)
            if wait_key in recovery_waits:
                cum_delay += recovery_waits[wait_key]
                route_sync_wait += recovery_waits[wait_key]

            # Apply cumulated delay to arrival and departure
            new_arrival = entry['arrival'] + cum_delay
            new_departure = entry['departure'] + cum_delay

            # Time window check (post-delay)
            if new_arrival > entry['due']:
                tardy = new_arrival - entry['due']
                total_tardiness += tardy * TARDINESS_COST_RATE
                violations['time_window'] += tardy
                # Was this caused by sync?
                if entry['arrival'] <= entry['due']:
                    cascaded_tardiness += tardy

            # Re-apply TW waiting (if early after delays, wait for TW)
            if new_arrival < entry['ready']:
                tw_wait = entry['ready'] - new_arrival
                new_arrival += tw_wait
                new_departure += tw_wait

            final.append({
                'cust_idx': entry['cust_idx'],
                'arrival': new_arrival,
                'departure': new_departure,
                'due': entry['due'],
            })

        total_sync_wait += route_sync_wait
        per_route_sync_wait.append(route_sync_wait)

        # Compute final makespan (return to depot)
        if final:
            last_cust = final[-1]['cust_idx']
            depot_dist = dist[last_cust][0]
            final_time = final[-1]['departure'] + depot_dist / TRUCK_SPEED
            # Add depot recovery waits
            depot_key = (ti, -1)
            if depot_key in recovery_waits:
                final_time += recovery_waits[depot_key]
                total_sync_wait += recovery_waits[depot_key]
            makespan = max(makespan, final_time)

        route_final_schedules.append(final)

    # ═══════════════════════════════════════════════════════════════════
    # Build sync_details and drone ops tracking
    # ═══════════════════════════════════════════════════════════════════
    for m in mission_assignments:
        ti = m['owner_ti']
        detail = {
            'i': m['i'], 'j': m['j'], 'k': m['k'],
            'drone_id': m['drone_id'],
            'launch_time': m['launch_time'],
            'drone_arrival_at_k': m['drone_arrival_at_k'],
            'drone_leg': m['drone_leg'],
            'wait_needed': m.get('wait_needed', 0.0),
        }
        sync_details.append(detail)

        # Compute actual recovery time (with sync waiting applied)
        if m['k'] == 0:
            recovery_time = m['drone_arrival_at_k']  # drone sets the depot arrival
        else:
            sched = route_final_schedules[ti] if ti < len(route_final_schedules) else []
            if m['recovery_pos'] < len(sched):
                recovery_time = max(
                    m['drone_arrival_at_k'],
                    sched[m['recovery_pos']]['arrival']
                )
            else:
                recovery_time = m['drone_arrival_at_k']

        truck_drone_ops[ti].append({
            'drone_id': m['drone_id'],
            'launch_time': m['launch_time'],
            'recovery_time': recovery_time,
            'launch_node': m['i'],
            'recovery_node': m['k'],
        })

    # ═══════════════════════════════════════════════════════════════════
    # Simultaneous flight constraint check
    # ═══════════════════════════════════════════════════════════════════
    for ti, ops in truck_drone_ops.items():
        if len(ops) <= 1:
            continue
        ops_sorted = sorted(ops, key=lambda o: o['launch_time'])
        active_at_times = []
        for op in ops_sorted:
            active_at_times.append((op['launch_time'], +1))
            active_at_times.append((op['recovery_time'], -1))
        active_at_times.sort(key=lambda x: x[0])
        current_active = 0
        for t, delta in active_at_times:
            current_active += delta
            if current_active > max_drones:
                violations['max_drones_per_truck'] += (current_active - max_drones)
                feasible = False

        for d_id in range(max_drones):
            d_ops = [o for o in ops_sorted if o['drone_id'] == d_id]
            for a_idx in range(len(d_ops)):
                for b_idx in range(a_idx + 1, len(d_ops)):
                    a, b = d_ops[a_idx], d_ops[b_idx]
                    if (a['launch_time'] < b['recovery_time'] and
                        b['launch_time'] < a['recovery_time']):
                        violations['sync'] += 50.0
                        feasible = False

    # ═══════════════════════════════════════════════════════════════════
    # Cost calculation
    # ═══════════════════════════════════════════════════════════════════
    vehicle_fixed = n_trucks * TRUCK_FIXED_COST + n_drones_used * DRONE_FIXED_COST
    distance_cost = total_truck_dist * TRUCK_DIST_COST_RATE + total_drone_dist * DRONE_DIST_COST_RATE
    total_cost = vehicle_fixed + distance_cost
    total_cost += total_sync_wait * TARDINESS_COST_RATE

    # Unserved customers
    all_customers = set(range(1, instance['n_customers'] + 1))
    unserved = all_customers - served_customers
    if unserved:
        feasible = False
        total_cost += len(unserved) * 1000.0

    violations['sync_wait'] = total_sync_wait

    drone_util = {
        'n_drones_used': n_drones_used,
        'n_drone_customers': len(drone_served_customers),
        'total_drone_distance': total_drone_dist,
        'drone_served_set': drone_served_customers,
        'max_drones_per_truck': max_drones,
        'per_truck_drone_counts': {ti: len(ops) for ti, ops in truck_drone_ops.items()},
    }

    return {
        'cost': total_cost,
        'tardiness': total_tardiness,
        'feasible': feasible,
        'violations': violations,
        'drone_util': drone_util,
        'sync_wait_time': total_sync_wait,
        'cascaded_tardiness': cascaded_tardiness,
        'makespan': makespan,
        'per_route_sync_wait': per_route_sync_wait,
        'sync_details': sync_details,
        'total_truck_dist': total_truck_dist,
        'total_drone_dist': total_drone_dist,
        'n_drones_used': n_drones_used,
        'n_trucks': n_trucks,
    }


def evaluate_no_sync(solution, instance):
    """
    Evaluate WITHOUT sync constraints (Model C behavior).

    This is the current _evaluate() logic but returns the same dict format
    as evaluate_sync_aware() for direct comparison.

    Key: truck does NOT wait for drones. Drone arrival after truck = soft violation.
    """
    customers = instance['customers']
    dist = instance['distance_matrix']
    n_trucks = len(solution.truck_routes)
    max_drones = getattr(solution, 'max_drones_per_truck', 2)

    total_cost = 0.0
    total_tardiness = 0.0
    total_sync_wait = 0.0
    feasible = True
    makespan = 0.0

    violations = {
        'capacity': 0,
        'time_window': 0,
        'drone_endurance': 0,
        'drone_capacity': 0,
        'sync': 0,
        'sync_wait': 0,
        'max_drones_per_truck': 0,
    }

    drone_served_customers = set()
    for mission in solution.drone_missions:
        drone_served_customers.add(mission[1])

    served_customers = set(drone_served_customers)
    total_truck_dist = 0.0
    total_drone_dist = 0.0
    n_drones_used = 0

    # Track node times for sync check
    node_arrival_times = {}
    node_departure_times = {}

    # ── Evaluate truck routes (no sync waiting) ──
    for route in solution.truck_routes:
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
            ready = c_data['ready_time']
            due = c_data['due_time']
            if current_time < ready:
                current_time = ready
            if current_time > due:
                tardy = current_time - due
                total_tardiness += tardy * TARDINESS_COST_RATE
                violations['time_window'] += tardy
            node_arrival_times[cust_idx] = current_time
            load += c_data['demand']
            if load > TRUCK_CAPACITY:
                violations['capacity'] += (load - TRUCK_CAPACITY)
                feasible = False
            current_time += c_data['service_time']
            node_departure_times[cust_idx] = current_time
            prev = cust_idx
        route_dist += dist[prev][0]
        current_time += dist[prev][0] / TRUCK_SPEED
        total_truck_dist += route_dist
        makespan = max(makespan, current_time)

    # ── Find truck for each mission ──
    def _find_truck(i, k):
        for ti, route in enumerate(solution.truck_routes):
            if (i == 0 or i in route) and (k == 0 or k in route):
                return ti
        return None

    truck_drone_ops = {ti: [] for ti in range(n_trucks)}

    # ── Evaluate drone missions (no sync waiting) ──
    for mission in solution.drone_missions:
        if len(mission) >= 4:
            i, j, k, drone_id = mission[0], mission[1], mission[2], mission[3]
        else:
            i, j, k = mission[0], mission[1], mission[2]
            drone_id = 0

        d_ij = _node_dist(i, j, dist, customers)
        d_jk = _node_dist(j, k, dist, customers)
        drone_leg = d_ij + d_jk
        total_drone_dist += drone_leg
        n_drones_used += 1

        if drone_leg > 6.0:
            violations['drone_endurance'] += (drone_leg - 6.0)
            feasible = False

        cj = customers[j - 1]
        if cj['demand'] > DRONE_CAPACITY:
            violations['drone_capacity'] += (cj['demand'] - DRONE_CAPACITY)
            feasible = False

        launch_truck = _find_truck(i, k)
        if launch_truck is None:
            violations['sync'] += 999.0
            feasible = False
            continue

        truck_depart_i = node_departure_times.get(i, 0.0) if i > 0 else 0.0
        truck_arrive_k = node_arrival_times.get(k, float('inf')) if k > 0 else float('inf')

        drone_flight_time = drone_leg / DRONE_SPEED + cj['service_time']
        drone_arrive_k = truck_depart_i + drone_flight_time

        # No sync: drone hovering is a soft tracking metric only
        if drone_arrive_k > truck_arrive_k:
            hover = drone_arrive_k - truck_arrive_k
            violations['sync'] += hover * 0.01

        # Record for simultaneous flight check
        truck_drone_ops[launch_truck].append({
            'drone_id': drone_id,
            'launch_time': truck_depart_i,
            'recovery_time': max(drone_arrive_k, truck_arrive_k),
            'launch_node': i,
            'recovery_node': k,
        })

    # ── Simultaneous flight check ──
    for ti, ops in truck_drone_ops.items():
        if len(ops) <= 1:
            continue
        ops_sorted = sorted(ops, key=lambda o: o['launch_time'])
        active_at_times = []
        for op in ops_sorted:
            active_at_times.append((op['launch_time'], +1))
            active_at_times.append((op['recovery_time'], -1))
        active_at_times.sort(key=lambda x: x[0])
        current_active = 0
        for t, delta in active_at_times:
            current_active += delta
            if current_active > max_drones:
                violations['max_drones_per_truck'] += (current_active - max_drones)
                feasible = False

        for d_id in range(max_drones):
            d_ops = [o for o in ops_sorted if o['drone_id'] == d_id]
            for a_idx in range(len(d_ops)):
                for b_idx in range(a_idx + 1, len(d_ops)):
                    a, b = d_ops[a_idx], d_ops[b_idx]
                    if (a['launch_time'] < b['recovery_time'] and
                        b['launch_time'] < a['recovery_time']):
                        violations['sync'] += 50.0
                        feasible = False

    # ── Cost ──
    vehicle_fixed = n_trucks * TRUCK_FIXED_COST + n_drones_used * DRONE_FIXED_COST
    distance_cost = total_truck_dist * TRUCK_DIST_COST_RATE + total_drone_dist * DRONE_DIST_COST_RATE
    total_cost = vehicle_fixed + distance_cost

    all_customers = set(range(1, instance['n_customers'] + 1))
    unserved = all_customers - served_customers
    if unserved:
        feasible = False
        total_cost += len(unserved) * 1000.0

    drone_util = {
        'n_drones_used': n_drones_used,
        'n_drone_customers': len(drone_served_customers),
        'total_drone_distance': total_drone_dist,
        'drone_served_set': drone_served_customers,
        'max_drones_per_truck': max_drones,
        'per_truck_drone_counts': {ti: len(ops) for ti, ops in truck_drone_ops.items()},
    }

    return {
        'cost': total_cost,
        'tardiness': total_tardiness,
        'feasible': feasible,
        'violations': violations,
        'drone_util': drone_util,
        'sync_wait_time': 0.0,   # no sync waiting in Model C
        'cascaded_tardiness': 0.0,
        'makespan': makespan,
        'per_route_sync_wait': [0.0] * n_trucks,
        'sync_details': [],
        'total_truck_dist': total_truck_dist,
        'total_drone_dist': total_drone_dist,
        'n_drones_used': n_drones_used,
        'n_trucks': n_trucks,
    }


# ── Sync-Aware Drone Insertion ──────────────────────────────────────────────

def insert_drones_sync_aware(truck_routes, instance, drone_endurance=4.0,
                              drone_speed=50.0, truck_speed=35.0,
                              drone_capacity=40.0, max_drones_per_truck=2,
                              min_saving=0.5, max_wait_penalty=60.0):
    """
    Sync-aware drone insertion: allows truck waiting at recovery instead of
    hard-rejecting missions where the drone is slower than the truck.

    Key difference from insert_cross_route_drones():
      - Instead of hard GO/NO-GO filter, computes wait_time needed
      - Adds wait_time * TARDINESS_COST_RATE to the effective cost
      - Only rejects if wait_time exceeds max_wait_penalty

    Args:
        truck_routes: list of lists of customer IDs
        instance: problem instance dict
        drone_endurance: max drone flight distance (km)
        drone_speed, truck_speed: km/h
        drone_capacity: max drone payload
        max_drones_per_truck: max drone missions per truck
        min_saving: minimum cost saving to accept (km equivalent)
        max_wait_penalty: max truck waiting time allowed (minutes)

    Returns:
        (new_truck_routes, drone_missions, total_saved, n_drones,
         drone_counts, sync_stats)
        sync_stats: dict with total_wait_time, n_wait_missions, per_mission waits
    """
    customers = instance['customers']
    depot = instance['depot']
    dist = instance['distance_matrix']

    n_trucks = len(truck_routes)
    routes = [list(r) for r in truck_routes]
    drone_missions = [[] for _ in range(n_trucks)]
    drone_counts = [0] * n_trucks
    total_saved = 0.0
    n_drones = 0

    # Sync statistics
    total_wait_time = 0.0
    n_wait_missions = 0
    per_mission_wait = []

    # Build customer → (truck_idx, position) map
    cust_to_truck = {}
    for ti, route in enumerate(routes):
        for pos, cid in enumerate(route):
            cust_to_truck[cid] = (ti, pos)

    drone_served = set()

    for _pass in range(3):
        candidates = []

        for j_cid in cust_to_truck:
            if j_cid in drone_served:
                continue
            src_truck, src_pos = cust_to_truck[j_cid]
            cj = customers[j_cid - 1]
            if cj['demand'] > drone_capacity:
                continue

            best_net_saving = 0
            best_mission = None

            for launch_truck in range(n_trucks):
                if launch_truck == src_truck:
                    continue
                if drone_counts[launch_truck] >= max_drones_per_truck:
                    continue

                route = routes[launch_truck]
                for i_pos in range(len(route)):
                    i_cid = route[i_pos]
                    k_cid = route[i_pos + 1] if i_pos + 1 < len(route) else 0

                    if i_cid > 0 and i_cid in drone_served:
                        continue
                    if k_cid > 0 and k_cid in drone_served:
                        continue

                    # Endurance check
                    d_ij = _node_dist(i_cid, j_cid, dist, customers)
                    d_jk = _node_dist(j_cid, k_cid, dist, customers)
                    if d_ij + d_jk > drone_endurance:
                        continue

                    # ── Sync-aware feasibility ──
                    truck_segment_time = _compute_truck_segment_time(
                        route, i_pos, k_cid, customers, dist, truck_speed)
                    drone_flight_time = (d_ij + d_jk) / drone_speed
                    drone_mission_time = drone_flight_time + cj['service_time']

                    service_k = (customers[k_cid - 1]['service_time']
                                 if k_cid > 0 else 30.0)

                    # Compute required truck waiting time at k
                    # If drone is faster: no wait needed
                    # If drone is slower: truck must wait = drone_mission_time - truck_segment_time
                    # But truck can use service time at k as buffer
                    wait_needed = max(0.0, drone_mission_time - truck_segment_time - service_k)

                    # Reject if wait exceeds threshold (would cause cascading TW issues)
                    if wait_needed > max_wait_penalty:
                        continue

                    # Cost saving calculation
                    prev_cid = routes[src_truck][src_pos - 1] if src_pos > 0 else 0
                    next_cid = routes[src_truck][src_pos + 1] if src_pos + 1 < len(routes[src_truck]) else 0

                    d_pj = _node_dist(prev_cid, j_cid, dist, customers)
                    d_jn = _node_dist(j_cid, next_cid, dist, customers)
                    d_pn = _node_dist(prev_cid, next_cid, dist, customers)

                    gross_saving = 2.0 * (d_pj + d_jn) - (2.0 * d_pn + 1.0 * (d_ij + d_jk))
                    # Subtract waiting penalty
                    wait_penalty = wait_needed * TARDINESS_COST_RATE / TRUCK_DIST_COST_RATE
                    net_saving = gross_saving - wait_penalty

                    if net_saving > best_net_saving:
                        best_net_saving = net_saving
                        best_mission = (launch_truck, i_cid, j_cid, k_cid,
                                       src_truck, src_pos, wait_needed, gross_saving)

            if best_mission is not None and best_net_saving > min_saving:
                candidates.append((best_net_saving, best_mission, j_cid))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[0], reverse=True)

        for net_saving, mission, j_cid in candidates:
            if j_cid in drone_served:
                continue

            (launch_truck, i_cid, j_cid2, k_cid,
             src_truck, src_pos, wait_needed, gross_saving) = mission

            if drone_counts[launch_truck] >= max_drones_per_truck:
                continue
            if i_cid > 0 and i_cid in drone_served:
                continue
            if k_cid > 0 and k_cid in drone_served:
                continue
            if i_cid > 0 and i_cid not in routes[launch_truck]:
                continue
            if k_cid > 0 and k_cid not in routes[launch_truck]:
                continue
            if j_cid not in cust_to_truck:
                continue
            curr_src_truck, curr_src_pos = cust_to_truck[j_cid]
            if curr_src_truck != src_truck:
                continue

            drone_id = drone_counts[launch_truck]

            # Apply the mission
            routes[src_truck].pop(curr_src_pos)
            drone_missions[launch_truck].append((i_cid, j_cid, k_cid, drone_id))
            drone_counts[launch_truck] += 1
            total_saved += net_saving
            n_drones += 1
            drone_served.add(j_cid)

            if wait_needed > 0.01:
                total_wait_time += wait_needed
                n_wait_missions += 1
            per_mission_wait.append({
                'i': i_cid, 'j': j_cid, 'k': k_cid,
                'wait_needed': wait_needed,
                'gross_saving': gross_saving,
                'net_saving': net_saving,
            })

            del cust_to_truck[j_cid]
            for new_pos, cid in enumerate(routes[src_truck]):
                cust_to_truck[cid] = (src_truck, new_pos)

        # Conflict resolution (same as original)
        _resolve_drone_conflicts(routes, drone_missions, drone_counts,
                                 drone_served, cust_to_truck, n_trucks)

    # Cleanup
    new_routes = [r for r in routes if r]
    all_drones = []
    for dm_list in drone_missions:
        all_drones.extend(dm_list)

    sync_stats = {
        'total_wait_time': total_wait_time,
        'n_wait_missions': n_wait_missions,
        'n_total_missions': n_drones,
        'per_mission': per_mission_wait,
    }

    return new_routes, all_drones, total_saved, n_drones, drone_counts, sync_stats


def _resolve_drone_conflicts(routes, drone_missions, drone_counts,
                              drone_served, cust_to_truck, n_trucks):
    """Resolve conflicts where launch/recovery node is drone-served."""
    for _ in range(10):
        fixed_any = False
        for ti in range(n_trucks):
            for mi, mission in enumerate(drone_missions[ti]):
                i_cid, j_cid, k_cid = mission[0], mission[1], mission[2]
                fixed_this = False
                if i_cid > 0 and i_cid in drone_served:
                    drone_served.discard(i_cid)
                    for t2 in range(n_trucks):
                        drone_missions[t2] = [m for m in drone_missions[t2]
                                              if m[1] != i_cid]
                        old_count = drone_counts[t2]
                        new_count = len(drone_missions[t2])
                        if new_count < old_count:
                            drone_counts[t2] = new_count
                    routes[ti].append(i_cid)
                    cust_to_truck[i_cid] = (ti, len(routes[ti]) - 1)
                    fixed_this = True
                if k_cid > 0 and k_cid in drone_served:
                    drone_served.discard(k_cid)
                    for t2 in range(n_trucks):
                        drone_missions[t2] = [m for m in drone_missions[t2]
                                              if m[1] != k_cid]
                        old_count = drone_counts[t2]
                        new_count = len(drone_missions[t2])
                        if new_count < old_count:
                            drone_counts[t2] = new_count
                    routes[ti].append(k_cid)
                    cust_to_truck[k_cid] = (ti, len(routes[ti]) - 1)
                    fixed_this = True
                if fixed_this:
                    fixed_any = True
                    break
            if fixed_any:
                break
        if not fixed_any:
            break


# ── Shared utilities ────────────────────────────────────────────────────────

def _node_dist(i, j, dist_matrix, customers):
    """Distance between two node indices (0=depot)."""
    if i == 0 and j == 0:
        return 0.0
    if i == 0:
        return math.sqrt((DEPOT[0] - customers[j - 1]['x']) ** 2 +
                         (DEPOT[1] - customers[j - 1]['y']) ** 2)
    if j == 0:
        return math.sqrt((DEPOT[0] - customers[i - 1]['x']) ** 2 +
                         (DEPOT[1] - customers[i - 1]['y']) ** 2)
    return dist_matrix[i][j]


def _compute_truck_segment_time(route, i_pos, k_cid, customers, dist_matrix,
                                 truck_speed=35.0):
    """
    Compute actual truck travel time from launch position i_pos to recovery
    node k_cid. Includes travel + service + waiting at intermediate customers.
    """
    total_time = 0.0
    current_time = 0.0
    prev_cid = route[i_pos]

    if k_cid == 0:
        end_pos = len(route)
    else:
        try:
            end_pos = route.index(k_cid, i_pos + 1)
        except ValueError:
            d_ik = _node_dist(prev_cid, k_cid, dist_matrix, customers)
            return d_ik / truck_speed

    for pos in range(i_pos + 1, end_pos + 1):
        if pos < len(route):
            curr_cid = route[pos]
        else:
            curr_cid = 0

        d = _node_dist(prev_cid, curr_cid, dist_matrix, customers)
        current_time += d / truck_speed

        if curr_cid == 0:
            break

        c = customers[curr_cid - 1]
        if current_time < c['ready_time']:
            current_time = c['ready_time']
        current_time += c['service_time']

        if curr_cid == k_cid:
            current_time -= c['service_time']
            break

        prev_cid = curr_cid

    return current_time


def compute_sync_wait_for_mission(route, i_pos, k_cid, drone_mission_time,
                                   customers, dist_matrix, truck_speed=35.0):
    """
    Compute how long the truck must wait at recovery point k for a drone.

    Args:
        route: truck route
        i_pos: launch position in route
        k_cid: recovery customer ID (0=depot)
        drone_mission_time: drone flight + service time
        customers, dist_matrix: instance data
        truck_speed: km/h

    Returns:
        wait_time: time truck must wait at k for drone (0 if drone is faster)
    """
    truck_segment_time = _compute_truck_segment_time(
        route, i_pos, k_cid, customers, dist_matrix, truck_speed)

    service_k = (customers[k_cid - 1]['service_time']
                 if k_cid > 0 else 30.0)

    wait_needed = max(0.0, drone_mission_time - truck_segment_time - service_k)
    return wait_needed


# ── Comparison utility ──────────────────────────────────────────────────────

def compare_sync_vs_nosync(solution, instance):
    """
    Evaluate the same solution with and without sync constraints.

    Returns:
        dict with side-by-side comparison of sync vs no-sync evaluation.
    """
    result_sync = evaluate_sync_aware(solution, instance)
    result_nosync = evaluate_no_sync(solution, instance)

    return {
        'solution': solution,
        'with_sync': result_sync,
        'without_sync': result_nosync,
        'delta': {
            'cost': result_sync['cost'] - result_nosync['cost'],
            'tardiness': result_sync['tardiness'] - result_nosync['tardiness'],
            'makespan': result_sync['makespan'] - result_nosync['makespan'],
            'sync_wait': result_sync['sync_wait_time'],
            'feasible_sync': result_sync['feasible'],
            'feasible_nosync': result_nosync['feasible'],
        },
    }
