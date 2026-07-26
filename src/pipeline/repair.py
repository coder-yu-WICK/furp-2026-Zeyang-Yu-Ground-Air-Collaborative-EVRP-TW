# -*- coding: utf-8 -*-
"""
IVND-Based Repair Operator — Week 6 Track B.

Takes a POMO + drone solution from Week 5 and applies focused local search
to repair residual time-window violations.

Strategy:
1. Strip drone missions → repair truck routes only
2. Evaluate using simple distance + tardiness cost
3. Accept moves that reduce total cost (distance + tardiness * penalty)
4. Re-insert drones on repaired routes
"""

import os, sys, random, math

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.problem_model import TruckDroneSolution

# Cost weights (weighted to prioritize tardiness reduction)
TRUCK_DIST_COST_RATE = 2.0
TARDINESS_COST_RATE = 5.0  # 2.5x vs distance — prioritize TW satisfaction


# ── Neighborhood Operators ────────────────────────────────────────────

def _relocate_one(routes):
    """Move one customer from position in one route to another position."""
    r = [list(rt) for rt in routes]
    valid = [i for i, rt in enumerate(r) if len(rt) > 0]
    if not valid:
        return r
    src = random.choice(valid)
    if len(r[src]) == 0:
        return r
    pos = random.randrange(len(r[src]))
    cust = r[src].pop(pos)
    # Insert into same or different route
    dst = random.choice(valid)  # can be same route
    ins = random.randrange(len(r[dst]) + 1) if r[dst] else 0
    r[dst].insert(ins, cust)
    return r


def _swap_two(routes):
    """Swap two customers within or between routes."""
    all_pos = [(ti, pi) for ti, rt in enumerate(routes)
               for pi in range(len(rt))]
    if len(all_pos) < 2:
        return [list(rt) for rt in routes]
    (t1, p1), (t2, p2) = random.sample(all_pos, 2)
    r = [list(rt) for rt in routes]
    r[t1][p1], r[t2][p2] = r[t2][p2], r[t1][p1]
    return r


def _two_opt_one_route(routes):
    """Reverse a segment within a single route."""
    valid = [i for i, rt in enumerate(routes) if len(rt) >= 3]
    if not valid:
        return [list(rt) for rt in routes]
    t = random.choice(valid)
    r = [list(rt) for rt in routes]
    i, j = sorted(random.sample(range(len(r[t])), 2))
    r[t][i:j+1] = list(reversed(r[t][i:j+1]))
    return r


def _two_opt_cross(routes):
    """Swap tails between two different routes (cross 2-opt)."""
    valid = [i for i, rt in enumerate(routes) if len(rt) >= 1]
    if len(valid) < 2:
        return [list(rt) for rt in routes]
    t1, t2 = random.sample(valid, 2)
    r = [list(rt) for rt in routes]
    if len(r[t1]) == 0 or len(r[t2]) == 0:
        return r
    pos1 = random.randrange(len(r[t1]))
    pos2 = random.randrange(len(r[t2]))
    # Swap tails
    tail1 = r[t1][pos1:]
    tail2 = r[t2][pos2:]
    r[t1][pos1:] = tail2
    r[t2][pos2:] = tail1
    return r


NEIGHBORHOODS = [
    _relocate_one,
    _swap_two,
    _two_opt_one_route,
    _two_opt_cross,
]


# ── Tardiness-Focused Evaluation ──────────────────────────────────────

def _route_tardiness(route, customers, depot, truck_speed=35.0):
    """Compute total tardiness for a single route."""
    if not route:
        return 0.0

    total_tard = 0.0
    current_time = 0.0
    prev_node = 0  # depot

    for cid in route:
        c = customers[cid - 1]
        # Travel time from previous node
        if prev_node == 0:
            dx = depot[0] - c['x']
            dy = depot[1] - c['y']
            travel = math.sqrt(dx*dx + dy*dy) / truck_speed
        else:
            prev_c = customers[prev_node - 1]
            dx = prev_c['x'] - c['x']
            dy = prev_c['y'] - c['y']
            travel = math.sqrt(dx*dx + dy*dy) / truck_speed

        current_time = max(current_time + travel, c['ready_time'])
        current_time += c['service_time']

        if current_time > c['due_time']:
            total_tard += current_time - c['due_time']

        prev_node = cid

    return total_tard


def _route_cost(routes, instance):
    """Compute truck-only cost: distance * rate + tardiness * penalty."""
    customers = instance['customers']
    depot = instance['depot']
    dist = instance['distance_matrix']
    truck_speed = 35.0

    total_dist = 0.0
    total_tard = 0.0
    current_time = 0.0

    for route in routes:
        prev = 0  # depot
        for cid in route:
            c = customers[cid - 1]
            if prev == 0:
                d = math.sqrt((depot[0] - c['x'])**2 + (depot[1] - c['y'])**2)
            else:
                d = dist[prev][cid]
            total_dist += d
            current_time += d / truck_speed
            current_time = max(current_time, c['ready_time'])
            current_time += c['service_time']
            if current_time > c['due_time']:
                total_tard += current_time - c['due_time']
            prev = cid

        # Return to depot
        if prev > 0:
            c = customers[prev - 1]
            total_dist += math.sqrt((depot[0] - c['x'])**2 + (depot[1] - c['y'])**2)

    return total_dist * TRUCK_DIST_COST_RATE + total_tard * TARDINESS_COST_RATE


# ── Route Simulation ─────────────────────────────────────────────────

def _simulate_route(route, customers, depot, truck_speed=35.0):
    """
    Forward-simulate a route to compute arrival times and per-customer tardiness.

    Returns:
        arrivals: list of arrival times at each position
        tardiness: list of tardiness values at each position
        departure_times: list of departure times at each position
        total_tardiness: float
        total_distance: float
        is_feasible: bool (True if no tardiness)
    """
    if not route:
        return [], [], [], 0.0, 0.0, True

    arrivals = []
    departures = []
    tardiness_vals = []
    total_tard = 0.0
    total_dist = 0.0
    current_time = 0.0
    prev_node = 0  # depot

    for cid in route:
        c = customers[cid - 1]

        # Travel from previous node
        if prev_node == 0:
            dx = depot[0] - c['x']
            dy = depot[1] - c['y']
            travel = math.sqrt(dx*dx + dy*dy) / truck_speed
        else:
            prev_c = customers[prev_node - 1]
            dx = prev_c['x'] - c['x']
            dy = prev_c['y'] - c['y']
            travel = math.sqrt(dx*dx + dy*dy) / truck_speed

        total_dist += travel * truck_speed  # convert back to distance
        arrival = current_time + travel
        start_time = max(arrival, c['ready_time'])
        arrivals.append(start_time)
        current_time = start_time + c['service_time']
        departures.append(current_time)

        # Tardiness = max(0, start_of_service - due_time)
        # Solomon TW standard: [ready_time, due_time] is for START of service
        tard = max(0.0, start_time - c['due_time'])
        tardiness_vals.append(tard)
        total_tard += tard

        prev_node = cid

    # Return to depot distance
    if prev_node > 0:
        c = customers[prev_node - 1]
        total_dist += math.sqrt((depot[0] - c['x'])**2 + (depot[1] - c['y'])**2)

    return arrivals, tardiness_vals, departures, total_tard, total_dist, total_tard == 0


def _find_tardy_segments(route, tardiness_vals):
    """
    Find contiguous segments of tardy customers.

    Returns list of (start_idx, end_idx) inclusive.
    """
    segments = []
    in_segment = False
    seg_start = 0

    for i, tard in enumerate(tardiness_vals):
        if tard > 0 and not in_segment:
            seg_start = i
            in_segment = True
        elif tard == 0 and in_segment:
            segments.append((seg_start, i - 1))
            in_segment = False

    if in_segment:
        segments.append((seg_start, len(tardiness_vals) - 1))

    return segments


def _extend_segment(start, end, route_len, before=1, after=1):
    """Extend a segment by context positions, clamped to route bounds."""
    return (max(0, start - before), min(route_len - 1, end + after))


def _node_dist_repair(i, j, customers, instance):
    """Distance between two node indices (0=depot), using customer coords."""
    if i == 0 and j == 0:
        return 0.0
    depot = instance['depot']
    if i == 0:
        cj = customers[j - 1]
        return math.sqrt((depot[0] - cj['x'])**2 + (depot[1] - cj['y'])**2)
    if j == 0:
        ci = customers[i - 1]
        return math.sqrt((depot[0] - ci['x'])**2 + (depot[1] - ci['y'])**2)
    ci = customers[i - 1]
    cj = customers[j - 1]
    return math.sqrt((ci['x'] - cj['x'])**2 + (ci['y'] - cj['y'])**2)


def _merge_segments(segments):
    """Merge overlapping segments."""
    if not segments:
        return []
    sorted_segs = sorted(segments)
    merged = [sorted_segs[0]]
    for seg in sorted_segs[1:]:
        if seg[0] <= merged[-1][1] + 1:  # adjacent or overlapping
            merged[-1] = (merged[-1][0], max(merged[-1][1], seg[1]))
        else:
            merged.append(seg)
    return merged


# ── Main Repair Loop ──────────────────────────────────────────────────

def repair_tardiness(solution, instance, max_iter=2000, seed=42,
                      max_drones_per_truck=2):
    """
    Targeted TW repair — reorder tardy routes by earliest due date.

    Strategy:
    1. For each route with tardiness > 0, sort by due_time (EDD)
    2. Also try a version sorted by TW midpoint
    3. Keep the better result
    4. Re-insert drones on repaired routes (with per-truck drone limits)

    This is a one-shot deterministic repair, not iterative search.
    POMO's distance optimization is mostly preserved since we only
    reorder within routes that are already broken (tardy).
    """
    random.seed(seed)

    routes = [list(r) for r in solution.truck_routes]
    tard_before = solution.tardiness
    customers = instance['customers']

    # RC2 Fix (Week 7): Skip repair when already feasible (tardiness ≈ 0).
    # For wide-TW instances, EDD reordering inflates cost without benefit.
    if tard_before <= 1e-6 or not any(r for r in routes):
        return solution, {'tardiness_before': tard_before, 'tardiness_after': tard_before,
                         'tardiness_reduction': 0.0, 'moves_accepted': 0,
                         'repair_skipped': True, 'reason': 'already_feasible'}

    # ── CRITICAL: Merge drone customers back into truck routes ──
    # The truck_routes already have drone-served customers removed.
    # If we repair without them, those customers are permanently lost
    # (they're not in any route, so re-insertion can't find them).
    # Solution: temporarily put drone customers back before repair,
    # then let drone re-insertion extract them again from the repaired routes.
    drone_customers = set()
    for mission in solution.drone_missions:
        _, j, _ = mission[0], mission[1], mission[2]
        drone_customers.add(j)

    if drone_customers:
        # Distribute drone customers back to routes (nearest-neighbor + capacity-aware)
        from src.config import TRUCK_CAPACITY
        routes_with_drones = [list(r) for r in routes]
        # Pre-compute route loads
        route_loads = [sum(customers[c-1]['demand'] for c in r) for r in routes_with_drones]
        for j_cid in sorted(drone_customers):
            cj = customers[j_cid - 1]
            demand_j = cj['demand']
            # Find best route that can accommodate this customer's demand
            best_route_idx = -1
            best_increase = float('inf')
            for ri, route in enumerate(routes_with_drones):
                if not route:
                    # Empty route always works
                    best_route_idx = ri
                    best_increase = 0.0
                    break
                # Capacity check: skip routes that would overflow
                if route_loads[ri] + demand_j > TRUCK_CAPACITY:
                    continue
                # Try inserting at each position, pick cheapest
                for pos in range(len(route) + 1):
                    prev_cid = route[pos - 1] if pos > 0 else 0
                    next_cid = route[pos] if pos < len(route) else 0
                    d_old = _node_dist_repair(prev_cid, next_cid, customers, instance)
                    d_new = (_node_dist_repair(prev_cid, j_cid, customers, instance) +
                             _node_dist_repair(j_cid, next_cid, customers, instance))
                    increase = d_new - d_old
                    if increase < best_increase:
                        best_increase = increase
                        best_route_idx = ri
            if best_route_idx >= 0:
                routes_with_drones[best_route_idx].append(j_cid)
                route_loads[best_route_idx] += demand_j
            else:
                # No route can take this customer — create a new route
                routes_with_drones.append([j_cid])
                route_loads.append(demand_j)
        routes = routes_with_drones

    # Identify tardy routes
    fixed_routes = []
    moves = 0
    for route in routes:
        if len(route) <= 2:
            fixed_routes.append(route)
            continue

        # Check if this route has tardiness
        # (simplified: check if TW midpoints are out of order)
        mids = [(cid, (customers[cid-1]['ready_time'] + customers[cid-1]['due_time']) / 2)
                for cid in route]
        due_times = [(cid, customers[cid-1]['due_time']) for cid in route]

        # Check if already sorted by due date
        sorted_by_due = [x[0] for x in sorted(due_times, key=lambda x: x[1])]
        sorted_by_mid = [x[0] for x in sorted(mids, key=lambda x: x[1])]

        if route == sorted_by_due or route == sorted_by_mid:
            fixed_routes.append(route)
            continue

        # Evaluate original vs EDD vs mid-sorted
        candidates = [route, sorted_by_due, sorted_by_mid]
        best = min(candidates, key=lambda r: _route_cost([r], instance))
        if best != route:
            moves += 1
        fixed_routes.append(best)

    # Re-insert drones (with max_drones_per_truck support)
    try:
        from src.pipeline.drone import insert_cross_route_drones
        end_val = 4.0
        final_routes, new_drones, _, n_drones, drone_counts = insert_cross_route_drones(
            fixed_routes, instance, drone_endurance=end_val,
            max_drones_per_truck=max_drones_per_truck)
        new_sol = TruckDroneSolution(final_routes, new_drones, instance,
                                      max_drones_per_truck=max_drones_per_truck)
    except Exception:
        new_sol = TruckDroneSolution(fixed_routes, [], instance,
                                      max_drones_per_truck=max_drones_per_truck)

    return new_sol, {
        'tardiness_before': tard_before,
        'tardiness_after': new_sol.tardiness,
        'tardiness_reduction': tard_before - new_sol.tardiness,
        'moves_accepted': moves,
    }


# ── P3: Smarter Partial Repair ────────────────────────────────────────

def repair_tardiness_partial(solution, instance, seed=42,
                             max_drones_per_truck=2):
    """
    P3: Targeted partial repair — only reorder the tardy segment of each route.

    Key insight: Full-route EDD destroys POMO's distance optimization on parts
    of the route that were already TW-feasible. This function:

    1. Forward-simulates each route to compute exact arrival times
    2. Identifies the minimal contiguous segment(s) containing tardy customers
    3. Applies EDD only to those segments (with 1-position context on each side)
    4. Preserves the rest of POMO's route ordering

    If partial repair fails to eliminate tardiness, falls back to full EDD.

    Returns:
        (solution, stats_dict)
    """
    random.seed(seed)

    routes = [list(r) for r in solution.truck_routes]
    tard_before = solution.tardiness
    customers = instance['customers']
    depot = instance['depot']

    if tard_before <= 1e-6 or not any(r for r in routes):
        return solution, {
            'tardiness_before': tard_before, 'tardiness_after': tard_before,
            'tardiness_reduction': 0.0, 'segments_repaired': 0,
            'partial_success': True, 'fallback_count': 0,
            'repair_skipped': True, 'reason': 'already_feasible',
        }

    # ── CRITICAL: Merge drone customers back into truck routes before repair ──
    drone_customers = set()
    for mission in solution.drone_missions:
        _, j, _ = mission[0], mission[1], mission[2]
        drone_customers.add(j)

    if drone_customers:
        from src.config import TRUCK_CAPACITY
        routes_with_drones = [list(r) for r in routes]
        route_loads = [sum(customers[c-1]['demand'] for c in r) for r in routes_with_drones]
        for j_cid in sorted(drone_customers):
            cj = customers[j_cid - 1]
            demand_j = cj['demand']
            best_route_idx = -1
            best_increase = float('inf')
            for ri, route in enumerate(routes_with_drones):
                if not route:
                    best_route_idx = ri
                    best_increase = 0.0
                    break
                if route_loads[ri] + demand_j > TRUCK_CAPACITY:
                    continue
                for pos in range(len(route) + 1):
                    prev_cid = route[pos - 1] if pos > 0 else 0
                    next_cid = route[pos] if pos < len(route) else 0
                    d_old = _node_dist_repair(prev_cid, next_cid, customers, instance)
                    d_new = (_node_dist_repair(prev_cid, j_cid, customers, instance) +
                             _node_dist_repair(j_cid, next_cid, customers, instance))
                    increase = d_new - d_old
                    if increase < best_increase:
                        best_increase = increase
                        best_route_idx = ri
            if best_route_idx >= 0:
                routes_with_drones[best_route_idx].append(j_cid)
                route_loads[best_route_idx] += demand_j
            else:
                routes_with_drones.append([j_cid])
                route_loads.append(demand_j)
        routes = routes_with_drones

    fixed_routes = []
    total_segments_repaired = 0
    fallback_count = 0

    for route in routes:
        if len(route) <= 2:
            fixed_routes.append(route)
            continue

        # Step 1: Forward-simulate to find exact tardy positions
        arrivals, tard_vals, departures, route_tard, route_dist, is_feasible = \
            _simulate_route(route, customers, depot)

        if is_feasible:
            fixed_routes.append(route)
            continue

        # Step 2: Find tardy segments
        raw_segments = _find_tardy_segments(route, tard_vals)

        if not raw_segments:
            fixed_routes.append(route)
            continue

        # Step 3: Extend segments with context and merge
        extended = [_extend_segment(s, e, len(route), before=1, after=1)
                    for s, e in raw_segments]
        merged = _merge_segments(extended)

        # Step 4: Apply partial EDD repair
        partial_route = list(route)  # copy
        for seg_start, seg_end in merged:
            segment = partial_route[seg_start:seg_end + 1]

            # Generate candidates for this segment
            due_times = [(cid, customers[cid-1]['due_time']) for cid in segment]
            mids = [(cid, (customers[cid-1]['ready_time'] + customers[cid-1]['due_time']) / 2)
                    for cid in segment]

            sorted_by_due = [x[0] for x in sorted(due_times, key=lambda x: x[1])]
            sorted_by_mid = [x[0] for x in sorted(mids, key=lambda x: x[1])]

            # Evaluate original segment vs EDD segment vs mid-sorted segment
            candidates = []
            for cand_seg in [segment, sorted_by_due, sorted_by_mid]:
                if cand_seg == segment:
                    candidates.append(segment)
                else:
                    candidates.append(cand_seg)

            # Pick best for this segment
            best_seg = segment  # default: keep original
            best_cost = float('inf')
            for cand_seg in candidates:
                test_route = list(partial_route)
                test_route[seg_start:seg_end + 1] = cand_seg
                cost = _route_cost([test_route], instance)
                if cost < best_cost:
                    best_cost = cost
                    best_seg = cand_seg

            if best_seg != segment:
                partial_route[seg_start:seg_end + 1] = best_seg
                total_segments_repaired += 1

        # Step 5: Verify partial repair worked
        _, _, _, partial_tard, _, partial_feasible = \
            _simulate_route(partial_route, customers, depot)

        if not partial_feasible:
            # Fallback: full EDD on this route
            due_times = [(cid, customers[cid-1]['due_time']) for cid in partial_route]
            mids = [(cid, (customers[cid-1]['ready_time'] + customers[cid-1]['due_time']) / 2)
                    for cid in partial_route]
            sorted_by_due = [x[0] for x in sorted(due_times, key=lambda x: x[1])]
            sorted_by_mid = [x[0] for x in sorted(mids, key=lambda x: x[1])]

            # Select candidate with ZERO tardiness first, then minimum cost.
            # Previous code used cost-minimization which could accept routes
            # with residual tardiness if they had lower distance (Bug: cost
            # weighting allowed tardy routes to "win" over zero-tardiness ones).
            best_candidate = None
            best_tardiness = float('inf')
            best_cost = float('inf')
            for cand in [partial_route, sorted_by_due, sorted_by_mid]:
                _, _, _, cand_tard, _, _ = _simulate_route(cand, customers, depot)
                cand_cost = _route_cost([cand], instance)
                if cand_tard < best_tardiness or (cand_tard == best_tardiness and cand_cost < best_cost):
                    best_tardiness = cand_tard
                    best_cost = cand_cost
                    best_candidate = cand
            partial_route = best_candidate
            fallback_count += 1

        fixed_routes.append(partial_route)

    # Step 6: Re-insert drones on repaired routes (with max_drones_per_truck)
    try:
        from src.pipeline.drone import insert_cross_route_drones
        final_routes, new_drones, _, n_drones, drone_counts = insert_cross_route_drones(
            fixed_routes, instance, drone_endurance=4.0,
            max_drones_per_truck=max_drones_per_truck)
        new_sol = TruckDroneSolution(final_routes, new_drones, instance,
                                      max_drones_per_truck=max_drones_per_truck)
    except Exception:
        new_sol = TruckDroneSolution(fixed_routes, [], instance,
                                      max_drones_per_truck=max_drones_per_truck)

    return new_sol, {
        'tardiness_before': tard_before,
        'tardiness_after': new_sol.tardiness,
        'tardiness_reduction': tard_before - new_sol.tardiness,
        'segments_repaired': total_segments_repaired,
        'partial_success': new_sol.tardiness == 0,
        'fallback_count': fallback_count,
    }


# ── Inter-Route Repair ─────────────────────────────────────────────────

def _eval_all_routes(routes, customers, depot):
    """Simulate all routes, return total tardiness and per-route details."""
    total_tard = 0.0
    route_details = []
    for route in routes:
        arrivals, tard_vals, deps, tard, dist, feas = _simulate_route(
            route, customers, depot)
        route_details.append({
            'route': route, 'arrivals': arrivals, 'tardiness_vals': tard_vals,
            'departures': deps, 'total_tardiness': tard, 'total_distance': dist,
            'is_feasible': feas,
        })
        total_tard += tard
    return total_tard, route_details


def _try_relocate(tardy_cid, src_route_idx, src_pos, dst_route_idx, dst_pos,
                  routes, customers, depot):
    """
    Try relocating one customer from src to dst at specific positions.
    Returns (new_routes, total_tardiness) or (None, inf) if invalid.
    """
    new_routes = [list(r) for r in routes]
    # Remove from source
    if src_route_idx < len(new_routes) and src_pos < len(new_routes[src_route_idx]):
        removed = new_routes[src_route_idx].pop(src_pos)
    else:
        return None, float('inf')
    # Insert at destination
    if dst_pos <= len(new_routes[dst_route_idx]):
        new_routes[dst_route_idx].insert(dst_pos, tardy_cid)
    else:
        new_routes[dst_route_idx].append(tardy_cid)

    # Quick capacity check
    from src.config import TRUCK_CAPACITY
    for route in new_routes:
        load = sum(customers[c-1]['demand'] for c in route)
        if load > TRUCK_CAPACITY:
            return None, float('inf')

    # Apply EDD reorder to affected routes
    new_routes[src_route_idx] = sorted(
        new_routes[src_route_idx],
        key=lambda cid: customers[cid-1]['due_time'])
    new_routes[dst_route_idx] = sorted(
        new_routes[dst_route_idx],
        key=lambda cid: customers[cid-1]['due_time'])

    # Evaluate
    total_tard, _ = _eval_all_routes(new_routes, customers, depot)
    return new_routes, total_tard


def _try_swap(cid1, route1_idx, pos1, cid2, route2_idx, pos2,
              routes, customers, depot):
    """Try swapping two customers between routes."""
    new_routes = [list(r) for r in routes]
    if (route1_idx < len(new_routes) and pos1 < len(new_routes[route1_idx]) and
        route2_idx < len(new_routes) and pos2 < len(new_routes[route2_idx])):
        new_routes[route1_idx][pos1], new_routes[route2_idx][pos2] = \
            new_routes[route2_idx][pos2], new_routes[route1_idx][pos1]
    else:
        return None, float('inf')

    # EDD reorder affected routes
    new_routes[route1_idx] = sorted(
        new_routes[route1_idx],
        key=lambda cid: customers[cid-1]['due_time'])
    if route2_idx != route1_idx:
        new_routes[route2_idx] = sorted(
            new_routes[route2_idx],
            key=lambda cid: customers[cid-1]['due_time'])

    total_tard, _ = _eval_all_routes(new_routes, customers, depot)
    return new_routes, total_tard


def repair_inter_route(solution, instance, max_iter=200, seed=42,
                       max_drones_per_truck=0):
    """
    Inter-route repair: move tardy customers between routes using
    best-improvement relocate + swap, with EDD reordering after each move.

    This addresses the limitation of intra-route-only EDD repair: when a
    route contains customers whose time windows are collectively infeasible
    (even in EDD order), the only fix is to move some customers to other
    routes with more slack.

    Strategy:
    1. Simulate all routes, identify tardy customers
    2. For each tardy customer, evaluate ALL possible relocations
    3. Accept the move that gives the largest tardiness reduction
    4. Also try swaps between tardy and non-tardy routes
    5. Apply EDD reordering after each accepted move
    6. Repeat until no improvement or max_iter

    Args:
        solution: TruckDroneSolution (truck-only, no drones)
        instance: problem instance dict
        max_iter: max iterations without improvement
        seed: random seed

    Returns:
        (repaired_solution, stats_dict)
    """
    random.seed(seed)
    customers = instance['customers']
    depot = instance['depot']

    routes = [list(r) for r in solution.truck_routes]
    tard_before = solution.tardiness

    if tard_before <= 1e-6 or len(routes) <= 1:
        return solution, {
            'tardiness_before': tard_before,
            'tardiness_after': tard_before,
            'moves_accepted': 0,
            'skipped': True, 'reason': 'already_feasible_or_single_route',
        }

    # Phase 1: Best-improvement relocate for each tardy customer
    moves_accepted = 0
    improved = True
    iteration = 0

    while improved and iteration < max_iter:
        improved = False
        iteration += 1

        # Simulate all routes
        total_tard, details = _eval_all_routes(routes, customers, depot)
        if total_tard <= 1e-6:
            break

        # Find all tardy customers (in tardy routes)
        tardy_customers = []
        for ri, detail in enumerate(details):
            if detail['total_tardiness'] > 0:
                for pos, (cid, tard) in enumerate(
                    zip(detail['route'], detail['tardiness_vals'])):
                    if tard > 0:
                        tardy_customers.append((cid, ri, pos, tard))

        if not tardy_customers:
            break

        # Sort by tardiness (most tardy first) — only try top-N to limit compute
        tardy_customers.sort(key=lambda x: x[3], reverse=True)
        max_to_try = min(len(tardy_customers), 15)

        best_improvement = 0
        best_move = None  # ('relocate', src_ri, src_pos, dst_ri, dst_pos) or ('swap', ...)

        for t_idx in range(max_to_try):
            cid, src_ri, src_pos, tard = tardy_customers[t_idx]

            # Try relocating to every position in every OTHER route
            for dst_ri in range(len(routes)):
                if dst_ri == src_ri:
                    continue
                for dst_pos in range(len(routes[dst_ri]) + 1):
                    new_routes, new_tard = _try_relocate(
                        cid, src_ri, src_pos, dst_ri, dst_pos,
                        routes, customers, depot)
                    if new_routes is None:
                        continue
                    improvement = total_tard - new_tard
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = ('relocate', src_ri, src_pos, dst_ri, dst_pos, cid)

            # Try swapping with customers in other routes
            for dst_ri in range(len(routes)):
                if dst_ri == src_ri:
                    continue
                for dst_pos in range(len(routes[dst_ri])):
                    other_cid = routes[dst_ri][dst_pos]
                    new_routes, new_tard = _try_swap(
                        cid, src_ri, src_pos, other_cid, dst_ri, dst_pos,
                        routes, customers, depot)
                    if new_routes is None:
                        continue
                    improvement = total_tard - new_tard
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = ('swap', src_ri, src_pos, dst_ri, dst_pos, cid, other_cid)

        # Accept best move
        if best_move is not None and best_improvement > 1e-6:
            if best_move[0] == 'relocate':
                _, src_ri, src_pos, dst_ri, dst_pos, cid = best_move
                new_routes, _ = _try_relocate(
                    cid, src_ri, src_pos, dst_ri, dst_pos,
                    routes, customers, depot)
                if new_routes is not None:
                    routes = new_routes
                    moves_accepted += 1
                    improved = True
            elif best_move[0] == 'swap':
                _, src_ri, src_pos, dst_ri, dst_pos, cid1, cid2 = best_move
                new_routes, _ = _try_swap(
                    cid1, src_ri, src_pos, cid2, dst_ri, dst_pos,
                    routes, customers, depot)
                if new_routes is not None:
                    routes = new_routes
                    moves_accepted += 1
                    improved = True

        # Clean up empty routes
        routes = [r for r in routes if r]

    # Build result
    final_sol = TruckDroneSolution(routes, [], instance,
                                    max_drones_per_truck=max_drones_per_truck)

    return final_sol, {
        'tardiness_before': tard_before,
        'tardiness_after': final_sol.tardiness,
        'tardiness_reduction': tard_before - final_sol.tardiness,
        'moves_accepted': moves_accepted,
        'iterations': iteration,
        'inter_route_success': final_sol.tardiness == 0,
    }


def repair_hybrid(solution, instance, max_iter=200, seed=42,
                  max_drones_per_truck=0):
    """
    Hybrid repair: intra-route EDD (partial) + inter-route relocate/swap.

    This is the recommended repair for tight-TW instances where intra-route
    EDD alone is insufficient. It first applies targeted partial EDD within
    each route, then uses inter-route operators for any remaining tardiness.
    """
    # Step 1: Intra-route (partial EDD)
    sol1, stats1 = repair_tardiness_partial(
        solution, instance, seed=seed,
        max_drones_per_truck=max_drones_per_truck)

    if sol1.tardiness <= 1e-6:
        stats1['inter_route_used'] = False
        return sol1, stats1

    # Step 2: Inter-route
    sol2, stats2 = repair_inter_route(
        sol1, instance, max_iter=max_iter, seed=seed + 1,
        max_drones_per_truck=max_drones_per_truck)

    return sol2, {
        'tardiness_before': solution.tardiness,
        'tardiness_after': sol2.tardiness,
        'tardiness_reduction': solution.tardiness - sol2.tardiness,
        'intra_route_stats': stats1,
        'inter_route_stats': stats2,
        'inter_route_used': True,
    }


# ── Capacity Repair ───────────────────────────────────────────────────

def repair_capacity(solution, instance, max_iter=200, seed=42):
    """
    Repair capacity violations by moving customers from overloaded routes
    to under-loaded routes. Uses best-improvement relocate.

    This is needed for 200c instances where POMO clustering sometimes
    creates routes exceeding TRUCK_CAPACITY.

    Strategy:
    1. Identify overloaded routes (load > TRUCK_CAPACITY)
    2. For each customer in overloaded routes, try relocating to all
       other routes that have spare capacity
    3. Accept the move that most reduces capacity violation
    4. Apply EDD reordering after each move
    5. Repeat until all routes within capacity or max_iter
    """
    from src.config import TRUCK_CAPACITY

    random.seed(seed)
    customers = instance['customers']
    depot = instance.get('depot', (8.0, 8.0))
    if isinstance(depot, list):
        depot = tuple(depot)

    routes = [list(r) for r in solution.truck_routes]
    cap_before = sum(
        max(0, sum(customers[c-1]['demand'] for c in r) - TRUCK_CAPACITY)
        for r in routes)

    moves_accepted = 0
    iteration = 0

    for iteration in range(max_iter):
        # Compute loads
        loads = [sum(customers[c-1]['demand'] for c in r) for r in routes]

        # Find overloaded and under-loaded routes
        overloaded = [(ri, loads[ri]) for ri in range(len(routes))
                      if loads[ri] > TRUCK_CAPACITY]
        if not overloaded:
            break  # All within capacity

        underloaded = [(ri, loads[ri]) for ri in range(len(routes))
                       if loads[ri] < TRUCK_CAPACITY]
        if not underloaded:
            # All routes full — create new route
            routes.append([])
            underloaded = [(len(routes)-1, 0)]

        best_improvement = 0.0
        best_move = None

        for src_ri, src_load in overloaded:
            route = routes[src_ri]
            for src_pos, cid in enumerate(route):
                demand = customers[cid-1]['demand']

                for dst_ri, dst_load in underloaded:
                    if src_ri == dst_ri:
                        continue
                    if dst_load + demand > TRUCK_CAPACITY:
                        continue

                    # Try inserting at each position in dst route
                    dst_route = routes[dst_ri]
                    for dst_pos in range(len(dst_route) + 1):
                        # Compute distance change
                        prev_src = route[src_pos-1] if src_pos > 0 else 0
                        next_src = route[src_pos+1] if src_pos < len(route)-1 else 0

                        prev_dst = dst_route[dst_pos-1] if dst_pos > 0 else 0
                        next_dst = dst_route[dst_pos] if dst_pos < len(dst_route) else 0

                        # Distance change
                        dist = instance['distance_matrix']
                        d_remove = dist[prev_src][next_src if next_src > 0 else 0] - \
                                   (dist[prev_src][cid] + dist[cid][next_src if next_src > 0 else 0])
                        d_add = (dist[prev_dst][cid] + dist[cid][next_dst if next_dst > 0 else 0]) - \
                                dist[prev_dst][next_dst if next_dst > 0 else 0]

                        # Improvement = capacity violation reduction (primary) + distance (secondary)
                        cap_reduction = min(src_load - TRUCK_CAPACITY, demand) * 1000.0
                        improvement = cap_reduction + (d_remove - d_add) * 0.01

                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_move = (src_ri, src_pos, dst_ri, dst_pos)

        if best_move is None:
            break

        src_ri, src_pos, dst_ri, dst_pos = best_move
        cid = routes[src_ri].pop(src_pos)
        routes[dst_ri].insert(dst_pos, cid)
        moves_accepted += 1

        # EDD reorder affected routes
        routes[src_ri] = sorted(routes[src_ri],
                                key=lambda cid: customers[cid-1]['due_time'])
        routes[dst_ri] = sorted(routes[dst_ri],
                                key=lambda cid: customers[cid-1]['due_time'])

    # Build repaired solution
    new_sol = TruckDroneSolution(routes, [], instance)
    cap_after = sum(
        max(0, sum(customers[c-1]['demand'] for c in r) - TRUCK_CAPACITY)
        for r in routes)

    return new_sol, {
        'capacity_violation_before': cap_before,
        'capacity_violation_after': cap_after,
        'moves_accepted': moves_accepted,
        'iterations': iteration + 1,
        'capacity_fixed': cap_after < 0.01,
    }
