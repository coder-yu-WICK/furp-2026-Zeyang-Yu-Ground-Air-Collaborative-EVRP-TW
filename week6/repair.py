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

import random
import math
from utils.problem_model import TruckDroneSolution

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
        current_time = max(arrival, c['ready_time'])
        arrivals.append(current_time)
        current_time += c['service_time']
        departures.append(current_time)

        tard = max(0.0, arrival + c['service_time'] - c['due_time'])
        # More precise: tardiness is based on completion time exceeding due_time
        if current_time > c['due_time']:
            tard = current_time - c['due_time']
        else:
            tard = 0.0
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

def repair_tardiness(solution, instance, max_iter=2000, seed=42):
    """
    Targeted TW repair — reorder tardy routes by earliest due date.

    Strategy:
    1. For each route with tardiness > 0, sort by due_time (EDD)
    2. Also try a version sorted by TW midpoint
    3. Keep the better result
    4. Re-insert drones on repaired routes

    This is a one-shot deterministic repair, not iterative search.
    POMO's distance optimization is mostly preserved since we only
    reorder within routes that are already broken (tardy).
    """
    random.seed(seed)

    routes = [list(r) for r in solution.truck_routes]
    tard_before = solution.tardiness
    customers = instance['customers']

    if tard_before == 0 or not any(r for r in routes):
        return solution, {'tardiness_before': 0.0, 'tardiness_after': 0.0,
                         'tardiness_reduction': 0.0, 'moves_accepted': 0}

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

    # Re-insert drones
    try:
        from drone_post_processing import insert_cross_route_drones
        end_val = 4.0
        final_routes, new_drones, _, n_drones = insert_cross_route_drones(
            fixed_routes, instance, drone_endurance=end_val)
        new_sol = TruckDroneSolution(final_routes, new_drones, instance)
    except Exception:
        new_sol = TruckDroneSolution(fixed_routes, [], instance)

    return new_sol, {
        'tardiness_before': tard_before,
        'tardiness_after': new_sol.tardiness,
        'tardiness_reduction': tard_before - new_sol.tardiness,
        'moves_accepted': moves,
    }


# ── P3: Smarter Partial Repair ────────────────────────────────────────

def repair_tardiness_partial(solution, instance, seed=42):
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

    if tard_before == 0 or not any(r for r in routes):
        return solution, {
            'tardiness_before': 0.0, 'tardiness_after': 0.0,
            'tardiness_reduction': 0.0, 'segments_repaired': 0,
            'partial_success': True, 'fallback_count': 0,
        }

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

    # Step 6: Re-insert drones on repaired routes
    try:
        from drone_post_processing import insert_cross_route_drones
        final_routes, new_drones, _, n_drones = insert_cross_route_drones(
            fixed_routes, instance, drone_endurance=4.0)
        new_sol = TruckDroneSolution(final_routes, new_drones, instance)
    except Exception:
        new_sol = TruckDroneSolution(fixed_routes, [], instance)

    return new_sol, {
        'tardiness_before': tard_before,
        'tardiness_after': new_sol.tardiness,
        'tardiness_reduction': tard_before - new_sol.tardiness,
        'segments_repaired': total_segments_repaired,
        'partial_success': new_sol.tardiness == 0,
        'fallback_count': fallback_count,
    }
