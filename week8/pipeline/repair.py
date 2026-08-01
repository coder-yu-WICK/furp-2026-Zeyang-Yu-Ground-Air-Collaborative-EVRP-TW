# -*- coding: utf-8 -*-
"""
EDD Repair Operators — Week 8 (truck-only, no drones).

Core contribution: Earliest Due Date (EDD) reordering for time-window repair.

Key insight (Jackson 1955): EDD ordering is provably optimal for minimizing
maximum lateness (Lmax) on a single machine. We apply this principle to each
truck route independently — reorder customers by due_date ascending to
eliminate time-window violations.

Truck-only versions (no drone merge/re-insert cycle):
  - repair_tardiness_truck: Full-route EDD reordering
  - repair_tardiness_partial_truck: Segment-level EDD (preserves POMO ordering
    on non-tardy segments, only repairs the problematic ones)
  - repair_inter_route: Moves tardy customers between routes (already truck-only)
  - repair_capacity: Capacity balancing (already truck-only)
"""
import os, sys, random, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from week8.core.problem_model import TruckSolution

# Cost weights for repair evaluation
TRUCK_DIST_COST_RATE = 2.0
TARDINESS_COST_RATE = 5.0  # 2.5x priority on TW satisfaction


# ═══════════════════════════════════════════════════════════════════════════
# Route Simulation Utilities
# ═══════════════════════════════════════════════════════════════════════════

def _simulate_route(route, customers, depot, truck_speed=35.0):
    """Simulate a truck route. Returns (arrivals, tardiness_vals, departures,
    total_tard, total_dist, is_feasible)."""
    arrivals = []
    tardiness_vals = []
    departures = []
    total_tard = 0.0
    total_dist = 0.0
    feasible = True

    prev = 0  # depot
    current_time = 0.0

    for cust_idx in route:
        c = customers[cust_idx - 1]
        # Distance from previous node
        if prev == 0:
            d = math.sqrt((depot[0] - c['x'])**2 + (depot[1] - c['y'])**2)
        else:
            prev_c = customers[prev - 1]
            d = math.sqrt((prev_c['x'] - c['x'])**2 + (prev_c['y'] - c['y'])**2)
        total_dist += d
        current_time += d / truck_speed

        # TW check
        if current_time < c['ready_time']:
            current_time = c['ready_time']
        arrivals.append(current_time)

        tard = max(0.0, current_time - c['due_time'])
        tardiness_vals.append(tard)
        total_tard += tard
        if tard > 0:
            feasible = False

        current_time += c['service_time']
        departures.append(current_time)
        prev = cust_idx

    # Return to depot
    if prev == 0:
        d = 0.0
    else:
        prev_c = customers[prev - 1]
        d = math.sqrt((prev_c['x'] - depot[0])**2 + (prev_c['y'] - depot[1])**2)
    total_dist += d

    return arrivals, tardiness_vals, departures, total_tard, total_dist, feasible


def _route_cost(routes, instance):
    """Compute total cost (distance + tardiness penalty) for routes."""
    customers = instance['customers']
    depot = instance['depot']
    total = 0.0
    for route in routes:
        if not route:
            continue
        _, _, _, tard, dist, _ = _simulate_route(route, customers, depot)
        total += dist * TRUCK_DIST_COST_RATE + tard * TARDINESS_COST_RATE
    return total


# ═══════════════════════════════════════════════════════════════════════════
# Tardy Segment Detection
# ═══════════════════════════════════════════════════════════════════════════

def _find_tardy_segments(route, tardiness_vals):
    """Find contiguous segments in a route that have tardiness > 0."""
    if not route or len(route) != len(tardiness_vals):
        return []
    segments = []
    start = None
    for i, tard in enumerate(tardiness_vals):
        if tard > 0 and start is None:
            start = i
        elif tard <= 0 and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(route)))
    return segments


def _merge_segments(segments):
    """Merge overlapping or adjacent segments."""
    if not segments:
        return []
    merged = [list(segments[0])]
    for seg in segments[1:]:
        if seg[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], seg[1])
        else:
            merged.append(list(seg))
    return [(s, e) for s, e in merged]


# ═══════════════════════════════════════════════════════════════════════════
# Repair Operator 1: Full-Route EDD (truck-only)
# ═══════════════════════════════════════════════════════════════════════════

def repair_tardiness_truck(solution, instance, max_iter=500, seed=42):
    """
    Full-route EDD repair — truck-only.

    For each truck route, reorder ALL customers by due_date ascending.
    This may increase distance but guarantees minimum Lmax per route.

    No drone merge/re-insert cycle — operates directly on truck_routes.

    Args:
        solution: TruckSolution
        instance: problem instance dict
        max_iter: max IVND iterations for inter-route improvement
        seed: random seed

    Returns:
        (TruckSolution, stats_dict)
    """
    random.seed(seed)
    customers = instance['customers']
    depot = instance['depot']

    routes = [list(rt) for rt in solution.truck_routes]
    tardiness_before = solution.tardiness

    # Phase 1: Full EDD reordering per route
    for i, route in enumerate(routes):
        if not route:
            continue
        # Sort by due_date ascending (EDD rule)
        routes[i] = sorted(route, key=lambda cid: customers[cid - 1]['due_time'])

    # Phase 2: Inter-route improvement (move tardy customers between routes)
    best_routes = [list(rt) for rt in routes]
    best_cost = _route_cost(best_routes, instance)
    moves_accepted = 0

    for iteration in range(max_iter):
        improved = False

        # Try relocate: move a customer from one route to another
        for _ in range(10):
            valid_routes = [i for i, rt in enumerate(best_routes) if len(rt) > 0]
            if len(valid_routes) < 1:
                continue
            src = random.choice(valid_routes)
            if len(best_routes[src]) == 0:
                continue
            pos = random.randrange(len(best_routes[src]))
            cust = best_routes[src][pos]

            # Try inserting into each route (including same)
            for dst in valid_routes:
                test_routes = [list(rt) for rt in best_routes]
                test_routes[src].pop(pos)
                ins = random.randrange(len(test_routes[dst]) + 1) if test_routes[dst] else 0
                test_routes[dst].insert(ins, cust)

                # After move, re-apply EDD to affected routes
                for ri in [src, dst]:
                    if test_routes[ri]:
                        test_routes[ri] = sorted(test_routes[ri],
                                                 key=lambda cid: customers[cid - 1]['due_time'])

                new_cost = _route_cost(test_routes, instance)
                if new_cost < best_cost - 0.01:
                    best_routes = test_routes
                    best_cost = new_cost
                    moves_accepted += 1
                    improved = True
                    break
            if improved:
                break

        if not improved:
            break

    repaired = TruckSolution(best_routes, instance)

    stats = {
        'tardiness_before': tardiness_before,
        'tardiness_after': repaired.tardiness,
        'tardiness_reduction': tardiness_before - repaired.tardiness,
        'moves_accepted': moves_accepted,
        'repair_type': 'full_edd_truck',
    }
    return repaired, stats


# ═══════════════════════════════════════════════════════════════════════════
# Repair Operator 2: Partial (Segment-Level) EDD (truck-only)
# ═══════════════════════════════════════════════════════════════════════════

def repair_tardiness_partial_truck(solution, instance, seed=42):
    """
    Partial (segment-level) EDD repair — truck-only.

    Only repairs route segments that have tardiness > 0, preserving
    POMO's distance-optimized ordering on non-tardy segments.

    Key insight: Full-route EDD destroys POMO's distance optimization on
    parts of the route that are already TW-feasible. Partial EDD only
    reorders the problematic segments, keeping the rest intact.

    For ≤50c instances, partial EDD generally produces lower-cost solutions.
    For 100c+, full EDD is more effective (repair phase transition at ~75c).

    No drone merge/re-insert cycle.

    Args:
        solution: TruckSolution
        instance: problem instance dict
        seed: random seed

    Returns:
        (TruckSolution, stats_dict)
    """
    random.seed(seed)
    customers = instance['customers']
    depot = instance['depot']

    routes = [list(rt) for rt in solution.truck_routes]
    tardiness_before = solution.tardiness

    segments_repaired = 0
    fallback_count = 0

    for i, route in enumerate(routes):
        if not route:
            continue

        # Find tardy segments
        _, tard_vals, _, _, _, _ = _simulate_route(route, customers, depot)
        segments = _find_tardy_segments(route, tard_vals)

        if not segments:
            continue

        # Merge adjacent segments
        segments = _merge_segments(segments)

        for start, end in segments:
            # Extract tardy segment + context (1 customer before and after if available)
            ctx_start = max(0, start - 1)
            ctx_end = min(len(route), end + 1)

            segment = route[start:end]
            context_before = route[ctx_start:start] if ctx_start < start else []
            context_after = route[end:ctx_end] if end < ctx_end else []

            # EDD reorder the tardy segment
            segment_edd = sorted(segment, key=lambda cid: customers[cid - 1]['due_time'])

            # Rebuild route: before + EDD segment + after
            new_route = route[:start] + segment_edd + route[end:]
            routes[i] = new_route
            segments_repaired += 1

            # Verify: if still tardy, fall back to full EDD for this route
            _, new_tard_vals, _, _, _, _ = _simulate_route(new_route, customers, depot)
            if any(t > 0 for t in new_tard_vals):
                routes[i] = sorted(route, key=lambda cid: customers[cid - 1]['due_time'])
                fallback_count += 1

    repaired = TruckSolution(routes, instance)

    stats = {
        'tardiness_before': tardiness_before,
        'tardiness_after': repaired.tardiness,
        'tardiness_reduction': tardiness_before - repaired.tardiness,
        'segments_repaired': segments_repaired,
        'fallback_count': fallback_count,
        'partial_success': repaired.tardiness <= 1e-6,
        'repair_type': 'partial_edd_truck',
    }
    return repaired, stats


# ═══════════════════════════════════════════════════════════════════════════
# Repair Operator 3: Inter-Route Relocate (already truck-only)
# ═══════════════════════════════════════════════════════════════════════════

def repair_inter_route(solution, instance, max_iter=200, seed=42):
    """
    Inter-route customer relocation — moves tardy customers between routes.

    Truck-only. No drone operations.
    """
    random.seed(seed)
    customers = instance['customers']
    depot = instance['depot']

    routes = [list(rt) for rt in solution.truck_routes]
    tardiness_before = solution.tardiness
    best_cost = _route_cost(routes, instance)
    moves = 0

    for _ in range(max_iter):
        improved = False
        valid = [i for i, rt in enumerate(routes) if len(rt) > 0]
        if len(valid) < 2:
            break

        for _ in range(20):
            src = random.choice(valid)
            if len(routes[src]) == 0:
                continue
            pos = random.randrange(len(routes[src]))
            cust = routes[src][pos]

            for dst in valid:
                if dst == src and len(routes[dst]) <= 1:
                    continue
                test = [list(rt) for rt in routes]
                test[src].pop(pos)
                ins_pos = random.randrange(len(test[dst]) + 1) if test[dst] else 0
                test[dst].insert(ins_pos, cust)

                new_cost = _route_cost(test, instance)
                if new_cost < best_cost - 0.01:
                    routes = test
                    best_cost = new_cost
                    moves += 1
                    improved = True
                    break
            if improved:
                break

        if not improved:
            break

    repaired = TruckSolution(routes, instance)
    return repaired, {
        'tardiness_before': tardiness_before,
        'tardiness_after': repaired.tardiness,
        'tardiness_reduction': tardiness_before - repaired.tardiness,
        'moves': moves,
        'repair_type': 'inter_route',
    }


# ═══════════════════════════════════════════════════════════════════════════
# Repair Operator 4: Capacity Balancing (already truck-only)
# ═══════════════════════════════════════════════════════════════════════════

def repair_capacity(solution, instance, max_iter=200, seed=42):
    """
    Capacity balancing — moves customers from overloaded to under-loaded routes.

    Truck-only. No drone operations.
    """
    random.seed(seed)
    from week8.config import TRUCK_CAPACITY

    routes = [list(rt) for rt in solution.truck_routes]
    customers = instance['customers']

    for _ in range(max_iter):
        # Find overloaded and underloaded routes
        overloaded = []
        underloaded = []
        for i, route in enumerate(routes):
            load = sum(customers[c - 1]['demand'] for c in route)
            if load > TRUCK_CAPACITY:
                overloaded.append((i, load - TRUCK_CAPACITY))
            else:
                underloaded.append((i, TRUCK_CAPACITY - load))

        if not overloaded:
            break

        improved = False
        for src_i, excess in overloaded:
            for dst_i, slack in underloaded:
                if src_i == dst_i:
                    continue
                # Find a customer in src that fits in dst
                for pos, cust in enumerate(routes[src_i]):
                    demand = customers[cust - 1]['demand']
                    if demand <= slack:
                        # Move it
                        routes[dst_i].append(routes[src_i].pop(pos))
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

        if not improved:
            break

    repaired = TruckSolution(routes, instance)
    return repaired, {
        'tardiness_before': solution.tardiness,
        'tardiness_after': repaired.tardiness,
        'tardiness_reduction': solution.tardiness - repaired.tardiness,
        'repair_type': 'capacity',
    }
