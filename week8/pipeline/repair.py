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
from week8.config import TRUCK_CAPACITY, BATTERY_CAPACITY, ENERGY_CONSUMPTION_RATE

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
# Repair Operator 2B: Forward Insertion Repair (NEW — replaces segment-EDD)
# ═══════════════════════════════════════════════════════════════════════════

def _simulate_route_from(route, start_idx, customers, depot, truck_speed=35.0):
    """
    Simulate route starting from a given index (with correct state up to that point).
    Returns (arrivals, tardiness_vals, total_tard, total_dist).
    """
    arrivals = [0.0] * len(route)
    tardiness_vals = [0.0] * len(route)
    total_tard = 0.0
    total_dist = 0.0

    prev = 0  # depot
    current_time = 0.0

    for i, cust_idx in enumerate(route):
        c = customers[cust_idx - 1]
        if prev == 0:
            d = math.sqrt((depot[0] - c['x'])**2 + (depot[1] - c['y'])**2)
        else:
            prev_c = customers[prev - 1]
            d = math.sqrt((prev_c['x'] - c['x'])**2 + (prev_c['y'] - c['y'])**2)
        total_dist += d
        current_time += d / truck_speed

        if current_time < c['ready_time']:
            current_time = c['ready_time']
        arrivals[i] = current_time

        tard = max(0.0, current_time - c['due_time'])
        tardiness_vals[i] = tard
        total_tard += tard

        current_time += c['service_time']
        prev = cust_idx

    # Return to depot
    if prev == 0:
        d = 0.0
    else:
        prev_c = customers[prev - 1]
        d = math.sqrt((prev_c['x'] - depot[0])**2 + (prev_c['y'] - depot[1])**2)
    total_dist += d

    return arrivals, tardiness_vals, total_tard, total_dist


def repair_forward_insertion(solution, instance, seed=42):
    """
    Forward Insertion Repair — surgically moves tardy customers earlier.

    PHILOSOPHY:
      Segment-level EDD reorder always fails because tardiness is caused by
      accumulated upstream travel time, not by bad ordering within the tardy
      segment. To fix tardiness, we must move the late customer FORWARD —
      past non-tardy customers — to give them more travel time budget.

    ALGORITHM:
      1. Simulate route → identify all tardy customers
      2. Sort by tardiness descending (worst first)
      3. For each tardy customer:
         a. Remove from current position
         b. Try inserting at every EARLIER position
         c. Score each candidate: new_distance + new_tardiness × TARD_COST_WEIGHT
         d. Pick the best position
         e. If it improves total cost → accept the move
      4. After processing all tardy customers, verify
      5. If still tardy → fallback to Full EDD for that route

    This is MORE SURGICAL than Full EDD (only moves tardy customers, not all)
    and MORE EFFECTIVE than segment-EDD (moves customers FORWARD, not just
    reorders within a late segment).

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

    total_moves = 0
    total_moves_attempted = 0
    fallback_count = 0
    moved_customer_ids = []  # Track which customers FI moved (for iterative re-opt)

    TARD_WEIGHT = 5.0   # tardiness penalty weight (same as _route_cost)

    for ri, route in enumerate(routes):
        if len(route) <= 1:
            continue

        # ── Step 1: Identify tardy customers ──
        _, tard_vals, _, _, _, _ = _simulate_route(route, customers, depot)
        tardy_customers = [
            (i, route[i], tard_vals[i])
            for i in range(len(route)) if tard_vals[i] > 0.01
        ]
        if not tardy_customers:
            continue

        # ── Step 1.5: Theorem 1 pre-judgment ──
        # Compute forward slack to estimate FI success probability.
        # If slack is far below tardiness, skip FI and go straight to EDD.
        total_tard = sum(t for _, _, t in tardy_customers)
        # Compute prefix slack for the earliest tardy customer
        first_tardy_pos = tardy_customers[0][0]
        prefix_slack = 0.0
        if first_tardy_pos > 0:
            # Simulate arrivals to compute available buffer
            arrivals, _, _, _, _, _ = _simulate_route(route, customers, depot)
            for p in range(first_tardy_pos):
                c = customers[route[p] - 1]
                buffer = c['due_time'] - arrivals[p]
                if buffer > 0:
                    prefix_slack += buffer

        slack_ratio = prefix_slack / total_tard if total_tard > 0.01 else float('inf')

        if slack_ratio < 0.3:
            # Theorem 1 condition strongly violated — FI unlikely to succeed.
            # Skip straight to EDD fallback, saving wasted FI attempts.
            route = sorted(route, key=lambda cid: customers[cid - 1]['due_time'])
            fallback_count += 1
            routes[ri] = route
            continue

        # Sort: most tardy first
        tardy_customers.sort(key=lambda x: x[2], reverse=True)

        # ── Step 2: Forward-insert each tardy customer ──
        # Process in order of tardiness
        processed = set()  # positions we've already moved (index shifts!)
        # Use customer IDs to track, not positions
        moved_customers = set()

        for orig_pos, cust_id, tard in tardy_customers:
            if cust_id in moved_customers:
                continue  # already moved in a previous iteration

            total_moves_attempted += 1

            # Find current position of this customer (may have shifted)
            try:
                current_pos = route.index(cust_id)
            except ValueError:
                continue

            # Only try EARLIER positions (forward insertion)
            if current_pos == 0:
                continue  # Already first, can't move earlier

            # ── Try each earlier position ──
            best_pos = current_pos  # default: don't move
            best_cost = float('inf')

            # Remove customer from route temporarily
            route_no_cust = route[:current_pos] + route[current_pos+1:]

            # Try inserting at positions 0, 1, ..., current_pos (earlier positions)
            for insert_pos in range(current_pos + 1):  # 0 to current_pos inclusive
                test_route = (route_no_cust[:insert_pos] +
                             [cust_id] +
                             route_no_cust[insert_pos:])

                # Simulate to evaluate
                _, new_tard_vals, total_tard, total_dist = _simulate_route_from(
                    test_route, 0, customers, depot)

                # Cost = distance_cost + tardiness_cost
                cost = total_dist * TRUCK_DIST_COST_RATE + total_tard * TARD_WEIGHT

                if cost < best_cost - 0.01:
                    best_cost = cost
                    best_pos = insert_pos

            # ── Cross-route FI: if intra-route can't fix, try other routes ──
            best_cross_ri = -1
            best_cross_pos = -1
            best_cross_cost = float('inf')

            if best_pos == current_pos and len(routes) >= 2:
                # This customer can't be fixed within its own route.
                # Lemma 2 extended to multi-route: moving to another truck's
                # route may give the customer an earlier service time.
                src_route_no_cust = route[:current_pos] + route[current_pos+1:]
                src_dist, src_cost = 0.0, 0.0
                if src_route_no_cust:
                    _, _, src_tard, src_dist = _simulate_route_from(
                        src_route_no_cust, 0, customers, depot)
                    src_cost = src_dist * TRUCK_DIST_COST_RATE + src_tard * TARD_WEIGHT

                for other_ri in range(len(routes)):
                    if other_ri == ri:
                        continue
                    other_route = routes[other_ri]

                    for insert_pos in range(len(other_route) + 1):
                        test_other = (other_route[:insert_pos] +
                                     [cust_id] +
                                     other_route[insert_pos:])

                        # Check TW feasibility of destination route
                        _, dst_tard_vals, dst_tard, dst_dist = _simulate_route_from(
                            test_other, 0, customers, depot)

                        # Combined cost = source without customer + destination with customer
                        combined_cost = src_cost + dst_dist * TRUCK_DIST_COST_RATE + dst_tard * TARD_WEIGHT

                        if combined_cost < best_cross_cost - 0.01:
                            best_cross_cost = combined_cost
                            best_cross_ri = other_ri
                            best_cross_pos = insert_pos

            # ── Accept if intra-route or cross-route improved ──
            if best_pos != current_pos:
                route = (route[:current_pos] + route[current_pos+1:])
                route = (route[:best_pos] + [cust_id] + route[best_pos:])
                total_moves += 1
                moved_customers.add(cust_id)
                moved_customer_ids.append(cust_id)
            elif best_cross_ri >= 0:
                # Cross-route move: remove from source, insert into destination
                routes[ri] = route[:current_pos] + route[current_pos+1:]
                dst_route = routes[best_cross_ri]
                routes[best_cross_ri] = (dst_route[:best_cross_pos] +
                                         [cust_id] +
                                         dst_route[best_cross_pos:])
                # Remove empty source routes
                if not routes[ri]:
                    routes.pop(ri)
                    # Adjust route index references
                    if best_cross_ri > ri:
                        best_cross_ri -= 1
                total_moves += 1
                moved_customers.add(cust_id)
                moved_customer_ids.append(cust_id)
                # Update `route` reference for subsequent processing
                if ri < len(routes):
                    route = routes[ri]
                else:
                    break  # Source route was removed, done with this customer

        # ── Step 3: Verify after all moves ──
        _, final_tard_vals, final_tard, _ = _simulate_route_from(
            route, 0, customers, depot)

        # If still tardy, fallback to Full EDD for this route
        if final_tard > 0.01:
            route = sorted(route, key=lambda cid: customers[cid - 1]['due_time'])
            fallback_count += 1

        routes[ri] = route

    repaired = TruckSolution(routes, instance)

    stats = {
        'tardiness_before': tardiness_before,
        'tardiness_after': repaired.tardiness,
        'tardiness_reduction': tardiness_before - repaired.tardiness,
        'moves_attempted': total_moves_attempted,
        'moves_accepted': total_moves,
        'fallback_count': fallback_count,
        'forward_insertion_success': total_moves > 0,
        'partial_success': repaired.tardiness <= 1e-6,
        'repair_type': 'forward_insertion',
        'moved_customer_ids': list(moved_customer_ids),
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


# ═══════════════════════════════════════════════════════════════════════════
# Repair Operator 5: Constrained Local Search (NEW — post-FI distance optimization)
# ═══════════════════════════════════════════════════════════════════════════

def _route_distance(route, customers, depot):
    """Compute total route distance (depot → customers → depot)."""
    if not route:
        return 0.0
    total = 0.0
    prev = 0
    for cust_idx in route:
        c = customers[cust_idx - 1]
        if prev == 0:
            total += math.sqrt((depot[0] - c['x'])**2 + (depot[1] - c['y'])**2)
        else:
            prev_c = customers[prev - 1]
            total += math.sqrt((prev_c['x'] - c['x'])**2 + (prev_c['y'] - c['y'])**2)
        prev = cust_idx
    # Return to depot
    if prev != 0:
        prev_c = customers[prev - 1]
        total += math.sqrt((prev_c['x'] - depot[0])**2 + (prev_c['y'] - depot[1])**2)
    return total


def _check_tw_feasible(route, customers, depot, truck_speed=35.0):
    """Quick TW check — returns True if no customer is tardy."""
    _, _, _, total_tard, _, feasible = _simulate_route(
        route, customers, depot, truck_speed)
    return feasible and total_tard <= 1e-6


def constrained_local_search(solution, instance, max_iter=100, seed=42):
    """
    Post-repair distance optimization with hard TW constraint.

    PHILOSOPHY:
      Forward Insertion guarantees TW feasibility by moving tardy customers
      forward. This can increase distance (detours). We now apply standard
      local search operators (2-opt, relocate) to reduce distance, but with
      a HARD CONSTRAINT: no move is accepted if it introduces ANY tardiness.

      This is the natural complement to FI:
        FI:     fix tardiness → TW feasible (may hurt distance)
        CLS:    fix distance → lower cost (guarantees TW stays feasible)

    ALGORITHM:
      1. For each route, try 2-opt (reverse subsequence)
         → accept if distance ↓ AND TW still feasible
      2. For each route, try relocate (move customer to new position)
         → accept if distance ↓ AND TW still feasible
      3. Repeat until no improvement or max_iter reached

    Args:
        solution: TruckSolution (must already be TW-feasible)
        instance: problem instance dict
        max_iter: max iterations of 2-opt + relocate
        seed: random seed

    Returns:
        (TruckSolution, stats_dict)
    """
    random.seed(seed)
    customers = instance['customers']
    depot = instance['depot']
    truck_speed = instance.get('truck_speed', 35.0)

    routes = [list(rt) for rt in solution.truck_routes]
    initial_cost = solution.cost

    total_2opt_moves = 0
    total_relocate_moves = 0

    for iteration in range(max_iter):
        improved = False

        # ── Phase 1: 2-opt (reverse subsequence) ─────────────────
        for ri, route in enumerate(routes):
            n = len(route)
            if n < 3:
                continue  # need at least 3 nodes for meaningful 2-opt

            orig_dist = _route_distance(route, customers, depot)

            # Try all i, j pairs (i < j-1, so the reversed segment has ≥ 2 nodes)
            best_ij = None
            best_dist = orig_dist

            for i in range(n - 1):
                for j in range(i + 2, n + 1):
                    # 2-opt: reverse route[i:j]
                    new_route = route[:i] + route[i:j][::-1] + route[j:]

                    # Hard TW constraint
                    if not _check_tw_feasible(new_route, customers, depot, truck_speed):
                        continue

                    new_dist = _route_distance(new_route, customers, depot)
                    if new_dist < best_dist - 0.01:
                        best_dist = new_dist
                        best_ij = (i, j)

            if best_ij is not None:
                i, j = best_ij
                routes[ri] = route[:i] + route[i:j][::-1] + route[j:]
                total_2opt_moves += 1
                improved = True

        # ── Phase 2: Relocate (move one customer) ─────────────────
        for ri, route in enumerate(routes):
            n = len(route)
            if n < 2:
                continue

            orig_dist = _route_distance(route, customers, depot)
            best_move = None
            best_dist = orig_dist

            for pos, cust in enumerate(route):
                # Try inserting at each other position
                route_no_cust = route[:pos] + route[pos+1:]
                for ins in range(n):  # 0 to n-1 (n positions total, but one is current)
                    if ins == pos:
                        continue
                    new_route = route_no_cust[:ins] + [cust] + route_no_cust[ins:]

                    # Hard TW constraint
                    if not _check_tw_feasible(new_route, customers, depot, truck_speed):
                        continue

                    new_dist = _route_distance(new_route, customers, depot)
                    if new_dist < best_dist - 0.01:
                        best_dist = new_dist
                        best_move = (pos, ins)

            if best_move is not None:
                pos, ins = best_move
                cust = route[pos]
                route_no_cust = route[:pos] + route[pos+1:]
                routes[ri] = route_no_cust[:ins] + [cust] + route_no_cust[ins:]
                total_relocate_moves += 1
                improved = True

        # ── Phase 3: Inter-route relocate (move customer between routes) ──
        n_routes = len(routes)
        if n_routes >= 2:
            # Compute initial total distance
            initial_total_dist = sum(
                _route_distance(r, customers, depot) for r in routes if r)

            best_move = None  # (src_ri, src_pos, dst_ri, dst_pos, cust)
            best_dist = initial_total_dist

            for src_ri in range(n_routes):
                src_route = routes[src_ri]
                if len(src_route) <= 1:
                    continue  # Don't empty a route completely

                for src_pos, cust in enumerate(src_route):
                    src_route_no = src_route[:src_pos] + src_route[src_pos+1:]

                    for dst_ri in range(n_routes):
                        if dst_ri == src_ri:
                            continue
                        dst_route = routes[dst_ri]

                        for dst_pos in range(len(dst_route) + 1):
                            new_src = src_route_no
                            new_dst = dst_route[:dst_pos] + [cust] + dst_route[dst_pos:]

                            # Hard TW constraint on BOTH routes
                            if not _check_tw_feasible(new_src, customers, depot, truck_speed):
                                continue
                            if not _check_tw_feasible(new_dst, customers, depot, truck_speed):
                                continue

                            # Compute new total distance
                            new_dist = (_route_distance(new_src, customers, depot) +
                                       _route_distance(new_dst, customers, depot))
                            # Other routes unchanged
                            for other_ri in range(n_routes):
                                if other_ri not in (src_ri, dst_ri):
                                    new_dist += _route_distance(routes[other_ri], customers, depot)

                            if new_dist < best_dist - 0.01:
                                best_dist = new_dist
                                best_move = (src_ri, src_pos, dst_ri, dst_pos, cust)

            if best_move is not None:
                src_ri, src_pos, dst_ri, dst_pos, cust = best_move
                routes[src_ri] = routes[src_ri][:src_pos] + routes[src_ri][src_pos+1:]
                routes[dst_ri] = routes[dst_ri][:dst_pos] + [cust] + routes[dst_ri][dst_pos:]
                # Remove empty routes
                routes = [r for r in routes if r]
                total_relocate_moves += 1  # Count as relocate
                improved = True

        if not improved:
            break

    repaired = TruckSolution(routes, instance)
    final_cost = repaired.cost

    stats = {
        'cls_2opt_moves': total_2opt_moves,
        'cls_relocate_moves': total_relocate_moves,
        'cls_total_moves': total_2opt_moves + total_relocate_moves,
        'cls_iterations': iteration + 1,
        'cls_cost_before': round(initial_cost, 2) if initial_cost else 0,
        'cls_cost_after': round(final_cost, 2),
        'cls_cost_reduction': round(initial_cost - final_cost, 2) if initial_cost else 0,
        'cls_cost_reduction_pct': round((initial_cost - final_cost) / initial_cost * 100, 1) if initial_cost else 0,
    }
    return repaired, stats


# ═══════════════════════════════════════════════════════════════════════════
# Repair Operator 6: Post-Repair Route Merging (reduce multi-trip overhead)
# ═══════════════════════════════════════════════════════════════════════════

def merge_routes_post_repair(solution, instance, seed=42):
    """
    Merge short routes after FI/EDD repair to reduce depot round-trips.

    PHILOSOPHY:
      On tight-TW instances (C1, R1), temporal feasibility checks may split
      clusters into many small routes (1-2 customers each). Each route incurs
      a depot→customer→depot fixed distance cost. By merging compatible short
      routes, we reduce the number of depot round-trips while maintaining
      TW feasibility.

      This is ONLY safe AFTER repair (FI/EDD) because:
      - All routes are already TW-feasible
      - Merged routes are EDD-sorted for TW check
      - Only merges that preserve TW=0 are accepted

    ALGORITHM:
      1. Sort routes by size (smallest first — merge short ones)
      2. For each pair of routes:
         a. Concatenate and EDD-sort
         b. Check TW feasibility
         c. If feasible → merge (accept greedily)
      3. Repeat until stable (no more merges possible)

    Args:
        solution: TruckSolution (must already be TW-feasible)
        instance: problem instance dict
        seed: random seed

    Returns:
        (TruckSolution, stats_dict)
    """
    random.seed(seed)
    customers = instance['customers']
    depot = instance['depot']
    truck_speed = instance.get('truck_speed', 35.0)

    routes = [list(rt) for rt in solution.truck_routes if len(rt) > 0]
    initial_n_routes = len(routes)
    initial_cost = solution.cost
    merges_done = 0

    # Iteratively merge until stable
    changed = True
    while changed:
        changed = False
        n = len(routes)
        if n <= 1:
            break

        # Sort by length: merge shortest first
        indices = sorted(range(n), key=lambda i: len(routes[i]))

        for ii in range(n):
            i = indices[ii]
            if not routes[i]:
                continue
            for jj in range(ii + 1, n):
                j = indices[jj]
                if not routes[j]:
                    continue

                # Try merging route j into route i
                merged = routes[i] + routes[j]
                # EDD sort for optimal TW feasibility
                merged_edd = sorted(merged, key=lambda cid: customers[cid - 1]['due_time'])

                # Check TW feasibility of merged route
                if not _check_tw_feasible(merged_edd, customers, depot, truck_speed):
                    continue

                # ── Check capacity: merged route must not exceed truck capacity ──
                total_demand = sum(customers[cid - 1]['demand'] for cid in merged_edd)
                if total_demand > TRUCK_CAPACITY + 0.01:
                    continue  # Would overload the truck

                # ── Check EV battery: merged route energy must be feasible ──
                merged_energy = _route_distance(merged_edd, customers, depot) * ENERGY_CONSUMPTION_RATE
                if merged_energy > BATTERY_CAPACITY + 0.01:
                    continue  # Single truck battery can't handle this route

                routes[i] = merged_edd
                routes[j] = []
                merges_done += 1
                changed = True
                break  # Restart scan with updated routes
            if changed:
                break

        # Remove empty routes
        routes = [rt for rt in routes if rt]

    repaired = TruckSolution(routes, instance)
    final_cost = repaired.cost
    final_n_routes = len(routes)

    stats = {
        'merge_routes_before': initial_n_routes,
        'merge_routes_after': final_n_routes,
        'merge_routes_reduced': initial_n_routes - final_n_routes,
        'merge_merges_done': merges_done,
        'merge_cost_before': round(initial_cost, 2) if initial_cost else 0,
        'merge_cost_after': round(final_cost, 2),
        'merge_cost_reduction': round(initial_cost - final_cost, 2) if initial_cost else 0,
        'merge_cost_reduction_pct': round((initial_cost - final_cost) / initial_cost * 100, 1) if initial_cost else 0,
    }
    return repaired, stats
