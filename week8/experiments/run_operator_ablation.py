#!/usr/bin/env python3
"""P1 Experiment: Operator Ablation — Forward Insertion vs Or-opt vs 2-opt* vs Relocate.

Tests whether Forward Insertion's "forward-only" constraint is what makes it
effective, or whether any local search repair operator would work equally well.

Operators compared:
  1. Forward Insertion (ours): move tardy customer → try ALL earlier positions
  2. Relocate: move tardy customer → try ALL positions (forward AND backward)
  3. Or-opt: move a segment of 1-3 consecutive customers to an earlier position
  4. 2-opt*: reverse a subsequence, accept if reduces cost

All operators use the same scoring function (distance + tardiness × 5.0)
and the same Full EDD fallback.

Runs on a representative subset: all 6 TW types × 4 scales = 24 instances.
"""
import os, sys, json, time, math, random, traceback
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from week8.config import (
    TRUCK_FLEET_CONFIGS, TRUCK_SPEED, TRUCK_DIST_COST_RATE,
)
from week8.core.data_loader import load_instance_from_disk
from week8.core.problem_model import TruckSolution

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Representative instances: one per TW type per scale
REPRESENTATIVES = {
    'RC1': 'RC101', 'RC2': 'RC201',
    'R1': 'R101', 'R2': 'R201',
    'C1': 'C101', 'C2': 'C201',
}
SCALES = [25, 50, 100, 200]

TARD_WEIGHT = 5.0


# ── Route Simulation Utilities ──────────────────────────────────────────

def simulate_route(route, customers, depot):
    """Simulate route. Returns (arrivals, tardiness_vals, total_tard, total_dist)."""
    arrivals = [0.0] * len(route)
    tardiness_vals = [0.0] * len(route)
    total_tard = 0.0
    total_dist = 0.0
    prev = 0
    current_time = 0.0

    for i, cid in enumerate(route):
        c = customers[cid - 1]
        if prev == 0:
            d = math.hypot(depot[0] - c['x'], depot[1] - c['y'])
        else:
            pc = customers[prev - 1]
            d = math.hypot(pc['x'] - c['x'], pc['y'] - c['y'])
        total_dist += d
        current_time += d / TRUCK_SPEED
        if current_time < c['ready_time']:
            current_time = c['ready_time']
        arrivals[i] = current_time
        tard = max(0.0, current_time - c['due_time'])
        tardiness_vals[i] = tard
        total_tard += tard
        current_time += c['service_time']
        prev = cid

    # Return to depot
    if prev == 0:
        d = 0.0
    else:
        pc = customers[prev - 1]
        d = math.hypot(pc['x'] - depot[0], pc['y'] - depot[1])
    total_dist += d

    return arrivals, tardiness_vals, total_tard, total_dist


def route_cost(route, customers, depot):
    """Cost = distance * DIST_RATE + tardiness * TARD_WEIGHT."""
    _, _, tard, dist = simulate_route(route, customers, depot)
    return dist * TRUCK_DIST_COST_RATE + tard * TARD_WEIGHT


# ── Repair Operators ────────────────────────────────────────────────────

def repair_forward_insertion(routes, customers, depot):
    """Forward Insertion: move each tardy customer → try ALL earlier positions."""
    total_moves = 0
    fallback_count = 0
    new_routes = [list(r) for r in routes]

    for ri, route in enumerate(new_routes):
        if len(route) <= 1:
            continue

        _, tard_vals, _, _ = simulate_route(route, customers, depot)
        tardy = [(i, route[i], tard_vals[i]) for i in range(len(route)) if tard_vals[i] > 0.01]
        if not tardy:
            continue

        tardy.sort(key=lambda x: x[2], reverse=True)
        moved = set()

        for orig_pos, cust_id, _ in tardy:
            if cust_id in moved:
                continue
            try:
                cur_pos = route.index(cust_id)
            except ValueError:
                continue
            if cur_pos == 0:
                continue

            best_pos, best_cost = cur_pos, float('inf')
            route_removed = route[:cur_pos] + route[cur_pos+1:]

            for insert_pos in range(cur_pos + 1):
                test = route_removed[:insert_pos] + [cust_id] + route_removed[insert_pos:]
                cost = route_cost(test, customers, depot)
                if cost < best_cost - 0.01:
                    best_cost = cost
                    best_pos = insert_pos

            if best_pos != cur_pos:
                route = route_removed[:best_pos] + [cust_id] + route_removed[best_pos:]
                total_moves += 1
                moved.add(cust_id)

        # Verify + fallback
        _, _, final_tard, _ = simulate_route(route, customers, depot)
        if final_tard > 0.01:
            route = sorted(route, key=lambda cid: customers[cid-1]['due_time'])
            fallback_count += 1

        new_routes[ri] = route

    return new_routes, total_moves, fallback_count


def repair_relocate(routes, customers, depot):
    """Relocate: move each tardy customer → try ALL positions (forward + backward)."""
    total_moves = 0
    fallback_count = 0
    new_routes = [list(r) for r in routes]

    for ri, route in enumerate(new_routes):
        if len(route) <= 1:
            continue

        _, tard_vals, _, _ = simulate_route(route, customers, depot)
        tardy = [(i, route[i], tard_vals[i]) for i in range(len(route)) if tard_vals[i] > 0.01]
        if not tardy:
            continue

        tardy.sort(key=lambda x: x[2], reverse=True)
        moved = set()

        for orig_pos, cust_id, _ in tardy:
            if cust_id in moved:
                continue
            try:
                cur_pos = route.index(cust_id)
            except ValueError:
                continue

            best_pos, best_cost = cur_pos, float('inf')
            route_removed = route[:cur_pos] + route[cur_pos+1:]

            for insert_pos in range(len(route_removed) + 1):
                test = route_removed[:insert_pos] + [cust_id] + route_removed[insert_pos:]
                cost = route_cost(test, customers, depot)
                if cost < best_cost - 0.01:
                    best_cost = cost
                    best_pos = insert_pos

            if best_pos != cur_pos:
                route = route_removed[:best_pos] + [cust_id] + route_removed[best_pos:]
                total_moves += 1
                moved.add(cust_id)

        _, _, final_tard, _ = simulate_route(route, customers, depot)
        if final_tard > 0.01:
            route = sorted(route, key=lambda cid: customers[cid-1]['due_time'])
            fallback_count += 1

        new_routes[ri] = route

    return new_routes, total_moves, fallback_count


def repair_oropt(routes, customers, depot):
    """Or-opt: move segments of 1-3 consecutive customers to earlier positions."""
    total_moves = 0
    fallback_count = 0
    new_routes = [list(r) for r in routes]

    for ri, route in enumerate(new_routes):
        if len(route) <= 2:
            continue

        _, tard_vals, _, _ = simulate_route(route, customers, depot)

        # Find all tardy customers
        tardy_positions = {i for i, t in enumerate(tard_vals) if t > 0.01}
        if not tardy_positions:
            continue

        improved = True
        iteration = 0
        while improved and iteration < 20:
            improved = False
            iteration += 1

            for seg_len in [1, 2, 3]:
                for start in range(len(route) - seg_len + 1):
                    segment = route[start:start + seg_len]
                    # Check if segment contains any tardy customer
                    if not any(p in tardy_positions for p in range(start, start + seg_len)):
                        continue

                    route_removed = route[:start] + route[start+seg_len:]
                    best_pos, best_cost = start, route_cost(route, customers, depot)

                    for insert_pos in range(start + 1):
                        test = route_removed[:insert_pos] + segment + route_removed[insert_pos:]
                        cost = route_cost(test, customers, depot)
                        if cost < best_cost - 0.01:
                            best_cost = cost
                            best_pos = insert_pos

                    if best_pos != start:
                        route = route_removed[:best_pos] + segment + route_removed[best_pos:]
                        _, new_tard, _, _ = simulate_route(route, customers, depot)
                        tardy_positions = {i for i, t in enumerate(new_tard) if t > 0.01}
                        total_moves += 1
                        improved = True
                        break
                if improved:
                    break

        _, _, final_tard, _ = simulate_route(route, customers, depot)
        if final_tard > 0.01:
            route = sorted(route, key=lambda cid: customers[cid-1]['due_time'])
            fallback_count += 1

        new_routes[ri] = route

    return new_routes, total_moves, fallback_count


def repair_two_opt_star(routes, customers, depot):
    """2-opt*: reverse subsequences if it reduces cost (focus on tardy segments)."""
    total_moves = 0
    fallback_count = 0
    new_routes = [list(r) for r in routes]

    for ri, route in enumerate(new_routes):
        if len(route) <= 2:
            continue

        _, tard_vals, _, _ = simulate_route(route, customers, depot)
        tardy_positions = {i for i, t in enumerate(tard_vals) if t > 0.01}

        improved = True
        iteration = 0
        while improved and iteration < 30:
            improved = False
            iteration += 1
            best_improvement = 0
            best_move = None

            for i in range(len(route) - 1):
                for j in range(i + 2, len(route) + 1):
                    # Only try 2-opt on segments containing or near tardy customers
                    segment_has_tardy = any(p in tardy_positions for p in range(i, min(j, len(route))))
                    if not segment_has_tardy and random.random() > 0.1:
                        continue

                    test = route[:i] + list(reversed(route[i:j])) + route[j:]
                    old_cost = route_cost(route, customers, depot)
                    new_cost = route_cost(test, customers, depot)
                    improvement = old_cost - new_cost

                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = (i, j)

            if best_move and best_improvement > 0.01:
                i, j = best_move
                route = route[:i] + list(reversed(route[i:j])) + route[j:]
                _, new_tard, _, _ = simulate_route(route, customers, depot)
                tardy_positions = {p for p, t in enumerate(new_tard) if t > 0.01}
                total_moves += 1
                improved = True

        _, _, final_tard, _ = simulate_route(route, customers, depot)
        if final_tard > 0.01:
            route = sorted(route, key=lambda cid: customers[cid-1]['due_time'])
            fallback_count += 1

        new_routes[ri] = route

    return new_routes, total_moves, fallback_count


# ── Main ────────────────────────────────────────────────────────────────

def get_pomo_routes(instance, n_trucks):
    """Get POMO solution routes before repair."""
    from week8.pipeline.pipeline import solve_evrptw
    result = solve_evrptw(instance, n_trucks=n_trucks, variant='budget_aware',
                          use_repair=False, repair_mode='forward',
                          n_runs=1, seed=42)
    if not result['solutions']:
        return None
    return result['solutions'][0].truck_routes


def main():
    print("=" * 70)
    print("OPERATOR ABLATION: FI vs Relocate vs Or-opt vs 2-opt*")
    print("=" * 70)

    results = {}

    for tw_type, inst_name in REPRESENTATIVES.items():
        for scale in SCALES:
            key = f"{inst_name}_{scale}c"
            n_trucks = TRUCK_FLEET_CONFIGS.get(scale, [2])[len(TRUCK_FLEET_CONFIGS.get(scale, [2]))//2]

            try:
                instance = load_instance_from_disk(key)
            except FileNotFoundError:
                print(f"  SKIP {key}")
                continue

            print(f"\n  {key} ({tw_type}, {n_trucks}t):")

            # Get POMO routes (no repair)
            try:
                routes = get_pomo_routes(instance, n_trucks)
                if routes is None:
                    print(f"    POMO FAILED")
                    continue
            except Exception as e:
                print(f"    POMO ERROR: {e}")
                continue

            customers = instance['customers']
            depot = instance['depot']

            # Compute base metrics
            base_sol = TruckSolution(routes, instance)
            print(f"    Base: cost={base_sol.cost:.0f} tard={base_sol.tardiness:.0f}")

            results[key] = {
                'tw_type': tw_type, 'scale': scale, 'n_trucks': n_trucks,
                'base_cost': round(base_sol.cost, 2),
                'base_tardiness': round(base_sol.tardiness, 2),
            }

            # Test each operator
            operators = [
                ('Forward Insertion', repair_forward_insertion),
                ('Relocate', repair_relocate),
                ('Or-opt', repair_oropt),
                ('2-opt*', repair_two_opt_star),
            ]

            for op_name, op_fn in operators:
                t0 = time.time()
                try:
                    new_routes, moves, fallbacks = op_fn(routes, customers, depot)
                    new_sol = TruckSolution(new_routes, instance)
                    elapsed = time.time() - t0

                    results[key][op_name] = {
                        'cost': round(new_sol.cost, 2),
                        'tardiness': round(new_sol.tardiness, 2),
                        'tw_feasible': new_sol.tardiness <= 1e-6,
                        'moves': moves,
                        'fallbacks': fallbacks,
                        'runtime': round(elapsed, 3),
                    }

                    feas = '✓' if new_sol.tardiness <= 1e-6 else '✗'
                    print(f"    {op_name:<20}: cost={new_sol.cost:>8.0f} "
                          f"tard={new_sol.tardiness:>6.0f} TW={feas} "
                          f"moves={moves} fb={fallbacks} {elapsed:.3f}s")
                except Exception as e:
                    print(f"    {op_name:<20}: ERROR {e}")
                    results[key][op_name] = {'error': str(e)}

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"ABLATION SUMMARY")
    print(f"{'='*70}")

    for op_name in ['Forward Insertion', 'Relocate', 'Or-opt', '2-opt*']:
        tw_ok = 0
        total = 0
        total_moves = 0
        total_fb = 0
        for key, data in results.items():
            if op_name in data and 'error' not in data[op_name]:
                total += 1
                if data[op_name]['tw_feasible']:
                    tw_ok += 1
                total_moves += data[op_name].get('moves', 0)
                total_fb += data[op_name].get('fallbacks', 0)
        if total > 0:
            print(f"  {op_name:<20}: TW={tw_ok}/{total} ({tw_ok/max(total,1)*100:.0f}%) "
                  f"moves={total_moves} fb={total_fb}")

    # Save
    out_path = os.path.join(RESULTS_DIR, 'exp_operator_ablation.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == '__main__':
    main()
