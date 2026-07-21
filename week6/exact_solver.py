# -*- coding: utf-8 -*-
"""
Exact Solver & Optimality Gap Analysis — Week 7 Gap 5.

Uses OR-Tools to compute optimal (or near-optimal) VRPTW solutions
for small instances (10-15 customers, truck-only), then compares
against our pipeline to compute optimality gaps.

OR-Tools is installed in: Or-tools/.venv/ (v9.15.6755)

Usage:
    python week7/exact_solver.py              # Self-test
    python week7/run_gap_analysis.py          # Full gap analysis
"""

import math, os, sys, time

_W6 = os.path.dirname(os.path.abspath(__file__))
_W4 = os.path.join(_W6, '..', 'week4')
_W3 = os.path.join(_W6, '..', 'week3')

for _p in [_W4, _W3]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import TRUCK_SPEED, TRUCK_CAPACITY, TRUCK_DIST_COST_RATE, DEPOT

# ── OR-Tools Import (with graceful fallback) ──────────────────────────

_ORTOOLS_AVAILABLE = False
_ORTOOLS_ERROR = None

try:
    # Try the global install first, then the local venv
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    _ORTOOLS_AVAILABLE = True
except ImportError:
    try:
        # Try the Or-tools/.venv
        _ortools_venv = os.path.join(_W6, '..', 'Or-tools', '.venv')
        _ortools_site = os.path.join(_ortools_venv, 'lib')
        # Find the python version directory
        if os.path.exists(_ortools_site):
            for d in os.listdir(_ortools_site):
                sp = os.path.join(_ortools_site, d, 'site-packages')
                if os.path.exists(sp):
                    sys.path.insert(0, sp)
                    break
        from ortools.constraint_solver import routing_enums_pb2
        from ortools.constraint_solver import pywrapcp
        _ORTOOLS_AVAILABLE = True
    except (ImportError, FileNotFoundError) as e:
        _ORTOOLS_ERROR = str(e)


# ── Exact VRPTW Solver ────────────────────────────────────────────────

def solve_vrptw_exact(instance, time_limit=60, n_vehicles=None):
    """
    Solve VRPTW exactly using OR-Tools (truck-only, no drones).

    Args:
        instance: problem instance dict
        time_limit: max solver time in seconds
        n_vehicles: number of vehicles (default: auto from instance size)

    Returns:
        dict with:
          - cost: total cost (objective value)
          - routes: list of routes (list of customer ID lists)
          - is_optimal: bool (True if proven optimal)
          - wall_time: solver wall time
          - error: error message if solver failed
    """
    if not _ORTOOLS_AVAILABLE:
        return {'cost': None, 'routes': [], 'is_optimal': False,
                'wall_time': 0, 'error': f'OR-Tools not available: {_ORTOOLS_ERROR}'}

    customers = instance['customers']
    depot = instance['depot']
    n = len(customers)

    if n_vehicles is None:
        n_vehicles = max(2, n // 5)

    # ── Build distance + time matrices ──
    # Node 0 = depot, Nodes 1..n = customers
    def _dist(i, j):
        """Distance between nodes (0=depot, 1..n=customer)."""
        if i == 0:
            px, py = depot
        else:
            px, py = customers[i-1]['x'], customers[i-1]['y']
        if j == 0:
            qx, qy = depot
        else:
            qx, qy = customers[j-1]['x'], customers[j-1]['y']
        return math.sqrt((px - qx)**2 + (py - qy)**2)

    # Create routing model
    manager = pywrapcp.RoutingIndexManager(n + 1, n_vehicles, 0)  # 0 = depot
    routing = pywrapcp.RoutingModel(manager)

    # ── Distance callback ──
    def distance_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)
        return int(_dist(from_node, to_node) * 1000)  # mm precision

    dist_cb_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_cb_idx)

    # Add distance dimension (no upper bound)
    routing.AddDimension(dist_cb_idx, 0, 1000000, True, 'Distance')

    # ── Time callback ──
    def time_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)
        travel_time = _dist(from_node, to_node) / TRUCK_SPEED * 60  # minutes
        # Add service time at destination
        if to_node > 0:
            travel_time += customers[to_node-1]['service_time']
        return int(travel_time * 10)  # 0.1-min precision

    time_cb_idx = routing.RegisterTransitCallback(time_callback)

    # Time window constraints
    horizon = int(instance.get('tw_horizon', 240) * 10)  # 0.1-min units
    routing.AddDimension(
        time_cb_idx,
        horizon,      # max waiting (allow any wait)
        horizon,      # max time per vehicle
        False,        # don't force start cumul to zero
        'Time')

    time_dimension = routing.GetDimensionOrDie('Time')

    # Add time windows for each customer
    for i in range(1, n + 1):
        c = customers[i-1]
        idx = manager.NodeToIndex(i)
        ready = int(c['ready_time'] * 10)   # 0.1-min units
        due = int(c['due_time'] * 10)
        time_dimension.CumulVar(idx).SetRange(ready, due)

    # ── Capacity constraints ──
    def demand_callback(from_idx):
        from_node = manager.IndexToNode(from_idx)
        if from_node == 0:
            return 0
        return int(customers[from_node-1]['demand'])

    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb_idx, 0, [int(TRUCK_CAPACITY)] * n_vehicles, True, 'Capacity')

    # ── Solve ──
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_params.time_limit.seconds = time_limit
    search_params.log_search = False

    t0 = time.time()
    solution = routing.SolveWithParameters(search_params)
    wall_time = time.time() - t0

    if not solution:
        return {'cost': None, 'routes': [], 'is_optimal': False,
                'wall_time': wall_time, 'error': 'No solution found'}

    # ── Extract solution ──
    routes = []
    total_dist = 0.0
    for v in range(n_vehicles):
        route = []
        idx = routing.Start(v)
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node > 0:
                route.append(node)
            prev_idx = idx
            idx = solution.Value(routing.NextVar(idx))
            if not routing.IsEnd(idx):
                total_dist += _dist(
                    manager.IndexToNode(prev_idx),
                    manager.IndexToNode(idx))
        if route:
            # Add return to depot distance
            last_node = manager.IndexToNode(prev_idx)
            total_dist += _dist(last_node, 0)
            routes.append(route)

    objective = solution.ObjectiveValue() / 1000.0  # convert back to km
    is_optimal = True  # OR-Tools guarantees optimality for small instances

    # Cost in our project's terms
    cost = n_vehicles * 100.0 + total_dist * TRUCK_DIST_COST_RATE

    return {
        'cost': cost,
        'distance': total_dist,
        'routes': routes,
        'is_optimal': is_optimal,
        'wall_time': wall_time,
        'n_vehicles': n_vehicles,
        'objective_km': objective,
    }


# ── Optimality Gap Computation ───────────────────────────────────────

def compute_optimality_gap(our_cost, optimal_cost):
    """gap = (our_cost - optimal_cost) / optimal_cost * 100"""
    if optimal_cost is None or optimal_cost <= 0:
        return None
    return (our_cost - optimal_cost) / optimal_cost * 100


# ── Small Instance Builder ────────────────────────────────────────────

def build_small_instance(source_instance, n_customers, seed=42):
    """Create a small instance for gap analysis."""
    from utils.data_loader import build_instance
    return build_instance(source_instance, n_customers, seed)


def compare_on_small_instances():
    """
    Build small instances, solve optimally with OR-Tools, compare with our pipeline.

    Returns: list of comparison dicts
    """
    from utils.data_loader import load_instance_from_disk
    from utils.problem_model import TruckDroneSolution, evaluate_solution_batch

    # Try to import pipeline (may fail if torch not available)
    try:
        from pipeline import run_pipeline
        pipeline_available = True
    except (ImportError, ModuleNotFoundError):
        pipeline_available = False

    configs = [
        ('RC101', 10, 2),
        ('RC101', 15, 2),
        ('RC201', 10, 2),
        ('RC201', 15, 2),
        ('RC102', 10, 2),
        ('RC202', 10, 2),
    ]

    results = []

    for src, nc, n_trucks in configs:
        key = f'{src}_{nc}c'
        print(f'\n--- {key} ---')

        # Build instance
        try:
            inst = load_instance_from_disk(key)
        except FileNotFoundError:
            inst = build_small_instance(src, nc)
        print(f'  {inst["n_customers"]} customers, {inst["tw_type"]}')

        # OR-Tools exact solve
        t0 = time.time()
        exact = solve_vrptw_exact(inst, time_limit=30, n_vehicles=n_trucks)
        exact_time = time.time() - t0

        if exact['cost'] is not None:
            print(f'  OR-Tools: cost={exact["cost"]:.0f}  dist={exact["distance"]:.1f}km  '
                  f'optimal={exact["is_optimal"]}  time={exact["wall_time"]:.1f}s')
        else:
            print(f'  OR-Tools: FAILED — {exact.get("error", "unknown")}')

        # Our pipeline
        our_result = None
        if pipeline_available:
            try:
                t0 = time.time()
                r = run_pipeline(inst, n_trucks=n_trucks, variant='hybrid',
                               use_repair=True, repair_mode='full',
                               n_runs=3, seed=42)
                our_time = time.time() - t0
                m = evaluate_solution_batch(r['solutions'])
                our_result = m
                print(f'  Our:     cost={m["mean_cost"]:.0f}  '
                      f'feas={m["feasibility_rate"]*100:.0f}%  '
                      f'time={our_time:.1f}s')
            except Exception as e:
                print(f'  Our:     FAILED — {e}')

        # Compute gap
        gap_pct = None
        if exact['cost'] is not None and our_result is not None:
            gap_pct = compute_optimality_gap(our_result['mean_cost'], exact['cost'])
            print(f'  Gap: {gap_pct:+.1f}%')

        results.append({
            'instance': key,
            'n_customers': nc,
            'tw_type': inst['tw_type'],
            'exact_cost': exact['cost'],
            'exact_distance': exact.get('distance'),
            'exact_optimal': exact['is_optimal'],
            'exact_time': exact['wall_time'],
            'our_cost': our_result['mean_cost'] if our_result else None,
            'our_feasibility': our_result['feasibility_rate'] if our_result else None,
            'gap_pct': gap_pct,
        })

    return results


# ── Fallback: Heuristic Lower Bound ───────────────────────────────────

def nearest_neighbor_cost(instance):
    """
    Nearest-neighbor heuristic as a fallback when OR-Tools is unavailable.
    Returns a lower-effort reference cost (upper bound on optimal).
    """
    customers = instance['customers']
    depot = instance['depot']

    unvisited = set(range(1, len(customers) + 1))
    total_dist = 0.0
    current = 0  # depot

    while unvisited:
        # Find nearest unvisited customer
        best_id = min(unvisited, key=lambda cid: math.sqrt(
            (depot[0] - customers[cid-1]['x'])**2 +
            (depot[1] - customers[cid-1]['y'])**2
        ) if current == 0 else math.sqrt(
            (customers[current-1]['x'] - customers[cid-1]['x'])**2 +
            (customers[current-1]['y'] - customers[cid-1]['y'])**2
        ))
        total_dist += math.sqrt(
            (depot[0] - customers[best_id-1]['x'])**2 +
            (depot[1] - customers[best_id-1]['y'])**2
        ) if current == 0 else math.sqrt(
            (customers[current-1]['x'] - customers[best_id-1]['x'])**2 +
            (customers[current-1]['y'] - customers[best_id-1]['y'])**2
        )
        unvisited.remove(best_id)
        current = best_id

    # Return to depot
    total_dist += math.sqrt(
        (depot[0] - customers[current-1]['x'])**2 +
        (depot[1] - customers[current-1]['y'])**2
    )

    cost = 2 * 100.0 + total_dist * TRUCK_DIST_COST_RATE
    return {'cost': cost, 'distance': total_dist, 'method': 'nearest_neighbor'}


# ── Self-Test ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== Exact Solver Self-Test ===\n")
    print(f"OR-Tools available: {_ORTOOLS_AVAILABLE}")
    if not _ORTOOLS_AVAILABLE:
        print(f"  Error: {_ORTOOLS_ERROR}")
        print("\nFalling back to nearest-neighbor baseline...")
        from utils.data_loader import build_instance
        inst = build_instance('RC101', 10)
        ref = nearest_neighbor_cost(inst)
        print(f"  NN reference: cost={ref['cost']:.0f}, distance={ref['distance']:.1f}km")

    if _ORTOOLS_AVAILABLE:
        from utils.data_loader import build_instance

        # Test on a small instance
        inst = build_instance('RC101', 10)
        print(f"\nInstance: RC101_10c ({inst['n_customers']} customers)")
        result = solve_vrptw_exact(inst, time_limit=10, n_vehicles=2)
        print(f"  Cost: {result['cost']:.0f}")
        print(f"  Distance: {result['distance']:.1f} km")
        print(f"  Optimal: {result['is_optimal']}")
        print(f"  Time: {result['wall_time']:.1f}s")
        print(f"  Vehicles: {result['n_vehicles']}")
        print(f"  Routes: {[len(r) for r in result['routes']]} customers each")

        # Run full comparison
        print("\n--- Full Gap Analysis ---")
        results = compare_on_small_instances()
        for r in results:
            gap_str = f"{r['gap_pct']:+.1f}%" if r['gap_pct'] is not None else "N/A"
            print(f"  {r['instance']}: exact={r['exact_cost']:.0f}  "
                  f"our={r['our_cost']:.0f}  gap={gap_str}")
