#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization: Pareto Fronts + Route Maps for Truck-Drone EVRP-TW.

Generates:
  1. Pareto Front scatter plots (Cost vs Tardiness, colored by method)
  2. Route maps (geographic layout with truck routes + drone missions)

Usage:
    python visualize.py                    # All configs
    python visualize.py --config 25c       # 25-customer only
    python visualize.py --pareto-only      # Only Pareto fronts
    python visualize.py --routes-only      # Only route maps
    python visualize.py --quick            # 4 representative configs only
"""

import os, sys, json, math, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try importing matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    HAS_MPL = True
except ImportError:
    print("ERROR: matplotlib required. Install: pip install matplotlib")
    HAS_MPL = False
    sys.exit(1)

import numpy as np
from collections import defaultdict

from config import (
    RESULTS_DIR, DATA_OUT_DIR, DEPOT, COORD_SCALE,
    RC1_INSTANCES, RC2_INSTANCES, CUSTOMER_SIZES,
    DRONE_ENDURANCE, VEHICLE_CONFIGS,
)
from utils.data_loader import load_instance_from_disk
from utils.problem_model import TruckDroneSolution, extract_pareto_front

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'visualizations')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Color scheme ───────────────────────────────────────────────────
COLORS = {
    'P-ACO': '#2ecc71',      # Green
    'NSGA-II': '#3498db',    # Blue
    'IVND': '#e74c3c',       # Red
    'No-Drone': '#f39c12',   # Orange
    'POMO': '#9b59b6',       # Purple
}
MARKERS = {
    'P-ACO': 'o',
    'NSGA-II': 's',
    'IVND': '^',
    'No-Drone': 'D',
    'POMO': 'v',
}


# ═══════════════════════════════════════════════════════════════════
#  PART 1: Pareto Front Visualization
# ═══════════════════════════════════════════════════════════════════

def compute_pareto_fronts_from_data(experiment_data, config_label):
    """
    For a given config, run each algorithm once to collect solution points.
    Returns dict: method -> [(cost, tardiness), ...]
    """
    from algorithms.no_drone import solve_no_drone
    from algorithms.paco import PACOSolver
    from algorithms.nsga2 import NSGA2Solver
    from algorithms.ivnd import IVNDSolver

    exp = experiment_data
    inst = load_instance_from_disk(exp['instance_key'])
    n = exp['n_customers']
    endurance = exp['endurance']
    n_t, n_d = exp['n_trucks'], exp['n_drones']

    points = {}

    # No-Drone
    try:
        sols = solve_no_drone(inst, pop_size=80, generations=120, seed=42)
        pareto = extract_pareto_front(sols)
        points['No-Drone'] = [(s.cost, s.tardiness) for s in pareto]
    except Exception as e:
        print(f"  No-Drone error: {e}")
        points['No-Drone'] = []

    # P-ACO
    try:
        solver = PACOSolver(inst, seed=42)
        sols, pareto = solver.solve(endurance=endurance)
        points['P-ACO'] = [(s.cost, s.tardiness) for s in pareto]
    except Exception as e:
        print(f"  P-ACO error: {e}")
        points['P-ACO'] = []

    # NSGA-II
    try:
        solver = NSGA2Solver(inst, n_trucks=n_t, n_drones=n_d, endurance=endurance, seed=42)
        sols, pareto = solver.solve()
        points['NSGA-II'] = [(s.cost, s.tardiness) for s in pareto]
    except Exception as e:
        print(f"  NSGA-II error: {e}")
        points['NSGA-II'] = []

    # IVND
    try:
        solver = IVNDSolver(inst, n_trucks=n_t, n_drones=n_d, endurance=endurance, seed=42)
        sols, pareto = solver.solve()
        points['IVND'] = [(s.cost, s.tardiness) for s in pareto]
    except Exception as e:
        print(f"  IVND error: {e}")
        points['IVND'] = []

    return points


def compute_joint_pareto(all_points):
    """Find the joint non-dominated front across all methods."""
    all_pts = []
    for method, pts in all_points.items():
        all_pts.extend(pts)
    if not all_pts:
        return []
    # Non-dominated filter
    nondom = []
    for i, (c1, t1) in enumerate(all_pts):
        dominated = False
        for j, (c2, t2) in enumerate(all_pts):
            if i == j:
                continue
            if c2 <= c1 and t2 <= t1 and (c2 < c1 or t2 < t1):
                dominated = True
                break
        if not dominated:
            nondom.append((c1, t1))
    nondom.sort(key=lambda p: p[0])
    return nondom


def plot_pareto_front(points_dict, title, save_path, joint_pareto=None):
    """Plot Pareto front scatter plot with method colors."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for method in ['No-Drone', 'P-ACO', 'NSGA-II', 'IVND']:
        pts = points_dict.get(method, [])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.scatter(xs, ys, c=COLORS[method], marker=MARKERS[method],
                   label=method, alpha=0.7, s=40, edgecolors='black', linewidth=0.3)

    # Joint Pareto front line
    if joint_pareto and len(joint_pareto) > 1:
        xs = [p[0] for p in joint_pareto]
        ys = [p[1] for p in joint_pareto]
        ax.plot(xs, ys, 'k--', linewidth=1.5, alpha=0.6, label='Joint Pareto Front')

    ax.set_xlabel('Travel Cost', fontsize=12)
    ax.set_ylabel('Tardiness', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


def generate_pareto_plots(results, subset=None):
    """Generate Pareto front plots for selected experiment configs."""
    print("\n" + "=" * 60)
    print("Generating Pareto Front Plots")
    print("=" * 60)

    configs = results if subset is None else [r for r in results if subset in r['label']]

    for exp in configs:
        label = exp['label']
        print(f"\n  {label}...")
        try:
            points = compute_pareto_fronts_from_data(exp, label)
            joint = compute_joint_pareto(points)

            tw_name = 'Tight TW' if exp['tw_type'] == 'RC1' else 'Wide TW'
            end_name = f"{exp['endurance']}km"
            title = f"Pareto Front: {exp['n_customers']}c {exp['tw_type']} {end_name} {exp['n_trucks']}T+{exp['n_drones']}D"

            save_name = f"pareto_{label}.png"
            plot_pareto_front(points, title, os.path.join(OUTPUT_DIR, save_name), joint)
        except Exception as e:
            print(f"    ERROR: {e}")


# ═══════════════════════════════════════════════════════════════════
#  PART 2: Route Visualization
# ═══════════════════════════════════════════════════════════════════

def plot_route_map(instance, truck_routes, drone_missions, title, save_path,
                   method_name=''):
    """Plot geographic route map with truck routes and drone missions."""
    fig, ax = plt.subplots(figsize=(10, 10))

    customers = instance['customers']
    depot = instance['depot']

    # Plot depot
    ax.scatter(depot[0], depot[1], c='red', marker='s', s=150,
               zorder=5, edgecolors='black', linewidth=1.5, label='Depot')

    # Plot customers
    cx = [c['x'] for c in customers]
    cy = [c['y'] for c in customers]
    ax.scatter(cx, cy, c='gray', marker='o', s=40, zorder=3,
               edgecolors='black', linewidth=0.5, alpha=0.7)

    # Label customer indices
    for c in customers:
        ax.annotate(str(c['id']), (c['x'], c['y']), textcoords="offset points",
                    xytext=(3, 3), fontsize=7, alpha=0.7)

    # Plot truck routes
    truck_colors = plt.cm.Set1(np.linspace(0, 1, max(len(truck_routes), 1)))
    for ri, route in enumerate(truck_routes):
        if not route:
            continue
        color = truck_colors[ri % len(truck_colors)]
        # Route = depot -> customers -> depot
        path = [depot] + [(customers[cid-1]['x'], customers[cid-1]['y']) for cid in route] + [depot]
        px, py = zip(*path)
        ax.plot(px, py, '-', color=color, linewidth=2, alpha=0.8,
                label=f'Truck {ri+1}' if ri == 0 else f'Truck {ri+1}')
        # Direction arrows
        for i in range(len(path) - 1):
            mid_x = (path[i][0] + path[i+1][0]) / 2
            mid_y = (path[i][1] + path[i+1][1]) / 2
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            ax.annotate('', xy=(mid_x + dx*0.05, mid_y + dy*0.05),
                        xytext=(mid_x - dx*0.05, mid_y - dy*0.05),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.6))

    # Plot drone missions (dashed lines)
    for mission in drone_missions:
        i, j, k = mission  # launch, customer, recovery
        launch_xy = depot if i == 0 else (customers[i-1]['x'], customers[i-1]['y'])
        cust_xy = (customers[j-1]['x'], customers[j-1]['y'])
        recov_xy = depot if k == 0 else (customers[k-1]['x'], customers[k-1]['y'])

        # Launch -> customer
        ax.plot([launch_xy[0], cust_xy[0]], [launch_xy[1], cust_xy[1]],
                '--', color='#e74c3c', linewidth=1.5, alpha=0.7)
        # Customer -> recovery
        ax.plot([cust_xy[0], recov_xy[0]], [cust_xy[1], recov_xy[1]],
                '--', color='#e74c3c', linewidth=1.5, alpha=0.7)
        # Mark drone customer
        ax.scatter(cust_xy[0], cust_xy[1], c='#e74c3c', marker='*', s=120,
                   zorder=6, edgecolors='black', linewidth=0.5)

    # Custom legend
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
               markersize=10, label='Depot'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=8, label='Customer'),
        Line2D([0], [0], color=truck_colors[0], linewidth=2, label='Truck Route'),
        Line2D([0], [0], color='#e74c3c', linewidth=1.5, linestyle='--',
               label='Drone Mission'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#e74c3c',
               markersize=10, label='Drone Customer'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    ax.set_xlabel('X (km)', fontsize=12)
    ax.set_ylabel('Y (km)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 16)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


def get_best_solution(instance, method, endurance, n_t, n_d):
    """Get the best feasible solution for a given method."""
    from algorithms.no_drone import solve_no_drone
    from algorithms.paco import PACOSolver
    from algorithms.nsga2 import NSGA2Solver
    from algorithms.ivnd import IVNDSolver

    best_sol = None
    best_cost = float('inf')

    if method == 'No-Drone':
        sols = solve_no_drone(instance, pop_size=80, generations=120, seed=42)
        for s in sols:
            if s.feasible and s.cost < best_cost:
                best_cost = s.cost
                best_sol = s

    elif method == 'P-ACO':
        solver = PACOSolver(instance, seed=42)
        sols, _ = solver.solve(endurance=endurance)
        for s in sols:
            if s.feasible and s.cost < best_cost:
                best_cost = s.cost
                best_sol = s

    elif method == 'NSGA-II':
        solver = NSGA2Solver(instance, n_trucks=n_t, n_drones=n_d, endurance=endurance, seed=42)
        sols, _ = solver.solve()
        for s in sols:
            if s.feasible and s.cost < best_cost:
                best_cost = s.cost
                best_sol = s

    elif method == 'IVND':
        solver = IVNDSolver(instance, n_trucks=n_t, n_drones=n_d, endurance=endurance, seed=42)
        sols, _ = solver.solve()
        for s in sols:
            if s.feasible and s.cost < best_cost:
                best_cost = s.cost
                best_sol = s

    return best_sol


def generate_route_maps(results, subset=None):
    """Generate route map visualizations for selected configs."""
    print("\n" + "=" * 60)
    print("Generating Route Maps")
    print("=" * 60)

    # Select representative configs: one per scale × TW type
    selected = []

    # 25c configs
    for tw in ['RC1', 'RC2']:
        for exp in results:
            if exp['n_customers'] == 25 and exp['tw_type'] == tw and 'medium' in exp['endurance_name']:
                selected.append(exp)
                break

    # 50c configs (4T+4D)
    for tw in ['RC1', 'RC2']:
        for exp in results:
            if (exp['n_customers'] == 50 and exp['tw_type'] == tw and
                exp['n_trucks'] == 4 and 'medium' in exp['endurance_name']):
                selected.append(exp)
                break

    # 100c configs (4T+4D)
    for tw in ['RC1', 'RC2']:
        for exp in results:
            if (exp['n_customers'] == 100 and exp['tw_type'] == tw and
                exp['n_trucks'] == 4 and 'medium' in exp['endurance_name']):
                selected.append(exp)
                break

    if subset:
        selected = [s for s in selected if subset in s['label']]

    for exp in selected:
        label = exp['label']
        print(f"\n  {label}...")
        try:
            inst = load_instance_from_disk(exp['instance_key'])

            for method in ['P-ACO', 'NSGA-II', 'No-Drone']:
                print(f"    {method}...")
                sol = get_best_solution(inst, method, exp['endurance'],
                                        exp['n_trucks'], exp['n_drones'])
                if sol is None:
                    print(f"      No feasible solution found")
                    continue

                tw_name = 'Tight TW' if exp['tw_type'] == 'RC1' else 'Wide TW'
                title = f"{method}: {exp['n_customers']}c {tw_name} ({exp['n_trucks']}T+{exp['n_drones']}D)\nCost={sol.cost:.1f}, Tardiness={sol.tardiness:.1f}"

                save_name = f"route_{method}_{label}.png"
                plot_route_map(inst, sol.truck_routes, sol.drone_missions,
                              title, os.path.join(OUTPUT_DIR, save_name), method)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════
#  PART 3: Main
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description='EVRP-TW Visualizations')
    parser.add_argument('--config', type=str, default=None,
                       help='Filter by config label, e.g. "25c" or "RC1"')
    parser.add_argument('--pareto-only', action='store_true')
    parser.add_argument('--routes-only', action='store_true')
    parser.add_argument('--quick', action='store_true',
                       help='Only 4 representative configs')
    args = parser.parse_args()

    # Load results
    results_path = os.path.join(RESULTS_DIR, 'results_20260702_152443_hv_fixed.json')
    if not os.path.exists(results_path):
        results_path = os.path.join(RESULTS_DIR, 'results_20260702_152443.json')
    with open(results_path) as f:
        results = json.load(f)
    print(f"Loaded {len(results)} experiment results")

    do_pareto = not args.routes_only
    do_routes = not args.pareto_only

    if args.quick:
        # 8 representative configs: 25c/50c × RC1/RC2 × medium
        subset = []
        seen = set()
        for exp in results:
            if 'medium' not in exp['endurance_name']:
                continue
            if exp['n_customers'] == 100:
                continue  # skip 100c for quick mode (slow P-ACO)
            key = (exp['n_customers'], exp['tw_type'])
            if key not in seen:
                seen.add(key)
                subset.append(exp)
        results = subset
        print(f"Quick mode: {len(results)} configs")

    if args.config:
        results = [r for r in results if args.config.lower() in r['label'].lower()]
        print(f"Filtered to {len(results)} configs")

    if do_pareto:
        generate_pareto_plots(results, args.config)

    if do_routes:
        generate_route_maps(results, args.config)

    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")
    print(f"Total files: {len(os.listdir(OUTPUT_DIR))}")


if __name__ == '__main__':
    main()
