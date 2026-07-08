#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 4 POMO-MT Visualization — Pareto Fronts + Petal Route Maps.

Generates:
  1. Pareto Front scatter plots (POMO-MT vs No-Drone, Cost vs Tardiness)
  2. Route maps (geographic layout with truck routes, petal pattern)

Usage:
    python visualize.py                     # All configs
    python visualize.py --quick             # Representative configs only
    python visualize.py --pareto-only       # Only Pareto fronts
    python visualize.py --routes-only       # Only route maps
"""

import os, sys, json, math, time, random
import importlib.util

# ── Path setup ────────────────────────────────────────────────────
_W4 = os.path.dirname(os.path.abspath(__file__))
_W3 = os.path.join(_W4, '..', 'week3')

# Strategy: week4 goes first so that `algorithms`, `utils`, `config` resolve
# to week4's versions (which are compatible copies/adapters). The No-Drone
# solver from week3 is loaded via importlib as a standalone module to avoid
# having week3's `algorithms` package pollute the import cache.
sys.path.insert(0, _W4)

# Load No-Drone solver from week3 as a standalone module.
# Using importlib avoids caching week3's `algorithms` package, which would
# shadow week4's `algorithms.pomo` sub-package.
_no_drone_spec = importlib.util.spec_from_file_location(
    "week3_no_drone",
    os.path.join(_W3, 'algorithms', 'no_drone.py')
)
_no_drone_mod = importlib.util.module_from_spec(_no_drone_spec)
sys.modules['week3_no_drone'] = _no_drone_mod
_no_drone_spec.loader.exec_module(_no_drone_mod)
solve_no_drone = _no_drone_mod.solve_no_drone

# Also load P-ACO, NSGA-II, IVND from week3 as standalone modules
_w3_algo_dir = os.path.join(_W3, 'algorithms')
for _name, _file in [('paco', 'paco.py'), ('nsga2', 'nsga2.py'), ('ivnd', 'ivnd.py')]:
    _spec = importlib.util.spec_from_file_location(
        f"week3_{_name}", os.path.join(_w3_algo_dir, _file)
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[f"week3_{_name}"] = _mod
    _spec.loader.exec_module(_mod)

PACOSolver = sys.modules['week3_paco'].PACOSolver
NSGA2Solver = sys.modules['week3_nsga2'].NSGA2Solver
IVNDSolver = sys.modules['week3_ivnd'].IVNDSolver

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
    RESULTS_DIR, DATA_OUT_DIR, DEPOT,
    RC1_INSTANCES, RC2_INSTANCES, CUSTOMER_SIZES,
    DRONE_ENDURANCE, VEHICLE_CONFIGS,
    TRUCK_CAPACITY, TRUCK_SPEED, BATTERY_CAPACITY,
)
from utils.data_loader import load_instance_from_disk
from utils.problem_model import TruckDroneSolution, extract_pareto_front
from pomo_multi_truck import run_pomo_multitruck, POMOMultiTruckSolver

OUTPUT_DIR = os.path.join(_W4, 'visualizations')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Color scheme ───────────────────────────────────────────────────
COLORS = {
    'POMO-MT': '#9b59b6',     # Purple
    'No-Drone': '#f39c12',    # Orange
}
MARKERS = {
    'POMO-MT': 'v',           # inverted triangle
    'No-Drone': 'D',          # diamond
}


# ═══════════════════════════════════════════════════════════════════
#  PART 1: Pareto Front Visualization
# ═══════════════════════════════════════════════════════════════════

def compute_pareto_fronts(exp):
    """
    Get Pareto points for POMO-MT and No-Drone for one config.
    - POMO-MT: from stored JSON (deduplicated)
    - No-Drone: re-run solver once to get fresh Pareto
    """
    inst = load_instance_from_disk(exp['instance_key'])
    points = {}

    # ── POMO-MT from stored JSON ──
    pomo_data = exp['methods'].get('POMO-MT', {})
    pomo_pts = pomo_data.get('pareto_points', [])
    if pomo_pts:
        # Deduplicate (deterministic output → all 10 repeats identical)
        seen = set()
        unique = []
        for (c, t) in pomo_pts:
            key = (round(c, 1), round(t, 1))
            if key not in seen:
                seen.add(key)
                unique.append((c, t))
        points['POMO-MT'] = unique
    else:
        # Fallback: use mean values
        mc = pomo_data.get('mean_cost', 0)
        mt = pomo_data.get('mean_tardiness', 0)
        points['POMO-MT'] = [(mc, mt)] if mc > 0 else []

    # ── No-Drone: re-run solver ──
    try:
        sols = solve_no_drone(inst, pop_size=80, generations=120, seed=42)
        pareto = extract_pareto_front(sols)
        points['No-Drone'] = [(s.cost, s.tardiness) for s in pareto if s.feasible]
        if not points['No-Drone']:
            # Fallback: any solution
            points['No-Drone'] = [(s.cost, s.tardiness) for s in pareto]
    except Exception as e:
        print(f"    No-Drone re-run error: {e}")
        # Use stored aggregate data as fallback
        nd = exp['methods'].get('No-Drone', {})
        if nd:
            points['No-Drone'] = [(nd.get('mean_cost', 0), nd.get('mean_tardiness', 0))]

    return points


def compute_joint_pareto(all_points):
    """Find the joint non-dominated front across all methods."""
    all_pts = []
    for pts in all_points.values():
        all_pts.extend(pts)
    if not all_pts:
        return []
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
    """Plot Pareto front scatter plot: POMO-MT vs No-Drone."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for method in ['No-Drone', 'POMO-MT']:
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


def generate_pareto_plots(results, quick=False):
    """Generate Pareto front plots for experiment configs."""
    print("\n" + "=" * 60)
    print("Generating Pareto Front Plots")
    print("=" * 60)

    if quick:
        # One config per (n_customers, tw_type) → 6 plots
        seen = set()
        selected = []
        for exp in results:
            key = (exp['n_customers'], exp['tw_type'], exp['source_instance'])
            if key not in seen:
                seen.add(key)
                selected.append(exp)
        configs = selected
        print(f"Quick mode: {len(configs)} configs")
    else:
        configs = results

    for exp in configs:
        label = exp['label']
        print(f"\n  {label}...")
        try:
            points = compute_pareto_fronts(exp)
            joint = compute_joint_pareto(points)

            end_name = f"{exp['endurance']}km"
            title = (f"Pareto Front: {exp['n_customers']}c {exp['tw_type']} "
                     f"{end_name} {exp['n_trucks']}T+{exp['n_drones']}D\n"
                     f"({exp['source_instance']})")

            save_name = f"pareto_{exp['source_instance']}_{label}.png"
            plot_pareto_front(points, title, os.path.join(OUTPUT_DIR, save_name), joint)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════
#  PART 2: Route Map Visualization (Petal Plots)
# ═══════════════════════════════════════════════════════════════════

def get_best_pomo_mt_solution(instance, n_trucks, seed=42):
    """Re-run POMO-MT and return best feasible solution with truck_routes."""
    result = run_pomo_multitruck(instance, n_runs=1, n_trucks=n_trucks,
                                 n_drones=0, endurance=None, seed=seed)
    solutions = result['solutions']
    best, best_cost = None, float('inf')
    for s in solutions:
        if s.feasible and s.cost < best_cost:
            best_cost = s.cost
            best = s
    if best is None and solutions:
        best = solutions[0]
    return best


def get_best_no_drone_solution(instance, seed=42):
    """Re-run No-Drone and return best feasible solution."""
    sols = solve_no_drone(instance, pop_size=80, generations=120, seed=seed)
    best, best_cost = None, float('inf')
    for s in sols:
        if s.feasible and s.cost < best_cost:
            best_cost = s.cost
            best = s
    if best is None and sols:
        best = sols[0]
    return best


def plot_route_map(instance, truck_routes, drone_missions, title, save_path,
                   method_name=''):
    """Plot geographic route map with truck routes (petal pattern)."""
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

    # Build customer coord lookup
    cust_coords = {c['id']: (c['x'], c['y']) for c in customers}

    # Plot truck routes (petal pattern from depot)
    n_routes = max(len(truck_routes), 1)
    truck_colors = plt.cm.Set1(np.linspace(0, 1, n_routes))
    for ri, route in enumerate(truck_routes):
        if not route:
            continue
        color = truck_colors[ri % len(truck_colors)]
        # Route = depot → customers → depot
        path = [depot]
        for cid in route:
            path.append(cust_coords.get(cid, depot))
        path.append(depot)
        px, py = zip(*path)
        label = f'Truck {ri+1}' if ri == 0 else None
        ax.plot(px, py, '-', color=color, linewidth=2, alpha=0.8,
                label=f'Truck {ri+1}')
        # Direction arrows at segment midpoints
        for i in range(len(path) - 1):
            mid_x = (path[i][0] + path[i+1][0]) / 2
            mid_y = (path[i][1] + path[i+1][1]) / 2
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            ax.annotate('', xy=(mid_x + dx*0.05, mid_y + dy*0.05),
                        xytext=(mid_x - dx*0.05, mid_y - dy*0.05),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.6))

    # Plot drone missions (dashed red lines) — only for No-Drone
    for mission in drone_missions:
        i, j, k = mission  # launch, customer, recovery
        launch_xy = depot if i == 0 else cust_coords.get(i, depot)
        cust_xy = cust_coords.get(j, depot)
        recov_xy = depot if k == 0 else cust_coords.get(k, depot)

        ax.plot([launch_xy[0], cust_xy[0]], [launch_xy[1], cust_xy[1]],
                '--', color='#e74c3c', linewidth=1.5, alpha=0.7)
        ax.plot([cust_xy[0], recov_xy[0]], [cust_xy[1], recov_xy[1]],
                '--', color='#e74c3c', linewidth=1.5, alpha=0.7)
        ax.scatter(cust_xy[0], cust_xy[1], c='#e74c3c', marker='*', s=120,
                   zorder=6, edgecolors='black', linewidth=0.5)

    # Custom legend
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
               markersize=10, label='Depot'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=8, label='Customer'),
        Line2D([0], [0], color=truck_colors[0], linewidth=2, label='Truck Route'),
    ]
    if drone_missions:
        legend_elements.extend([
            Line2D([0], [0], color='#e74c3c', linewidth=1.5, linestyle='--',
                   label='Drone Mission'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='#e74c3c',
                   markersize=10, label='Drone Customer'),
        ])
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


def generate_route_maps(results):
    """Generate petal-style route maps for representative configs."""
    print("\n" + "=" * 60)
    print("Generating Route Maps")
    print("=" * 60)

    # Select representative configs: one per (n_customers, tw_type, source_instance)
    # Prefer medium endurance, smallest fleet
    selected = []

    for n_cust in [25, 50, 100]:
        for tw in ['RC1', 'RC2']:
            for src in (RC1_INSTANCES if tw == 'RC1' else RC2_INSTANCES):
                # Find matching config with medium endurance, smallest fleet
                best = None
                for exp in results:
                    if (exp['n_customers'] == n_cust and
                        exp['tw_type'] == tw and
                        exp['source_instance'] == src and
                        'medium' in exp['endurance_name']):
                        if best is None or exp['n_trucks'] < best['n_trucks']:
                            best = exp
                if best:
                    selected.append(best)
                    break  # one per (scale, TW) — first source instance

    print(f"Selected {len(selected)} configs for route maps")

    for exp in selected:
        label = exp['label']
        print(f"\n  {label}...")
        try:
            inst = load_instance_from_disk(exp['instance_key'])
            n_t = exp['n_trucks']

            # ── POMO-MT route map ──
            print(f"    POMO-MT...")
            sol = get_best_pomo_mt_solution(inst, n_trucks=n_t, seed=42)
            if sol is not None:
                tw_name = 'Tight TW' if exp['tw_type'] == 'RC1' else 'Wide TW'
                title = (f"POMO-MT: {exp['n_customers']}c {tw_name} "
                         f"({exp['n_trucks']}T, {len(sol.truck_routes)} routes)\n"
                         f"Cost={sol.cost:.1f}, Tardiness={sol.tardiness:.1f}")
                save_name = f"route_POMO-MT_{label}.png"
                plot_route_map(inst, sol.truck_routes, sol.drone_missions,
                              title, os.path.join(OUTPUT_DIR, save_name), 'POMO-MT')
            else:
                print(f"      No solution found")

            # ── No-Drone route map ──
            print(f"    No-Drone...")
            sol_nd = get_best_no_drone_solution(inst, seed=42)
            if sol_nd is not None:
                tw_name = 'Tight TW' if exp['tw_type'] == 'RC1' else 'Wide TW'
                title = (f"No-Drone: {exp['n_customers']}c {tw_name} "
                         f"({exp['n_trucks']}T)\n"
                         f"Cost={sol_nd.cost:.1f}, Tardiness={sol_nd.tardiness:.1f}")
                save_name = f"route_No-Drone_{label}.png"
                plot_route_map(inst, sol_nd.truck_routes, sol_nd.drone_missions,
                              title, os.path.join(OUTPUT_DIR, save_name), 'No-Drone')
            else:
                print(f"      No solution found")
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════
#  PART 3: Summary Dashboard
# ═══════════════════════════════════════════════════════════════════

def generate_summary_plot(results):
    """Generate a summary bar chart comparing POMO-MT vs No-Drone."""
    print("\n" + "=" * 60)
    print("Generating Summary Dashboard")
    print("=" * 60)

    # Aggregate by (n_customers, tw_type)
    groups = defaultdict(lambda: {'POMO-MT': {'cost': [], 'tard': []},
                                   'No-Drone': {'cost': [], 'tard': []}})

    for exp in results:
        key = (exp['n_customers'], exp['tw_type'])
        for method in ['POMO-MT', 'No-Drone']:
            m = exp['methods'].get(method, {})
            if m:
                groups[key][method]['cost'].append(m.get('mean_cost', 0))
                groups[key][method]['tard'].append(m.get('mean_tardiness', 0))

    # Sort keys
    sorted_keys = sorted(groups.keys(), key=lambda k: (k[0], k[1]))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── Cost subplot ──
    ax = axes[0]
    x_labels = [f"{n}c\n{tw}" for n, tw in sorted_keys]
    x = np.arange(len(sorted_keys))
    width = 0.35

    pomo_costs = [np.mean(groups[k]['POMO-MT']['cost']) for k in sorted_keys]
    nd_costs = [np.mean(groups[k]['No-Drone']['cost']) for k in sorted_keys]
    pomo_costs_std = [np.std(groups[k]['POMO-MT']['cost']) for k in sorted_keys]
    nd_costs_std = [np.std(groups[k]['No-Drone']['cost']) for k in sorted_keys]

    bars1 = ax.bar(x - width/2, pomo_costs, width, yerr=pomo_costs_std,
                   label='POMO-MT', color=COLORS['POMO-MT'], edgecolor='black', linewidth=0.5,
                   capsize=4)
    bars2 = ax.bar(x + width/2, nd_costs, width, yerr=nd_costs_std,
                   label='No-Drone', color=COLORS['No-Drone'], edgecolor='black', linewidth=0.5,
                   capsize=4)

    # Annotate reduction %
    for i, (pc, nc) in enumerate(zip(pomo_costs, nd_costs)):
        if nc > 0:
            reduction = (nc - pc) / nc * 100
            ax.annotate(f'-{reduction:.0f}%', (x[i], pc),
                       textcoords="offset points", xytext=(0, -15),
                       ha='center', fontsize=8, fontweight='bold', color='#27ae60')

    ax.set_ylabel('Mean Travel Cost', fontsize=12)
    ax.set_title('Cost Comparison: POMO-MT vs No-Drone', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # ── Tardiness subplot ──
    ax = axes[1]
    pomo_tards = [np.mean(groups[k]['POMO-MT']['tard']) for k in sorted_keys]
    nd_tards = [np.mean(groups[k]['No-Drone']['tard']) for k in sorted_keys]
    pomo_tards_std = [np.std(groups[k]['POMO-MT']['tard']) for k in sorted_keys]
    nd_tards_std = [np.std(groups[k]['No-Drone']['tard']) for k in sorted_keys]

    # Use log scale for tardiness since values span orders of magnitude
    ax.bar(x - width/2, pomo_tards, width, yerr=pomo_tards_std,
           label='POMO-MT', color=COLORS['POMO-MT'], edgecolor='black', linewidth=0.5,
           capsize=4)
    ax.bar(x + width/2, nd_tards, width, yerr=nd_tards_std,
           label='No-Drone', color=COLORS['No-Drone'], edgecolor='black', linewidth=0.5,
           capsize=4)

    ax.set_ylabel('Mean Tardiness', fontsize=12)
    ax.set_title('Tardiness Comparison: POMO-MT vs No-Drone', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'summary_dashboard.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: summary_dashboard.png")


def generate_feasibility_plot(results):
    """Generate a runtime/feasibility comparison plot."""
    print("\nGenerating Feasibility & Runtime Plot")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by n_customers
    scales = sorted(set(exp['n_customers'] for exp in results))
    pomo_times = []
    nd_times = []
    pomo_time_std = []
    nd_time_std = []

    for sc in scales:
        pt = [exp['methods']['POMO-MT'].get('mean_runtime', 0)
              for exp in results if exp['n_customers'] == sc
              and 'POMO-MT' in exp['methods']]
        nt = [exp['methods']['No-Drone'].get('mean_runtime', 0)
              for exp in results if exp['n_customers'] == sc
              and 'No-Drone' in exp['methods']]
        pomo_times.append(np.mean(pt) if pt else 0)
        nd_times.append(np.mean(nt) if nt else 0)
        pomo_time_std.append(np.std(pt) if pt else 0)
        nd_time_std.append(np.std(nt) if nt else 0)

    x = np.arange(len(scales))
    width = 0.35

    ax.bar(x - width/2, pomo_times, width, yerr=pomo_time_std,
           label='POMO-MT', color=COLORS['POMO-MT'], edgecolor='black', linewidth=0.5,
           capsize=4)
    ax.bar(x + width/2, nd_times, width, yerr=nd_time_std,
           label='No-Drone', color=COLORS['No-Drone'], edgecolor='black', linewidth=0.5,
           capsize=4)

    # Annotate speedup
    for i, (pt, nt) in enumerate(zip(pomo_times, nd_times)):
        if nt > 0:
            speedup = nt / pt if pt > 0 else float('inf')
            label = f'{speedup:.1f}×' if speedup < 100 else '∞'
            ax.annotate(label, (x[i] - width/2, pt),
                       textcoords="offset points", xytext=(0, 10),
                       ha='center', fontsize=8, fontweight='bold', color='#8e44ad')

    ax.set_ylabel('Mean Runtime (seconds)', fontsize=12)
    ax.set_title('Runtime Comparison by Problem Scale', fontsize=14, fontweight='bold')
    ax.set_xlabel('Customer Count', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}c' for s in scales], fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'runtime_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: runtime_comparison.png")


# ═══════════════════════════════════════════════════════════════════
#  PART 4: All-Methods Comparison (POMO-MT vs Week 3 methods)
# ═══════════════════════════════════════════════════════════════════

# Extended color/marker scheme for all 5 methods
ALL_COLORS = {
    'POMO-MT':  '#9b59b6',    # Purple
    'P-ACO':    '#2ecc71',    # Green
    'NSGA-II':  '#3498db',    # Blue
    'IVND':     '#e74c3c',    # Red
    'No-Drone': '#f39c12',    # Orange
}
ALL_MARKERS = {
    'POMO-MT':  'v',          # inverted triangle
    'P-ACO':    'o',          # circle
    'NSGA-II':  's',          # square
    'IVND':     '^',          # triangle
    'No-Drone': 'D',          # diamond
}
METHOD_ORDER = ['POMO-MT', 'P-ACO', 'NSGA-II', 'IVND', 'No-Drone']


def load_week3_results():
    """Load Week 3 aggregate results for cross-method comparison."""
    w3_paths = [
        os.path.join(_W3, 'results', 'results_20260702_152443_hv_fixed.json'),
        os.path.join(_W3, 'results', 'results_20260702_152443.json'),
    ]
    for p in w3_paths:
        if os.path.exists(p):
            with open(p) as f:
                data = json.load(f)
            # Index by label
            indexed = {}
            for exp in data:
                indexed[exp['label']] = exp
            print(f"Loaded {len(indexed)} Week 3 results from {os.path.basename(p)}")
            return indexed
    print("WARNING: Week 3 results not found")
    return {}


def generate_all_methods_comparison(pomo_results):
    """Aggregate comparison: POMO-MT vs all 4 Week 3 methods using stored data."""
    print("\n" + "=" * 60)
    print("Generating All-Methods Aggregate Comparison")
    print("=" * 60)

    w3_data = load_week3_results()
    if not w3_data:
        print("  SKIP: No Week 3 data available")
        return

    # ── Aggregate by (n_customers, tw_type) ──
    groups = defaultdict(lambda: defaultdict(lambda: {'cost': [], 'tard': [], 'feas': [], 'runtime': [], 'hv': []}))

    for exp in pomo_results:
        key = (exp['n_customers'], exp['tw_type'])
        label = exp['label']

        # POMO-MT from pomo results
        pm = exp['methods'].get('POMO-MT', {})
        if pm:
            groups[key]['POMO-MT']['cost'].append(pm.get('mean_cost', 0))
            groups[key]['POMO-MT']['tard'].append(pm.get('mean_tardiness', 0))
            groups[key]['POMO-MT']['feas'].append(pm.get('feasibility_rate', 0) * 100)
            groups[key]['POMO-MT']['runtime'].append(pm.get('mean_runtime', 0))
            groups[key]['POMO-MT']['hv'].append(pm.get('hypervolume', 0))

        # Week 3 methods from stored data
        w3_exp = w3_data.get(label, {})
        for method in ['No-Drone', 'P-ACO', 'NSGA-II', 'IVND']:
            m = w3_exp.get('methods', {}).get(method, {})
            if m:
                groups[key][method]['cost'].append(m.get('mean_cost', 0))
                groups[key][method]['tard'].append(m.get('mean_tardiness', 0))
                groups[key][method]['feas'].append(m.get('feasibility_rate', 0) * 100)
                groups[key][method]['runtime'].append(m.get('mean_runtime', 0))
                groups[key][method]['hv'].append(m.get('hypervolume', 0))

    sorted_keys = sorted(groups.keys(), key=lambda k: (k[0], k[1]))
    x_labels = [f"{n}c\n{tw}" for n, tw in sorted_keys]
    methods_present = [m for m in METHOD_ORDER if any(groups[k][m]['cost'] for k in sorted_keys)]

    # ── Figure 1: Cost + Tardiness ──
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    n_methods = len(methods_present)
    x = np.arange(len(sorted_keys))
    width = 0.8 / n_methods

    # Cost subplot
    ax = axes[0]
    for mi, method in enumerate(methods_present):
        offset = (mi - (n_methods - 1) / 2) * width
        vals = [np.mean(groups[k][method]['cost']) for k in sorted_keys]
        stds = [np.std(groups[k][method]['cost']) for k in sorted_keys]
        ax.bar(x + offset, vals, width, yerr=stds,
               label=method, color=ALL_COLORS[method], edgecolor='black',
               linewidth=0.3, capsize=3, alpha=0.9)

    ax.set_ylabel('Mean Travel Cost', fontsize=12)
    ax.set_title('Cost Comparison: All Methods', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

    # Tardiness subplot
    ax = axes[1]
    for mi, method in enumerate(methods_present):
        offset = (mi - (n_methods - 1) / 2) * width
        vals = [np.mean(groups[k][method]['tard']) for k in sorted_keys]
        stds = [np.std(groups[k][method]['tard']) for k in sorted_keys]
        ax.bar(x + offset, vals, width, yerr=stds,
               label=method, color=ALL_COLORS[method], edgecolor='black',
               linewidth=0.3, capsize=3, alpha=0.9)

    ax.set_ylabel('Mean Tardiness', fontsize=12)
    ax.set_title('Tardiness Comparison: All Methods', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'all_methods_cost_tardiness.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: all_methods_cost_tardiness.png")

    # ── Figure 2: Feasibility + Runtime ──
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Feasibility subplot
    ax = axes[0]
    for mi, method in enumerate(methods_present):
        offset = (mi - (n_methods - 1) / 2) * width
        vals = [np.mean(groups[k][method]['feas']) for k in sorted_keys]
        ax.bar(x + offset, vals, width,
               label=method, color=ALL_COLORS[method], edgecolor='black',
               linewidth=0.3, alpha=0.9)
        # Annotate values below 100%
        for ki, k in enumerate(sorted_keys):
            v = np.mean(groups[k][method]['feas'])
            if v < 99:
                ax.annotate(f'{v:.0f}%', (x[ki] + offset, v + 1),
                           ha='center', fontsize=6, fontweight='bold', color=ALL_COLORS[method])

    ax.set_ylabel('Feasibility Rate (%)', fontsize=12)
    ax.set_title('Feasibility Comparison: All Methods', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

    # Runtime subplot (log scale due to P-ACO's extreme runtime)
    ax = axes[1]
    for mi, method in enumerate(methods_present):
        offset = (mi - (n_methods - 1) / 2) * width
        vals = [max(np.mean(groups[k][method]['runtime']), 0.001) for k in sorted_keys]
        ax.bar(x + offset, vals, width,
               label=method, color=ALL_COLORS[method], edgecolor='black',
               linewidth=0.3, alpha=0.9)

    ax.set_ylabel('Mean Runtime (s, log scale)', fontsize=12)
    ax.set_title('Runtime Comparison: All Methods', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_yscale('log')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'all_methods_feasibility_runtime.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: all_methods_feasibility_runtime.png")

    # ── Figure 3: Hypervolume (multi-objective quality) ──
    fig, ax = plt.subplots(figsize=(12, 7))
    for mi, method in enumerate(methods_present):
        offset = (mi - (n_methods - 1) / 2) * width
        vals = [np.mean(groups[k][method]['hv']) / 1e6 for k in sorted_keys]
        stds = [np.std(groups[k][method]['hv']) / 1e6 for k in sorted_keys]
        ax.bar(x + offset, vals, width, yerr=stds,
               label=method, color=ALL_COLORS[method], edgecolor='black',
               linewidth=0.3, capsize=3, alpha=0.9)

    ax.set_ylabel('Hypervolume (×10⁶)', fontsize=12)
    ax.set_title('Multi-Objective Quality (Hypervolume): All Methods', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.legend(fontsize=10, ncol=3)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'all_methods_hypervolume.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: all_methods_hypervolume.png")

    # ── Figure 4: Trade-off Radar (Cost vs Tardiness scatter per method) ──
    fig, ax = plt.subplots(figsize=(10, 8))
    for method in methods_present:
        xs = [np.mean(groups[k][method]['cost']) for k in sorted_keys]
        ys = [np.mean(groups[k][method]['tard']) for k in sorted_keys]
        ax.scatter(xs, ys, c=ALL_COLORS[method], marker=ALL_MARKERS[method],
                   label=method, alpha=0.8, s=80, edgecolors='black', linewidth=0.5)
        # Annotate each point with scale+TW
        for ki, k in enumerate(sorted_keys):
            ax.annotate(f"{k[0]}c\n{k[1]}", (xs[ki], ys[ki]),
                       textcoords="offset points", xytext=(5, 5),
                       fontsize=6, alpha=0.8)

    ax.set_xlabel('Mean Travel Cost', fontsize=12)
    ax.set_ylabel('Mean Tardiness', fontsize=12)
    ax.set_title('Objective Space: All Methods (mean values)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'all_methods_objective_space.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: all_methods_objective_space.png")


# ═══════════════════════════════════════════════════════════════════
#  PART 5: Full 5-Method Pareto Scatter (re-runs solvers, slower)
# ═══════════════════════════════════════════════════════════════════

def compute_all_methods_pareto(exp):
    """Run all 5 methods on one config to get Pareto scatter points.
    POMO-MT from stored JSON; others re-run once."""
    inst = load_instance_from_disk(exp['instance_key'])
    endurance = exp['endurance']
    n_t, n_d = exp['n_trucks'], exp['n_drones']
    points = {}

    # ── POMO-MT from stored JSON ──
    pomo_pts = exp['methods'].get('POMO-MT', {}).get('pareto_points', [])
    if pomo_pts:
        seen = set()
        unique = []
        for (c, t) in pomo_pts:
            key = (round(c, 1), round(t, 1))
            if key not in seen:
                seen.add(key)
                unique.append((c, t))
        points['POMO-MT'] = unique

    # ── No-Drone ──
    try:
        sols = solve_no_drone(inst, pop_size=80, generations=120, seed=42)
        pareto = extract_pareto_front(sols)
        points['No-Drone'] = [(s.cost, s.tardiness) for s in pareto if s.feasible] or \
                              [(s.cost, s.tardiness) for s in pareto]
    except Exception as e:
        print(f"      No-Drone error: {e}")
        points['No-Drone'] = []

    # ── P-ACO ──
    try:
        solver = PACOSolver(inst, seed=42)
        sols, pareto = solver.solve(endurance=endurance)
        points['P-ACO'] = [(s.cost, s.tardiness) for s in pareto if s.feasible] or \
                           [(s.cost, s.tardiness) for s in pareto]
    except Exception as e:
        print(f"      P-ACO error: {e}")
        points['P-ACO'] = []

    # ── NSGA-II ──
    try:
        solver = NSGA2Solver(inst, n_trucks=n_t, n_drones=n_d, endurance=endurance, seed=42)
        sols, pareto = solver.solve()
        points['NSGA-II'] = [(s.cost, s.tardiness) for s in pareto if s.feasible] or \
                             [(s.cost, s.tardiness) for s in pareto]
    except Exception as e:
        print(f"      NSGA-II error: {e}")
        points['NSGA-II'] = []

    # ── IVND ──
    try:
        solver = IVNDSolver(inst, n_trucks=n_t, n_drones=n_d, endurance=endurance, seed=42)
        sols, pareto = solver.solve()
        points['IVND'] = [(s.cost, s.tardiness) for s in pareto if s.feasible] or \
                          [(s.cost, s.tardiness) for s in pareto]
    except Exception as e:
        print(f"      IVND error: {e}")
        points['IVND'] = []

    return points


def plot_full_pareto(points_dict, title, save_path):
    """Plot 5-method Pareto scatter with joint front."""
    fig, ax = plt.subplots(figsize=(12, 8))

    for method in METHOD_ORDER:
        pts = points_dict.get(method, [])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.scatter(xs, ys, c=ALL_COLORS[method], marker=ALL_MARKERS[method],
                   label=method, alpha=0.7, s=35, edgecolors='black', linewidth=0.3)

    # Joint Pareto front
    all_pts = []
    for pts in points_dict.values():
        all_pts.extend(pts)
    if all_pts:
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
        if len(nondom) > 1:
            xs = [p[0] for p in nondom]
            ys = [p[1] for p in nondom]
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


def generate_full_pareto_plots(pomo_results):
    """Generate 5-method Pareto plots for representative configs.
    Re-runs all solvers — slower but shows actual scatter distributions."""
    print("\n" + "=" * 60)
    print("Generating Full 5-Method Pareto Plots (re-running solvers)")
    print("=" * 60)

    # Select representative configs: one per (n_customers, tw_type, source_instance)
    # Limit to 25c and 50c (skip 100c — P-ACO takes ~16 min per 100c config)
    selected = []
    seen = set()
    for exp in pomo_results:
        if exp['n_customers'] >= 100:
            continue
        key = (exp['n_customers'], exp['tw_type'], exp['source_instance'])
        if key not in seen and 'medium' in exp['endurance_name']:
            seen.add(key)
            selected.append(exp)

    # Further limit: one per (n_customers, tw_type) → 4 configs max
    seen2 = set()
    limited = []
    for exp in selected:
        key = (exp['n_customers'], exp['tw_type'])
        if key not in seen2:
            seen2.add(key)
            limited.append(exp)

    print(f"Selected {len(limited)} configs: 25c/50c × RC1/RC2")
    print("Estimated time: ~5-8 minutes (P-ACO is slowest at ~2 min per 50c config)")

    for exp in limited:
        label = exp['label']
        print(f"\n  [{label}] Running all 5 methods...")
        try:
            points = compute_all_methods_pareto(exp)
            end_name = f"{exp['endurance']}km"
            title = (f"Pareto Front: {exp['n_customers']}c {exp['tw_type']} "
                     f"{end_name} {exp['n_trucks']}T+{exp['n_drones']}D\n"
                     f"({exp['source_instance']}) — All Methods")

            save_name = f"full_pareto_{exp['source_instance']}_{label}.png"
            plot_full_pareto(points, title, os.path.join(OUTPUT_DIR, save_name))

            # Report point counts
            counts = ', '.join(f"{m}: {len(pts)} pts" for m, pts in points.items() if pts)
            print(f"    → {counts}")
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════
#  PART 6: Main
# ═══════════════════════════════════════════════════════════════════

def find_results_file():
    """Auto-detect the most recent POMO-MT results JSON."""
    results_dir = os.path.join(_W4, 'results')
    if not os.path.isdir(results_dir):
        # Fall back to week3 results dir
        results_dir = os.path.join(_W3, 'results')

    candidates = []
    for f in os.listdir(results_dir):
        if f.startswith('pomo_multitruck_') and f.endswith('.json'):
            candidates.append(os.path.join(results_dir, f))

    if candidates:
        return max(candidates, key=os.path.getmtime)
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Week 4 POMO-MT Visualizations')
    parser.add_argument('--pareto-only', action='store_true',
                       help='Only generate Pareto front plots')
    parser.add_argument('--routes-only', action='store_true',
                       help='Only generate route maps')
    parser.add_argument('--quick', action='store_true',
                       help='Only representative configs (faster)')
    parser.add_argument('--full', action='store_true',
                       help='Generate full 5-method Pareto plots (re-runs solvers, slower)')
    parser.add_argument('--all-methods', action='store_true',
                       help='Generate all-methods aggregate comparison charts')
    args = parser.parse_args()

    # ── Load results ──
    results_path = find_results_file()
    if results_path is None:
        # Fallback: try loading from week3 results dir
        results_path = os.path.join(_W3, 'results', 'pomo_multitruck_20260707_152155.json')

    if not os.path.exists(results_path):
        print(f"ERROR: No results file found. Run experiments first.")
        print(f"Tried: {results_path}")
        sys.exit(1)

    with open(results_path) as f:
        results = json.load(f)
    print(f"Loaded {len(results)} experiment results from:")
    print(f"  {results_path}")

    # ── Verify data integrity ──
    n_pomo = sum(1 for r in results if 'POMO-MT' in r.get('methods', {}))
    n_nd = sum(1 for r in results if 'No-Drone' in r.get('methods', {}))
    print(f"  POMO-MT: {n_pomo}/{len(results)} configs")
    print(f"  No-Drone: {n_nd}/{len(results)} configs")

    # ── Generate visualizations ──
    do_pareto = not args.routes_only
    do_routes = not args.pareto_only

    if do_pareto:
        generate_pareto_plots(results, quick=args.quick)
        # Summary plots (always generate these if doing Pareto)
        try:
            generate_summary_plot(results)
            generate_feasibility_plot(results)
        except Exception as e:
            print(f"  Summary plots error: {e}")

    if do_routes:
        generate_route_maps(results)

    # ── All-methods aggregate comparison (always run unless routes-only) ──
    if do_pareto:
        try:
            generate_all_methods_comparison(results)
        except Exception as e:
            print(f"  All-methods comparison error: {e}")
            import traceback
            traceback.print_exc()

    # ── Full 5-method Pareto scatter (opt-in, re-runs solvers) ──
    if args.full:
        try:
            generate_full_pareto_plots(results)
        except Exception as e:
            print(f"  Full Pareto error: {e}")
            import traceback
            traceback.print_exc()

    # ── Report ──
    files = os.listdir(OUTPUT_DIR)
    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {OUTPUT_DIR}")
    print(f"Total files: {len(files)}")
    for f in sorted(files):
        fpath = os.path.join(OUTPUT_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {f} ({size_kb:.0f} KB)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
