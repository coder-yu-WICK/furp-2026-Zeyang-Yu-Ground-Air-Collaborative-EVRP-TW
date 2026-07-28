#!/usr/bin/env python3
"""
Publication-Quality Route Map Visualizations for FURP 2026 Paper.

Generates geographic route maps showing:
  - Customer distribution with depot
  - Truck routes (colored by route, with direction arrows)
  - Drone missions (dashed lines: launch → customer → recovery)
  - Drone launch/recovery points and drone-served customers
  - Charging stations (for EV models)
  - Time window context (tight vs wide)

Produces 4 route map figures:
  fig7a: RC-type — No-Drone vs 2-Drone comparison (RC101_50c)
  fig7b: R-type — No-Drone vs 2-Drone comparison (R101_50c)
  fig7c: C-type — clustered customer pattern (C101_50c)
  fig7d: EV Model B — charging station visits (RC101_50c, stress params)
"""

import os, sys, json, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Add parent for imports (following week7 convention)
_W7 = os.path.dirname(os.path.abspath(__file__))
for p in [os.path.join(_W7, '..', 'week3'),
          os.path.join(_W7, '..', 'week4'),
          os.path.join(_W7, '..', 'week5'),
          os.path.join(_W7, '..', 'week6'),
          _W7]:
    if p not in sys.path:
        sys.path.insert(0, p)

OUTPUT_DIR = os.path.join(_W7, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Publication Style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# ── Color Palette ─────────────────────────────────────────────────────────────
TRUCK_COLORS = ['#0072B2', '#E69F00', '#009E73', '#D55E00',
                '#CC79A7', '#56B4E9', '#F0E442', '#000000']
DRONE_COLOR = '#CC3333'       # red for drone missions
DRONE_CUST_COLOR = '#FF6666'  # light red for drone customers
DEPOT_COLOR = '#000000'       # black depot
CS_COLOR = '#33CC33'          # green for charging stations
CUST_COLOR = '#AAAAAA'        # grey for regular customers
LAUNCH_COLOR = '#FF9933'      # orange for launch points
RECOVERY_COLOR = '#9933FF'     # purple for recovery points


def load_instance(name):
    """Load a Solomon instance from week3/data/."""
    from utils.data_loader import load_instance_from_disk
    return load_instance_from_disk(name)


def get_routes(instance, n_trucks, max_drones_per_truck, seed=42,
               tw_type=None):
    """
    Run the pipeline and return the best solution's routes.
    Returns: (truck_routes, drone_missions, cost, tardiness, feasible)
    """
    from pipeline import solve_with_repair

    # Determine variant based on tw_type
    if tw_type and tw_type.startswith('C'):
        variant = 'cw_savings'  # CW-Savings for C-type
    elif tw_type and tw_type.startswith('R1'):
        variant = 'cw_savings'  # CW-Savings for R1 tight TW
    else:
        variant = 'hybrid'

    result = solve_with_repair(
        instance, n_trucks=n_trucks,
        variant=variant,
        use_repair=True,
        repair_mode='full',
        n_runs=1,
        seed=seed,
        max_drones_per_truck=max_drones_per_truck,
    )

    if not result['solutions']:
        return None, None, 0, 0, False

    # Take best feasible solution (lowest cost among feasible)
    best = None
    best_cost = float('inf')
    for s in result['solutions']:
        if s.feasible and s.cost < best_cost:
            best_cost = s.cost
            best = s
    if best is None:
        best = min(result['solutions'], key=lambda s: s.cost)

    return (best.truck_routes, best.drone_missions,
            best.cost, best.tardiness, best.feasible)


def get_ev_routes(instance, n_trucks, battery_capacity=30.0,
                  energy_rate=2.0, charging_model='linear', seed=42):
    """
    Run EV pipeline with charging station insertion.
    Returns: (truck_routes_with_cs, drone_missions, cost, ev_feasible,
              n_charges, charge_time)
    """
    from pipeline import solve_with_repair
    from ev_problem_model import (
        insert_charging_stops, simulate_route_ev, EVTruckDroneSolution
    )
    from utils.problem_model import TruckDroneSolution

    # Step 1: Get truck-only routes (no drones for EV study)
    result = solve_with_repair(
        instance, n_trucks=n_trucks,
        variant='cw_savings',  # compact routes for EV
        use_repair=True,
        repair_mode='full',
        n_runs=1,
        seed=seed,
        max_drones_per_truck=0,  # no drones
    )

    if not result['solutions']:
        return None, None, 0, False, 0, 0

    best = min(result['solutions'], key=lambda s: s.cost)
    truck_routes = best.truck_routes

    # Step 2: Insert charging stops
    from config import CHARGING_STATIONS, BATTERY_CAPACITY
    import numpy as np

    customers = instance['customers']
    n_cust = len(customers)

    # Build distance matrix
    coords = [(instance['depot'][0], instance['depot'][1])]
    for c in customers:
        coords.append((c['x'], c['y']))
    for cs in CHARGING_STATIONS:
        coords.append((cs[0], cs[1]))

    n_all = len(coords)
    dist = np.zeros((n_all, n_all))
    for i in range(n_all):
        for j in range(n_all):
            dist[i, j] = np.sqrt(
                (coords[i][0] - coords[j][0])**2 +
                (coords[i][1] - coords[j][1])**2
            )

    # Build CS coords dict: node_id -> (x, y)
    cs_coords_dict = {}
    for i, cs in enumerate(CHARGING_STATIONS):
        cs_coords_dict[n_cust + i + 1] = (cs[0], cs[1])

    # Insert charging stops
    routes_with_cs, cs_stats = insert_charging_stops(
        truck_routes, customers, dist, instance,
        battery_capacity=battery_capacity,
        energy_rate=energy_rate,
    )
    total_charge_time = cs_stats.get('total_charge_time', 0)
    total_charges = cs_stats.get('total_charges', 0)

    # Step 3: Simulate EV
    ev_feasible = True
    total_energy_vio = 0.0
    all_states = []
    for route in routes_with_cs:
        result = simulate_route_ev(
            route, customers, dist, cs_coords_dict, instance['depot'],
            battery_capacity=battery_capacity,
            energy_rate=energy_rate,
            charging_model=charging_model,
        )
        all_states.append(result)
        if result.get('energy_violation', 0) > 0:
            ev_feasible = False
            total_energy_vio += result['energy_violation']

    return (routes_with_cs, [], best.cost, ev_feasible,
            total_charges, total_charge_time)


def plot_route_map(instance, truck_routes, drone_missions, title, save_path,
                   charging_stations=None, ev_routes=False,
                   highlight_nodes=None, tw_annotations=False):
    """
    Publication-quality geographic route map.

    Args:
        instance: problem instance dict with 'customers' and 'depot'
        truck_routes: list of lists of customer IDs (including CS node IDs for EV)
        drone_missions: list of (i, j, k) tuples
        title: figure title
        save_path: output PNG path
        charging_stations: list of (x, y) tuples for CS locations
        ev_routes: if True, truck routes may contain CS node IDs > n_customers
        highlight_nodes: dict mapping node_id -> {'color': ..., 'marker': ..., 'label': ...}
        tw_annotations: if True, annotate customers with ready_time/due_time
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    customers = instance['customers']
    depot = instance['depot']
    n_customers = len(customers)

    # Build coordinate lookup
    cust_coords = {c['id']: (c['x'], c['y']) for c in customers}
    # CS nodes have IDs > n_customers
    cs_coords_lookup = {}
    if charging_stations:
        for i, cs in enumerate(charging_stations):
            cs_id = n_customers + i + 1
            cs_coords_lookup[cs_id] = cs

    # ── Plot depot ──
    ax.scatter(depot[0], depot[1], c=DEPOT_COLOR, marker='s', s=180,
               zorder=10, edgecolors='white', linewidth=1.5, label='Depot')
    ax.annotate('Depot', (depot[0], depot[1]),
                textcoords="offset points", xytext=(8, 8),
                fontsize=9, fontweight='bold', color=DEPOT_COLOR)

    # ── Plot customers ──
    cx = [c['x'] for c in customers]
    cy = [c['y'] for c in customers]
    ax.scatter(cx, cy, c=CUST_COLOR, marker='o', s=50, zorder=3,
               edgecolors='white', linewidth=0.5, alpha=0.8)

    # ── Plot charging stations (if any) ──
    if charging_stations:
        csx = [cs[0] for cs in charging_stations]
        csy = [cs[1] for cs in charging_stations]
        ax.scatter(csx, csy, c=CS_COLOR, marker='P', s=200, zorder=8,
                   edgecolors='white', linewidth=1.5, label='Charging Station')
        for i, (x, y) in enumerate(charging_stations):
            ax.annotate(f'CS{i+1}', (x, y),
                       textcoords="offset points", xytext=(5, -12),
                       fontsize=8, fontweight='bold', color=CS_COLOR)

    # ── Plot truck routes ──
    n_routes = max(len(truck_routes), 1)
    route_colors = TRUCK_COLORS[:n_routes] if n_routes <= len(TRUCK_COLORS) else \
                   plt.cm.tab20(np.linspace(0, 1, n_routes))

    for ri, route in enumerate(truck_routes):
        if not route:
            continue
        color = route_colors[ri % len(route_colors)]

        # Build path: depot → nodes → depot
        path = [(depot[0], depot[1])]
        for node_id in route:
            if node_id in cust_coords:
                path.append(cust_coords[node_id])
            elif node_id in cs_coords_lookup:
                path.append(cs_coords_lookup[node_id])
            # skip unknown nodes
        path.append((depot[0], depot[1]))

        px, py = zip(*path)
        ax.plot(px, py, '-', color=color, linewidth=2.5, alpha=0.85,
                label=f'Truck {ri+1}', zorder=4)

        # Direction arrows at segment midpoints
        for i in range(len(path) - 1):
            mid_x = (path[i][0] + path[i+1][0]) / 2
            mid_y = (path[i][1] + path[i+1][1]) / 2
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            seg_len = np.sqrt(dx**2 + dy**2)
            if seg_len > 0.01:
                dx_n, dy_n = dx/seg_len, dy/seg_len
                ax.annotate('', xy=(mid_x + dx_n*0.15, mid_y + dy_n*0.15),
                           xytext=(mid_x - dx_n*0.15, mid_y - dy_n*0.15),
                           arrowprops=dict(arrowstyle='->', color=color,
                                          lw=1.5, alpha=0.7), zorder=5)

    # ── Plot drone missions ──
    drone_launch_nodes = set()
    drone_recovery_nodes = set()
    drone_customers = set()

    for mission in drone_missions:
        if len(mission) >= 4:
            i, j, k, drone_id = mission[:4]  # launch, customer, recovery, drone_id
        else:
            i, j, k = mission  # launch, customer, recovery

        launch_xy = (depot[0], depot[1]) if i == 0 else cust_coords.get(i)
        cust_xy = cust_coords.get(j)
        recov_xy = (depot[0], depot[1]) if k == 0 else cust_coords.get(k)

        if launch_xy is None or cust_xy is None or recov_xy is None:
            continue

        drone_launch_nodes.add(i)
        drone_recovery_nodes.add(k)
        drone_customers.add(j)

        # Launch → customer (dashed)
        ax.plot([launch_xy[0], cust_xy[0]], [launch_xy[1], cust_xy[1]],
                '--', color=DRONE_COLOR, linewidth=1.8, alpha=0.8, zorder=5)
        # Customer → recovery (dashed)
        ax.plot([cust_xy[0], recov_xy[0]], [cust_xy[1], recov_xy[1]],
                '--', color=DRONE_COLOR, linewidth=1.8, alpha=0.8, zorder=5)

    # ── Mark drone-served customers ──
    for cid in drone_customers:
        xy = cust_coords.get(cid)
        if xy:
            ax.scatter(xy[0], xy[1], c=DRONE_CUST_COLOR, marker='*', s=180,
                       zorder=9, edgecolors='darkred', linewidth=0.8)

    # ── Mark launch points (if they are customers) ──
    for nid in drone_launch_nodes:
        if nid == 0:
            continue  # depot already marked
        xy = cust_coords.get(nid)
        if xy and nid not in drone_customers:
            ax.scatter(xy[0], xy[1], c=LAUNCH_COLOR, marker='D', s=90,
                       zorder=8, edgecolors='white', linewidth=0.5, alpha=0.8)

    # ── Mark recovery points ──
    for nid in drone_recovery_nodes:
        if nid == 0:
            continue
        xy = cust_coords.get(nid)
        if xy and nid not in drone_customers and nid not in drone_launch_nodes:
            ax.scatter(xy[0], xy[1], c=RECOVERY_COLOR, marker='s', s=90,
                       zorder=8, edgecolors='white', linewidth=0.5, alpha=0.8)

    # ── Legend ──
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=DEPOT_COLOR,
               markersize=10, label='Depot'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CUST_COLOR,
               markersize=8, label='Customer'),
        Line2D([0], [0], color=route_colors[0], linewidth=2.5,
               label='Truck Route'),
    ]
    if drone_missions:
        legend_elements.extend([
            Line2D([0], [0], color=DRONE_COLOR, linewidth=1.8, linestyle='--',
                   label='Drone Flight'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor=DRONE_CUST_COLOR,
                   markersize=12, markeredgecolor='darkred', markeredgewidth=0.8,
                   label='Drone Customer'),
        ])
        if drone_launch_nodes - {0}:
            legend_elements.append(
                Line2D([0], [0], marker='D', color='w', markerfacecolor=LAUNCH_COLOR,
                       markersize=8, label='Launch Point'))
        if drone_recovery_nodes - {0}:
            legend_elements.append(
                Line2D([0], [0], marker='s', color='w', markerfacecolor=RECOVERY_COLOR,
                       markersize=8, label='Recovery Point'))
    if charging_stations:
        legend_elements.append(
            Line2D([0], [0], marker='P', color='w', markerfacecolor=CS_COLOR,
                   markersize=12, label='Charging Station'))

    ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
              framealpha=0.9, edgecolor='gray')

    # ── Labels and styling ──
    ax.set_xlabel('X (km)', fontsize=11)
    ax.set_ylabel('Y (km)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)

    # Compute dynamic bounds with padding
    all_x = [depot[0]] + cx
    all_y = [depot[1]] + cy
    if charging_stations:
        all_x.extend(cs[0] for cs in charging_stations)
        all_y.extend(cs[1] for cs in charging_stations)

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    pad = max((x_max - x_min), (y_max - y_min)) * 0.08
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15, linestyle='--')

    plt.tight_layout()
    fig.savefig(save_path, facecolor='white')
    plt.close(fig)
    print(f'  Saved: {os.path.basename(save_path)}')


def generate_all_route_maps():
    """
    Generate route map figures for the paper:
      - fig7a_no_drone: RC101_50c truck-only
      - fig7b_2drone: RC101_50c with 2 drones
      - fig7c_r_type: R101_50c with 2 drones
      - fig7d_c_type: C101_50c with 2 drones
      - fig7e_ev: RC101_50c EV Model B with charging stations
    """
    from config import CHARGING_STATIONS

    print('=' * 60)
    print('GENERATING ROUTE MAP FIGURES')
    print('=' * 60)

    # ── Representative instances ──
    instances = [
        ('RC101_50c', 'RC1', 4, 'RC101 (Tight TW)'),
        ('R101_50c', 'R1', 4, 'R101 (Random, Tight TW)'),
        ('C101_50c', 'C1', 4, 'C101 (Clustered, Tight TW)'),
        ('RC201_50c', 'RC2', 2, 'RC201 (Wide TW)'),
    ]

    for inst_name, tw_type, n_trucks, label in instances:
        print(f'\n--- {inst_name} ({label}) ---')

        try:
            instance = load_instance(inst_name)
        except Exception as e:
            print(f'  SKIP: Cannot load {inst_name}: {e}')
            continue

        n_customers = len(instance['customers'])
        print(f'  Loaded: {n_customers} customers')

        # ── Fig A: No-Drone (truck only) ──
        print(f'  Running: No-Drone...')
        t0 = time.time()
        routes_nd, drones_nd, cost_nd, tard_nd, feas_nd = get_routes(
            instance, n_trucks, max_drones_per_truck=0, tw_type=tw_type)
        print(f'    Cost={cost_nd:.0f}, Feas={feas_nd}, Time={time.time()-t0:.1f}s')

        if routes_nd:
            title_nd = (f'{label}: Truck-Only (No Drones)\n'
                       f'{n_trucks} trucks, Cost={cost_nd:.0f}, '
                       f'Feasible={"Yes" if feas_nd else "No"}')
            plot_route_map(
                instance, routes_nd, [], title_nd,
                os.path.join(OUTPUT_DIR, f'fig_route_nd_{inst_name}.png'))

        # ── Fig B: 2-Drone (full pipeline) ──
        print(f'  Running: 2-Drone...')
        t0 = time.time()
        routes_2d, drones_2d, cost_2d, tard_2d, feas_2d = get_routes(
            instance, n_trucks, max_drones_per_truck=2, tw_type=tw_type)
        print(f'    Cost={cost_2d:.0f}, n_drones={len(drones_2d)}, '
              f'Feas={feas_2d}, Time={time.time()-t0:.1f}s')

        if routes_2d:
            drone_save = ((cost_nd - cost_2d) / cost_nd * 100) if cost_nd > 0 else 0
            title_2d = (f'{label}: 2 Drones/Truck\n'
                       f'{n_trucks} trucks, {len(drones_2d)} drone missions, '
                       f'Cost={cost_2d:.0f}\n'
                       f'Drone Savings: {drone_save:.1f}% vs Truck-Only')
            plot_route_map(
                instance, routes_2d, drones_2d, title_2d,
                os.path.join(OUTPUT_DIR, f'fig_route_2drone_{inst_name}.png'))

        # ── Fig C: 1-Drone comparison ──
        if tw_type in ('RC1', 'RC2'):  # Only for RC types
            print(f'  Running: 1-Drone...')
            routes_1d, drones_1d, cost_1d, tard_1d, feas_1d = get_routes(
                instance, n_trucks, max_drones_per_truck=1, tw_type=tw_type)
            print(f'    Cost={cost_1d:.0f}, n_drones={len(drones_1d)}, Feas={feas_1d}')

            if routes_1d:
                title_1d = (f'{label}: 1 Drone/Truck\n'
                           f'{n_trucks} trucks, {len(drones_1d)} drone missions, '
                           f'Cost={cost_1d:.0f}')
                plot_route_map(
                    instance, routes_1d, drones_1d, title_1d,
                    os.path.join(OUTPUT_DIR, f'fig_route_1drone_{inst_name}.png'))

    # ── Fig D: EV Model B with charging stations ──
    print(f'\n--- EV Charging Route Maps ---')
    ev_instance_name = 'RC101_50c'
    try:
        instance = load_instance(ev_instance_name)
        print(f'  Loaded: {ev_instance_name} ({len(instance["customers"])} customers)')

        # Use stress params to force CS visits
        for batt, erate in [(25, 2.0), (30, 2.0)]:
            print(f'  Running EV: battery={batt}kWh, energy_rate={erate}kWh/km...')
            t0 = time.time()
            routes_ev, drones_ev, cost_ev, ev_feas, n_cs, cs_time = get_ev_routes(
                instance, n_trucks=3, battery_capacity=batt,
                energy_rate=erate, charging_model='linear')
            print(f'    CS visits={n_cs}, charge_time={cs_time:.1f}min, '
                  f'EV-feas={ev_feas}, Time={time.time()-t0:.1f}s')

            if routes_ev:
                cs_list = [(8.0, 8.0), (4.0, 12.0), (12.0, 4.0)]  # from config
                title_ev = (f'EV Model B (Linear Charging): {ev_instance_name}\n'
                           f'Battery={batt}kWh, Energy={erate}kWh/km, '
                           f'3 Trucks\n'
                           f'{n_cs} charging stops, {cs_time:.0f} min total charge time, '
                           f'EV-Feasible={"Yes" if ev_feas else "No"}')
                plot_route_map(
                    instance, routes_ev, [], title_ev,
                    os.path.join(OUTPUT_DIR,
                               f'fig_route_ev_{ev_instance_name}_batt{batt}.png'),
                    charging_stations=cs_list, ev_routes=True)

    except Exception as e:
        print(f'  EV route map failed: {e}')
        import traceback
        traceback.print_exc()

    print(f'\n{"=" * 60}')
    print(f'All figures saved to: {OUTPUT_DIR}/')
    print('Done.')


def generate_panel_figure():
    """
    Generate a 2×2 panel figure comparing route structures across instance types.
    Panel layout:
      top-left:  RC101 (mixed, tight TW) — 2-Drone
      top-right: R101  (random, tight TW) — 2-Drone
      bot-left:  C101  (clustered, tight TW) — 2-Drone
      bot-right: RC101  — EV Model B with charging stations
    """
    from config import CHARGING_STATIONS

    print('\n' + '=' * 60)
    print('GENERATING 2×2 PANEL FIGURE')
    print('=' * 60)

    configs = [
        ('RC101_50c', 'RC1', 4, 'RC101 (Mixed, Tight TW)\n2 Drones/Truck'),
        ('R101_50c',  'R1',  4, 'R101 (Random, Tight TW)\n2 Drones/Truck'),
        ('C101_50c',  'C1',  4, 'C101 (Clustered, Tight TW)\n2 Drones/Truck'),
        # EV will be handled separately
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    axes = axes.flatten()

    for ai, (inst_name, tw_type, n_trucks, label) in enumerate(configs):
        if ai >= 3:
            break
        print(f'\n  Panel {ai+1}: {inst_name}')

        try:
            instance = load_instance(inst_name)
        except Exception as e:
            print(f'    SKIP: {e}')
            continue

        # Get 2-Drone solution
        routes, drones, cost, tard, feas = get_routes(
            instance, n_trucks, max_drones_per_truck=2, tw_type=tw_type)

        if routes is None:
            print(f'    No solution found')
            continue

        ax = axes[ai]
        customers = instance['customers']
        depot = instance['depot']
        cust_coords = {c['id']: (c['x'], c['y']) for c in customers}

        # Plot depot
        ax.scatter(depot[0], depot[1], c=DEPOT_COLOR, marker='s', s=150,
                   zorder=10, edgecolors='white', linewidth=1.5)

        # Plot customers
        cx = [c['x'] for c in customers]
        cy = [c['y'] for c in customers]
        ax.scatter(cx, cy, c=CUST_COLOR, marker='o', s=35, zorder=3,
                   edgecolors='white', linewidth=0.3, alpha=0.8)

        # Plot truck routes
        n_routes = max(len(routes), 1)
        route_colors = TRUCK_COLORS[:n_routes] if n_routes <= len(TRUCK_COLORS) else \
                       plt.cm.tab20(np.linspace(0, 1, n_routes))

        for ri, route in enumerate(routes):
            if not route:
                continue
            color = route_colors[ri % len(route_colors)]
            path = [(depot[0], depot[1])]
            for nid in route:
                if nid in cust_coords:
                    path.append(cust_coords[nid])
            path.append((depot[0], depot[1]))
            px, py = zip(*path)
            ax.plot(px, py, '-', color=color, linewidth=2, alpha=0.85,
                    label=f'T{ri+1}')

        # Plot drone missions
        drone_count = 0
        for mission in drones:
            if len(mission) >= 4:
                i, j, k, drone_id = mission[:4]
            else:
                i, j, k = mission
            launch_xy = (depot[0], depot[1]) if i == 0 else cust_coords.get(i)
            cust_xy = cust_coords.get(j)
            recov_xy = (depot[0], depot[1]) if k == 0 else cust_coords.get(k)
            if launch_xy is None or cust_xy is None or recov_xy is None:
                continue
            ax.plot([launch_xy[0], cust_xy[0]], [launch_xy[1], cust_xy[1]],
                    '--', color=DRONE_COLOR, linewidth=1.5, alpha=0.75)
            ax.plot([cust_xy[0], recov_xy[0]], [cust_xy[1], recov_xy[1]],
                    '--', color=DRONE_COLOR, linewidth=1.5, alpha=0.75)
            ax.scatter(cust_xy[0], cust_xy[1], c=DRONE_CUST_COLOR, marker='*',
                       s=120, zorder=9, edgecolors='darkred', linewidth=0.5)
            drone_count += 1

        ax.set_title(f'{label}\nCost={cost:.0f}, {drone_count} drones, '
                    f'Feas={"Yes" if feas else "No"}',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('X (km)', fontsize=9)
        ax.set_ylabel('Y (km)', fontsize=9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.15, linestyle='--')

        # Dynamic bounds
        all_x = [depot[0]] + cx
        all_y = [depot[1]] + cy
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        pad = max((x_max - x_min), (y_max - y_min)) * 0.08
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)

    # ── Panel 4: EV Model B ──
    print(f'\n  Panel 4: EV RC101_50c')
    try:
        instance = load_instance('RC101_50c')
        routes_ev, _, cost_ev, ev_feas, n_cs, cs_time = get_ev_routes(
            instance, n_trucks=3, battery_capacity=30.0,
            energy_rate=2.0, charging_model='linear')

        if routes_ev:
            ax = axes[3]
            customers = instance['customers']
            depot = instance['depot']
            cust_coords = {c['id']: (c['x'], c['y']) for c in customers}
            n_customers = len(customers)

            # Charging stations
            cs_list = [(8.0, 8.0), (4.0, 12.0), (12.0, 4.0)]
            cs_coords_lookup = {}
            for i, cs in enumerate(cs_list):
                cs_id = n_customers + i + 1
                cs_coords_lookup[cs_id] = cs

            ax.scatter(depot[0], depot[1], c=DEPOT_COLOR, marker='s', s=150,
                       zorder=10, edgecolors='white', linewidth=1.5)

            cx = [c['x'] for c in customers]
            cy = [c['y'] for c in customers]
            ax.scatter(cx, cy, c=CUST_COLOR, marker='o', s=35, zorder=3,
                       edgecolors='white', linewidth=0.3, alpha=0.8)

            # Charging stations
            csx = [cs[0] for cs in cs_list]
            csy = [cs[1] for cs in cs_list]
            ax.scatter(csx, csy, c=CS_COLOR, marker='P', s=180, zorder=8,
                       edgecolors='white', linewidth=1.5)
            for i, (x, y) in enumerate(cs_list):
                ax.annotate(f'CS{i+1}', (x, y), textcoords="offset points",
                           xytext=(5, -12), fontsize=8, fontweight='bold',
                           color='darkgreen')

            # Routes with CS stops
            n_routes = max(len(routes_ev), 1)
            route_colors = TRUCK_COLORS[:n_routes]
            for ri, route in enumerate(routes_ev):
                if not route:
                    continue
                color = route_colors[ri % len(route_colors)]
                path = [(depot[0], depot[1])]
                for nid in route:
                    if nid in cust_coords:
                        path.append(cust_coords[nid])
                    elif nid in cs_coords_lookup:
                        path.append(cs_coords_lookup[nid])
                path.append((depot[0], depot[1]))
                px, py = zip(*path)

                # Solid line between customers, dotted near CS
                ax.plot(px, py, '-', color=color, linewidth=2, alpha=0.85)

                # Highlight CS segments
                for i in range(len(path) - 1):
                    # Check if this segment involves a CS
                    is_cs_seg = False
                    mid_x = (path[i][0] + path[i+1][0]) / 2
                    mid_y = (path[i][1] + path[i+1][1]) / 2
                    if any(np.sqrt((path[i][0] - cs[0])**2 + (path[i][1] - cs[1])**2) < 0.01
                           for cs in cs_list):
                        is_cs_seg = True
                    if any(np.sqrt((path[i+1][0] - cs[0])**2 + (path[i+1][1] - cs[1])**2) < 0.01
                           for cs in cs_list):
                        is_cs_seg = True

            ax.set_title(f'EV Model B (Linear Charging)\n'
                        f'Battery=30kWh, Energy=2.0kWh/km, 3 Trucks\n'
                        f'{n_cs} charging stops, EV-Feas={"Yes" if ev_feas else "No"}',
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('X (km)', fontsize=9)
            ax.set_ylabel('Y (km)', fontsize=9)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.15, linestyle='--')

            all_x = [depot[0]] + cx + csx
            all_y = [depot[1]] + cy + csy
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            pad = max((x_max - x_min), (y_max - y_min)) * 0.08
            ax.set_xlim(x_min - pad, x_max + pad)
            ax.set_ylim(y_min - pad, y_max + pad)
    except Exception as e:
        print(f'    EV panel failed: {e}')
        import traceback
        traceback.print_exc()

    # Shared legend
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=DEPOT_COLOR,
               markersize=10, label='Depot'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CUST_COLOR,
               markersize=8, label='Customer'),
        Line2D([0], [0], color=TRUCK_COLORS[0], linewidth=2, label='Truck Route'),
        Line2D([0], [0], color=DRONE_COLOR, linewidth=1.5, linestyle='--',
               label='Drone Flight'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=DRONE_CUST_COLOR,
               markersize=10, markeredgecolor='darkred', markeredgewidth=0.5,
               label='Drone Customer'),
        Line2D([0], [0], marker='P', color='w', markerfacecolor=CS_COLOR,
               markersize=10, label='Charging Station'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=6,
              fontsize=10, framealpha=0.9, edgecolor='gray',
              bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Route Structure Comparison Across Instance Types',
                 fontweight='bold', fontsize=15, y=1.01)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig7_route_map_panel.png')
    fig.savefig(out, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'\n  Saved: fig7_route_map_panel.png')


def generate_comparison_figure():
    """
    Side-by-side No-Drone vs 2-Drone comparison for RC101_50c.
    Shows the geometric impact of drone insertion.
    """
    print('\n' + '=' * 60)
    print('GENERATING NO-DRONE vs 2-DRONE COMPARISON')
    print('=' * 60)

    inst_name = 'RC101_50c'
    instance = load_instance(inst_name)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

    for ax, max_d, label, title_suffix in [
        (ax1, 0, 'Truck-Only', 'No Drones'),
        (ax2, 2, '2 Drones/Truck', '2 Drones/Truck'),
    ]:
        print(f'  Running: {label}...')
        routes, drones, cost, tard, feas = get_routes(
            instance, 4, max_drones_per_truck=max_d, tw_type='RC1')

        customers = instance['customers']
        depot = instance['depot']
        cust_coords = {c['id']: (c['x'], c['y']) for c in customers}

        ax.scatter(depot[0], depot[1], c=DEPOT_COLOR, marker='s', s=150,
                   zorder=10, edgecolors='white', linewidth=1.5)

        cx = [c['x'] for c in customers]
        cy = [c['y'] for c in customers]
        ax.scatter(cx, cy, c=CUST_COLOR, marker='o', s=30, zorder=3,
                   edgecolors='white', linewidth=0.3, alpha=0.8)

        if routes:
            n_routes = max(len(routes), 1)
            route_colors = TRUCK_COLORS[:n_routes] if n_routes <= len(TRUCK_COLORS) else \
                           plt.cm.tab20(np.linspace(0, 1, n_routes))
            for ri, route in enumerate(routes):
                if not route:
                    continue
                color = route_colors[ri % len(route_colors)]
                path = [(depot[0], depot[1])]
                for nid in route:
                    if nid in cust_coords:
                        path.append(cust_coords[nid])
                path.append((depot[0], depot[1]))
                px, py = zip(*path)
                ax.plot(px, py, '-', color=color, linewidth=2.5, alpha=0.85,
                        label=f'Truck {ri+1}')

                # Direction arrows
                for i in range(len(path) - 1):
                    mid_x = (path[i][0] + path[i+1][0]) / 2
                    mid_y = (path[i][1] + path[i+1][1]) / 2
                    dx = path[i+1][0] - path[i][0]
                    dy = path[i+1][1] - path[i][1]
                    seg_len = np.sqrt(dx**2 + dy**2)
                    if seg_len > 0.01:
                        dx_n, dy_n = dx/seg_len, dy/seg_len
                        ax.annotate('', xy=(mid_x + dx_n*0.12, mid_y + dy_n*0.12),
                                   xytext=(mid_x - dx_n*0.12, mid_y - dy_n*0.12),
                                   arrowprops=dict(arrowstyle='->', color=color,
                                                  lw=1.2, alpha=0.6))

            # Drone missions
            drone_count = 0
            for mission in drones:
                if len(mission) >= 4:
                    i, j, k, drone_id = mission[:4]
                else:
                    i, j, k = mission
                launch_xy = (depot[0], depot[1]) if i == 0 else cust_coords.get(i)
                cust_xy = cust_coords.get(j)
                recov_xy = (depot[0], depot[1]) if k == 0 else cust_coords.get(k)
                if launch_xy is None or cust_xy is None or recov_xy is None:
                    continue
                ax.plot([launch_xy[0], cust_xy[0]], [launch_xy[1], cust_xy[1]],
                        '--', color=DRONE_COLOR, linewidth=1.8, alpha=0.8)
                ax.plot([cust_xy[0], recov_xy[0]], [cust_xy[1], recov_xy[1]],
                        '--', color=DRONE_COLOR, linewidth=1.8, alpha=0.8)
                ax.scatter(cust_xy[0], cust_xy[1], c=DRONE_CUST_COLOR, marker='*',
                           s=150, zorder=9, edgecolors='darkred', linewidth=0.8)
                drone_count += 1

            ax.set_title(f'{title_suffix}\n'
                        f'Cost={cost:.0f}, {len(routes)} routes, '
                        f'{drone_count} drone missions\n'
                        f'Feasible={"Yes" if feas else "No"}',
                        fontsize=13, fontweight='bold')
        else:
            ax.set_title(f'{title_suffix}\nNo solution found',
                        fontsize=13, fontweight='bold')

        ax.set_xlabel('X (km)', fontsize=11)
        ax.set_ylabel('Y (km)', fontsize=11)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.15, linestyle='--')

        all_x = [depot[0]] + cx
        all_y = [depot[1]] + cy
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        pad = max((x_max - x_min), (y_max - y_min)) * 0.08
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)

    fig.suptitle('Impact of Drone Integration: Truck-Only vs 2 Drones/Truck\n'
                 'RC101 — 50 Customers, Tight Time Windows, 4 Trucks',
                 fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_route_comparison_nd_vs_2d.png')
    fig.savefig(out, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: fig_route_comparison_nd_vs_2d.png')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate publication-quality route map figures')
    parser.add_argument('--quick', action='store_true',
                       help='Only generate comparison figure (faster)')
    parser.add_argument('--panel', action='store_true',
                       help='Only generate 2x2 panel figure')
    parser.add_argument('--comparison', action='store_true',
                       help='Only generate No-Drone vs 2-Drone comparison')
    args = parser.parse_args()

    if args.comparison:
        generate_comparison_figure()
    elif args.panel:
        generate_panel_figure()
    elif args.quick:
        generate_comparison_figure()
    else:
        generate_all_route_maps()
        generate_panel_figure()
        generate_comparison_figure()


if __name__ == '__main__':
    main()
