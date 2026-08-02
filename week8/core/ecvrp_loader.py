#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E-CVRP Benchmark Instance Loader.

Parses .evrp files from Mavrovouniotis et al. (IEEE CEC 2020) benchmark set
and converts them to the standard week8 instance format.

Format specification:
  - Specification part: NAME, TYPE, COMMENT, OPTIMAL_VALUE, VEHICLES, DIMENSION,
    STATIONS, CAPACITY, ENERGY_CAPACITY, ENERGY_CONSUMPTION, EDGE_WEIGHT_TYPE
  - Data part: NODE_COORD_SECTION, DEMAND_SECTION, STATIONS_COORD_SECTION, DEPOT_SECTION

Key differences from Solomon TW instances:
  - No time windows (ready_time=0, due_time=inf)
  - No service times (set to 0)
  - Charging stations as separate nodes with coordinates
  - Battery capacity + energy consumption rate specified per instance
  - Vehicle cargo capacity specified per instance
"""

import os
import sys
import json
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from week8.config import COORD_SCALE


def parse_evrp_file(filepath):
    """
    Parse a single .evrp file.

    Returns dict with standard week8 instance format.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    spec = {}
    section = None
    node_coords = {}     # node_id -> (x, y)
    demands = {}         # node_id -> demand
    station_ids = set()  # set of charging station node IDs
    depot_id = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect sections
        if line.startswith('NODE_COORD_SECTION'):
            section = 'node_coord'
            continue
        elif line.startswith('DEMAND_SECTION'):
            section = 'demand'
            continue
        elif line.startswith('STATIONS_COORD_SECTION') or line.startswith('STATION_COORD_SECTION'):
            section = 'station_coord'
            continue
        elif line.startswith('DEPOT_SECTION'):
            section = 'depot'
            continue
        elif line == 'EOF':
            break

        if section is None:
            # Parse spec fields
            if ':' in line:
                key, _, value = line.partition(':')
                key = key.strip().upper()
                value = value.strip()
                spec[key] = value
        elif section == 'node_coord':
            parts = line.split()
            if len(parts) >= 3:
                nid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                node_coords[nid] = (x, y)
        elif section == 'demand':
            parts = line.split()
            if len(parts) >= 2:
                nid = int(parts[0])
                d = int(parts[1])
                demands[nid] = d
        elif section == 'station_coord':
            parts = line.split()
            if len(parts) >= 1:
                nid = int(parts[0])
                station_ids.add(nid)
        elif section == 'depot':
            if line == '-1':
                section = None
            else:
                parts = line.split()
                if len(parts) >= 1:
                    depot_id = int(parts[0])

    # ── Determine depot coordinates ──
    if depot_id and depot_id in node_coords:
        depot_xy = node_coords[depot_id]
    elif 1 in node_coords:
        depot_xy = node_coords[1]
        depot_id = 1
    else:
        depot_xy = (0.0, 0.0)

    # ── Identify customer nodes ──
    # Customers = all nodes except depot and charging stations
    customer_ids = sorted([
        nid for nid in node_coords
        if nid != depot_id and nid not in station_ids
    ])

    # ── Build customer list ──
    customers = []
    for i, cid in enumerate(customer_ids):
        coord = node_coords[cid]
        # Keep raw coordinates (E-CVRP uses different scale than Solomon)
        # Published optimal values are in raw coordinate units
        x_scaled = coord[0]
        y_scaled = coord[1]
        customers.append({
            'id': i + 1,  # 1-indexed in standard format
            'original_id': cid,
            'x': x_scaled,
            'y': y_scaled,
            'demand': demands.get(cid, 0),
            'ready_time': 0.0,
            'due_time': 1e9,  # effectively infinite (no TW in E-CVRP)
            'service_time': 0.0,
        })

    # ── Build charging station list ──
    charging_stations = []
    for sid in sorted(station_ids):
        if sid in node_coords:
            coord = node_coords[sid]
            charging_stations.append({
                'id': sid,
                'x': coord[0],
                'y': coord[1],
            })

    # ── Compute depot (raw coordinates — E-CVRP doesn't use COORD_SCALE) ──
    depot_scaled = (depot_xy[0], depot_xy[1])

    # ── Compute distance matrix (depot + customers) ──
    n_cust = len(customers)
    points = [depot_scaled] + [(c['x'], c['y']) for c in customers]
    dist_matrix = [[0.0] * (n_cust + 1) for _ in range(n_cust + 1)]
    for i in range(n_cust + 1):
        for j in range(n_cust + 1):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            dist_matrix[i][j] = math.sqrt(dx * dx + dy * dy)

    # ── Parse spec values ──
    name = spec.get('NAME', os.path.basename(filepath))
    n_vehicles = int(spec.get('VEHICLES', 2))
    battery_capacity = float(spec.get('ENERGY_CAPACITY', 100))
    energy_consumption = float(spec.get('ENERGY_CONSUMPTION', 1.0))
    cargo_capacity = float(spec.get('CAPACITY', 200))
    optimal_value = spec.get('OPTIMAL_VALUE', '-')
    if optimal_value != '-':
        try:
            optimal_value = float(optimal_value)
        except ValueError:
            optimal_value = None
    else:
        optimal_value = None

    return {
        'name': name.replace('.evrp', ''),
        'source': 'mavrovouniotis_cec2020',
        'depot': depot_scaled,
        'n_customers': len(customers),
        'customers': customers,
        'distance_matrix': dist_matrix,
        'charging_stations': charging_stations,
        # E-CVRP specific parameters
        'n_vehicles': n_vehicles,
        'battery_capacity': battery_capacity,
        'energy_consumption_rate': energy_consumption,
        'cargo_capacity': cargo_capacity,
        'optimal_value': optimal_value,
        # No time windows
        'tw_type': 'none',
        'tw_horizon': 1e9,
        # Instance-specific truck capacity (overrides config default)
        'truck_capacity': cargo_capacity,
    }


def load_all_ecvrp_instances(benchmark_dir=None):
    """
    Parse all .evrp files in the benchmark directory.

    Args:
        benchmark_dir: path to e_cvrp_benchmark directory

    Returns:
        dict of name -> instance dict
    """
    if benchmark_dir is None:
        benchmark_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'e_cvrp_benchmark'
        )

    instances = {}
    for fname in sorted(os.listdir(benchmark_dir)):
        if fname.endswith('.evrp'):
            filepath = os.path.join(benchmark_dir, fname)
            try:
                inst = parse_evrp_file(filepath)
                instances[inst['name']] = inst
            except Exception as e:
                print(f"  ERROR parsing {fname}: {e}")

    return instances


def save_ecvrp_instances(instances, data_dir=None):
    """Save parsed instances to JSON files in the data directory."""
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data'
        )

    os.makedirs(data_dir, exist_ok=True)

    for name, inst in instances.items():
        out_path = os.path.join(data_dir, f'{name}.json')
        serializable = {
            'name': inst['name'],
            'source': inst['source'],
            'depot': list(inst['depot']),
            'n_customers': inst['n_customers'],
            'customers': inst['customers'],
            'distance_matrix': inst['distance_matrix'],
            'charging_stations': inst['charging_stations'],
            'n_vehicles': inst['n_vehicles'],
            'battery_capacity': inst['battery_capacity'],
            'energy_consumption_rate': inst['energy_consumption_rate'],
            'cargo_capacity': inst['cargo_capacity'],
            'optimal_value': inst['optimal_value'],
            'tw_type': inst['tw_type'],
            'tw_horizon': inst['tw_horizon'],
            'truck_capacity': inst['truck_capacity'],
        }
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, default=str)

    return data_dir


def load_ecvrp_instance_from_disk(name, data_dir=None):
    """Load a pre-built E-CVRP instance from disk."""
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data'
        )
    filepath = os.path.join(data_dir, f'{name}.json')
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data['depot'] = tuple(data['depot'])
    return data


def print_instance_summary(instances):
    """Print a summary table of all instances."""
    print(f"{'Name':<25} {'N':>5} {'Veh':>4} {'Batt':>6} {'EC':>5} {'Cap':>6} {'Opt':>8}")
    print("-" * 65)
    for name in sorted(instances.keys(), key=lambda n: instances[n]['n_customers']):
        inst = instances[name]
        n = inst['n_customers']
        v = inst['n_vehicles']
        b = inst['battery_capacity']
        e = inst['energy_consumption_rate']
        c = inst['cargo_capacity']
        o = inst['optimal_value']
        o_str = f"{o:.0f}" if o else "-"
        print(f"{name:<25} {n:>5} {v:>4} {b:>6.0f} {e:>5.1f} {c:>6.0f} {o_str:>8}")


if __name__ == '__main__':
    print("Parsing E-CVRP benchmark instances...")
    instances = load_all_ecvrp_instances()
    print(f"\nParsed {len(instances)} instances.\n")
    print_instance_summary(instances)

    print("\nSaving to data directory...")
    save_ecvrp_instances(instances)
    print("Done.")
