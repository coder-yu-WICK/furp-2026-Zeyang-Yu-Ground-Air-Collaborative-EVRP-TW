# -*- coding: utf-8 -*-
"""
Load Solomon VRPTW instances from py-ga-VRPTW JSON format,
extract subsets, scale coordinates, and set depot.
"""

import json
import math
import os
import random
from itertools import combinations

from config import (
    DATA_JSON_DIR, DATA_OUT_DIR,
    COORD_SCALE, DEPOT,
    RC1_INSTANCES, RC2_INSTANCES,
    R1_INSTANCES, R2_INSTANCES,
    C1_INSTANCES, C2_INSTANCES,
    CUSTOMER_SIZES, TW_TYPES,
)


def load_solomon_json(instance_name):
    """Load a single Solomon instance from py-ga-VRPTW JSON."""
    filepath = os.path.join(DATA_JSON_DIR, f'{instance_name}.json')
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_customers(instance_json, n_customers, seed=42):
    """
    Extract first n_customers from a Solomon instance.
    Returns list of customer dicts with scaled coordinates.
    """
    rng = random.Random(seed)
    all_customers = []
    for key, val in instance_json.items():
        if key.startswith('customer_'):
            cid = int(key.split('_')[1])
            all_customers.append({
                'id': cid,
                'x': val['coordinates']['x'] * COORD_SCALE,
                'y': val['coordinates']['y'] * COORD_SCALE,
                'demand': val['demand'],
                'ready_time': val['ready_time'],
                'due_time': val['due_time'],
                'service_time': val['service_time'],
            })
    all_customers.sort(key=lambda c: c['id'])
    if n_customers <= len(all_customers):
        selected = all_customers[:n_customers]
    else:
        selected = all_customers
    return selected


def compute_distance_matrix(customers, depot):
    """Compute Euclidean distance matrix between all points (depot + customers)."""
    points = [depot] + [(c['x'], c['y']) for c in customers]
    n = len(points)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            matrix[i][j] = math.sqrt(dx * dx + dy * dy)
    return matrix


def build_instance(instance_name, n_customers, seed=42):
    """
    Build a complete problem instance for truck-drone EVRP-TW.

    Returns dict with:
      - name, depot, customers, distance_matrix
      - tw_type ('RC1' or 'RC2')
    """
    raw = load_solomon_json(instance_name)
    customers = extract_customers(raw, n_customers, seed)
    dist_matrix = compute_distance_matrix(customers, DEPOT)

    # Detect TW type from instance name
    if instance_name.startswith('RC1'):
        tw_type = 'RC1'
    elif instance_name.startswith('RC2'):
        tw_type = 'RC2'
    elif instance_name.startswith('R1'):
        tw_type = 'R1'
    elif instance_name.startswith('R2'):
        tw_type = 'R2'
    elif instance_name.startswith('C1'):
        tw_type = 'C1'
    elif instance_name.startswith('C2'):
        tw_type = 'C2'
    else:
        tw_type = 'RC1'  # fallback

    return {
        'name': f'{instance_name}_{n_customers}c',
        'source_instance': instance_name,
        'depot': DEPOT,
        'n_customers': len(customers),
        'customers': customers,
        'distance_matrix': dist_matrix,
        'tw_type': tw_type,
        'tw_horizon': TW_TYPES[tw_type]['horizon'],
    }


def build_all_instances():
    """Build all experiment instances and save to JSON."""
    instances = {}
    all_source = (RC1_INSTANCES + RC2_INSTANCES +
                  R1_INSTANCES + R2_INSTANCES +
                  C1_INSTANCES + C2_INSTANCES)

    for src in all_source:
        for n in CUSTOMER_SIZES:
            inst = build_instance(src, n)
            key = inst['name']
            instances[key] = inst

            # Save to disk
            out_path = os.path.join(DATA_OUT_DIR, f'{key}.json')
            # Convert to serializable format
            serializable = {
                'name': inst['name'],
                'source_instance': inst['source_instance'],
                'depot': list(inst['depot']),
                'n_customers': inst['n_customers'],
                'customers': inst['customers'],
                'distance_matrix': inst['distance_matrix'],
                'tw_type': inst['tw_type'],
                'tw_horizon': inst['tw_horizon'],
            }
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(serializable, f, indent=2)
            print(f'  Saved: {key}.json')

    return instances


def load_instance_from_disk(name):
    """Load a pre-built instance from disk."""
    filepath = os.path.join(DATA_OUT_DIR, f'{name}.json')
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data['depot'] = tuple(data['depot'])
    return data


if __name__ == '__main__':
    print('Building all experiment instances...')
    build_all_instances()
    print('Done.')
