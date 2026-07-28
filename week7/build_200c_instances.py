#!/usr/bin/env python3
"""
Generate proper 200-customer Solomon instances by combining two instances
of the same type. Each Solomon source has exactly 100 customers, so we
merge Instance A (customers 1-100) + Instance B (customers 101-200).

Type mapping (representative pairs):
  RC1: RC101 + RC102    RC2: RC201 + RC202
  R1:  R101 + R102      R2:  R201 + R202
  C1:  C101 + C102      C2:  C201 + C202
"""

import json, os, sys, math, random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'week3'))
from config import COORD_SCALE, DEPOT, DATA_OUT_DIR, DATA_JSON_DIR

# Pairs for 200c generation — one pair per Solomon type
TYPE_PAIRS = {
    'RC1': ('RC101', 'RC102'),
    'RC2': ('RC201', 'RC202'),
    'R1':  ('R101',  'R102'),
    'R2':  ('R201',  'R202'),
    'C1':  ('C101',  'C102'),
    'C2':  ('C201',  'C202'),
}


def load_solomon_json(instance_name):
    filepath = os.path.join(DATA_JSON_DIR, f'{instance_name}.json')
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_customers_from_json(instance_json, id_offset=0):
    """Extract all customers from a Solomon JSON, optionally offset IDs."""
    customers = []
    for key, val in instance_json.items():
        if key.startswith('customer_'):
            cid = int(key.split('_')[1]) + id_offset
            customers.append({
                'id': cid,
                'x': val['coordinates']['x'] * COORD_SCALE,
                'y': val['coordinates']['y'] * COORD_SCALE,
                'demand': val['demand'],
                'ready_time': val['ready_time'],
                'due_time': val['due_time'],
                'service_time': val['service_time'],
            })
    customers.sort(key=lambda c: c['id'])
    return customers


def compute_distance_matrix(customers, depot):
    points = [depot] + [(c['x'], c['y']) for c in customers]
    n = len(points)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            matrix[i][j] = math.sqrt(dx*dx + dy*dy)
    return matrix


def build_200c_instance(tw_type, inst_a_name, inst_b_name):
    """Combine two 100c Solomon instances into one 200c instance."""
    raw_a = load_solomon_json(inst_a_name)
    raw_b = load_solomon_json(inst_b_name)

    custs_a = extract_customers_from_json(raw_a, id_offset=0)     # IDs 1-100
    custs_b = extract_customers_from_json(raw_b, id_offset=100)   # IDs 101-200
    all_custs = custs_a + custs_b

    dist_matrix = compute_distance_matrix(all_custs, DEPOT)

    instance = {
        'name': f'{inst_a_name}_200c',
        'source_instance': inst_a_name,
        'source_instance_b': inst_b_name,
        'depot': list(DEPOT),
        'n_customers': 200,
        'customers': all_custs,
        'distance_matrix': dist_matrix,
        'tw_type': tw_type,
        'tw_horizon': max(c['due_time'] for c in all_custs),
    }

    out_path = os.path.join(DATA_OUT_DIR, f'{inst_a_name}_200c.json')
    with open(out_path, 'w') as f:
        json.dump(instance, f, indent=2)
    return instance


def main():
    print("Building 200c instances (combining same-type pairs)...")
    os.makedirs(DATA_OUT_DIR, exist_ok=True)

    for tw_type, (inst_a, inst_b) in TYPE_PAIRS.items():
        inst = build_200c_instance(tw_type, inst_a, inst_b)
        n = inst['n_customers']
        print(f"  {inst['name']}: {n} customers, TW horizon={inst['tw_horizon']:.0f}")

    print(f"\nDone. 6 instances saved to {DATA_OUT_DIR}/")


if __name__ == '__main__':
    main()
