#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Time-Window Augmented CEVRP Instances.

Takes the 24 Mavrovouniotis CEVRP benchmark instances and adds realistic
time windows using a Solomon-style generation scheme. This creates EVRP-TW
instances that allow direct comparison: can methods designed for CEVRP (no TW)
handle time windows?

TW generation rules (Solomon-inspired):
  - ready_time = max(0, travel_time_from_depot - slack)
  - due_time = ready_time + TW_width
  - Tight TW (type 1): width = 10-30 time units
  - Wide TW (type 2): width = 60-120 time units
  - Mixed: 50% tight, 50% wide per instance
  - Service time = 10 (standard Solomon)
  - Speed = 1.0 (so distance ≈ travel time in same units)

The TW horizon is set to accommodate the longest possible route.
"""

import os, sys, json, math, random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from week8.core.ecvrp_loader import load_all_ecvrp_instances

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

SPEED = 1.0          # distance units per time unit
SERVICE_TIME = 10.0  # standard Solomon service time
TIGHT_WIDTH = (10, 30)
WIDE_WIDTH = (60, 120)


def compute_depot_distances(customers, depot):
    """Compute Euclidean distance from depot to each customer."""
    dists = []
    for c in customers:
        d = math.hypot(c['x'] - depot[0], c['y'] - depot[1])
        dists.append(d)
    return dists


def generate_time_windows(customers, depot, seed=42):
    """
    Generate Solomon-style time windows for CEVRP customers.

    Returns: list of (ready_time, due_time) pairs, and the TW horizon.
    """
    rng = random.Random(seed)
    n = len(customers)
    depot_dists = compute_depot_distances(customers, depot)
    max_dist = max(depot_dists) if depot_dists else 100

    tw_data = []
    for i, c in enumerate(customers):
        # Base ready time: proportional to distance from depot
        travel_time = depot_dists[i] / SPEED

        # Determine if tight or wide TW (alternating by index for mix)
        if i % 2 == 0:
            # Tight TW
            width = rng.uniform(*TIGHT_WIDTH)
            # ready_time: can start a bit before travel time (slack)
            ready_time = max(0.0, travel_time - rng.uniform(0, width * 0.3))
        else:
            # Wide TW
            width = rng.uniform(*WIDE_WIDTH)
            ready_time = max(0.0, travel_time - rng.uniform(0, width * 0.5))

        due_time = ready_time + width

        tw_data.append({
            'ready_time': round(ready_time, 2),
            'due_time': round(due_time, 2),
            'service_time': SERVICE_TIME,
            'tw_type': 'tight' if i % 2 == 0 else 'wide',
        })

    # Horizon: accommodate the latest due_time + return travel
    max_due = max(t['due_time'] for t in tw_data)
    max_return = max(depot_dists) / SPEED if depot_dists else 50
    horizon = max_due + max_return + SERVICE_TIME * 2

    return tw_data, round(horizon, 2)


def augment_instance_with_tw(instance, seed=42):
    """
    Add time windows to an E-CVRP instance. Returns a new instance dict.
    """
    customers = instance['customers']
    depot = instance['depot']

    tw_data, horizon = generate_time_windows(customers, depot, seed=seed)

    new_customers = []
    for c, tw in zip(customers, tw_data):
        new_c = dict(c)
        new_c['ready_time'] = tw['ready_time']
        new_c['due_time'] = tw['due_time']
        new_c['service_time'] = tw['service_time']
        new_c['_tw_type'] = tw['tw_type']
        new_customers.append(new_c)

    new_inst = dict(instance)
    new_inst['customers'] = new_customers
    new_inst['tw_type'] = 'mixed'  # override from 'none'
    new_inst['tw_horizon'] = horizon
    # Keep truck_capacity and other CEVRP-specific fields

    return new_inst


def main():
    print("Generating TW-augmented CEVRP instances...")
    benchmark_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  'e_cvrp_benchmark')
    instances = load_all_ecvrp_instances(benchmark_dir)

    tw_instances = {}
    for name, inst in instances.items():
        tw_inst = augment_instance_with_tw(inst, seed=42)
        tw_instances[name] = tw_inst

        n_tight = sum(1 for c in tw_inst['customers'] if c.get('_tw_type') == 'tight')
        n_wide = sum(1 for c in tw_inst['customers'] if c.get('_tw_type') == 'wide')
        print(f"  {name}: {tw_inst['n_customers']}c, horizon={tw_inst['tw_horizon']:.0f}, "
              f"tight={n_tight}, wide={n_wide}")

    # Save
    for name, inst in tw_instances.items():
        out_path = os.path.join(DATA_DIR, f'{name}_tw.json')
        serializable = {
            'name': inst['name'] + '_tw',
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
        # Clean up internal TW type marker
        for c in serializable['customers']:
            c.pop('_tw_type', None)

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, default=str)

    print(f"\nSaved {len(tw_instances)} TW-augmented instances to {DATA_DIR}/")
    print("Done.")


if __name__ == '__main__':
    main()
