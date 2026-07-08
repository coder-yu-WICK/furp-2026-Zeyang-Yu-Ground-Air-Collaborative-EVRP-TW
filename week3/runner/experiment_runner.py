# -*- coding: utf-8 -*-
"""
Unified experiment runner that executes all algorithm-method combinations
across all instance configurations and collects results.
"""

import json
import os
import time
import sys
from datetime import datetime
from collections import defaultdict

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    RESULTS_DIR, DATA_OUT_DIR,
    RC1_INSTANCES, RC2_INSTANCES,
    CUSTOMER_SIZES, VEHICLE_CONFIGS,
    DRONE_ENDURANCE, N_REPEATS, RANDOM_SEEDS,
    HV_REFERENCE,
)
from utils.data_loader import build_all_instances, load_instance_from_disk
from utils.problem_model import (
    evaluate_solution_batch, extract_pareto_front,
    hypervolume_2d,
)
from algorithms.no_drone import run_no_drone
from algorithms.paco import run_paco
from algorithms.nsga2 import run_nsga2
from algorithms.ivnd import run_ivnd


def get_instance_key(source_instance, n_customers):
    """Generate instance key."""
    return f'{source_instance}_{n_customers}c'


def run_experiment(instance, config, methods):
    """
    Run one experiment configuration.

    Args:
        instance: problem instance dict
        config: dict with n_trucks, n_drones, endurance
        methods: dict of method_name → run_function

    Returns:
        dict of method_name → results_dict
    """
    n_trucks = config['n_trucks']
    n_drones = config['n_drones']
    endurance = config['endurance']
    endurance_name = config['endurance_name']

    results = {}
    tw_type = instance['tw_type']

    for method_name, run_fn in methods.items():
        print(f'    Running {method_name}...')
        t0 = time.time()

        if method_name == 'No-Drone':
            result = run_fn(instance, n_runs=N_REPEATS, seed=42)
        elif method_name == 'P-ACO':
            result = run_fn(instance, n_runs=N_REPEATS, endurance=endurance, seed=42)
        elif method_name == 'NSGA-II':
            result = run_fn(instance, n_trucks=n_trucks, n_drones=n_drones,
                           endurance=endurance, n_runs=N_REPEATS, seed=42)
        elif method_name == 'IVND':
            result = run_fn(instance, n_trucks=n_trucks, n_drones=n_drones,
                           endurance=endurance, n_runs=N_REPEATS, seed=42)
        else:
            continue

        elapsed = time.time() - t0
        print(f'      Done in {elapsed:.1f}s')

        # Compute aggregate metrics
        metrics = evaluate_solution_batch(result['solutions'])
        metrics['mean_runtime'] = result['mean_runtime']
        metrics['std_runtime'] = result['std_runtime']

        results[method_name] = metrics

    return results


def build_experiment_matrix():
    """Build the full experiment matrix."""
    matrix = []

    all_source_instances = RC1_INSTANCES + RC2_INSTANCES

    for src_inst in all_source_instances:
        for n_cust in CUSTOMER_SIZES:
            instance_key = get_instance_key(src_inst, n_cust)
            try:
                instance = load_instance_from_disk(instance_key)
            except FileNotFoundError:
                print(f'  WARNING: Instance {instance_key} not found, skipping')
                continue

            vehicle_configs = VEHICLE_CONFIGS[n_cust]

            for n_t, n_d in vehicle_configs:
                for end_name, end_val in DRONE_ENDURANCE.items():
                    config = {
                        'n_trucks': n_t,
                        'n_drones': n_d,
                        'endurance': end_val,
                        'endurance_name': end_name,
                    }
                    tw_type = 'RC1' if src_inst.startswith('RC1') else 'RC2'
                    label = f'{n_cust}c_{tw_type}_{end_name}_{n_t}T+{n_d}D'
                    matrix.append({
                        'instance_key': instance_key,
                        'source_instance': src_inst,
                        'n_customers': n_cust,
                        'tw_type': tw_type,
                        'config': config,
                        'label': label,
                    })

    return matrix


def run_all_experiments():
    """Run the full experiment suite."""
    # Ensure data is built
    print('=' * 70)
    print('Building experiment instances...')
    print('=' * 70)
    build_all_instances()

    # Build experiment matrix
    print('\n' + '=' * 70)
    print('Building experiment matrix...')
    print('=' * 70)
    matrix = build_experiment_matrix()
    print(f'Total experiment configurations: {len(matrix)}')

    methods = {
        'No-Drone': run_no_drone,
        'P-ACO': run_paco,
        'NSGA-II': run_nsga2,
        'IVND': run_ivnd,
    }

    all_results = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for idx, exp in enumerate(matrix):
        print(f'\n{"="*70}')
        print(f'Experiment {idx+1}/{len(matrix)}: {exp["label"]}')
        print(f'  Instance: {exp["instance_key"]} ({exp["n_customers"]} customers, {exp["tw_type"]})')
        print(f'  Config: {exp["config"]["n_trucks"]}T+{exp["config"]["n_drones"]}D, endurance={exp["config"]["endurance"]}km')
        print(f'{"="*70}')

        instance = load_instance_from_disk(exp['instance_key'])
        results = run_experiment(instance, exp['config'], methods)

        # Store results
        exp_result = {
            'label': exp['label'],
            'instance_key': exp['instance_key'],
            'source_instance': exp['source_instance'],
            'n_customers': exp['n_customers'],
            'tw_type': exp['tw_type'],
            'n_trucks': exp['config']['n_trucks'],
            'n_drones': exp['config']['n_drones'],
            'endurance': exp['config']['endurance'],
            'endurance_name': exp['config']['endurance_name'],
            'methods': {},
        }

        for method_name, metrics in results.items():
            exp_result['methods'][method_name] = {
                'mean_cost': metrics['mean_cost'],
                'std_cost': metrics['std_cost'],
                'mean_tardiness': metrics['mean_tardiness'],
                'std_tardiness': metrics['std_tardiness'],
                'feasibility_rate': metrics['feasibility_rate'],
                'hypervolume': metrics['hypervolume'],
                'pareto_points': metrics.get('pareto_points', []),
                'drone_solution_pct': metrics['drone_solution_pct'],
                'avg_drone_missions': metrics['avg_drone_missions'],
                'pareto_size': metrics['pareto_size'],
                'best_cost': metrics['best_cost'],
                'best_tardiness': metrics['best_tardiness'],
                'mean_runtime': metrics['mean_runtime'],
                'std_runtime': metrics['std_runtime'],
            }

        all_results.append(exp_result)

        # Save intermediate results
        interim_path = os.path.join(RESULTS_DIR, f'interim_{timestamp}.json')
        with open(interim_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)

    # Save final results
    final_path = os.path.join(RESULTS_DIR, f'results_{timestamp}.json')
    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    print(f'\n{"="*70}')
    print(f'All experiments complete!')
    print(f'Results saved to: {final_path}')
    print(f'{"="*70}')

    return all_results, final_path


if __name__ == '__main__':
    run_all_experiments()
