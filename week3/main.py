#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 3 Main Entry Point
=======================
Run all Truck-Drone EVRP-TW experiments from VS Code or terminal.

Usage:
    python main.py              # Run all experiments
    python main.py --quick      # Quick test (1 repeat, small instances only)
    python main.py --report     # Generate report from existing results
"""

import sys
import os

# Ensure the week3 directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from runner.experiment_runner import run_all_experiments


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Week 3 Truck-Drone EVRP-TW Experiments')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode: 1 repeat, 25c only')
    parser.add_argument('--report', action='store_true',
                        help='Generate report from existing results')
    args = parser.parse_args()

    if args.quick:
        print('=' * 70)
        print('QUICK TEST MODE')
        print('=' * 70)
        # Override config for quick test
        import config
        config.N_REPEATS = 1
        config.CUSTOMER_SIZES = [25]
        config.PACO['iterations'] = 20
        config.NSGA2['generations'] = 20
        config.IVND['max_iterations'] = 30
        print('Running quick smoke test...')

    if args.report:
        from utils.report_generator import generate_report
        generate_report()
        return

    # Run experiments
    results, result_path = run_all_experiments()

    print(f'\nResults saved to: {result_path}')
    print('\nTo generate the report, run: python main.py --report')


if __name__ == '__main__':
    main()
