#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge POMO-only results with existing experiment results.

Reads existing experiment results and POMO results, merges them by
matching on the experiment label. Produces a combined results file
that can be used with the report generator.

Usage:
    python merge_results.py
    python merge_results.py --existing results/XXX.json --pomo results/pomo_YYY.json
"""

import json
import os
import sys
import glob
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def find_latest(pattern, exclude_pattern=None):
    """Find latest result file matching pattern."""
    files = glob.glob(os.path.join(RESULTS_DIR, pattern))
    if exclude_pattern:
        files = [f for f in files if exclude_pattern not in os.path.basename(f)]
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def merge_results(existing_path, pomo_path):
    """
    Merge POMO results into existing results.

    For each experiment config in the existing results, adds POMO method
    if available in the pomo results.
    """
    with open(existing_path, 'r', encoding='utf-8') as f:
        existing = json.load(f)

    with open(pomo_path, 'r', encoding='utf-8') as f:
        pomo_results = json.load(f)

    # Build lookup for POMO results by label
    pomo_by_label = {}
    for exp in pomo_results:
        label = exp.get('label', '')
        # Normalize label for matching
        pomo_by_label[label] = exp

    # Merge POMO into existing
    merged_count = 0
    for exp in existing:
        label = exp.get('label', '')
        if label in pomo_by_label:
            pomo_exp = pomo_by_label[label]
            if 'POMO' in pomo_exp.get('methods', {}):
                exp['methods']['POMO'] = pomo_exp['methods']['POMO']
                merged_count += 1

    # Also add any POMO-only experiments not in existing
    existing_labels = {exp.get('label', '') for exp in existing}
    for label, pomo_exp in pomo_by_label.items():
        if label not in existing_labels:
            existing.append(pomo_exp)
            merged_count += 1

    # Save merged
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(RESULTS_DIR, f'merged_results_{timestamp}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2)

    print(f'Merged {merged_count} POMO entries')
    print(f'Total experiments: {len(existing)}')
    print(f'Saved to: {out_path}')

    # Quick summary
    print_summary(existing)

    return out_path


def print_summary(results):
    """Print compact summary of all methods."""
    methods = ['P-ACO', 'NSGA-II', 'IVND', 'No-Drone', 'POMO']
    active = [m for m in methods if any(m in exp.get('methods', {}) for exp in results)]

    if not active:
        return

    print()
    header = f"{'Label':<42}"
    for m in active:
        header += f' {m:>10}'
    print(header)
    print('-' * (42 + 11 * len(active)))

    for exp in results:
        label = exp.get('label', '?')[:41]
        row = f'{label:<42}'
        for m in active:
            hv = exp.get('methods', {}).get(m, {}).get('hypervolume', 0)
            row += f' {hv:>10.0f}'
        print(row)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Merge POMO results with existing')
    parser.add_argument('--existing', type=str, default=None,
                        help='Path to existing results JSON')
    parser.add_argument('--pomo', type=str, default=None,
                        help='Path to POMO results JSON')
    args = parser.parse_args()

    # Find files
    existing_path = args.existing or find_latest('interim_*.json', 'hv_fixed')
    pomo_path = args.pomo or find_latest('pomo_interim_*.json')

    if not existing_path:
        print("ERROR: No existing results found.")
        print("Specify with --existing path/to/results.json")
        sys.exit(1)

    if not pomo_path:
        print("ERROR: No POMO results found.")
        print("Specify with --pomo path/to/pomo_results.json")
        sys.exit(1)

    print(f'Existing results: {os.path.basename(existing_path)}')
    print(f'POMO results: {os.path.basename(pomo_path)}')

    merge_results(existing_path, pomo_path)


if __name__ == '__main__':
    main()
