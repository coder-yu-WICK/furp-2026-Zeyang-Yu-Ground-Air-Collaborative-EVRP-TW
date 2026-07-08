#!/usr/bin/env python3
"""
Standalone HV fixer — no imports from week3, safe to run while main.py is active.

Reads interim result JSON, recalculates Hypervolume with correct algorithm
and auto-scaling reference point, writes to a NEW file (never touches the
one being written by main.py).

Usage:
    python fix_hv.py              # fix latest result file
    python fix_hv.py --watch 60   # re-run every 60 seconds
    python fix_hv.py --all        # fix all result files
"""

import json
import os
import sys
import time

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


# ── Correct 2D Hypervolume ────────────────────────────────────────

def _nondom(pairs):
    """Filter list of (cost, tard) to non-dominated subset."""
    if len(pairs) <= 1:
        return pairs
    # Sort by cost
    sp = sorted(pairs, key=lambda p: p[0])
    result = []
    best_t = float('inf')
    for c, t in sp:
        if t < best_t:
            result.append((c, t))
            best_t = t
    return result


def compute_hv(pairs, ref_cost=5000.0, ref_tard=20000.0):
    """
    Correct 2D HV for minimization.

    Auto-expands reference point to cover all observed points at 1.2x.
    Algorithm: non-dominated sort → sweep rectangles.
    """
    if not pairs:
        return 0.0

    max_c = max(c for c, _ in pairs)
    max_t = max(t for _, t in pairs)
    rc = max(ref_cost, max_c * 1.2, 1.0)
    rt = max(ref_tard, max_t * 1.2, 1.0)

    pts = [(c, t) for c, t in pairs if c <= rc and t <= rt]
    if not pts:
        return 0.0

    nondom = _nondom(pts)

    hv = 0.0
    prev_t = rt
    for c, t in nondom:
        if t < prev_t:
            hv += (rc - c) * (prev_t - t)
            prev_t = t

    return hv


# ── File handling ──────────────────────────────────────────────────

def find_result_files():
    files = []
    for f in os.listdir(RESULTS_DIR):
        if f.endswith('.json') and ('interim_' in f or 'results_' in f):
            if '_hv_fixed' not in f:  # skip our own output
                files.append(os.path.join(RESULTS_DIR, f))
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files


def fix_file(input_path):
    """Recalculate HV for all experiments in a result file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    for exp in results:
        if 'methods' not in exp:
            continue

        for method_name, metrics in exp['methods'].items():
            # Use stored pareto_points if available (new format)
            pareto_pts = metrics.get('pareto_points', [])

            if pareto_pts:
                # Already have Pareto points, just recompute HV
                new_hv = compute_hv([tuple(p) for p in pareto_pts])
            else:
                # Old format: estimate from mean cost/tardiness
                # (single-point approximation, less accurate but nonzero)
                mc = metrics.get('mean_cost', 0)
                mt = metrics.get('mean_tardiness', 0)
                if mc > 0 or mt > 0:
                    new_hv = compute_hv([(mc, mt)])
                else:
                    new_hv = 0.0

            old_hv = metrics.get('hypervolume', 0)
            metrics['hypervolume'] = new_hv

    # Write to new file
    base = os.path.basename(input_path)
    name, ext = os.path.splitext(base)
    out_path = os.path.join(RESULTS_DIR, f'{name}_hv_fixed{ext}')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    return out_path, results


def print_summary(results):
    """Print a compact HV summary table."""
    print()
    print(f"{'Label':<45} {'P-ACO':>10} {'NSGA-II':>10} {'IVND':>10} {'No-Drone':>10}")
    print('-' * 85)
    for exp in results:
        label = exp.get('label', '?')[:44]
        row = [label]
        for m in ['P-ACO', 'NSGA-II', 'IVND', 'No-Drone']:
            hv = exp.get('methods', {}).get(m, {}).get('hypervolume', 0)
            row.append(f'{hv:>10.0f}')
        print(' '.join(row))


# ── Main ───────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    watch_interval = None
    process_all = '--all' in args

    for i, a in enumerate(args):
        if a == '--watch' and i + 1 < len(args):
            watch_interval = int(args[i + 1])

    while True:
        files = find_result_files()
        if not files:
            print("No result files found.")
        else:
            targets = files if process_all else [files[0]]
            for f in targets:
                mtime = os.path.getmtime(f)
                age = time.time() - mtime
                print(f"[{time.strftime('%H:%M:%S')}] {os.path.basename(f)} "
                      f"(modified {age:.0f}s ago)")
                out, results = fix_file(f)
                print(f"  → {os.path.basename(out)}")
                print_summary(results)

        if watch_interval is None:
            break
        print(f'\nWaiting {watch_interval}s...')
        time.sleep(watch_interval)


if __name__ == '__main__':
    main()
