#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 6 P2: Compare fine-tuned POMO model vs original on W5 benchmark.

Runs both models through the W5 pipeline on all 12 benchmark instances
and reports cost, tardiness, feasibility differences.

Usage:
    python week6/compare_models.py
    python week6/compare_models.py --original week4/algorithms/pomo/checkpoints/best_model.pt
    python week6/compare_models.py --finetuned week6/checkpoints/best_finetuned.pt
"""

import os, sys, json, time, copy
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_SCRIPT_DIR)
_W4 = os.path.join(_PROJECT, 'week4')
_W5 = os.path.join(_PROJECT, 'week5')

for _p in [_W5, _W4, _SCRIPT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import numpy as np

from config import RC1_INSTANCES, RC2_INSTANCES, CUSTOMER_SIZES
from utils.data_loader import load_instance_from_disk, build_all_instances
from utils.problem_model import evaluate_solution_batch
from pomo_mt_improved import run_pomo_improved


def run_benchmark(model_path, model_label, instances_configs, n_runs=3):
    """
    Run W5 pipeline with a given POMO checkpoint on all benchmark instances.

    Uses tw_aware_drone variant (the best for RC2, and adaptive_tw_drone for RC1
    as established by P1 meta-learner).

    Args:
        model_path: path to POMO checkpoint
        model_label: display label
        instances_configs: list of (instance_key, n_trucks, tw_type)
        n_runs: repeats per instance

    Returns:
        list of result dicts
    """
    # Monkey-patch: make run_pomo_improved use our model path
    import pomo_mt_improved as pmi
    original_model_path = None

    # Find the ImprovedPOMOSolver class and override model_path
    # We need to patch the solver creation inside run_pomo_improved
    original_solve = pmi.ImprovedPOMOSolver.__init__

    def patched_init(self, model_path_arg=None, device='cpu'):
        # Force our model path
        if model_path_arg is None:
            model_path_arg = model_path
        original_solve(self, model_path_arg, device)

    pmi.ImprovedPOMOSolver.__init__ = patched_init

    results = []
    for idx, (inst_key, n_trucks, tw_type) in enumerate(instances_configs):
        inst = load_instance_from_disk(inst_key)

        # Select best variant per TW type (from P1 finding)
        variant = 'adaptive_tw_drone' if tw_type == 'RC1' else 'tw_aware_drone'

        print(f'  [{idx+1}/{len(instances_configs)}] {inst_key} ({variant})...',
              end=' ', flush=True)
        t0 = time.time()

        try:
            r = run_pomo_improved(
                inst, n_runs=n_runs, n_trucks=n_trucks,
                endurance='medium', seed=42,
                variant=variant, tw_beta=0.4,
            )
            m = evaluate_solution_batch(r['solutions'])
            dt = time.time() - t0

            result = {
                'instance': inst_key,
                'variant': variant,
                'tw_type': tw_type,
                'mean_cost': m['mean_cost'],
                'std_cost': m.get('std_cost', 0),
                'mean_tardiness': m['mean_tardiness'],
                'std_tardiness': m.get('std_tardiness', 0),
                'feasibility_rate': m['feasibility_rate'],
                'avg_drone_missions': m.get('avg_drone_missions', 0),
                'mean_runtime': r.get('mean_runtime', dt),
            }
            print(f'cost={m["mean_cost"]:.0f} tard={m["mean_tardiness"]:.0f} '
                  f'feas={m["feasibility_rate"]*100:.0f}% drones={m.get("avg_drone_missions",0):.1f} {dt:.0f}s')
        except Exception as e:
            print(f'ERROR: {e}')
            result = {
                'instance': inst_key,
                'variant': variant,
                'tw_type': tw_type,
                'mean_cost': float('inf'),
                'std_cost': 0,
                'mean_tardiness': float('inf'),
                'std_tardiness': 0,
                'feasibility_rate': 0.0,
                'avg_drone_missions': 0,
                'mean_runtime': 0,
            }

        results.append(result)

    # Restore original
    pmi.ImprovedPOMOSolver.__init__ = original_solve
    return results


def print_comparison(orig_results, ft_results):
    """Print side-by-side comparison table."""
    print(f'\n{"="*100}')
    print(f'MODEL COMPARISON: Original vs Fine-Tuned')
    print(f'{"="*100}')

    # Match results by instance
    ft_by_inst = {r['instance']: r for r in ft_results}

    header = (f'  {"Instance":<16s} {"Variant":<20s} '
              f'{"Orig Cost":>10s} {"FT Cost":>10s} {"ΔCost%":>8s} '
              f'{"Orig Tard":>10s} {"FT Tard":>10s} {"ΔTard":>8s} '
              f'{"Orig Feas":>8s} {"FT Feas":>8s}')
    print(header)
    print('  ' + '-' * 96)

    total_orig_cost = 0
    total_ft_cost = 0
    total_orig_tard = 0
    total_ft_tard = 0
    n_better = 0
    n_worse = 0
    n_same = 0

    for orig in orig_results:
        ft = ft_by_inst.get(orig['instance'], {})
        oc = orig['mean_cost']
        fc = ft.get('mean_cost', float('inf'))
        ot = orig['mean_tardiness']
        ft_t = ft.get('mean_tardiness', float('inf'))

        if oc > 0 and fc < float('inf'):
            delta = (fc - oc) / oc * 100
            delta_str = f'{delta:+.1f}%'
        else:
            delta_str = '---'

        if ot >= 0 and ft_t < float('inf'):
            delta_t = ft_t - ot
            delta_t_str = f'{delta_t:+.0f}'
        else:
            delta_t_str = '---'

        print(f'  {orig["instance"]:<16s} {orig["variant"]:<20s} '
              f'{oc:>10.0f} {fc:>10.0f} {delta_str:>8s} '
              f'{ot:>10.0f} {ft_t:>10.0f} {delta_t_str:>8s} '
              f'{orig["feasibility_rate"]*100:>7.0f}% {ft.get("feasibility_rate",0)*100:>7.0f}%')

        if fc < float('inf'):
            total_orig_cost += oc
            total_ft_cost += fc
            total_orig_tard += ot
            total_ft_tard += ft_t

            if fc < oc * 0.99:
                n_better += 1
            elif fc > oc * 1.01:
                n_worse += 1
            else:
                n_same += 1

    # Totals
    n = len(orig_results)
    print('  ' + '-' * 96)
    print(f'  {"TOTAL/AVG":<16s} {"":<20s} '
          f'{total_orig_cost:>10.0f} {total_ft_cost:>10.0f} '
          f'{(total_ft_cost-total_orig_cost)/max(total_orig_cost,1)*100:>+7.1f}% '
          f'{total_orig_tard:>10.0f} {total_ft_tard:>10.0f} '
          f'{total_ft_tard-total_orig_tard:>+8.0f} '
          f'{">>>>>":>8s} {">>>>>":>8s}')

    print(f'\n  Instances improved: {n_better}/{n}')
    print(f'  Instances unchanged (±1%): {n_same}/{n}')
    print(f'  Instances degraded: {n_worse}/{n}')

    return {
        'n_better': n_better, 'n_worse': n_worse, 'n_same': n_same,
        'total_orig_cost': total_orig_cost, 'total_ft_cost': total_ft_cost,
        'total_orig_tard': total_orig_tard, 'total_ft_tard': total_ft_tard,
    }


def plot_comparison(orig_results, ft_results, output_dir):
    """Plot side-by-side comparison."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    instances = [r['instance'] for r in orig_results]
    orig_cost = [r['mean_cost'] for r in orig_results]
    ft_cost = [r['mean_cost'] for r in ft_results]
    orig_tard = [r['mean_tardiness'] for r in orig_results]
    ft_tard = [r['mean_tardiness'] for r in ft_results]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Cost comparison
    ax = axes[0]
    x = np.arange(len(instances))
    w = 0.35
    ax.bar(x - w/2, orig_cost, w, label='Original', color='#3498DB', alpha=0.8)
    ax.bar(x + w/2, ft_cost, w, label='Fine-Tuned', color='#E74C3C', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(instances, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean Cost')
    ax.set_title('Cost Comparison: Original vs Fine-Tuned')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Panel 2: Tardiness comparison
    ax = axes[1]
    ax.bar(x - w/2, orig_tard, w, label='Original', color='#3498DB', alpha=0.8)
    ax.bar(x + w/2, ft_tard, w, label='Fine-Tuned', color='#E74C3C', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(instances, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean Tardiness')
    ax.set_title('Tardiness Comparison: Original vs Fine-Tuned')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Week 6 P2: POMO Fine-Tuning — Benchmark Comparison',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'finetune_benchmark_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Plot saved: {path}')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=str,
                       default=os.path.join(_W4, 'algorithms', 'pomo', 'checkpoints', 'best_model.pt'))
    parser.add_argument('--finetuned', type=str,
                       default=os.path.join(_SCRIPT_DIR, 'checkpoints', 'best_finetuned.pt'))
    parser.add_argument('--runs', type=int, default=1,
                       help='Repeats per instance (default: 1, use 3 for stats)')
    parser.add_argument('--quick', action='store_true',
                       help='25c + 50c only (skip 100c)')
    parser.add_argument('--output-dir', type=str,
                       default=os.path.join(_SCRIPT_DIR, 'results'))
    args = parser.parse_args()

    if not os.path.exists(args.finetuned):
        print(f'ERROR: Fine-tuned model not found: {args.finetuned}')
        print('Run fine-tuning first: python week6/pomo_finetune.py')
        return

    print(f'Original:  {args.original}')
    print(f'Fine-tuned: {args.finetuned}')
    print(f'Repeats:   {args.runs}')

    # Build instance configs
    build_all_instances()

    sizes = [25, 50] if args.quick else [25, 50, 100]
    configs = []
    for src in RC1_INSTANCES + RC2_INSTANCES:
        for nc in sizes:
            inst_key = f'{src}_{nc}c'
            try:
                load_instance_from_disk(inst_key)
            except FileNotFoundError:
                continue
            n_trucks = 2 if nc <= 25 else 4
            tw_type = 'RC1' if src.startswith('RC1') else 'RC2'
            configs.append((inst_key, n_trucks, tw_type))

    print(f'Instances: {len(configs)}')

    # ── Run original model ──
    print(f'\n--- Original Model ---')
    orig_results = run_benchmark(args.original, 'Original', configs, n_runs=args.runs)

    # ── Run fine-tuned model ──
    print(f'\n--- Fine-Tuned Model ---')
    ft_results = run_benchmark(args.finetuned, 'Fine-Tuned', configs, n_runs=args.runs)

    # ── Compare ──
    summary = print_comparison(orig_results, ft_results)

    # ── Save ──
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_path = os.path.join(args.output_dir, f'finetune_comparison_{timestamp}.json')
    with open(result_path, 'w') as f:
        json.dump({
            'original': orig_results,
            'finetuned': ft_results,
            'summary': summary,
        }, f, indent=2)
    print(f'\nResults saved: {result_path}')

    # ── Plot ──
    plot_comparison(orig_results, ft_results,
                    os.path.join(_SCRIPT_DIR, 'visualizations'))


if __name__ == '__main__':
    main()
