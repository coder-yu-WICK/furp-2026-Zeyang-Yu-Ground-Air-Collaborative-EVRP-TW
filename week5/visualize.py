#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 5 Visualization — Ablation Study Plots.

Generates:
  1. Ablation comparison: 4 variants × cost/tardiness/feasibility/runtime
  2. TW-aware clustering effect: tardiness reduction vs beta parameter
  3. Drone impact: cost savings from drone post-processing
  4. Pareto scatter: 4 variants + Week 3 methods on representative configs
  5. Route maps: TW-aware vs spatial-only clustering comparison
  6. Summary dashboard

Usage:
    python visualize.py                          # All plots
    python visualize.py --quick                  # Representative subset
    python visualize.py --results <path.json>    # Specific results file
"""

import json, os, sys, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_W5 = os.path.dirname(os.path.abspath(__file__))
_W4 = os.path.join(_W5, '..', 'week4')
_W3 = os.path.join(_W5, '..', 'week3')
sys.path.insert(0, _W5)
sys.path.insert(0, _W4)

VIS_DIR = os.path.join(_W5, 'visualizations')
os.makedirs(VIS_DIR, exist_ok=True)

# ── Color Scheme ─────────────────────────────────────────────────────

COLORS = {
    'baseline':       '#9b59b6',  # purple (Week 4 POMO-MT)
    'tw_aware':       '#1abc9c',  # teal
    'drone_only':     '#e67e22',  # orange
    'tw_aware_drone': '#27ae60',  # green
    'adaptive_tw':    '#2980b9',  # blue
    'adaptive_tw_drone': '#16a085', # dark teal
    'angle':          '#c0392b',  # dark red
    'angle_drone':    '#d35400',  # dark orange
    'hybrid':         '#8e44ad',  # dark purple
    'hybrid_drone':   '#2c3e50',  # dark blue-gray
    'No-Drone':       '#f39c12',  # yellow-orange
    'P-ACO':          '#2ecc71',  # green
    'NSGA-II':        '#3498db',  # blue
    'IVND':           '#e74c3c',  # red
}

VARIANT_LABELS = {
    'baseline':       'Baseline (spatial, no drones)',
    'tw_aware':       'TW-Aware (spatio-temporal, no drones)',
    'drone_only':     'Drone-PP (spatial + drones)',
    'tw_aware_drone': 'TW-Aware+Drone',
    'adaptive_tw':    'Adaptive TW (auto-threshold)',
    'adaptive_tw_drone': 'Adaptive TW+Drone+ReOpt',
    'angle':          'Angle-Based (petal)',
    'angle_drone':    'Angle+Drone',
    'hybrid':         'Hybrid (auto-select)',
    'hybrid_drone':   'Hybrid+Drone ★',
}


# ── Data Loading ─────────────────────────────────────────────────────

def load_results(results_path):
    """Load experiment results JSON."""
    with open(results_path) as f:
        return json.load(f)


def aggregate_by_scale(results, methods):
    """Aggregate results by customer scale for each method."""
    scales = {25: [], 50: [], 100: []}
    for exp in results:
        nc = exp['n_customers']
        for method in methods:
            m = exp['methods'].get(method)
            if m and m.get('feasibility_rate', 0) > 0:
                scales[nc].append({
                    'method': method,
                    'cost': m['mean_cost'],
                    'tardiness': m['mean_tardiness'],
                    'feasibility': m['feasibility_rate'],
                    'runtime': m.get('mean_runtime', 0),
                    'hypervolume': m.get('hypervolume', 0),
                    'pareto_size': m.get('pareto_size', 0),
                    'drone_missions': m.get('avg_drone_missions', 0),
                })
    return scales


# ── Plot 1: Ablation Bar Charts ──────────────────────────────────────

def plot_ablation_bars(results, methods, save_path):
    """Grouped bar charts: cost, tardiness, feasibility, runtime by scale."""
    scales_data = aggregate_by_scale(results, methods)
    scale_labels = ['25 customers', '50 customers', '100 customers']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    metrics = [
        ('cost', 'Mean Cost', axes[0, 0]),
        ('tardiness', 'Mean Tardiness', axes[0, 1]),
        ('feasibility', 'Feasibility Rate (%)', axes[1, 0]),
        ('runtime', 'Mean Runtime (s)', axes[1, 1]),
    ]

    x = np.arange(len(scale_labels))
    width = 0.18
    n_methods = len(methods)

    for metric_name, ylabel, ax in metrics:
        for i, method in enumerate(methods):
            values = []
            for nc in [25, 50, 100]:
                entries = [e[metric_name] for e in scales_data[nc] if e['method'] == method]
                values.append(np.mean(entries) if entries else 0)

            offset = (i - n_methods/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=VARIANT_LABELS.get(method, method),
                         color=COLORS.get(method, '#888888'), edgecolor='white', linewidth=0.5)

            # Annotate values
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                           f'{val:.0f}' if metric_name != 'feasibility' else f'{val*100:.0f}%',
                           ha='center', va='bottom', fontsize=7, rotation=90)

        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(scale_labels)
        if metric_name == 'runtime':
            ax.set_yscale('log')
        if metric_name == 'cost':
            ax.legend(fontsize=8, loc='upper left')

    fig.suptitle('Week 5 Ablation Study: POMO-MT Variants by Scale', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {save_path}')


# ── Plot 2: TW-Aware Effect (Tardiness Reduction) ────────────────────

def plot_tw_aware_effect(results, save_path):
    """Scatter: tardiness reduction (TW-aware vs baseline) by config."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, tw_type in enumerate(['RC1', 'RC2']):
        ax = axes[ax_idx]
        points_25, points_50, points_100 = [], [], []

        for exp in results:
            if exp['tw_type'] != tw_type:
                continue
            base = exp['methods'].get('baseline', {})
            tw = exp['methods'].get('tw_aware', {})
            if not base or not tw:
                continue

            base_tard = base.get('mean_tardiness', 0)
            tw_tard = tw.get('mean_tardiness', 0)
            base_cost = base.get('mean_cost', 0)
            tw_cost = tw.get('mean_cost', 0)

            if base_tard > 0:
                reduction = (base_tard - tw_tard) / base_tard * 100
                cost_change = (tw_cost - base_cost) / base_cost * 100
                pt = (cost_change, reduction)
                if exp['n_customers'] == 25:
                    points_25.append(pt)
                elif exp['n_customers'] == 50:
                    points_50.append(pt)
                else:
                    points_100.append(pt)

        for pts, label, marker, color in [
            (points_25, '25c', 'o', '#3498db'),
            (points_50, '50c', 's', '#2ecc71'),
            (points_100, '100c', '^', '#e74c3c'),
        ]:
            if pts:
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, c=color, marker=marker, label=label, alpha=0.7, s=60)

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xlabel('Cost Change (%)')
        ax.set_ylabel('Tardiness Reduction (%)')
        ax.set_title(f'{tw_type} ({"Tight" if tw_type=="RC1" else "Wide"} TW)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Effect of TW-Aware Clustering: Tardiness Reduction vs Cost Change',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {save_path}')


# ── Plot 3: Drone Impact ─────────────────────────────────────────────

def plot_drone_impact(results, save_path):
    """Bar chart: cost savings from drone post-processing."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax_idx, nc in enumerate([25, 50, 100]):
        ax = axes[ax_idx]
        labels, savings_base, savings_tw = [], [], []

        for exp in results:
            if exp['n_customers'] != nc:
                continue
            base = exp['methods'].get('baseline', {})
            drone = exp['methods'].get('drone_only', {})
            tw_drone = exp['methods'].get('tw_aware_drone', {})

            if base and drone:
                saving = (base.get('mean_cost', 0) - drone.get('mean_cost', 0))
                savings_base.append(saving)
            if base and tw_drone:
                saving_tw = (base.get('mean_cost', 0) - tw_drone.get('mean_cost', 0))
                savings_tw.append(saving_tw)
            labels.append(exp['label'][:25])

        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w/2, savings_base, w, label='Drone-PP (spatial)', color=COLORS['drone_only'])
        ax.bar(x + w/2, savings_tw, w, label='TW-Aware+Drone', color=COLORS['tw_aware_drone'])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_ylabel('Cost Saving')
        ax.set_title(f'{nc} Customers')
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        if ax_idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle('Cost Savings from Drone Post-Processing', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {save_path}')


# ── Plot 4: Multi-Method Pareto Scatter ──────────────────────────────

def plot_pareto_scatter(results, save_dir):
    """Pareto scatter for representative configs: all variants + Week 3 methods."""
    all_methods = ['baseline', 'tw_aware', 'drone_only', 'tw_aware_drone',
                   'No-Drone', 'P-ACO', 'NSGA-II', 'IVND']

    # Select representative configs
    representative = [
        ('RC101', 25, 'RC1', 'medium', '2T+2D'),
        ('RC101', 50, 'RC1', 'medium', '4T+4D'),
        ('RC201', 25, 'RC2', 'medium', '2T+2D'),
        ('RC201', 50, 'RC2', 'medium', '4T+4D'),
    ]

    for src_inst, nc, tw, end, fleet in representative:
        label = f'{nc}c_{tw}_{end}_{fleet}'
        exp = next((e for e in results
                    if e['source_instance'] == src_inst
                    and e['n_customers'] == nc
                    and e['endurance_name'] == end
                    and f'{e["n_trucks"]}T+{e["n_drones"]}D' == fleet), None)

        if not exp:
            continue

        fig, ax = plt.subplots(figsize=(10, 8))

        for method in all_methods:
            m = exp['methods'].get(method, {})
            pts = m.get('pareto_points', [])
            if not pts:
                # Use mean as single point
                if m.get('mean_cost') and m.get('mean_tardiness'):
                    pts = [(m['mean_cost'], m['mean_tardiness'])]

            if pts:
                xs, ys = zip(*pts)
                marker = 'D' if method in VARIANT_LABELS else 'o'
                size = 80 if method in VARIANT_LABELS else 40
                alpha = 0.9 if method in VARIANT_LABELS else 0.5
                ax.scatter(xs, ys, c=COLORS.get(method, '#888'), marker=marker,
                          s=size, alpha=alpha, edgecolors='white', linewidth=0.5,
                          label=VARIANT_LABELS.get(method, method))

        # Joint Pareto front
        all_pts = []
        for method in all_methods:
            m = exp['methods'].get(method, {})
            for c, t in m.get('pareto_points', []):
                all_pts.append((c, t))
        if all_pts:
            all_pts.sort()
            nondom = []
            best_t = float('inf')
            for c, t in all_pts:
                if t < best_t:
                    nondom.append((c, t))
                    best_t = t
            if nondom:
                xs, ys = zip(*nondom)
                ax.plot(xs, ys, 'k--', linewidth=1.5, alpha=0.5, label='Joint Pareto')

        ax.set_xlabel('Cost')
        ax.set_ylabel('Tardiness')
        ax.set_title(f'{src_inst} — {nc} customers, {tw}, {fleet}')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.2)

        save_path = os.path.join(save_dir, f'pareto_w5_{src_inst}_{label}.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {save_path}')


# ── Plot 5: Cost-Tardiness Trade-off ─────────────────────────────────

def plot_tradeoff(results, methods, save_path):
    """Scatter plot: cost vs tardiness for all methods (aggregated)."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for method in methods:
        costs, tards, scales = [], [], []
        for exp in results:
            m = exp['methods'].get(method)
            if m and m.get('mean_cost', 0) > 0:
                costs.append(m['mean_cost'])
                tards.append(m['mean_tardiness'])
                scales.append(exp['n_customers'])

        if costs:
            sizes = [s*2 for s in scales]
            ax.scatter(costs, tards, c=COLORS.get(method, '#888'),
                      s=sizes, alpha=0.6, edgecolors='white', linewidth=0.5,
                      label=VARIANT_LABELS.get(method, method))

            # Draw arrow showing scale trend
            for nc in [25, 50, 100]:
                pts = [(c, t) for c, t, s in zip(costs, tards, scales) if s == nc]
                if pts:
                    cx, cy = np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts])
                    ax.annotate(f'{nc}c', (cx, cy), textcoords="offset points",
                              xytext=(5, 5), fontsize=8, alpha=0.7)

    ax.set_xlabel('Mean Cost')
    ax.set_ylabel('Mean Tardiness')
    ax.set_title('Cost-Tardiness Trade-off: All Methods', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {save_path}')


# ── Plot 6: Summary Dashboard ────────────────────────────────────────

def plot_summary_dashboard(results, methods, save_path):
    """4-panel summary: cost, tardiness, feasibility, hypervolume by scale."""
    scales_data = aggregate_by_scale(results, methods)
    scale_labels = ['25', '50', '100']
    metric_names = ['cost', 'tardiness', 'feasibility', 'hypervolume']
    titles = ['Mean Cost', 'Mean Tardiness', 'Feasibility Rate', 'Hypervolume']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for ax, metric, title in zip(axes.flat, metric_names, titles):
        x = np.arange(len(scale_labels))
        n = len(methods)
        width = 0.7 / n

        for i, method in enumerate(methods):
            means, stds = [], []
            for nc in [25, 50, 100]:
                vals = [e[metric] for e in scales_data[nc] if e['method'] == method]
                means.append(np.mean(vals) if vals else 0)
                stds.append(np.std(vals) if vals else 0)

            offset = (i - n/2 + 0.5) * width
            ax.bar(x + offset, means, width, yerr=stds,
                  color=COLORS.get(method, '#888'),
                  label=VARIANT_LABELS.get(method, method),
                  edgecolor='white', linewidth=0.5, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(scale_labels)
        ax.set_xlabel('Customers')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.2, axis='y')
        if metric == 'cost':
            ax.legend(fontsize=7)

    fig.suptitle('Week 5 Summary Dashboard: POMO-MT Ablation Study',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {save_path}')


# ── Plot 7: Feasibility & Runtime Comparison ─────────────────────────

def plot_feasibility_runtime(results, methods, save_path):
    """Feasibility and runtime comparison."""
    scales_data = aggregate_by_scale(results, methods)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Feasibility
    ax = axes[0]
    scales = [25, 50, 100]
    x = np.arange(len(scales))
    width = 0.7 / len(methods)
    for i, method in enumerate(methods):
        vals = []
        for nc in scales:
            entries = [e['feasibility'] for e in scales_data[nc] if e['method'] == method]
            vals.append(np.mean(entries)*100 if entries else 0)
        offset = (i - len(methods)/2 + 0.5) * width
        ax.bar(x + offset, vals, width, color=COLORS.get(method, '#888'),
              label=VARIANT_LABELS.get(method, method), edgecolor='white')
    ax.set_ylabel('Feasibility Rate (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}c' for s in scales])
    ax.set_title('Feasibility Rate')
    ax.legend(fontsize=7)
    ax.set_ylim(0, 105)

    # Runtime
    ax = axes[1]
    for i, method in enumerate(methods):
        vals = []
        for nc in scales:
            entries = [e['runtime'] for e in scales_data[nc] if e['method'] == method]
            vals.append(np.mean(entries) if entries else 0)
        offset = (i - len(methods)/2 + 0.5) * width
        ax.bar(x + offset, vals, width, color=COLORS.get(method, '#888'),
              label=VARIANT_LABELS.get(method, method), edgecolor='white')
    ax.set_ylabel('Mean Runtime (s)')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}c' for s in scales])
    ax.set_title('Runtime (log scale)')

    fig.suptitle('Feasibility & Runtime: POMO-MT Variants', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {save_path}')


# ── Route Maps ───────────────────────────────────────────────────────

def plot_route_maps(results, save_dir):
    """Generate route maps comparing TW-aware vs baseline clustering."""
    # Route maps require re-running solvers to extract actual routes
    # For now, we note this in the report as a future enhancement
    print('  Route maps: requires re-running solvers (see visualize.py --routes)')


# ── Main ─────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default=None,
                       help='Path to results JSON')
    parser.add_argument('--quick', action='store_true',
                       help='Representative subset only')
    parser.add_argument('--routes', action='store_true',
                       help='Generate route maps (re-runs solvers)')
    args = parser.parse_args()

    # Find results file
    if args.results:
        results_path = args.results
    else:
        results_dir = os.path.join(_W5, 'results')
        json_files = sorted([f for f in os.listdir(results_dir) if f.endswith('.json')])
        if not json_files:
            print("No results found. Run experiments first: python run_experiments.py --quick")
            return
        results_path = os.path.join(results_dir, json_files[-1])

    print(f'Loading results: {results_path}')
    results = load_results(results_path)
    print(f'  {len(results)} configurations')

    # Methods to compare
    w5_variants = [v for v in ['baseline', 'tw_aware', 'drone_only', 'tw_aware_drone',
                                'adaptive_tw', 'adaptive_tw_drone',
                                'angle', 'angle_drone', 'hybrid', 'hybrid_drone',
                                'hybrid_drone_no_reopt']
                   if any(v in exp['methods'] for exp in results)]
    w3_methods = [m for m in ['No-Drone', 'P-ACO', 'NSGA-II', 'IVND']
                  if any(m in exp['methods'] for exp in results)]

    print(f'  Week 5 variants found: {w5_variants}')
    print(f'  Week 3 methods found: {w3_methods}')

    # Generate plots
    if w5_variants:
        plot_ablation_bars(results, w5_variants,
                          os.path.join(VIS_DIR, 'ablation_bars.png'))
        plot_summary_dashboard(results, w5_variants,
                              os.path.join(VIS_DIR, 'summary_dashboard.png'))
        plot_feasibility_runtime(results, w5_variants,
                                os.path.join(VIS_DIR, 'feasibility_runtime.png'))
        plot_tradeoff(results, w5_variants + w3_methods,
                     os.path.join(VIS_DIR, 'cost_tardiness_tradeoff.png'))

    if 'tw_aware' in w5_variants and 'baseline' in w5_variants:
        plot_tw_aware_effect(results,
                            os.path.join(VIS_DIR, 'tw_aware_effect.png'))

    if any(v in w5_variants for v in ['drone_only', 'tw_aware_drone']):
        plot_drone_impact(results,
                         os.path.join(VIS_DIR, 'drone_impact.png'))

    # Pareto scatter for representative configs
    all_methods = w5_variants + w3_methods
    if all_methods:
        plot_pareto_scatter(results, VIS_DIR)

    if args.routes:
        plot_route_maps(results, VIS_DIR)

    print(f'\nDone! {len(os.listdir(VIS_DIR))} files in {VIS_DIR}/')


if __name__ == '__main__':
    main()
