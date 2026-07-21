#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 6 Visualization — W5 Pipeline vs W5 + IVND Repair comparison.
"""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_W6 = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_W6, 'results')
VIZ_DIR = os.path.join(_W6, 'visualizations')
os.makedirs(VIZ_DIR, exist_ok=True)


def load_latest_results():
    """Find the most recent week6 result file."""
    files = sorted([f for f in os.listdir(RESULTS_DIR) if f.startswith('week6_pipeline_')])
    if not files:
        raise FileNotFoundError("No results found in results/")
    path = os.path.join(RESULTS_DIR, files[-1])
    with open(path) as f:
        return json.load(f), files[-1]


def plot_tardiness_comparison(results, timestamp):
    """Bar chart: W5 baseline vs W5 + repair tardiness by instance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, tw_type in enumerate(['RC1', 'RC2']):
        ax = axes[ax_idx]
        exps = [e for e in results if e['tw_type'] == tw_type]
        if not exps:
            continue

        labels = []
        w5_tards, rp_tards = [], []
        w5_costs, rp_costs = [], []

        for e in sorted(exps, key=lambda x: (x['n_customers'], x['instance_key'])):
            b = e['methods']['w5_baseline']
            rp = e['methods']['w5_plus_repair']
            labels.append(f'{e["instance_key"]}')
            w5_tards.append(b['mean_tardiness'])
            rp_tards.append(rp['mean_tardiness'])
            w5_costs.append(b['mean_cost'])
            rp_costs.append(rp['mean_cost'])

        x = np.arange(len(labels))
        w = 0.35
        bars1 = ax.bar(x - w/2, w5_tards, w, label='W5 Baseline', color='#E74C3C', alpha=0.8)
        bars2 = ax.bar(x + w/2, rp_tards, w, label='W5 + IVND Repair', color='#27AE60', alpha=0.8)

        ax.set_xlabel('Instance')
        ax.set_ylabel('Tardiness')
        ax.set_title(f'{tw_type} (Tight TW)' if tw_type == 'RC1' else f'{tw_type} (Wide TW)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Annotate reduction %
        for i in range(len(labels)):
            if w5_tards[i] > 0:
                red = (w5_tards[i] - rp_tards[i]) / w5_tards[i] * 100
                ax.annotate(f'-{red:.0f}%', (x[i] + w/2, rp_tards[i]),
                          textcoords="offset points", xytext=(0, 5),
                          ha='center', fontsize=7, fontweight='bold', color='#27AE60')

    fig.suptitle('IVND Repair Effect on Tardiness', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(VIZ_DIR, 'repair_tardiness_bars.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def plot_cost_vs_tardiness_scatter(results, timestamp):
    """Scatter: cost vs tardiness before and after repair."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for e in results:
        b = e['methods']['w5_baseline']
        rp = e['methods']['w5_plus_repair']

        # Arrow from baseline to repair
        ax.annotate('', xy=(rp['mean_tardiness'], rp['mean_cost']),
                    xytext=(b['mean_tardiness'], b['mean_cost']),
                    arrowprops=dict(arrowstyle='->', color='#3498DB',
                                   lw=1.5, alpha=0.7))

        color = '#E74C3C' if e['tw_type'] == 'RC1' else '#2980B9'
        marker = 'o' if e['n_customers'] == 25 else ('s' if e['n_customers'] == 50 else 'D')
        size = 60 + e['n_customers'] * 0.5

        ax.scatter(b['mean_tardiness'], b['mean_cost'], c=color, marker=marker,
                  s=size, alpha=0.6, edgecolors='black', linewidth=0.5,
                  label='_nolegend_')
        ax.scatter(rp['mean_tardiness'], rp['mean_cost'], c=color, marker=marker,
                  s=size, alpha=1.0, edgecolors='black', linewidth=1.5,
                  label='_nolegend_')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C',
               markersize=8, label='RC1 (tight TW)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2980B9',
               markersize=8, label='RC2 (wide TW)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=6, label='25c'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
               markersize=8, label='50c'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
               markersize=10, label='100c'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.set_xlabel('Tardiness')
    ax.set_ylabel('Cost')
    ax.set_title('Cost vs Tardiness: W5 Baseline → W5 + Repair')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(VIZ_DIR, 'repair_cost_tardiness.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def plot_summary_dashboard(results, timestamp):
    """4-panel summary dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: Tardiness by scale and TW type
    ax = axes[0, 0]
    categories = ['RC1\n25c', 'RC1\n50c', 'RC1\n100c', 'RC2\n25c', 'RC2\n50c', 'RC2\n100c']
    w5_tards, rp_tards = [], []
    for cat in categories:
        tw, sc = cat.split('\n')
        nc = int(sc.replace('c', ''))
        exps = [e for e in results if e['tw_type'] == tw and e['n_customers'] == nc]
        if exps:
            w5_tards.append(np.mean([e['methods']['w5_baseline']['mean_tardiness'] for e in exps]))
            rp_tards.append(np.mean([e['methods']['w5_plus_repair']['mean_tardiness'] for e in exps]))
        else:
            w5_tards.append(0); rp_tards.append(0)

    x = np.arange(len(categories))
    ax.bar(x - 0.2, w5_tards, 0.35, label='W5 Baseline', color='#E74C3C', alpha=0.8)
    ax.bar(x + 0.2, rp_tards, 0.35, label='W5 + Repair', color='#27AE60', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(categories)
    ax.set_ylabel('Tardiness'); ax.set_title('Tardiness by Scale & TW Type')
    ax.legend(); ax.grid(axis='y', alpha=0.3)

    # Panel 2: Cost change
    ax = axes[0, 1]
    cost_increases = []
    for cat in categories:
        tw, sc = cat.split('\n')
        nc = int(sc.replace('c', ''))
        exps = [e for e in results if e['tw_type'] == tw and e['n_customers'] == nc]
        if exps:
            w5_c = np.mean([e['methods']['w5_baseline']['mean_cost'] for e in exps])
            rp_c = np.mean([e['methods']['w5_plus_repair']['mean_cost'] for e in exps])
            cost_increases.append((rp_c - w5_c) / max(w5_c, 1) * 100)
        else:
            cost_increases.append(0)

    colors = ['#E74C3C' if v > 30 else '#F39C12' if v > 15 else '#27AE60' for v in cost_increases]
    ax.bar(x, cost_increases, 0.5, color=colors, alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(categories)
    ax.set_ylabel('Cost Increase (%)'); ax.set_title('Cost Increase from Repair')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)

    # Panel 3: Tardiness reduction %
    ax = axes[1, 0]
    reductions = []
    for cat in categories:
        tw, sc = cat.split('\n')
        nc = int(sc.replace('c', ''))
        exps = [e for e in results if e['tw_type'] == tw and e['n_customers'] == nc]
        if exps:
            w5_t = np.mean([e['methods']['w5_baseline']['mean_tardiness'] for e in exps])
            rp_t = np.mean([e['methods']['w5_plus_repair']['mean_tardiness'] for e in exps])
            reductions.append((w5_t - rp_t) / max(w5_t, 1) * 100)
        else:
            reductions.append(0)

    ax.bar(x, reductions, 0.5, color='#27AE60', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(categories)
    ax.set_ylabel('Tardiness Reduction (%)'); ax.set_title('Tardiness Reduction from Repair')
    for i, v in enumerate(reductions):
        ax.annotate(f'{v:.0f}%', (x[i], v), textcoords="offset points",
                   xytext=(0, 5), ha='center', fontweight='bold', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Panel 4: Runtime comparison
    ax = axes[1, 1]
    w5_times, rp_times = [], []
    for cat in categories:
        tw, sc = cat.split('\n')
        nc = int(sc.replace('c', ''))
        exps = [e for e in results if e['tw_type'] == tw and e['n_customers'] == nc]
        if exps:
            w5_times.append(np.mean([e['methods']['w5_baseline'].get('mean_runtime', 0.5) for e in exps]))
            rp_times.append(np.mean([e['methods']['w5_plus_repair'].get('mean_runtime', 0.7) for e in exps]))
        else:
            w5_times.append(0); rp_times.append(0)

    ax.bar(x - 0.2, w5_times, 0.35, label='W5 Baseline', color='#3498DB', alpha=0.8)
    ax.bar(x + 0.2, rp_times, 0.35, label='W5 + Repair', color='#9B59B6', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(categories)
    ax.set_ylabel('Runtime (s)'); ax.set_title('Runtime Comparison')
    ax.legend(); ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Week 6: Pipeline + IVND Repair — Summary Dashboard', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(VIZ_DIR, 'repair_dashboard.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default=None,
                       help='Path to results JSON')
    args = parser.parse_args()

    if args.results:
        with open(args.results) as f:
            results = json.load(f)
        timestamp = os.path.basename(args.results).replace('.json', '')
    else:
        results, fname = load_latest_results()
        timestamp = fname.replace('.json', '')

    print(f'Loaded {len(results)} experiment results')
    plot_tardiness_comparison(results, timestamp)
    plot_cost_vs_tardiness_scatter(results, timestamp)
    plot_summary_dashboard(results, timestamp)
    print('Done!')


if __name__ == '__main__':
    main()
