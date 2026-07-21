#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 6 P3 Visualization — Baseline vs Full EDD vs Partial EDD comparison.
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


def load_latest_p3_results():
    """Find the most recent P3 result file."""
    files = sorted([f for f in os.listdir(RESULTS_DIR) if f.startswith('week6_p3_')])
    if not files:
        raise FileNotFoundError("No P3 results found in results/")
    path = os.path.join(RESULTS_DIR, files[-1])
    with open(path) as f:
        return json.load(f), files[-1]


def plot_three_way_cost_comparison(results, timestamp):
    """Panel 1: 3-way cost bar chart grouped by scale and TW type."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, tw_type in enumerate(['RC1', 'RC2']):
        ax = axes[ax_idx]
        categories = []
        baseline_costs, full_costs, partial_costs = [], [], []

        for nc in [25, 50, 100]:
            exps = [e for e in results if e['tw_type'] == tw_type and e['n_customers'] == nc]
            if not exps:
                continue
            categories.append(f'{nc}c')
            baseline_costs.append(np.mean([e['methods']['w5_baseline']['mean_cost'] for e in exps]))
            full_costs.append(np.mean([e['methods']['full_edd']['mean_cost'] for e in exps]))
            partial_costs.append(np.mean([e['methods']['partial_edd']['mean_cost'] for e in exps]))

        x = np.arange(len(categories))
        w = 0.25
        ax.bar(x - w, baseline_costs, w, label='W5 Baseline', color='#E74C3C', alpha=0.7)
        ax.bar(x, full_costs, w, label='Full EDD', color='#3498DB', alpha=0.8)
        ax.bar(x + w, partial_costs, w, label='Partial EDD (P3)', color='#27AE60', alpha=0.8)

        ax.set_xlabel('Problem Size')
        ax.set_ylabel('Mean Cost')
        ax.set_title(f'{tw_type} (Tight TW)' if tw_type == 'RC1' else f'{tw_type} (Wide TW)')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        # Annotate partial vs full delta
        for i in range(len(categories)):
            delta_pct = (partial_costs[i] - full_costs[i]) / max(full_costs[i], 1) * 100
            color = '#27AE60' if delta_pct <= 0 else '#E74C3C'
            ax.annotate(f'{delta_pct:+.1f}%', (x[i] + w, partial_costs[i]),
                      textcoords="offset points", xytext=(0, 5),
                      ha='center', fontsize=8, fontweight='bold', color=color)

    fig.suptitle('P3: 3-Way Cost Comparison — Baseline vs Full EDD vs Partial EDD',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(VIZ_DIR, 'p3_cost_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def plot_partial_vs_full_delta(results, timestamp):
    """Panel 2: Cost delta (partial - full) by instance scale, with explanation."""
    fig, ax = plt.subplots(figsize=(12, 6))

    instances = []
    deltas_pct = []
    fallback_rates = []
    colors = []

    for e in sorted(results, key=lambda x: (x['n_customers'], x['tw_type'], x['instance_key'])):
        fe = e['methods']['full_edd']
        pe = e['methods']['partial_edd']
        delta = (pe['mean_cost'] - fe['mean_cost']) / max(fe['mean_cost'], 1) * 100
        instances.append(e['instance_key'])
        deltas_pct.append(delta)

        # Fallback rate
        fr = pe.get('avg_fallback_count', 0)
        # Estimate number of routes
        n_routes = e['n_trucks']
        fallback_rates.append(min(fr / max(n_routes, 1), 1.0) * 100)

        colors.append('#27AE60' if delta <= 0 else '#E74C3C')

    x = np.arange(len(instances))
    bars = ax.bar(x, deltas_pct, 0.5, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Overlay fallback rate as text
    for i in range(len(instances)):
        if abs(deltas_pct[i]) > 0.1:
            ax.annotate(f'FB:{fallback_rates[i]:.0f}%', (x[i], deltas_pct[i]),
                      textcoords="offset points",
                      xytext=(0, 5 if deltas_pct[i] >= 0 else -12),
                      ha='center', fontsize=7, color='#7F8C8D')

    ax.axhline(y=0, color='black', linewidth=1.5, linestyle='-')
    ax.set_xticks(x)
    ax.set_xticklabels(instances, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Cost Change: Partial vs Full EDD (%)')
    ax.set_title('P3: Partial vs Full EDD — Cost Delta by Instance\n(Green=Partial Better, Red=Full Better, FB%=Fallback Rate)',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add region shading
    ax.axvspan(-0.5, 7.5, alpha=0.05, color='green', label='25c/50c: Partial wins')
    ax.axvspan(7.5, 11.5, alpha=0.05, color='red', label='100c: Full wins')
    ax.legend(fontsize=9, loc='lower left')

    plt.tight_layout()
    path = os.path.join(VIZ_DIR, 'p3_partial_vs_full_delta.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def plot_scale_analysis(results, timestamp):
    """Panel 3: Aggregated analysis by scale — when partial repair works."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    scales = [25, 50, 100]
    tw_types = ['RC1', 'RC2']

    # Subplot 1: Cost delta by scale
    ax = axes[0]
    for twt in tw_types:
        deltas = []
        for nc in scales:
            exps = [e for e in results if e['tw_type'] == twt and e['n_customers'] == nc]
            if exps:
                fe_c = np.mean([e['methods']['full_edd']['mean_cost'] for e in exps])
                pe_c = np.mean([e['methods']['partial_edd']['mean_cost'] for e in exps])
                deltas.append((pe_c - fe_c) / max(fe_c, 1) * 100)
            else:
                deltas.append(0)
        ax.plot(scales, deltas, 'o-', linewidth=2, markersize=8,
               label=f'{twt} ({"Tight" if twt=="RC1" else "Wide"} TW)')
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Problem Size (customers)')
    ax.set_ylabel('Cost Change: Partial vs Full (%)')
    ax.set_title('Cost Impact by Scale')
    ax.legend()
    ax.grid(alpha=0.3)

    # Subplot 2: Fallback rate by scale
    ax = axes[1]
    for twt in tw_types:
        fbs = []
        for nc in scales:
            exps = [e for e in results if e['tw_type'] == twt and e['n_customers'] == nc]
            if exps:
                avg_fb = np.mean([e['methods']['partial_edd'].get('avg_fallback_count', 0) for e in exps])
                n_trucks = exps[0]['n_trucks']
                fbs.append(avg_fb / n_trucks * 100)
            else:
                fbs.append(0)
        ax.plot(scales, fbs, 's-', linewidth=2, markersize=8, label=twt)
    ax.set_xlabel('Problem Size (customers)')
    ax.set_ylabel('Routes Needing Full Fallback (%)')
    ax.set_title('Fallback Rate by Scale')
    ax.legend()
    ax.grid(alpha=0.3)

    # Subplot 3: Segments repaired by scale
    ax = axes[2]
    for twt in tw_types:
        segs = []
        for nc in scales:
            exps = [e for e in results if e['tw_type'] == twt and e['n_customers'] == nc]
            if exps:
                avg_seg = np.mean([e['methods']['partial_edd'].get('avg_segments_repaired', 0) for e in exps])
                segs.append(avg_seg)
            else:
                segs.append(0)
        ax.plot(scales, segs, 'D-', linewidth=2, markersize=8, label=twt)
    ax.set_xlabel('Problem Size (customers)')
    ax.set_ylabel('Segments Repaired')
    ax.set_title('Repair Segments by Scale')
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle('P3: Scale-Dependent Analysis — Why Partial Repair Fails on 100c',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(VIZ_DIR, 'p3_scale_analysis.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def plot_summary_dashboard(results, timestamp):
    """Panel 4: Comprehensive P3 dashboard."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    categories = ['RC1\n25c', 'RC1\n50c', 'RC1\n100c', 'RC2\n25c', 'RC2\n50c', 'RC2\n100c']

    def get_vals(method, key):
        vals = []
        for cat in categories:
            tw, sc = cat.split('\n')
            nc = int(sc.replace('c', ''))
            exps = [e for e in results if e['tw_type'] == tw and e['n_customers'] == nc]
            if exps:
                vals.append(np.mean([e['methods'][method][key] for e in exps]))
            else:
                vals.append(0)
        return vals

    x = np.arange(len(categories))
    w = 0.25

    # (0,0): Cost comparison
    ax = axes[0, 0]
    ax.bar(x - w, get_vals('w5_baseline', 'mean_cost'), w, label='Baseline', color='#E74C3C', alpha=0.7)
    ax.bar(x, get_vals('full_edd', 'mean_cost'), w, label='Full EDD', color='#3498DB', alpha=0.8)
    ax.bar(x + w, get_vals('partial_edd', 'mean_cost'), w, label='Partial EDD', color='#27AE60', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylabel('Cost'); ax.set_title('Cost by Category')
    ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)

    # (0,1): Tardiness comparison
    ax = axes[0, 1]
    ax.bar(x - w, get_vals('w5_baseline', 'mean_tardiness'), w, label='Baseline', color='#E74C3C', alpha=0.7)
    ax.bar(x, get_vals('full_edd', 'mean_tardiness'), w, label='Full EDD', color='#3498DB', alpha=0.8)
    ax.bar(x + w, get_vals('partial_edd', 'mean_tardiness'), w, label='Partial EDD', color='#27AE60', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylabel('Tardiness'); ax.set_title('Tardiness by Category')
    ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)

    # (0,2): Partial vs Full cost delta %
    ax = axes[0, 2]
    deltas = []
    for cat in categories:
        tw, sc = cat.split('\n')
        nc = int(sc.replace('c', ''))
        exps = [e for e in results if e['tw_type'] == tw and e['n_customers'] == nc]
        if exps:
            fe = np.mean([e['methods']['full_edd']['mean_cost'] for e in exps])
            pe = np.mean([e['methods']['partial_edd']['mean_cost'] for e in exps])
            deltas.append((pe - fe) / max(fe, 1) * 100)
        else:
            deltas.append(0)
    colors = ['#27AE60' if d <= 0 else '#E74C3C' for d in deltas]
    ax.bar(x, deltas, 0.5, color=colors, alpha=0.8)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylabel('Cost Delta (%)'); ax.set_title('Partial vs Full: Cost Impact')
    for i, d in enumerate(deltas):
        ax.annotate(f'{d:+.1f}%', (x[i], d), textcoords="offset points",
                   xytext=(0, 5 if d >= 0 else -12), ha='center', fontsize=8, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # (1,0): Drone missions
    ax = axes[1, 0]
    ax.bar(x - w, get_vals('w5_baseline', 'avg_drone_missions'), w, label='Baseline', color='#E74C3C', alpha=0.7)
    ax.bar(x, get_vals('full_edd', 'avg_drone_missions'), w, label='Full EDD', color='#3498DB', alpha=0.8)
    ax.bar(x + w, get_vals('partial_edd', 'avg_drone_missions'), w, label='Partial EDD', color='#27AE60', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylabel('Drone Missions'); ax.set_title('Drone Utilization')
    ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)

    # (1,1): Fallback rate by category
    ax = axes[1, 1]
    fbs = []
    for cat in categories:
        tw, sc = cat.split('\n')
        nc = int(sc.replace('c', ''))
        exps = [e for e in results if e['tw_type'] == tw and e['n_customers'] == nc]
        if exps:
            avg_fb = np.mean([e['methods']['partial_edd'].get('avg_fallback_count', 0) for e in exps])
            n_trucks = exps[0]['n_trucks']
            fbs.append(avg_fb / n_trucks * 100)
        else:
            fbs.append(0)
    colors_fb = ['#F39C12' if f > 50 else '#27AE60' for f in fbs]
    ax.bar(x, fbs, 0.5, color=colors_fb, alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylabel('Fallback Rate (%)'); ax.set_title('Routes Needing Full Fallback')
    for i, f in enumerate(fbs):
        ax.annotate(f'{f:.0f}%', (x[i], f), textcoords="offset points",
                   xytext=(0, 5), ha='center', fontsize=8, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # (1,2): Segments repaired
    ax = axes[1, 2]
    segs = []
    for cat in categories:
        tw, sc = cat.split('\n')
        nc = int(sc.replace('c', ''))
        exps = [e for e in results if e['tw_type'] == tw and e['n_customers'] == nc]
        if exps:
            segs.append(np.mean([e['methods']['partial_edd'].get('avg_segments_repaired', 0) for e in exps]))
        else:
            segs.append(0)
    ax.bar(x, segs, 0.5, color='#9B59B6', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylabel('Count'); ax.set_title('Segments Repaired (Partial EDD)')
    for i, s in enumerate(segs):
        ax.annotate(f'{s:.0f}', (x[i], s), textcoords="offset points",
                   xytext=(0, 5), ha='center', fontsize=8, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Week 6 P3: Smarter Partial Repair — Full Dashboard',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(VIZ_DIR, 'p3_dashboard.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default=None)
    args = parser.parse_args()

    if args.results:
        with open(args.results) as f:
            results = json.load(f)
        timestamp = os.path.basename(args.results).replace('.json', '')
    else:
        results, fname = load_latest_p3_results()
        timestamp = fname.replace('.json', '')

    print(f'Loaded {len(results)} P3 experiment results')
    plot_three_way_cost_comparison(results, timestamp)
    plot_partial_vs_full_delta(results, timestamp)
    plot_scale_analysis(results, timestamp)
    plot_summary_dashboard(results, timestamp)
    print('Done!')


if __name__ == '__main__':
    main()
