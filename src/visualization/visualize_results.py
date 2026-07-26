#!/usr/bin/env python3
"""
Visualization script for FURP 2026 Workshop Paper.
Generates publication-quality plots from experiment results.
"""

import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from collections import defaultdict

_BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_BASE, '..', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style setup
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

# Color scheme
METHOD_COLORS = {
    'Ours (2-Drone)': '#2196F3',
    'Ours (1-Drone)': '#64B5F6',
    'Ours (No Drone)': '#BBDEFB',
    'Ours (No EDD)': '#FF9800',
    'Ours (Partial EDD)': '#FFB74D',
    'CW-Savings': '#4CAF50',
    'Sweep+NN': '#9E9E9E',
    'KM+NN': '#9E9E9E',
    'KM+2opt': '#9E9E9E',
    'NSGA-II': '#F44336',
    'P-ACO': '#E91E63',
    'IVND': '#9C27B0',
}

METHOD_SHORT = {
    'ours_full': 'Ours (2-Drone)',
    'ours_1drone': 'Ours (1-Drone)',
    'ours_no_drone': 'Ours (No Drone)',
    'ours_no_edd': 'Ours (No EDD)',
    'ours_partial_edd': 'Ours (Partial EDD)',
    'cw_savings': 'CW-Savings',
    'sweep_nn': 'Sweep+NN',
    'kmeans_nn': 'KM+NN',
    'kmeans_2opt': 'KM+2opt',
    'nsga2': 'NSGA-II',
    'paco': 'P-ACO',
    'ivnd': 'IVND',
}

INSTANCE_LABELS = {
    'RC101': 'RC101\n(RC1)', 'RC201': 'RC201\n(RC2)',
    'R101': 'R101\n(R1)', 'R201': 'R201\n(R2)',
    'C101': 'C101\n(C1)', 'C201': 'C201\n(C2)',
}


def load_results(json_path):
    with open(json_path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Method Comparison -- Cost + Feasibility
# ═══════════════════════════════════════════════════════════════════════════

def plot_method_comparison(results, size_label, output_name):
    """Bar chart: cost by method x instance, colored by feasibility."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    our_methods = ['ours_full', 'ours_1drone', 'ours_no_drone']
    baseline_methods = ['cw_savings', 'sweep_nn']
    all_plot_methods = our_methods + baseline_methods

    for ai, entry in enumerate(results):
        ax = axes[ai]
        ik = entry['instance_key']
        tw = entry['tw_type']
        src = entry['source_instance']
        methods = entry['methods']

        x_labels = []
        costs = []
        feas_colors = []
        for mk in all_plot_methods:
            if mk not in methods:
                continue
            m = methods[mk]
            if m['mean_cost'] >= 1e8:
                continue
            short = METHOD_SHORT.get(mk, mk)
            x_labels.append(short)
            costs.append(m['mean_cost'])
            feas_colors.append('#2E7D32' if m['feasibility_rate'] >= 0.99 else '#C62828')

        x = np.arange(len(x_labels))
        bars = ax.bar(x, costs, color=feas_colors, edgecolor='white', linewidth=0.5)

        # Annotate with cost values
        for bar, cost in zip(bars, costs):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(costs)*0.02,
                    f'{cost:.0f}', ha='center', va='bottom', fontsize=7, rotation=45)

        ax.set_title(f'{src} ({tw})', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Cost (distance units)')
        ax.grid(axis='y', alpha=0.3)

        # Add feasibility legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2E7D32', label='100% Feasible'),
            Patch(facecolor='#C62828', label='Infeasible'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=7)

    fig.suptitle(f'Method Comparison -- {size_label}', fontweight='bold', fontsize=15)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, output_name)
    fig.savefig(out_path)
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Drone Impact by Instance Type
# ═══════════════════════════════════════════════════════════════════════════

def plot_drone_impact(results, size_label, output_name):
    """Show cost savings from drones by instance type."""
    fig, ax = plt.subplots(figsize=(10, 6))

    types = []
    savings_2d = []
    savings_1d = []

    for entry in results:
        tw = entry['tw_type']
        methods = entry['methods']

        full = methods.get('ours_full', {})
        nd = methods.get('ours_no_drone', {})
        one = methods.get('ours_1drone', {})

        fc = full.get('mean_cost', 0)
        nc = nd.get('mean_cost', 0)
        oc = one.get('mean_cost', 0)

        if nc > 0:
            s2 = (nc - fc) / nc * 100
            s1 = (nc - oc) / nc * 100
        else:
            s2 = s1 = 0

        types.append(tw)
        savings_2d.append(s2)
        savings_1d.append(s1)

    x = np.arange(len(types))
    width = 0.35

    bars1 = ax.bar(x - width/2, savings_1d, width, label='1 Drone/Truck',
                   color='#64B5F6', edgecolor='white')
    bars2 = ax.bar(x + width/2, savings_2d, width, label='2 Drones/Truck',
                   color='#1565C0', edgecolor='white')

    # Annotate
    for bar, val in zip(bars1, savings_1d):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, savings_2d):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(types, fontsize=11)
    ax.set_ylabel('Cost Savings vs No-Drone (%)', fontsize=12)
    ax.set_title(f'Drone Impact by Instance Type -- {size_label}', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.8)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, output_name)
    fig.savefig(out_path)
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Pipeline Ablation
# ═══════════════════════════════════════════════════════════════════════════

def plot_pipeline_ablation(results, size_label, output_name):
    """Show contribution of each pipeline component."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    ablation_methods = ['ours_no_edd', 'ours_partial_edd', 'ours_no_drone',
                        'ours_1drone', 'ours_full']
    labels = ['No EDD\n(POMO raw)', 'Partial\nEDD', 'Full EDD\n(No Drone)',
              '+1 Drone', '+2 Drones\n(Full)']

    for ai, entry in enumerate(results):
        ax = axes[ai]
        src = entry['source_instance']
        tw = entry['tw_type']
        methods = entry['methods']

        costs_ablation = []
        feases = []
        for mk in ablation_methods:
            m = methods.get(mk, {})
            costs_ablation.append(m.get('mean_cost', 0) if m.get('mean_cost', 1e9) < 1e8 else 0)
            feases.append(m.get('feasibility_rate', 0))

        x = np.arange(len(labels))
        colors = ['#FF9800', '#FFB74D', '#BBDEFB', '#64B5F6', '#1565C0']
        bars = ax.bar(x, costs_ablation, color=colors, edgecolor='white')

        for bar, cost, feas in zip(bars, costs_ablation, feases):
            if cost > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(costs_ablation)*0.02,
                        f'{cost:.0f}', ha='center', fontsize=7, rotation=45)
            if feas < 0.99:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 0.5,
                        'x', ha='center', fontsize=14, color='red', fontweight='bold')

        ax.set_title(f'{src} ({tw})', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, fontsize=8)
        ax.set_ylabel('Cost')
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f'Pipeline Ablation -- {size_label}', fontweight='bold', fontsize=15)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, output_name)
    fig.savefig(out_path)
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: EV Ablation
# ═══════════════════════════════════════════════════════════════════════════

def plot_ev_ablation(results_200c, output_name):
    """EV Model A vs B vs C comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    types = []
    costs_a, costs_b, costs_c = [], [], []
    ev_feas_b, ev_feas_c = [], []

    for entry in results_200c:
        tw = entry['tw_type']
        methods = entry.get('methods', {})
        ev = entry.get('ev_methods', {})

        base = methods.get('ours_full', {}).get('mean_cost', 0)
        evl = ev.get('ev_linear', {})
        evn = ev.get('ev_nonlinear', {})

        if base > 0 and base < 1e8:
            types.append(tw)
            costs_a.append(base)
            costs_b.append(evl.get('mean_cost', 0))
            costs_c.append(evn.get('mean_cost', 0))
            ev_feas_b.append(evl.get('ev_feasibility_rate', 0) * 100)
            ev_feas_c.append(evn.get('ev_feasibility_rate', 0) * 100)

    # Cost comparison
    x = np.arange(len(types))
    width = 0.25
    ax1.bar(x - width, costs_a, width, label='Model A (No EV)', color='#BBDEFB')
    ax1.bar(x, costs_b, width, label='Model B (Linear)', color='#42A5F5')
    ax1.bar(x + width, costs_c, width, label='Model C (Non-linear)', color='#1565C0')

    ax1.set_xticks(x)
    ax1.set_xticklabels(types)
    ax1.set_ylabel('Cost')
    ax1.set_title('Cost Comparison: EV Models')
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.3)

    # EV feasibility
    ax2.bar(x - width/2, ev_feas_b, width, label='Model B (Linear)', color='#42A5F5')
    ax2.bar(x + width/2, ev_feas_c, width, label='Model C (Non-linear)', color='#1565C0')
    ax2.set_xticks(x)
    ax2.set_xticklabels(types)
    ax2.set_ylabel('EV Feasibility (%)')
    ax2.set_title('EV Feasibility Rate')
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('EV Ablation Study -- 200c', fontweight='bold', fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, output_name)
    fig.savefig(out_path)
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5: Scale Comparison (50c vs 100c vs 200c)
# ═══════════════════════════════════════════════════════════════════════════

def plot_scale_comparison(results_dict, output_name):
    """Compare our method across scales (50c, 100c, 200c)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    instance_types = ['RC101', 'RC201', 'R101', 'R201', 'C101', 'C201']

    for ai, itype in enumerate(instance_types):
        ax = axes[ai]
        scales = []
        costs = []

        for size_label, results in results_dict.items():
            for entry in results:
                if entry['source_instance'] == itype:
                    full = entry['methods'].get('ours_full', {})
                    if full.get('mean_cost', 1e9) < 1e8:
                        scales.append(size_label)
                        costs.append(full['mean_cost'])

        if scales and costs:
            ax.bar(scales, costs, color='#2196F3', edgecolor='white')
            for i, (s, c) in enumerate(zip(scales, costs)):
                ax.text(i, c + max(costs)*0.03, f'{c:.0f}', ha='center', fontsize=9)

        ax.set_title(itype, fontweight='bold')
        ax.set_ylabel('Cost')
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Our Method (2-Drone) Across Scales', fontweight='bold', fontsize=15)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, output_name)
    fig.savefig(out_path)
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6: Summary Table (Text)
# ═══════════════════════════════════════════════════════════════════════════

def generate_summary_table(results, size_label, output_name):
    """Generate a clean markdown summary table."""
    lines = []
    lines.append(f'## {size_label} Results Summary')
    lines.append('')
    lines.append('| Instance | Ours (2D) | Ours (1D) | Ours (ND) | CW-Sav | Feas% | DroneD% |')
    lines.append('|----------|-----------|-----------|-----------|--------|-------|---------|')

    for entry in results:
        ik = entry['instance_key']
        m = entry['methods']
        full = m.get('ours_full', {})
        one = m.get('ours_1drone', {})
        nd = m.get('ours_no_drone', {})
        cw = m.get('cw_savings', {})

        fc = full.get('mean_cost', 0)
        oc = one.get('mean_cost', 0)
        nc = nd.get('mean_cost', 0)
        cc = cw.get('mean_cost', 0)
        feas = full.get('feasibility_rate', 0) * 100
        drone_delta = (nc - fc) / nc * 100 if nc > 0 else 0

        lines.append(f'| {ik} | {fc:.0f} | {oc:.0f} | {nc:.0f} | {cc:.0f} | {feas:.0f}% | {drone_delta:+.1f}% |')

    table_text = '\n'.join(lines)

    out_path = os.path.join(OUTPUT_DIR, output_name)
    with open(out_path, 'w') as f:
        f.write(table_text)
    print(f'  Saved: {out_path}')


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print('GENERATING VISUALIZATIONS')
    print('=' * 60)

    # Load available result files
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')

    # 200c results (latest)
    v2_200c_files = sorted(
        [f for f in os.listdir(results_dir) if f.startswith('week7_tier0_200c_') and f.endswith('.json')],
        reverse=True)
    v2_200c = os.path.join(results_dir, v2_200c_files[0]) if v2_200c_files else None
    if v2_200c and os.path.exists(v2_200c):
        r_200c = load_results(v2_200c)
        print(f'\nLoaded 200c: {len(r_200c)} instances')

        # Figure 1: Method comparison
        plot_method_comparison(r_200c, '200 Customers', 'fig1_method_comparison_200c.png')

        # Figure 2: Drone impact
        plot_drone_impact(r_200c, '200 Customers', 'fig2_drone_impact_200c.png')

        # Figure 3: Pipeline ablation
        plot_pipeline_ablation(r_200c, '200 Customers', 'fig3_pipeline_ablation_200c.png')

        # Figure 4: EV ablation
        plot_ev_ablation(r_200c, 'fig4_ev_ablation_200c.png')

        # Summary table
        generate_summary_table(r_200c, '200c', 'summary_table_200c.md')

    # Fast results (50c/100c) -- latest
    fast_files = sorted(
        [f for f in os.listdir(results_dir) if f.startswith('week7_tier0_fast_') and f.endswith('.json')],
        reverse=True)
    v2_fast = os.path.join(results_dir, fast_files[0]) if fast_files else None
    if v2_fast and os.path.exists(v2_fast):
        r_fast = load_results(v2_fast)
        if isinstance(r_fast, list) and len(r_fast) > 0:
            print(f'\nLoaded 50c/100c (interim): {len(r_fast)} instances')

            # Split by size
            r_50c = [e for e in r_fast if e.get('n_customers') == 50]
            r_100c = [e for e in r_fast if e.get('n_customers') == 100]

            if r_100c:
                plot_method_comparison(r_100c, '100 Customers', 'fig1_method_comparison_100c.png')
                plot_drone_impact(r_100c, '100 Customers', 'fig2_drone_impact_100c.png')
                generate_summary_table(r_100c, '100c', 'summary_table_100c.md')

            if r_50c:
                plot_method_comparison(r_50c, '50 Customers', 'fig1_method_comparison_50c.png')
                plot_drone_impact(r_50c, '50 Customers', 'fig2_drone_impact_50c.png')
                generate_summary_table(r_50c, '50c', 'summary_table_50c.md')

    print(f'\n{"=" * 60}')
    print(f'All figures saved to: {OUTPUT_DIR}')
    print('Done.')


if __name__ == '__main__':
    main()
