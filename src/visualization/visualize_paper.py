#!/usr/bin/env python3
"""
Publication-quality visualizations for FURP 2026 Workshop Paper.
Truck-Drone EVRP-TW: EDD Repair + Cross-Route Drone Optimization.

Generates 6 figures + 3 tables for the workshop paper.
"""

import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch, FancyBboxPatch
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_BASE, '..', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Publication Style ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# ── Color Palette (colorblind-friendly) ────────────────────────────────────
C_OURS_2D   = '#0072B2'  # deep blue
C_OURS_1D   = '#56B4E9'  # light blue
C_OURS_ND   = '#CCE5FF'  # pale blue
C_CW        = '#009E73'  # green
C_PACO      = '#E69F00'  # orange
C_NSGA2     = '#F0E442'  # yellow
C_IVND      = '#D55E00'  # vermillion
C_BASELINE  = '#999999'  # grey
C_NEGATIVE  = '#CC3333'  # red (for infeasible)

TYPE_COLORS = {
    'RC1': '#0072B2', 'RC2': '#56B4E9',
    'R1':  '#E69F00', 'R2':  '#F0E442',
    'C1':  '#009E73', 'C2':  '#66CC99',
}
TYPE_MARKERS = {'RC1': 'o', 'RC2': 's', 'R1': 'D', 'R2': '^', 'C1': 'v', 'C2': 'p'}

METHOD_ORDER = ['ours_full', 'ours_1drone', 'ours_no_drone',
                'cw_savings', 'nsga2', 'paco', 'ivnd']
METHOD_SHORT = {
    'ours_full': 'Ours\n(2-Drone)', 'ours_1drone': 'Ours\n(1-Drone)',
    'ours_no_drone': 'Ours\n(No Drone)', 'cw_savings': 'CW-\nSavings',
    'nsga2': 'NSGA-II', 'paco': 'P-ACO', 'ivnd': 'IVND',
}
INSTANCE_LABEL = {
    'RC101': 'RC101\n(RC1)', 'RC201': 'RC201\n(RC2)',
    'R101': 'R101\n(R1)',   'R201': 'R201\n(R2)',
    'C101': 'C101\n(C1)',   'C201': 'C201\n(C2)',
}

# ── Helpers ─────────────────────────────────────────────────────────────────

def load_latest(pattern):
    """Load latest result file matching pattern."""
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    files = sorted([f for f in os.listdir(results_dir)
                    if f.startswith(pattern) and f.endswith('.json')], reverse=True)
    if not files:
        return None, None
    path = os.path.join(results_dir, files[0])
    with open(path) as f:
        return json.load(f), files[0]


def safe_mean(mdata, key='mean_cost'):
    v = mdata.get(key, 1e9)
    return v if v < 1e8 else None


def method_cost(methods, mk):
    m = methods.get(mk, {})
    return m.get('mean_cost', 1e9) if m.get('mean_cost', 1e9) < 1e8 else None


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Comprehensive Method Comparison (200c)
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_comprehensive_comparison(results_200c, results_100c, results_50c):
    """
    3x2 grid: rows = scales (50c, 100c, 200c), cols = cost + feasibility.
    Shows our methods vs baselines across all scales.
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 12),
                              gridspec_kw={'width_ratios': [3, 1]})

    datasets = [(results_50c, '50 Customers'), (results_100c, '100 Customers'),
                (results_200c, '200 Customers')]

    for row, (dataset, scale_label) in enumerate(datasets):
        if dataset is None:
            continue

        ax_cost = axes[row, 0]
        ax_feas = axes[row, 1]

        # ── Cost subplot ──
        instances = sorted(set(e['source_instance'] for e in dataset))
        plot_methods = ['ours_full', 'ours_1drone', 'ours_no_drone', 'cw_savings']
        colors = [C_OURS_2D, C_OURS_1D, C_OURS_ND, C_CW]
        names = ['Ours\n2-Drone', 'Ours\n1-Drone', 'Ours\nNo Drone', 'CW-\nSavings']

        x = np.arange(len(instances))
        width = 0.18
        n_methods = len(plot_methods)

        for mi, (mk, color, name) in enumerate(zip(plot_methods, colors, names)):
            costs = []
            feas_flags = []
            for entry in dataset:
                src = entry['source_instance']
                if src in instances:
                    m = entry['methods'].get(mk, {})
                    c = m.get('mean_cost', 1e9)
                    f = m.get('feasibility_rate', 0)
                    costs.append(c if c < 1e8 else 0)
                    feas_flags.append(f >= 0.99)
            offset = (mi - (n_methods-1)/2) * width
            bars = ax_cost.bar(x + offset, costs, width, color=color, label=name,
                              edgecolor='white', linewidth=0.3)

        ax_cost.set_xticks(x)
        ax_cost.set_xticklabels([INSTANCE_LABEL.get(i, i) for i in instances], fontsize=8)
        ax_cost.set_ylabel('Total Cost', fontsize=10)
        ax_cost.set_title(f'{scale_label} -- Cost Comparison', fontweight='bold', fontsize=11)
        if row == 0:
            ax_cost.legend(loc='upper left', fontsize=7, ncol=2)

        # ── Feasibility subplot ──
        all_methods = ['ours_full', 'ours_1drone', 'ours_no_drone',
                       'ours_no_edd', 'ours_partial_edd',
                       'cw_savings', 'nsga2', 'paco', 'ivnd']
        present = [m for m in all_methods if any(
            m in entry['methods'] for entry in dataset)]
        method_feas = []
        method_names = []
        method_colors = []

        for mk in present:
            feas_rates = []
            for entry in dataset:
                m = entry['methods'].get(mk, {})
                fr = m.get('feasibility_rate', 0)
                feas_rates.append(fr)
            avg_feas = np.mean(feas_rates) * 100 if feas_rates else 0

            if mk.startswith('ours'):
                clr = C_OURS_2D if 'full' in mk else (C_OURS_1D if '1drone' in mk else C_OURS_ND)
            elif 'cw' in mk:
                clr = C_CW
            else:
                clr = C_BASELINE

            method_feas.append(avg_feas)
            method_names.append(METHOD_SHORT.get(mk, mk))
            method_colors.append(clr)

        y_pos = range(len(method_names))
        bars = ax_feas.barh(y_pos, method_feas, color=method_colors, edgecolor='white', height=0.6)
        ax_feas.set_yticks(y_pos)
        ax_feas.set_yticklabels(method_names, fontsize=8)
        ax_feas.set_xlabel('Feasibility Rate (%)', fontsize=10)
        ax_feas.set_xlim(0, 105)
        ax_feas.set_title(f'{scale_label}\nFeasibility', fontweight='bold', fontsize=11)
        ax_feas.axvline(x=99, color='green', linestyle='--', linewidth=0.8, alpha=0.5)

        # Label bars
        for bar, val in zip(bars, method_feas):
            if val > 0:
                ax_feas.text(min(val + 1, 103), bar.get_y() + bar.get_height()/2,
                           f'{val:.0f}%', va='center', fontsize=7,
                           color='darkgreen' if val >= 99 else 'darkred')

    fig.suptitle('Method Comparison Across Scales', fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig1_comprehensive_comparison.png')
    fig.savefig(out, facecolor='white')
    plt.close(fig)
    print(f'  [fig1] {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Drone Impact -- Cost Savings by Instance Type
# ═══════════════════════════════════════════════════════════════════════════════

def fig2_drone_impact(results_200c, results_100c, results_50c):
    """
    Show drone cost savings (2-Drone vs No-Drone) across instance types and scales.
    Grouped bar chart: 6 instance types x 3 scales.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    type_order = ['RC1', 'RC2', 'R1', 'R2', 'C1', 'C2']
    scales = ['50c', '100c', '200c']
    scale_colors = ['#BBDEFB', '#42A5F5', '#1565C0']

    x = np.arange(len(type_order))
    width = 0.25

    all_savings = {}
    for scale_label, dataset in [('50c', results_50c), ('100c', results_100c), ('200c', results_200c)]:
        if dataset is None:
            continue
        savings = {}
        for entry in dataset:
            tw = entry['tw_type']
            nd_ = entry['methods'].get('ours_no_drone', {})
            full_ = entry['methods'].get('ours_full', {})
            nc = nd_.get('mean_cost', 1e9)
            fc = full_.get('mean_cost', 1e9)
            if nc < 1e8 and fc < 1e8 and nc > 0:
                savings[tw] = (nc - fc) / nc * 100
            else:
                savings[tw] = 0
        all_savings[scale_label] = savings

    for si, (scale_label, color) in enumerate(zip(scales, scale_colors)):
        vals = [all_savings.get(scale_label, {}).get(t, 0) for t in type_order]
        offset = (si - 1) * width
        bars = ax.bar(x + offset, vals, width, label=scale_label, color=color,
                     edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0.5:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                       f'{val:.1f}%', ha='center', fontsize=7, fontweight='bold', color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(type_order, fontsize=11)
    ax.set_ylabel('Cost Reduction vs Truck-Only (%)', fontsize=11)
    ax.set_title('Drone Impact by Instance Type and Scale', fontweight='bold', fontsize=13)
    ax.legend(title='Scale', fontsize=9, title_fontsize=10)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_ylim(bottom=-2)

    # Add annotation about C-type
    ax.annotate('C-type: drones\nnet-negative at 200c',
                xy=(4, 0), xytext=(4.5, -0.5),
                fontsize=7, color='darkred', fontstyle='italic',
                arrowprops=dict(arrowstyle='->', color='darkred', lw=0.8))

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig2_drone_impact.png')
    fig.savefig(out, facecolor='white')
    plt.close(fig)
    print(f'  [fig2] {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Pipeline Ablation -- Component Contribution
# ═══════════════════════════════════════════════════════════════════════════════

def fig3_pipeline_ablation(results_200c, results_100c):
    """
    Waterfall-style: show each component's marginal contribution.
    POMO raw -> +EDD Partial -> +EDD Full -> +1 Drone -> +2 Drones
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for dataset, scale_label, ax_row in [(results_100c, '100c', 0), (results_200c, '200c', 1)]:
        if dataset is None:
            continue

        for ai, entry in enumerate(dataset):
            if ai >= 3 and scale_label == '100c':
                break
            ax = axes[ax_row * 3 + (ai % 3)]
            src = entry['source_instance']
            tw = entry['tw_type']
            methods = entry['methods']

            steps = [
                ('ours_no_edd', 'POMO\nRaw', '#FF9800'),
                ('ours_partial_edd', '+Partial\nEDD', '#FFB74D'),
                ('ours_no_drone', '+Full\nEDD', '#BBDEFB'),
                ('ours_1drone', '+1 Drone\n/Truck', '#64B5F6'),
                ('ours_full', '+2 Drones\n/Truck', '#1565C0'),
            ]

            costs = []
            feas_list = []
            for mk, _, _ in steps:
                m = methods.get(mk, {})
                c = m.get('mean_cost', 0) if m.get('mean_cost', 1e9) < 1e8 else 0
                f = m.get('feasibility_rate', 0)
                costs.append(c)
                feas_list.append(f)

            colors = [c for _, _, c in steps]
            labels = [l for _, l, _ in steps]

            # Filter out zeros
            valid = [(c, l, clr, f) for c, l, clr, f in zip(costs, labels, colors, feas_list) if c > 0]
            if not valid:
                continue
            costs_v, labels_v, colors_v, feas_v = zip(*valid)

            x = np.arange(len(costs_v))
            bars = ax.bar(x, costs_v, color=colors_v, edgecolor='white', linewidth=0.5)

            # Mark infeasible steps
            for bi, (bar, c, f) in enumerate(zip(bars, costs_v, feas_v)):
                if c > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(costs_v)*0.03,
                           f'{c:.0f}', ha='center', fontsize=8, fontweight='bold', rotation=0)
                if f < 0.99:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 0.5,
                           'x', ha='center', fontsize=18, color='red', fontweight='bold')

            # Arrow showing improvement
            ax.annotate('', xy=(len(costs_v)-1, costs_v[-1]),
                       xytext=(len(costs_v)-2, costs_v[-2]),
                       arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

            ax.set_xticks(x)
            ax.set_xticklabels(labels_v, fontsize=7)
            ax.set_title(f'{src} ({tw}) -- {scale_label}', fontweight='bold', fontsize=10)
            ax.set_ylabel('Cost')

    # Hide unused subplots
    for ai in range(len(dataset), 6):
        if ai < 6:
            pass  # keep them

    fig.suptitle('Pipeline Ablation: Marginal Contribution of Each Component',
                 fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig3_pipeline_ablation.png')
    fig.savefig(out, facecolor='white')
    plt.close(fig)
    print(f'  [fig3] {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: EV Ablation -- Models A/B/C
# ═══════════════════════════════════════════════════════════════════════════════

def fig4_ev_study(results_200c):
    """
    Left: Cost comparison Model A vs B vs C.
    Right: EV feasibility + energy violation.
    """
    if results_200c is None:
        print('  [fig4] No 200c data, skipping')
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    type_order = ['RC1', 'RC2', 'R1', 'R2', 'C1', 'C2']
    types_found = []
    costs_a, costs_b, costs_c = [], [], []
    ev_feas_b, ev_feas_c = [], []
    e_vio_b, e_vio_c = [], []

    for tw in type_order:
        entry = next((e for e in results_200c if e['tw_type'] == tw), None)
        if entry is None:
            continue
        types_found.append(tw)

        base = method_cost(entry['methods'], 'ours_full') or 0
        evl = entry.get('ev_methods', {}).get('ev_linear', {})
        evn = entry.get('ev_methods', {}).get('ev_nonlinear', {})

        costs_a.append(base)
        costs_b.append(safe_mean(evl, 'mean_cost') or base)
        costs_c.append(safe_mean(evn, 'mean_cost') or base)
        ev_feas_b.append(evl.get('ev_feasibility_rate', 0) * 100)
        ev_feas_c.append(evn.get('ev_feasibility_rate', 0) * 100)
        e_vio_b.append(safe_mean(evl, 'mean_energy_violation') or 0)
        e_vio_c.append(safe_mean(evn, 'mean_energy_violation') or 0)

    x = np.arange(len(types_found))
    width = 0.25

    # Cost comparison
    ax1.bar(x - width, costs_a, width, label='Model A\n(No EV)', color='#BBDEFB', edgecolor='white')
    ax1.bar(x, costs_b, width, label='Model B\n(+Linear)', color='#42A5F5', edgecolor='white')
    ax1.bar(x + width, costs_c, width, label='Model C\n(+Non-linear)', color='#1565C0', edgecolor='white')
    ax1.set_xticks(x)
    ax1.set_xticklabels(types_found, fontsize=10)
    ax1.set_ylabel('Total Cost', fontsize=11)
    ax1.set_title('EV Model Cost Comparison (200c)', fontweight='bold')
    ax1.legend(fontsize=8)

    # EV metrics
    ax2_twin = ax2.twinx()
    b1 = ax2.bar(x - width/2, ev_feas_b, width, label='EV Feasibility (B)',
                color='#42A5F5', edgecolor='white')
    b2 = ax2.bar(x + width/2, ev_feas_c, width, label='EV Feasibility (C)',
                color='#1565C0', edgecolor='white')
    ax2.plot(x, e_vio_b, 'D-', color='red', markersize=6, label='E-Violation (B)')
    ax2.plot(x, e_vio_c, 's-', color='darkred', markersize=6, label='E-Violation (C)')

    ax2.set_xticks(x)
    ax2.set_xticklabels(types_found, fontsize=10)
    ax2.set_ylabel('EV Feasibility Rate (%)', fontsize=11)
    ax2.set_title('EV Feasibility & Violations (200c)', fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2_twin.set_ylim(0, max(max(e_vio_b), max(e_vio_c), 1) * 1.2)

    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper right')

    fig.suptitle('EV Ablation Study -- Battery: 100 kWh | Energy: 1.5 kWh/km | 8 Trucks',
                 fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig4_ev_ablation.png')
    fig.savefig(out, facecolor='white')
    plt.close(fig)
    print(f'  [fig4] {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Gap-to-CW-Savings Heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def fig5_gap_heatmap(results_200c, results_100c, results_50c):
    """
    Show cost gap (%) between Ours-Full and CW-Savings.
    Positive = Ours more expensive, Negative = Ours cheaper.
    """
    datasets = [
        ('50c', results_50c),
        ('100c', results_100c),
        ('200c', results_200c),
    ]
    type_order = ['RC1', 'RC2', 'R1', 'R2', 'C1', 'C2']

    # Build matrix
    matrix = np.zeros((len(datasets), len(type_order)))
    annot = [['' for _ in type_order] for _ in range(len(datasets))]

    for ri, (scale_label, dataset) in enumerate(datasets):
        if dataset is None:
            matrix[ri, :] = np.nan
            continue
        for ci, tw in enumerate(type_order):
            entry = next((e for e in dataset if e['tw_type'] == tw), None)
            if entry is None:
                matrix[ri, ci] = np.nan
                continue
            fc = method_cost(entry['methods'], 'ours_full')
            cc = method_cost(entry['methods'], 'cw_savings')
            if fc and cc and cc > 0:
                gap = (fc - cc) / cc * 100
                matrix[ri, ci] = gap
                annot[ri][ci] = f'{gap:+.0f}%'
            else:
                matrix[ri, ci] = np.nan

    fig, ax = plt.subplots(figsize=(8, 3.5))

    cmap = plt.cm.RdYlGn_r  # Red = Ours worse, Green = Ours better
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=-30, vmax=100)

    # Annotate
    for ri in range(len(datasets)):
        for ci in range(len(type_order)):
            if not np.isnan(matrix[ri, ci]):
                color = 'white' if abs(matrix[ri, ci]) > 50 else 'black'
                ax.text(ci, ri, annot[ri][ci], ha='center', va='center',
                       fontsize=10, fontweight='bold', color=color)

    ax.set_xticks(range(len(type_order)))
    ax.set_xticklabels(type_order, fontsize=11)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels([d[0] for d in datasets], fontsize=11)
    ax.set_title('Cost Gap: Ours (2-Drone) vs CW-Savings', fontweight='bold', fontsize=13)
    ax.set_xlabel('Instance Type', fontsize=11)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Gap (%) -- Red=Ours Worse, Green=Ours Better', fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig5_gap_heatmap.png')
    fig.savefig(out, facecolor='white')
    plt.close(fig)
    print(f'  [fig5] {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Drone Mission Statistics (200c)
# ═══════════════════════════════════════════════════════════════════════════════

def fig6_drone_stats(results_200c):
    """Show n_drones, drone savings, and per-truck utilization at 200c."""
    if results_200c is None:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    type_order = ['RC1', 'RC2', 'R1', 'R2', 'C1', 'C2']
    n_drones_list = []
    savings_list = []
    types_present = []

    for tw in type_order:
        entry = next((e for e in results_200c if e['tw_type'] == tw), None)
        if entry is None:
            continue
        types_present.append(tw)

        # n_drones from per_run
        per_run = entry['methods'].get('ours_full', {}).get('per_run', [])
        drone_counts = [r.get('n_drones', 0) for r in per_run]
        n_drones_list.append(np.mean(drone_counts) if drone_counts else 0)

        # Savings
        nc = method_cost(entry['methods'], 'ours_no_drone')
        fc = method_cost(entry['methods'], 'ours_full')
        if nc and fc and nc > 0:
            savings_list.append((nc - fc) / nc * 100)
        else:
            savings_list.append(0)

    x = np.arange(len(types_present))

    # Left: n_drones bar chart
    colors_drone = [TYPE_COLORS.get(t, C_BASELINE) for t in types_present]
    bars = ax1.bar(x, n_drones_list, color=colors_drone, edgecolor='white')
    ax1.set_xticks(x)
    ax1.set_xticklabels(types_present, fontsize=11)
    ax1.set_ylabel('Avg Drone Missions', fontsize=11)
    ax1.set_title('Drone Utilization at 200c', fontweight='bold')
    for bar, val in zip(bars, n_drones_list):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                   f'{val:.0f}', ha='center', fontsize=10, fontweight='bold')

    # Right: scatter savings vs n_drones
    for ti, tw in enumerate(types_present):
        ax2.scatter(savings_list[ti], n_drones_list[ti], c=TYPE_COLORS.get(tw, C_BASELINE),
                   s=200, marker=TYPE_MARKERS.get(tw, 'o'), edgecolors='black',
                   linewidth=0.5, zorder=5, label=tw)

    ax2.set_xlabel('Cost Savings vs No-Drone (%)', fontsize=11)
    ax2.set_ylabel('Avg Drone Missions', fontsize=11)
    ax2.set_title('Savings vs Drone Count (200c)', fontweight='bold')
    ax2.legend(fontsize=8, loc='lower right')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Drone Mission Analysis -- 200 Customers', fontweight='bold', fontsize=13)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig6_drone_stats.png')
    fig.savefig(out, facecolor='white')
    plt.close(fig)
    print(f'  [fig6] {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# TABLES
# ═══════════════════════════════════════════════════════════════════════════════

def generate_tables(results_200c, results_100c, results_50c):
    """Generate LaTeX-format tables for the paper."""
    out_dir = os.path.join(OUTPUT_DIR, 'tables')
    os.makedirs(out_dir, exist_ok=True)

    type_order = ['RC1', 'RC2', 'R1', 'R2', 'C1', 'C2']

    # ── Table 1: Main Results ──
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Main Experimental Results -- Ours (2-Drone) vs Baselines}')
    lines.append(r'\label{tab:main_results}')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{lcccccccc}')
    lines.append(r'\toprule')
    lines.append(r'&& \multicolumn{3}{c}{Cost} && \multicolumn{2}{c}{Drone} & \\')
    lines.append(r'\cmidrule{3-5} \cmidrule{7-8}')
    lines.append(r'Instance & Scale & Ours-2D & Ours-1D & CW-Sav & Feas\% & $\Delta$Drone\% & n-Drones & Gap-CW\% \\')
    lines.append(r'\midrule')

    for scale, dataset in [('50', results_50c), ('100', results_100c), ('200', results_200c)]:
        if dataset is None:
            continue
        for entry in dataset:
            ik = entry['instance_key']
            src = entry['source_instance']
            tw = entry['tw_type']
            nc = entry['n_customers']
            methods = entry['methods']

            fc = method_cost(methods, 'ours_full')
            oc = method_cost(methods, 'ours_1drone')
            cc = method_cost(methods, 'cw_savings')
            nd_cost = method_cost(methods, 'ours_no_drone')
            feas = methods.get('ours_full', {}).get('feasibility_rate', 0) * 100

            drone_save = (nd_cost - fc) / nd_cost * 100 if nd_cost and fc and nd_cost > 0 else 0
            gap_cw = (fc - cc) / cc * 100 if fc and cc and cc > 0 else 0

            # n_drones from per_run
            per_run = methods.get('ours_full', {}).get('per_run', [])
            drone_counts = [r.get('n_drones', 0) for r in per_run]
            avg_d = np.mean(drone_counts) if drone_counts else 0

            bold_fc = r'\textbf{' + f'{fc:.0f}' + '}' if feas >= 99 else f'{fc:.0f}'
            lines.append(f'  {src} & {nc}c & {bold_fc} & {oc:.0f} & {cc:.0f} & '
                        f'{feas:.0f}\% & {drone_save:+.1f}\% & {avg_d:.0f} & {gap_cw:+.0f}\% \\\\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    with open(os.path.join(out_dir, 'table1_main_results.tex'), 'w') as f:
        f.write('\n'.join(lines))
    print(f'  [table1] {out_dir}/table1_main_results.tex')

    # ── Table 2: EV Ablation ──
    if results_200c:
        lines2 = []
        lines2.append(r'\begin{table}[htbp]')
        lines2.append(r'\centering')
        lines2.append(r'\caption{EV Model Ablation -- 200 Customers, 8 Trucks, 100 kWh Battery}')
        lines2.append(r'\label{tab:ev_ablation}')
        lines2.append(r'\small')
        lines2.append(r'\begin{tabular}{lccccc}')
        lines2.append(r'\toprule')
        lines2.append(r'Type & $\Delta$Cost (B) & $\Delta$Cost (C) & EV-Feas (B) & EV-Feas (C) & E-Violation \\')
        lines2.append(r'\midrule')

        for entry in results_200c:
            tw = entry['tw_type']
            base = method_cost(entry['methods'], 'ours_full') or 0
            evl = entry.get('ev_methods', {}).get('ev_linear', {})
            evn = entry.get('ev_methods', {}).get('ev_nonlinear', {})
            bc = safe_mean(evl, 'mean_cost') or base
            nc = safe_mean(evn, 'mean_cost') or base
            bf = evl.get('ev_feasibility_rate', 0) * 100
            nf = evn.get('ev_feasibility_rate', 0) * 100
            ev = safe_mean(evl, 'mean_energy_violation') or 0

            db = (bc - base) / base * 100 if base > 0 else 0
            dn = (nc - base) / base * 100 if base > 0 else 0
            lines2.append(f'  {tw} & {db:+.1f}\% & {dn:+.1f}\% & {bf:.0f}\% & {nf:.0f}\% & {ev:.1f} kWh \\\\')

        lines2.append(r'\bottomrule')
        lines2.append(r'\end{tabular}')
        lines2.append(r'\end{table}')

        with open(os.path.join(out_dir, 'table2_ev_ablation.tex'), 'w') as f:
            f.write('\n'.join(lines2))
        print(f'  [table2] {out_dir}/table2_ev_ablation.tex')

    # ── Table 3: 50c/100c Summary (Markdown) ──
    for scale_name, dataset in [('50c', results_50c), ('100c', results_100c)]:
        if dataset is None:
            continue
        md_lines = [f'## {scale_name} Results -- All Methods', '',
                    '| Instance | Ours-2D | Ours-1D | Ours-ND | CW-Sav | '
                    'Feas% | Tard | DroneD% | n-Drones | Gap-CW% |',
                    '|----------|---------|---------|---------|--------|'
                    '-------|------|---------|----------|---------|']

        for entry in dataset:
            ik = entry['instance_key']
            methods = entry['methods']
            fc = method_cost(methods, 'ours_full') or 0
            oc = method_cost(methods, 'ours_1drone') or 0
            nc = method_cost(methods, 'ours_no_drone') or 0
            cc = method_cost(methods, 'cw_savings') or 0
            ft = methods.get('ours_full', {}).get('mean_tardiness', 0)
            feas = methods.get('ours_full', {}).get('feasibility_rate', 0) * 100
            per_run = methods.get('ours_full', {}).get('per_run', [])
            drone_counts = [r.get('n_drones', 0) for r in per_run]
            avg_d = np.mean(drone_counts) if drone_counts else 0
            ds = (nc - fc) / nc * 100 if nc > 0 else 0
            gw = (fc - cc) / cc * 100 if cc > 0 else 0
            md_lines.append(f'| {ik} | {fc:.0f} | {oc:.0f} | {nc:.0f} | {cc:.0f} | '
                          f'{feas:.0f}% | {ft:.0f} | {ds:+.1f}% | {avg_d:.0f} | {gw:+.0f}% |')

        path = os.path.join(out_dir, f'table_{scale_name}_all_methods.md')
        with open(path, 'w') as f:
            f.write('\n'.join(md_lines))
        print(f'  [table] {path}')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 60)
    print('GENERATING PUBLICATION-QUALITY FIGURES')
    print('=' * 60)

    # Load all data
    r_200c, fn_200c = load_latest('week7_tier0_200c_')
    r_fast, fn_fast = load_latest('week7_tier0_fast_')

    print(f'\nData:')
    print(f'  200c: {fn_200c} ({len(r_200c) if r_200c else 0} instances)')
    print(f'  Fast: {fn_fast} ({len(r_fast) if r_fast else 0} instances)')

    r_50c = [e for e in r_fast if e.get('n_customers') == 50] if r_fast else None
    r_100c = [e for e in r_fast if e.get('n_customers') == 100] if r_fast else None

    print(f'\nGenerating figures...')

    # Figure 1: Comprehensive comparison across scales
    fig1_comprehensive_comparison(r_200c, r_100c, r_50c)

    # Figure 2: Drone impact by type and scale
    fig2_drone_impact(r_200c, r_100c, r_50c)

    # Figure 3: Pipeline ablation
    fig3_pipeline_ablation(r_200c, r_100c)

    # Figure 4: EV study
    fig4_ev_study(r_200c)

    # Figure 5: Gap heatmap
    fig5_gap_heatmap(r_200c, r_100c, r_50c)

    # Figure 6: Drone statistics
    fig6_drone_stats(r_200c)

    # Figure 7: Route map panel (2x2 -- 3 instance types + EV)
    print(f'\nGenerating route maps...')
    try:
        from src.visualization.fig_route_maps import generate_panel_figure, generate_comparison_figure
        generate_panel_figure()
        generate_comparison_figure()
        print('  Route maps complete.')
    except Exception as e:
        print(f'  Route maps skipped: {e}')

    # Tables
    print(f'\nGenerating tables...')
    generate_tables(r_200c, r_100c, r_50c)

    print(f'\n{"=" * 60}')
    print(f'Output: {OUTPUT_DIR}/')
    print(f'Tables: {OUTPUT_DIR}/tables/')
    print('Done.')


if __name__ == '__main__':
    main()
