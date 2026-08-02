#!/usr/bin/env python3
"""
Comprehensive Visualization — Week 8 Updated.
Reads from the new sweep_*.json files and generates all report figures.

Figures:
  1. TW Feasibility — Ours vs Classical (224 instances)
  2. Tardiness Comparison (log scale)
  3. Forward Insertion vs Old Partial EDD — Fallback Reduction Matrix
  4. Zero-Fallback Rate by TW Type (Heatmap)
  5. Fallback vs Scale (Line Chart)
  6. Cost Impact Scatter
  7. Summary Dashboard (4-panel)
  8. EV Binding Analysis
  9. Statistical Tests Visualization
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
RESULTS_DIR = os.path.join(_BASE, '..', 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Style ──
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'legend.fontsize': 8, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
})

# ── Colors ──
C_OURS = '#0072B2'; C_OLD = '#CC3333'; C_NSGA2 = '#F0E442'
C_PACO = '#E69F00'; C_IVND = '#D55E00'; C_GREY = '#999999'
TW_TYPES = ['RC1', 'RC2', 'R1', 'R2', 'C1', 'C2']
TW_COLORS = {'RC1': '#E41A1C', 'RC2': '#377EB8', 'R1': '#4DAF4A',
             'R2': '#984EA3', 'C1': '#FF7F00', 'C2': '#A65628'}
SCALES = [25, 50, 100, 200]

def load_json(name):
    path = os.path.join(RESULTS_DIR, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1: TW Feasibility — Ours vs Classical (ALL 224 instances)
# ══════════════════════════════════════════════════════════════════════════

def fig1_tw_feasibility():
    fi = load_json('sweep_forward_insertion.json')
    cl = load_json('sweep_classical_baselines.json')

    if not fi or not cl:
        print('  FIG1: Data not available, skipping')
        return

    # Aggregate TW feasibility by method across ALL instances
    methods = ['POMO+Forward\nInsertion', 'NSGA-II', 'P-ACO', 'IVND']
    tw_feas = {m: 0 for m in methods}
    total = 0

    for key in fi:
        if 'error' in fi[key] or key not in cl:
            continue
        total += 1
        if fi[key].get('new_tw_feasible', False):
            tw_feas['POMO+Forward\nInsertion'] += 1
        for m in ['NSGA-II', 'P-ACO', 'IVND']:
            if m in cl[key] and 'error' not in cl[key][m]:
                if cl[key][m].get('tw_feasible', False):
                    tw_feas[m] += 1

    # Also compute by scale × type
    by_group = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'feas': 0}))
    for key in fi:
        if 'error' in fi[key] or key not in cl:
            continue
        scale = fi[key].get('scale')
        tw_type = fi[key].get('tw_type')
        if scale is None or tw_type is None:
            continue
        g = f'{tw_type}'
        by_group[scale][g]['total'] += 1
        if fi[key].get('new_tw_feasible', False):
            by_group[scale][g]['feas'] += 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # (a) Overall bar chart
    ax = axes[0]
    names = list(methods)
    rates = [tw_feas[m]/max(total,1)*100 for m in methods]
    colors = [C_OURS, C_NSGA2, C_PACO, C_IVND]
    bars = ax.bar(range(len(methods)), rates, color=colors, edgecolor='white', linewidth=0.8)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f'{rate:.1f}%', ha='center', fontweight='bold', fontsize=12)
    ax.set_ylabel('TW Feasibility Rate (%)')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_title(f'(a) Overall TW Feasibility ({total} instances)', fontweight='bold')

    # (b) By scale heatmap
    ax = axes[1]
    matrix = np.zeros((len(SCALES), len(TW_TYPES)))
    for i, scale in enumerate(SCALES):
        for j, tw in enumerate(TW_TYPES):
            bg = by_group[scale].get(tw, {'total': 0, 'feas': 0})
            matrix[i, j] = bg['feas'] / max(bg['total'], 1) * 100

    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(len(TW_TYPES)))
    ax.set_xticklabels(TW_TYPES)
    ax.set_yticks(range(len(SCALES)))
    ax.set_yticklabels([f'{s}c' for s in SCALES])
    for i in range(len(SCALES)):
        for j in range(len(TW_TYPES)):
            ax.text(j, i, f'{matrix[i,j]:.0f}%', ha='center', va='center',
                   fontweight='bold', fontsize=9,
                   color='white' if matrix[i,j] < 50 else 'black')
    ax.set_title('(b) Our TW Feasibility by Scale × Type', fontweight='bold')
    plt.colorbar(im, ax=ax, label='TW Feasibility (%)')

    fig.suptitle('Figure 1: Time-Window Feasibility — Forward Insertion vs Classical Methods',
                 fontweight='bold', fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig1_tw_feasibility.png'))
    plt.close()
    print('  Saved: fig1_tw_feasibility.png')


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Tardiness — Ours vs Classical (log scale)
# ══════════════════════════════════════════════════════════════════════════

def fig2_tardiness():
    fi = load_json('sweep_forward_insertion.json')
    cl = load_json('sweep_classical_baselines.json')

    if not fi or not cl:
        print('  FIG2: Data not available, skipping')
        return

    methods = ['POMO+FI', 'NSGA-II', 'P-ACO', 'IVND']
    method_tards = {m: [] for m in methods}

    for key in fi:
        if 'error' in fi[key] or key not in cl:
            continue
        method_tards['POMO+FI'].append(max(fi[key].get('new_tard', 0), 0.01))
        for m in ['NSGA-II', 'P-ACO', 'IVND']:
            if m in cl[key] and 'error' not in cl[key][m]:
                method_tards[m].append(max(cl[key][m].get('tardiness', 0), 0.01))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # (a) Box plot
    ax = axes[0]
    data = [method_tards[m] for m in methods]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    ax.set_xticklabels(methods)
    colors = [C_OURS, C_NSGA2, C_PACO, C_IVND]
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_yscale('log')
    ax.set_ylabel('Tardiness (log scale)')
    ax.set_title('(a) Tardiness Distribution', fontweight='bold')

    # (b) Mean tardiness
    ax = axes[1]
    means = [np.mean(method_tards[m]) for m in methods]
    bars = ax.bar(range(len(methods)), means, color=colors, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, means):
        label = f'{val:.0f}' if val >= 1 else '≈0'
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.05,
                label, ha='center', fontweight='bold', fontsize=9)
    ax.set_ylabel('Mean Tardiness')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods)
    ax.set_yscale('log')
    ax.set_title('(b) Mean Tardiness', fontweight='bold')

    fig.suptitle('Figure 2: Tardiness — Forward Insertion Eliminates TW Violations',
                 fontweight='bold', fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_tardiness.png'))
    plt.close()
    print('  Saved: fig2_tardiness.png')


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Forward Insertion vs Old Partial EDD — Fallback Reduction
# ══════════════════════════════════════════════════════════════════════════

def fig3_fallback_reduction():
    fi = load_json('sweep_forward_insertion.json')

    if not fi:
        print('  FIG3: Data not available, skipping')
        return

    # Aggregate by scale × tw_type
    by_group = defaultdict(lambda: defaultdict(lambda: {
        'old_fb': 0, 'new_fb': 0, 'moves': 0, 'fi_ok': 0, 'tw_ok': 0, 'count': 0}))

    for key, data in fi.items():
        if 'error' in data:
            continue
        scale = data.get('scale')
        tw = data.get('tw_type')
        if scale is None or tw is None:
            continue
        bg = by_group[scale][tw]
        bg['old_fb'] += data.get('old_fallback_count', 0)
        bg['new_fb'] += data.get('new_fallback_count', 0)
        bg['moves'] += data.get('new_moves_accepted', 0)
        if data.get('new_fi_success', False):
            bg['fi_ok'] += 1
        if data.get('new_tw_feasible', False):
            bg['tw_ok'] += 1
        bg['count'] += 1

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Grouped bar: old vs new fallback by scale×type
    ax = axes[0, 0]
    x = np.arange(len(TW_TYPES))
    width = 0.2
    for si, scale in enumerate(SCALES):
        old_vals = [by_group[scale][tw]['old_fb'] for tw in TW_TYPES]
        new_vals = [by_group[scale][tw]['new_fb'] for tw in TW_TYPES]
        offset = (si - 1.5) * width * 2
        ax.bar(x + offset, old_vals, width, label=f'{scale}c Old', color=C_OLD, alpha=0.5)
        ax.bar(x + offset + width, new_vals, width, label=f'{scale}c New', color=C_OURS, alpha=0.8)
    ax.set_ylabel('Total Fallback Count')
    ax.set_xticks(x)
    ax.set_xticklabels(TW_TYPES)
    ax.set_title('(a) Fallback Count: Old vs New', fontweight='bold')
    ax.legend(fontsize=6, ncol=2)

    # (b) Zero-fallback rate heatmap
    ax = axes[0, 1]
    matrix = np.zeros((len(SCALES), len(TW_TYPES)))
    for i, scale in enumerate(SCALES):
        for j, tw in enumerate(TW_TYPES):
            bg = by_group[scale][tw]
            matrix[i, j] = bg['tw_ok'] / max(bg['count'], 1) * 100
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(len(TW_TYPES))); ax.set_xticklabels(TW_TYPES)
    ax.set_yticks(range(len(SCALES))); ax.set_yticklabels([f'{s}c' for s in SCALES])
    for i in range(len(SCALES)):
        for j in range(len(TW_TYPES)):
            ax.text(j, i, f'{matrix[i,j]:.0f}%', ha='center', va='center',
                   fontweight='bold', color='white' if matrix[i,j] < 50 else 'black')
    ax.set_title('(b) Zero-Fallback Rate (%)', fontweight='bold')
    plt.colorbar(im, ax=ax)

    # (c) Fallback vs scale line
    ax = axes[1, 0]
    for tw in TW_TYPES:
        old_line = [by_group[s][tw]['old_fb'] for s in SCALES]
        new_line = [by_group[s][tw]['new_fb'] for s in SCALES]
        ax.plot(SCALES, old_line, '--', color=TW_COLORS[tw], alpha=0.4, linewidth=1)
        ax.plot(SCALES, new_line, 'o-', color=TW_COLORS[tw], label=tw, linewidth=2)
    ax.set_xlabel('Scale (customers)'); ax.set_ylabel('Total Fallback Count')
    ax.set_title('(c) Fallback vs Scale', fontweight='bold')
    ax.legend(fontsize=7); ax.set_xticks(SCALES)

    # (d) FI moves vs fallback reduction scatter
    ax = axes[1, 1]
    for tw in TW_TYPES:
        xs, ys = [], []
        for scale in SCALES:
            bg = by_group[scale][tw]
            reduction = bg['old_fb'] - bg['new_fb']
            moves = bg['moves']
            xs.append(moves / max(bg['count'], 1))
            ys.append(reduction / max(bg['count'], 1))
        ax.scatter(xs, ys, c=TW_COLORS[tw], label=tw, s=80, edgecolors='white', linewidth=0.5)
    ax.set_xlabel('Avg Forward Moves per Instance')
    ax.set_ylabel('Avg Fallback Reduction per Instance')
    ax.set_title('(d) Moves vs Reduction', fontweight='bold')
    ax.legend(fontsize=7)
    ax.axhline(y=0, color='grey', linestyle='--', alpha=0.3)

    fig.suptitle('Figure 3: Forward Insertion — Fallback Reduction Analysis (224 Instances)',
                 fontweight='bold', fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig3_fallback_analysis.png'))
    plt.close()
    print('  Saved: fig3_fallback_analysis.png')


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Cost Impact
# ══════════════════════════════════════════════════════════════════════════

def fig4_cost_impact():
    fi = load_json('sweep_forward_insertion.json')

    if not fi:
        print('  FIG4: Data not available, skipping')
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # (a) Cost scatter: old vs new
    ax = axes[0]
    for tw in TW_TYPES:
        xs, ys = [], []
        for key, data in fi.items():
            if 'error' in data or data.get('tw_type') != tw:
                continue
            xs.append(data.get('old_cost', 0))
            ys.append(data.get('new_cost', 0))
        ax.scatter(xs, ys, c=TW_COLORS[tw], label=tw, s=20, alpha=0.6, edgecolors='none')
    max_val = max(max(xs, default=5000), max(ys, default=5000))
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Equal cost')
    ax.set_xlabel('Old Cost (Partial EDD)')
    ax.set_ylabel('New Cost (Forward Insertion)')
    ax.set_title('(a) Cost: Old vs New', fontweight='bold')
    ax.legend(fontsize=7)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}'))

    # (b) Cost reduction histogram
    ax = axes[1]
    cost_changes = []
    for key, data in fi.items():
        if 'error' in data:
            continue
        old = data.get('old_cost', 0)
        new = data.get('new_cost', 0)
        if old > 0:
            cost_changes.append((new - old) / old * 100)

    ax.hist(cost_changes, bins=40, color=C_OURS, edgecolor='white', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=np.median(cost_changes), color='black', linestyle='-', alpha=0.7,
              label=f'Median: {np.median(cost_changes):+.1f}%')
    ax.set_xlabel('Cost Change (%)')
    ax.set_ylabel('Number of Instances')
    ax.set_title('(b) Cost Change Distribution', fontweight='bold')
    ax.legend(fontsize=8)

    fig.suptitle('Figure 4: Cost Impact — Forward Insertion Preserves POMO Distance Structure',
                 fontweight='bold', fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig4_cost_impact.png'))
    plt.close()
    print('  Saved: fig4_cost_impact.png')


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Summary Dashboard (4 panels)
# ══════════════════════════════════════════════════════════════════════════

def fig5_dashboard():
    fi = load_json('sweep_forward_insertion.json')

    if not fi:
        print('  FIG5: Data not available, skipping')
        return

    # Compute aggregate stats
    total_old_fb = 0; total_new_fb = 0; total_moves = 0
    fi_success = 0; tw_feasible = 0; total = 0

    by_type = defaultdict(lambda: {'old_fb': 0, 'new_fb': 0, 'count': 0, 'tw_ok': 0, 'fi_ok': 0})

    for key, data in fi.items():
        if 'error' in data: continue
        total += 1
        tw = data.get('tw_type', '?')
        total_old_fb += data.get('old_fallback_count', 0)
        total_new_fb += data.get('new_fallback_count', 0)
        total_moves += data.get('new_moves_accepted', 0)
        if data.get('new_fi_success', False): fi_success += 1
        if data.get('new_tw_feasible', False): tw_feasible += 1
        by_type[tw]['old_fb'] += data.get('old_fallback_count', 0)
        by_type[tw]['new_fb'] += data.get('new_fallback_count', 0)
        by_type[tw]['count'] += 1
        if data.get('new_tw_feasible', False): by_type[tw]['tw_ok'] += 1
        if data.get('new_fi_success', False): by_type[tw]['fi_ok'] += 1

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Key metrics summary
    ax = axes[0, 0]
    ax.axis('off')
    metrics = [
        ('Instances', f'{total}'),
        ('Old Total Fallbacks', f'{total_old_fb:.0f}'),
        ('New Total Fallbacks', f'{total_new_fb:.0f}'),
        ('Fallback Reduction', f'{(total_old_fb-total_new_fb)/max(total_old_fb,1)*100:.1f}%'),
        ('Zero-Fallback Rate', f'{tw_feasible/max(total,1)*100:.1f}%'),
        ('FI Success Rate', f'{fi_success/max(total,1)*100:.1f}%'),
        ('TW Feasibility', f'{tw_feasible}/{total} ({tw_feasible/max(total,1)*100:.1f}%)'),
        ('Avg Moves/Instance', f'{total_moves/max(total,1):.1f}'),
    ]
    for i, (label, value) in enumerate(metrics):
        y = 0.9 - i * 0.11
        ax.text(0.1, y, label, fontsize=12, va='center')
        ax.text(0.7, y, value, fontsize=14, va='center', fontweight='bold', color=C_OURS)
    ax.set_title('(a) Key Metrics', fontweight='bold', fontsize=12)

    # (b) Fallback by type (grouped bar)
    ax = axes[0, 1]
    x = np.arange(len(TW_TYPES))
    width = 0.35
    old_vals = [by_type[t]['old_fb'] for t in TW_TYPES]
    new_vals = [by_type[t]['new_fb'] for t in TW_TYPES]
    ax.bar(x - width/2, old_vals, width, label='Old Partial EDD', color=C_OLD, alpha=0.7)
    ax.bar(x + width/2, new_vals, width, label='Forward Insertion', color=C_OURS, alpha=0.7)
    for i, (o, n) in enumerate(zip(old_vals, new_vals)):
        pct = (o - n) / max(o, 1) * 100
        ax.text(i, max(o, n) + 5, f'{pct:.0f}%↓', ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(TW_TYPES)
    ax.set_ylabel('Total Fallback Count')
    ax.set_title('(b) Fallback by TW Type', fontweight='bold')
    ax.legend(fontsize=8)

    # (c) Zero-fallback by type
    ax = axes[1, 0]
    rates = [by_type[t]['tw_ok']/max(by_type[t]['count'],1)*100 for t in TW_TYPES]
    bars = ax.bar(range(len(TW_TYPES)), rates, color=[TW_COLORS[t] for t in TW_TYPES], edgecolor='white')
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
               f'{rate:.0f}%', ha='center', fontweight='bold', fontsize=10)
    ax.set_xticks(range(len(TW_TYPES))); ax.set_xticklabels(TW_TYPES)
    ax.set_ylabel('Zero-Fallback Rate (%)')
    ax.set_ylim(0, 115)
    ax.set_title('(c) Zero-Fallback by TW Type', fontweight='bold')

    # (d) FI success rate by type
    ax = axes[1, 1]
    rates = [by_type[t]['fi_ok']/max(by_type[t]['count'],1)*100 for t in TW_TYPES]
    bars = ax.bar(range(len(TW_TYPES)), rates, color=[TW_COLORS[t] for t in TW_TYPES], edgecolor='white')
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
               f'{rate:.0f}%', ha='center', fontweight='bold', fontsize=10)
    ax.set_xticks(range(len(TW_TYPES))); ax.set_xticklabels(TW_TYPES)
    ax.set_ylabel('FI Success Rate (%)')
    ax.set_ylim(0, 115)
    ax.set_title('(d) Forward Insertion Success Rate', fontweight='bold')

    fig.suptitle('Figure 5: Summary Dashboard — Forward Insertion Results (224 Instances)',
                 fontweight='bold', fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig5_dashboard.png'))
    plt.close()
    print('  Saved: fig5_dashboard.png')


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 6: EV Binding Analysis
# ══════════════════════════════════════════════════════════════════════════

def fig6_ev_analysis():
    ev = load_json('sweep_ev_binding.json')

    if not ev:
        print('  FIG6: EV data not available, skipping')
        return

    battery_levels = [55, 40, 30, 25]
    models = ['none', 'linear', 'nonlinear']

    # Count feasible by battery level
    by_battery = {b: {'feas': 0, 'total': 0, 'n_charges': []} for b in battery_levels}

    for key, data in ev.items():
        if 'error' in data or 'ev_results' not in data:
            continue
        for b in battery_levels:
            by_battery[b]['total'] += 1
            # Check if ANY charging model is feasible at this battery level
            any_feas = False
            for m in models:
                ek = f'{m}_{b}kWh'
                if ek in data['ev_results']:
                    er = data['ev_results'][ek]
                    if er.get('ev_feasible', False):
                        any_feas = True
                        by_battery[b]['n_charges'].append(er.get('n_charges', 0))
            if any_feas:
                by_battery[b]['feas'] += 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # (a) EV feasibility by battery level
    ax = axes[0]
    x = range(len(battery_levels))
    rates = [by_battery[b]['feas']/max(by_battery[b]['total'],1)*100 for b in battery_levels]
    colors_list = ['#2E7D32', '#66BB6A', '#FFA726', '#EF5350']
    bars = ax.bar(x, rates, color=colors_list, edgecolor='white', linewidth=0.8)
    for bar, rate, b in zip(bars, rates, battery_levels):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
               f'{rate:.0f}%\n({b}kWh)', ha='center', fontweight='bold', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{b} kWh' for b in battery_levels])
    ax.set_ylabel('EV Feasible (%)')
    ax.set_ylim(0, 115)
    ax.set_title('(a) EV Feasibility by Battery Capacity', fontweight='bold')

    # (b) Avg charges needed at binding levels
    ax = axes[1]
    x = range(len(battery_levels))
    avg_charges = [np.mean(by_battery[b]['n_charges']) if by_battery[b]['n_charges'] else 0 for b in battery_levels]
    bars = ax.bar(x, avg_charges, color=colors_list, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, avg_charges):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
               f'{val:.1f}', ha='center', fontweight='bold', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{b} kWh' for b in battery_levels])
    ax.set_ylabel('Avg Charging Stops per Instance')
    ax.set_title('(b) Avg Charging Stops at Feasible Battery Levels', fontweight='bold')

    fig.suptitle('Figure 6: EV Battery Constraint Analysis — Binding Regime Identification',
                 fontweight='bold', fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig6_ev_analysis.png'))
    plt.close()
    print('  Saved: fig6_ev_analysis.png')


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 7: Statistical Tests
# ══════════════════════════════════════════════════════════════════════════

def fig7_statistics():
    stats = load_json('sweep_statistics.json')

    if not stats or 'rankings' not in stats:
        print('  FIG7: Statistics not available, skipping')
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # (a) Method rankings
    ax = axes[0]
    rankings = stats['rankings']
    methods = list(rankings.keys())
    ranks = [rankings[m] for m in methods]
    colors = [C_OURS if 'Forward' in m else C_NSGA2 if 'NSGA' in m else C_PACO if 'ACO' in m else C_IVND for m in methods]

    bars = ax.barh(range(len(methods)), ranks, color=colors, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, ranks):
        ax.text(bar.get_width()+0.05, bar.get_y()+bar.get_height()/2,
               f'{val:.2f}', va='center', fontweight='bold')
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel('Average Rank (lower = better)')
    ax.set_title('(a) Friedman Test Rankings', fontweight='bold')
    ax.invert_yaxis()

    # Annotate Friedman stat
    fr = stats.get('friedman', {})
    ax.text(0.95, 0.05,
            f"Friedman: χ²={fr.get('chi2', 0):.2f}\np={fr.get('p_value', 0):.2e}\nNemenyi CD={fr.get('cd', 0):.2f}",
            transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # (b) Wilcoxon pairwise
    ax = axes[1]
    wilcoxon = stats.get('wilcoxon', {})
    if wilcoxon:
        baselines = list(wilcoxon.keys())
        p_values = [wilcoxon[b]['p_value'] for b in baselines]
        colors_w = [C_NSGA2, C_PACO, C_IVND][:len(baselines)]
        bars = ax.bar(range(len(baselines)), p_values, color=colors_w, edgecolor='white', linewidth=0.8)
        ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='α=0.05')
        ax.axhline(y=0.01, color='red', linestyle=':', alpha=0.3, label='α=0.01')
        for bar, p in zip(bars, p_values):
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                   f'p={p:.4f} {sig}', ha='center', fontweight='bold', fontsize=9)
        ax.set_xticks(range(len(baselines)))
        ax.set_xticklabels([f'vs {b}' for b in baselines], fontsize=9)
        ax.set_ylabel('p-value (Wilcoxon signed-rank)')
        ax.set_title('(b) Wilcoxon Pairwise Tests', fontweight='bold')
        ax.legend(fontsize=7)
        ax.set_yscale('log')

    fig.suptitle('Figure 7: Statistical Validation — Friedman + Wilcoxon Tests',
                 fontweight='bold', fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig7_statistics.png'))
    plt.close()
    print('  Saved: fig7_statistics.png')


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 8: Pipeline Schematic (updated)
# ══════════════════════════════════════════════════════════════════════════

def fig8_pipeline():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 5); ax.axis('off')

    steps = [
        {'x': 1, 'y': 2.5, 'w': 2.2, 'h': 2.0, 'color': '#E3F2FD', 'border': '#1565C0',
         'num': '1', 'title': 'Budget-Aware\nClustering',
         'sub': 'Spatial K-means\nTargeted splits at\n75% time budget'},
        {'x': 4, 'y': 2.5, 'w': 2.2, 'h': 2.0, 'color': '#E8F5E9', 'border': '#2E7D32',
         'num': '2', 'title': 'POMO\nNeural Routing',
         'sub': 'Transformer Encoder\n8-fold Augmentation\nCVRP Transfer Learning'},
        {'x': 7, 'y': 2.5, 'w': 2.2, 'h': 2.0, 'color': '#FFF3E0', 'border': '#E65100',
         'num': '3', 'title': 'Forward Insertion\nRepair [core]',
         'sub': 'Surgical: move tardy\ncustomers forward\nJackson 1955 fallback\n97% fallback reduction'},
        {'x': 10, 'y': 2.5, 'w': 2.2, 'h': 2.0, 'color': '#FCE4EC', 'border': '#C62828',
         'num': '4', 'title': 'EV\nEvaluation',
         'sub': 'Model A: No charging\nModel B: Linear\nModel C: Non-linear\nBinding analysis'},
    ]

    for step in steps:
        rect = plt.Rectangle((step['x'], step['y']-step['h']/2), step['w'], step['h'],
                            facecolor=step['color'], edgecolor=step['border'],
                            linewidth=2, alpha=0.9, zorder=2)
        ax.add_patch(rect)
        circle = plt.Circle((step['x']+step['w']/2, step['y']+step['h']/2-0.15),
                           0.3, facecolor=step['border'], edgecolor='white', linewidth=1.5, zorder=3)
        ax.add_patch(circle)
        ax.text(step['x']+step['w']/2, step['y']+step['h']/2-0.15, step['num'],
               ha='center', va='center', color='white', fontweight='bold', fontsize=12, zorder=4)
        ax.text(step['x']+step['w']/2, step['y']+step['h']/2-0.6, step['title'],
               ha='center', va='center', fontweight='bold', fontsize=9, color=step['border'], zorder=4)
        ax.text(step['x']+step['w']/2, step['y']-0.1, step['sub'],
               ha='center', va='center', fontsize=7, color='#555555', zorder=4, linespacing=1.3)

    for i in range(len(steps)-1):
        ax.annotate('', xy=(steps[i+1]['x'], steps[i]['y']),
                   xytext=(steps[i]['x']+steps[i]['w']+0.05, steps[i]['y']),
                   arrowprops=dict(arrowstyle='->', color='#666666', lw=2.5))

    ax.annotate('CORE CONTRIBUTION\nSurgical forward moves\n97% fallback reduction\n80% zero-fallback',
               xy=(8.1, 4.0), fontsize=8, fontweight='bold', color='#E65100',
               ha='center', linespacing=1.3,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF8E1', edgecolor='#E65100', alpha=0.9))

    ax.text(7, 0.3, '"Where to go" (POMO) + "In what order" (Forward Insertion) → Surgical TW repair preserves distance optimization',
           ha='center', fontsize=9, fontstyle='italic', color='#333333',
           bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='#CCCCCC', alpha=0.7))

    ax.set_title('Pipeline: Budget-Aware Clustering → POMO Neural Routing → Forward Insertion Repair → EV Evaluation',
                fontweight='bold', fontsize=13, pad=15)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig8_pipeline.png'))
    plt.close()
    print('  Saved: fig8_pipeline.png')


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('Generating comprehensive Week 8 figures...\n')

    for fn in [fig1_tw_feasibility, fig2_tardiness, fig3_fallback_reduction,
               fig4_cost_impact, fig5_dashboard, fig6_ev_analysis,
               fig7_statistics, fig8_pipeline]:
        try:
            fn()
        except Exception as e:
            print(f'  ERROR in {fn.__name__}: {e}')
            import traceback; traceback.print_exc()

    print(f'\nDone! Figures saved to: {OUTPUT_DIR}')
