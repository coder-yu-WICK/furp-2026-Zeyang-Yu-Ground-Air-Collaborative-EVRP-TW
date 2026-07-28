#!/usr/bin/env python3
"""
Generate FURP 2026 Showcase Poster.
Output: FURP_Showcase.pdf in repository root.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os, sys

# ── Paths ──
_W7 = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_W7)
for p in [os.path.join(_W7, '..', 'week3'),
          os.path.join(_W7, '..', 'week4'),
          os.path.join(_W7, '..', 'week5'),
          os.path.join(_W7, '..', 'week6'),
          _W7]:
    if p not in sys.path:
        sys.path.insert(0, p)

FIGS_DIR = os.path.join(_W7, 'figures')
OUTPUT = os.path.join(_PROJ, 'FURP_Showcase.pdf')

# ── Style ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
})

# Color palette
C_BLUE = '#0D3B66'
C_ORANGE = '#EE6C4D'
C_GREEN = '#2A9D8F'
C_YELLOW = '#E9C46A'
C_LIGHT = '#F4F1DE'
C_DARK = '#1A1A2E'
C_GREY = '#6C757D'

# ── Poster dimensions (A0: 84.1 x 118.9 cm, landscape) ──
FIG_W, FIG_H = 48, 33  # inches (~A0 landscape at 72 dpi equivalent)


def draw_section_box(ax, x, y, w, h, color, alpha=0.08):
    """Draw a rounded section background."""
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.02",
                          facecolor=color, edgecolor=color,
                          alpha=alpha, linewidth=2,
                          transform=ax.transAxes)
    ax.add_patch(rect)


def draw_section_title(ax, x, y, text, color=C_BLUE):
    """Draw a section title."""
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=16, fontweight='bold', color=color,
            va='top', ha='left')


def create_poster():
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # ═══════════════════════════════════════════════════════════════
    # HEADER
    # ═══════════════════════════════════════════════════════════════
    # Top bar
    ax.axhline(y=0.92, xmin=0, xmax=1, color=C_BLUE, linewidth=3)
    rect = FancyBboxPatch((0, 0.92), 1, 0.08,
                          boxstyle="round,pad=0",
                          facecolor=C_BLUE, edgecolor='none',
                          transform=ax.transAxes)
    ax.add_patch(rect)

    # Title
    ax.text(0.5, 0.97, 'Ground-Air Collaborative EVRP-TW',
            transform=ax.transAxes, fontsize=38, fontweight='bold',
            color='white', ha='center', va='center')
    ax.text(0.5, 0.94, 'Hybrid Optimization for Truck-Drone Delivery',
            transform=ax.transAxes, fontsize=22, fontweight='normal',
            color=C_YELLOW, ha='center', va='center')

    # Author line
    ax.text(0.5, 0.915, 'FURP 2026 — Final Showcase  |  July 2026',
            transform=ax.transAxes, fontsize=12, color=C_GREY,
            ha='center', va='center', style='italic')

    # ═══════════════════════════════════════════════════════════════
    # COLUMN 1: Problem + Method (left: 0.02–0.32)
    # ═══════════════════════════════════════════════════════════════
    L1, R1 = 0.02, 0.32
    y_top = 0.89

    # ── Research Question ──
    draw_section_box(ax, L1, 0.78, R1-L1, 0.11, C_BLUE)
    draw_section_title(ax, L1+0.01, 0.885, 'Research Question', C_BLUE)
    ax.text(L1+0.01, 0.865, 'How do charging strategies and truck-drone',
            transform=ax.transAxes, fontsize=13, va='top')
    ax.text(L1+0.01, 0.847, 'coordination affect the feasibility and',
            transform=ax.transAxes, fontsize=13, va='top')
    ax.text(L1+0.01, 0.829, 'efficiency of EVRP-TW?',
            transform=ax.transAxes, fontsize=13, va='top', fontweight='bold')
    ax.text(L1+0.01, 0.805, 'First method to simultaneously address VRPTW + EV + '
            'Truck-Drone + Sync at 200-customer scale.',
            transform=ax.transAxes, fontsize=10, va='top', style='italic',
            color=C_GREY)

    # ── Four-Model Ablation ──
    draw_section_box(ax, L1, 0.55, R1-L1, 0.215, C_GREEN)
    draw_section_title(ax, L1+0.01, 0.755, 'Four-Model Ablation', C_GREEN)

    models = [
        ('A: Baseline', 'Truck + TW + EDD Repair', C_GREY),
        ('B: +Linear EV', 'Constant-rate charging, CS insertion', '#457B9D'),
        ('C: +Non-linear EV', 'Piecewise charging (fast→slow)', '#1D3557'),
        ('D: +Synchronization', 'Launch-recovery + cascading delays', C_BLUE),
    ]
    for i, (name, desc, color) in enumerate(models):
        yy = 0.735 - i * 0.042
        ax.text(L1+0.01, yy, name, transform=ax.transAxes, fontsize=12,
                fontweight='bold', color=color, va='top')
        ax.text(L1+0.24, yy, desc, transform=ax.transAxes, fontsize=10,
                color=C_DARK, va='top')

    # Pipeline
    ax.text(L1+0.01, 0.565, 'Pipeline: Construction → Repair → Drone → EV → Sync',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            color=C_DARK, va='top',
            bbox=dict(boxstyle='round', facecolor=C_LIGHT, alpha=0.8))

    # ── Instance Types ──
    draw_section_box(ax, L1, 0.41, R1-L1, 0.125, C_ORANGE)
    draw_section_title(ax, L1+0.01, 0.525, 'Benchmark', C_ORANGE)
    ax.text(L1+0.01, 0.505, 'Solomon VRPTW: 6 types × 3 scales (50/100/200c)',
            transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')
    ax.text(L1+0.01, 0.485, 'RC1/RC2 (mixed) | R1/R2 (random) | C1/C2 '
            '(clustered)\n14 methods: Ours (5 variants) + 9 classical baselines',
            transform=ax.transAxes, fontsize=10, va='top')

    # ── Literature Positioning ──
    draw_section_box(ax, L1, 0.04, R1-L1, 0.35, C_BLUE)
    draw_section_title(ax, L1+0.01, 0.38, 'Literature Positioning', C_BLUE)

    lit_text = (
        'Our work sits at the intersection of three research streams:\n\n'
        '• VRPTW (Solomon 1987): Time-window constrained routing\n'
        '• E-VRP (Schneider et al. 2014; Keskin & Çatay 2016): EV battery\n'
        '  and charging constraints with linear/non-linear profiles\n'
        '• Truck-Drone (Murray & Chu 2015; Yin et al. 2023): Collaborative\n'
        '  delivery with cross-route drone insertion\n\n'
        'No published method addresses all three simultaneously.\n'
        'Closest: Liu et al. (2024) — truck-drone+TW, no EV, 100c max.\n'
        'Schneider et al. (2014) — EV+TW, no drone, 100c max.'
    )
    ax.text(L1+0.01, 0.36, lit_text, transform=ax.transAxes,
            fontsize=9.5, va='top', linespacing=1.3)

    # ═══════════════════════════════════════════════════════════════
    # COLUMN 2: Main Results (center: 0.34–0.66)
    # ═══════════════════════════════════════════════════════════════
    L2, R2 = 0.34, 0.66

    # ── Key Metrics Box ──
    draw_section_box(ax, L2, 0.76, R2-L2, 0.13, C_BLUE)
    draw_section_title(ax, L2+0.01, 0.88, 'Key Results at a Glance', C_BLUE)

    metrics = [
        ('100%', 'TW Feasibility\n(all 18 instances)', C_GREEN),
        ('17.4%', 'Avg Drone Cost\nSavings (50-100c)', C_BLUE),
        ('0%', 'Classical Methods\nTW Feasibility', C_ORANGE),
        ('74%', 'Instances with\nSync Wait Time', '#1D3557'),
    ]
    for i, (val, label, color) in enumerate(metrics):
        xx = L2 + 0.01 + i * 0.08
        ax.text(xx, 0.855, val, transform=ax.transAxes, fontsize=24,
                fontweight='bold', color=color, va='center')
        ax.text(xx + 0.03, 0.855, label, transform=ax.transAxes, fontsize=8.5,
                color=C_DARK, va='center')

    # ── Experiment 1: Scale Test ──
    draw_section_box(ax, L2, 0.48, R2-L2, 0.265, C_GREEN)
    draw_section_title(ax, L2+0.01, 0.735, 'Exp 1: Scale Test (50/100/200c)', C_GREEN)

    # Mini table
    table_data = [
        ['Instance', 'Ours-2D', 'CW-Sav', 'DroneΔ%', 'Feas%'],
        ['RC101_50c',  '1091', '1610', '+32.2%', '100%'],
        ['RC201_100c', '2643', '3140', '+15.8%', '100%'],
        ['R101_200c',  '4120', '7077', '+41.8%', '100%'],
        ['C101_200c',  '3615', '3615',  '0.0%',  '100%'],
    ]
    for ri, row in enumerate(table_data):
        yy = 0.715 - ri * 0.026
        for ci, cell in enumerate(row):
            xx = L2 + 0.01 + ci * 0.065
            fw = 'bold' if ri == 0 else 'normal'
            fs = 9 if ri == 0 else 8.5
            color = C_GREEN if (ri > 0 and ci == 3 and '+' in str(cell)) else C_DARK
            ax.text(xx, yy, str(cell), transform=ax.transAxes, fontsize=fs,
                    fontweight=fw, color=color, va='center')

    ax.text(L2+0.01, 0.57, '→ Classical methods (NSGA-II, P-ACO, IVND): 0% TW '
            'feasibility — massive violations',
            transform=ax.transAxes, fontsize=9, va='top', color=C_ORANGE,
            fontweight='bold')
    ax.text(L2+0.01, 0.552, '→ EDD Repair is the decisive component: 50→100% '
            'feasibility improvement',
            transform=ax.transAxes, fontsize=9, va='top', color=C_GREEN)
    ax.text(L2+0.01, 0.534, '→ C-type at 200c: drones net-negative — composite '
            'fallback rejects all',
            transform=ax.transAxes, fontsize=9, va='top', color=C_GREY)

    # ── Experiment 2: EV Charging ──
    draw_section_box(ax, L2, 0.32, R2-L2, 0.145, '#457B9D')
    draw_section_title(ax, L2+0.01, 0.455, 'Exp 2: EV Charging Study (Stress Test)', '#1D3557')

    ev_table = [
        ['Battery',  'ΔCost(B-A)', 'ΔCost(C-B)', 'CS/Inst', 'EV-Feas'],
        ['25 kWh',   '+13.7%',     '+7.1%',      '6.9',     '4.8%'],
        ['30 kWh',   '+7.2%',      '+0.7%',      '3.4',     '14.3%'],
        ['100 kWh',  '0.0%',       '0.0%',       '0.0',     '100%'],
    ]
    for ri, row in enumerate(ev_table):
        yy = 0.435 - ri * 0.021
        for ci, cell in enumerate(row):
            xx = L2 + 0.01 + ci * 0.065
            fw = 'bold' if ri == 0 else 'normal'
            fs = 8.5 if ri == 0 else 8
            ax.text(xx, yy, str(cell), transform=ax.transAxes, fontsize=fs,
                    fontweight=fw, color=C_DARK, va='center')

    ax.text(L2+0.01, 0.345, '→ At 100 kWh (default): EV constraints non-binding — '
            'legitimate fleet finding',
            transform=ax.transAxes, fontsize=8.5, va='top', color=C_GREY)
    ax.text(L2+0.01, 0.332, '→ Non-linear: bidirectional effect (−8% at low SOC, '
            '+7.1% at high SOC)',
            transform=ax.transAxes, fontsize=8.5, va='top', color='#1D3557')

    # ── Experiment 3: Synchronization ──
    draw_section_box(ax, L2, 0.16, R2-L2, 0.145, C_BLUE)
    draw_section_title(ax, L2+0.01, 0.295, 'Exp 3: Sync Study (Model C vs D)', C_BLUE)

    sync_table = [
        ['Metric',      'Model C',  'Model D',  'Δ'],
        ['Avg Cost',    '1114.2',   '1101.7',   '−12.5'],
        ['Sync Wait',   '0.0 min',  '44.6 min', '+44.6'],
        ['n-Drones',    '4.2',      '6.8',      '+2.6'],
    ]
    for ri, row in enumerate(sync_table):
        yy = 0.276 - ri * 0.022
        for ci, cell in enumerate(row):
            xx = L2 + 0.01 + ci * 0.082
            fw = 'bold' if ri == 0 else 'normal'
            fs = 9 if ri == 0 else 8.5
            ax.text(xx, yy, str(cell), transform=ax.transAxes, fontsize=fs,
                    fontweight=fw, color=C_DARK, va='center')

    ax.text(L2+0.01, 0.188, '→ 74% of instances require truck waiting when sync '
            'is properly modeled',
            transform=ax.transAxes, fontsize=8.5, va='top', color=C_BLUE)
    ax.text(L2+0.01, 0.173, '→ Two-pass cascading delay algorithm propagates '
            'waits through all subsequent nodes',
            transform=ax.transAxes, fontsize=8.5, va='top', color=C_GREY)

    # ═══════════════════════════════════════════════════════════════
    # COLUMN 3: Visuals + Failure Cases + Conclusions (0.68–0.98)
    # ═══════════════════════════════════════════════════════════════
    L3, R3 = 0.68, 0.98

    # ── Route Map Figure ──
    draw_section_box(ax, L3, 0.65, R3-L3, 0.24, C_BLUE)
    draw_section_title(ax, L3+0.01, 0.88, 'Route Maps: Truck-Drone Geometry', C_BLUE)

    # Try to embed route map images
    route_imgs = [
        os.path.join(FIGS_DIR, 'fig_route_comparison_nd_vs_2d.png'),
        os.path.join(FIGS_DIR, 'fig7_route_map_panel.png'),
    ]
    for i, img_path in enumerate(route_imgs):
        if os.path.exists(img_path):
            try:
                img = plt.imread(img_path)
                xx = L3 + 0.005 + i * 0.15
                ax.imshow(img, extent=[xx, xx+0.145, 0.67, 0.87],
                         aspect='auto', transform=ax.transAxes)
                label = ['Truck-Only vs 2-Drone\n(RC101, 50 customers)',
                        '2×2 Panel: 3 Instance\nTypes + EV Model'][i]
                ax.text(xx + 0.07, 0.66, label, transform=ax.transAxes,
                       fontsize=7, ha='center', va='top', style='italic')
            except Exception:
                pass

    # ── Failure Cases ──
    draw_section_box(ax, L3, 0.45, R3-L3, 0.185, C_ORANGE)
    draw_section_title(ax, L3+0.01, 0.625, 'Failure Cases (5 Systematic)', C_ORANGE)

    failures = [
        ('1. Battery Starvation', '30 kWh, 50c route → energy violation'),
        ('2. TW Tightening', 'Scale due_times ×50% → tardiness 850+'),
        ('3. Capacity Exceeded', 'Route load >200 → infeasible'),
        ('4. Sync Failure', 'Drone faster than truck → 44.6 min avg wait'),
        ('5. Charging Necessity', 'No CS, >40c route → battery depletion'),
    ]
    for i, (name, desc) in enumerate(failures):
        yy = 0.608 - i * 0.03
        ax.text(L3+0.01, yy, name, transform=ax.transAxes, fontsize=10,
                fontweight='bold', color=C_ORANGE, va='top')
        ax.text(L3+0.17, yy, desc, transform=ax.transAxes, fontsize=8.5,
                color=C_DARK, va='top')

    # ── Method Comparison Bar ──
    draw_section_box(ax, L3, 0.28, R3-L3, 0.155, C_GREEN)
    draw_section_title(ax, L3+0.01, 0.425, 'Method Landscape', C_GREEN)

    methods_list = [
        'Ours-2D → 100% feasible, 17.4% drone savings',
        'Ours-1D → 100% feasible, 13.1% savings',
        'CW-Savings → 100% feasible, drone-unfriendly',
        'POMO-Raw → 50% feasible, no repair',
        'NSGA-II → 0% feasible, tardiness=12,450',
        'P-ACO → 0% feasible, tardiness=9,850',
        'IVND → 0% feasible, tardiness=11,200',
    ]
    for i, m in enumerate(methods_list):
        yy = 0.408 - i * 0.018
        color = C_GREEN if '100%' in m else (C_ORANGE if '0%' in m else C_GREY)
        ax.text(L3+0.01, yy, m, transform=ax.transAxes, fontsize=8.5,
                color=color, va='top')

    # ── Conclusions ──
    draw_section_box(ax, L3, 0.04, R3-L3, 0.225, C_BLUE)
    draw_section_title(ax, L3+0.01, 0.255, 'Conclusions & Future Work', C_BLUE)

    conclusions = (
        '1. Hybrid neural-classical pipeline achieves 100% TW feasibility\n'
        '   at 200c — the first method to do so with drones + EV + sync.\n'
        '2. EDD repair is the decisive component (50%→100% feasibility).\n'
        '3. Drone integration provides 13–17% cost savings across scales.\n'
        '4. EV constraints non-binding at standard fleet parameters — a\n'
        '   practical finding for urban fleet operators.\n'
        '5. C-type instances structurally unsuitable for drone delivery.\n'
        '6. 74% of instances require truck waiting under proper sync.\n\n'
        'Future: ALNS integration, inter-route repair for R1/C1,\n'
        'improved CS heuristic, real-world instance validation.'
    )
    ax.text(L3+0.01, 0.238, conclusions, transform=ax.transAxes,
            fontsize=9.5, va='top', linespacing=1.4, color=C_DARK)

    # ── Footer ──
    ax.axhline(y=0.02, xmin=0, xmax=1, color=C_BLUE, linewidth=2)
    ax.text(0.5, 0.01, 'FURP 2026 — Ground-Air Collaborative EVRP-TW  |  '
            'Hybrid Optimization for Truck-Drone Delivery  |  Final Report: week7/final_report.pdf',
            transform=ax.transAxes, fontsize=9, color=C_GREY,
            ha='center', va='center')

    # ── Save ──
    fig.savefig(OUTPUT, facecolor='white', dpi=72, bbox_inches='tight')
    plt.close(fig)
    print(f'Poster saved: {OUTPUT}')
    print(f'Size: {os.path.getsize(OUTPUT) / 1024 / 1024:.1f} MB')


if __name__ == '__main__':
    create_poster()
