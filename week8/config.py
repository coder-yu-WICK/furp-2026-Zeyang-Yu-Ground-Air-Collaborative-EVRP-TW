# -*- coding: utf-8 -*-
"""
Week 8 Configuration — EVRP-TW with EDD Repair (truck-only, no drones).

Teacher feedback: classmate chose truck+drone basic model, so we differentiate
by focusing on EVRP-TW + EDD repair as our unique contribution.

Pipeline: Clustering → POMO Neural Routing → EDD Repair → EV Evaluation (A/B/C)
"""
import os

# ── Paths ──────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BASE_DIR)
DATA_JSON_DIR = os.path.join(_PROJECT_ROOT, 'py-ga-VRPTW', 'data', 'json')
DATA_DIR = os.path.join(_BASE_DIR, 'data')
RESULTS_DIR = os.path.join(_BASE_DIR, 'results')
FIGURES_DIR = os.path.join(_BASE_DIR, 'figures')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Dataset & Instance Configuration ───────────────────────────────
RC1_INSTANCES = ['RC101', 'RC102', 'RC103', 'RC104', 'RC105', 'RC106', 'RC107', 'RC108']
RC2_INSTANCES = ['RC201', 'RC202', 'RC203', 'RC204', 'RC205', 'RC206', 'RC207', 'RC208']
R1_INSTANCES  = ['R101', 'R102', 'R103', 'R104', 'R105', 'R106', 'R107', 'R108',
                 'R109', 'R110', 'R111', 'R112']
R2_INSTANCES  = ['R201', 'R202', 'R203', 'R204', 'R205', 'R206', 'R207', 'R208',
                 'R209', 'R210', 'R211']
C1_INSTANCES  = ['C101', 'C102', 'C103', 'C104', 'C105', 'C106', 'C107', 'C108', 'C109']
C2_INSTANCES  = ['C201', 'C202', 'C203', 'C204', 'C205', 'C206', 'C207', 'C208']

ALL_INSTANCE_LISTS = {
    'RC1': RC1_INSTANCES, 'RC2': RC2_INSTANCES,
    'R1': R1_INSTANCES,   'R2': R2_INSTANCES,
    'C1': C1_INSTANCES,   'C2': C2_INSTANCES,
}

COORD_SCALE = 0.16        # Solomon [0,100] → Urban [0,16] km
DEPOT = (8.0, 8.0)
CUSTOMER_SIZES = [25, 50, 100, 200]

TW_TYPES = {
    'RC1': {'name': 'tight', 'horizon': 120.0},
    'RC2': {'name': 'wide',  'horizon': 240.0},
    'R1':  {'name': 'tight', 'horizon': 120.0},
    'R2':  {'name': 'wide',  'horizon': 240.0},
    'C1':  {'name': 'tight', 'horizon': 120.0},
    'C2':  {'name': 'wide',  'horizon': 240.0},
}

# ── Vehicle Configuration (truck-only) ─────────────────────────────
TRUCK_SPEED = 35.0          # km/h
TRUCK_CAPACITY = 200.0

# Fleet sizing: customer_size → list of truck counts to test
TRUCK_FLEET_CONFIGS = {
    25:  [2],
    50:  [4, 6],
    100: [4, 6, 8],
    200: [8, 10],
}

# ── Cost Parameters ─────────────────────────────────────────────────
TRUCK_FIXED_COST = 100.0    # per truck
TRUCK_DIST_COST_RATE = 2.0  # per km
TARDINESS_COST_RATE = 1.0   # per time unit (primary optimization weight)

# ── EV Charging Parameters (for electric trucks) ────────────────────
CHARGING_STATIONS = [
    (8.0, 8.0),    # depot as charging station
    (4.0, 12.0),   # NW quadrant
    (12.0, 4.0),   # SE quadrant
]
CHARGING_RATE = 1.0              # kWh per time unit (linear)
BATTERY_CAPACITY = 55.0          # kWh (scaled for 16×16 km urban zone)
# Rationale: 55 kWh / 1.5 kWh/km = 36.7 km range. City max one-way ~11 km.
# C-type clustered routes (13-37 kWh): always feasible.
# RC/R-type routes (26-90 kWh): binding on ~40-50% of routes.
# This creates a research-valuable constraint that differentiates instance types.
ENERGY_CONSUMPTION_RATE = 1.5    # kWh/km

# Nonlinear charging segments: (soc_low, soc_high, rate_multiplier)
CHARGING_SEGMENTS = [
    (0.0, 0.2, 1.5),    # 0-20%: fast charge
    (0.2, 0.8, 1.0),    # 20-80%: normal charge
    (0.8, 1.0, 0.5),    # 80-100%: slow charge
]

# ── Algorithm Hyperparameters ──────────────────────────────────────

# P-ACO (Das et al. 2020, IEEE TITS)
PACO = {
    'ants_25c': 50,
    'ants_50c': 80,
    'ants_100c': 120,
    'iterations': 100,
    'alpha': 1.0,
    'beta': 2.0,
    'rho': 0.15,
    'q0': 0.5,
    'Q_cost': 120,
    'Q_tard': 60,
    'tau0_cost': 1.0,
    'tau0_tard': 1.0,
}

# NSGA-II (Deb et al. 2002, IEEE TEC)
NSGA2 = {
    'pop_25c': 50,
    'pop_50c': 80,
    'pop_100c': 150,
    'generations': 120,
    'crossover_pb': 0.9,
    'mutation_pb': 0.1,
    'crossover_eta': 20,
    'mutation_eta': 20,
    'tournament_size': 2,
}

# IVND — truck-only neighborhoods (Wu et al. 2022, IEEE TITS)
IVND = {
    'max_iterations': 200,
    'tabu_tenure': 15,
    'shaking_k_max': 5,
    'temperature_initial': 100.0,
    'cooling_rate': 0.95,
    'neighborhood_structures': [
        'relocate_truck',
        'swap_truck',
        'two_opt_truck',
    ],
}

# ── POMO (Neural Construction) Parameters ──────────────────────────
POMO = {
    'embedding_dim': 128,
    'encoder_layers': 6,
    'heads': 8,
    'qkv_dim': 16,
    'ff_hidden': 512,
    'logit_clipping': 10.0,
    'training_epochs': 80,
    'episodes_per_epoch': 400,
    'batch_size': 64,
    'lr': 1e-4,
    'weight_decay': 1e-6,
    'use_augmentation': True,
}

# ── TW-Aware Clustering Parameters ─────────────────────────────────
TW_AWARE = {
    'alpha': 1.0,         # spatial weight
    'beta': 0.5,          # temporal weight
    'tw_horizon_rc1': 120.0,
    'tw_horizon_rc2': 240.0,
}

# ── Clustering Variants (truck-only, no drone variants) ────────────
CLUSTERING_VARIANTS = [
    'baseline',          # K-means spatial only
    'tw_aware',          # TW-aware two-phase
    'adaptive_tw',       # Adaptive TW clustering
    'angle',             # Angle-based petal
    'hybrid',            # RC1→spatiotemporal, RC2→tw_aware
    'spatiotemporal',    # Joint spatio-temporal K-means (NEW)
]

# ── IVND Repair Parameters ─────────────────────────────────────────
IVND_REPAIR = {
    'max_iterations': 500,
    'tabu_tenure': 15,
    'temperature': 0.5,
    'cooling_rate': 0.95,
    'improvement_threshold': 0.0,
}

# ── Pipeline Variants ──────────────────────────────────────────────
PIPELINE_VARIANTS = [
    'w5_baseline',        # Clustering + POMO, no repair
    'w5_plus_repair',     # + EDD repair
]

# ── Experiment Parameters ──────────────────────────────────────────
N_REPEATS = 10
RANDOM_SEEDS = list(range(42, 42 + N_REPEATS))
HV_REFERENCE = (170.0, 140.0)
