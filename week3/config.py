# -*- coding: utf-8 -*-
"""
Master configuration for Truck-Drone EVRP-TW experiments.
All parameters are centralized here for reproducibility.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_JSON_DIR = os.path.join(PROJECT_ROOT, 'py-ga-VRPTW', 'data', 'json')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DATA_OUT_DIR = os.path.join(BASE_DIR, 'data')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_OUT_DIR, exist_ok=True)

# ── Dataset & Instance Configuration ───────────────────────────────
# Solomon instances — all 56 instances across 6 types
RC1_INSTANCES = ['RC101', 'RC102', 'RC103', 'RC104', 'RC105', 'RC106', 'RC107', 'RC108']
RC2_INSTANCES = ['RC201', 'RC202', 'RC203', 'RC204', 'RC205', 'RC206', 'RC207', 'RC208']
R1_INSTANCES  = ['R101', 'R102', 'R103', 'R104', 'R105', 'R106', 'R107', 'R108',
                 'R109', 'R110', 'R111', 'R112']
R2_INSTANCES  = ['R201', 'R202', 'R203', 'R204', 'R205', 'R206', 'R207', 'R208',
                 'R209', 'R210', 'R211']
C1_INSTANCES  = ['C101', 'C102', 'C103', 'C104', 'C105', 'C106', 'C107', 'C108', 'C109']
C2_INSTANCES  = ['C201', 'C202', 'C203', 'C204', 'C205', 'C206', 'C207', 'C208']

# All instance lists grouped by type
ALL_INSTANCE_LISTS = {
    'RC1': RC1_INSTANCES, 'RC2': RC2_INSTANCES,
    'R1': R1_INSTANCES,   'R2': R2_INSTANCES,
    'C1': C1_INSTANCES,   'C2': C2_INSTANCES,
}

# Coordinate scaling: Solomon [0, 100] → Urban [0, 16] km
COORD_SCALE = 0.16

# Depot location in urban coordinates (km)
DEPOT = (8.0, 8.0)

# Customer sizes to test
CUSTOMER_SIZES = [25, 50, 100, 200]

# Time window types — 6 Solomon categories
TW_TYPES = {
    'RC1': {'name': 'tight', 'horizon': 120.0},
    'RC2': {'name': 'wide',  'horizon': 240.0},
    'R1':  {'name': 'tight', 'horizon': 120.0},
    'R2':  {'name': 'wide',  'horizon': 240.0},
    'C1':  {'name': 'tight', 'horizon': 120.0},
    'C2':  {'name': 'wide',  'horizon': 240.0},
}

# ── Vehicle Configuration ──────────────────────────────────────────
TRUCK_SPEED = 35.0      # km/h
DRONE_SPEED = 50.0      # km/h
TRUCK_CAPACITY = 200.0
DRONE_CAPACITY = 40.0
DRONE_ENDURANCE = {
    'medium': 4.0,      # km
    'high': 6.0,        # km
}

# Vehicle fleet: customer_size → list of (trucks, drones)
VEHICLE_CONFIGS = {
    25:  [(2, 2)],
    50:  [(4, 4), (6, 6)],
    100: [(4, 4), (6, 6), (8, 8)],
    200: [(8, 8), (10, 10)],
}

# ── Cost Parameters ─────────────────────────────────────────────────
TRUCK_FIXED_COST = 100.0    # per truck
DRONE_FIXED_COST = 0.0      # per drone
TRUCK_DIST_COST_RATE = 2.0  # per km
DRONE_DIST_COST_RATE = 1.0  # per km
TARDINESS_COST_RATE = 1.0   # per time unit (priority weight)

# ── Algorithm Hyperparameters ──────────────────────────────────────

# P-ACO (from DOI: 10.1109/TITS.2020.2992549)
PACO = {
    'ants_25c': 50,
    'ants_50c': 80,
    'ants_100c': 120,
    'iterations': 100,
    'alpha': 1.0,          # pheromone weight
    'beta': 2.0,           # heuristic weight
    'rho': 0.15,           # evaporation rate
    'q0': 0.5,             # exploitation vs exploration
    'Q_cost': 120,         # pheromone deposit for cost
    'Q_tard': 60,           # pheromone deposit for tardiness
    'tau0_cost': 1.0,
    'tau0_tard': 1.0,
}

# NSGA-II (from Deb et al. 2002)
NSGA2 = {
    'pop_25c': 50,
    'pop_50c': 80,
    'pop_100c': 150,
    'generations': 120,
    'crossover_pb': 0.9,
    'mutation_pb': 0.1,
    'crossover_eta': 20,   # SBX distribution index
    'mutation_eta': 20,    # polynomial mutation distribution index
    'tournament_size': 2,
}

# IVND (from DOI: 10.1109/TITS.2022.3181282)
IVND = {
    'max_iterations': 200,
    'tabu_tenure': 15,
    'shaking_k_max': 5,
    'temperature_initial': 100.0,
    'cooling_rate': 0.95,
    'neighborhood_structures': [
        'relocate_truck',
        'relocate_drone',
        'swap_truck',
        'swap_drone',
        'two_opt_truck',
        'drone_to_truck',
        'truck_to_drone',
    ],
}

# ── Experiment Parameters ──────────────────────────────────────────
N_REPEATS = 10              # runs per configuration
RANDOM_SEEDS = list(range(42, 42 + N_REPEATS))

# Multi-objective reference point for Hypervolume
HV_REFERENCE = (170.0, 140.0)

# ── Nonlinear Charging (from DOI: S1366554526002978) ────────────────
# Piecewise linear approximation of nonlinear charging curve
CHARGING_STATIONS = [
    (8.0, 8.0),   # depot also serves as charging station
    (4.0, 12.0),
    (12.0, 4.0),
]
CHARGING_RATE = 1.0          # kWh per time unit (linear approximation)
BATTERY_CAPACITY = 100.0     # kWh for electric trucks
# Nonlinear charging: f(SOC) = piecewise segments (SOC_bins → rate)
# Simplified from the paper's Model 3
CHARGING_SEGMENTS = [
    (0.0, 0.2, 1.5),    # 0-20% SOC: fast charge, 1.5× rate
    (0.2, 0.8, 1.0),    # 20-80% SOC: normal charge, 1.0× rate
    (0.8, 1.0, 0.5),    # 80-100% SOC: slow charge, 0.5× rate
]
