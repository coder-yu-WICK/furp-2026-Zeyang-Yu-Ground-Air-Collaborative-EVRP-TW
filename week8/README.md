# Week 8 — EVRP-TW with EDD Repair (Truck-Only)

> **Teacher Guidance (2026-07-29):** Classmate chose truck+drone basic model.  
> Differentiate by removing truck-drone collaboration and focusing on  
> **EVRP-TW + EDD Repair** as our unique contribution.

---

## Research Positioning

| Before (Week 3-7) | After (Week 8+) |
|---|---|
| Truck-Drone Collaborative EVRP-TW | **EVRP-TW** (Electric Vehicle Routing with Time Windows) |
| 4 models: A/B/C/D (D = drone sync) | **3 models: A/B/C** (EV charging only) |
| 12 methods including drone variants | **Truck-only methods** |
| Story: drones + EV + TW | **Story: EDD Repair** for TW feasibility |

### Unique Contribution
**EDD (Earliest Due Date) Repair** — a simple scheduling heuristic (Jackson 1955)
that reorders customers by due_date ascending. Provably optimal for minimizing
maximum lateness (Lmax) on a single route. Applied to EVRP-TW as a post-processing
step after POMO neural routing, achieving **100% time-window feasibility** in
sub-second time, where NSGA-II, P-ACO, and IVND all achieve 0%.

---

## Pipeline

```
Solomon Instance
  → 1. Clustering (Hybrid: RC1→Angle Petal, RC2→Adaptive TW)
  → 2. POMO Neural Routing (Pre-trained Transformer, 8-fold augmentation)
  → 3. EDD Repair (Adaptive: ≤50c Partial EDD, 100c Full EDD)
  → 4. EV Evaluation (Model A: baseline, B: linear charging, C: non-linear)
```

**Removed:** Cross-route drone insertion, drone post-processing, drone re-optimization,
sync evaluation (Model D).

---

## Directory Structure

```
week8/
├── README.md
├── config.py                    # Drone-free config
├── core/
│   ├── problem_model.py         # TruckSolution class
│   └── data_loader.py           # Solomon instance loading
├── algorithms/
│   ├── nsga2.py                 # NSGA-II (truck-only)
│   ├── paco.py                  # P-ACO (truck-only)
│   ├── ivnd.py                  # IVND (truck-only, 3 neighborhoods)
│   ├── no_drone.py              # GA truck-only baseline
│   └── pomo/                    # POMO neural network (5 files)
├── pipeline/
│   ├── pipeline.py              # solve_evrptw() main entry
│   ├── repair.py                # EDD repair operators (truck-only)
│   ├── clustering.py            # TW-aware two-phase clustering
│   ├── adaptive_clustering.py   # Adaptive/angle/hybrid strategies
│   ├── cluster_feasibility.py   # Temporal feasibility pre-check
│   ├── pomo_solver.py           # POMO multi-truck solver
│   └── pomo_multitruck.py       # K-means + POMO routing
├── ev/
│   └── ev_model.py              # EVTruckSolution (A/B/C charging)
├── experiments/
│   ├── clustering_baselines.py  # Sweep, CW-Savings, K-means baselines
│   ├── gap_analysis.py          # OR-Tools optimality gap
│   ├── statistical_tests.py     # Friedman + Wilcoxon
│   └── analyze_results.py       # Results analysis
├── visualization/
│   ├── visualize_paper.py
│   ├── visualize_results.py
│   └── fig_route_maps.py
├── data/ → ../week3/data/       # Symlink to 224 Solomon instances
└── results/                     # Experiment output
```

---

## Quick Start

```bash
cd "/Users/jackalwick/Desktop/Truck-Drone EVRP-TW"

# Test imports
.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from week8.config import TRUCK_SPEED, BATTERY_CAPACITY
from week8.core.problem_model import TruckSolution
from week8.core.data_loader import load_instance_from_disk
from week8.pipeline.repair import repair_tardiness_truck
print('All imports OK')
"

# Run pipeline on a single instance
.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from week8.pipeline.pipeline import solve_evrptw
from week8.core.data_loader import load_instance_from_disk
inst = load_instance_from_disk('RC101_25c')
result = solve_evrptw(inst, n_trucks=2, variant='hybrid', use_repair=True, repair_mode='full', n_runs=1)
sol = result['solutions'][0]
print(f'Cost: {sol.cost:.0f}, Tardiness: {sol.tardiness:.0f}, Feasible: {sol.feasible}')
print(f'Repair: tardiness {result[\"repair_stats\"][\"tardiness_before\"]:.0f} → {result[\"repair_stats\"][\"tardiness_after\"]:.0f}')
"
```

---

## Key Changes from src/

| File | Change |
|------|--------|
| `config.py` | Removed all DRONE_* params, MAX_DRONES_PER_TRUCK, DRONE_PP, drone variants |
| `core/problem_model.py` | `TruckDroneSolution` → `TruckSolution`, removed ~60% drone eval code |
| `algorithms/nsga2.py` | Removed drone chromosome flags, simplified decode |
| `algorithms/paco.py` | Removed 3D drone pheromone, drone candidate construction |
| `algorithms/ivnd.py` | Removed 4 drone neighborhoods, kept 3 truck neighborhoods |
| `pipeline/pipeline.py` | New `solve_evrptw()`, removed drone insertion step |
| `pipeline/repair.py` | New truck-only EDD operators (no merge/re-insert cycle) |
| `pipeline/pomo_solver.py` | Removed drone post-processing step, 5 → 5 clustering variants |
| `ev/ev_model.py` | `EVTruckSolution(TruckSolution)`, removed drone/sync code |
| **Deleted** | `drone.py`, `drone_reopt.py`, `sync.py`, `run_sync_study.py` |

---

## Experiment Design

### Core Comparison
| Method | Type | TW Feasibility | Notes |
|--------|------|---------------|-------|
| **Ours (POMO + EDD)** | Neural + Heuristic | **100%** | Pipeline with EDD repair |
| NSGA-II | Evolutionary | 0% | Truck-only, no TW repair |
| P-ACO | Swarm | 0% | Truck-only, no TW repair |
| IVND | Local Search | 0% | Truck-only, no TW repair |
| CW-Savings | Constructive | 100% | But drone-unfriendly routes |
| Sweep+NN | Constructive | ~25% | Polar sweep + nearest neighbor |

### Ablation Dimensions
- **Repair strategy**: No repair vs Partial EDD vs Full EDD
- **EV charging**: Model A (none) vs B (linear) vs C (non-linear)
- **Scale**: 25c / 50c / 100c / 200c
- **Instance types**: RC1, RC2, R1, R2, C1, C2 (56 × 4 = 224 instances)

### Key Hypothesis
EDD repair is the decisive component — it alone bridges the gap from ~50% TW
feasibility (POMO raw) to 100%. The decomposition "POMO decides what-goes-where,
EDD decides in-what-order" separates two concerns that classical methods conflate.

---

## References

1. Jackson, J.R. (1955). Scheduling a production line to minimize maximum tardiness. *Management Science Research Report 43*, UCLA.
2. Kwon, Y.D. et al. (2020). POMO: Policy Optimization with Multiple Optima for Reinforcement Learning. *NeurIPS*.
3. Deb, K. et al. (2002). A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II. *IEEE TEC*, 6(2).
4. Das, D. et al. (2020). Synchronized Truck and Drone Routing in Package Delivery Logistics. *IEEE TITS*, 22(9).
5. Wu, G. et al. (2022). Collaborative Truck-Drone Routing for Contactless Parcel Delivery. *IEEE TITS*, 23(12).
6. Schneider, M. et al. (2014). The Electric Vehicle-Routing Problem with Time Windows. *Transportation Science*, 48(4).
7. Montoya, A. et al. (2017). The EVRP with Nonlinear Charging Function. *TR-B*, 103.

---

*Last updated: 2026-08-01*
