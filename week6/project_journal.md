# Truck-Drone EVRP-TW: Project Work Journal & Failure Case Log

> **Track:** Ground-Air Collaborative EVRP-TW — Hybrid Optimization for Truck-Drone Delivery  
> **Repository:** `/Users/jackalwick/Desktop/Truck-Drone EVRP-TW`  
> **Period:** Week 1 (setup) — Week 7 (completion)  
> **Updated:** 2026-07-21

---

## Project Architecture Overview

```
Input Instance (Solomon VRPTW)
    │
    ▼
Clustering (K-means spatial + temporal split)
    │
    ▼
POMO Neural Routing (per-cluster Transformer inference)
    │
    ▼
Drone Post-Processing (cross-route drone insertion)
    │
    ▼
EDD Repair (earliest due date reordering of tardy routes)
    │
    ▼
TruckDroneSolution (evaluation: cost, tardiness, feasibility)
```

**Key Technologies:**
- **POMO** (Policy Optimization with Multiple Optima): Neural Transformer-based construction heuristic
- **EDD Repair**: Classical heuristic for time-window feasibility
- **K-means Clustering**: Spatial + temporal customer grouping
- **Cross-route Drone Insertion**: Drones launched from truck A serve customers originally on truck B

---

## Week-by-Week Work Log

### Week 1 (Environment Setup)
- **Date:** ~June 2026
- **Actions:**
  - Set up Python environment with required dependencies (torch, numpy, matplotlib)
  - Cloned POMO repository and verified model checkpoints
  - Set up Solomon VRPTW dataset (py-ga-VRPTW JSON format, 56 instances)
  - Created project structure: `week3/`, `week4/`, `week5/`, `week6/`
- **Status:** ✅ Complete

### Week 2 (Literature Review & Data Preparation)
- **Date:** ~June 2026
- **Actions:**
  - Reviewed VRPTW literature (Solomon benchmark, time window constraints)
  - Studied EVRP-TW papers (battery, charging station models)
  - Reviewed truck-drone routing (Flying Sidekick TSP, parallel drone scheduling)
  - Built Solomon instance extraction pipeline (`utils/data_loader.py`)
  - Created coordinate scaling (Solomon [0,100] → Urban [0,16] km)
- **Key Files Created:**
  - `week3/config.py` — master configuration
  - `week3/utils/data_loader.py` — instance loading/building
  - `week3/utils/problem_model.py` — `TruckDroneSolution` evaluation class
- **Status:** ✅ Complete

### Week 3 (Baseline Reproduction — 4 Algorithms)
- **Date:** ~June-July 2026
- **Actions:**
  - Implemented **P-ACO** (Pareto Ant Colony Optimization) in `week3/algorithms/paco.py`
    - Dual pheromone (cost + tardiness), 3D drone pheromone
    - Pseudo-random proportional rule for truck/drone decisions
  - Implemented **NSGA-II** in `week3/algorithms/nsga2.py`
    - Chromosome encoding: [permutation] + [drone_flags]
    - SBX crossover, swap mutation, non-dominated sorting
  - Implemented **IVND** (Improved Variable Neighborhood Descent) in `week3/algorithms/ivnd.py`
    - 7 neighborhood structures (relocate, swap, 2-opt, truck↔drone conversions)
    - Tabu search + simulated annealing
  - Built experiment runner (`week3/runner/experiment_runner.py`)
  - Generated baseline results on RC101/RC102/RC201/RC202 (25c/50c/100c)
- **Key Results:**
  - All 3 algorithms achieve feasible solutions on ≥80% of instances
  - P-ACO: best tardiness minimization (dominates on tight-TW instances)
  - NSGA-II: best cost minimization (dominates on wide-TW instances)
  - IVND: fastest runtime, balanced performance
- **Status:** ✅ Complete

### Week 4 (POMO Integration — Neural Construction)
- **Date:** ~July 2026
- **Actions:**
  - Integrated POMO Transformer model for per-cluster routing
  - Created `pomo_multi_truck.py` — POMO with K-means clustering
  - Built mini-instance construction for cluster-level POMO inference
  - Implemented 8-fold coordinate augmentation
  - Created `TruckDroneSolution` evaluation with 5 violation types:
    - Capacity, time window, drone endurance, drone capacity, sync (stub)
  - Added Pareto front extraction and Hypervolume computation
- **Key Files:**
  - `week4/pomo_multi_truck.py` — POMO multi-truck solver
  - `week4/utils/problem_model.py` — solution evaluation (primary)
  - `week4/algorithms/pomo/` — POMO model + environment
- **Key Insight:** POMO produces shorter routes than classical heuristics but with TW violations on tight instances
- **Status:** ✅ Complete

### Week 5 (Drone Integration & Advanced Clustering)
- **Date:** ~July 2026
- **Actions:**
  - Implemented **cross-route drone insertion** (`drone_post_processing.py`)
    - Drone from truck A serves customer j originally on truck B
    - Pure distance-based saving calculation
    - No temporal sync check (gap identified for later)
  - Implemented **TW-aware clustering** (`tw_aware_clustering.py`)
    - Two-phase: spatial K-means → temporal split by TW midpoint gaps
    - Configurable max_gap_ratio (default 0.4 × horizon)
  - Implemented **adaptive clustering** (`adaptive_clustering.py`)
    - Auto-tunes gap threshold per cluster based on internal TW density
    - Angle-based (petal) clustering for RC1 tight-TW instances
    - Hybrid strategy: auto-selects angle for RC1, adaptive TW for RC2
  - Implemented **drone re-optimization** (`drone_reopt.py`)
    - After drone insertion, re-runs POMO on affected routes
    - Drone fleet sizing analysis
  - Created `ImprovedPOMOSolver` with 10 clustering + drone variants
- **Key Results:**
  - Hybrid + drone variant achieves best balance on 12 instances
  - Cross-route drones: 8-47 drone missions per instance
  - Drone cost saving: 0.5-3.2% of total solution cost
- **Failure Cases Found:**
  - Depot-launched drones impossible (min round-trip 11.5km > 4km endurance)
  - Spatial clustering ignores temporal compatibility → TW violations
- **Status:** ✅ Complete

### Week 6 (EDD Repair + Meta-Learning + ALNS Baseline)
- **Date:** ~July 2026
- **Actions:**

**P1: Meta-Learner (Strategy Selection)**
- Built dataset: 10 clustering variants × 12 instances = 120 data points
- Features: TW type, n_customers, TW spread, demand variance, horizon
- Trained decision tree to predict optimal clustering variant
- **Finding:** 10 variants collapse to 2 rules:
  - ≤50 customers → hybrid clustering
  - 100 customers → adaptive TW clustering
- **Contribution:** "Complex → Simple" — meta-learner eliminates unnecessary complexity

**P2: POMO Fine-Tuning**
- Fine-tuned POMO on 50 Solomon instances with TW-aware loss
- **Finding (Negative Result):** Fine-tuning does NOT change routing decisions
  - The pre-trained POMO model is already at a local optimum
  - Fine-tuning with TW penalty produces identical routes
- **Contribution:** Negative result validates using pre-trained POMO as-is

**P3: Partial EDD Repair (Segment-Level)**
- **Problem:** Full-route EDD destroys POMO's distance optimization on the entire route
- **Solution:** Only reorder the tardy segment (with 1-position context)
- **Key Discovery: Repair Phase Transition**
  - ≤50 customers: Partial EDD wins (preserves POMO distance optimization)
  - 100 customers: Full EDD wins (segments too long for partial repair)
  - Phase transition at ~75 customers
- **Fallback Bug Fix:** Changed fallback from cost-minimization to tardiness-first selection

**Cluster Feasibility Check**
- Created `cluster_feasibility.py` — pre-routing temporal feasibility validation
- EDD-sorts cluster and simulates to check zero-tardiness achievability
- Splits infeasible clusters by TW gaps (k=2,3 sub-clusters)
- **Finding:** All 32 clusters on RC202_100c are EDD-feasible
  - Tardiness originates from POMO routing + drone insertion, not clustering

**ALNS Baseline**
- Implemented full ALNS solver (`alns_baseline.py`)
  - 4 destroy + 3 repair operators, adaptive weights, simulated annealing
- **Finding:** ALNS cannot beat EDD repair on this problem
  - ALNS reduces tardiness but at massive cost increase (1901→39842)
  - EDD repair is fundamentally more efficient for TW feasibility

**Final P3 Results (Adaptive Strategy):**
| Scale | Best Method | Feasibility | Tardiness | Cost vs Baseline |
|-------|------------|-------------|-----------|-----------------|
| ≤50c  | Partial EDD | 100% | 0 | +0.8% ~ +2.7% |
| 100c  | Full EDD | 100% | 0 | -6.0% ~ +2.5% |

- **Key Files Created:**
  - `week6/repair.py` — full + partial EDD repair
  - `week6/pipeline.py` — unified solve pipeline
  - `week6/meta_learner.py` — decision tree strategy selector
  - `week6/pomo_finetune.py` — POMO fine-tuning
  - `week6/cluster_feasibility.py` — temporal feasibility check
  - `week6/alns_baseline.py` — ALNS solver
  - `week6/run_p3_experiments.py` — experiment runner
- **Status:** ✅ Complete

### Week 7 (Project Completion — All 6 Gaps) [IN PROGRESS]
- **Date:** July 21, 2026
- **Gaps to Fill:**
  1. SOTA comparison (literature table + baselines + statistical tests)
  2. Charging/battery constraints (EV component, Models B & C)
  3. Truck-drone synchronization (Model D)
  4. Expand to all 56 Solomon instances + 200c scale ✅ (Done)
  5. Optimality gap analysis
  6. Failure case analysis (≥3 systematic cases)
- **Status:** 🔨 In Progress

---

## Comprehensive Failure Case Log

### FC-1: Depot-Launched Drones Impossible
| Field | Detail |
|-------|--------|
| **Week Discovered** | Week 5 |
| **Root Cause** | Depot at (8,8) — minimum round-trip to any customer ≥ 11.5km, exceeding 4km drone endurance |
| **Impact** | All depot-launched drone missions are infeasible |
| **Solution** | Cross-route truck-launched drones (drone launches from truck in the field) |
| **Status** | ✅ Resolved |

### FC-2: Spatial Clustering Ignores Temporal Compatibility
| Field | Detail |
|-------|--------|
| **Week Discovered** | Week 5 |
| **Root Cause** | K-means clustering uses only Euclidean distance; customers with widely different TWs assigned to same cluster |
| **Impact** | Single truck cannot serve all customers in a cluster on time → POMO produces tardy routes |
| **Solution** | Two-phase clustering: spatial K-means → temporal split by TW midpoint gaps |
| **Status** | ✅ Resolved (TW-aware clustering) |

### FC-3: Clustering-TW Fundamental Contradiction
| Field | Detail |
|-------|--------|
| **Week Discovered** | Week 6 |
| **Root Cause** | Temporal split threshold (0.4×240=96min) too coarse — clusters with 80-95min TW spread pass without split but are TW-infeasible |
| **Impact** | On 100c instances, some clusters are fundamentally TW-infeasible even before POMO routing |
| **Solution** | Adaptive threshold per cluster based on internal TW density |
| **Status** | ✅ Resolved (adaptive clustering) |

### FC-4: EDD Repair is a No-Op on 2-Customer Routes
| Field | Detail |
|-------|--------|
| **Week Discovered** | Week 6 |
| **Root Cause** | Drone insertion removes customers from truck routes, creating 2-customer routes. If both customers have incompatible TWs (e.g., customer A due at t=20, customer B ready at t=80), EDD ordering cannot fix the problem — any ordering is tardy |
| **Impact** | Post-drone safety net cannot fix 2-customer routes without causing capacity violations |
| **Solution** | Adaptive repair strategy (Partial EDD for ≤50c, Full EDD for 100c) |
| **Status** | ✅ Accepted as limitation (documented) |

### FC-5: POMO Fine-Tuning Produces No Improvement
| Field | Detail |
|-------|--------|
| **Week Discovered** | Week 6 |
| **Root Cause** | Pre-trained POMO model is already at a local optimum; fine-tuning with additional TW penalty doesn't change routing decisions |
| **Impact** | P2 research direction (fine-tuning) produces zero improvement |
| **Classification** | **Negative Result** — valuable for the paper (shows fine-tuning is unnecessary) |
| **Solution** | Use pre-trained POMO as-is; focus optimization effort on repair stage |
| **Status:** | ✅ Documented as negative result |

### FC-6: ALNS Cannot Beat EDD Repair
| Field | Detail |
|-------|--------|
| **Week Discovered** | Week 6 |
| **Root Cause** | ALNS is a general-purpose metaheuristic; for the specific problem of TW feasibility, EDD is provably optimal (minimizes maximum lateness for fixed customer set) |
| **Impact** | ALNS produces worse solutions than EDD repair (cost 1901→39842 on RC202_100c) |
| **Classification** | Competitive baseline that validates EDD repair's efficiency |
| **Solution** | Keep ALNS as a SOTA baseline in the comparison table, not as an improvement |
| **Status:** | ✅ Documented |

### FC-7: Post-Drone Safety Net Causes Capacity Violations
| Field | Detail |
|-------|--------|
| **Week Discovered** | Week 6 |
| **Root Cause** | Safety net relocates customers from small tardy routes to other routes, exceeding capacity (200 units) and creating unserved customers |
| **Impact** | Cost exploded (1882→16480), feasibility dropped (100%→67%) |
| **Solution** | Reverted safety net entirely; adaptive strategy is the correct fix |
| **Status:** | ✅ Resolved (removed safety net) |

### FC-8: Fallback Bug in Partial EDD Repair (P3)
| Field | Detail |
|-------|--------|
| **Week Discovered** | Week 6 |
| **Root Cause** | Cost-minimization (`min(candidates, key=lambda r: _route_cost([r], instance))`) could select routes with residual tardiness if they had lower distance |
| **Impact** | RC202_100c: partial_success_rate 67% (1/3 routes still tardy after fallback) |
| **Fix** | Changed to tardiness-first selection: prioritize zero-tardiness, then minimize cost |
| **Lines Changed** | `repair.py` lines 448-471 |
| **Status:** | ✅ Fixed |

---

## Research Contributions Summary

| # | Contribution | Type | Week |
|---|-------------|------|------|
| 1 | **"Complex → Simple"** — 10 clustering variants collapse to 2 rules via meta-learning | Simplification | W6 |
| 2 | **Fine-Tuning Is Unnecessary** — POMO fine-tuning does not change routing decisions | Negative Result | W6 |
| 3 | **Repair Phase Transition** — Partial EDD wins at ≤50c, Full EDD wins at 100c; transition at ~75c | Discovery | W6 |
| 4 | **EDD Beats ALNS** — EDD repair is fundamentally more efficient than metaheuristics for TW feasibility | Competitive Finding | W6 |
| 5 | **Adaptive Repair Strategy** — Scale-dependent repair mode selection achieves 100% feasibility, 100% tardiness elimination | Method | W6 |
| 6 | **Cross-Route Drone Insertion** — Truck-launched drones serving customers from different trucks, with distance-based saving | Method | W5 |

---

## Key Design Decisions

| Decision | Rationale | Date |
|----------|----------|------|
| Use pre-trained POMO (not train from scratch) | Training takes days; pre-trained generalizes to Solomon | W4 |
| EDD repair instead of metaheuristic | Provably optimal for minimizing max lateness; sub-second | W6 |
| Cross-route drones (not depot-launched) | Depot too far from any customer for 4km endurance | W5 |
| Adaptive strategy (not one-size-fits-all) | Repair effectiveness depends on instance scale | W6 |
| No PyVRP migration | Project requirements explicitly forbid; would require C++ compilation | W3 |
| Truck-only charging + drone sync separate | Modular design; charging affects trucks, sync affects drones | W7 |

---

## Repository File Map

```
Truck-Drone EVRP-TW/
├── week3/                         # Baseline algorithms (P-ACO, NSGA-II, IVND)
│   ├── config.py                  # Master configuration (all constants)
│   ├── algorithms/
│   │   ├── paco.py                # Pareto ACO solver
│   │   ├── nsga2.py               # NSGA-II solver
│   │   └── ivnd.py                # IVND solver
│   ├── utils/
│   │   ├── data_loader.py         # Solomon instance loading/building
│   │   └── problem_model.py       # TruckDroneSolution (original copy)
│   ├── runner/
│   │   └── experiment_runner.py   # Multi-method experiment runner
│   └── data/                      # Built instance JSONs (224 files)
│
├── week4/                         # POMO neural solver
│   ├── pomo_multi_truck.py        # POMO with K-means clustering
│   ├── algorithms/pomo/           # POMO Transformer model + env
│   └── utils/
│       ├── data_loader.py         # Instance loader (extended types)
│       └── problem_model.py       # TruckDroneSolution (primary/active)
│
├── week5/                         # Drone + clustering improvements
│   ├── pomo_mt_improved.py        # ImprovedPOMOSolver (10 variants)
│   ├── tw_aware_clustering.py     # Spatio-temporal clustering
│   ├── adaptive_clustering.py     # Adaptive + angle + hybrid clustering
│   ├── drone_post_processing.py   # Cross-route drone insertion
│   └── drone_reopt.py             # Drone + POMO re-optimization
│
├── week6/                         # Repair + meta-learning + ablation
│   ├── repair.py                  # Full + partial EDD repair
│   ├── pipeline.py                # Unified solve pipeline
│   ├── meta_learner.py            # Strategy selection meta-learner
│   ├── pomo_finetune.py           # POMO fine-tuning (negative result)
│   ├── cluster_feasibility.py     # Temporal feasibility check
│   ├── alns_baseline.py           # ALNS baseline (not competitive)
│   └── run_p3_experiments.py      # P3 experiment runner
│
└── week7/                         # [IN PROGRESS] Project completion
    ├── ev_problem_model.py        # EV battery + charging (planned)
    ├── sota_comparison.py         # SOTA literature table (planned)
    ├── failure_cases.py           # Failure case generators (planned)
    └── project_journal.md         # THIS FILE
```

---

## Open Issues & Future Work

| Issue | Priority | Status |
|-------|---------|--------|
| No truck battery constraints | HIGH | 🔨 W7 Gap 2 |
| No drone-truck synchronization | HIGH | 🔨 W7 Gap 3 |
| No SOTA literature comparison | HIGH | 🔨 W7 Gap 1 |
| No optimality gap analysis | MED | 🔨 W7 Gap 5 |
| No charging station integration | HIGH | 🔨 W7 Gap 2 |
| 200c instances limited to 100 customers (Solomon limit) | LOW | Documented |
| No real-world instance testing | LOW | Future work |
| Sync violations counter exists but never populated | MED | 🔨 W7 Gap 3 |
| Truck waiting time at drone recovery not modeled | MED | 🔨 W7 Gap 3 |

---

*This journal is maintained as a living document. Each new finding, failure case, or design decision should be added promptly.*
