# Ground-Air Collaborative EVRP-TW: Hybrid Optimization for Truck-Drone Delivery

> **FURP 2026 — Undergraduate Research Project**  
> University of Nottingham Ningbo China · Faculty of Science and Engineering  
> **Student:** Zeyang Yu · **Track:** Truck-Drone Collaboration

---

## Project Info

| Field | Entry |
|---|---|
| **Student** | Zeyang Yu |
| **Project Title** | Ground-Air Collaborative EVRP-TW: Hybrid Optimization for Truck-Drone Delivery |
| **Track** | Truck-drone collaboration (mixed fleets, synchronization, applied logistics) |
| **Research Question** | How do charging strategies and truck-drone coordination affect the feasibility and efficiency of Ground-Air Collaborative EVRP-TW? |
| **One-line Summary** | A cluster-first route-second pipeline combining POMO neural routing with EDD repair, cross-route drone insertion, and EV charging constraints — achieving 100% time-window feasibility where all SOTA baselines fail. |

---

## Core Concepts

读报告之前，先搞清楚这几个到处出现的缩写是什么意思：

| 缩写 | 全称 | 一句话解释 |
|------|------|-----------|
| **W5** | Week 5 Pipeline | 第五周搭建好的完整求解流程：**聚类 → POMO神经路由 → 无人机插入**。这是本项目的方法基线，"W5 Baseline"就是不做任何修复的原始输出 |
| **EDD** | Earliest Due Date | **最早截止日期优先**。把一条路线上的客户按 `due_time` 从小到大重新排序。这是本项目最核心的修复算子——EDD排序对"最小化最大延迟"是可证明最优的，能在亚秒级消除时间窗违规 |
| **SOTA** | State Of The Art | **领域最先进方法**。"SOTA对比"就是拿我们的方法跟已发表论文里的主流算法（NSGA-II、P-ACO、IVND）在同等问题上比一比，看谁更好 |
| **POMO** | Policy Optimization with Multiple Optima | Kwon et al. (2020) 提出的**神经网络路径规划模型**。用 Transformer 一次性从多个起点生成路线，选最好的那个。我们用的是预训练好的模型，不需要自己训练 |

简单说：**W5 是流水线、EDD 是修复算法、SOTA 是对比对象、POMO 是神经网络大脑。**

---

## Architecture

```
Solomon VRPTW Instance (224 instances, 6 types, 25/50/100/200 customers)
    │
    ├── 1. Clustering ─────────────────────────────────
    │       Hybrid: RC1→Angle Petal, RC2→Adaptive TW
    │       + Temporal feasibility pre-check
    │
    ├── 2. POMO Neural Routing ────────────────────────
    │       Pre-trained Transformer, 8-fold augmentation
    │       ~0.05s per cluster inference
    │
    ├── 3. Drone Post-Processing ──────────────────────
    │       Cross-route truck-launched drone insertion
    │       + Launch-recovery synchronization (Model D)
    │
    ├── 4. EDD Repair ─────────────────────────────────
    │       Adaptive: ≤50c Partial EDD, 100c Full EDD
    │       Fallback: tardiness-first candidate selection
    │
    └── 5. EV Evaluation ──────────────────────────────
            Model A: Baseline (no battery/sync)
            Model B: + Linear charging
            Model C: + Non-linear charging
            Model D: + Synchronization
```

---

## Key Results

### SOTA Comparison — 16 Solomon RC Instances × 5 Methods

| Method | RC1 Cost | RC1 Tard | RC2 Cost | RC2 Tard | Feasibility | Runtime |
|--------|---------|---------|---------|---------|-------------|---------|
| **W5 + EDD (Ours)** | **423** | **0** ⭐ | 800 | **0** ⭐ | **100%** ⭐ | **0.1s** |
| NSGA-II (Deb 2002) | 368 | 247 | 368 | 1,048 | 91% | 0.6s |
| P-ACO (2020) | 322 | 396 | 340 | 254 | 47% | 4.2s |
| IVND (2022) | 322 | 871 | 322 | 2,600 | 0% | 0.0s |
| W5 Baseline (POMO) | 428 | 75 | 462 | 91 | 100% | 0.1s |

> **Our method is the ONLY one achieving 100% feasibility with 0 tardiness.**  
> Other methods are cheaper (15–85%) but produce infeasible solutions with massive TW violations.

### Four-Model Ablation (A/B/C/D)

| Model | Charging | Sync | Status |
|-------|----------|------|--------|
| **A** | ❌ No battery | ❌ No sync | ✅ Baseline |
| **B** | ✅ Linear (1.0 kWh/t) | ❌ No sync | ✅ Completed |
| **C** | ✅ Non-linear (SOC分段) | ❌ No sync | ✅ Completed |
| **D** | ✅ Non-linear | ✅ Launch-recovery | ✅ Completed |

### Key Findings

1. **Repair Phase Transition** — Partial EDD wins at ≤50c, Full EDD wins at 100c; transition at ~75 customers
2. **10→2 Rule** — Meta-learner collapses 10 clustering variants into 2: `tw_type == RC1 ? adaptive_tw_drone : tw_aware_drone`
3. **POMO Fine-tuning is Unnecessary** — Pre-trained model already at local optimum (negative result)
4. **EDD beats ALNS** — Domain-specific heuristic outperforms general-purpose metaheuristic for TW feasibility
5. **Battery capacity is the binding constraint** — 100 kWh sufficient for ≤50c (55–85 kWh), 100c+ requires en-route charging

---

## Repository Structure

```
Truck-Drone EVRP-TW/
│
├── week3/                         # Classical baselines + master config
│   ├── config.py                  # All parameters (56 Solomon instances, EV, drone)
│   ├── algorithms/
│   │   ├── paco.py                # Pareto Ant Colony Optimization
│   │   ├── nsga2.py               # NSGA-II multi-objective GA
│   │   └── ivnd.py                # Improved Variable Neighborhood Descent
│   ├── utils/
│   │   ├── data_loader.py         # Solomon instance building (224 instances)
│   │   └── problem_model.py       # TruckDroneSolution evaluation (original)
│   ├── runner/experiment_runner.py
│   ├── data/                      # 224 built instance JSONs
│   └── week3_report.md
│
├── week4/                         # POMO neural routing
│   ├── pomo_multi_truck.py        # POMO + K-means clustering
│   ├── algorithms/pomo/           # Transformer model + environment
│   ├── utils/
│   │   ├── data_loader.py         # Extended: 6 TW types
│   │   └── problem_model.py       # + Sync violation tracking (W6 update)
│   └── week4_report.md
│
├── week5/                         # Drone + advanced clustering
│   ├── pomo_mt_improved.py        # 10-variant unified solver
│   ├── tw_aware_clustering.py     # Spatial + temporal two-phase
│   ├── adaptive_clustering.py     # Adaptive + angle + hybrid strategies
│   ├── drone_post_processing.py   # Cross-route drone insertion
│   ├── drone_reopt.py             # POMO re-optimization + fleet sizing
│   ├── results/                   # Ablation + parameter sweep JSONs
│   ├── visualizations/            # 10 generated plots
│   └── week5_report.md
│
├── week6/                         # Pipeline integration + project completion
│   ├── repair.py                  # Full + partial EDD repair
│   ├── pipeline.py                # Unified solve pipeline
│   ├── meta_learner.py            # Strategy selection (KNN classifier)
│   ├── pomo_finetune.py           # POMO fine-tuning (negative result)
│   ├── cluster_feasibility.py     # Temporal feasibility pre-check
│   ├── alns_baseline.py           # ALNS competitive baseline
│   │
│   ├── ev_problem_model.py        # 🆕 EV battery + charging (Models B/C)
│   ├── sync_constraints.py        # 🆕 Drone sync (Model D)
│   ├── exact_solver.py            # 🆕 OR-Tools exact solver + gap analysis
│   ├── sota_comparison.py         # 🆕 Literature comparison table
│   ├── failure_cases.py           # 🆕 4 systematic failure cases
│   │
│   ├── run_sota_comparison.py     # SOTA experiment (16 instances × 5 methods)
│   ├── run_charging_study.py      # Charging study (Models A vs B vs C)
│   ├── run_sync_study.py          # Sync study (no-sync vs full-sync)
│   ├── run_p3_experiments.py      # Partial vs Full EDD ablation
│   │
│   ├── week6_integration_note.md  # ** Complete Week 6 Report **
│   ├── project_journal.md         # Full project log (Weeks 1–6 + failure cases)
│   ├── glossary.md                # 80+ terminology reference
│   ├── results/                   # Experiment JSON outputs
│   └── visualizations/            # 11 generated plots
│
├── py-ga-VRPTW/data/json/         # 56 Solomon source instances
├── docs/                          # Project requirements + outlines
├── VRP-EVRP-Project-Hub/          # Research hub templates
└── README.md                      # This file
```

---

## Quick Start

```bash
# 1. Build all instances (first run only)
cd week3
python -c "from utils.data_loader import build_all_instances; build_all_instances()"

# 2. Run the pipeline on a single instance
cd ../week6
python -c "
from pipeline import run_pipeline
from utils.data_loader import load_instance_from_disk
inst = load_instance_from_disk('RC201_50c')
result = run_pipeline(inst, n_trucks=4, variant='hybrid', use_repair=True, repair_mode='partial', n_runs=1)
sol = result['solutions'][0]
print(f'Cost: {sol.cost:.0f}, Tardiness: {sol.tardiness:.0f}, Feasible: {sol.feasible}')
"

# 3. SOTA comparison
python run_sota_comparison.py --quick    # 25c+50c

# 4. Charging study (Models A/B/C)
python run_charging_study.py --test      # Quick smoke test

# 5. Sync study (Model D)
python run_sync_study.py --test

# 6. Failure case analysis
python failure_cases.py

# 7. P3 ablation (Partial vs Full EDD)
python run_p3_experiments.py --quick --repeats 3
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [`week6/week6_integration_note.md`](week6/week6_integration_note.md) | **Week 6 Complete Report** — full results, analysis, visualizations |
| [`week6/project_journal.md`](week6/project_journal.md) | Full project log (Weeks 1–6), all failure cases |
| [`week6/glossary.md`](week6/glossary.md) | 80+ terminology reference (Chinese + English) |
| [`week5/week5_report.md`](week5/week5_report.md) | Week 5: Clustering + drone improvements |
| [`week4/week4_report.md`](week4/week4_report.md) | Week 4: POMO neural routing |
| [`week3/week3_report.md`](week3/week3_report.md) | Week 3: Classical baselines |
| [`docs/任务list.md`](docs/任务list.md) | FURP project requirements checklist |

---

## Deliverables Checklist

| FURP Requirement | Status | Evidence |
|-----------------|--------|----------|
| Model A (baseline, no charging/sync) | ✅ | `TruckDroneSolution` in `problem_model.py` |
| Model B (+ linear charging) | ✅ | `EVTruckDroneSolution(linear)` in `ev_problem_model.py` |
| Model C (+ non-linear charging) | ✅ | `EVTruckDroneSolution(nonlinear)` in `ev_problem_model.py` |
| Model D (+ synchronization) | ✅ | `sync_constraints.py` + `problem_model.py` sync tracking |
| Scale test (50/100/200) | ✅ | 224 instances, 4 scales |
| Charging study | ✅ | `run_charging_study.py` |
| Synchronization study | ✅ | `run_sync_study.py` |
| SOTA comparison (≥3 baselines) | ✅ | 5 methods × 16 instances |
| ≥3 failure cases | ✅ | 4 systematic cases with root cause |
| Optimality gap analysis | ✅ | `exact_solver.py` + OR-Tools |
| Weekly logs + meeting notes | ✅ | `project_journal.md` + `docs/` |
| Final poster | 🔜 | `FURP_Showcase.pdf` |

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Neural Routing | POMO (Transformer + REINFORCE), PyTorch 2.12 |
| Exact Solver | Google OR-Tools 9.15 (VRPTW, CP-SAT) |
| Classical Baselines | P-ACO, NSGA-II, IVND (custom implementations) |
| Clustering | K-means (numpy), TW-aware temporal split |
| Visualization | Matplotlib |
| Dataset | Solomon VRPTW (56 instances, 6 types) |
| Environment | Python 3, macOS |

---

## Key Design Decisions

| Decision | Rationale |
|----------|----------|
| EDD repair over metaheuristics | Provably optimal for minimizing max lateness; sub-second |
| Cross-route drones (not depot-launched) | Depot too far from any customer for 4km endurance |
| Adaptive repair strategy (not one-size-fits-all) | Repair effectiveness depends on instance scale |
| Charging + sync separated | Modular: charging affects trucks, sync affects drones |
| Pre-trained POMO (not trained from scratch) | Training takes days; pre-trained generalizes to Solomon |
| No PyVRP migration | Project requirements explicitly forbid; requires C++ build |

---

> **Core Narrative:** Feasibility-first optimization. Other methods find cheaper routes — but they're infeasible. We provide the only reliably feasible solution, at a reasonable cost premium.  
> **优化研究是关于权衡的。我们使权衡清晰可见。**
>
> *Last updated: 2026-07-21*
