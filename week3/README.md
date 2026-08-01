# Week 3: Truck-Drone EVRP-TW Experiment Framework

## Quick Start

```bash
# 1. Quick smoke test (1 repeat, 25 customers only, ~2-3 minutes)
python main.py --quick

# 2. Full experiment (10 repeats, all scales, ~2-4 hours)
python main.py

# 3. Generate report from results
python main.py --report
```

## File Structure

```
week3/
├── main.py                    # Entry point - run from VS Code or terminal
├── config.py                  # All parameters (dataset, vehicle, algorithm, experiment)
├── README.md                  # This file
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py         # Solomon instance loading, subset extraction, coordinate scaling
│   ├── problem_model.py       # TruckDroneSolution, cost/tardiness evaluation, Pareto, HV
│   └── report_generator.py    # Automated markdown report from results JSON
│
├── algorithms/
│   ├── __init__.py
│   ├── no_drone.py            # Pure truck GA baseline (VRPTW)
│   ├── paco.py                # P-ACO (DOI: 10.1109/TITS.2020.2992549)
│   ├── nsga2.py               # NSGA-II (Deb et al. 2002)
│   └── ivnd.py                # IVND (DOI: 10.1109/TITS.2022.3181282)
│
├── runner/
│   ├── __init__.py
│   └── experiment_runner.py   # Unified experiment execution engine
│
├── data/                      # Generated problem instances (JSON)
└── results/                   # Experiment results (JSON)
```

## Experiment Matrix

| Scale | TW Types | Vehicle Configs | Endurance | Methods | Repeats |
|-------|----------|-----------------|-----------|---------|---------|
| 25c | RC1, RC2 | 2T+2D | 4km, 6km | P-ACO, NSGA-II, IVND, No-Drone, POMO | 10 |
| 50c | RC1, RC2 | 4T+4D, 6T+6D | 4km, 6km | P-ACO, NSGA-II, IVND, No-Drone, POMO | 10 |
| 100c | RC1, RC2 | 4T+4D, 6T+6D, 8T+8D | 4km, 6km | P-ACO, NSGA-II, IVND, No-Drone, POMO | 10 |

**Total**: 4 instances x 3 scales x 6 configs x 2 endurance x 5 methods x 10 repeats = **~960 runs**

## POMO (Policy Optimization with Multiple Optima)

POMO is a deep reinforcement learning method based on Transformers (Kwon et al., NeurIPS 2020).
It uses REINFORCE with a shared baseline across multiple trajectories starting from different nodes.

### POMO Quick Start

```bash
# 1. Train the POMO model (requires PyTorch)
python -m algorithms.pomo.train --epochs 200 --problem-size 50

# 2. Run POMO experiments only (does NOT re-run other methods)
python run_pomo_experiments.py

# 3. Or train + run in one command
python run_pomo_experiments.py --train

# 4. Merge with existing results
python merge_results.py

# 5. Generate combined report
python main.py --report
```

### POMO Architecture
- **Encoder**: 6-layer Transformer encoder over customer nodes
  - Node features: (x, y, demand, ready_time, due_time, service_time)
  - Depot features: (x, y)
- **Decoder**: Attention-based pointer network
  - Query: last_node + load + time + battery
  - Multi-head attention (8 heads) → single-head → softmax
- **Training**: REINFORCE with POMO group baseline
  - POMO initialization: each trajectory starts from a different customer
  - Baseline = mean reward across all POMO trajectories
- **Inference**: Greedy decoding + 8-fold coordinate augmentation
- **Drone integration**: Greedy post-processing insertion into truck routes

## Key Parameters

| Parameter | Value |
|-----------|-------|
| Depot | (8.0, 8.0) |
| Coordinate scale | [0,100] → [0,16] km |
| Truck speed | 35 km/h |
| Drone speed | 50 km/h |
| Truck capacity | 200.0 |
| Drone capacity | 40.0 |
| Truck fixed cost | 100.0 |
| Drone fixed cost | 0.0 |

## Evaluation Metrics

- **Cost**: Fixed costs + distance-based costs
- **Tardiness**: Time window violation penalty
- **Hypervolume (HV)**: Multi-objective Pareto front quality
- **Drone Utilization**: % solutions with drones + avg missions/solution
- **Feasibility**: % runs producing feasible solutions
- **Runtime**: Wall-clock time per run

## Hardware and Environment

| Item | Specification |
|------|---------------|
| **Model** | MacBook Air (Mac16,13) |
| **Chip** | Apple M4 |
| **Cores** | 10 (4 performance + 6 efficiency) |
| **Memory** | 16 GB |
| **OS** | macOS 15.7.7 (Sequoia) |
| **Architecture** | arm64 (Apple Silicon) |
| **Python** | 3.14.0 |
| **Key dependencies** | NumPy (optional), Matplotlib (optional) |

## References

- P-ACO: Das et al., "Synchronized Truck and Drone Routing", IEEE TITS, 2020 [DOI: 10.1109/TITS.2020.2992549](https://doi.org/10.1109/TITS.2020.2992549)
- NSGA-II: Deb et al., "A Fast and Elitist Multiobjective Genetic Algorithm", IEEE TEC, 2002
- IVND: Wu et al., "Collaborative Truck-Drone Routing for Contactless Parcel Delivery", IEEE TITS, 2022 [DOI: 10.1109/TITS.2022.3181282](https://doi.org/10.1109/TITS.2022.3181282)
- Nonlinear Charging: Montoya et al., "The Electric Vehicle Routing Problem with Nonlinear Charging Function", TRB, 2017
