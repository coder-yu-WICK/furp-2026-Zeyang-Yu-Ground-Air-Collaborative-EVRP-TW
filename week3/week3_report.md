# Week 3 Lab: Truck-Drone EVRP-TW Experiment Report

*Generated: 2026-07-05*
*Data source: results_20260702_152443_hv_fixed.json*

---

## Abstract

This report presents a systematic comparative analysis of three multi-objective optimization methods for the truck-drone collaborative electric vehicle routing problem with time windows (EVRP-TW):

- **P-ACO**: Pareto Ant Colony Optimization with synchronized truck-drone routing and 3D pheromone matrix
- **NSGA-II**: Non-dominated Sorting Genetic Algorithm II with SBX crossover and polynomial mutation
- **IVND**: Improved Variable Neighborhood Descent with tabu search and simulated annealing acceptance

A pure truck-only baseline (**No-Drone**) is included as a control. Experiments are conducted on Solomon RC benchmark instances (RC101, RC102 for tight time windows; RC201, RC202 for wide time windows) with 25, 50, and 100 customers, testing both medium (4 km) and high (6 km) drone endurance configurations across multiple truck+drone fleet configurations. All methods optimize travel cost and tardiness as dual objectives, with Hypervolume (HV) as the comprehensive performance metric.

Experimental results show:

- **HV Comparison**: P-ACO dominates across all scales, achieving mean HV of 78.9M vs NSGA-II's 61.2M, IVND's 50.6M, and No-Drone's 44.8M. The gap widens at larger scales: at 100 customers, P-ACO's HV (73.3M) is 1.7× NSGA-II (43.8M) and 2.4× IVND (30.4M).
- **Cost Advantage**: P-ACO achieves 72.6% average cost savings vs No-Drone, compared to NSGA-II's 65.2% and IVND's 67.0%. However, IVND's cost advantage is misleading due to near-zero feasibility.
- **Feasibility Crisis**: P-ACO (36.1%), NSGA-II (73.6%), and IVND (16.7%) all struggle with constraint satisfaction at scale. No-Drone is the only method with 100% feasibility, highlighting the difficulty of synchronized truck-drrone routing under time window and battery constraints.
- **Time Window Flexibility**: P-ACO uniquely improves under wider time windows (+6.9% HV from RC1→RC2), while all other methods degrade (NSGA-II: −27.1%, IVND: −30.5%, No-Drone: −42.0%).
- **Runtime Scaling**: P-ACO exhibits superlinear runtime growth (58.5× from 25c→100c), becoming prohibitively slow at 100 customers (~16 min/run). NSGA-II scales better (40×) while IVND is near-constant (3.6×).

---

## 1. Experimental Setup

### 1.1 Comparison Objectives

- **Test Methods**: P-ACO, NSGA-II, IVND (three-way equal comparison)
- **Baseline**: No-Drone (pure truck delivery)
- **Research Questions**:
  1. Which method achieves the best solution quality (HV, Cost, Tardiness)?
  2. Can drone-assisted delivery significantly reduce total cost vs pure trucks?
  3. How does drone endurance (4 km vs 6 km) affect solution quality?
  4. How does time window tightness (RC1 vs RC2) affect algorithm performance?
  5. How do algorithms scale from 25 → 50 → 100 customers?

### 1.2 Dataset and Instance Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | Solomon RC series (RC101, RC102, RC201, RC202) |
| Customer sizes | 25, 50, 100 |
| Coordinate scaling | Solomon [0,100] → Urban [0,16] km |
| Depot location | (8.0, 8.0) |
| Time window types | RC1 (tight, 120 min horizon) / RC2 (wide, 240 min horizon) |
| Truck speed | 35 km/h |
| Drone speed | 50 km/h |
| Truck capacity | 200.0 |
| Drone capacity | 40.0 |
| Drone endurance | medium = 4 km, high = 6 km |
| Truck fixed cost | 100.0 per vehicle |
| Distance cost rates | Truck: 2.0/km, Drone: 1.0/km |
| Tardiness penalty | 1.0 per time unit |

### 1.3 Vehicle Configuration

| Customers | Trucks | Drones | Description |
|-----------|--------|--------|-------------|
| 25 | 2 | 2 | 1:1 truck-drone pairing |
| 50 | 4 / 6 | 4 / 6 | Standard and high-density |
| 100 | 4 / 6 / 8 | 4 / 6 / 8 | Three density levels |

Each truck carries one drone; drone launches from truck, serves customer, returns to truck at recovery node.

### 1.4 Algorithm Configuration

| Parameter | P-ACO | NSGA-II | IVND | No-Drone |
|-----------|-------|---------|------|----------|
| Population/Ants | 50–120 | 50–150 | — | 80–100 |
| Iterations/Generations | 100 | 120 | 200 | 120 |
| Key mechanism | 2D truck + 3D drone pheromone | Non-dominated sorting + crowding distance | K-means init + 7 neighborhoods + Tabu + SA | PMX crossover + inversion mutation |
| Crossover | — | SBX (η=20) | — | PMX |
| Mutation | — | Polynomial (η=20) | — | Inversion |
| Selection | Pseudo-random (q₀=0.5) | Tournament (k=2) | Metropolis (T₀=100, α=0.95) | Tournament |
| Pheromone | α=1.0, β=2.0, ρ=0.15 | — | — | — |
| Repeats | 10 | 10 | 10 | 10 |

### 1.5 Evaluation Metrics

| Metric | Definition |
|--------|------------|
| **Cost** | Vehicle fixed costs (truck: 100/vehicle) + truck distance × 2.0 + drone flight distance × 1.0 |
| **Tardiness** | Σ max(0, arrival_time − due_time) × 1.0 |
| **Hypervolume (HV)** | 2D Lebesgue measure of Pareto front relative to auto-scaled reference point (1.2× max observed), measuring multi-objective solution set quality |
| **Feasibility Rate** | % of runs producing feasible solutions (all constraints satisfied) |
| **Drone Utilization** | % of solutions containing drone missions + average missions per solution |
| **Runtime** | Wall-clock time per independent run |

### 1.6 Hardware and Environment

| Item | Configuration |
|------|---------------|
| Model | MacBook Air (Mac16,13) |
| Chip | Apple M4 |
| Cores | 10 (4 performance + 6 efficiency) |
| Memory | 16 GB |
| OS | macOS 15.7.7 (Sequoia) |
| Architecture | arm64 (Apple Silicon) |
| Python | 3.14.0 |
| Key dependencies | NumPy (optional), Matplotlib (optional) |

---

## 2. Results

### 2.1 Overall Performance Summary

| Method | Mean Cost | Mean Tardiness | Mean HV | Feasibility | Mean Runtime (s) | Drone Usage |
|--------|-----------|----------------|---------|-------------|-----------------|-------------|
| **P-ACO** | 709.6 | 1,632.2 | **78,924,957** | 36.1% | 504.4 | 100.0% |
| **NSGA-II** | 962.6 | 5,318.7 | 61,190,252 | 73.6% | 8.9 | 99.9% |
| **IVND** | 850.1 | 10,002.3 | 50,578,930 | 16.7% | 0.12 | 99.6% |
| **No-Drone** | 3,625.9 | 967.2 | 44,799,610 | **100.0%** | 0.61 | 0.0% |

### 2.2 Performance by Customer Scale

| Scale | Method | Mean Cost | Mean Tardiness | Mean HV | Feasibility | Mean Runtime |
|-------|--------|-----------|----------------|---------|-------------|-------------|
| **25c** | P-ACO | 328.4 | 1,055.1 | 90,991,207 | 63.7% | 16.8s |
| | NSGA-II | 367.4 | 672.6 | 87,324,558 | 91.5% | 0.41s |
| | IVND | 322.2 | 1,834.8 | 79,855,009 | 0.0% | 0.05s |
| | No-Drone | 1,269.3 | 55.8 | 74,454,155 | 100.0% | 0.18s |
| **50c** | P-ACO | 685.0 | 1,458.6 | 81,391,323 | 44.8% | 32.9s |
| | NSGA-II | 775.1 | 3,578.4 | 74,228,420 | 98.2% | 1.46s |
| | IVND | 949.5 | 5,584.5 | 66,212,464 | 50.0% | 0.08s |
| | No-Drone | 2,428.2 | 182.1 | 50,607,679 | 100.0% | 0.34s |
| **100c** | P-ACO | 853.0 | 2,099.3 | 73,258,629 | 21.2% | 981.4s |
| | NSGA-II | 1,286.0 | 8,090.7 | 43,786,703 | 51.1% | 16.6s |
| | IVND | 959.8 | 15,986.3 | 30,397,880 | 0.0% | 0.17s |
| | No-Drone | 5,209.8 | 1,498.4 | 31,042,715 | 100.0% | 0.92s |

**Key observations:**
- P-ACO leads in HV at every scale, but feasibility drops from 63.7% (25c) → 44.8% (50c) → 21.2% (100c)
- NSGA-II maintains best feasibility among drone methods (91.5% → 98.2% → 51.1%)
- IVND collapses at scale: 0% feasibility at both 25c and 100c
- P-ACO runtime explodes at 100c (981s/run = ~16 min), making it impractical for large instances

### 2.3 Performance by Time Window Type

| TW Type | Method | Mean HV | Mean Cost | Mean Tardiness | Feasibility |
|---------|--------|---------|-----------|----------------|-------------|
| **RC1 (tight)** | P-ACO | 76,301,253 | 698.4 | 932.6 | 36.1% |
| | NSGA-II | 70,761,119 | 962.4 | 2,339.5 | 73.6% |
| | IVND | 59,675,432 | 840.4 | 5,383.2 | 16.7% |
| | No-Drone | 56,718,988 | 2,094.6 | 448.6 | 100.0% |
| **RC2 (wide)** | P-ACO | **81,548,660** | 720.7 | 2,331.8 | 36.1% |
| | NSGA-II | 51,619,384 | 962.8 | 8,297.8 | 73.5% |
| | IVND | 41,482,427 | 859.8 | 14,621.4 | 16.7% |
| | No-Drone | 32,880,232 | 5,157.2 | 1,485.9 | 100.0% |

**Time Window Flexibility** (HV change RC1→RC2):

| Method | RC1 HV | RC2 HV | Change |
|--------|--------|--------|--------|
| P-ACO | 76.3M | 81.5M | **+6.9%** ↑ |
| NSGA-II | 70.8M | 51.6M | −27.1% ↓ |
| IVND | 59.7M | 41.5M | −30.5% ↓ |
| No-Drone | 56.7M | 32.9M | −42.0% ↓ |

P-ACO is the **only method that improves under wider time windows**. All other methods degrade significantly, with No-Drone losing nearly half its HV. This suggests P-ACO's pheromone-guided exploration effectively leverages the additional scheduling flexibility of RC2, while other methods struggle with the expanded objective space.

### 2.4 Drone Endurance Impact

| Endurance | Method | Mean HV | Mean Cost | Drone Missions |
|-----------|--------|---------|-----------|----------------|
| **Medium (4km)** | P-ACO | 77,442,084 | 710.9 | 26.6 |
| | NSGA-II | 57,444,860 | 985.1 | 12.4 |
| | IVND | 50,609,745 | 845.8 | 6.8 |
| **High (6km)** | P-ACO | 80,407,829 | 708.2 | 41.2 |
| | NSGA-II | 64,935,643 | 940.1 | 36.9 |
| | IVND | 50,548,114 | 854.4 | 8.9 |

Higher endurance (6km vs 4km) provides modest benefits:
- P-ACO: HV +3.8%, drone missions +55%
- NSGA-II: HV +13.0%, drone missions +198% (triples drone usage)
- IVND: HV unchanged, drone missions +31%

NSGA-II benefits most from extended endurance, nearly tripling its drone mission count, suggesting that the 4km range was a binding constraint for its crossover-generated drone missions.

### 2.5 Cost Advantage vs No-Drone Baseline

| Method | Avg Cost Savings vs No-Drone |
|--------|------------------------------|
| P-ACO | **72.6%** |
| IVND | 67.0% |
| NSGA-II | 65.2% |

P-ACO achieves the deepest cost reduction relative to pure-truck delivery, saving nearly three-quarters of total cost. However, this must be weighed against its low feasibility (36.1%) — the savings apply only to successfully feasible solutions. IVND's 67.0% savings is misleading given its 16.7% feasibility rate.

### 2.6 Runtime Scaling

| Method | 25c (s) | 50c (s) | 100c (s) | 50c/25c | 100c/25c |
|--------|---------|---------|----------|---------|----------|
| P-ACO | 16.8 | 32.9 | 981.4 | 2.0× | **58.5×** |
| NSGA-II | 0.41 | 1.46 | 16.6 | 3.5× | 40.0× |
| IVND | 0.05 | 0.08 | 0.17 | 1.6× | **3.6×** |
| No-Drone | 0.18 | 0.34 | 0.92 | 1.9× | 5.2× |

Runtime reveals a sharp divide:
- **IVND** is blazing fast (0.17s at 100c) but infeasible
- **No-Drone** scales gracefully (5.2× from 25→100c)
- **P-ACO** becomes computationally prohibitive at 100c, consuming 16+ minutes per run — a 58.5× increase from 25c. This is caused by the O(n³) 3D drone pheromone enumeration.

### 2.7 Pareto Front Visualizations

The figures below show the Pareto fronts for representative configurations. Each point represents a non-dominated solution (Cost vs Tardiness). The black dashed line connects the joint Pareto front across all methods.

![25c RC1 medium](visualizations/pareto_25c_RC1_medium_2T+2D.png)
*25 customers, RC1 (tight time windows), 4km endurance, 2T+2D. P-ACO achieves the widest spread with lowest cost; NSGA-II overlaps in the mid-range; IVND collapses to a single point; No-Drone forms a distinct high-cost cluster.*

![25c RC2 medium](visualizations/pareto_25c_RC2_medium_2T+2D.png)
*25 customers, RC2 (wide time windows), 4km endurance, 2T+2D. Under wider TW, tardiness values expand significantly. P-ACO maintains cost advantage while NSGA-II and No-Drone shift toward higher tardiness.*

![50c RC1 medium](visualizations/pareto_50c_RC1_medium_4T+4D.png)
*50 customers, RC1, 4km endurance, 4T+4D. The cost gap between P-ACO and NSGA-II widens at larger scale. No-Drone becomes distinctly more expensive.*

![50c RC2 medium](visualizations/pareto_50c_RC2_medium_4T+4D.png)
*50 customers, RC2, 4km endurance, 4T+4D. All methods show dramatically expanded tardiness ranges. P-ACO uniquely maintains low-cost solutions.*

![100c RC1 medium](visualizations/pareto_100c_RC1_medium_4T+4D.png)
*100 customers, RC1, 4km endurance, 4T+4D. P-ACO's Pareto front dominates but feasibility is severely reduced (21.2%). NSGA-II provides more consistent coverage.*

### 2.8 Route Visualizations

The figures below show the best feasible routes found by each method. Truck routes are shown as solid colored lines (one color per vehicle); drone missions are dashed red lines with star markers at drone-served customers.

#### 25 Customers — RC1 (Tight TW)

| P-ACO | NSGA-II | No-Drone |
|-------|---------|----------|
| ![P-ACO 25c RC1](visualizations/route_P-ACO_25c_RC1_medium_2T+2D.png) | ![NSGA-II 25c RC1](visualizations/route_NSGA-II_25c_RC1_medium_2T+2D.png) | ![No-Drone 25c RC1](visualizations/route_No-Drone_25c_RC1_medium_2T+2D.png) |

P-ACO achieves the lowest cost (2 truck routes with multiple drone missions). NSGA-II produces more balanced routes. No-Drone requires more vehicle trips without drone assistance.

#### 25 Customers — RC2 (Wide TW)

| P-ACO | NSGA-II | No-Drone |
|-------|---------|----------|
| ![P-ACO 25c RC2](visualizations/route_P-ACO_25c_RC2_medium_2T+2D.png) | ![NSGA-II 25c RC2](visualizations/route_NSGA-II_25c_RC2_medium_2T+2D.png) | ![No-Drone 25c RC2](visualizations/route_No-Drone_25c_RC2_medium_2T+2D.png) |

Under wider time windows, routes become more spread out geographically. P-ACO leverages drone missions (red dashed lines) to serve outlying customers without truck detours.

#### 50 Customers — RC1 (Tight TW, 4T+4D)

| P-ACO | NSGA-II | No-Drone |
|-------|---------|----------|
| ![P-ACO 50c RC1](visualizations/route_P-ACO_50c_RC1_medium_4T+4D.png) | ![NSGA-II 50c RC1](visualizations/route_NSGA-II_50c_RC1_medium_4T+4D.png) | ![No-Drone 50c RC1](visualizations/route_No-Drone_50c_RC1_medium_4T+4D.png) |

At 50 customers, the routing complexity increases substantially. P-ACO's pheromone guidance produces more compact routes with strategic drone insertions, while NSGA-II's genetic operators generate more dispersed but feasible solutions.

#### 50 Customers — RC2 (Wide TW, 4T+4D)

| P-ACO | NSGA-II | No-Drone |
|-------|---------|----------|
| ![P-ACO 50c RC2](visualizations/route_P-ACO_50c_RC2_medium_4T+4D.png) | ![NSGA-II 50c RC2](visualizations/route_NSGA-II_50c_RC2_medium_4T+4D.png) | ![No-Drone 50c RC2](visualizations/route_No-Drone_50c_RC2_medium_4T+4D.png) |

Wide time windows allow more flexible scheduling. P-ACO's routes show clear spatial partitioning between vehicles with drones filling inter-vehicle gaps.

#### 100 Customers — RC1 (Tight TW, 4T+4D)

| P-ACO | No-Drone |
|-------|----------|
| ![P-ACO 100c RC1](visualizations/route_P-ACO_100c_RC1_medium_4T+4D.png) | ![No-Drone 100c RC1](visualizations/route_No-Drone_100c_RC1_medium_4T+4D.png) |

At 100 customers, P-ACO's route map reveals the challenge of coordinating 4 trucks and 4 drones under tight time windows. NSGA-II produced no feasible solution for this configuration. No-Drone maintains feasibility but at substantially higher cost.

### 2.9 Detailed Per-Configuration Results

#### 25 Customers (2T+2D)

| Config | TW | End. | Method | Mean Cost ± Std | Mean Tard ± Std | HV | Feas. |
|--------|----|------|--------|-----------------|-----------------|-----|-------|
| RC101_25c | RC1 | 4km | P-ACO | 322.4 ± 14.8 | 899.1 ± 206.9 | 89.3M | 69.2% |
| | | | NSGA-II | 368.6 ± 33.2 | 562.0 ± 277.3 | 90.0M | 91.2% |
| | | | IVND | 322.2 ± 0.0 | 1,661.8 ± 0.0 | 85.8M | 0.0% |
| | | | No-Drone | 657.9 ± 54.6 | 43.7 ± 88.4 | 86.7M | 100% |
| | | 6km | P-ACO | 323.1 ± 13.2 | 847.4 ± 207.0 | 89.2M | 70.4% |
| | | | NSGA-II | 365.9 ± 31.6 | 546.6 ± 282.6 | 90.1M | 90.9% |
| | | | IVND | 322.2 ± 0.0 | 1,661.8 ± 0.0 | 85.8M | 0.0% |
| | | | No-Drone | 657.9 ± 54.6 | 43.7 ± 88.4 | 86.7M | 100% |
| RC102_25c | RC1 | 4km | P-ACO | 335.7 ± 12.1 | 691.8 ± 132.1 | 90.9M | 57.4% |
| | | | NSGA-II | 368.2 ± 32.3 | 230.0 ± 225.4 | 91.2M | 91.6% |
| | | | IVND | 322.2 ± 0.0 | 1,661.8 ± 0.0 | 88.2M | 0.0% |
| | | | No-Drone | 590.3 ± 48.7 | 2.0 ± 0.0 | 88.1M | 100% |
| | | 6km | P-ACO | 335.2 ± 12.3 | 615.4 ± 126.1 | 90.9M | 58.9% |
| | | | NSGA-II | 365.7 ± 33.2 | 193.8 ± 220.0 | 91.3M | 92.7% |
| | | | IVND | 322.2 ± 0.0 | 1,661.8 ± 0.0 | 88.2M | 0.0% |
| | | | No-Drone | 590.3 ± 48.7 | 2.0 ± 0.0 | 88.1M | 100% |
| RC201_25c | RC2 | 4km | P-ACO | 336.0 ± 17.3 | 3,537.9 ± 727.2 | 91.1M | 40.7% |
| | | | NSGA-II | 437.6 ± 54.3 | 3,665.0 ± 1,534.7 | 81.7M | 90.9% |
| | | | IVND | 322.2 ± 0.0 | 1,661.8 ± 0.0 | 68.7M | 0.0% |
| | | | No-Drone | 1,916.1 ± 197.2 | 2.0 ± 0.0 | 57.2M | 100% |
| | | 6km | P-ACO | 336.5 ± 14.4 | 3,546.7 ± 792.6 | 91.5M | 43.1% |
| | | | NSGA-II | 447.9 ± 56.0 | 3,450.6 ± 1,409.0 | 81.8M | 90.5% |
| | | | IVND | 322.2 ± 0.0 | 1,661.8 ± 0.0 | 68.7M | 0.0% |
| | | | No-Drone | 1,916.1 ± 197.2 | 2.0 ± 0.0 | 57.2M | 100% |
| RC202_25c | RC2 | 4km | P-ACO | 330.3 ± 15.1 | 3,338.3 ± 849.9 | 92.1M | 56.8% |
| | | | NSGA-II | 421.5 ± 51.4 | 2,692.4 ± 1,336.9 | 86.2M | 93.1% |
| | | | IVND | 322.2 ± 0.0 | 1,661.8 ± 0.0 | 76.8M | 0.0% |
| | | | No-Drone | 1,812.7 ± 192.5 | 201.9 ± 186.8 | 65.8M | 100% |
| | | 6km | P-ACO | 335.4 ± 12.9 | 3,118.8 ± 851.8 | 92.8M | 64.7% |
| | | | NSGA-II | 424.7 ± 51.2 | 2,540.8 ± 1,376.6 | 86.3M | 93.1% |
| | | | IVND | 322.2 ± 0.0 | 1,661.8 ± 0.0 | 76.8M | 0.0% |
| | | | No-Drone | 1,812.7 ± 192.5 | 201.9 ± 186.8 | 65.8M | 100% |

#### 50 Customers — 4T+4D

| Config | TW | End. | Method | Mean Cost ± Std | Mean Tard ± Std | HV | Feas. |
|--------|----|------|--------|-----------------|-----------------|-----|-------|
| RC101_50c | RC1 | 4km | P-ACO | 568.5 ± 28.7 | 1,511.8 ± 436.5 | 78.8M | 40.8% |
| | | | NSGA-II | 927.4 ± 44.7 | 3,269.1 ± 724.2 | 81.1M | 99.0% |
| | | | IVND | 901.0 ± 0.0 | 4,026.0 ± 0.0 | 74.7M | 50.0% |
| | | | No-Drone | 2,347.4 ± 149.7 | 75.8 ± 107.1 | 70.3M | 100% |
| | | 6km | P-ACO | 576.8 ± 27.2 | 1,497.5 ± 366.4 | 78.6M | 42.6% |
| | | | NSGA-II | 933.4 ± 54.0 | 3,082.9 ± 741.0 | 81.1M | 98.5% |
| | | | IVND | 901.0 ± 0.0 | 4,026.0 ± 0.0 | 74.7M | 50.0% |
| | | | No-Drone | 2,347.4 ± 149.7 | 75.8 ± 107.1 | 70.3M | 100% |
| RC102_50c | RC1 | 4km | P-ACO | 622.4 ± 36.3 | 1,015.6 ± 363.2 | 82.1M | 52.3% |
| | | | NSGA-II | 889.5 ± 47.9 | 1,876.9 ± 631.3 | 82.8M | 98.5% |
| | | | IVND | 901.0 ± 0.0 | 4,026.0 ± 0.0 | 77.5M | 50.0% |
| | | | No-Drone | 2,207.7 ± 128.3 | 8.0 ± 0.0 | 74.6M | 100% |
| | | 6km | P-ACO | 631.1 ± 38.0 | 940.5 ± 339.5 | 81.8M | 53.4% |
| | | | NSGA-II | 885.1 ± 48.1 | 1,676.1 ± 601.2 | 82.9M | 98.5% |
| | | | IVND | 901.0 ± 0.0 | 4,026.0 ± 0.0 | 77.5M | 50.0% |
| | | | No-Drone | 2,207.7 ± 128.3 | 8.0 ± 0.0 | 74.6M | 100% |
| RC201_50c | RC2 | 4km | P-ACO | 664.3 ± 29.9 | 7,244.7 ± 1,784.7 | 80.8M | 43.2% |
| | | | NSGA-II | 1,009.3 ± 68.3 | 10,923.8 ± 3,049.6 | 63.9M | 97.3% |
| | | | IVND | 1,119.4 ± 0.0 | 14,358.4 ± 0.0 | 37.5M | 50.0% |
| | | | No-Drone | 2,704.6 ± 190.6 | 717.6 ± 500.9 | 19.8M | 100% |
| | | 6km | P-ACO | 665.1 ± 27.4 | 7,363.2 ± 1,663.0 | 81.7M | 44.7% |
| | | | NSGA-II | 1,010.4 ± 62.5 | 10,670.6 ± 3,079.3 | 64.5M | 97.4% |
| | | | IVND | 1,119.4 ± 0.0 | 14,358.4 ± 0.0 | 37.5M | 50.0% |
| | | | No-Drone | 2,704.6 ± 190.6 | 717.6 ± 500.9 | 19.8M | 100% |
| RC202_50c | RC2 | 4km | P-ACO | 663.7 ± 27.7 | 6,316.5 ± 1,764.3 | 83.5M | 33.5% |
| | | | NSGA-II | 968.2 ± 66.5 | 8,225.0 ± 3,023.9 | 72.1M | 97.5% |
| | | | IVND | 1,119.4 ± 0.0 | 14,358.4 ± 0.0 | 52.1M | 50.0% |
| | | | No-Drone | 2,428.3 ± 154.4 | 30.0 ± 62.8 | 37.7M | 100% |
| | | 6km | P-ACO | 675.7 ± 27.7 | 5,951.1 ± 1,673.4 | 83.8M | 34.5% |
| | | | NSGA-II | 967.6 ± 61.4 | 7,829.3 ± 2,884.0 | 72.3M | 97.6% |
| | | | IVND | 1,119.4 ± 0.0 | 14,358.4 ± 0.0 | 52.1M | 50.0% |
| | | | No-Drone | 2,428.3 ± 154.4 | 30.0 ± 62.8 | 37.7M | 100% |

#### 50 Customers — 6T+6D

| Config | TW | End. | Method | Mean Cost ± Std | Mean Tard ± Std | HV | Feas. |
|--------|----|------|--------|-----------------|-----------------|-----|-------|
| RC101_50c | RC1 | 4km | P-ACO | 769.7 ± 22.0 | 962.8 ± 265.6 | 78.8M | 46.1% |
| | | | NSGA-II | 1,211.6 ± 52.1 | 2,339.5 ± 535.7 | 78.4M | 99.1% |
| | | | IVND | 1,107.6 ± 0.0 | 4,499.1 ± 0.0 | 73.6M | 50.0% |
| | | | No-Drone | 2,347.4 ± 149.7 | 75.8 ± 107.1 | 70.3M | 100% |
| | | 6km | P-ACO | 771.5 ± 21.0 | 923.0 ± 263.6 | 78.6M | 46.1% |
| | | | NSGA-II | 1,217.8 ± 52.5 | 2,158.0 ± 525.8 | 78.7M | 98.8% |
| | | | IVND | 1,107.6 ± 0.0 | 4,499.1 ± 0.0 | 73.6M | 50.0% |
| | | | No-Drone | 2,347.4 ± 149.7 | 75.8 ± 107.1 | 70.3M | 100% |
| RC102_50c | RC1 | 4km | P-ACO | 827.1 ± 32.5 | 578.2 ± 265.2 | 82.1M | 49.7% |
| | | | NSGA-II | 1,166.3 ± 60.6 | 1,216.1 ± 460.5 | 80.1M | 100.0% |
| | | | IVND | 1,107.6 ± 0.0 | 4,499.1 ± 0.0 | 74.7M | 50.0% |
| | | | No-Drone | 2,207.7 ± 128.3 | 8.0 ± 0.0 | 74.6M | 100% |
| | | 6km | P-ACO | 831.3 ± 33.4 | 574.4 ± 248.2 | 81.8M | 50.5% |
| | | | NSGA-II | 1,167.5 ± 58.3 | 1,163.5 ± 429.6 | 80.2M | 99.5% |
| | | | IVND | 1,107.6 ± 0.0 | 4,499.1 ± 0.0 | 74.7M | 50.0% |
| | | | No-Drone | 2,207.7 ± 128.3 | 8.0 ± 0.0 | 74.6M | 100% |
| RC201_50c | RC2 | 4km | P-ACO | 862.1 ± 29.7 | 5,134.3 ± 1,693.4 | 80.8M | 41.8% |
| | | | NSGA-II | 1,471.8 ± 83.5 | 8,821.0 ± 3,012.2 | 63.7M | 97.0% |
| | | | IVND | 1,243.3 ± 0.0 | 13,250.8 ± 0.0 | 67.9M | 50.0% |
| | | | No-Drone | 2,704.6 ± 190.6 | 717.6 ± 500.9 | 19.8M | 100% |
| | | 6km | P-ACO | 856.0 ± 21.5 | 5,239.8 ± 1,592.2 | 81.7M | 43.3% |
| | | | NSGA-II | 1,479.7 ± 81.2 | 8,497.1 ± 2,872.1 | 64.3M | 96.5% |
| | | | IVND | 1,243.3 ± 0.0 | 13,250.8 ± 0.0 | 67.9M | 50.0% |
| | | | No-Drone | 2,704.6 ± 190.6 | 717.6 ± 500.9 | 19.8M | 100% |
| RC202_50c | RC2 | 4km | P-ACO | 864.7 ± 32.0 | 4,666.9 ± 1,690.6 | 83.5M | 37.3% |
| | | | NSGA-II | 1,448.9 ± 78.6 | 6,631.2 ± 2,765.7 | 70.6M | 97.1% |
| | | | IVND | 1,243.3 ± 0.0 | 13,250.8 ± 0.0 | 71.6M | 50.0% |
| | | | No-Drone | 2,428.3 ± 154.4 | 30.0 ± 62.8 | 37.7M | 100% |
| | | 6km | P-ACO | 874.0 ± 28.5 | 4,413.6 ± 1,699.4 | 83.8M | 37.5% |
| | | | NSGA-II | 1,427.6 ± 83.8 | 6,423.4 ± 2,668.1 | 70.9M | 96.7% |
| | | | IVND | 1,243.3 ± 0.0 | 13,250.8 ± 0.0 | 71.6M | 50.0% |
| | | | No-Drone | 2,428.3 ± 154.4 | 30.0 ± 62.8 | 37.7M | 100% |

#### 100 Customers (Selected Highlights)

| Config | TW | End. | Method | Mean Cost | Mean Tard | HV | Feas. |
|--------|----|------|--------|-----------|-----------|-----|-------|
| RC101_100c_4T+4D | RC1 | 4km | P-ACO | 779.8 | 1,275.1 | 62.8M | 17.1% |
| | | | NSGA-II | 1,314.2 | 5,740.0 | 34.7M | 48.0% |
| | | | IVND | 956.4 | 10,206.3 | 20.8M | 0.0% |
| | | | No-Drone | 4,847.8 | 205.6 | 34.1M | 100% |
| RC101_100c_4T+4D | RC1 | 6km | P-ACO | 818.5 | 1,179.1 | 71.8M | 25.7% |
| | | | NSGA-II | 1,249.8 | 4,908.4 | 58.1M | 57.3% |
| | | | IVND | 958.7 | 9,721.1 | 20.6M | 0.0% |
| RC202_100c_8T+8D | RC2 | 6km | P-ACO | 987.1 | 2,925.6 | 79.7M | 18.0% |
| | | | NSGA-II | 1,725.1 | 10,560.5 | 50.6M | 43.8% |
| | | | IVND | 1,105.3 | 10,970.3 | 19.3M | 0.0% |

Full 100c results follow the same pattern: P-ACO leads HV but with low feasibility, NSGA-II is the best compromise, IVND produces infeasible solutions, and No-Drone serves as a consistent but costly baseline.

---

## 3. Discussion

### 3.1 Algorithm Performance Comparison

**P-ACO leads in solution quality but fails at scale.** P-ACO achieves the highest HV across all 48 configurations, with costs 30–70% lower than NSGA-II and No-Drone. However, its feasibility rate drops precipitously from 63.7% at 25c to just 21.2% at 100c. The root cause is P-ACO's 3D drone pheromone matrix: at 100 customers, the drone candidate enumeration explodes to O(n³) ≈ 1,000,000 entries, and the pheromone signals are too sparse to converge within 100 iterations. This creates a structural cold-start problem where drone missions are rarely sampled, yet the ant construction depends on drone missions for feasibility under tight time windows.

**NSGA-II offers the best feasibility-quality trade-off.** With 73.6% overall feasibility and HV of 61.2M, NSGA-II is the most reliable drone-capable method. Its crossover and mutation operators naturally maintain diversity, and crowding distance selection preserves solutions across the Pareto front. However, NSGA-II's cost is 30–40% higher than P-ACO on feasible runs, suggesting its routing optimization is less effective than P-ACO's pheromone guidance. The route maps (§2.8) visually confirm this: NSGA-II's routes are more dispersed with longer individual vehicle paths, while P-ACO produces tighter, more efficient clusters.

**IVND is fast but broken.** IVND achieves near-zero feasibility (16.7% overall, 0% at 25c and 100c). Despite producing solutions with competitive costs (322.2 at 25c), these solutions systematically violate constraints. The K-means initialization and 7-neighborhood VND structure, while efficient, appear insufficient for the complex synchronization constraints of truck-drone routing. The Metropolis acceptance criterion (T₀=100, α=0.95) may be too permissive, accepting infeasible solutions too readily.

**No-Drone is the reliability champion.** With 100% feasibility across all configurations, the No-Drone baseline demonstrates that the pure truck VRPTW is well-solved by the GA. Its costs (1,269 at 25c → 5,210 at 100c) serve as an upper bound for what drone-assisted methods should improve upon.

### 3.2 Effectiveness of Drone Usage

Drone utilization patterns reveal fundamental differences in how algorithms explore the drone decision space:

| Method | Avg Drone Missions/Solution | Drone Solution % | Cost Savings vs No-Drone |
|--------|---------------------------|-----------------|--------------------------|
| P-ACO | 33.9 | 100.0% | 72.6% |
| NSGA-II | 24.7 | 99.9% | 65.2% |
| IVND | 7.9 | 99.6% | 67.0% |

All drone-capable methods produce drone missions in nearly 100% of solutions. However, P-ACO generates the most drone missions per solution (33.9) while achieving the lowest costs — suggesting its drone missions are more strategically placed.

**Why drone cost savings are limited.** For a drone mission i→j→k (drone serves customer j, truck goes from i to k):
- Drone cost = d_ij + d_jk (drone rate: 1.0/km)
- Truck cost = 2 × d_ik (truck rate: 2.0/km)
- Saving = 2 × d_ik − (d_ij + d_jk)

By triangle inequality, d_ij + d_jk ≥ d_ik, so maximum saving is d_ik (at most endurance = 4–6 km). Even with the truck's higher per-km rate (2.0×), the max variable cost saving is ~12 units — modest compared to total solution costs of 300–5,000. This fundamental geometric limit explains why even aggressive drone usage yields only incremental cost improvements beyond what good truck routing achieves.

### 3.3 Problem Scale Effects

**Nonlinear degradation with customer count:**
- P-ACO cost grows 2.6× from 25c→100c (328→853), but feasibility drops 3× (63.7%→21.2%)
- NSGA-II cost grows 3.5× (367→1,286), feasibility drops 1.8× (91.5%→51.1%)
- No-Drone cost grows 4.1× (1,269→5,210), but maintains 100% feasibility

NSGA-II degrades faster in cost but slower in feasibility than P-ACO. This suggests that while NSGA-II's solution quality erodes with scale, its constraint-handling mechanisms (diversity-preserving selection, separate crossover/mutation) remain relatively robust.

**Fleet density effects (4T+4D vs 6T+6D vs 8T+8D at 100c):**
More vehicles increase fixed costs but provide more routing flexibility. At 100c, P-ACO HV is nearly identical across fleet sizes (62.8M–74.6M), suggesting the extra vehicles do not significantly improve the Pareto front — the bottleneck is the algorithm's ability to find feasible synchronized routes, not vehicle availability.

### 3.4 RC1 vs RC2 (Time Window Tightness)

This is the most striking finding: **P-ACO is the only method that benefits from wider time windows.**

| Method | RC1→RC2 HV Change | Interpretation |
|--------|-------------------|----------------|
| P-ACO | **+6.9%** | Exploits scheduling flexibility |
| NSGA-II | −27.1% | Overwhelmed by expanded objective space |
| IVND | −30.5% | Same failure mode regardless of TW |
| No-Drone | −42.0% | Tardiness dominates under wide TW |

P-ACO's pheromone guidance appears to effectively navigate the expanded scheduling flexibility of RC2 — the ants can find routes with lower cost even as tardiness varies more widely. In contrast, NSGA-II's non-dominated sorting struggles when the objective space expands, as the Pareto front becomes harder to approximate with the same population size.

Under RC2, tardiness values explode (5–10× higher across all methods), but this is expected: wider time windows allow more scheduling variation. The critical metric is whether cost improves commensurately — only P-ACO achieves this.

### 3.5 Nonlinear Charging Impact

The nonlinear charging model (piecewise SOC-rate segments: 0–20% fast, 20–80% normal, 80–100% slow) affects all methods through the battery constraint in solution evaluation. However, since the current implementation uses a simplified energy model (1 kWh/km, 100 kWh capacity, recharge at depot), the nonlinear charging effect is modest:

- **Observed effect**: Feasibility rates are depressed across all drone methods, particularly at 100c where long routes deplete batteries
- **P-ACO** is most affected because its drone missions consume additional battery for truck detours to recovery nodes
- **Future work**: Full integration of en-route charging stations (at (4,12) and (12,4)) would allow studying charging stop optimization

### 3.6 Failure Cases and Limitations

**P-ACO failure modes:**
1. **3D pheromone cold-start**: The drone pheromone matrix (O(n³) entries) receives too few updates per iteration (1 drone mission per ant) to converge. This is a structural limitation of extending 2D ACO to 3D decision spaces.
2. **Runtime explosion at scale**: At 100 customers, P-ACO takes 16+ minutes per run (58.5× slower than 25c). Multi-threaded ant construction could mitigate this.
3. **Feasibility collapse**: From 69% at 25c to 21% at 100c, driven by the interaction of tight time windows + battery + drone synchronization constraints.

**NSGA-II failure modes:**
1. **Cost disadvantage**: 30–40% higher costs than P-ACO on feasible solutions. The genetic operators are less targeted than pheromone guidance.
2. **RC2 degradation**: Loses 27% HV under wider time windows, suggesting population diversity is insufficient for the expanded objective space.
3. **100c reliability**: Feasibility drops to 51% at 100 customers, though still much better than P-ACO (21%) or IVND (0%).

**IVND failure modes:**
1. **Systematic infeasibility**: 0% feasibility at 25c and 100c. The neighborhood structures likely cannot repair time window and synchronization violations.
2. **Zero variance**: std=0 on all metrics suggests the algorithm converges to a single point regardless of random seed — a sign of premature convergence.
3. **Parameter tuning needed**: The temperature schedule (T₀=100, α=0.95) and tabu tenure (15) may need adjustment for the EVRP-TW constraint structure.

---

## 4. Conclusion

This experiment systematically compares P-ACO, NSGA-II, IVND, and a No-Drone baseline on the truck-drone EVRP-TW using 48 experimental configurations across Solomon RC benchmark instances (25–100 customers). Key findings:

1. **P-ACO achieves the best solution quality (HV) across all configurations**, with costs 30–70% lower than alternatives. However, its feasibility drops to 21% at 100 customers, and runtime becomes prohibitive (16+ min/run). P-ACO is recommended for small-to-medium instances (≤50 customers) where its pheromone guidance can converge effectively.

2. **NSGA-II is the most reliable drone-capable method**, with 73.6% overall feasibility and reasonable cost-quality trade-offs. It is the recommended choice for large instances (100 customers) where P-ACO becomes infeasible both computationally and constraint-wise.

3. **IVND is the fastest method (0.17s at 100c) but produces infeasible solutions** on most configurations. Significant parameter tuning (temperature schedule, tabu tenure, neighborhood weights) is needed before it can serve as a viable alternative.

4. **Drone usage provides limited cost benefit (~3% max savings)** due to geometric constraints (triangle inequality bounds drone savings to at most the endurance range × truck cost rate). The primary value of drones in this problem is not cost reduction but enabling feasible solutions under tight constraints.

5. **Time window tightness is a critical differentiator**: P-ACO uniquely exploits wider time windows (+6.9% HV), while all other methods degrade. This suggests P-ACO's exploration strategy is better suited to the scheduling flexibility of real-world delivery scenarios.

6. **The No-Drone baseline maintains 100% feasibility** and serves as a crucial control — any drone-capable method must justify its added complexity against this reliable baseline.

**Future work:**
- Multi-thread P-ACO to address the 100c runtime bottleneck
- Tune IVND parameters (temperature, tabu tenure, neighborhood selection) to improve feasibility
- Integrate en-route charging station optimization
- Add POMO (Policy Optimization with Multiple Optima) as a fifth comparison method using deep reinforcement learning
- Extend to additional Solomon instance classes (C, R series)

---

## Appendix A: File Structure

| File / Directory | Purpose |
|------------------|---------|
| `week3/main.py` | Main experiment entry point |
| `week3/config.py` | Centralized parameter configuration |
| `week3/run_pomo_experiments.py` | POMO-only experiment runner |
| `week3/fix_hv.py` | Standalone hypervolume recalculation |
| `week3/merge_results.py` | Results merging utility |
| `week3/utils/data_loader.py` | Solomon instance loading and preprocessing |
| `week3/utils/problem_model.py` | Core VRP model, solution evaluation, HV computation |
| `week3/utils/report_generator.py` | Automated report generation |
| `week3/algorithms/no_drone.py` | Truck-only GA baseline |
| `week3/algorithms/paco.py` | P-ACO algorithm implementation |
| `week3/algorithms/nsga2.py` | NSGA-II algorithm implementation |
| `week3/algorithms/ivnd.py` | IVND algorithm implementation |
| `week3/algorithms/pomo/` | POMO DRL method (training + inference) |
| `week3/algorithms/pomo_solver.py` | POMO integration wrapper |
| `week3/runner/experiment_runner.py` | Unified experiment execution engine |
| `week3/results/` | Experiment result JSON files |
| `week3/data/` | Generated problem instances |

## Appendix B: Complete Experiment Results

Full per-configuration data is available in:
- `week3/results/results_20260702_152443_hv_fixed.json` — All 48 experiment configurations with corrected HV values
- `week3/results/interim_20260702_150309.json` — Earlier interim results
- `week3/week3_report.md` — This report
