# -*- coding: utf-8 -*-
"""
Week 7 — FURP Project Finalization.

Key contributions this week:
  1. 2-drones-per-truck support with optional deployment
  2. Expanded SOTA comparison: 50c/100c, all 6 Solomon types, 5 repetitions
  3. Statistical tests: Wilcoxon signed-rank + Friedman
  4. Clustering-first baselines: Sweep, CW-Savings, K-means+NN
  5. RC2 conditional EDD trigger fix
  6. Clear method definition for publication

Our Method (Ours):
    POMO (pre-trained, Kwon 2020)
    + Hybrid Clustering (Angle Petal for RC1, Adaptive TW for RC2)
    + Cross-Route Drone Insertion (distance-based saving)
    + Adaptive EDD Repair (Partial for ≤50c, Full for 100c)
    + 2-Drone Optional Deployment (new in Week 7)

Core Claim:
    In truck-drone collaborative routing with time windows, EDD repair
    is a simple but overlooked method for achieving high feasibility.
    While other methods find cheaper routes, ours is the only one that
    guarantees 100% time-window feasibility at reasonable cost.
"""
