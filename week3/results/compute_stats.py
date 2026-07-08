import json
import statistics
from collections import defaultdict

# Load data
with open('/Users/jackalwick/Desktop/Truck-Drone EVRP-TW/week3/results/results_20260702_152443_hv_fixed.json', 'r') as f:
    experiments = json.load(f)

methods = ["P-ACO", "NSGA-II", "IVND", "No-Drone"]
N = len(experiments)
print(f"Total experiments: {N}")

# Helper: extract metrics for a method across all experiments
def extract(method, key):
    return [exp["methods"][method][key] for exp in experiments]

# ========== 1. Overall averages ==========
print("\n" + "="*80)
print("1. OVERALL AVERAGES (across all 48 experiments)")
print("="*80)
print(f"{'Method':<12} {'Cost':>12} {'Tardiness':>14} {'HV':>16} {'Feas%':>8} {'Runtime':>10} {'Drone%':>8}")
for m in methods:
    mc = statistics.mean(extract(m, "mean_cost"))
    mt = statistics.mean(extract(m, "mean_tardiness"))
    mhv = statistics.mean(extract(m, "hypervolume"))
    mf = statistics.mean(extract(m, "feasibility_rate")) * 100
    mr = statistics.mean(extract(m, "mean_runtime"))
    md = statistics.mean(extract(m, "drone_solution_pct"))
    print(f"{m:<12} {mc:>12.2f} {mt:>14.2f} {mhv:>16.2f} {mf:>7.2f}% {mr:>10.2f} {md:>7.2f}%")

# ========== 2. By customer scale ==========
print("\n" + "="*80)
print("2. BY CUSTOMER SCALE (25c, 50c, 100c)")
print("="*80)
for scale in [25, 50, 100]:
    exps = [e for e in experiments if e["n_customers"] == scale]
    print(f"\n--- {scale} customers ({len(exps)} experiments) ---")
    print(f"{'Method':<12} {'Mean HV':>16} {'Mean Cost':>14} {'Feas%':>8}")
    for m in methods:
        hv_vals = [exp["methods"][m]["hypervolume"] for exp in exps]
        cost_vals = [exp["methods"][m]["mean_cost"] for exp in exps]
        feas_vals = [exp["methods"][m]["feasibility_rate"] * 100 for exp in exps]
        print(f"{m:<12} {statistics.mean(hv_vals):>16.2f} {statistics.mean(cost_vals):>14.2f} {statistics.mean(feas_vals):>7.2f}%")

# ========== 3. By TW type ==========
print("\n" + "="*80)
print("3. BY TW TYPE (RC1 tight, RC2 wide)")
print("="*80)
for tw in ["RC1", "RC2"]:
    exps = [e for e in experiments if e["tw_type"] == tw]
    print(f"\n--- {tw} ({len(exps)} experiments) ---")
    print(f"{'Method':<12} {'Mean HV':>16} {'Mean Cost':>14} {'Feas%':>8}")
    for m in methods:
        hv_vals = [exp["methods"][m]["hypervolume"] for exp in exps]
        cost_vals = [exp["methods"][m]["mean_cost"] for exp in exps]
        feas_vals = [exp["methods"][m]["feasibility_rate"] * 100 for exp in exps]
        print(f"{m:<12} {statistics.mean(hv_vals):>16.2f} {statistics.mean(cost_vals):>14.2f} {statistics.mean(feas_vals):>7.2f}%")

# ========== 4. By endurance ==========
print("\n" + "="*80)
print("4. BY ENDURANCE (medium 4km, high 6km)")
print("="*80)
for end in ["medium", "high"]:
    exps = [e for e in experiments if e["endurance_name"] == end]
    print(f"\n--- {end} ({len(exps)} experiments) ---")
    print(f"{'Method':<12} {'Mean HV':>16} {'Mean Cost':>14}")
    for m in methods:
        hv_vals = [exp["methods"][m]["hypervolume"] for exp in exps]
        cost_vals = [exp["methods"][m]["mean_cost"] for exp in exps]
        print(f"{m:<12} {statistics.mean(hv_vals):>16.2f} {statistics.mean(cost_vals):>14.2f}")

# ========== 5. Best and worst per scale (by HV) ==========
print("\n" + "="*80)
print("5. BEST AND WORST METHOD PER SCALE (by mean HV)")
print("="*80)
for scale in [25, 50, 100]:
    exps = [e for e in experiments if e["n_customers"] == scale]
    print(f"\n--- {scale} customers ---")
    scores = {}
    for m in methods:
        hv_vals = [exp["methods"][m]["hypervolume"] for exp in exps]
        scores[m] = statistics.mean(hv_vals)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print(f"  Best:  {ranked[0][0]} (HV={ranked[0][1]:.2f})")
    print(f"  Worst: {ranked[-1][0]} (HV={ranked[-1][1]:.2f})")
    for m, hv in ranked:
        print(f"    {m}: {hv:.2f}")

# ========== 6. Cost Advantage ==========
print("\n" + "="*80)
print("6. COST ADVANTAGE: % savings vs No-Drone")
print("="*80)
print(f"{'Method':<12} {'Avg Savings%':>14}")
for m in ["P-ACO", "NSGA-II", "IVND"]:
    savings = []
    for exp in experiments:
        nd_cost = exp["methods"]["No-Drone"]["mean_cost"]
        m_cost = exp["methods"][m]["mean_cost"]
        saving = (nd_cost - m_cost) / nd_cost * 100
        savings.append(saving)
    print(f"{m:<12} {statistics.mean(savings):>14.2f}%")

# ========== 7. Time Window Flexibility ==========
print("\n" + "="*80)
print("7. TIME WINDOW FLEXIBILITY: (HV_RC2 - HV_RC1) / HV_RC1 * 100")
print("="*80)
for m in methods:
    rc1_hv = statistics.mean([exp["methods"][m]["hypervolume"] for exp in experiments if exp["tw_type"] == "RC1"])
    rc2_hv = statistics.mean([exp["methods"][m]["hypervolume"] for exp in experiments if exp["tw_type"] == "RC2"])
    gap = (rc2_hv - rc1_hv) / rc1_hv * 100
    print(f"{m:<12} RC1={rc1_hv:>14.2f}  RC2={rc2_hv:>14.2f}  Gap={gap:>8.2f}%")

# ========== 8. Drone Utilization ==========
print("\n" + "="*80)
print("8. DRONE UTILIZATION: Avg drone mission count")
print("="*80)
for m in ["P-ACO", "NSGA-II", "IVND"]:
    missions = extract(m, "avg_drone_missions")
    print(f"{m:<12} {statistics.mean(missions):>14.2f} missions")

# ========== 9. Runtime scaling ==========
print("\n" + "="*80)
print("9. RUNTIME SCALING: runtime growth 25c -> 50c -> 100c")
print("="*80)
print(f"{'Method':<12} {'25c':>10} {'50c':>10} {'100c':>10} {'50c/25c':>10} {'100c/25c':>10}")
for m in methods:
    rt_25 = statistics.mean([exp["methods"][m]["mean_runtime"] for exp in experiments if exp["n_customers"] == 25])
    rt_50 = statistics.mean([exp["methods"][m]["mean_runtime"] for exp in experiments if exp["n_customers"] == 50])
    rt_100 = statistics.mean([exp["methods"][m]["mean_runtime"] for exp in experiments if exp["n_customers"] == 100])
    ratio_50_25 = rt_50 / rt_25 if rt_25 > 0 else float('inf')
    ratio_100_25 = rt_100 / rt_25 if rt_25 > 0 else float('inf')
    print(f"{m:<12} {rt_25:>10.4f} {rt_50:>10.4f} {rt_100:>10.4f} {ratio_50_25:>10.1f}x {ratio_100_25:>10.1f}x")

# ========== 10. Top-3 configurations per method by HV ==========
print("\n" + "="*80)
print("10. TOP-3 CONFIGURATIONS PER METHOD (by HV)")
print("="*80)
for m in methods:
    ranked = sorted(experiments, key=lambda e: e["methods"][m]["hypervolume"], reverse=True)
    print(f"\n--- {m} ---")
    for i, exp in enumerate(ranked[:3]):
        hv = exp["methods"][m]["hypervolume"]
        print(f"  {i+1}. {exp['label']}  HV={hv:.2f}  Cost={exp['methods'][m]['mean_cost']:.2f}  Feas={exp['methods'][m]['feasibility_rate']*100:.1f}%")

# ========== BONUS: Summary table ==========
print("\n" + "="*80)
print("SUMMARY TABLE FOR REPORT")
print("="*80)

# Method comparison matrix
print("\nMethod | Avg Cost | Avg Tardiness | Avg HV | Feas% | Runtime(s) | Drone% | CostSave%")
print("-" * 90)
for m in methods:
    mc = statistics.mean(extract(m, "mean_cost"))
    mt = statistics.mean(extract(m, "mean_tardiness"))
    mhv = statistics.mean(extract(m, "hypervolume"))
    mf = statistics.mean(extract(m, "feasibility_rate")) * 100
    mr = statistics.mean(extract(m, "mean_runtime"))
    md = statistics.mean(extract(m, "drone_solution_pct"))
    if m == "No-Drone":
        cs = 0
    else:
        cs = statistics.mean([(exp["methods"]["No-Drone"]["mean_cost"] - exp["methods"][m]["mean_cost"]) / exp["methods"]["No-Drone"]["mean_cost"] * 100 for exp in experiments])
    print(f"{m:<8} | {mc:>9.2f} | {mt:>13.2f} | {mhv:>12.0f} | {mf:>5.1f}% | {mr:>9.2f} | {md:>6.1f}% | {cs:>7.1f}%")

# Scale summary
print("\nScale | Method | Avg HV | Avg Cost | Feas%")
print("-" * 55)
for scale in [25, 50, 100]:
    exps = [e for e in experiments if e["n_customers"] == scale]
    for m in methods:
        hv = statistics.mean([exp["methods"][m]["hypervolume"] for exp in exps])
        cost = statistics.mean([exp["methods"][m]["mean_cost"] for exp in exps])
        feas = statistics.mean([exp["methods"][m]["feasibility_rate"] for exp in exps]) * 100
        print(f"{scale}c   | {m:<8} | {hv:>12.0f} | {cost:>9.2f} | {feas:>5.1f}%")

# TW summary
print("\nTW | Method | Avg HV | Avg Cost | Feas%")
print("-" * 55)
for tw in ["RC1", "RC2"]:
    exps = [e for e in experiments if e["tw_type"] == tw]
    for m in methods:
        hv = statistics.mean([exp["methods"][m]["hypervolume"] for exp in exps])
        cost = statistics.mean([exp["methods"][m]["mean_cost"] for exp in exps])
        feas = statistics.mean([exp["methods"][m]["feasibility_rate"] for exp in exps]) * 100
        print(f"{tw:<4} | {m:<8} | {hv:>12.0f} | {cost:>9.2f} | {feas:>5.1f}%")

# Endurance summary
print("\nEndurance | Method | Avg HV | Avg Cost")
print("-" * 50)
for end in ["medium", "high"]:
    exps = [e for e in experiments if e["endurance_name"] == end]
    for m in methods:
        hv = statistics.mean([exp["methods"][m]["hypervolume"] for exp in exps])
        cost = statistics.mean([exp["methods"][m]["mean_cost"] for exp in exps])
        print(f"{end:<10} | {m:<8} | {hv:>12.0f} | {cost:>9.2f}")

print("\nDone.")
