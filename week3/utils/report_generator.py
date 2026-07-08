# -*- coding: utf-8 -*-
"""
Generate Week 3 experiment report from results JSON.
Produces a markdown report following the sample.md structure.
"""

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RESULTS_DIR, HV_REFERENCE


def find_latest_results():
    """Find the most recent results JSON file."""
    files = [f for f in os.listdir(RESULTS_DIR)
             if f.startswith('results_') and f.endswith('.json')]
    if not files:
        # Try interim
        files = [f for f in os.listdir(RESULTS_DIR)
                 if f.startswith('interim_') and f.endswith('.json')]
    if not files:
        return None
    files.sort(reverse=True)
    return os.path.join(RESULTS_DIR, files[0])


def generate_summary_table(results):
    """Generate the per-configuration summary table."""
    rows = []
    for exp in results:
        label = exp['label']
        n_cust = exp['n_customers']
        tw = exp['tw_type']
        end = exp['endurance_name']
        n_t = exp['n_trucks']
        n_d = exp['n_drones']

        for method in ['P-ACO', 'NSGA-II', 'IVND', 'No-Drone', 'POMO']:
            if method not in exp['methods']:
                continue
            m = exp['methods'][method]
            rows.append({
                'config': f'{n_cust}c_{tw}_{end}_{n_t}T+{n_d}D',
                'tw': tw,
                'endurance': end,
                'method': method,
                'mean_cost': m['mean_cost'],
                'std_cost': m['std_cost'],
                'mean_tardiness': m['mean_tardiness'],
                'std_tardiness': m['std_tardiness'],
                'hv': m['hypervolume'],
                'feasibility': m['feasibility_rate'],
                'drone_pct': m['drone_solution_pct'],
                'avg_drone_missions': m['avg_drone_missions'],
                'runtime': m['mean_runtime'],
            })
    return rows


def generate_aggregate_table(rows):
    """Generate aggregate statistics by scale and method."""
    from collections import defaultdict
    agg = defaultdict(lambda: {'costs': [], 'tards': [], 'hvs': [], 'runtimes': [],
                                'feasibility': [], 'drone_pct': [], 'drone_missions': []})

    for r in rows:
        scale = r['config'].split('_')[0]  # e.g., "25c"
        key = (scale, r['method'])

        # Determine vehicle config
        if '4T+4D' in r['config']:
            vc = '4T+4D'
        elif '6T+6D' in r['config']:
            vc = '6T+6D'
        elif '8T+8D' in r['config']:
            vc = '8T+8D'
        elif '2T+2D' in r['config']:
            vc = '2T+2D'
        else:
            vc = ''

        full_key = (scale, vc, r['method'])
        agg[full_key]['costs'].append(r['mean_cost'])
        agg[full_key]['tards'].append(r['mean_tardiness'])
        agg[full_key]['hvs'].append(r['hv'])
        agg[full_key]['runtimes'].append(r['runtime'])
        agg[full_key]['feasibility'].append(r['feasibility'])
        agg[full_key]['drone_pct'].append(r['drone_pct'])
        agg[full_key]['drone_missions'].append(r['avg_drone_missions'])

    result = []
    for (scale, vc, method), vals in sorted(agg.items()):
        result.append({
            'scale': scale,
            'config': vc,
            'method': method,
            'feasibility': f"{sum(vals['feasibility'])/len(vals['feasibility'])*100:.0f}%",
            'mean_cost': f"{sum(vals['costs'])/len(vals['costs']):.1f}",
            'mean_tardiness': f"{sum(vals['tards'])/len(vals['tards']):.1f}",
            'mean_hv': f"{sum(vals['hvs'])/len(vals['hvs']):.0f}",
            'mean_runtime': f"{sum(vals['runtimes'])/len(vals['runtimes']):.1f}",
        })
    return result


def generate_report():
    """Generate the full markdown report."""
    results_path = find_latest_results()
    if not results_path:
        print("ERROR: No results found. Run experiments first.")
        return

    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    rows = generate_summary_table(results)
    agg = generate_aggregate_table(rows)

    report = []
    report.append('# Week 3 Lab: Truck-Drone EVRP-TW Experiment Report')
    report.append('')
    report.append(f'*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}*')
    report.append(f'*Data source: {os.path.basename(results_path)}*')
    report.append('')
    report.append('---')
    report.append('')

    # ── Abstract ──
    report.append('## Abstract')
    report.append('')
    report.append(
        'This report presents a systematic comparative analysis of three multi-objective '
        'optimization methods for the truck-drone collaborative routing problem:'
    )
    report.append('')
    report.append('- **P-ACO**: Pareto Ant Colony Optimization for synchronized truck-drone routing')
    report.append('- **NSGA-II**: Non-dominated Sorting Genetic Algorithm II')
    report.append('- **IVND**: Improved Variable Neighborhood Descent with tabu search and SA acceptance')
    report.append('- **POMO**: Policy Optimization with Multiple Optima (DRL-based constructive heuristic)')
    report.append('')
    report.append(
        'A pure truck-only baseline (**No-Drone**) is included as a control. '
        'Experiments are conducted on Solomon RC benchmark instances (RC101, RC102 for tight time windows; '
        'RC201, RC202 for wide time windows) with 25, 50, and 100 customers. '
        'All methods optimize travel cost and tardiness as dual objectives, '
        'with Hypervolume (HV) as the comprehensive performance metric. '
        'Nonlinear charging efficiency is considered as an additional evaluation dimension.'
    )
    report.append('')
    report.append('---')
    report.append('')

    # ── 1. Experimental Setup ──
    report.append('## 1. Experimental Setup')
    report.append('')
    report.append('### 1.1 Comparison Objectives')
    report.append('')
    report.append('- **Test Methods**: P-ACO, NSGA-II, IVND (three-way equal comparison)')
    report.append('- **Baseline**: No-Drone (pure truck delivery)')
    report.append('- **Research Questions**:')
    report.append('  1. Which method achieves the best solution quality (HV, Cost, Tardiness)?')
    report.append('  2. Can drone-assisted delivery significantly reduce total cost vs pure trucks?')
    report.append('  3. How does drone endurance (4 km vs 6 km) affect solution quality?')
    report.append('  4. How does time window tightness (RC1 vs RC2) affect performance?')
    report.append('  5. What is the impact of nonlinear charging on overall efficiency?')
    report.append('')
    report.append('### 1.2 Dataset and Instance Configuration')
    report.append('')
    report.append('| Parameter | Value |')
    report.append('|-----------|-------|')
    report.append('| Dataset | Solomon RC series (RC101, RC102, RC201, RC202) |')
    report.append('| Customer sizes | 25, 50, 100 |')
    report.append('| Coordinate scaling | Solomon [0,100] → Urban [0,16] km |')
    report.append('| Depot location | (8.0, 8.0) |')
    report.append('| Time window types | RC1 (tight, 120 min) / RC2 (wide, 240 min) |')
    report.append('| Truck speed | 35 km/h |')
    report.append('| Drone speed | 50 km/h |')
    report.append('| Truck capacity | 200.0 |')
    report.append('| Drone capacity | 40.0 |')
    report.append('| Drone endurance | medium = 4 km, high = 6 km |')
    report.append('')
    report.append('### 1.3 Vehicle Configuration')
    report.append('')
    report.append('| Customers | Trucks | Drones |')
    report.append('|-----------|--------|--------|')
    report.append('| 25 | 2 | 2 |')
    report.append('| 50 | 4 / 6 | 4 / 6 |')
    report.append('| 100 | 4 / 6 / 8 | 4 / 6 / 8 |')
    report.append('')
    report.append('### 1.4 Algorithm Configuration')
    report.append('')
    report.append('| Parameter | P-ACO | NSGA-II | IVND | POMO |')
    report.append('|-----------|-------|---------|------|------|')
    report.append('| Population/Ants | 50-120 | 50-150 | — | — |')
    report.append('| Iterations/Generations | 100 | 120 | 200 | 200 epochs × 1000 episodes |')
    report.append('| Key mechanism | 3D pheromone | Non-dominated sorting | Tabu + SA + VND | Transformer + REINFORCE |')
    report.append('| Repeats | 10 | 10 | 10 | 10 (POMO starts per instance) |')
    report.append('')
    report.append('### 1.5 Evaluation Metrics')
    report.append('')
    report.append('| Metric | Definition |')
    report.append('|--------|------------|')
    report.append('| **Cost** | Vehicle fixed costs + truck distance x 2.0 + drone distance x 1.0 |')
    report.append('| **Tardiness** | Σ max(0, arrival - TW end) x priority weight |')
    report.append('| **Hypervolume (HV)** | Area covered by Pareto front relative to reference point (170, 140) |')
    report.append('| **Drone Utilization** | % solutions with drone missions + avg missions per solution |')
    report.append('| **Feasibility** | % of runs producing feasible solutions |')
    report.append('| **Runtime** | Wall-clock time per run |')
    report.append('')
    report.append('### 1.6 Hardware and Environment')
    report.append('')
    report.append('| Item | Configuration |')
    report.append('|------|---------------|')
    report.append('| Model | MacBook Air (Mac16,13) |')
    report.append('| Chip | Apple M4 |')
    report.append('| Cores | 10 (4 performance + 6 efficiency) |')
    report.append('| Memory | 16 GB |')
    report.append('| OS | macOS 15.7.7 (Sequoia) |')
    report.append('| Architecture | arm64 (Apple Silicon) |')
    report.append('| Python | 3.14.0 |')
    report.append('| Key dependencies | NumPy (optional), Matplotlib (optional) |')
    report.append('')
    report.append('---')
    report.append('')

    # ── 2. Results ──
    report.append('## 2. Results')
    report.append('')
    report.append('### 2.1 Per-Configuration Results')
    report.append('')

    # Per-config table
    report.append('| Config | Method | Cost (mean±std) | Tardiness (mean±std) | HV | Feas. | Drone% | Runtime(s) |')
    report.append('|--------|--------|----------------|---------------------|----|-------|--------|-----------|')
    for r in rows:
        report.append(
            f'| {r["config"]} | {r["method"]} | '
            f'{r["mean_cost"]:.1f} ± {r["std_cost"]:.1f} | '
            f'{r["mean_tardiness"]:.1f} ± {r["std_tardiness"]:.1f} | '
            f'{r["hv"]:.0f} | '
            f'{r["feasibility"]*100:.0f}% | '
            f'{r["drone_pct"]:.1f}% | '
            f'{r["runtime"]:.1f} |'
        )
    report.append('')

    # Aggregate table
    report.append('### 2.2 Aggregate Statistics')
    report.append('')
    report.append('| Scale | Config | Method | Feasibility | Mean Cost | Mean Tardiness | Mean HV | Mean Runtime(s) |')
    report.append('|-------|--------|--------|-------------|-----------|----------------|---------|-----------------|')
    for a in agg:
        report.append(
            f'| {a["scale"]} | {a["config"]} | {a["method"]} | '
            f'{a["feasibility"]} | '
            f'{a["mean_cost"]} | {a["mean_tardiness"]} | '
            f'{a["mean_hv"]} | {a["mean_runtime"]} |'
        )
    report.append('')

    # ── 3. Discussion ──
    report.append('## 3. Discussion')
    report.append('')
    report.append('### 3.1 Algorithm Performance Comparison')
    report.append('')
    report.append('*Analysis will be populated with data-driven observations after experiments complete.*')
    report.append('')
    report.append('### 3.2 Effectiveness of Drone Usage')
    report.append('')
    report.append('*Analysis of drone utilization rates across different methods and configurations.*')
    report.append('')
    report.append('### 3.3 Problem Scale Effects')
    report.append('')
    report.append('*How performance degrades as customer count increases from 25 to 100.*')
    report.append('')
    report.append('### 3.4 RC1 vs RC2 (Time Window Tightness)')
    report.append('')
    report.append('*Impact of tight vs wide time windows on each algorithm.*')
    report.append('')
    report.append('### 3.5 Nonlinear Charging Impact')
    report.append('')
    report.append('*Effect of piecewise nonlinear charging rates on EVRP solutions.*')
    report.append('')
    report.append('### 3.6 Failure Cases and Limitations')
    report.append('')
    report.append('*Discussion of when and why each method fails.*')
    report.append('')
    report.append('---')
    report.append('')

    # ── 4. Conclusion ──
    report.append('## 4. Conclusion')
    report.append('')
    report.append('*Conclusions will be drawn from experimental results.*')
    report.append('')
    report.append('---')
    report.append('')

    # ── Appendix ──
    report.append('## Appendix: File Structure')
    report.append('')
    report.append('| File / Directory | Purpose |')
    report.append('|------------------|---------|')
    report.append('| `week3/main.py` | Main entry point |')
    report.append('| `week3/config.py` | Centralized parameter configuration |')
    report.append('| `week3/utils/data_loader.py` | Solomon instance loading and preprocessing |')
    report.append('| `week3/utils/problem_model.py` | Core VRP model, solution evaluation, HV computation |')
    report.append('| `week3/utils/report_generator.py` | Automated report generation |')
    report.append('| `week3/algorithms/no_drone.py` | Truck-only GA baseline |')
    report.append('| `week3/algorithms/paco.py` | P-ACO algorithm implementation |')
    report.append('| `week3/algorithms/nsga2.py` | NSGA-II algorithm implementation |')
    report.append('| `week3/algorithms/ivnd.py` | IVND algorithm implementation |')
    report.append('| `week3/runner/experiment_runner.py` | Unified experiment execution engine |')
    report.append('| `week3/results/` | Experiment result JSON files |')
    report.append('| `week3/data/` | Generated problem instances |')

    # Write report
    report_path = os.path.join(os.path.dirname(RESULTS_DIR), 'week3_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f'Report generated: {report_path}')
    return report_path


if __name__ == '__main__':
    generate_report()
