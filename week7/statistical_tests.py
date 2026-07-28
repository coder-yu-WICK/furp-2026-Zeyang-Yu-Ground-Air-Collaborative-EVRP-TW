# -*- coding: utf-8 -*-
"""
Statistical Tests for SOTA Comparison — Week 7.

Implements non-parametric statistical tests for comparing multiple
optimization methods across multiple problem instances:

  1. Wilcoxon Signed-Rank Test (pairwise, two methods)
  2. Friedman Test (multi-method comparison)
  3. Friedman with post-hoc Nemenyi test (critical difference)

All implementations are pure Python (no scipy dependency required),
using standard statistical approximations.

Usage:
    from week7.statistical_tests import (
        wilcoxon_signed_rank_test,
        friedman_test,
        friedman_nemenyi_posthoc,
        compute_method_rankings,
    )
"""

import math
import json


# ── Helper: Standard Normal Distribution ─────────────────────────────────

def _normal_cdf(x):
    """Cumulative distribution function of standard normal."""
    # Abramowitz & Stegun approximation
    if x < -8.0:
        return 0.0
    if x > 8.0:
        return 1.0
    # Constants for approximation
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x_abs = abs(x)
    t = 1.0 / (1.0 + p * x_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x_abs * x_abs / 2.0)
    return 0.5 * (1.0 + sign * y)


def _normal_ppf(p):
    """Percent point function (inverse CDF) of standard normal.

    Uses the approximation from Odeh & Evans (1974).
    """
    if p <= 0.0:
        return -float('inf')
    if p >= 1.0:
        return float('inf')
    if p < 0.5:
        return -_normal_ppf(1.0 - p)

    # Rational approximation for 0.5 <= p < 1.0
    a0 = 2.50662823884
    a1 = -18.61500062529
    a2 = 41.39119773534
    a3 = -25.44106049637
    b0 = -8.47351093090
    b1 = 23.08336743743
    b2 = -21.06224101826
    b3 = 3.13082909833
    c0 = 0.3374754822726147
    c1 = 0.9761690190917186
    c2 = 0.1607979714918209
    c3 = 0.0276438810333863
    c4 = 0.0038405729373609
    c5 = 0.0003951896511919
    c6 = 0.0000321767881768
    c7 = 0.0000002888167364
    c8 = 0.000000003960315187

    y = p - 0.5
    if abs(y) < 0.42:
        r = y * y
        return y * (((a3 * r + a2) * r + a1) * r + a0) / ((((b3 * r + b2) * r + b1) * r + b0) * r + 1.0)
    else:
        if y > 0:
            r = math.sqrt(-math.log(1.0 - p))
        else:
            r = math.sqrt(-math.log(p))
        return ((((c7 * r + c6) * r + c5) * r + c4) * r + c3) * r + c2 * r + c1 + c0 / r


# ── Chi-Square Distribution ──────────────────────────────────────────────

def _chi2_cdf(x, df):
    """CDF of chi-square distribution with df degrees of freedom.

    Uses the Wilson-Hilferty approximation for df > 2, and exact
    computation for df <= 2.
    """
    if x <= 0.0:
        return 0.0
    if df <= 0:
        return 1.0

    if df == 1:
        return 2.0 * _normal_cdf(math.sqrt(x)) - 1.0
    elif df == 2:
        return 1.0 - math.exp(-x / 2.0)
    else:
        # Wilson-Hilferty transformation
        z = ((x / df) ** (1.0/3.0) - 1.0 + 2.0/(9.0*df)) / math.sqrt(2.0/(9.0*df))
        return _normal_cdf(z)


# ── Studentized Range Distribution ───────────────────────────────────────

def _q_critical(alpha, k, df=float('inf')):
    """Critical values for the studentized range distribution (q).

    For post-hoc Nemenyi test. Uses tabulated values for alpha=0.05 and 0.01
    with df=inf (most common case for Friedman post-hoc).

    Args:
        alpha: significance level (0.05 or 0.01)
        k: number of methods being compared
        df: degrees of freedom (default: inf for Nemenyi test)

    Returns:
        Critical q value
    """
    # Tabulated q values for alpha=0.05, df=inf
    Q_005 = {
        2: 2.772, 3: 3.314, 4: 3.633, 5: 3.858,
        6: 4.030, 7: 4.170, 8: 4.286, 9: 4.387, 10: 4.474,
        11: 4.552, 12: 4.622, 13: 4.685, 14: 4.743, 15: 4.796,
    }
    # Tabulated q values for alpha=0.01, df=inf
    Q_001 = {
        2: 3.643, 3: 4.120, 4: 4.403, 5: 4.603,
        6: 4.757, 7: 4.882, 8: 4.987, 9: 5.078, 10: 5.157,
        11: 5.227, 12: 5.290, 13: 5.348, 14: 5.400, 15: 5.448,
    }

    table = Q_005 if alpha == 0.05 else Q_001
    if k in table:
        return table[k]
    # Extrapolate for larger k
    if k <= 2:
        return table[2]
    return table[15] + (k - 15) * 0.05


# ── Wilcoxon Signed-Rank Test ────────────────────────────────────────────

def wilcoxon_signed_rank_test(method_a_results, method_b_results,
                               alternative='two-sided'):
    """
    Wilcoxon Signed-Rank Test for pairwise comparison of two methods.

    Tests whether the median difference between paired observations
    (same instance, two methods) is zero.

    Args:
        method_a_results: list of performance values for method A
                          (one value per instance)
        method_b_results: list of performance values for method B
                          (one value per instance, same order as A)
        alternative: 'two-sided' (default), 'greater' (A > B), or 'less' (A < B)

    Returns:
        dict with 'statistic', 'p_value', 'z_statistic', 'n_pairs',
        'n_nonzero', 'alternative', 'significant' (at alpha=0.05)
    """
    n = len(method_a_results)
    if n != len(method_b_results):
        raise ValueError(f"Unequal sample sizes: {len(method_a_results)} vs {len(method_b_results)}")
    if n < 5:
        return {
            'statistic': 0, 'p_value': 1.0, 'z_statistic': 0,
            'n_pairs': n, 'n_nonzero': 0, 'alternative': alternative,
            'significant': False,
            'warning': 'Sample size < 5; results unreliable'
        }

    # Compute differences
    diffs = [a - b for a, b in zip(method_a_results, method_b_results)]

    # Remove zero differences
    nonzero = [(i, d) for i, d in enumerate(diffs) if d != 0]
    n_nonzero = len(nonzero)

    if n_nonzero == 0:
        return {
            'statistic': 0, 'p_value': 1.0, 'z_statistic': 0,
            'n_pairs': n, 'n_nonzero': 0, 'alternative': alternative,
            'significant': False,
        }

    # Rank absolute differences
    abs_diffs = [(abs(d), d) for _, d in nonzero]
    abs_diffs.sort(key=lambda x: x[0])

    # Assign ranks (average for ties)
    ranks = [0.0] * n_nonzero
    i = 0
    while i < n_nonzero:
        j = i
        while j < n_nonzero and abs_diffs[j][0] == abs_diffs[i][0]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # +1 for 1-indexed, +1 for exclusive j
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    # Compute W+ (sum of ranks for positive differences)
    w_plus = sum(r for r, (_, d) in zip(ranks, abs_diffs) if d > 0)
    w_minus = sum(r for r, (_, d) in zip(ranks, abs_diffs) if d < 0)

    # Test statistic
    if alternative == 'two-sided':
        statistic = min(w_plus, w_minus)
    elif alternative == 'greater':
        statistic = w_minus  # H0: A <= B, test if A > B (fewer negative ranks)
    else:  # 'less'
        statistic = w_plus  # H0: A >= B, test if A < B

    # Normal approximation (with continuity correction)
    mean_w = n_nonzero * (n_nonzero + 1) / 4.0
    std_w = math.sqrt(n_nonzero * (n_nonzero + 1) * (2 * n_nonzero + 1) / 24.0)

    if std_w > 0:
        z = (w_plus - mean_w - 0.5 * (1 if w_plus > mean_w else -1)) / std_w
    else:
        z = 0.0

    # P-value
    if alternative == 'two-sided':
        p_value = 2.0 * min(_normal_cdf(z), 1.0 - _normal_cdf(z))
    elif alternative == 'greater':
        p_value = 1.0 - _normal_cdf(z)
    else:  # 'less'
        p_value = _normal_cdf(z)

    return {
        'statistic': statistic,
        'p_value': p_value,
        'z_statistic': z,
        'n_pairs': n,
        'n_nonzero': n_nonzero,
        'alternative': alternative,
        'significant': p_value < 0.05,
    }


# ── Friedman Test ────────────────────────────────────────────────────────

def friedman_test(results_matrix, method_names=None):
    """
    Friedman test for multiple method comparison across instances.

    Non-parametric equivalent of repeated-measures ANOVA.
    Null hypothesis: all methods have the same performance distribution.

    Args:
        results_matrix: list of lists, shape (n_instances, n_methods)
                        Each row = one instance, each col = one method
                        Lower values = better performance
        method_names: optional list of method names

    Returns:
        dict with 'statistic', 'p_value', 'df', 'n_instances', 'n_methods',
        'method_names', 'rankings', 'significant', 'critical_value'
    """
    n = len(results_matrix)      # instances
    k = len(results_matrix[0])   # methods

    if method_names is None:
        method_names = [f"Method_{i+1}" for i in range(k)]

    if n < 3:
        return {
            'statistic': 0, 'p_value': 1.0, 'df': k - 1,
            'n_instances': n, 'n_methods': k,
            'method_names': method_names,
            'rankings': None, 'significant': False,
            'warning': 'Sample size < 3; test unreliable',
        }

    # Compute ranks for each instance (1 = best = lowest value)
    all_ranks = []
    for row in results_matrix:
        # Sort indices by value
        sorted_pairs = sorted(enumerate(row), key=lambda x: x[1])
        ranks_row = [0.0] * k
        i = 0
        while i < k:
            j = i
            while j < k and sorted_pairs[j][1] == sorted_pairs[i][1]:
                j += 1
            avg_rank = (i + j + 2) / 2.0
            for idx in range(i, j):
                ranks_row[sorted_pairs[idx][0]] = avg_rank
            i = j
        all_ranks.append(ranks_row)

    # Average rank for each method
    avg_ranks = [0.0] * k
    for row in all_ranks:
        for j in range(k):
            avg_ranks[j] += row[j]
    avg_ranks = [r / n for r in avg_ranks]

    # Friedman statistic
    # Q = (12n / (k(k+1))) * sum(R_j^2) - 3n(k+1)
    r_squared_sum = sum(r * r for r in avg_ranks)
    statistic = (12.0 * n / (k * (k + 1))) * r_squared_sum - 3.0 * n * (k + 1)

    # Degrees of freedom
    df = k - 1

    # P-value from chi-square distribution
    p_value = 1.0 - _chi2_cdf(statistic, df)

    # Critical value at alpha=0.05
    # chi-square critical values for common df
    chi2_crit = {
        1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070,
        6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919, 10: 18.307,
    }
    critical_value = chi2_crit.get(df, df + 1.645 * math.sqrt(2 * df))

    return {
        'statistic': statistic,
        'p_value': p_value,
        'df': df,
        'n_instances': n,
        'n_methods': k,
        'method_names': method_names,
        'rankings': {name: rank for name, rank in zip(method_names, avg_ranks)},
        'significant': p_value < 0.05,
        'critical_value': critical_value,
    }


# ── Nemenyi Post-Hoc Test ────────────────────────────────────────────────

def friedman_nemenyi_posthoc(results_matrix, method_names=None, alpha=0.05):
    """
    Nemenyi post-hoc test after significant Friedman test.

    Computes critical difference (CD) and determines which pairs of
    methods are significantly different.

    CD = q_alpha * sqrt(k*(k+1) / (6*n))

    where q_alpha is the studentized range critical value.

    Args:
        results_matrix: same format as friedman_test
        method_names: optional list of method names
        alpha: significance level (default 0.05)

    Returns:
        dict with 'critical_difference', 'avg_ranks', 'method_names',
        'significant_pairs', 'cd_diagram_groups'
    """
    n = len(results_matrix)
    k = len(results_matrix[0])

    if method_names is None:
        method_names = [f"Method_{i+1}" for i in range(k)]

    # Compute average ranks (same as Friedman)
    all_ranks = []
    for row in results_matrix:
        sorted_pairs = sorted(enumerate(row), key=lambda x: x[1])
        ranks_row = [0.0] * k
        i = 0
        while i < k:
            j = i
            while j < k and sorted_pairs[j][1] == sorted_pairs[i][1]:
                j += 1
            avg_rank = (i + j + 2) / 2.0
            for idx in range(i, j):
                ranks_row[sorted_pairs[idx][0]] = avg_rank
            i = j
        all_ranks.append(ranks_row)

    avg_ranks = [sum(r[j] for r in all_ranks) / n for j in range(k)]

    # Critical difference
    q = _q_critical(alpha, k)
    cd = q * math.sqrt(k * (k + 1) / (6.0 * n))

    # Find significant pairs
    significant_pairs = []
    rank_order = sorted(enumerate(avg_ranks), key=lambda x: x[1])

    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(avg_ranks[i] - avg_ranks[j])
            if diff > cd:
                significant_pairs.append({
                    'method_a': method_names[i],
                    'method_b': method_names[j],
                    'rank_diff': diff,
                    'significant': True,
                })

    # Build CD diagram groups (methods not significantly different)
    # Simple greedy grouping
    sorted_by_rank = sorted(zip(method_names, avg_ranks), key=lambda x: x[1])
    groups = []
    i = 0
    while i < k:
        group = [sorted_by_rank[i][0]]
        j = i + 1
        while j < k:
            diff = sorted_by_rank[j][1] - sorted_by_rank[i][1]
            if diff <= cd:
                group.append(sorted_by_rank[j][0])
                j += 1
            else:
                break
        groups.append(group)
        i = j

    return {
        'critical_difference': cd,
        'avg_ranks': {name: rank for name, rank in zip(method_names, avg_ranks)},
        'method_names': method_names,
        'significant_pairs': significant_pairs,
        'cd_diagram_groups': groups,
        'alpha': alpha,
        'q_critical': q,
    }


# ── Method Rankings ──────────────────────────────────────────────────────

def compute_method_rankings(all_results, metrics=None):
    """
    Compute method rankings across all instances for multiple metrics.

    Args:
        all_results: list of dicts (output from experiment runner), each with:
            - 'instance_key': str
            - 'methods': dict mapping method_name → {'mean_cost': float, ...}
        metrics: list of metric keys to rank by (default: ['mean_cost', 'mean_tardiness'])

    Returns:
        dict with per-metric rankings and aggregate
    """
    if metrics is None:
        metrics = ['mean_cost', 'mean_tardiness']

    method_names = list(all_results[0]['methods'].keys())

    rankings = {}
    for metric in metrics:
        # Collect all instance-method values
        matrix = []
        for result in all_results:
            row = []
            for mname in method_names:
                val = result['methods'][mname].get(metric, float('inf'))
                row.append(val)
            matrix.append(row)

        # Run Friedman
        friedman_result = friedman_test(matrix, method_names)
        rankings[metric] = friedman_result

    # Aggregate ranking (average rank across metrics)
    if len(metrics) > 1:
        agg_ranks = {name: 0.0 for name in method_names}
        for metric, result in rankings.items():
            for name, rank in result['rankings'].items():
                agg_ranks[name] += rank
        for name in agg_ranks:
            agg_ranks[name] /= len(metrics)

        # Sort
        sorted_agg = sorted(agg_ranks.items(), key=lambda x: x[1])
        rankings['aggregate'] = {
            'rankings': dict(sorted_agg),
            'best_method': sorted_agg[0][0] if sorted_agg else None,
        }

    return rankings


# ── Print/Report Helpers ─────────────────────────────────────────────────

def print_wilcoxon_results(results, method_a, method_b):
    """Pretty-print Wilcoxon test results."""
    print(f"\n  Wilcoxon Signed-Rank Test: {method_a} vs {method_b}")
    print(f"  {'─' * 50}")
    print(f"    Statistic (W): {results['statistic']:.2f}")
    print(f"    Z-statistic:   {results['z_statistic']:.3f}")
    print(f"    P-value:       {results['p_value']:.4f}")
    print(f"    N (non-zero):  {results['n_nonzero']}/{results['n_pairs']}")
    print(f"    Significant:   {'YES ⭐' if results['significant'] else 'no'} "
          f"(α=0.05, {results['alternative']})")


def print_friedman_results(results):
    """Pretty-print Friedman test results."""
    print(f"\n  Friedman Test ({results['n_methods']} methods, "
          f"{results['n_instances']} instances)")
    print(f"  {'─' * 50}")
    print(f"    Statistic (Q): {results['statistic']:.2f}")
    print(f"    df:            {results['df']}")
    print(f"    P-value:       {results['p_value']:.4f}")
    print(f"    Significant:   {'YES ⭐' if results['significant'] else 'no'} "
          f"(α=0.05)")

    if results['rankings']:
        print(f"\n    Average Rankings (lower = better):")
        sorted_ranks = sorted(results['rankings'].items(), key=lambda x: x[1])
        for name, rank in sorted_ranks:
            bar = '█' * int(rank)
            print(f"      {name:<25s}  {rank:5.2f}  {bar}")


def print_nemenyi_results(results):
    """Pretty-print Nemenyi post-hoc test results."""
    print(f"\n  Nemenyi Post-Hoc Test (α={results['alpha']})")
    print(f"  {'─' * 50}")
    print(f"    Critical Difference (CD): {results['critical_difference']:.3f}")
    print(f"    Q-critical:               {results['q_critical']:.3f}")

    if results['significant_pairs']:
        print(f"\n    Significant Differences:")
        for pair in results['significant_pairs']:
            print(f"      {pair['method_a']} vs {pair['method_b']}: "
                  f"rank diff={pair['rank_diff']:.2f} > CD={results['critical_difference']:.2f}")
    else:
        print(f"\n    No significant pairwise differences found.")

    if results.get('cd_diagram_groups'):
        print(f"\n    CD Diagram Groups (methods connected if diff ≤ CD):")
        for i, group in enumerate(results['cd_diagram_groups']):
            print(f"      Group {i+1}: {', '.join(group)}")


# ── Full Statistical Report ──────────────────────────────────────────────

def full_statistical_report(all_results, output_json=None):
    """
    Generate a complete statistical report from experiment results.

    Args:
        all_results: list of experiment result dicts
        output_json: optional path to save JSON report

    Returns:
        dict with all test results
    """
    method_names = list(all_results[0]['methods'].keys())
    n_instances = len(all_results)

    # Build results matrices for cost and tardiness
    cost_matrix = []
    tard_matrix = []
    for result in all_results:
        cost_row = [result['methods'][m]['mean_cost'] for m in method_names]
        tard_row = [result['methods'][m]['mean_tardiness'] for m in method_names]
        cost_matrix.append(cost_row)
        tard_matrix.append(tard_row)

    report = {
        'n_instances': n_instances,
        'n_methods': len(method_names),
        'method_names': method_names,
    }

    # Friedman test
    report['friedman_cost'] = friedman_test(cost_matrix, method_names)
    report['friedman_tardiness'] = friedman_test(tard_matrix, method_names)

    # Nemenyi post-hoc (only if Friedman significant)
    if report['friedman_cost']['significant']:
        report['nemenyi_cost'] = friedman_nemenyi_posthoc(cost_matrix, method_names)
    if report['friedman_tardiness']['significant']:
        report['nemenyi_tardiness'] = friedman_nemenyi_posthoc(tard_matrix, method_names)

    # Pairwise Wilcoxon: our method vs each baseline
    our_method = 'w5_edd_full'  # our best method
    if our_method in method_names:
        our_idx = method_names.index(our_method)
        report['wilcoxon_cost'] = {}
        report['wilcoxon_tardiness'] = {}

        for mname in method_names:
            if mname == our_method:
                continue
            other_idx = method_names.index(mname)

            # Cost comparison (lower is better → test if ours < other)
            cost_ours = [row[our_idx] for row in cost_matrix]
            cost_other = [row[other_idx] for row in cost_matrix]
            report['wilcoxon_cost'][mname] = wilcoxon_signed_rank_test(
                cost_ours, cost_other, alternative='less')

            # Tardiness comparison
            tard_ours = [row[our_idx] for row in tard_matrix]
            tard_other = [row[other_idx] for row in tard_matrix]
            report['wilcoxon_tardiness'][mname] = wilcoxon_signed_rank_test(
                tard_ours, tard_other, alternative='less')

    if output_json:
        # Convert to serializable dict
        serializable = _make_serializable(report)
        with open(output_json, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"Statistical report saved to {output_json}")

    return report


def _make_serializable(obj):
    """Convert report dict to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, bool):
        return obj
    elif isinstance(obj, (int, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, 6)
    elif obj is None:
        return None
    return str(obj)
