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
    from src.experiments.statistical_tests import (
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

    # Rational approximation for the inverse normal CDF
    a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
    b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
    c = [0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
         0.0276438810333863, 0.0038405729373609, 0.0003951896511919,
         0.0000321767881768, 0.0000002888167364, 0.0000003960315187]

    if p < 0.5:
        y = math.sqrt(-2.0 * math.log(p))
    else:
        y = math.sqrt(-2.0 * math.log(1.0 - p))

    num = ((c[8] * y + c[7]) * y + c[6]) * y + c[5]
    num = ((num + c[4]) * y + c[3]) * y + c[2]
    num = num * y + c[1]
    num = num * y + c[0]
    num = num * y * y * y

    if p < 0.5:
        return -num
    return num


# ── Helper: Chi-Square Distribution ─────────────────────────────────────

def _chi2_cdf(x, df):
    """CDF of chi-square distribution with df degrees of freedom."""
    if x <= 0:
        return 0.0
    # Wilson-Hilferty approximation
    z = ((x / df) ** (1.0 / 3.0) - 1.0 + 2.0 / (9.0 * df)) / math.sqrt(2.0 / (9.0 * df))
    return _normal_cdf(z)


def _chi2_ppf(p, df):
    """Inverse CDF (percent point function) for chi-square."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return float('inf')

    # Wilson-Hilferty approximation inverted
    z = _normal_ppf(p)
    x = df * (1.0 - 2.0 / (9.0 * df) + z * math.sqrt(2.0 / (9.0 * df))) ** 3
    return max(0.0, x)


# ── Helper: Studentized Range Distribution ──────────────────────────────

def _studentized_range_ppf(p, k, df):
    """Approximate critical value for studentized range q distribution.

    Used for Nemenyi post-hoc test. Approximated via Copenhaver & Holland (1988).
    For df=inf (which is the case for Nemenyi), use normal approximation.
    """
    # For infinite df, use simplified approximation
    if df >= 1000:
        # Conservative Bonferroni-style approximation
        alpha = 1.0 - p
        alpha_adj = alpha / (k * (k - 1) / 2.0)
        z = _normal_ppf(1.0 - alpha_adj / 2.0)
        return z * math.sqrt(2.0)

    # General case: interpolation from tabulated values for common k
    # For k=2, q = t-value * sqrt(2)
    if k == 2:
        return _t_ppf(1.0 - (1.0 - p) / 2.0, df) * math.sqrt(2.0)

    # Approximate using tabulated studentized range values
    # Common values for alpha=0.05, df=inf
    q_table = {
        2: 2.772, 3: 3.314, 4: 3.633, 5: 3.858, 6: 4.030,
        7: 4.170, 8: 4.286, 9: 4.387, 10: 4.474,
        11: 4.552, 12: 4.622, 13: 4.685, 14: 4.743, 15: 4.796,
        16: 4.845, 17: 4.891, 18: 4.934, 19: 4.974, 20: 5.012,
    }

    if k in q_table:
        return q_table[k]

    # Extrapolate
    if k > 20:
        x1, x2 = 19, 20
        y1, y2 = q_table[19], q_table[20]
        slope = (y2 - y1) / (x2 - x1)
        return y2 + slope * (k - 20)

    return 5.0  # Fallback


def _t_ppf(p, df):
    """Approximate t-distribution percent point function."""
    if p <= 0.0:
        return -float('inf')
    if p >= 1.0:
        return float('inf')
    if df <= 0:
        return _normal_ppf(p)
    if df >= 1000:
        return _normal_ppf(p)

    z = _normal_ppf(p)
    # Use the formula from Abramowitz & Stegun 26.7.5
    z2 = z * z
    z3 = z2 * z
    z5 = z3 * z2
    z7 = z5 * z2

    t = z + (z3 + z) / (4.0 * df)
    t += (5.0 * z5 + 16.0 * z3 + 3.0 * z) / (96.0 * df * df)
    t += (3.0 * z7 + 19.0 * z5 + 17.0 * z3 - 15.0 * z) / (384.0 * df * df * df)

    return t


# ═══════════════════════════════════════════════════════════════════════════════
# Wilcoxon Signed-Rank Test
# ═══════════════════════════════════════════════════════════════════════════════

def wilcoxon_signed_rank_test(sample_a, sample_b, alternative='two-sided'):
    """
    Wilcoxon signed-rank test for paired samples.

    Tests whether the distribution of differences (a - b) is symmetric about zero.

    Args:
        sample_a: list of values for method A
        sample_b: list of values for method B (same length)
        alternative: 'two-sided' (default), 'less', or 'greater'

    Returns:
        dict with: statistic (W+), p_value, z_statistic, n_pairs,
                   median_diff, mean_diff, effect_size
    """
    n = len(sample_a)
    if n != len(sample_b):
        raise ValueError("Samples must have the same length")
    if n == 0:
        return {'statistic': 0, 'p_value': 1.0, 'n_pairs': 0}

    # Compute differences
    diffs = []
    for a, b in zip(sample_a, sample_b):
        d = a - b
        if abs(d) > 1e-12:  # Exclude zero differences
            diffs.append(d)

    n_nonzero = len(diffs)
    if n_nonzero == 0:
        return {'statistic': 0, 'p_value': 1.0, 'n_pairs': n, 'n_nonzero': 0}

    # Rank absolute differences
    abs_diffs = [abs(d) for d in diffs]
    # Sort and compute ranks (average ranks for ties)
    indexed = [(i, v) for i, v in enumerate(abs_diffs)]
    indexed.sort(key=lambda x: x[1])

    ranks = [0.0] * n_nonzero
    i = 0
    while i < n_nonzero:
        j = i
        while j < n_nonzero and abs(indexed[j][1] - indexed[i][1]) < 1e-12:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-based rank average
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j

    # Sum of ranks for positive differences (W+)
    w_plus = sum(ranks[i] for i in range(n_nonzero) if diffs[i] > 0)

    # Test statistic
    # For large n, use normal approximation
    mean_w = n_nonzero * (n_nonzero + 1.0) / 4.0
    # Variance corrected for ties
    tie_groups = {}
    for r in ranks:
        tie_groups[r] = tie_groups.get(r, 0) + 1
    var_w = n_nonzero * (n_nonzero + 1.0) * (2.0 * n_nonzero + 1.0) / 24.0
    for t in tie_groups.values():
        if t > 1:
            var_w -= (t * t * t - t) / 48.0
    var_w = max(var_w, 1e-12)

    z = (w_plus - mean_w) / math.sqrt(var_w)

    # P-value
    if alternative == 'two-sided':
        p_value = 2.0 * (1.0 - _normal_cdf(abs(z)))
    elif alternative == 'less':
        p_value = _normal_cdf(z)
    elif alternative == 'greater':
        p_value = 1.0 - _normal_cdf(z)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    # Effect size: r = z / sqrt(n)
    effect_size = abs(z) / math.sqrt(n) if n > 0 else 0.0

    return {
        'statistic': w_plus,
        'p_value': p_value,
        'z_statistic': z,
        'n_pairs': n,
        'n_nonzero': n_nonzero,
        'mean_diff': sum(diffs) / n_nonzero if n_nonzero > 0 else 0.0,
        'median_diff': sorted(diffs)[len(diffs) // 2] if diffs else 0.0,
        'effect_size': effect_size,
        'significant_05': p_value < 0.05,
        'significant_01': p_value < 0.01,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Friedman Test
# ═══════════════════════════════════════════════════════════════════════════════

def friedman_test(method_scores):
    """
    Friedman test for comparing k methods across n instances.

    Tests the null hypothesis that all methods have the same performance.

    Args:
        method_scores: list of lists, shape (n_instances, k_methods)
                       Each inner list contains scores for all methods on one instance.

    Returns:
        dict with: statistic (chi-squared), p_value, df, k (n_methods),
                   N (n_instances), method_ranks
    """
    N = len(method_scores)
    if N == 0:
        return {'statistic': 0, 'p_value': 1.0, 'df': 0, 'k': 0, 'N': 0}

    k = len(method_scores[0])
    if k < 2:
        return {'statistic': 0, 'p_value': 1.0, 'df': 0, 'k': k, 'N': N}

    # Compute ranks per instance (1 = best, k = worst)
    ranks = []
    for row in method_scores:
        # Sort indices by scores (smaller score = better rank)
        indexed = [(i, v) for i, v in enumerate(row)]
        indexed.sort(key=lambda x: x[1])

        # Assign ranks, handling ties (average ranks)
        inst_ranks = [0.0] * k
        i = 0
        while i < k:
            j = i
            while j < k and abs(indexed[j][1] - indexed[i][1]) < 1e-12:
                j += 1
            avg_rank = (i + j + 1) / 2.0  # 1-based rank average
            for m in range(i, j):
                inst_ranks[indexed[m][0]] = avg_rank
            i = j
        ranks.append(inst_ranks)

    # Average rank per method
    avg_ranks = [0.0] * k
    for j in range(k):
        avg_ranks[j] = sum(ranks[i][j] for i in range(N)) / N

    # Friedman statistic
    R_sum_sq = sum((sum(ranks[i][j] for i in range(N))) ** 2 for j in range(k))

    # Chi-squared = 12 * N / (k * (k+1)) * (sum(R_j^2) - k * (k+1)^2 / 4)
    stat = (12.0 * N) / (k * (k + 1.0)) * (R_sum_sq / (N * N) - N * k * (k + 1.0) ** 2 / 4.0)

    # Correction for ties
    tie_sum = 0.0
    for i in range(N):
        for j in range(k):
            tie_sum += ranks[i][j] ** 2
    expected_tie_sum = N * k * (k + 1.0) * (2.0 * k + 1.0) / 6.0
    if abs(expected_tie_sum - tie_sum) > 1e-12:
        correction = 1.0 - (tie_sum - expected_tie_sum) / (N * (k - 1.0) * expected_tie_sum)
        if correction > 1e-12:
            stat /= correction

    df = k - 1
    p_value = 1.0 - _chi2_cdf(stat, df) if stat > 0 else 1.0

    return {
        'statistic': stat,
        'p_value': p_value,
        'df': df,
        'k': k,
        'N': N,
        'method_ranks': avg_ranks,
        'significant_05': p_value < 0.05,
        'significant_01': p_value < 0.01,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Nemenyi Post-Hoc Test
# ═══════════════════════════════════════════════════════════════════════════════

def friedman_nemenyi_posthoc(method_scores, alpha=0.05):
    """
    Nemenyi post-hoc test after Friedman test.

    Two methods are significantly different if their average rank difference
    exceeds the critical difference (CD).

    Args:
        method_scores: same format as friedman_test
        alpha: significance level (default 0.05)

    Returns:
        dict with: cd (critical difference), k, N, method_ranks,
                   significant_pairs (list of (i, j, diff, significant))
    """
    N = len(method_scores)
    if N == 0:
        return {'cd': 0, 'k': 0, 'N': 0, 'significant_pairs': []}

    k = len(method_scores[0])

    # Get ranks from Friedman
    friedman_result = friedman_test(method_scores)
    avg_ranks = friedman_result['method_ranks']

    # Critical difference: CD = q_alpha * sqrt(k*(k+1)/(6*N))
    # q_alpha from studentized range distribution with k groups, df=inf
    q_alpha = _studentized_range_ppf(1.0 - alpha, k, float('inf'))
    cd = q_alpha * math.sqrt(k * (k + 1.0) / (6.0 * N))

    # Find significant pairs
    significant_pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(avg_ranks[i] - avg_ranks[j])
            sig = diff > cd + 1e-12
            if sig or diff > 0.5 * cd:  # Report marginal ones too
                significant_pairs.append({
                    'i': i, 'j': j,
                    'rank_diff': diff,
                    'critical_diff': cd,
                    'significant': sig,
                    'marginal': not sig and diff > 0.3 * cd,
                })

    return {
        'cd': cd,
        'k': k,
        'N': N,
        'alpha': alpha,
        'q_alpha': q_alpha,
        'method_ranks': avg_ranks,
        'significant_pairs': significant_pairs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Method Rankings
# ═══════════════════════════════════════════════════════════════════════════════

def compute_method_rankings(method_scores, instance_names=None, method_names=None):
    """
    Compute per-instance and aggregate rankings for methods.

    Args:
        method_scores: list of lists, shape (n_instances, k_methods)
        instance_names: optional list of instance names
        method_names: optional list of method names

    Returns:
        dict with rankings table
    """
    N = len(method_scores)
    if N == 0:
        return {}

    k = len(method_scores[0])

    if method_names is None:
        method_names = [f'Method_{i}' for i in range(k)]
    if instance_names is None:
        instance_names = [f'Instance_{i}' for i in range(N)]

    per_instance = []
    for i, row in enumerate(method_scores):
        indexed = [(j, v) for j, v in enumerate(row)]
        indexed.sort(key=lambda x: x[1])
        ranks = [0] * k
        for rank, (j, v) in enumerate(indexed):
            ranks[j] = rank + 1
        per_instance.append({
            'instance': instance_names[i],
            'ranks': ranks,
            'best_method': method_names[indexed[0][0]],
            'best_score': indexed[0][1],
            'worst_method': method_names[indexed[-1][0]],
            'worst_score': indexed[-1][1],
        })

    avg_ranks = friedman_test(method_scores)['method_ranks']
    ranked_methods = sorted(
        [(method_names[i], avg_ranks[i]) for i in range(k)],
        key=lambda x: x[1]
    )

    return {
        'per_instance': per_instance,
        'avg_ranks': {name: rank for name, rank in ranked_methods},
        'ranked_methods': ranked_methods,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Full Report
# ═══════════════════════════════════════════════════════════════════════════════

def full_statistical_report(method_scores, instance_names=None, method_names=None):
    """
    Run full statistical analysis and return a comprehensive report.

    Args:
        method_scores: list of lists (instances x methods)
        instance_names: optional labels
        method_names: optional labels

    Returns:
        dict with friedman, nemenyi, pairwise_wilcoxon, rankings
    """
    N = len(method_scores)
    if N == 0:
        return {}

    k = len(method_scores[0])

    if method_names is None:
        method_names = [f'M{i}' for i in range(k)]

    # Friedman
    fried = friedman_test(method_scores)
    fried['method_names'] = method_names

    # Nemenyi
    nem = friedman_nemenyi_posthoc(method_scores, alpha=0.05)
    nem['method_names'] = method_names

    # Pairwise Wilcoxon (all vs all)
    wilcoxon_results = {}
    for i in range(k):
        for j in range(i + 1, k):
            a_scores = [row[i] for row in method_scores]
            b_scores = [row[j] for row in method_scores]
            key = f'{method_names[i]}_vs_{method_names[j]}'
            wilcoxon_results[key] = wilcoxon_signed_rank_test(a_scores, b_scores)

    # Rankings
    rankings = compute_method_rankings(method_scores, instance_names, method_names)

    return {
        'friedman': fried,
        'nemenyi': nem,
        'pairwise_wilcoxon': wilcoxon_results,
        'rankings': rankings,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Pretty Printing
# ═══════════════════════════════════════════════════════════════════════════════

def print_wilcoxon_results(results, method_a='A', method_b='B'):
    """Print formatted Wilcoxon test results."""
    print(f"\n  Wilcoxon Signed-Rank Test: {method_a} vs {method_b}")
    print(f"  {'-'*50}")
    print(f"    W+ statistic:     {results['statistic']:.2f}")
    print(f"    Z statistic:      {results['z_statistic']:.3f}")
    print(f"    p-value:          {results['p_value']:.4f}")
    print(f"    Mean diff:        {results['mean_diff']:.3f}")
    print(f"    Median diff:      {results['median_diff']:.3f}")
    print(f"    Effect size (r):  {results['effect_size']:.3f}")
    print(f"    N pairs:          {results['n_pairs']} ({results['n_nonzero']} non-zero)")
    sig = "SIGNIFICANT" if results['significant_05'] else "not significant"
    print(f"    Result:           {sig} at alpha=0.05")
    if results['significant_01']:
        print(f"                      SIGNIFICANT at alpha=0.01")


def print_friedman_results(results, method_names=None):
    """Print formatted Friedman test results."""
    print(f"\n  Friedman Test ({results['k']} methods, {results['N']} instances)")
    print(f"  {'-'*60}")
    print(f"    Chi-squared:      {results['statistic']:.3f}")
    print(f"    df:               {results['df']}")
    print(f"    p-value:          {results['p_value']:.6f}")
    sig = "SIGNIFICANT" if results['significant_05'] else "not significant"
    print(f"    Result:           {sig} at alpha=0.05")

    print(f"\n    Average Ranks:")
    ranks = list(enumerate(results['method_ranks']))
    ranks.sort(key=lambda x: x[1])
    for i, (method_idx, avg_rank) in enumerate(ranks):
        name = method_names[method_idx] if method_names else f"M{method_idx}"
        print(f"      {i+1}. {name:<20s} {avg_rank:.2f}")


def print_nemenyi_results(results, method_names=None):
    """Print formatted Nemenyi post-hoc results."""
    print(f"\n  Nemenyi Post-Hoc Test (alpha={results['alpha']})")
    print(f"  {'-'*60}")
    print(f"    Critical Difference (CD): {results['cd']:.3f}")
    print(f"    q_alpha:                 {results['q_alpha']:.3f}")

    if not results['significant_pairs']:
        print(f"    No significant pairwise differences found.")
        return

    print(f"\n    Significant/marginal pairwise differences:")
    for pair in results['significant_pairs']:
        i, j = pair['i'], pair['j']
        name_i = method_names[i] if method_names else f"M{i}"
        name_j = method_names[j] if method_names else f"M{j}"
        label = "***" if pair['significant'] else "(marginal)"
        print(f"      {name_i:<20s} vs {name_j:<20s} "
              f"diff={pair['rank_diff']:.3f} (CD={pair['critical_diff']:.3f}) {label}")
