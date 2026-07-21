#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 6 P1: Evaluate Meta-Learner vs Hybrid Rule vs Oracle.

Trains on all training data (leave-one-out CV), then compares:
  1. Meta-learner (KNN) prediction accuracy
  2. Hybrid rule (RC1→adaptive_tw_drone, RC2→tw_aware_drone) accuracy
  3. Feature importance: which instance features drive variant selection?

Usage:
    python evaluate_meta.py                              # Use latest training data
    python evaluate_meta.py --data results/meta_training_XXXX.json
"""

import json, os, sys, math
import numpy as np

_W6 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _W6)

from meta_learner import (
    MetaLearner, KNNClassifier, hybrid_rule,
    extract_features, label_best_variant,
    FEATURE_NAMES, ALL_VARIANTS, VARIANT_NAMES,
    permutation_importance,
)

# ── Matplotlib setup ──
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_latest_training_data():
    """Find most recent meta training data file."""
    results_dir = os.path.join(_W6, 'results')
    files = sorted([f for f in os.listdir(results_dir)
                    if f.startswith('meta_training_') and f.endswith('.json')])
    if not files:
        raise FileNotFoundError("No meta_training_*.json found in results/")
    path = os.path.join(results_dir, files[-1])
    with open(path) as f:
        return json.load(f), path


def evaluate_k_values(training_data, k_values=None):
    """Test different K values with LOOCV."""
    if k_values is None:
        k_values = [1, 3, 5]

    instances = []
    labels_tardiness = []
    labels_cost = []

    for d in training_data:
        # Use features from training data
        feats = np.array(d['features'])
        instances.append(feats)
        labels_tardiness.append(d['best_variant_tardiness'])
        labels_cost.append(d['best_variant_cost'])

    print(f'Dataset: {len(instances)} instances')
    print(f'Unique best variants (tardiness): {set(labels_tardiness)}')
    print(f'Unique best variants (cost): {set(labels_cost)}')
    print()

    results = {}
    for k in k_values:
        knn = KNNClassifier(k=k)
        knn.fit(instances, labels_tardiness)

        correct = 0
        for i in range(len(instances)):
            X_loo = [x for j, x in enumerate(instances) if j != i]
            y_loo = [y for j, y in enumerate(labels_tardiness) if j != i]
            loo_knn = KNNClassifier(k=k)
            loo_knn.fit(X_loo, y_loo)
            pred = loo_knn.predict(instances[i])
            if pred == labels_tardiness[i]:
                correct += 1

        acc = correct / len(instances)
        results[k] = acc
        print(f'K={k}: LOOCV accuracy = {correct}/{len(instances)} = {acc:.1%}')

    return results


def compare_strategies(training_data, k=3):
    """
    Compare three strategies:
      1. Meta-learner (KNN)
      2. Hybrid rule (hard-coded)
      3. Oracle (always picks best)

    Uses leave-one-out to be fair: for each test instance, train on the
    other 11, predict the held-out one.
    """
    n = len(training_data)
    instances = [np.array(d['features']) for d in training_data]
    labels = [d['best_variant_tardiness'] for d in training_data]
    hybrid_picks = [d['hybrid_rule_pick'] for d in training_data]

    results = {
        'meta_correct': 0,
        'hybrid_correct': 0,
        'meta_worse_than_hybrid': 0,
        'meta_better_than_hybrid': 0,
        'details': [],
    }

    for i in range(n):
        # Leave one out
        X_train = [x for j, x in enumerate(instances) if j != i]
        y_train = [y for j, y in enumerate(labels) if j != i]

        knn = KNNClassifier(k=k)
        knn.fit(X_train, y_train)

        meta_pred = knn.predict(instances[i])
        hybrid_pred = hybrid_picks[i]
        oracle = labels[i]

        meta_correct = (meta_pred == oracle)
        hybrid_correct = (hybrid_pred == oracle)

        if meta_correct:
            results['meta_correct'] += 1
        if hybrid_correct:
            results['hybrid_correct'] += 1

        if meta_correct and not hybrid_correct:
            results['meta_better_than_hybrid'] += 1
        elif hybrid_correct and not meta_correct:
            results['meta_worse_than_hybrid'] += 1

        # Get cost/tardiness of each pick
        d = training_data[i]
        meta_metrics = d['variant_results'].get(meta_pred, {})
        hybrid_metrics = d['variant_results'].get(hybrid_pred, {})
        oracle_metrics = d['variant_results'].get(oracle, {})

        results['details'].append({
            'instance': d['instance_key'],
            'oracle': oracle,
            'meta_pred': meta_pred,
            'hybrid_pred': hybrid_pred,
            'meta_correct': meta_correct,
            'hybrid_correct': hybrid_correct,
            'oracle_cost': oracle_metrics.get('mean_cost', 0),
            'oracle_tard': oracle_metrics.get('mean_tardiness', 0),
            'meta_cost': meta_metrics.get('mean_cost', 0),
            'meta_tard': meta_metrics.get('mean_tardiness', 0),
            'hybrid_cost': hybrid_metrics.get('mean_cost', 0),
            'hybrid_tard': hybrid_metrics.get('mean_tardiness', 0),
        })

    return results


def compute_cost_penalty(results):
    """
    Compute the cost penalty of using meta-learner vs oracle.
    How much worse is the meta-learner's pick compared to the best possible?
    """
    penalties = []
    for d in results['details']:
        oracle_cost = d['oracle_cost']
        meta_cost = d['meta_cost']
        hybrid_cost = d['hybrid_cost']

        if oracle_cost > 0:
            meta_penalty = (meta_cost - oracle_cost) / oracle_cost * 100
            hybrid_penalty = (hybrid_cost - oracle_cost) / oracle_cost * 100
            penalties.append({
                'instance': d['instance'],
                'meta_penalty_pct': meta_penalty,
                'hybrid_penalty_pct': hybrid_penalty,
                'meta_tard_penalty': d['meta_tard'] - d['oracle_tard'],
                'hybrid_tard_penalty': d['hybrid_tard'] - d['oracle_tard'],
            })

    avg_meta_penalty = np.mean([p['meta_penalty_pct'] for p in penalties])
    avg_hybrid_penalty = np.mean([p['hybrid_penalty_pct'] for p in penalties])
    avg_meta_tard = np.mean([p['meta_tard_penalty'] for p in penalties])
    avg_hybrid_tard = np.mean([p['hybrid_tard_penalty'] for p in penalties])

    return {
        'avg_cost_penalty': {'meta': avg_meta_penalty, 'hybrid': avg_hybrid_penalty},
        'avg_tard_penalty': {'meta': avg_meta_tard, 'hybrid': avg_hybrid_tard},
        'per_instance': penalties,
    }


def plot_feature_importance(training_data, k=3):
    """Plot permutation feature importance."""
    if not HAS_MPL:
        print('Matplotlib not available, skipping plot.')
        return

    instances = [np.array(d['features']) for d in training_data]
    labels = [d['best_variant_tardiness'] for d in training_data]

    knn = KNNClassifier(k=k)
    knn.fit(instances, labels)

    importances = permutation_importance(knn, instances, labels, n_repeats=20)

    # Sort by importance
    sorted_imp = sorted(importances.items(), key=lambda x: x[1][0], reverse=True)

    names = [x[0] for x in sorted_imp]
    means = [x[1][0] for x in sorted_imp]
    stds = [x[1][1] for x in sorted_imp]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#27AE60' if m > 0 else '#E74C3C' for m in means]
    ax.barh(range(len(names)), means, xerr=stds, color=colors, alpha=0.8,
            capsize=3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Accuracy Drop When Permuted')
    ax.set_title(f'Feature Importance for Variant Selection (K={k})')
    plt.tight_layout()

    viz_dir = os.path.join(_W6, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    path = os.path.join(viz_dir, 'meta_feature_importance.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def plot_confusion_summary(training_data):
    """Plot which variants get confused with which."""
    if not HAS_MPL:
        return

    labels = [d['best_variant_tardiness'] for d in training_data]
    from collections import Counter
    counts = Counter(labels)

    fig, ax = plt.subplots(figsize=(10, 5))
    variants = sorted(counts.keys())
    values = [counts[v] for v in variants]
    colors = ['#27AE60' if 'drone' in v else '#3498DB' for v in variants]
    ax.bar(variants, values, color=colors, alpha=0.8)
    ax.set_ylabel('Number of Instances')
    ax.set_title('Best Variant Distribution Across Training Instances')
    ax.tick_params(axis='x', labelrotation=45)
    for i, v in enumerate(values):
        ax.annotate(str(v), (i, v), textcoords="offset points",
                   xytext=(0, 5), ha='center', fontweight='bold')
    plt.tight_layout()

    viz_dir = os.path.join(_W6, 'visualizations')
    path = os.path.join(viz_dir, 'meta_variant_distribution.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def plot_strategy_comparison(cmp_results):
    """Plot meta-learner vs hybrid rule performance."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Accuracy
    ax = axes[0]
    n = len(cmp_results['details'])
    strategies = ['Meta-Learner\n(KNN)', 'Hybrid\nRule']
    correct = [cmp_results['meta_correct'], cmp_results['hybrid_correct']]
    incorrect = [n - c for c in correct]
    x = np.arange(len(strategies))
    ax.bar(x, correct, 0.4, label='Correct', color='#27AE60', alpha=0.8)
    ax.bar(x, incorrect, 0.4, bottom=correct, label='Wrong', color='#E74C3C', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylabel('Instances')
    ax.set_title(f'Prediction Accuracy (n={n})')
    ax.legend()
    for i, (c, tot) in enumerate(zip(correct, [n, n])):
        ax.annotate(f'{c}/{tot}\n({c/tot*100:.0f}%)', (i, c/2),
                   ha='center', fontweight='bold', fontsize=11, color='white')

    # Panel 2: Cost penalty vs oracle
    ax = axes[1]
    penalties = compute_cost_penalty(cmp_results)
    ax.bar(['Meta-Learner', 'Hybrid Rule'],
           [penalties['avg_cost_penalty']['meta'],
            penalties['avg_cost_penalty']['hybrid']],
           color=['#3498DB', '#F39C12'], alpha=0.8)
    ax.set_ylabel('Avg Cost Increase vs Oracle (%)')
    ax.set_title('Cost Penalty vs Oracle')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Week 6 P1: Meta-Learner vs Hybrid Rule Comparison',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    viz_dir = os.path.join(_W6, 'visualizations')
    path = os.path.join(viz_dir, 'meta_strategy_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def print_detailed_report(cmp_results, penalties, k=3):
    """Print comprehensive evaluation report."""
    n = len(cmp_results['details'])

    print(f'\n{"="*70}')
    print(f'P1 META-LEARNER EVALUATION REPORT')
    print(f'{"="*70}')
    print(f'Instances: {n}')
    print(f'KNN k-value: {k}')
    print()

    # ── Accuracy ──
    print(f'--- Accuracy ---')
    print(f'  Meta-learner:  {cmp_results["meta_correct"]}/{n} '
          f'({cmp_results["meta_correct"]/n*100:.1f}%)')
    print(f'  Hybrid rule:   {cmp_results["hybrid_correct"]}/{n} '
          f'({cmp_results["hybrid_correct"]/n*100:.1f}%)')
    print(f'  Meta better than hybrid: {cmp_results["meta_better_than_hybrid"]} instances')
    print(f'  Hybrid better than meta: {cmp_results["meta_worse_than_hybrid"]} instances')
    print()

    # ── Cost Penalty ──
    print(f'--- Cost Penalty vs Oracle ---')
    print(f'  Meta-learner avg:  {penalties["avg_cost_penalty"]["meta"]:+.1f}%')
    print(f'  Hybrid rule avg:   {penalties["avg_cost_penalty"]["hybrid"]:+.1f}%')
    print(f'  Meta tard penalty:  {penalties["avg_tard_penalty"]["meta"]:+.1f}')
    print(f'  Hybrid tard penalty: {penalties["avg_tard_penalty"]["hybrid"]:+.1f}')
    print()

    # ── Per-instance detail ──
    print(f'--- Per-Instance Detail ---')
    header = f'  {"Instance":<16s} {"Oracle":<22s} {"Meta":<22s} {"Hybrid":<22s} {"Meta✓":>5s} {"Hyb✓":>5s}'
    print(header)
    print('  ' + '-' * 100)
    for d in cmp_results['details']:
        print(f'  {d["instance"]:<16s} {d["oracle"]:<22s} {d["meta_pred"]:<22s} '
              f'{d["hybrid_pred"]:<22s} {"✓" if d["meta_correct"] else "✗":>5s} '
              f'{"✓" if d["hybrid_correct"] else "✗":>5s}')
    print()

    # ── Cost comparison ──
    print(f'--- Cost Comparison ---')
    print(f'  {"Instance":<16s} {"Oracle Cost":>10s} {"Meta Cost":>10s} {"Hybrid Cost":>10s} '
          f'{"Meta Δ%":>8s} {"Hybrid Δ%":>8s}')
    print('  ' + '-' * 70)
    for d in cmp_results['details']:
        oc = d['oracle_cost']
        mc = d['meta_cost']
        hc = d['hybrid_cost']
        md = (mc - oc) / max(oc, 1) * 100
        hd = (hc - oc) / max(oc, 1) * 100
        print(f'  {d["instance"]:<16s} {oc:>10.0f} {mc:>10.0f} {hc:>10.0f} '
              f'{md:>+7.1f}% {hd:>+7.1f}%')

    # ── Key takeaways ──
    print(f'\n{"="*70}')
    print('KEY TAKEAWAYS')
    print(f'{"="*70}')

    if cmp_results['meta_correct'] == cmp_results['hybrid_correct']:
        print('Meta-learner matches hybrid rule performance.')
        print('The hard-coded rule (hybrid_drone for all) is already optimal.')
        print('This validates the W5 hybrid strategy design.')
    elif cmp_results['meta_correct'] > cmp_results['hybrid_correct']:
        diff = cmp_results['meta_correct'] - cmp_results['hybrid_correct']
        print(f'Meta-learner identifies {diff} more correct variants than hybrid rule.')
        print('Instance-specific features provide additional signal beyond TW type alone.')
    else:
        diff = cmp_results['hybrid_correct'] - cmp_results['meta_correct']
        print(f'Hybrid rule outperforms meta-learner by {diff} instances.')
        print('With only {n} training instances, KNN lacks sufficient data.')
        print('More instances or different features may help.')

    if penalties['avg_cost_penalty']['meta'] < penalties['avg_cost_penalty']['hybrid']:
        print(f'Meta-learner reduces cost penalty by '
              f'{penalties["avg_cost_penalty"]["hybrid"] - penalties["avg_cost_penalty"]["meta"]:.1f}% '
              f'vs hybrid rule.')
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None,
                       help='Path to training data JSON')
    parser.add_argument('--k', type=int, default=3,
                       help='KNN k-value (default: 3)')
    parser.add_argument('--quick', action='store_true',
                       help='Skip plots, text-only report')
    args = parser.parse_args()

    # Load data
    if args.data:
        with open(args.data) as f:
            training_data = json.load(f)
        print(f'Loaded: {args.data}')
    else:
        training_data, path = load_latest_training_data()
        print(f'Loaded: {path}')

    print(f'{len(training_data)} training instances\n')

    # 1. Evaluate K values
    print('--- LOOCV Accuracy by K ---')
    k_results = evaluate_k_values(training_data, k_values=[1, 3, 5])

    # 2. Compare strategies
    best_k = args.k
    print(f'\n--- Strategy Comparison (K={best_k}) ---')
    cmp_results = compare_strategies(training_data, k=best_k)
    penalties = compute_cost_penalty(cmp_results)

    # 3. Report
    print_detailed_report(cmp_results, penalties, k=best_k)

    # 4. Visualizations
    if not args.quick:
        print('--- Generating Visualizations ---')
        plot_feature_importance(training_data, k=best_k)
        plot_confusion_summary(training_data)
        plot_strategy_comparison(cmp_results)

    print('\nDone!')


if __name__ == '__main__':
    main()
