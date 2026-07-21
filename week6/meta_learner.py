# -*- coding: utf-8 -*-
"""
Week 6 P1: Meta-Learner for W5 Variant Selection.

Given instance features, predict which W5 variant will perform best
(lowest tardiness, then lowest cost as tiebreaker).

Uses KNN classifier (no sklearn dependency) with feature importance
analysis via permutation test.

Usage:
    from meta_learner import MetaLearner
    ml = MetaLearner()
    ml.fit(features_list, labels_list)
    best_variant = ml.predict(instance)
"""

import math
import numpy as np
from collections import Counter

# All W5 variants the meta-learner can choose from
ALL_VARIANTS = [
    'baseline',
    'tw_aware',
    'drone_only',
    'tw_aware_drone',
    'adaptive_tw',
    'adaptive_tw_drone',
    'angle',
    'angle_drone',
    'hybrid',
    'hybrid_drone',
]

# Human-readable variant names for reporting
VARIANT_NAMES = {
    'baseline': 'Spatial (no drone)',
    'tw_aware': 'TW-Aware (no drone)',
    'drone_only': 'Spatial + Drone',
    'tw_aware_drone': 'TW-Aware + Drone',
    'adaptive_tw': 'Adaptive TW (no drone)',
    'adaptive_tw_drone': 'Adaptive TW + Drone',
    'angle': 'Angle Petal (no drone)',
    'angle_drone': 'Angle Petal + Drone',
    'hybrid': 'Hybrid (no drone)',
    'hybrid_drone': 'Hybrid + Drone',
}

# Feature names in order (must match extract_features output)
FEATURE_NAMES = [
    'n_customers',
    'tw_type_rc2',         # binary: 0=RC1, 1=RC2
    'tw_horizon',
    'spatial_spread_x',
    'spatial_spread_y',
    'spatial_radius',
    'avg_tw_width',
    'tw_density',
    'tw_spread',
    'tw_overlap_ratio',
    'demand_total',
    'demand_mean',
    'demand_std',
    'avg_service_time',
    'n_trucks_needed',
    'max_tw_gap',
    'avg_tw_gap',
    'customer_density',
]


def extract_features(instance):
    """
    Extract 18 numerical features from a problem instance.

    Args:
        instance: dict with 'customers', 'depot', 'tw_type', 'tw_horizon'

    Returns:
        np.ndarray of shape (18,) — normalized feature vector
    """
    customers = instance['customers']
    n = len(customers)
    depot = instance['depot']

    xs = np.array([c['x'] for c in customers])
    ys = np.array([c['y'] for c in customers])
    ready_times = np.array([c['ready_time'] for c in customers])
    due_times = np.array([c['due_time'] for c in customers])
    tw_widths = due_times - ready_times
    tw_midpoints = (ready_times + due_times) / 2.0
    demands = np.array([c['demand'] for c in customers])
    service_times = np.array([c['service_time'] for c in customers])

    tw_horizon = instance.get('tw_horizon', 120.0 if instance.get('tw_type') == 'RC1' else 240.0)
    tw_type = instance.get('tw_type', 'RC1')

    # Spatial features
    spatial_spread_x = float(np.std(xs))
    spatial_spread_y = float(np.std(ys))
    dists_from_depot = np.sqrt((xs - depot[0])**2 + (ys - depot[1])**2)
    spatial_radius = float(np.max(dists_from_depot))

    # Temporal features
    avg_tw_width = float(np.mean(tw_widths))
    tw_density = avg_tw_width / max(tw_horizon, 1.0)
    tw_spread = float(np.max(due_times) - np.min(ready_times))

    # TW overlap: for each customer, fraction of others whose TW overlaps with it
    overlap_counts = []
    for i in range(n):
        # Customer j's TW overlaps with i if ready_j <= due_i AND ready_i <= due_j
        overlaps = np.sum(
            (ready_times <= due_times[i]) & (due_times >= ready_times[i])
        )
        overlap_counts.append((overlaps - 1) / max(n - 1, 1))  # exclude self
    tw_overlap_ratio = float(np.mean(overlap_counts))

    # Demand features
    demand_total = float(np.sum(demands))
    demand_mean = float(np.mean(demands))
    demand_std = float(np.std(demands))

    # Service time
    avg_service_time = float(np.mean(service_times))

    # Fleet sizing
    n_trucks_needed = max(1, math.ceil(demand_total / 200.0))

    # TW gap features (sorted by midpoint)
    sorted_mids = np.sort(tw_midpoints)
    gaps = np.diff(sorted_mids) if len(sorted_mids) > 1 else np.array([0.0])
    max_tw_gap = float(np.max(gaps))
    avg_tw_gap = float(np.mean(gaps))

    # Customer density (customers per unit area, approximate)
    area = max(spatial_spread_x * spatial_spread_y * 4.0, 1.0)  # ~2σ × 2σ bounding box
    customer_density = n / area

    feats = np.array([
        n,
        1.0 if tw_type == 'RC2' else 0.0,
        tw_horizon,
        spatial_spread_x,
        spatial_spread_y,
        spatial_radius,
        avg_tw_width,
        tw_density,
        tw_spread,
        tw_overlap_ratio,
        demand_total,
        demand_mean,
        demand_std,
        avg_service_time,
        n_trucks_needed,
        max_tw_gap,
        avg_tw_gap,
        customer_density,
    ], dtype=np.float64)

    return feats


def label_best_variant(variant_results, prefer='tardiness'):
    """
    Given results for all variants on one instance, pick the best.

    Args:
        variant_results: dict mapping variant_name → dict with keys
                        'mean_cost', 'mean_tardiness', 'feasibility_rate'
        prefer: 'tardiness' (default) or 'cost'

    Returns:
        best variant name (str)
    """
    # Filter: only consider feasible variants
    feasible = {
        v: r for v, r in variant_results.items()
        if r.get('feasibility_rate', 0) >= 0.5
    }

    if not feasible:
        # If no variant is feasible, pick lowest tardiness
        return min(variant_results, key=lambda v: variant_results[v].get('mean_tardiness', 1e9))

    if prefer == 'tardiness':
        # Primary: lowest tardiness, secondary: lowest cost
        best = min(feasible, key=lambda v: (
            feasible[v].get('mean_tardiness', 1e9),
            feasible[v].get('mean_cost', 1e9),
        ))
    else:
        # Primary: lowest cost among those with acceptable tardiness
        min_tard = min(r.get('mean_tardiness', 1e9) for r in feasible.values())
        threshold = min_tard * 1.1 + 10.0  # within 10% or 10 units of best
        acceptable = {
            v: r for v, r in feasible.items()
            if r.get('mean_tardiness', 1e9) <= threshold
        }
        best = min(acceptable, key=lambda v: acceptable[v].get('mean_cost', 1e9))

    return best


def hybrid_rule(instance):
    """
    Current hard-coded rule from W5 hybrid strategy.

    Returns:
        variant name (str)
    """
    tw_type = instance.get('tw_type', 'RC1')
    if tw_type == 'RC1':
        return 'hybrid_drone'  # delegates to adaptive_tw
    else:
        return 'hybrid_drone'  # delegates to tw_aware


# ── KNN Classifier ───────────────────────────────────────────────────────

class KNNClassifier:
    """Simple K-Nearest Neighbors classifier."""

    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.X_mean = None
        self.X_std = None

    def fit(self, X, y):
        """
        Args:
            X: list of feature vectors (np.ndarray each)
            y: list of labels (str each)
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)

        # Z-score normalization
        self.X_mean = self.X_train.mean(axis=0)
        self.X_std = self.X_train.std(axis=0)
        self.X_std[self.X_std == 0] = 1.0  # avoid division by zero

    def _normalize(self, X):
        return (X - self.X_mean) / self.X_std

    def predict(self, X):
        """Predict single instance or batch."""
        X = np.atleast_2d(np.array(X))
        X_norm = self._normalize(X)
        train_norm = self._normalize(self.X_train)

        preds = []
        for x in X_norm:
            dists = np.sqrt(((train_norm - x) ** 2).sum(axis=1))
            knn_idx = np.argsort(dists)[:self.k]
            knn_labels = self.y_train[knn_idx]
            # Majority vote (break ties by closest neighbor)
            counter = Counter(knn_labels)
            max_count = max(counter.values())
            top = [l for l, c in counter.items() if c == max_count]
            if len(top) == 1:
                preds.append(top[0])
            else:
                # Tie: use closest neighbor among tied
                for idx in knn_idx:
                    if self.y_train[idx] in top:
                        preds.append(self.y_train[idx])
                        break
                else:
                    preds.append(top[0])

        if len(preds) == 1:
            return preds[0]
        return np.array(preds)

    def predict_proba(self, X):
        """Return probability distribution over classes."""
        X = np.atleast_2d(np.array(X))
        X_norm = self._normalize(X)
        train_norm = self._normalize(self.X_train)

        classes = sorted(set(self.y_train))
        all_probs = []

        for x in X_norm:
            dists = np.sqrt(((train_norm - x) ** 2).sum(axis=1))
            knn_idx = np.argsort(dists)[:self.k]
            knn_labels = self.y_train[knn_idx]
            knn_dists = dists[knn_idx]

            # Weight by inverse distance
            probs = {}
            for label, d in zip(knn_labels, knn_dists):
                w = 1.0 / max(d, 1e-8)
                probs[label] = probs.get(label, 0) + w

            total = sum(probs.values())
            all_probs.append({c: probs.get(c, 0) / total for c in classes})

        return all_probs


# ── Feature Importance via Permutation ───────────────────────────────────

def permutation_importance(model, X_test, y_test, n_repeats=10):
    """
    Compute feature importance by permuting each feature and measuring
    accuracy drop.

    Args:
        model: fitted KNNClassifier
        X_test: feature matrix (n_samples, n_features)
        y_test: true labels (list of str)

    Returns:
        dict mapping feature_name → (mean_drop, std_drop)
    """
    X = np.array(X_test)
    y = np.array(y_test)

    baseline_acc = np.mean(model.predict(X) == y)

    importances = {}
    for fi, fname in enumerate(FEATURE_NAMES):
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            np.random.shuffle(X_perm[:, fi])
            perm_acc = np.mean(model.predict(X_perm) == y)
            drops.append(baseline_acc - perm_acc)
        importances[fname] = (np.mean(drops), np.std(drops))

    return importances


# ── Meta-Learner Class ───────────────────────────────────────────────────

class MetaLearner:
    """
    Meta-learner for W5 variant selection.

    Usage:
        ml = MetaLearner(k=3)
        ml.fit(training_instances, training_labels)
        # training_labels: list of (instance, best_variant) tuples
        variant = ml.predict(new_instance)
    """

    def __init__(self, k=3):
        self.k = k
        self.knn = KNNClassifier(k=k)
        self.fitted = False

    def fit(self, instances_and_labels):
        """
        Args:
            instances_and_labels: list of (instance_dict, best_variant_str) tuples
        """
        X = []
        y = []
        for inst, label in instances_and_labels:
            X.append(extract_features(inst))
            y.append(label)

        self.X_train = X
        self.y_train = y
        self.knn.fit(X, y)
        self.fitted = True
        print(f'Meta-learner fitted on {len(y)} instances, '
              f'{len(set(y))} unique variants')

    def predict(self, instance):
        """Predict best variant for a new instance."""
        if not self.fitted:
            raise RuntimeError("Meta-learner not fitted. Call fit() first.")
        feats = extract_features(instance)
        return self.knn.predict(feats)

    def predict_with_confidence(self, instance):
        """Predict with probability distribution over variants."""
        if not self.fitted:
            raise RuntimeError("Meta-learner not fitted. Call fit() first.")
        feats = extract_features(instance)
        variant = self.knn.predict(feats)
        probs = self.knn.predict_proba(feats)[0]
        return variant, probs

    def leave_one_out_accuracy(self):
        """Compute leave-one-out cross-validation accuracy."""
        if not self.fitted:
            raise RuntimeError("Not fitted.")
        correct = 0
        for i in range(len(self.X_train)):
            # Leave one out
            X_loo = [x for j, x in enumerate(self.X_train) if j != i]
            y_loo = [y for j, y in enumerate(self.y_train) if j != i]
            model = KNNClassifier(k=self.k)
            model.fit(X_loo, y_loo)
            pred = model.predict(self.X_train[i])
            if pred == self.y_train[i]:
                correct += 1
        return correct / len(self.y_train)

    def feature_importance(self):
        """Compute permutation feature importance via LOOCV."""
        return permutation_importance(self.knn, self.X_train, self.y_train)


# ── Cli ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Quick test with dummy data
    import os, sys, json

    _W6 = os.path.dirname(os.path.abspath(__file__))
    _W5 = os.path.join(_W6, '..', 'week5')
    _W4 = os.path.join(_W6, '..', 'week4')
    sys.path.insert(0, _W5)
    sys.path.insert(1, _W4)

    from utils.data_loader import load_instance_from_disk, build_all_instances
    build_all_instances()

    inst = load_instance_from_disk('RC101_25c')
    feats = extract_features(inst)

    print('Feature extraction test on RC101_25c:')
    for name, val in zip(FEATURE_NAMES, feats):
        print(f'  {name:25s} = {val:.4f}')

    print(f'\nFeature vector shape: {feats.shape}')
    print(f'n_trucks_needed: {feats[14]:.0f}')
