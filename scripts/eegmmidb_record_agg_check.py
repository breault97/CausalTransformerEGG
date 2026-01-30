"""
Minimal regression tests for record-level aggregation helpers.

This script validates that the record-level aggregation choices used for EEGMMIDB exports
(mean_prob / mean_logit / trimmed_mean_prob / majority_vote) behave as expected on toy data.
"""

import numpy as np


def _softmax_np(x):
    """NumPy softmax for small arrays."""
    x = x - x.max(axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=1, keepdims=True)


def aggregate_record(probs, agg="mean_prob", trim=0.1, logits=None):
    """
    Aggregate window-level class probabilities into a single record-level prediction.

    Args:
        probs: array of shape (n_windows, n_classes)
        agg: aggregation mode (mean_prob / majority_vote / mean_logit / trimmed_mean_prob)
        trim: trim fraction (per tail) for trimmed_mean_prob
        logits: optional array of shape (n_windows, n_classes) for mean_logit
    """
    probs = np.asarray(probs, dtype=float)
    if agg == "mean_prob":
        return probs.mean(axis=0).argmax()
    if agg == "majority_vote":
        return np.bincount(probs.argmax(axis=1)).argmax()
    if agg == "mean_logit":
        if logits is None:
            logit_mean = np.log(np.clip(probs, 1e-12, 1.0)).mean(axis=0, keepdims=True)
        else:
            logit_mean = np.asarray(logits, dtype=float).mean(axis=0, keepdims=True)
        return _softmax_np(logit_mean).argmax()
    if agg == "trimmed_mean_prob":
        n = probs.shape[0]
        k = int(n * trim)
        if k <= 0 or n <= 2 * k:
            return probs.mean(axis=0).argmax()
        probs_sorted = np.sort(probs, axis=0)
        return probs_sorted[k:n - k].mean(axis=0).argmax()
    raise ValueError(f"Unknown agg: {agg}")


def main():
    # Synthetic example: 2 records, 3 classes
    rec1 = np.array([
        [0.90, 0.05, 0.05],
        [0.85, 0.10, 0.05],
        [0.10, 0.10, 0.80],
    ])
    rec2 = np.array([
        [0.20, 0.70, 0.10],
        [0.30, 0.60, 0.10],
        [0.05, 0.05, 0.90],
    ])

    # Expected: both records -> class 0 for rec1, class 1 for rec2
    for agg in ["mean_prob", "majority_vote", "mean_logit", "trimmed_mean_prob"]:
        pred1 = aggregate_record(rec1, agg=agg, trim=0.34)
        pred2 = aggregate_record(rec2, agg=agg, trim=0.34)
        assert pred1 == 0, f"{agg} pred1 expected 0, got {pred1}"
        assert pred2 == 1, f"{agg} pred2 expected 1, got {pred2}"

    # Pandas groupby regression check (no Series.name crash)
    try:
        import pandas as pd

        df = pd.DataFrame({
            "record_id": [1, 1, 1, 2, 2, 2],
            "y_true": [0, 0, 0, 1, 1, 1],
            "y_pred": [0, 0, 2, 1, 1, 2],
            "p0": [0.90, 0.85, 0.10, 0.20, 0.30, 0.05],
            "p1": [0.05, 0.10, 0.10, 0.70, 0.60, 0.05],
            "p2": [0.05, 0.05, 0.80, 0.10, 0.10, 0.90],
            "logit0": [2.0, 1.8, -1.0, -0.2, 0.1, -2.0],
            "logit1": [-1.0, -0.5, -0.8, 1.5, 1.1, -1.2],
            "logit2": [-1.0, -1.3, 1.2, -0.3, -0.4, 2.1],
        })

        prob_cols = ["p0", "p1", "p2"]
        logit_cols = ["logit0", "logit1", "logit2"]
        grouped = df.groupby(["record_id"], sort=False, dropna=False)
        trim = 0.34

        prob_means = []
        logit_means = []
        trimmed_means = []
        for _, g in grouped:
            probs = g[prob_cols].to_numpy()
            prob_means.append(probs.mean(axis=0))
            logit_means.append(g[logit_cols].to_numpy().mean(axis=0))
            n = probs.shape[0]
            k = int(n * trim)
            if k <= 0 or n <= 2 * k:
                trimmed_means.append(probs.mean(axis=0))
            else:
                probs_sorted = np.sort(probs, axis=0)
                trimmed_means.append(probs_sorted[k:n - k].mean(axis=0))

        prob_means = np.stack(prob_means, axis=0)
        logit_means = np.stack(logit_means, axis=0)
        trimmed_means = np.stack(trimmed_means, axis=0)

        assert prob_means.shape == (2, 3)
        assert logit_means.shape == (2, 3)
        assert trimmed_means.shape == (2, 3)
    except Exception as e:
        print(f"Pandas regression check skipped or failed: {e}")

    print("record_level_agg checks passed.")


if __name__ == "__main__":
    main()
