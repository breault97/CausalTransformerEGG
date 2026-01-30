from typing import Dict, Any

import numpy as np


def evaluate_subject_level(probs: np.ndarray,
                           labels: np.ndarray,
                           subject_ids: np.ndarray,
                           agg: str = "mean_prob") -> Dict[str, Any]:
    """
    Aggregate window-level probabilities to subject-level predictions.

    probs: (N, C) probability per window
    labels: (N,) true labels per window
    subject_ids: (N,) subject id per window
    agg: mean_prob | vote
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels).astype(int)
    subject_ids = np.asarray(subject_ids)

    if probs.ndim != 2:
        raise ValueError("probs must be 2D (N, C).")
    if labels.shape[0] != probs.shape[0] or subject_ids.shape[0] != probs.shape[0]:
        raise ValueError("probs, labels, and subject_ids must have matching length.")

    subjects = np.unique(subject_ids)
    n_classes = probs.shape[1]

    subj_probs = []
    subj_true = []
    subj_pred = []

    for s in subjects:
        idx = subject_ids == s
        if not np.any(idx):
            continue
        p = probs[idx]
        y = labels[idx]
        if agg in ("vote", "majority", "majority_vote"):
            pred = int(np.bincount(p.argmax(axis=1), minlength=n_classes).argmax())
            mean_probs = p.mean(axis=0)
        else:
            mean_probs = p.mean(axis=0)
            pred = int(mean_probs.argmax())
        true = int(np.bincount(y, minlength=n_classes).argmax())
        subj_probs.append(mean_probs)
        subj_true.append(true)
        subj_pred.append(pred)

    subj_probs = np.stack(subj_probs, axis=0) if subj_probs else np.zeros((0, n_classes))
    subj_true = np.asarray(subj_true, dtype=int)
    subj_pred = np.asarray(subj_pred, dtype=int)

    try:
        from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
    except Exception:
        balanced_accuracy_score = None
        f1_score = None
        confusion_matrix = None

    balanced_acc = None
    macro_f1 = None
    cm = None
    if subj_true.size and balanced_accuracy_score is not None:
        balanced_acc = float(balanced_accuracy_score(subj_true, subj_pred))
        macro_f1 = float(f1_score(subj_true, subj_pred, average="macro", zero_division=0))
        cm = confusion_matrix(subj_true, subj_pred, labels=list(range(n_classes)))

    return {
        "subject_ids": subjects.astype(int),
        "subject_probs": subj_probs,
        "y_true": subj_true,
        "y_pred": subj_pred,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
    }
