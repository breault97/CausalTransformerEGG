"""
Model-free quality-control tests for EEG quality classification (EEGMMIDB).

This script runs fast sanity checks using a simple logistic regression on flattened windows:
- Label permutation (must collapse to chance if the pipeline is healthy)
- Subject-ID prediction (high accuracy can indicate leakage/subject bias)
- Optional channel ablation (requires knowing the flattened input layout)

It is intentionally lightweight and does not depend on the main CT model.
"""

import logging
import os
import random
from typing import Dict, Tuple, List

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.metrics.eval import evaluate_subject_level

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _set_seed(seed: int):
    """Seed Python and NumPy RNGs for deterministic sanity checks."""
    random.seed(seed)
    np.random.seed(seed)


def _flatten_windows(split_data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a split dict with padded sequences into a flat (X, y, subject_ids) dataset.

    Expects:
    - `current_covariates`: (N, T, F)
    - `outputs`:            (N, T, 1) or (N, T)
    - `active_entries`:     (N, T, 1) or (N, T)

    Returns:
    - X: (n_active_windows, F)
    - y: (n_active_windows,)
    - subject_ids: (n_active_windows,) or None if not present in `split_data`
    """
    covs = split_data["current_covariates"]
    labels = split_data["outputs"].reshape(-1)
    active = split_data["active_entries"].reshape(-1).astype(bool)
    X = covs.reshape(-1, covs.shape[-1])[active]
    y = labels[active].astype(int)
    subject_ids = None
    if "subject_id" in split_data:
        subj = split_data["subject_id"]
        subj_rep = np.repeat(subj, covs.shape[1])
        subject_ids = subj_rep[active]
    return X, y, subject_ids


def _fit_logreg(X_train: np.ndarray, y_train: np.ndarray):
    """Fit a multinomial logistic regression baseline (scikit-learn)."""
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as e:
        raise ImportError("scikit-learn is required for quality tests.") from e
    clf = LogisticRegression(max_iter=200, n_jobs=1, multi_class="auto")
    clf.fit(X_train, y_train)
    return clf


def test_permutation(X_train, y_train, X_test, y_test) -> Dict[str, float]:
    """Train on permuted labels; metrics should be near chance for a healthy pipeline."""
    y_perm = np.random.permutation(y_train)
    clf = _fit_logreg(X_train, y_perm)
    y_pred = clf.predict(X_test)
    try:
        from sklearn.metrics import balanced_accuracy_score, f1_score
    except Exception as e:
        logger.warning(f"sklearn missing for metrics: {e}")
        return {}
    return {
        "perm_balanced_acc": float(balanced_accuracy_score(y_test, y_pred)),
        "perm_macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }


def test_subject_prediction(X_train, subj_train, X_test, subj_test) -> Dict[str, float]:
    """Train a classifier to predict subject ID; high accuracy can indicate leakage/bias."""
    clf = _fit_logreg(X_train, subj_train)
    y_pred = clf.predict(X_test)
    acc = float(np.mean(y_pred == subj_test))
    return {"subject_id_acc": acc}


def test_channel_ablation(X_train, y_train, X_test, y_test,
                          input_channels: int, input_samples: int,
                          ablate_channels: List[int]) -> Dict[str, float]:
    """
    Zero-out selected channels (after reshaping) and re-run the simple classifier.

    This is mainly a sensitivity check and requires a known `(input_channels, input_samples)`
    reshape for the flattened features.
    """
    X_train_r = X_train.reshape(X_train.shape[0], input_channels, input_samples)
    X_test_r = X_test.reshape(X_test.shape[0], input_channels, input_samples)
    X_train_r[:, ablate_channels, :] = 0.0
    X_test_r[:, ablate_channels, :] = 0.0
    clf = _fit_logreg(X_train_r.reshape(X_train.shape[0], -1), y_train)
    y_pred = clf.predict(X_test_r.reshape(X_test.shape[0], -1))
    try:
        from sklearn.metrics import balanced_accuracy_score, f1_score
    except Exception as e:
        logger.warning(f"sklearn missing for metrics: {e}")
        return {}
    return {
        "ablation_balanced_acc": float(balanced_accuracy_score(y_test, y_pred)),
        "ablation_macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }


@hydra.main(version_base="1.3", config_name="config.yaml", config_path="../config/")
def main(args: DictConfig):
    OmegaConf.set_struct(args, False)
    logger.info("\n" + OmegaConf.to_yaml(args, resolve=True))

    _set_seed(int(args.exp.seed))
    if getattr(args, "fold_index", None) is not None:
        args.dataset.fold_index = int(args.fold_index)
    dataset_collection = instantiate(args.dataset, _recursive_=True)
    dataset_collection.process_data_multi()

    X_train, y_train, subj_train = _flatten_windows(dataset_collection.train_f.data)
    X_test, y_test, subj_test = _flatten_windows(dataset_collection.test_f.data)

    results = {}
    results.update(test_permutation(X_train, y_train, X_test, y_test))

    if subj_train is not None and subj_test is not None:
        results.update(test_subject_prediction(X_train, subj_train, X_test, subj_test))

    # Channel ablation (requires reshape params)
    input_channels = getattr(args.dataset, "input_channels", None)
    input_samples = getattr(args.dataset, "input_samples", None)
    ablate_channels = getattr(args.exp, "ablate_channels", [])
    if input_channels and input_samples and ablate_channels:
        results.update(test_channel_ablation(
            X_train, y_train, X_test, y_test,
            int(input_channels), int(input_samples),
            [int(c) for c in ablate_channels]
        ))

    out_path = os.path.join(os.getcwd(), "quality_tests_results.json")
    try:
        import json
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved quality test results to {out_path}")
    except Exception as e:
        logger.warning(f"Failed to save test results: {e}")


if __name__ == "__main__":
    main()
