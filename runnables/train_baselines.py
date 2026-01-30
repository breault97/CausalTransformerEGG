import logging
import os
import random
import time
import gc
from typing import Dict, Tuple, List

import hydra
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
if "matplotlib.pyplot" not in os.sys.modules:
    matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from src.metrics.eval import evaluate_subject_level
from src.models.baselines import EEGNet, ShallowConvNet, SimpleCNN1D, CSP_LDA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _flatten_windows(split_data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    record_ids = None
    if "record_id" in split_data:
        rec = split_data["record_id"]
        rec_rep = np.repeat(rec, covs.shape[1])
        record_ids = rec_rep[active]
    return X, y, subject_ids, record_ids


def _reshape_features(X: np.ndarray, input_channels: int = None, input_samples: int = None) -> np.ndarray:
    n, f = X.shape
    if input_channels is None and input_samples is None:
        input_channels = 1
        input_samples = f
    elif input_channels is None:
        if f % input_samples != 0:
            raise ValueError("input_samples does not divide feature dimension.")
        input_channels = f // input_samples
    elif input_samples is None:
        if f % input_channels != 0:
            raise ValueError("input_channels does not divide feature dimension.")
        input_samples = f // input_channels
    if input_channels * input_samples != f:
        raise ValueError("input_channels * input_samples must equal feature dimension.")
    return X.reshape(n, input_channels, input_samples)


def _evaluate_window_level(probs: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    try:
        from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
    except Exception as e:
        logger.warning(f"sklearn missing for metrics: {e}")
        return {}
    y_pred = probs.argmax(axis=1)
    return {
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def _evaluate_record_level(probs: np.ndarray, y_true: np.ndarray, record_ids: np.ndarray,
                           agg: str = "mean_prob") -> Dict[str, float]:
    if record_ids is None:
        return {}
    try:
        from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
    except Exception as e:
        logger.warning(f"sklearn missing for record metrics: {e}")
        return {}
    record_ids = np.asarray(record_ids)
    uniq = np.unique(record_ids)
    n_classes = probs.shape[1]
    rec_true = []
    rec_pred = []
    for rid in uniq:
        idx = record_ids == rid
        if not np.any(idx):
            continue
        p = probs[idx]
        y = y_true[idx]
        if agg in ("vote", "majority", "majority_vote"):
            pred = int(np.bincount(p.argmax(axis=1), minlength=n_classes).argmax())
        else:
            pred = int(p.mean(axis=0).argmax())
        true = int(np.bincount(y, minlength=n_classes).argmax())
        rec_true.append(true)
        rec_pred.append(pred)
    rec_true = np.asarray(rec_true, dtype=int)
    rec_pred = np.asarray(rec_pred, dtype=int)
    if rec_true.size == 0:
        return {}
    return {
        "balanced_acc": float(balanced_accuracy_score(rec_true, rec_pred)),
        "macro_f1": float(f1_score(rec_true, rec_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(rec_true, rec_pred, labels=list(range(n_classes))),
    }


def _save_confusion_matrix(cm: np.ndarray, labels: List[str], path: str, title: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        im = ax.imshow(cm, interpolation="nearest")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
    finally:
        plt.close(fig)


def _train_torch_model(model: torch.nn.Module,
                       train_loader: DataLoader,
                       val_loader: DataLoader,
                       lr: float,
                       weight_decay: float,
                       max_epochs: int,
                       device: torch.device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).long()  # CrossEntropyLoss requires target to be torch.long
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.info(f"    [Epoch {epoch + 1}/{max_epochs}] completed.")

        # Light validation pass to keep logs consistent
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                _ = [model(xv.to(device)) for xv, _ in val_loader]


def _run_baseline(cfg: DictConfig, backbone_name: str):
    seed = int(cfg.exp.seed)
    _set_seed(seed)

    fold_index = int(cfg.fold_index)
    n_folds = int(getattr(cfg.dataset, "n_folds", 5))
    cfg.dataset.fold_index = fold_index

    dataset_collection = instantiate(cfg.dataset, _recursive_=True)
    dataset_collection.process_data_multi()

    X_train, y_train, _, _train_records = _flatten_windows(dataset_collection.train_f.data)
    X_val, y_val, val_subjects, _val_records = _flatten_windows(dataset_collection.val_f.data)
    X_test, y_test, test_subjects, test_records = _flatten_windows(dataset_collection.test_f.data)

    input_channels = getattr(cfg.dataset, "input_channels", None)
    input_samples = getattr(cfg.dataset, "input_samples", None)
    if input_channels is not None:
        input_channels = int(input_channels)
    if input_samples is not None:
        input_samples = int(input_samples)

    X_train_r = _reshape_features(X_train, input_channels, input_samples)
    X_val_r = _reshape_features(X_val, input_channels, input_samples)
    X_test_r = _reshape_features(X_test, input_channels, input_samples)

    run_name = f"{backbone_name}_fold{fold_index}"
    mlflow.set_tracking_uri(str(cfg.exp.mlflow_uri))
    mlflow.set_experiment(f"baselines/{cfg.dataset.name}")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "backbone": backbone_name,
            "fold_index": fold_index,
            "n_folds": n_folds,
        })
        if getattr(dataset_collection, "scaler_", None) is not None:
            mlflow.log_dict(dataset_collection.scaler_.to_loggable(), "scalers/per_fold_scaler.json")
        if getattr(dataset_collection, "labeler_", None) is not None:
            labeler = dataset_collection.labeler_
            if hasattr(labeler, "to_loggable"):
                mlflow.log_dict(labeler.to_loggable(), "labelers/quantile_labeler.json")

        baseline_cfg = OmegaConf.to_container(cfg.model.baseline, resolve=True)
        baseline_cfg.pop("_target_", None)

        if backbone_name.lower() == "csp_lda":
            logger.info(f"    Training {backbone_name}...")
            model = CSP_LDA(**baseline_cfg)
            model.fit(X_train_r, y_train)
            logger.info(f"    ...done.")
            logger.info(f"    Inferring with {backbone_name}...")
            probs = model.predict_proba(X_test_r)
            logger.info(f"    ...done.")
        else:
            # Torch CNN baselines
            n_channels = X_train_r.shape[1]
            n_samples = X_train_r.shape[2]
            baseline_cfg["n_channels"] = n_channels
            baseline_cfg["n_samples"] = n_samples
            baseline_cfg["n_classes"] = int(baseline_cfg.get("n_classes", 3))

            model = instantiate(cfg.model.baseline, **baseline_cfg)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            batch_size = int(cfg.model.get("batch_size", 128))
            max_epochs = int(cfg.model.get("max_epochs", 30))
            if max_epochs <= 0:
                raise ValueError(f"model.max_epochs must be positive, but got {max_epochs}.")
            lr = float(cfg.model.optimizer.get("learning_rate", 1e-3))
            wd = float(cfg.model.optimizer.get("weight_decay", 0.0))

            train_loader = DataLoader(
                TensorDataset(torch.tensor(X_train_r, dtype=torch.float32), torch.tensor(y_train)),
                batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(
                TensorDataset(torch.tensor(X_val_r, dtype=torch.float32), torch.tensor(y_val)),
                batch_size=batch_size, shuffle=False
            )
            logger.info(f"    Training {backbone_name} for {max_epochs} epochs...")
            _train_torch_model(model, train_loader, val_loader, lr, wd, max_epochs, device)
            logger.info(f"    ...done.")

            logger.info(f"    Inferring with {backbone_name}...")
            model.eval()
            with torch.no_grad():
                logits = model(torch.tensor(X_test_r, dtype=torch.float32).to(device))
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            logger.info(f"    ...done.")

        logger.info("    Exporting metrics and artifacts...")
        window_metrics = _evaluate_window_level(probs, y_test)
        if window_metrics:
            mlflow.log_metric("window_balanced_acc", window_metrics["balanced_acc"])
            mlflow.log_metric("window_macro_f1", window_metrics["macro_f1"])
            labels = getattr(cfg.dataset, "label_names", ["0", "1", "2"])
            if "confusion_matrix" in window_metrics and window_metrics["confusion_matrix"] is not None:
                cm_path = os.path.join(os.getcwd(), f"confusion_matrix_test_window_{run_name}.png")
                _save_confusion_matrix(window_metrics["confusion_matrix"], labels,
                                       cm_path, f"Confusion Matrix (window, {run_name})")
                mlflow.log_artifact(cm_path, artifact_path="confusion_matrices")

        record_agg = getattr(cfg.exp, "record_level_agg", "mean_prob")
        record_metrics = _evaluate_record_level(probs, y_test, test_records, agg=record_agg)
        if record_metrics:
            mlflow.log_metric("record_balanced_acc", record_metrics["balanced_acc"])
            mlflow.log_metric("record_macro_f1", record_metrics["macro_f1"])
            labels = getattr(cfg.dataset, "label_names", ["0", "1", "2"])
            if "confusion_matrix" in record_metrics and record_metrics["confusion_matrix"] is not None:
                cm_path = os.path.join(os.getcwd(), f"confusion_matrix_test_record_{run_name}.png")
                _save_confusion_matrix(record_metrics["confusion_matrix"], labels,
                                       cm_path, f"Confusion Matrix (record, {run_name})")
                mlflow.log_artifact(cm_path, artifact_path="confusion_matrices")
        mlflow.log_param("record_level_agg", record_agg)

        if test_subjects is not None:
            subj_metrics = evaluate_subject_level(probs, y_test, test_subjects)
            if subj_metrics.get("balanced_accuracy") is not None:
                mlflow.log_metric("subject_balanced_acc", subj_metrics["balanced_accuracy"])
            if subj_metrics.get("macro_f1") is not None:
                mlflow.log_metric("subject_macro_f1", subj_metrics["macro_f1"])
            if subj_metrics.get("confusion_matrix") is not None:
                labels = getattr(cfg.dataset, "label_names", ["0", "1", "2"])
                cm_path = os.path.join(os.getcwd(), f"confusion_matrix_test_subject_{run_name}.png")
                _save_confusion_matrix(subj_metrics["confusion_matrix"], labels,
                                       cm_path, f"Confusion Matrix (subject, {run_name})")
                mlflow.log_artifact(cm_path, artifact_path="confusion_matrices")
        logger.info("    ...done.")


@hydra.main(version_base="1.3", config_name="config.yaml", config_path="../config/")
def main(args: DictConfig):
    OmegaConf.set_struct(args, False)
    logger.info("\n" + OmegaConf.to_yaml(args, resolve=True))

    baseline_list = getattr(args.exp, "baseline_list", None)
    if baseline_list is None:
        baseline_list = [str(args.model.name)]
    if isinstance(baseline_list, str):
        baseline_list = [baseline_list]

    # --- New sequential fold logic ---
    run_all_folds = bool(getattr(args.exp, "run_all_folds", False))
    if run_all_folds:
        n_folds = int(getattr(args.exp, "n_folds", 5))
        fold_indices = range(n_folds)
        logger.info(f"Sequential mode: running all {n_folds} folds.")
    else:
        cli_fold_index = getattr(args, "fold_index", 0)
        fold_indices = [int(cli_fold_index)]
        logger.info(f"Single mode: running fold {cli_fold_index} only.")

    for fold_idx in fold_indices:
        logger.info(f"===== STARTING FOLD {fold_idx}/{len(fold_indices)} =====")
        # Create a mutable copy for this fold to avoid state leakage
        base_cfg = OmegaConf.create(OmegaConf.to_container(args, resolve=False))
        base_cfg.fold_index = fold_idx  # Set fold for this iteration

        n_baselines = len(baseline_list)
        for i, backbone in enumerate(baseline_list):
            t0 = time.time()
            logger.info(f"=== START baseline {i+1}/{n_baselines}: {backbone} (fold {fold_idx}) ===")
            backbone_cfg_path = hydra.utils.to_absolute_path(os.path.join("config", "backbone", f"{backbone}.yaml"))
            if not os.path.exists(backbone_cfg_path):
                logger.warning(f"Backbone config not found: {backbone_cfg_path}; skipping {backbone}.")
                continue
            
            backbone_cfg = OmegaConf.load(backbone_cfg_path)
            # Important: merge fold-specific cfg into the merged experiment+backbone cfg
            cfg = OmegaConf.merge(base_cfg, backbone_cfg)
            cfg.fold_index = fold_idx # Ensure it's set correctly after merge

            # The _run_baseline function is now simplified to run only one fold
            # by reading cfg.fold_index.
            _run_baseline(cfg, backbone)
            
            # Cleanup VRAM after each baseline model run
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            elapsed = time.time() - t0
            logger.info(f"=== END baseline {i+1}/{n_baselines}: {backbone} (fold {fold_idx}, elapsed: {elapsed:.1f}s) ===")
        logger.info(f"===== COMPLETED FOLD {fold_idx}/{len(fold_indices)} =====")




if __name__ == "__main__":
    main()
