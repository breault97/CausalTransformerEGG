"""
Hydra entrypoint for CT training/evaluation (including EEGMMIDB classification).

This script is used by the EEG experiments in `config/experiment/eegmmidb_ct_*.yaml`.

Sanity flags (environment variables):
- `CT_SHUFFLE_TRAIN_LABELS=1`: permute training labels (should drop to chance).
- `CT_SHUFFLE_TRAIN_INPUTS=1` + `CT_SHUFFLE_TRAIN_INPUTS_MODE=...`: input shuffling modes.
- `CT_ONLY_SPLIT_CHECK=1`: build/validate splits and exit early.
"""

import logging
import os
import hydra
import torch
import mlflow
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
import inspect
import math

from src.models.utils import AlphaRise, FilteringMlFlowLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.double)

def _env_flag_01(name: str, default: str = "0") -> str:
    """Return "1" if env var is exactly "1" (after strip), else "0"."""
    val = os.getenv(name, default)
    return "1" if str(val).strip() == "1" else "0"

def _env_str(name: str, default: str = "") -> str:
    return str(os.getenv(name, default)).strip()


def _get_sanity_flags_from_env():
    """Read and normalize sanity/debug flags from environment variables."""
    return {
        "CT_SHUFFLE_TRAIN_LABELS": _env_flag_01("CT_SHUFFLE_TRAIN_LABELS", "0"),
        "CT_SHUFFLE_TRAIN_INPUTS": _env_flag_01("CT_SHUFFLE_TRAIN_INPUTS", "0"),
        "CT_SHUFFLE_TRAIN_INPUTS_MODE": _env_str("CT_SHUFFLE_TRAIN_INPUTS_MODE", ""),
        "CT_ONLY_SPLIT_CHECK": _env_flag_01("CT_ONLY_SPLIT_CHECK", "0"),
    }


def _log_sanity_flags_to_mlflow(mlf_logger, sanity_flags: dict) -> None:
    """Persist sanity/debug flags as both MLflow params and tags for easier auditing."""
    if mlf_logger is None:
        return
    try:
        run_id = mlf_logger.run_id
        client = mlf_logger.experiment
        for k, v in sanity_flags.items():
            # Make flags easily grep-able in both exports: params.json and tags.json
            client.log_param(run_id, k, v)
            client.set_tag(run_id, k, v)
    except Exception as e:
        logger.warning(f"Failed to log sanity flags to MLflow: {e}")


def _resolve_precision(exp_cfg):
    precision = getattr(exp_cfg, "precision", 32)
    if isinstance(precision, str):
        p = precision.lower()
        if p in ("16", "16-mixed", "fp16"):
            return 16
        if p in ("32", "32-true", "fp32"):
            return 32
        if p in ("bf16", "bfloat16"):
            # Lightning 1.4 does not fully support bf16; fall back to 16 with a warning.
            logger.warning("bf16 precision requested but not supported in this Lightning version; using fp16.")
            return 16
        try:
            return int(precision)
        except Exception:
            return 32
    try:
        return int(precision)
    except Exception:
        return 32


def _normalize_treatment_mode(mode):
    if mode is None:
        return None
    m = str(mode).strip().lower()
    if m in ("none", "no", "off", "false", "null", "0"):
        return "none"
    return m


def _as_int(val):
    if val is None:
        return None
    if isinstance(val, str) and val.strip() == "???":
        return None
    try:
        return int(val)
    except Exception:
        return None


def _validate_treatment_config(args: DictConfig):
    mode = _normalize_treatment_mode(getattr(args.dataset, "treatment_mode", None))
    dataset_dim = _as_int(getattr(args.dataset, "treatment_dim", None))
    model_dim = _as_int(getattr(args.model, "dim_treatments", None))

    if mode == "none" and dataset_dim not in (0, None):
        raise ValueError(
            "Config error: dataset.treatment_mode=none requires dataset.treatment_dim=0. "
            f"Got dataset.treatment_dim={dataset_dim}."
        )
    if dataset_dim is not None and model_dim is not None and dataset_dim != model_dim:
        raise ValueError(
            "Config error: model.dim_treatments must match dataset.treatment_dim. "
            f"Got model.dim_treatments={model_dim}, dataset.treatment_dim={dataset_dim}."
        )


def _get_dataloader_kwargs(args):
    exp_workers = getattr(args.exp, "num_workers", None)
    data_workers = getattr(args.dataset, "num_workers", 0)
    if exp_workers is None:
        try:
            num_workers = int(data_workers)
        except Exception:
            num_workers = 0
    else:
        try:
            num_workers = max(0, int(exp_workers))
            if data_workers is not None:
                try:
                    data_workers_int = int(data_workers)
                    if data_workers_int != num_workers and not getattr(_get_dataloader_kwargs, "_warned_workers", False):
                        logger.info(
                            f"DataLoader: using exp.num_workers={num_workers} "
                            f"(dataset.num_workers={data_workers_int})."
                        )
                        _get_dataloader_kwargs._warned_workers = True
                except Exception:
                    pass
        except Exception:
            logger.warning("Invalid exp.num_workers; falling back to dataset.num_workers.")
            try:
                num_workers = int(data_workers)
            except Exception:
                num_workers = 0
    pin_memory = bool(getattr(args.exp, "pin_memory", False))
    persistent_workers = bool(getattr(args.exp, "persistent_workers", False))
    prefetch_factor = getattr(args.exp, "prefetch_factor", None)
    if num_workers <= 0:
        if (persistent_workers or prefetch_factor is not None) and not getattr(_get_dataloader_kwargs, "_warned_flags", False):
            logger.warning("DataLoader: num_workers=0, ignoring persistent_workers/prefetch_factor.")
            _get_dataloader_kwargs._warned_flags = True
        persistent_workers = False
        prefetch_factor = None
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if persistent_workers:
        kwargs["persistent_workers"] = True
    if prefetch_factor is not None:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return kwargs


def _trainer_device_kwargs(args):
    gpus = eval(str(args.exp.gpus))
    sig = inspect.signature(Trainer.__init__)
    if "devices" in sig.parameters:
        accelerator = "gpu" if gpus else "cpu"
        return {"accelerator": accelerator, "devices": gpus}
    return {"gpus": gpus}


def _resolve_log_every_n_steps(args, dataset_collection):
    configured = getattr(args.exp, "log_every_n_steps", None)
    if configured is not None:
        try:
            return max(1, int(configured))
        except Exception:
            pass
    try:
        batch_size = int(args.model.multi.batch_size)
        # drop_last=True in train dataloader -> use floor for steps/epoch
        steps = len(dataset_collection.train_f) // batch_size
        steps = max(1, int(steps))
        return max(1, min(50, steps))
    except Exception:
        return 50


def run(args: DictConfig):
    """
    Training / evaluation script for CT (Causal Transformer).

    This entrypoint supports both regression-style tasks (original repo) and EEG quality
    classification experiments (EEGMMIDB) configured through Hydra.

    Args:
        args: arguments of run as DictConfig

    Returns:
        results: a dict of aggregated metrics (classification or regression, depending on config)
    """

    results = {}

    # Non-strict access to fields
    OmegaConf.set_struct(args, False)
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)
    logger.info('\n' + OmegaConf.to_yaml(args, resolve=True))

    _validate_treatment_config(args)

    # Force MLflow tracking URI (avoid Hydra file:// outputs mlruns)
    os.environ["MLFLOW_TRACKING_URI"] = str(args.exp.mlflow_uri)
    mlflow.set_tracking_uri(str(args.exp.mlflow_uri))

    sanity_flags = _get_sanity_flags_from_env()
    inputs_mode = sanity_flags.get("CT_SHUFFLE_TRAIN_INPUTS_MODE", "")
    if sanity_flags.get("CT_SHUFFLE_TRAIN_INPUTS", "0") == "1" and inputs_mode == "":
        inputs_mode = "intra_sequence_lockstep"
    logger.info(
        "SANITY FLAGS: "
        f"LABELS={sanity_flags['CT_SHUFFLE_TRAIN_LABELS']}, "
        f"INPUTS={sanity_flags['CT_SHUFFLE_TRAIN_INPUTS']}"
        + (f", INPUTS_MODE={inputs_mode}" if sanity_flags.get("CT_SHUFFLE_TRAIN_INPUTS", "0") == "1" else "")
        + ", "
        f"ONLY_SPLIT_CHECK={sanity_flags['CT_ONLY_SPLIT_CHECK']}"
    )

    # Create MLflow logger early so CT_ONLY_SPLIT_CHECK=1 exits are still traceable.
    if args.exp.logging:
        experiment_name = f'{args.model.name}/{args.dataset.name}'
        mlf_logger = FilteringMlFlowLogger(
            filter_submodels=[], experiment_name=experiment_name, tracking_uri=args.exp.mlflow_uri
        )
        _log_sanity_flags_to_mlflow(mlf_logger, sanity_flags)
        try:
            artifacts_path = hydra.utils.to_absolute_path(
                mlf_logger.experiment.get_run(mlf_logger.run_id).info.artifact_uri
            )
        except Exception as e:
            logger.warning(f"Failed to resolve MLflow artifacts_path: {e}")
            artifacts_path = None
    else:
        mlf_logger = None
        artifacts_path = None

    # Allow top-level fold_index / n_folds overrides for CV
    if getattr(args, "fold_index", None) is not None:
        args.dataset.fold_index = int(args.fold_index)
    if getattr(args, "n_folds", None) is not None:
        args.dataset.n_folds = int(args.n_folds)

    # Initialisation of data
    seed_everything(args.exp.seed, workers=True)
    try:
        dataset_collection = instantiate(args.dataset, _recursive_=True)
        dataset_collection.process_data_multi()
    except SystemExit:
        # Ensure MLflow run is cleanly closed for split-check runs.
        if mlf_logger is not None:
            try:
                mlf_logger.experiment.set_terminated(mlf_logger.run_id)
            except Exception as e:
                logger.warning(f"Failed to set MLflow run terminated: {e}")
            try:
                mlflow.end_run(status="FINISHED")
            except Exception:
                pass
        raise

    if bool(getattr(args.exp, "multi_task", False)):
        dataset_name = str(getattr(args.dataset, "name", "")).lower()
        if dataset_name == "eegmmidb":
            raise ValueError("multi_task (shock localization) is disabled for eegmmidb quality classification.")

    # args.model.dim_outcomes = dataset_collection.train_f.data['outputs'].shape[-1]
    out_dim = dataset_collection.train_f.data['outputs'].shape[-1]
    prev_out_dim = dataset_collection.train_f.data['prev_outputs'].shape[-1]
    args.model.dim_outcomes_input = prev_out_dim

    if args.exp.task == 'classification':
        # One-hot labels -> out_dim is number of classes
        # class-index labels -> out_dim is 1
        if args.model.dim_outcomes is None or str(args.model.dim_outcomes) == "???":
            if out_dim > 1:
                args.model.dim_outcomes = out_dim   # One-hot
            else:
                raise ValueError("dim_outcomes must be specified for classification.")
        else:
            args.model.dim_outcomes = int(args.model.dim_outcomes)   # Keep number of classes
    else:    # Regression
        args.model.dim_outcomes = out_dim
        args.model.dim_outcomes_input = prev_out_dim

    args.model.dim_treatments = dataset_collection.train_f.data['current_treatments'].shape[-1]
    args.model.dim_vitals = dataset_collection.train_f.data['vitals'].shape[-1] if dataset_collection.has_vitals else 0
    args.model.dim_static_features = dataset_collection.train_f.data['static_features'].shape[-1]

    # Train_callbacks
    multimodel_callbacks = [AlphaRise(rate=args.exp.alpha_rate)]
    early_stopping_cb = None
    if args.exp.task == 'classification':
        if getattr(args.exp, "early_stopping", True):
            logger.info("EarlyStopping enabled.")
            monitor = getattr(args.exp, "early_stopping_monitor", "multi_val_f1_macro")
            mode = getattr(args.exp, "early_stopping_mode", "max")
            patience = int(getattr(args.exp, "early_stopping_patience", 20))
            min_delta = float(getattr(args.exp, "early_stopping_min_delta", 0.0))
            early_stopping_cb = EarlyStopping(
                monitor=monitor,
                mode=mode,
                patience=patience,
                min_delta=min_delta,
            )
            multimodel_callbacks.append(early_stopping_cb)
        else:
            logger.info("EarlyStopping disabled (exp.early_stopping=False).")
        if getattr(args.exp, "checkpoint_best", True):
            monitor = getattr(args.exp, "checkpoint_monitor", "multi_val_f1_macro")
            mode = getattr(args.exp, "checkpoint_mode", "max")
            save_top_k = int(getattr(args.exp, "checkpoint_save_top_k", 1))
            filename = getattr(args.exp, "checkpoint_filename", f"best-{{epoch:02d}}-{{{monitor}:.4f}}")
            multimodel_callbacks.append(ModelCheckpoint(
                dirpath=os.getcwd(),
                monitor=monitor,
                mode=mode,
                save_top_k=save_top_k,
                filename=filename,
            ))

    # MLflow callbacks + artifacts
    if mlf_logger is not None:
        multimodel_callbacks += [LearningRateMonitor(logging_interval='epoch')]
        # Log fold scaler / labeler parameters for reproducibility
        try:
            run_id = mlf_logger.run_id
            client = mlf_logger.experiment
            if getattr(dataset_collection, "scaler_", None) is not None:
                client.log_dict(run_id, dataset_collection.scaler_.to_loggable(), "scalers/per_fold_scaler.json")
            if getattr(dataset_collection, "labeler_", None) is not None:
                labeler = dataset_collection.labeler_
                if hasattr(labeler, "to_loggable"):
                    client.log_dict(run_id, labeler.to_loggable(), "labelers/quantile_labeler.json")
        except Exception as e:
            logger.warning(f"Failed to log scaler/labeler artifacts: {e}")

    # ============================== Initialisation & Training of multimodel ==============================
    multimodel = instantiate(args.model.multi, args, dataset_collection, _recursive_=False)
    if args.model.multi.tune_hparams:
        multimodel.finetune(resources_per_trial=args.model.multi.resources_per_trial)

    precision = _resolve_precision(args.exp)
    trainer_device_kwargs = _trainer_device_kwargs(args)
    log_every_n_steps = _resolve_log_every_n_steps(args, dataset_collection)
    grad_clip_val = float(getattr(args.model.multi, "max_grad_norm", 0.0) or 0.0)
    grad_clip_algorithm = str(getattr(args.exp, "gradient_clip_algorithm", "norm")).lower()
    trainer_kwargs = dict(
        **trainer_device_kwargs,
        logger=mlf_logger,
        max_epochs=args.exp.max_epochs,
        callbacks=multimodel_callbacks,
        terminate_on_nan=True,
        gradient_clip_val=grad_clip_val if grad_clip_val > 0.0 else 0.0,
        accumulate_grad_batches=getattr(args.exp, "accumulate_grad_batches", 1),
        precision=precision,
    )
    trainer_sig = inspect.signature(Trainer.__init__)
    if "gradient_clip_algorithm" in trainer_sig.parameters:
        trainer_kwargs["gradient_clip_algorithm"] = grad_clip_algorithm
    if "log_every_n_steps" in trainer_sig.parameters:
        trainer_kwargs["log_every_n_steps"] = log_every_n_steps

    multimodel_trainer = Trainer(**trainer_kwargs)
    multimodel_trainer.fit(multimodel)
    if early_stopping_cb is not None and getattr(early_stopping_cb, "stopped_epoch", 0):
        logger.info(f"EarlyStopping stopped at epoch={early_stopping_cb.stopped_epoch} "
                    f"best_score={getattr(early_stopping_cb, 'best_score', None)}")

    # Evaluation
    dl_kwargs = _get_dataloader_kwargs(args)

    if args.exp.task == 'classification':
        # Factual validation
        val_dataloader = DataLoader(dataset_collection.val_f, batch_size=args.dataset.val_batch_size, shuffle=False,
                                    **dl_kwargs)
        multimodel_trainer.validate(multimodel, dataloaders=val_dataloader)

        if hasattr(dataset_collection, 'test_f'):
            test_dataloader = DataLoader(dataset_collection.test_f, batch_size=args.dataset.val_batch_size, shuffle=False,
                                         **dl_kwargs)
            multimodel_trainer.test(multimodel, dataloaders=test_dataloader)

        logger.info("Classification run: metrics are logged via Lightning (e.g., *_acc, *_ce_loss).")
        mlf_logger.experiment.set_terminated(mlf_logger.run_id) if args.exp.logging else None
        return results

    # Regression path (keep existing RMSE logic)

    val_dataloader = DataLoader(dataset_collection.val_f, batch_size=args.dataset.val_batch_size, shuffle=False,
                                **dl_kwargs)
    multimodel_trainer.test(multimodel, dataloaders=val_dataloader)
    val_rmse_orig, val_rmse_all = multimodel.get_normalised_masked_rmse(dataset_collection.val_f)
    logger.info(f'Val normalised RMSE (all): {val_rmse_all}; Val normalised RMSE (orig): {val_rmse_orig}')

    encoder_results = {}
    if hasattr(dataset_collection, 'test_cf_one_step'):  # Test one_step_counterfactual rmse
        test_rmse_orig, test_rmse_all, test_rmse_last = multimodel.get_normalised_masked_rmse(
            dataset_collection.test_cf_one_step, one_step_counterfactual=True
        )
        logger.info(f'Test normalised RMSE (all): {test_rmse_all}; '
                    f'Test normalised RMSE (orig): {test_rmse_orig}; '
                    f'Test normalised RMSE (only counterfactual): {test_rmse_last}')
        encoder_results = {
            'encoder_val_rmse_all': val_rmse_all,
            'encoder_val_rmse_orig': val_rmse_orig,
            'encoder_test_rmse_all': test_rmse_all,
            'encoder_test_rmse_orig': test_rmse_orig,
            'encoder_test_rmse_last': test_rmse_last
        }
    elif hasattr(dataset_collection, 'test_f'):  # Test factual rmse
        test_rmse_orig, test_rmse_all = multimodel.get_normalised_masked_rmse(dataset_collection.test_f)
        logger.info(f'Test normalised RMSE (all): {test_rmse_all}; '
                    f'Test normalised RMSE (orig): {test_rmse_orig}.')
        encoder_results = {
            'encoder_val_rmse_all': val_rmse_all,
            'encoder_val_rmse_orig': val_rmse_orig,
            'encoder_test_rmse_all': test_rmse_all,
            'encoder_test_rmse_orig': test_rmse_orig
        }

    mlf_logger.log_metrics(encoder_results) if args.exp.logging else None
    results.update(encoder_results)

    test_rmses = {}
    if hasattr(dataset_collection, 'test_cf_treatment_seq'):  # Test n_step_counterfactual rmse
        test_rmses = multimodel.get_normalised_n_step_rmses(dataset_collection.test_cf_treatment_seq)
    elif hasattr(dataset_collection, 'test_f_multi'):  # Test n_step_factual rmse
        test_rmses = multimodel.get_normalised_n_step_rmses(dataset_collection.test_f_multi)
    test_rmses = {f'{k+2}-step': v for (k, v) in enumerate(test_rmses)}

    logger.info(f'Test normalised RMSE (n-step prediction): {test_rmses}')
    decoder_results = {
        'decoder_val_rmse_all': val_rmse_all,
        'decoder_val_rmse_orig': val_rmse_orig
    }
    decoder_results.update({('decoder_test_rmse_' + k): v for (k, v) in test_rmses.items()})

    mlf_logger.log_metrics(decoder_results) if args.exp.logging else None
    results.update(decoder_results)

    mlf_logger.experiment.set_terminated(mlf_logger.run_id) if args.exp.logging else None
    return results


@hydra.main(version_base="1.3", config_name=f'config.yaml', config_path='../config/')
def main(args: DictConfig):
    return run(args)


if __name__ == "__main__":
    main()
