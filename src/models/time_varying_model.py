"""
Core time-varying model training/evaluation logic (PyTorch Lightning).

EEGMMIDB-specific additions in this module include:
- Record/subject-level aggregation utilities for EEG quality classification outputs
  (e.g., `record_level_agg=mean_prob|mean_logit|trimmed_mean_prob|majority_vote`).
- Optional sanity modes controlled by environment variables:
  - `CT_SHUFFLE_TRAIN_INPUTS=1` with `CT_SHUFFLE_TRAIN_INPUTS_MODE=...` for input shuffling.
    In particular, `inter_batch_decouple` permutes *inputs* across the batch while keeping labels
    fixed, which should drive performance toward chance if the pipeline is healthy.

Only comments/docstrings are modified in this file for documentation purposes.
"""

import torch.optim as optim
from pytorch_lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
import torch
from typing import Union
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import logging
import inspect
import math
import numpy as np
from copy import deepcopy
from pytorch_lightning import Trainer
from torch_ema import ExponentialMovingAverage
from typing import List
from tqdm import tqdm
import os
import json
import tempfile
import sys
import matplotlib
if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
plt.ioff()
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import MLFlowLogger

from src.data import RealDatasetCollection, SyntheticDatasetCollection
from src.models.utils import grad_reverse, BRTreatmentOutcomeHead, AlphaRise, bce, focal_loss
from src.metrics.eval import evaluate_subject_level

logger = logging.getLogger(__name__)


def train_eval_factual(args: dict, train_f: Dataset, val_f: Dataset, orig_hparams: DictConfig, input_size: int, model_cls,
                       tuning_criterion='rmse', **kwargs):
    """
    Globally defined method, used for ray tuning
    :param args: Hyperparameter configuration
    :param train_f: Factual train dataset
    :param val_f: Factual val dataset
    :param orig_hparams: DictConfig of original hyperparameters
    :param input_size: Input size of model, infuences concrete hyperparameter configuration
    :param model_cls: class of model
    :param kwargs: Other args
    """
    try:
        from ray import tune
    except ImportError:
        logger.error("ray[tune] not found, but is required for train_eval_factual.")
        raise

    OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)
    new_params = deepcopy(orig_hparams)
    model_cls.set_hparams(new_params.model, args, input_size, model_cls.model_type)
    if model_cls.model_type == 'decoder':
        # Passing encoder takes too much memory
        encoder_r_size = new_params.model.encoder.br_size if 'br_size' in new_params.model.encoder \
            else new_params.model.encoder.seq_hidden_units  # Using either br_size or Memory adapter
        model = model_cls(new_params, encoder_r_size=encoder_r_size, **kwargs).double()
    else:
        model = model_cls(new_params, **kwargs).double()

    train_loader = DataLoader(train_f, shuffle=True, batch_size=new_params.model[model_cls.model_type]['batch_size'],
                              drop_last=True)
    trainer = Trainer(gpus=eval(str(new_params.exp.gpus))[:1],
                      logger=None,
                      max_epochs=new_params.exp.max_epochs,
                      progress_bar_refresh_rate=0,
                      gradient_clip_val=new_params.model[model_cls.model_type]['max_grad_norm']
                      if 'max_grad_norm' in new_params.model[model_cls.model_type] else None,
                      callbacks=[AlphaRise(rate=new_params.exp.alpha_rate)])
    trainer.fit(model, train_dataloader=train_loader)

    if tuning_criterion == 'rmse':
        val_rmse_orig, val_rmse_all = model.get_normalised_masked_rmse(val_f)
        tune.report(val_rmse_orig=val_rmse_orig, val_rmse_all=val_rmse_all)
    elif tuning_criterion == 'bce':
        val_bce_orig, val_bce_all = model.get_masked_bce(val_f)
        tune.report(val_bce_orig=val_bce_orig, val_bce_all=val_bce_all)
    else:
        raise NotImplementedError()


class TimeVaryingCausalModel(LightningModule):
    """
    Abstract class for models, estimating counterfactual outcomes over time
    """

    model_type = None  # Will be defined in subclasses
    possible_model_types = None  # Will be defined in subclasses
    tuning_criterion = None

    def __init__(self,
                 args: DictConfig,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 autoregressive: bool = None,
                 has_vitals: bool = None,
                 bce_weights: np.array = None,
                 **kwargs):
        """
        Args:
            args: DictConfig of model hyperparameters
            dataset_collection: Dataset collection
            autoregressive: Flag of including previous outcomes to modelling
            has_vitals: Flag of vitals in dataset
            bce_weights: Re-weight BCE if used
            **kwargs: Other arguments
        """
        super().__init__()
        self.dataset_collection = dataset_collection
        if dataset_collection is not None:
            self.autoregressive = self.dataset_collection.autoregressive
            self.has_vitals = self.dataset_collection.has_vitals
            self.bce_weights = None  # Will be calculated, when calling preparing data
        else:
            self.autoregressive = autoregressive
            self.has_vitals = has_vitals
            self.bce_weights = bce_weights
            print(self.bce_weights)

        # General datasets parameters
        self.dim_treatments = args.model.dim_treatments
        self.dim_vitals = args.model.dim_vitals
        self.dim_static_features = args.model.dim_static_features
        self.dim_outcome = args.model.dim_outcomes
        self.dim_outcome_input = getattr(args.model, "dim_outcomes_input", self.dim_outcome)

        self.input_size = None  # Will be defined in subclasses

        self.save_hyperparameters(args)  # Will be logged to mlflow

    def _get_optimizer(self, param_optimizer: list):
        no_decay = ['bias', 'layer_norm']
        sub_args = self.hparams.model[self.model_type]
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': sub_args['optimizer']['weight_decay'],
            },
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        lr = sub_args['optimizer']['learning_rate']
        optimizer_cls = sub_args['optimizer']['optimizer_cls']
        if optimizer_cls.lower() == 'adamw':
            optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr)
        elif optimizer_cls.lower() == 'adam':
            optimizer = optim.Adam(optimizer_grouped_parameters, lr=lr)
        elif optimizer_cls.lower() == 'sgd':
            optimizer = optim.SGD(optimizer_grouped_parameters, lr=lr,
                                  momentum=sub_args['optimizer']['momentum'])
        else:
            raise NotImplementedError()

        return optimizer

    def _resolve_scheduler_t_max(self, params: dict, interval: str) -> int:
        t_max = params.get("t_max", None)
        if t_max is not None:
            try:
                t_max = int(t_max)
            except Exception:
                t_max = None
        interval = str(interval or "epoch").lower()

        if t_max is None:
            if interval in ("step", "steps"):
                est = None
                if self.trainer is not None:
                    est = getattr(self.trainer, "estimated_stepping_batches", None)
                    if est is None:
                        num_batches = getattr(self.trainer, "num_training_batches", None)
                        max_epochs = getattr(self.hparams.exp, "max_epochs", None)
                        try:
                            if num_batches is not None and max_epochs is not None:
                                est = int(num_batches) * int(max_epochs)
                        except Exception:
                            est = None
                if est is not None:
                    t_max = est
                else:
                    t_max = 100
            else:
                max_epochs = getattr(self.hparams.exp, "max_epochs", None)
                try:
                    t_max = int(max_epochs)
                except Exception:
                    t_max = None
                if t_max is None or t_max <= 0:
                    t_max = 100

        if t_max <= 0:
            t_max = 1
        return int(t_max)

    def _build_lr_scheduler(self, optimizer):
        opt_cfg = self.hparams.model[self.model_type].get('optimizer', {})
        sched_cfg = opt_cfg.get('lr_scheduler', False)
        if sched_cfg is None or sched_cfg is False:
            return None

        sched_type = "exponential"
        params = {}

        if isinstance(sched_cfg, DictConfig):
            sched_cfg = OmegaConf.to_container(sched_cfg, resolve=True)

        if isinstance(sched_cfg, dict):
            sched_type = str(sched_cfg.get("type", sched_cfg.get("name", "exponential"))).lower()
            params.update({k: v for k, v in sched_cfg.items() if k not in ("type", "name")})
        elif isinstance(sched_cfg, str):
            sched_type = sched_cfg.lower()
        elif isinstance(sched_cfg, bool):
            sched_type = "exponential"

        extra_params = opt_cfg.get("lr_scheduler_params", {})
        if isinstance(extra_params, DictConfig):
            extra_params = OmegaConf.to_container(extra_params, resolve=True)
        if isinstance(extra_params, dict):
            for k, v in extra_params.items():
                params.setdefault(k, v)

        interval = str(params.get("interval", "epoch"))
        frequency = int(params.get("frequency", 1))

        if sched_type in ("exponential", "exp"):
            gamma = float(params.get("gamma", 0.99))
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
            return {"scheduler": scheduler, "interval": interval, "frequency": frequency}
        if sched_type in ("step", "steplr"):
            step_size = int(params.get("step_size", 10))
            gamma = float(params.get("gamma", 0.1))
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            return {"scheduler": scheduler, "interval": interval, "frequency": frequency}
        if sched_type in ("cosine", "cosineannealing", "cosineannealinglr"):
            t_max = self._resolve_scheduler_t_max(params, interval)
            eta_min = float(params.get("eta_min", 0.0))
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
            return {"scheduler": scheduler, "interval": interval, "frequency": frequency}
        if sched_type in ("cosine_warmup", "warmup_cosine"):
            t_max = self._resolve_scheduler_t_max(params, interval)
            warmup_pct = float(params.get("warmup_pct", 0.05))
            warmup_steps = max(1, int(t_max * warmup_pct))
            eta_min = float(params.get("eta_min", 0.0))

            def lr_lambda(epoch):
                if epoch < warmup_steps:
                    return float(epoch + 1) / float(warmup_steps)
                progress = float(epoch - warmup_steps) / float(max(1, t_max - warmup_steps))
                cos_val = 0.5 * (1.0 + math.cos(math.pi * progress))
                if eta_min <= 0.0:
                    return cos_val
                # map cosine to [eta_min, 1.0]
                return eta_min + (1.0 - eta_min) * cos_val

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            return {"scheduler": scheduler, "interval": interval, "frequency": frequency}
        if sched_type in ("plateau", "reduceonplateau", "reducelronplateau"):
            task = str(getattr(self.hparams.exp, "task", "classification")).lower()
            default_monitor = f"{self.model_type}_val_f1_macro" if task == "classification" else f"{self.model_type}_val_loss"
            monitor = str(params.get("monitor", default_monitor))
            mode = str(params.get("mode", "max" if task == "classification" else "min")).lower()
            factor = float(params.get("factor", 0.1))
            patience = int(params.get("patience", 10))
            min_lr = float(params.get("min_lr", 0.0))
            threshold = float(params.get("threshold", 1e-4))
            cooldown = int(params.get("cooldown", 0))
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
                min_lr=min_lr,
                threshold=threshold,
                cooldown=cooldown,
            )
            return {"scheduler": scheduler, "interval": interval, "frequency": frequency, "monitor": monitor}

        raise ValueError(
            f"Unknown lr_scheduler type '{sched_type}'. "
            "Supported: exponential, step, cosine, cosine_warmup, plateau."
        )

    def _has_lr_scheduler(self) -> bool:
        opt_cfg = self.hparams.model[self.model_type].get('optimizer', {})
        sched_cfg = opt_cfg.get('lr_scheduler', False)
        if sched_cfg is None:
            return False
        if isinstance(sched_cfg, bool):
            return sched_cfg
        if isinstance(sched_cfg, str) and sched_cfg.lower() in ("false", "none", "null", "no"):
            return False
        return True

    def _get_lr_schedulers(self, optimizer):
        if not isinstance(optimizer, list):
            lr_scheduler_cfg = self._build_lr_scheduler(optimizer)
            if lr_scheduler_cfg is None:
                return {"optimizer": optimizer}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_cfg}
        configs = []
        for opt in optimizer:
            lr_scheduler_cfg = self._build_lr_scheduler(opt)
            if lr_scheduler_cfg is None:
                raise RuntimeError("lr_scheduler enabled but scheduler config resolved to None.")
            configs.append({"optimizer": opt, "lr_scheduler": lr_scheduler_cfg})
        return configs

    def configure_optimizers(self):
        optimizer = self._get_optimizer(list(self.named_parameters()))
        if self._has_lr_scheduler():
            return self._get_lr_schedulers(optimizer)
        return optimizer

    def _get_num_workers(self) -> int:
        exp_workers = getattr(self.hparams.exp, "num_workers", None)
        data_workers = getattr(self.hparams.dataset, "num_workers", 0)
        if exp_workers is None:
            try:
                return int(data_workers)
            except Exception:
                return 0
        try:
            return max(0, int(exp_workers))
        except Exception:
            logger.warning("Invalid exp.num_workers; falling back to dataset.num_workers.")
            try:
                return int(data_workers)
            except Exception:
                return 0

    def _get_dataloader_kwargs(self) -> dict:
        num_workers = self._get_num_workers()
        pin_memory = bool(getattr(self.hparams.exp, "pin_memory", False))
        persistent_workers = bool(getattr(self.hparams.exp, "persistent_workers", False))
        prefetch_factor = getattr(self.hparams.exp, "prefetch_factor", None)
        if num_workers <= 0:
            if (persistent_workers or prefetch_factor is not None) and not getattr(self, "_warned_dl_flags", False):
                logger.warning("DataLoader: num_workers=0, ignoring persistent_workers/prefetch_factor.")
                self._warned_dl_flags = True
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

    def train_dataloader(self) -> DataLoader:
        sub_args = self.hparams.model[self.model_type]
        return DataLoader(
            self.dataset_collection.train_f,
            shuffle=True,
            batch_size=sub_args['batch_size'],
            drop_last=True,
            **self._get_dataloader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_collection.val_f,
            batch_size=self.hparams.dataset.val_batch_size,
            **self._get_dataloader_kwargs(),
        )

    def get_predictions(self, dataset: Dataset) -> np.array:
        raise NotImplementedError()

    def get_propensity_scores(self, dataset: Dataset) -> np.array:
        raise NotImplementedError()

    def get_representations(self, dataset: Dataset) -> np.array:
        raise NotImplementedError()

    def get_autoregressive_predictions(self, dataset: Dataset) -> np.array:
        logger.info(f'Autoregressive Prediction for {dataset.subset_name}.')
        if self.model_type == 'decoder':  # CRNDecoder / EDCTDecoder / RMSN Decoder

            predicted_outputs = np.zeros((len(dataset), self.hparams.dataset.projection_horizon, self.dim_outcome))
            for t in range(self.hparams.dataset.projection_horizon):
                logger.info(f't = {t + 2}')

                outputs_scaled = self.get_predictions(dataset)
                predicted_outputs[:, t] = outputs_scaled[:, t]

                if t < (self.hparams.dataset.projection_horizon - 1):
                    dataset.data['prev_outputs'][:, t + 1, :] = outputs_scaled[:, t, :]
        else:
            raise NotImplementedError()

        return predicted_outputs

    def get_masked_bce(self, dataset: Dataset):
        logger.info(f'BCE calculation for {dataset.subset_name}.')
        treatment_pred = torch.tensor(self.get_propensity_scores(dataset))
        current_treatments = torch.tensor(dataset.data['current_treatments'])

        bce = (self.bce_loss(treatment_pred, current_treatments, kind='predict')).unsqueeze(-1).numpy()
        bce = bce * dataset.data['active_entries']

        # Calculation like in original paper (Masked-Averaging over datapoints (& outputs) and then non-masked time axis)
        bce_orig = bce.sum(0).sum(-1) / dataset.data['active_entries'].sum(0).sum(-1)
        bce_orig = bce_orig.mean()

        # Masked averaging over all dimensions at once
        bce_all = bce.sum() / dataset.data['active_entries'].sum()

        return bce_orig, bce_all

    def get_normalised_masked_rmse(self, dataset: Dataset, one_step_counterfactual=False):
        logger.info(f'RMSE calculation for {dataset.subset_name}.')
        outputs_scaled = self.get_predictions(dataset)
        unscale = self.hparams.exp.unscale_rmse
        percentage = self.hparams.exp.percentage_rmse

        if unscale:
            output_stds, output_means = dataset.scaling_params['output_stds'], dataset.scaling_params['output_means']
            outputs_unscaled = outputs_scaled * output_stds + output_means

            # Batch-wise masked-MSE calculation is tricky, thus calculating for full dataset at once
            mse = ((outputs_unscaled - dataset.data['unscaled_outputs']) ** 2) * dataset.data['active_entries']
        else:
            # Batch-wise masked-MSE calculation is tricky, thus calculating for full dataset at once
            mse = ((outputs_scaled - dataset.data['outputs']) ** 2) * dataset.data['active_entries']

        # Calculation like in original paper (Masked-Averaging over datapoints (& outputs) and then non-masked time axis)
        mse_orig = mse.sum(0).sum(-1) / dataset.data['active_entries'].sum(0).sum(-1)
        mse_orig = mse_orig.mean()
        rmse_normalised_orig = np.sqrt(mse_orig) / dataset.norm_const

        # Masked averaging over all dimensions at once
        mse_all = mse.sum() / dataset.data['active_entries'].sum()
        rmse_normalised_all = np.sqrt(mse_all) / dataset.norm_const

        if percentage:
            rmse_normalised_orig *= 100.0
            rmse_normalised_all *= 100.0

        if one_step_counterfactual:
            # Only considering last active entry with actual counterfactuals
            num_samples, time_dim, output_dim = dataset.data['active_entries'].shape
            last_entries = dataset.data['active_entries'] - np.concatenate([dataset.data['active_entries'][:, 1:, :],
                                                                            np.zeros((num_samples, 1, output_dim))], axis=1)
            if unscale:
                mse_last = ((outputs_unscaled - dataset.data['unscaled_outputs']) ** 2) * last_entries
            else:
                mse_last = ((outputs_scaled - dataset.data['outputs']) ** 2) * last_entries

            mse_last = mse_last.sum() / last_entries.sum()
            rmse_normalised_last = np.sqrt(mse_last) / dataset.norm_const

            if percentage:
                rmse_normalised_last *= 100.0

            return rmse_normalised_orig, rmse_normalised_all, rmse_normalised_last

        return rmse_normalised_orig, rmse_normalised_all

    def get_normalised_n_step_rmses(self, dataset: Dataset, datasets_mc: List[Dataset] = None):
        logger.info(f'RMSE calculation for {dataset.subset_name}.')
        assert self.model_type == 'decoder' or self.model_type == 'multi' or self.model_type == 'g_net' or \
               self.model_type == 'msm_regressor'
        assert hasattr(dataset, 'data_processed_seq')

        unscale = self.hparams.exp.unscale_rmse
        percentage = self.hparams.exp.percentage_rmse
        outputs_scaled = self.get_autoregressive_predictions(dataset if datasets_mc is None else datasets_mc)

        if unscale:
            output_stds, output_means = dataset.scaling_params['output_stds'], dataset.scaling_params['output_means']
            outputs_unscaled = outputs_scaled * output_stds + output_means

            mse = ((outputs_unscaled - dataset.data_processed_seq['unscaled_outputs']) ** 2) \
                * dataset.data_processed_seq['active_entries']
        else:
            mse = ((outputs_scaled - dataset.data_processed_seq['outputs']) ** 2) * dataset.data_processed_seq['active_entries']

        nan_idx = np.unique(np.where(np.isnan(dataset.data_processed_seq['outputs']))[0])
        not_nan = np.array([i for i in range(outputs_scaled.shape[0]) if i not in nan_idx])

        # Calculation like in original paper (Masked-Averaging over datapoints (& outputs) and then non-masked time axis)
        mse_orig = mse[not_nan].sum(0).sum(-1) / dataset.data_processed_seq['active_entries'][not_nan].sum(0).sum(-1)
        rmses_normalised_orig = np.sqrt(mse_orig) / dataset.norm_const

        if percentage:
            rmses_normalised_orig *= 100.0

        return rmses_normalised_orig

    @staticmethod
    def set_hparams(model_args: DictConfig, new_args: dict, input_size: int, model_type: str):
        raise NotImplementedError()

    def finetune(self, resources_per_trial: dict):
        """
        Hyperparameter tuning with ray[tune]
        """
        try:
            import ray
            from ray import tune
            from ray import ray_constants
        except ImportError as e:
            raise ImportError("ray[tune] is required for hyperparameter tuning. Please install with: pip install 'ray[tune]'") from e

        # Set large function size threshold for Ray to avoid serialization errors with large models/datasets.
        ray_constants.FUNCTION_SIZE_ERROR_THRESHOLD = 10**8  # ~ 100Mb

        self.prepare_data()
        sub_args = self.hparams.model[self.model_type]
        logger.info(f"Running hyperparameters selection with {sub_args['tune_range']} trials")
        ray.init(num_gpus=len(eval(str(self.hparams.exp.gpus))), num_cpus=4, include_dashboard=False,
                 _redis_max_memory=ray_constants.FUNCTION_SIZE_ERROR_THRESHOLD)

        hparams_grid = {k: tune.choice(v) for k, v in sub_args['hparams_grid'].items()}
        analysis = tune.run(tune.with_parameters(train_eval_factual,
                                                 input_size=self.input_size,
                                                 model_cls=self.__class__,
                                                 tuning_criterion=self.tuning_criterion,
                                                 train_f=deepcopy(self.dataset_collection.train_f),
                                                 val_f=deepcopy(self.dataset_collection.val_f),
                                                 orig_hparams=self.hparams,
                                                 autoregressive=self.autoregressive,
                                                 has_vitals=self.has_vitals,
                                                 bce_weights=self.bce_weights,
                                                 projection_horizon=self.projection_horizon
                                                 if hasattr(self, 'projection_horizon') else None),
                            resources_per_trial=resources_per_trial,
                            metric=f"val_{self.tuning_criterion}_all",
                            mode="min",
                            config=hparams_grid,
                            num_samples=sub_args['tune_range'],
                            name=f"{self.__class__.__name__}{self.model_type}",
                            max_failures=3)
        ray.shutdown()

        logger.info(f"Best hyperparameters found: {analysis.best_config}.")
        logger.info("Resetting current hyperparameters to best values.")
        self.set_hparams(self.hparams.model, analysis.best_config, self.input_size, self.model_type)

        self.__init__(self.hparams,
                      dataset_collection=self.dataset_collection,
                      encoder=self.encoder if hasattr(self, 'encoder') else None,
                      propensity_treatment=self.propensity_treatment if hasattr(self, 'propensity_treatment') else None,
                      propensity_history=self.propensity_history if hasattr(self, 'propensity_history') else None)
        return self

    def visualize(self, dataset: Dataset, index=0, artifacts_path=None):
        pass

    def bce_loss(self, treatment_pred, current_treatments, kind='predict'):
        mode = str(getattr(self.hparams.dataset, "treatment_mode", "none")).lower()
        if mode in ("none", "no", "off", "false", "null", "0") or self.dim_treatments == 0 \
                or treatment_pred is None or current_treatments is None or current_treatments.numel() == 0:
            if current_treatments is not None:
                shape = current_treatments.shape[:-1]
                return torch.zeros(shape, dtype=current_treatments.dtype, device=current_treatments.device)
            if treatment_pred is not None:
                shape = treatment_pred.shape[:-1]
                return torch.zeros(shape, dtype=treatment_pred.dtype, device=treatment_pred.device)
            return torch.zeros((), device=self.device)

        bce_weights = torch.tensor(self.bce_weights).type_as(current_treatments) if self.hparams.exp.bce_weight else None

        if kind == 'predict':
            bce_loss = bce(treatment_pred, current_treatments, mode, bce_weights)
        elif kind == 'confuse':
            uniform_treatments = torch.ones_like(current_treatments)
            if mode == 'multiclass':
                uniform_treatments *= 1 / current_treatments.shape[-1]
            elif mode == 'multilabel':
                uniform_treatments *= 0.5
            bce_loss = bce(treatment_pred, uniform_treatments, mode)
        else:
            raise NotImplementedError()
        return bce_loss

    def on_fit_start(self) -> None:  # Issue with logging not yet existing parameters in MlFlow
        if self.trainer.logger is not None:
            self.trainer.logger.filter_submodels = list(self.possible_model_types - {self.model_type})
        self._log_labeling_artifacts()

    def on_fit_end(self) -> None:  # Issue with logging not yet existing parameters in MlFlow
        if self.trainer.logger is not None:
            self.trainer.logger.filter_submodels = list(self.possible_model_types)


class BRCausalModel(TimeVaryingCausalModel):
    """
    Abstract class for models, estimating counterfactual outcomes over time with balanced representations
    """

    model_type = None  # Will be defined in subclasses
    possible_model_types = None   # Will be defined in subclasses
    tuning_criterion = 'rmse'

    def __init__(self, args: DictConfig,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 autoregressive: bool = None,
                 has_vitals: bool = None,
                 bce_weights: np.array = None,
                 **kwargs):
        """
        Args:
            args: DictConfig of model hyperparameters
            dataset_collection: Dataset collection
            autoregressive: Flag of including previous outcomes to modelling
            has_vitals: Flag of vitals in dataset
            bce_weights: Re-weight BCE if used
            **kwargs: Other arguments
        """
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)

        # Balancing representation training parameters
        self.balancing = args.exp.balancing
        self.alpha = args.exp.alpha  # Used for gradient-reversal
        self.update_alpha = args.exp.update_alpha
        self._class_weights = None
        self.multi_task = bool(getattr(args.exp, "multi_task", False))
        self.lambda_shock = float(getattr(args.exp, "lambda_shock", 0.3))
        self.outcome_loss_weight = float(getattr(args.exp, "outcome_loss_weight", 1.0))
        self._shock_errors = {"val": [], "test": []}
        self._shock_t_hist = {"train": [], "val": [], "test": []}
        self._shock_valid_counts = {
            "train": {"valid": 0, "total": 0},
            "val": {"valid": 0, "total": 0},
            "test": {"valid": 0, "total": 0},
        }
        self._checked_multi_task_train = False

    def configure_optimizers(self):
        if not hasattr(self, 'br_treatment_outcome_head'):
            raise RuntimeError(
                "CT/EDCT not fully initialised: br_treatment_outcome_head missing. "
                "Check config/backbone/ct.yaml (seq_hidden_units, br_size, fc_hidden_units, dropout_rate, etc.)."
            )
        if self.multi_task:
            optimizer = self._get_optimizer(list(self.named_parameters()))
            if self._has_lr_scheduler():
                return self._get_lr_schedulers(optimizer)
            return optimizer
        balancing_mode = None if self.balancing is None else str(self.balancing).lower()
        if self.dim_treatments == 0 and balancing_mode not in (None, "none"):
            logger.warning("balancing set but dim_treatments=0; disabling balancing in optimizer setup.")
            balancing_mode = None
        use_ema = self._use_weights_ema()
        if balancing_mode in (None, "none"):
            if bool(getattr(self.hparams.exp, "weights_ema", False)):
                logger.warning("weights_ema=True ignored because balancing=none.")
            use_ema = False

        if balancing_mode in (None, "none") or (balancing_mode == 'grad_reverse' and not use_ema):  # one optimizer
            optimizer = self._get_optimizer(list(self.named_parameters()))

            if self._has_lr_scheduler():
                return self._get_lr_schedulers(optimizer)

            return optimizer

        else:  # two optimizers - simultaneous gradient descent update
            treatment_head_params = \
                ['br_treatment_outcome_head.' + s for s in self.br_treatment_outcome_head.treatment_head_params]
            treatment_head_params = \
                [k for k in dict(self.named_parameters()) for param in treatment_head_params if k.startswith(param)]
            non_treatment_head_params = [k for k in dict(self.named_parameters()) if k not in treatment_head_params]

            assert len(treatment_head_params + non_treatment_head_params) == len(list(self.named_parameters()))

            treatment_head_params = [(k, v) for k, v in dict(self.named_parameters()).items() if k in treatment_head_params]
            non_treatment_head_params = [(k, v) for k, v in dict(self.named_parameters()).items()
                                         if k in non_treatment_head_params]

            if use_ema:
                self.ema_treatment = ExponentialMovingAverage([par[1] for par in treatment_head_params],
                                                              decay=self.hparams.exp.beta)
                self.ema_non_treatment = ExponentialMovingAverage([par[1] for par in non_treatment_head_params],
                                                                  decay=self.hparams.exp.beta)

            treatment_head_optimizer = self._get_optimizer(treatment_head_params)
            non_treatment_head_optimizer = self._get_optimizer(non_treatment_head_params)

            if self._has_lr_scheduler():
                return self._get_lr_schedulers([non_treatment_head_optimizer, treatment_head_optimizer])

            return [non_treatment_head_optimizer, treatment_head_optimizer]

    def optimizer_step(self, epoch: int = None, batch_idx: int = None, optimizer=None, optimizer_idx: int = None, *args,
                       **kwargs) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, *args, **kwargs)
        if self._use_weights_ema() and optimizer_idx == 0:
            self.ema_non_treatment.update()
        elif self._use_weights_ema() and optimizer_idx == 1:
            self.ema_treatment.update()

    def _calculate_bce_weights(self) -> None:
        if self.dim_treatments == 0:
            logger.warning("dim_treatments=0; skipping BCE weight computation.")
            return
        if self.hparams.dataset.treatment_mode == 'multiclass':
            current_treatments = self.dataset_collection.train_f.data['current_treatments']
            current_treatments = current_treatments.reshape(-1, current_treatments.shape[-1])
            current_treatments = current_treatments[self.dataset_collection.train_f.data['active_entries'].flatten().astype(bool)]
            current_treatments = np.argmax(current_treatments, axis=1)

            self.bce_weights = len(current_treatments) / np.bincount(current_treatments) / len(np.bincount(current_treatments))
        else:
            raise NotImplementedError()

    def on_fit_start(self) -> None:  # Issue with logging not yet existing parameters in MlFlow
        if self.trainer.logger is not None:
            self.trainer.logger.filter_submodels = ['decoder'] if self.model_type == 'encoder' else ['encoder']
        self._log_labeling_artifacts()

    def on_fit_end(self) -> None:  # Issue with logging not yet existing parameters in MlFlow
        if self.trainer.logger is not None:
            self.trainer.logger.filter_submodels = ['encoder', 'decoder']

    def on_train_start(self) -> None:
        # Ensure labeling artifacts are logged after sanity checks.
        self._log_labeling_artifacts()

    def _use_weights_ema(self) -> bool:
        return bool(getattr(self.hparams.exp, "weights_ema", False)) and self.balancing not in (None, "none")

    def _extract_class_targets(self, outputs: torch.Tensor) -> torch.Tensor:
        # outputs: (B, T, C) one-hot or (B, T, 1) labels
        if outputs.ndim >= 3 and outputs.shape[-1] > 1:
            return torch.argmax(outputs, dim=-1).long()
        return outputs.squeeze(-1).long()

    def _compute_class_weights_from_train(self) -> np.ndarray:
        if self.dataset_collection is None or self.dataset_collection.train_f is None:
            raise RuntimeError("Dataset collection is not available for class-weight computation.")
        outputs = self.dataset_collection.train_f.data.get("outputs")
        active = self.dataset_collection.train_f.data.get("active_entries")
        if outputs is None or active is None:
            raise RuntimeError("Train data missing outputs/active_entries for class-weight computation.")

        y = outputs
        if y.ndim >= 3 and y.shape[-1] > 1:
            y = np.argmax(y, axis=-1)
        y = y.reshape(-1)
        mask = active.reshape(-1).astype(bool)
        y = y[mask]
        y = y.astype(int, copy=False)

        c = int(self.dim_outcome)
        counts = np.bincount(y, minlength=c)
        total = counts.sum()
        if total == 0:
            raise RuntimeError("Empty training labels after masking; cannot compute class weights.")
        if np.any(counts == 0):
            logger.warning(f"Some classes have zero count in training data: {counts.tolist()}. "
                           "Weights for missing classes will be set to 0.")
        counts_safe = np.where(counts == 0, 1, counts)
        weights = total / (c * counts_safe.astype(float))
        weights = np.where(counts == 0, 0.0, weights)
        return weights.astype(np.float32)

    def _resolve_class_weights(self) -> torch.Tensor:
        if self.hparams.exp.task != "classification":
            return None
        mode = getattr(self.hparams.exp, "class_weights_mode", "none")
        if mode is None or str(mode).lower() in ("none", "null", "false"):
            return None

        if str(mode).lower() == "manual":
            weights = getattr(self.hparams.exp, "class_weights", None)
            if weights is None:
                logger.warning("class_weights_mode=manual but exp.class_weights is None; ignoring.")
                return None
            try:
                weights = [float(w) for w in weights]
            except (ValueError, TypeError):
                logger.error("class_weights must be a list of floats; ignoring manual weights.")
                return None
            
            if len(weights) != int(self.dim_outcome):
                logger.error(f"class_weights length ({len(weights)}) does not match dim_outcome ({self.dim_outcome}); ignoring manual weights.")
                return None
            if any(w < 0 for w in weights):
                logger.error("class_weights must all be non-negative; ignoring manual weights.")
                return None
            
            self._class_weights = torch.tensor(weights, dtype=torch.float32)
            logger.info(f"Using manual class weights: {self._class_weights.tolist()}")

        elif str(mode).lower() == "balanced":
            if self._class_weights is None:
                try:
                    weights = self._compute_class_weights_from_train()
                    self._class_weights = torch.tensor(weights, dtype=torch.float32)
                    logger.info(f"Using computed balanced class weights: {weights.tolist()}")
                except Exception as e:
                    logger.warning(f"Failed to compute balanced class weights: {e}")
                    return None
        else:
            logger.warning(f"Unknown class_weights_mode={mode}; ignoring.")
            return None

        if self._class_weights is None:
            return None
        return self._class_weights.to(device=self.device, dtype=torch.float32)

    def _classification_loss(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
        """
        logits: (B, T, C)
        targets: (B, T)
        mask: (B, T) float/bool
        """
        logits_f = logits.reshape(-1, logits.shape[-1])
        tgt_f = targets.reshape(-1)
        mask_f = mask.reshape(-1).float()

        weights = self._resolve_class_weights()
        use_focal = bool(getattr(self.hparams.exp, "use_focal_loss", False))
        gamma = float(getattr(self.hparams.exp, "focal_gamma", 2.0))
        label_smoothing = float(getattr(self.hparams.exp, "label_smoothing", 0.0))
        if label_smoothing < 0.0 or label_smoothing >= 1.0:
            logger.warning("label_smoothing should be in [0, 1); ignoring.")
            label_smoothing = 0.0

        if use_focal:
            per_example = focal_loss(logits_f, tgt_f, gamma=gamma, weight=weights, reduction="none")
        else:
            if label_smoothing > 0.0:
                if not hasattr(self, "_ce_supports_label_smoothing"):
                    try:
                        sig = inspect.signature(F.cross_entropy)
                        self._ce_supports_label_smoothing = "label_smoothing" in sig.parameters
                    except Exception:
                        self._ce_supports_label_smoothing = False
                if self._ce_supports_label_smoothing:
                    per_example = F.cross_entropy(
                        logits_f, tgt_f, reduction="none", weight=weights, label_smoothing=label_smoothing
                    )
                else:
                    if not getattr(self, "_warned_label_smoothing", False):
                        logger.warning("label_smoothing requested but torch does not support it; ignoring.")
                        self._warned_label_smoothing = True
                    per_example = F.cross_entropy(logits_f, tgt_f, reduction="none", weight=weights)
            else:
                per_example = F.cross_entropy(logits_f, tgt_f, reduction="none", weight=weights)

        loss = (per_example * mask_f).sum() / mask_f.sum().clamp(min=1.0)
        return loss

    @staticmethod
    def _mask_fill_value(dtype: torch.dtype) -> float:
        if dtype in (torch.float16, torch.bfloat16):
            return -1e4
        try:
            return float(torch.finfo(dtype).min)
        except Exception:
            return -1e9

    def _pool_br(self, br: torch.Tensor, active_entries: torch.Tensor) -> torch.Tensor:
        mask = active_entries.squeeze(-1).float()
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (br * mask.unsqueeze(-1)).sum(dim=1) / denom

    def _compute_multi_task_losses(self, batch: dict, br: torch.Tensor):
        if not hasattr(self, "shock_head") or not hasattr(self, "outcome_head"):
            raise RuntimeError("multi_task enabled but shock_head/outcome_head not initialized.")
        if "y_outcome" not in batch:
            raise RuntimeError("multi_task requires batch['y_outcome'] (record-level labels).")
        if "t_shock" not in batch:
            raise RuntimeError("multi_task requires batch['t_shock'] (shock index).")

        active_entries = batch["active_entries"]
        outcome_logits = self.outcome_head(self._pool_br(br, active_entries))
        y_outcome = batch["y_outcome"].long()
        outcome_loss = F.cross_entropy(outcome_logits, y_outcome)

        shock_logits = self.shock_head(br).squeeze(-1)  # (B,T)
        mask_bt = active_entries.squeeze(-1).bool()
        shock_logits = shock_logits.masked_fill(~mask_bt, self._mask_fill_value(shock_logits.dtype))
        t_shock = batch["t_shock"].long()
        if shock_logits.shape[1] > 0:
            t_shock = torch.clamp(t_shock, min=0, max=shock_logits.shape[1] - 1)

        valid = mask_bt.any(dim=1)
        if "shock_valid" in batch:
            valid = valid & batch["shock_valid"].view(-1).bool()

        shock_loss_for_total = torch.zeros((), dtype=shock_logits.dtype, device=shock_logits.device)
        shock_loss_log = torch.tensor(float("nan"), dtype=shock_logits.dtype, device=shock_logits.device)
        if valid.any():
            shock_loss_for_total = F.cross_entropy(shock_logits[valid], t_shock[valid])
            shock_loss_log = shock_loss_for_total

        total_loss = self.outcome_loss_weight * outcome_loss + self.lambda_shock * shock_loss_for_total
        shock_valid_frac = valid.float().mean() if valid.numel() > 0 else torch.zeros((), device=valid.device)
        return (total_loss, outcome_loss, shock_loss_for_total, shock_loss_log, shock_valid_frac,
                outcome_logits, shock_logits, mask_bt, t_shock, valid)

    def training_step(self, batch, batch_ind, optimizer_idx=0):
        """
        Lightning training step.

        EEGMMIDB sanity flags (train-only):
        - `CT_SHUFFLE_TRAIN_INPUTS=1` enables input shuffling sanity checks.
          - MODE=intra_sequence_lockstep: shuffles timesteps within each sequence (order-only).
          - MODE=inter_batch_decouple: permutes batch-level *inputs* while keeping labels fixed
            (strong decoupling sanity; expected to drop to chance in classification).
        """
        sanity_shuffle_inputs = (str(os.getenv("CT_SHUFFLE_TRAIN_INPUTS", "0")).strip() == "1")
        if sanity_shuffle_inputs:
            mode = str(os.getenv("CT_SHUFFLE_TRAIN_INPUTS_MODE", "")).strip().lower() or "intra_sequence_lockstep"
            if mode == "inter_batch_decouple" and isinstance(batch, dict):
                # Strong sanity: decouple inputs↔labels by permuting batch-level inputs while keeping labels fixed.
                b = None
                device = None
                for k in ("prev_outputs", "current_covariates", "prev_treatments", "current_treatments", "vitals", "static_features"):
                    x = batch.get(k, None)
                    if isinstance(x, torch.Tensor) and x.ndim >= 1:
                        b = int(x.shape[0])
                        device = x.device
                        break

                if b is not None and b > 1 and device is not None:
                    perm = torch.randperm(b, device=device)
                    identity = torch.arange(b, device=device)
                    if torch.equal(perm, identity):
                        perm = torch.roll(perm, shifts=1, dims=0)
                    if int(perm[0].item()) == 0:
                        perm = perm.clone()
                        perm[0], perm[1] = perm[1], perm[0]

                    # Keep this list explicit and conservative: permute only model inputs, never labels/masks/IDs.
                    keys_to_permute = (
                        "prev_outputs",
                        "current_covariates",
                        "prev_treatments",
                        "current_treatments",
                        "vitals",
                        "static_features",
                    )
                    seen = set()
                    permuted_keys = []

                    def _probe_scalar(t: torch.Tensor) -> float:
                        with torch.no_grad():
                            if t.ndim >= 3:
                                return float(t[0, 0].mean().detach().cpu().item())
                            if t.ndim == 2:
                                return float(t[0].mean().detach().cpu().item())
                            return float(t.reshape(-1)[0].detach().cpu().item())

                    probe_key = None
                    probe_before = float("nan")
                    for k in ("prev_outputs", "current_covariates", "prev_treatments", "current_treatments", "vitals", "static_features"):
                        x = batch.get(k, None)
                        if isinstance(x, torch.Tensor) and x.ndim >= 1 and int(x.shape[0]) == b:
                            probe_key = k
                            probe_before = _probe_scalar(x)
                            break

                    for k in keys_to_permute:
                        if k in seen:
                            continue
                        seen.add(k)
                        x = batch.get(k, None)
                        if isinstance(x, torch.Tensor) and x.ndim >= 1 and int(x.shape[0]) == b:
                            batch[k] = x[perm]
                            permuted_keys.append(k)

                    probe_after = float("nan")
                    if probe_key is not None and probe_key in permuted_keys:
                        probe_after = _probe_scalar(batch[probe_key])

                    if not getattr(self, "_sanity_logged_inter_batch_decouple", False):
                        is_zero = True
                        try:
                            is_zero = (self.trainer is None) or bool(getattr(self.trainer, "is_global_zero", True))
                        except Exception:
                            is_zero = True
                        if is_zero:
                            logger.warning(
                                "SANITY INPUT-SHUFFLE (train, inter-batch-decouple): "
                                f"CT_SHUFFLE_TRAIN_INPUTS=1, MODE={mode}, B={b}, "
                                f"perm_head={perm[:min(5, b)].detach().cpu().tolist()}, "
                                f"probe_key={probe_key}, probe_before={probe_before:.6g}, probe_after={probe_after:.6g}, "
                                f"permuted_keys={permuted_keys}"
                            )
                        self._sanity_logged_inter_batch_decouple = True
            elif mode not in ("intra_sequence_lockstep", "inter_batch_decouple"):
                if not getattr(self, "_sanity_warned_unknown_input_shuffle_mode", False):
                    logger.warning(
                        "SANITY: Unknown CT_SHUFFLE_TRAIN_INPUTS_MODE='%s'; treating as 'intra_sequence_lockstep' (no train-step shuffle).",
                        mode,
                    )
                    self._sanity_warned_unknown_input_shuffle_mode = True

        for par in self.parameters():
            par.requires_grad = True
        if self.multi_task and self.hparams.exp.task == 'classification':
            # Multi-task: record-level outcome + shock localization
            _, outcome_pred, br = self(batch)
            (total_loss, outcome_loss, shock_loss_for_total, shock_loss_log, shock_valid_frac,
             _, _, _, t_shock, valid) = self._compute_multi_task_losses(batch, br)

            if not torch.isfinite(total_loss):
                raise RuntimeError("multi_task total_loss is not finite.")

            if not self._checked_multi_task_train:
                if not torch.isfinite(outcome_loss):
                    raise RuntimeError("multi_task outcome_loss is not finite on first batch.")
                if float(outcome_loss.detach().cpu().item()) <= 0.2:
                    raise RuntimeError("multi_task outcome_loss too small on first batch (<=0.2).")
                expected = self.outcome_loss_weight * outcome_loss + self.lambda_shock * shock_loss_for_total
                diff = torch.abs(total_loss - expected)
                tol = 1e-4 + 1e-3 * float(torch.abs(expected).detach().cpu().item())
                if float(diff.detach().cpu().item()) > tol:
                    raise RuntimeError("multi_task total_loss does not match weighted sum.")
                self._checked_multi_task_train = True

            self._update_shock_stats("train", valid, t_shock)

            targets_win = self._extract_class_targets(batch["outputs"])
            mask_win = batch["active_entries"].squeeze(-1)
            window_outcome_loss = self._classification_loss(outcome_pred, targets_win, mask_win)

            self.log(f'{self.model_type}_train_total_loss', total_loss, on_epoch=True, on_step=False,
                     prog_bar=True, sync_dist=True)
            self.log(f'{self.model_type}_train_loss', total_loss, on_epoch=True, on_step=False, sync_dist=True)
            self.log("loss", total_loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
            self.log(f'{self.model_type}_train_outcome_loss_record_head', outcome_loss,
                     on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_train_outcome_loss', outcome_loss,
                     on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_train_outcome_loss_window', window_outcome_loss,
                     on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_train_shock_loss', shock_loss_log, on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_train_shock_valid_frac', shock_valid_frac, on_epoch=True, on_step=False,
                     sync_dist=True)
            return total_loss

        balancing_mode = None if self.balancing is None else str(self.balancing).lower()

        if optimizer_idx == 0:  # grad reversal or domain confusion representation update
            if self._use_weights_ema():
                with self.ema_treatment.average_parameters():
                    treatment_pred, outcome_pred, _ = self(batch)
            else:
                treatment_pred, outcome_pred, _ = self(batch)

            ##########################################################################################################
            # Original implementation before Classifier head refactoring
            #
            # mse_loss = F.mse_loss(outcome_pred, batch['outputs'], reduce=False)
            # if self.balancing == 'grad_reverse':
            #     bce_loss = self.bce_loss(treatment_pred, batch['current_treatments'].double(), kind='predict')
            # elif self.balancing == 'domain_confusion':
            #     bce_loss = self.bce_loss(treatment_pred, batch['current_treatments'].double(), kind='confuse')
            #     bce_loss = self.br_treatment_outcome_head.alpha * bce_loss
            # else:
            #     raise NotImplementedError()
            #
            # # Masking for shorter sequences
            # # Attention! Averaging across all the active entries (= sequence masks) for full batch
            # bce_loss = (batch['active_entries'].squeeze(-1) * bce_loss).sum() / batch['active_entries'].sum()
            # mse_loss = (batch['active_entries'] * mse_loss).sum() / batch['active_entries'].sum()
            #
            # loss = bce_loss + mse_loss
            ##########################################################################################################

            # BCE domain/treatment loss (optional)
            bce_loss_raw = torch.zeros((), dtype=outcome_pred.dtype, device=outcome_pred.device)
            if balancing_mode in ('grad_reverse', 'domain_confusion'):
                if self.dim_treatments == 0 or treatment_pred is None:
                    if not getattr(self, "_warned_no_treatments", False):
                        logger.warning("balancing requested but dim_treatments=0; skipping treatment loss.")
                        self._warned_no_treatments = True
                    balancing_mode = None
                else:
                    if balancing_mode == 'grad_reverse':
                        bce_loss_raw = self.bce_loss(treatment_pred, batch['current_treatments'].double(), kind='predict')
                    elif balancing_mode == 'domain_confusion':
                        bce_loss_raw = self.bce_loss(treatment_pred, batch['current_treatments'].double(), kind='confuse')
                        bce_loss_raw = self.br_treatment_outcome_head.alpha * bce_loss_raw

                    # Masking for shorter sequences
                    bce_loss_raw = (batch['active_entries'].squeeze(-1) * bce_loss_raw).sum() / batch['active_entries'].sum()

            bce_coef = 1.0
            if self.hparams.exp.task == 'classification':
                bce_coef = float(getattr(self.hparams.exp, "classification_bce_coef", 0.0))
            domain_confusion_coef = float(getattr(self.hparams.exp, "domain_confusion_coef", 1.0))
            bce_loss = bce_loss_raw * bce_coef * domain_confusion_coef

            # Outcome loss classification or regression
            if self.hparams.exp.task == 'classification':
                targets = self._extract_class_targets(batch['outputs'])  # (B,T)
                mask_bt = batch['active_entries'].squeeze(-1)            # (B,T)
                outcome_loss = self._classification_loss(outcome_pred, targets, mask_bt)
                loss = outcome_loss + bce_loss

                # Logs for classification
                self.log(f'{self.model_type}_train_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
                self.log(f'{self.model_type}_train_bce_loss', bce_loss, on_epoch=True, on_step=False, sync_dist=True)
                self.log(f'{self.model_type}_train_bce_loss_raw', bce_loss_raw, on_epoch=True, on_step=False, sync_dist=True)
                self.log(f'{self.model_type}_train_ce_loss', outcome_loss, on_epoch=True, on_step=False, sync_dist=True)

            else:      # Regression
                mse_loss = F.mse_loss(outcome_pred, batch['outputs'], reduce=False)
                mse_loss = (batch['active_entries'] * mse_loss).sum() / batch['active_entries'].sum()

                loss = bce_loss + mse_loss

                # Logs for regression
                self.log(f'{self.model_type}_train_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
                self.log(f'{self.model_type}_train_bce_loss', bce_loss, on_epoch=True, on_step=False, sync_dist=True)
                self.log(f'{self.model_type}_train_bce_loss_raw', bce_loss_raw, on_epoch=True, on_step=False, sync_dist=True)
                self.log(f'{self.model_type}_train_mse_loss', mse_loss, on_epoch=True, on_step=False, sync_dist=True)

            self.log(f'{self.model_type}_alpha', self.br_treatment_outcome_head.alpha, on_epoch=True, on_step=False, sync_dist=True)

            return loss

        elif optimizer_idx == 1:  # domain classifier update
            if self.dim_treatments == 0:
                return torch.zeros((), device=self.device)
            if self._use_weights_ema():
                with self.ema_non_treatment.average_parameters():
                    treatment_pred, _, _ = self(batch, detach_treatment=True)
            else:
                treatment_pred, _, _ = self(batch, detach_treatment=True)

            bce_loss = self.bce_loss(treatment_pred, batch['current_treatments'].double(), kind='predict')
            if balancing_mode == 'domain_confusion':
                bce_loss = self.br_treatment_outcome_head.alpha * bce_loss

            # Masking for shorter sequences
            # Attention! Averaging across all the active entries (= sequence masks) for full batch
            bce_loss = (batch['active_entries'].squeeze(-1) * bce_loss).sum() / batch['active_entries'].sum()
            bce_coef = 1.0
            if self.hparams.exp.task == 'classification':
                bce_coef = float(getattr(self.hparams.exp, "classification_bce_coef", 0.0))
            domain_confusion_coef = float(getattr(self.hparams.exp, "domain_confusion_coef", 1.0))
            bce_loss_scaled = bce_loss * bce_coef * domain_confusion_coef
            self.log(f'{self.model_type}_train_bce_loss_cl', bce_loss_scaled, on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_train_bce_loss_cl_raw', bce_loss, on_epoch=True, on_step=False, sync_dist=True)

            return bce_loss_scaled

    # Classification evaluation helpers (confusion matrix, metrics)
    def _reset_confmat(self, stage: str):
        if self.hparams.exp.task != 'classification':
            return
        c = int(self.dim_outcome)
        cm = torch.zeros((c, c), dtype=torch.long, device=self.device)
        if stage == "val":
            self._cm_val = cm
        elif stage == "test":
            self._cm_test = cm

    def _update_confmat(self, stage: str, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
        # preds, targets: (B,T) ; mask: (B,T) bool/0-1
        if self.hparams.exp.task != 'classification':
            return
        c = int(self.dim_outcome)

        preds_f = preds.reshape(-1)
        targets_f = targets.reshape(-1)
        mask_f = mask.reshape(-1).bool()

        preds_m = preds_f[mask_f]
        targets_m = targets_f[mask_f]

        idx = targets_m * c + preds_m
        cm_add = torch.bincount(idx, minlength=c * c).reshape(c, c)

        if stage == "val":
            self._cm_val += cm_add
        elif stage == "test":
            self._cm_test += cm_add

    def _log_confmat(self, stage: str):
        if self.hparams.exp.task != 'classification':
            return
        if self.trainer is not None and hasattr(self.trainer, "is_global_zero") and not self.trainer.is_global_zero:
            return

        cm = self._cm_val if stage == "val" else self._cm_test
        cm_cpu = cm.detach().cpu().numpy()
        total = cm_cpu.sum()
        if total == 0:
            return

        # Global accuracy
        acc = float((cm_cpu.diagonal().sum()) / total)
        self.log(f"{self.model_type}_{stage}_acc", acc, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)

        # Precision/Recall/F1 by classes
        tp = cm_cpu.diagonal().astype(float)
        fp = cm_cpu.sum(axis=0).astype(float) - tp
        fn = cm_cpu.sum(axis=1).astype(float) - tp

        eps = 1e-12
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        macro_f1 = float(np.mean(f1))
        balanced_acc = float(np.mean(recall))
        self.log(f"{self.model_type}_{stage}_f1_macro", macro_f1, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f"{self.model_type}_{stage}_balanced_acc", balanced_acc, on_epoch=True, on_step=False, sync_dist=True)

        for k in range(len(tp)):
            self.log(f"{self.model_type}_{stage}_precision_c{k}", float(precision[k]), on_epoch=True, on_step=False, sync_dist=True)
            self.log(f"{self.model_type}_{stage}_recall_c{k}", float(recall[k]), on_epoch=True, on_step=False, sync_dist=True)
            self.log(f"{self.model_type}_{stage}_f1_c{k}", float(f1[k]), on_epoch=True, on_step=False, sync_dist=True)

        # Save confusion matrix (MLflow)
        client, run_id = self._get_mlflow_client_and_run_id()
        if client is None or run_id is None:
            logger.warning("MLflow logger not found; confusion matrix artifact not logged.")
            return

        labels = self._get_label_names()

        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, f"confusion_matrix_{stage}.png")

        fig, ax = plt.subplots(figsize=(6, 5))
        try:
            im = ax.imshow(cm_cpu, interpolation="nearest")
            ax.set_title(f"Confusion Matrix ({stage})")
            fig.colorbar(im, ax=ax)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

            # valeurs dans les cases
            for i in range(cm_cpu.shape[0]):
                for j in range(cm_cpu.shape[1]):
                    ax.text(j, i, str(cm_cpu[i, j]), ha="center", va="center")

            fig.tight_layout()
            fig.savefig(path, dpi=200, bbox_inches="tight")
        finally:
            plt.close(fig)

        # log MLflow artifact
        try:
            logger.debug(f"Logging confusion matrix to MLflow run_id={run_id}: {path}")
            client.log_artifact(run_id, path, artifact_path="confusion_matrices")
        except Exception as e:
            logger.warning(f"Failed to log confusion matrix artifact: {e}")

    def _resolve_subset_name(self, stage: str, dataloader_idx: int) -> str:
        # Works even when user passes custom dataloaders to trainer.test(...)
        try:
            if stage == "val" and hasattr(self.trainer, "val_dataloaders"):
                dl = self.trainer.val_dataloaders[dataloader_idx]
            elif stage == "test" and hasattr(self.trainer, "test_dataloaders"):
                dl = self.trainer.test_dataloaders[dataloader_idx]
            else:
                dl = None
            if dl is not None and hasattr(dl, "dataset") and hasattr(dl.dataset, "subset_name"):
                return dl.dataset.subset_name
        except Exception:
            pass
        return stage

    def _get_label_names(self):
        names = getattr(self.hparams.dataset, "label_names", None)
        c = int(self.dim_outcome)
        if names is not None:
            names = list(names)
            if len(names) == c:
                return [str(n) for n in names]
            logger.warning("dataset.label_names length mismatch with dim_outcome; falling back to numeric labels.")
        return [str(i) for i in range(c)]

    def _get_mlflow_client_and_run_id(self):
        if self.trainer is None:
            return None, None
        loggers = []
        if hasattr(self.trainer, "loggers") and self.trainer.loggers:
            loggers = self.trainer.loggers if isinstance(self.trainer.loggers, list) else [self.trainer.loggers]
        elif getattr(self.trainer, "logger", None) is not None:
            loggers = [self.trainer.logger]
        for lg in loggers:
            if isinstance(lg, MLFlowLogger):
                return lg.experiment, lg.run_id
        return None, None

    def _log_labeling_artifacts(self):
        if self.dataset_collection is None:
            return
        if self.trainer is not None and getattr(self.trainer, "sanity_checking", False):
            return
        if self.trainer is not None and hasattr(self.trainer, "is_global_zero") and not self.trainer.is_global_zero:
            return

        paths = []
        for attr in ("labeling_report_path", "labeling_thresholds_path", "shock_target_report_path"):
            path = getattr(self.dataset_collection, attr, None)
            if path and os.path.exists(path):
                paths.append(path)

        if not paths:
            return

        client, run_id = self._get_mlflow_client_and_run_id()
        if client is None or run_id is None:
            logger.warning("MLflow logger not found; labeling artifacts not logged.")
            return

        for path in paths:
            try:
                client.log_artifact(run_id, path, artifact_path="labels")
            except Exception as e:
                logger.warning(f"Failed to log labeling artifact {path}: {e}")

    def _init_pred_buffers(self):
        if not hasattr(self, "_pred_samples"):
            self._pred_samples = {"val": [], "test": []}

    def _reset_pred_samples(self, stage: str):
        if not getattr(self.hparams.exp, "log_predictions", False):
            return
        if stage != "val":
            return
        self._init_pred_buffers()
        self._pred_samples[stage] = []

    def _init_pred_full_buffers(self):
        if not hasattr(self, "_pred_full"):
            self._pred_full = {"val": [], "test": []}

    def _reset_pred_full(self, stage: str):
        if not getattr(self.hparams.exp, "export_predictions", False):
            return
        self._init_pred_full_buffers()
        self._pred_full[stage] = []

    def _collect_pred_samples(self, stage: str, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
        if not getattr(self.hparams.exp, "log_predictions", False):
            return
        if stage != "val":
            return
        if self.trainer is not None and hasattr(self.trainer, "is_global_zero") and not self.trainer.is_global_zero:
            return
        if self.trainer is not None and getattr(self.trainer, "sanity_checking", False):
            return

        self._init_pred_buffers()
        limit = int(getattr(self.hparams.exp, "log_predictions_n", 200))
        if limit <= 0:
            return
        remaining = limit - len(self._pred_samples[stage])
        if remaining <= 0:
            return

        mask_f = mask.reshape(-1).bool()
        if mask_f.sum() == 0:
            return

        logits_f = logits.reshape(-1, logits.shape[-1])[mask_f]
        targets_f = targets.reshape(-1)[mask_f]
        probs = torch.softmax(logits_f, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        n = probs.shape[0]
        if n > remaining:
            idx = torch.randperm(n, device=probs.device)[:remaining]
            probs = probs[idx]
            preds = preds[idx]
            targets_f = targets_f[idx]

        probs_cpu = probs.detach().cpu().numpy()
        preds_cpu = preds.detach().cpu().numpy()
        targets_cpu = targets_f.detach().cpu().numpy()
        for i in range(len(targets_cpu)):
            self._pred_samples[stage].append({
                "y_true": int(targets_cpu[i]),
                "y_pred": int(preds_cpu[i]),
                "probs": probs_cpu[i].tolist(),
            })

    def _collect_pred_full(self, stage: str, logits: torch.Tensor, targets: torch.Tensor,
                           mask: torch.Tensor, batch: dict):
        if not getattr(self.hparams.exp, "export_predictions", False):
            return
        if self.hparams.exp.task != "classification":
            return
        if self.trainer is not None and hasattr(self.trainer, "is_global_zero") and not self.trainer.is_global_zero:
            return
        if self.trainer is not None and getattr(self.trainer, "sanity_checking", False):
            return

        self._init_pred_full_buffers()

        mask_f = mask.reshape(-1).bool()
        if mask_f.sum() == 0:
            return

        logits_f = logits.reshape(-1, logits.shape[-1])[mask_f]
        targets_f = targets.reshape(-1)[mask_f]
        probs = torch.softmax(logits_f, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        entry = {
            "y_true": targets_f.detach().cpu().numpy(),
            "y_pred": preds.detach().cpu().numpy(),
            "probs": probs.detach().cpu().numpy(),
        }
        if bool(getattr(self.hparams.exp, "export_predictions_logits", False)):
            entry["logits"] = logits_f.detach().cpu().numpy()

        # Optional identifiers if present in batch
        b, t = mask.shape[0], mask.shape[1]
        time_idx = torch.arange(t, device=mask.device).view(1, -1).expand(b, t).reshape(-1)[mask_f]
        entry["time_idx"] = time_idx.detach().cpu().numpy()

        id_keys = ["original_index", "index", "subject_id", "record_id", "run_id"]
        for k in id_keys:
            if k in batch:
                seq_ids = batch[k]
                if isinstance(seq_ids, torch.Tensor) and seq_ids.numel() == b:
                    seq_ids = seq_ids.view(-1, 1).expand(b, t).reshape(-1)[mask_f]
                    entry[k] = seq_ids.detach().cpu().numpy()

        self._pred_full[stage].append(entry)

    def _collect_pred_full_record_head(self, stage: str, logits: torch.Tensor, targets: torch.Tensor, batch: dict):
        if not getattr(self.hparams.exp, "export_predictions", False):
            return
        if self.hparams.exp.task != "classification":
            return
        if self.trainer is not None and hasattr(self.trainer, "is_global_zero") and not self.trainer.is_global_zero:
            return
        if self.trainer is not None and getattr(self.trainer, "sanity_checking", False):
            return

        self._init_pred_full_buffers()

        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        entry = {
            "y_true": targets.detach().cpu().numpy(),
            "y_pred": preds.detach().cpu().numpy(),
            "probs": probs.detach().cpu().numpy(),
        }
        if bool(getattr(self.hparams.exp, "export_predictions_logits", False)):
            entry["logits"] = logits.detach().cpu().numpy()

        id_keys = ["original_index", "index", "subject_id", "record_id", "run_id"]
        for k in id_keys:
            if k in batch:
                seq_ids = batch[k]
                if isinstance(seq_ids, torch.Tensor) and seq_ids.numel() == logits.shape[0]:
                    entry[k] = seq_ids.detach().cpu().numpy()

        self._pred_full[stage].append(entry)

    @rank_zero_only
    def _log_prediction_artifacts(self, stage: str):
        if not getattr(self.hparams.exp, "log_predictions", False):
            return
        if stage != "val":
            return
        if self.trainer is not None and getattr(self.trainer, "sanity_checking", False):
            return
        client, run_id = self._get_mlflow_client_and_run_id()
        if client is None or run_id is None:
            logger.warning("MLflow logger not found; prediction artifacts not logged.")
            return

        self._init_pred_buffers()
        samples = self._pred_samples.get(stage, [])
        if not samples:
            return

        try:
            import pandas as pd
            from sklearn.metrics import classification_report
        except Exception as e:
            logger.warning(f"Prediction artifact logging skipped: {e}")
            return

        labels = self._get_label_names()
        rows = []
        for s in samples:
            row = {
                "y_true": s["y_true"],
                "y_pred": s["y_pred"],
                "y_true_label": labels[s["y_true"]] if s["y_true"] < len(labels) else str(s["y_true"]),
                "y_pred_label": labels[s["y_pred"]] if s["y_pred"] < len(labels) else str(s["y_pred"]),
            }
            probs = s["probs"]
            for k in range(len(labels)):
                row[f"p{k}"] = float(probs[k]) if k < len(probs) else 0.0
            rows.append(row)

        df = pd.DataFrame(rows)
        tmpdir = tempfile.mkdtemp()
        csv_path = os.path.join(tmpdir, "predictions_val_sample.csv")
        report_path = os.path.join(tmpdir, "classification_report_val_sample.txt")
        df.to_csv(csv_path, index=False)

        y_true = df["y_true"].tolist()
        y_pred = df["y_pred"].tolist()
        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(labels))),
            target_names=labels,
            digits=4,
            zero_division=0
        )
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        try:
            logger.debug(f"Logging prediction artifacts to MLflow run_id={run_id}: {csv_path}, {report_path}")
            client.log_artifact(run_id, csv_path, artifact_path="predictions")
            client.log_artifact(run_id, report_path, artifact_path="predictions")
        except Exception as e:
            logger.warning(f"Failed to log prediction artifacts: {e}")

        self._pred_samples[stage] = []

    @rank_zero_only
    def _log_full_prediction_artifacts(self, stage: str):
        if not getattr(self.hparams.exp, "export_predictions", False):
            return
        if self.hparams.exp.task != "classification":
            return
        if self.trainer is not None and getattr(self.trainer, "sanity_checking", False):
            return

        self._init_pred_full_buffers()
        chunks = self._pred_full.get(stage, [])
        if not chunks:
            return

        try:
            import pandas as pd
            from sklearn.metrics import classification_report
        except Exception as e:
            logger.warning(f"Full prediction artifact logging skipped: {e}")
            return

        # Concatenate chunks
        y_true = np.concatenate([c["y_true"] for c in chunks])
        y_pred = np.concatenate([c["y_pred"] for c in chunks])
        probs = np.concatenate([c["probs"] for c in chunks])

        data = {
            "y_true": y_true.astype(int),
            "y_pred": y_pred.astype(int),
        }

        # Add optional identifiers
        for key in ["time_idx", "original_index", "index", "subject_id", "record_id", "run_id"]:
            if all(key in c for c in chunks):
                data[key] = np.concatenate([c[key] for c in chunks])

        labels = self._get_label_names()
        data["y_true_label"] = [labels[i] if i < len(labels) else str(i) for i in data["y_true"]]
        data["y_pred_label"] = [labels[i] if i < len(labels) else str(i) for i in data["y_pred"]]

        # Probabilities/logits
        if bool(getattr(self.hparams.exp, "export_predictions_probs", True)):
            for k in range(len(labels)):
                data[f"p{k}"] = probs[:, k] if probs.shape[1] > k else 0.0
        if bool(getattr(self.hparams.exp, "export_predictions_logits", False)):
            # Reconstruct logits only if stored; otherwise skip
            if all("logits" in c for c in chunks):
                logits = np.concatenate([c["logits"] for c in chunks])
                for k in range(len(labels)):
                    data[f"logit{k}"] = logits[:, k] if logits.shape[1] > k else 0.0

        df = pd.DataFrame(data)

        # Output paths (Hydra run dir is cwd)
        out_dir = os.getcwd()
        csv_path = os.path.join(out_dir, f"predictions_{stage}.csv")
        report_csv = os.path.join(out_dir, f"classification_report_{stage}.csv")
        report_txt = os.path.join(out_dir, f"classification_report_{stage}.txt")
        df.to_csv(csv_path, index=False)

        report_dict = classification_report(
            data["y_true"],
            data["y_pred"],
            labels=list(range(len(labels))),
            target_names=labels,
            digits=4,
            zero_division=0,
            output_dict=True,
        )
        pd.DataFrame(report_dict).transpose().to_csv(report_csv)
        with open(report_txt, "w", encoding="utf-8") as f:
            f.write(classification_report(
                data["y_true"],
                data["y_pred"],
                labels=list(range(len(labels))),
                target_names=labels,
                digits=4,
                zero_division=0
            ))

        # Record-level aggregation + exports
        record_df = self._export_record_level_artifacts(stage, df, labels)
        # Subject-level aggregation + exports
        subject_df = self._export_subject_level_artifacts(stage, df, labels)

        # MLflow artifact logging
        client, run_id = self._get_mlflow_client_and_run_id()
        if client is not None and run_id is not None:
            try:
                client.log_artifact(run_id, csv_path, artifact_path="predictions")
                client.log_artifact(run_id, report_csv, artifact_path="predictions")
                client.log_artifact(run_id, report_txt, artifact_path="predictions")
            except Exception as e:
                logger.warning(f"Failed to log full prediction artifacts: {e}")

        # Post-run report (window/record/subject + calibration)
        try:
            self._generate_post_run_report(stage, df, record_df, subject_df, labels)
        except Exception as e:
            logger.warning(f"Post-run report generation failed: {e}")

        self._pred_full[stage] = []

    def _confusion_matrix_np(self, y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
        if y_true.size == 0 or y_pred.size == 0:
            return np.zeros((n_classes, n_classes), dtype=np.int64)
        idx = y_true.astype(int) * n_classes + y_pred.astype(int)
        cm = np.bincount(idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
        return cm

    def _normalize_confusion(self, cm: np.ndarray) -> np.ndarray:
        cm = cm.astype(np.float64)
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum == 0, 1.0, row_sum)
        return cm / row_sum

    def _metrics_from_cm(self, cm: np.ndarray) -> dict:
        total = float(cm.sum())
        if total <= 0:
            return {"accuracy": None, "balanced_acc": None, "macro_f1": None}
        tp = np.diag(cm).astype(float)
        fp = cm.sum(axis=0).astype(float) - tp
        fn = cm.sum(axis=1).astype(float) - tp
        eps = 1e-12
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        accuracy = float(tp.sum() / total)
        return {
            "accuracy": accuracy,
            "balanced_acc": float(np.mean(recall)),
            "macro_f1": float(np.mean(f1)),
        }

    def _save_confusion_plot(self, cm: np.ndarray, labels: list, path: str, title: str, normalize: bool = False):
        fig, ax = plt.subplots(figsize=(6, 5))
        try:
            data = self._normalize_confusion(cm) if normalize else cm.astype(float)
            im = ax.imshow(data, interpolation="nearest")
            ax.set_title(title)
            fig.colorbar(im, ax=ax)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if normalize:
                        txt = f"{data[i, j]:.2f}"
                    else:
                        txt = str(int(data[i, j]))
                    ax.text(j, i, txt, ha="center", va="center")
            fig.tight_layout()
            fig.savefig(path, dpi=200, bbox_inches="tight")
        finally:
            plt.close(fig)

    def _confidence_hist(self, probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> dict:
        if probs is None or probs.size == 0:
            return {"bins": [], "counts": [], "accuracy": []}
        conf = probs.max(axis=1)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        counts, _ = np.histogram(conf, bins=bins)
        acc = []
        y_pred = probs.argmax(axis=1)
        for i in range(n_bins):
            mask = (conf >= bins[i]) & (conf < bins[i + 1])
            if mask.any():
                acc.append(float((y_pred[mask] == y_true[mask]).mean()))
            else:
                acc.append(None)
        return {"bins": bins.tolist(), "counts": counts.tolist(), "accuracy": acc}

    def _brier_score(self, probs: np.ndarray, y_true: np.ndarray, n_classes: int) -> float:
        if probs is None or probs.size == 0:
            return None
        one_hot = np.zeros((y_true.size, n_classes), dtype=np.float64)
        one_hot[np.arange(y_true.size), y_true.astype(int)] = 1.0
        return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    def _level_report(self, name: str, df, labels: list) -> tuple:
        if df is None or df.empty:
            return None, []
        y_true = df["y_true"].to_numpy().astype(int)
        y_pred = df["y_pred"].to_numpy().astype(int)
        n_classes = len(labels)
        cm = self._confusion_matrix_np(y_true, y_pred, n_classes)
        cm_norm = self._normalize_confusion(cm)
        metrics = self._metrics_from_cm(cm)

        prob_cols = [f"p{k}" for k in range(n_classes) if f"p{k}" in df.columns]
        probs = df[prob_cols].to_numpy() if prob_cols else None

        dist_true = {labels[i]: int((y_true == i).sum()) for i in range(n_classes)}
        dist_pred = {labels[i]: int((y_pred == i).sum()) for i in range(n_classes)}

        level = {
            "n_samples": int(len(y_true)),
            "metrics": metrics,
            "confusion_matrix": {
                "raw": cm.tolist(),
                "normalized": cm_norm.tolist(),
            },
            "distribution": {
                "y_true": dist_true,
                "y_pred": dist_pred,
            },
        }
        artifacts = []
        if probs is not None and probs.size > 0:
            level["confidence"] = {
                "mean_max_prob": float(probs.max(axis=1).mean()),
                "brier": self._brier_score(probs, y_true, n_classes),
                "histogram": self._confidence_hist(probs, y_true, n_bins=10),
            }
        return level, artifacts

    def _collect_baseline_metrics(self, fold_index: int, dataset_name: str):
        try:
            from mlflow.tracking import MlflowClient
        except Exception:
            return []
        try:
            client = MlflowClient()
            exp = client.get_experiment_by_name(f"baselines/{dataset_name}")
            if exp is None:
                return []
            filter_str = ""
            if fold_index is not None:
                filter_str = f"params.fold_index = '{int(fold_index)}'"
            runs = client.search_runs([exp.experiment_id], filter_string=filter_str,
                                      order_by=["attributes.start_time DESC"])
        except Exception:
            return []
        results = {}
        for run in runs:
            params = run.data.params if run.data else {}
            backbone = params.get("backbone") or run.data.tags.get("mlflow.runName", "baseline")
            if backbone in results:
                continue
            metrics = run.data.metrics if run.data else {}
            results[backbone] = {
                "model": backbone,
                "record_macro_f1": metrics.get("record_macro_f1"),
                "record_balanced_acc": metrics.get("record_balanced_acc"),
                "record_level_agg": params.get("record_level_agg"),
                "run_id": run.info.run_id,
            }
        return list(results.values())

    def _generate_post_run_report(self, stage: str, window_df, record_df, subject_df, labels: list):
        """
        Generate a compact post-run quality report for EEG classification.

        Inputs:
        - `window_df` / `record_df` / `subject_df`: prediction DataFrames containing at least
          `y_true`, `y_pred`, and (optionally) per-class probabilities/logits columns.
        - `labels`: list of human-readable class names.

        Side effects:
        - Writes `quality_report_{stage}.json` and `quality_report_{stage}.md` into the run directory.
        - Optionally saves and logs normalized confusion matrices and confidence histograms to MLflow.
        """
        if self.hparams.exp.task != "classification":
            return
        out_dir = os.getcwd()

        report = {
            "stage": stage,
            "labels": labels,
            "aggregations": {
                "record": getattr(self.hparams.exp, "record_level_agg", "mean_prob"),
                "subject": getattr(self.hparams.exp, "subject_level_agg", "mean_prob"),
            },
            "levels": {},
        }
        if record_df is not None and "agg_used_record" in record_df.columns and not record_df.empty:
            report["aggregations"]["record"] = str(record_df["agg_used_record"].iloc[0])
        if subject_df is not None and "agg_used_subject" in subject_df.columns and not subject_df.empty:
            report["aggregations"]["subject"] = str(subject_df["agg_used_subject"].iloc[0])

        level_artifacts = []
        window_level, _ = self._level_report("window", window_df, labels)
        record_level, _ = self._level_report("record", record_df, labels)
        subject_level, _ = self._level_report("subject", subject_df, labels)
        if window_level is not None:
            report["levels"]["window"] = window_level
        if record_level is not None:
            report["levels"]["record"] = record_level
        if subject_level is not None:
            report["levels"]["subject"] = subject_level

        # Optional baseline comparison (test only)
        if stage == "test" and record_level is not None:
            dataset_name = str(getattr(self.hparams.dataset, "name", "dataset"))
            fold_index = getattr(self.hparams.dataset, "fold_index", None)
            baselines = self._collect_baseline_metrics(fold_index, dataset_name)
            report["baseline_comparison"] = baselines

        # Write JSON report
        json_path = os.path.join(out_dir, f"quality_report_{stage}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # Write Markdown report
        md_path = os.path.join(out_dir, f"quality_report_{stage}.md")
        lines = [
            f"# EEGMMIDB Quality Report ({stage})",
            "",
            f"- record_level_agg: {report['aggregations']['record']}",
            f"- subject_level_agg: {report['aggregations']['subject']}",
            "",
        ]
        for level_name, level_data in report["levels"].items():
            metrics = level_data.get("metrics", {})
            lines.append(f"## {level_name.capitalize()} level")
            lines.append(f"- samples: {level_data.get('n_samples')}")
            lines.append(f"- accuracy: {metrics.get('accuracy')}")
            lines.append(f"- balanced_acc: {metrics.get('balanced_acc')}")
            lines.append(f"- macro_f1: {metrics.get('macro_f1')}")
            conf = level_data.get("confidence")
            if conf:
                lines.append(f"- mean_max_prob: {conf.get('mean_max_prob')}")
                lines.append(f"- brier: {conf.get('brier')}")
            lines.append("")

        if "baseline_comparison" in report:
            lines.append("## Baseline comparison (record-level)")
            lines.append("| model | record_macro_f1 | record_balanced_acc |")
            lines.append("| --- | --- | --- |")
            # Current model
            if record_level is not None:
                lines.append("| CT (this run) | "
                             f"{record_level['metrics'].get('macro_f1')} | "
                             f"{record_level['metrics'].get('balanced_acc')} |")
            for row in report["baseline_comparison"]:
                lines.append(f"| {row.get('model')} | {row.get('record_macro_f1')} | {row.get('record_balanced_acc')} |")
            lines.append("")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        # Normalized confusion matrices + confidence histograms
        plot_paths = []
        for level_name, df in (("window", window_df), ("record", record_df), ("subject", subject_df)):
            if df is None or df.empty:
                continue
            y_true = df["y_true"].to_numpy().astype(int)
            y_pred = df["y_pred"].to_numpy().astype(int)
            cm = self._confusion_matrix_np(y_true, y_pred, len(labels))
            norm_path = os.path.join(out_dir, f"confusion_matrix_{stage}_{level_name}_norm.png")
            self._save_confusion_plot(cm, labels, norm_path,
                                      f"Confusion Matrix ({stage}, {level_name}, normalized)", normalize=True)
            plot_paths.append(norm_path)

            prob_cols = [f"p{k}" for k in range(len(labels)) if f"p{k}" in df.columns]
            if prob_cols:
                probs = df[prob_cols].to_numpy()
                conf = probs.max(axis=1)
                hist_path = os.path.join(out_dir, f"confidence_hist_{stage}_{level_name}.png")
                fig, ax = plt.subplots(figsize=(6, 4))
                try:
                    ax.hist(conf, bins=10, range=(0.0, 1.0), color="#4C72B0", alpha=0.85)
                    ax.set_title(f"Confidence Histogram ({stage}, {level_name})")
                    ax.set_xlabel("max(p)")
                    ax.set_ylabel("count")
                    fig.tight_layout()
                    fig.savefig(hist_path, dpi=200, bbox_inches="tight")
                finally:
                    plt.close(fig)
                plot_paths.append(hist_path)

        # Log artifacts to MLflow
        client, run_id = self._get_mlflow_client_and_run_id()
        if client is not None and run_id is not None:
            try:
                client.log_artifact(run_id, json_path, artifact_path="reports")
                client.log_artifact(run_id, md_path, artifact_path="reports")
                for p in plot_paths:
                    if os.path.exists(p):
                        client.log_artifact(run_id, p, artifact_path="reports")
            except Exception as e:
                logger.warning(f"Failed to log report artifacts: {e}")

    def _export_record_level_artifacts(self, stage: str, df, labels):
        """
        Aggregate window-level predictions into record-level predictions and artifacts.

        Expected `df` columns:
        - `record_id` (preferred) or a grouping key (`subject_id` + `run_id`)
        - `y_true` (int class id) and probability columns `p0..pK-1`
        - optional logit columns `logit0..logitK-1` (enables true `mean_logit`)

        Aggregation modes (via `exp.record_level_agg`):
        - `mean_prob`: mean of probabilities per record
        - `mean_logit`: mean of logits per record (falls back to a log-prob proxy if logits missing)
        - `trimmed_mean_prob`: trimmed mean of probabilities (robust to outliers)
        - `majority_vote`: majority vote of window-level argmax
        """
        if df is None or df.empty:
            return None
        if self.hparams.exp.task != "classification":
            return None

        prob_cols = [f"p{k}" for k in range(len(labels)) if f"p{k}" in df.columns]
        if not prob_cols:
            return None
        logit_cols = [f"logit{k}" for k in range(len(labels)) if f"logit{k}" in df.columns]

        if "record_id" in df.columns:
            group_cols = ["record_id"]
        elif "subject_id" in df.columns and "run_id" in df.columns:
            group_cols = ["subject_id", "run_id"]
        elif "original_index" in df.columns:
            group_cols = ["original_index"]
        else:
            return None

        trim = float(getattr(self.hparams.exp, "record_level_trim", 0.1))
        if trim < 0.0:
            trim = 0.0

        grouped = df.groupby(group_cols, sort=False, dropna=False)
        keys = []
        prob_means = []
        y_true_list = []
        y_pred_vote_list = []
        logit_means = []
        trimmed_means = []

        def _trimmed_mean_np(arr):
            n = arr.shape[0]
            k = int(n * trim)
            if k <= 0 or n <= 2 * k:
                return arr.mean(axis=0)
            arr_sorted = np.sort(arr, axis=0)
            return arr_sorted[k:n - k].mean(axis=0)

        for key, g in grouped:
            keys.append(key)
            probs = g[prob_cols].to_numpy()
            prob_means.append(probs.mean(axis=0))
            y_true_list.append(int(g["y_true"].value_counts().idxmax()))
            if "y_pred" in g.columns:
                y_pred_vote_list.append(int(g["y_pred"].value_counts().idxmax()))
            if logit_cols:
                logit_means.append(g[logit_cols].to_numpy().mean(axis=0))
            trimmed_means.append(_trimmed_mean_np(probs))

        if not prob_means:
            return None

        prob_means = np.stack(prob_means, axis=0)
        trimmed_means = np.stack(trimmed_means, axis=0)
        if logit_cols:
            logit_means = np.stack(logit_means, axis=0)

        import pandas as pd
        if len(group_cols) == 1:
            record_df = pd.DataFrame({group_cols[0]: keys})
        else:
            record_df = pd.DataFrame(list(keys), columns=group_cols)
        for i, col in enumerate(prob_cols):
            record_df[col] = prob_means[:, i]
        record_df["y_true"] = y_true_list
        record_df["y_pred_mean"] = prob_means.argmax(axis=1)
        if y_pred_vote_list:
            record_df["y_pred_vote"] = y_pred_vote_list

        # Optional aggregations for stability
        y_pred_mean_logit = None
        y_pred_trim = None
        used_logit_proxy = False

        def _softmax_np(x):
            x = x - x.max(axis=1, keepdims=True)
            exp = np.exp(x)
            return exp / exp.sum(axis=1, keepdims=True)

        if logit_cols:
            probs_logit = _softmax_np(logit_means)
            y_pred_mean_logit = probs_logit.argmax(axis=1)
        else:
            if not getattr(self, "_warned_missing_logits", False):
                logger.warning("record_level_agg=mean_logit requested but logits not exported; using mean log-prob proxy.")
                self._warned_missing_logits = True
            # Fall back to mean log-prob if logits are not exported.
            logit_mean = np.log(np.clip(prob_means, 1e-12, 1.0))
            probs_logit = _softmax_np(logit_mean)
            y_pred_mean_logit = probs_logit.argmax(axis=1)
            used_logit_proxy = True

        y_pred_trim = trimmed_means.argmax(axis=1)

        # Attach subject/run if available
        if "subject_id" in df.columns and "subject_id" not in group_cols:
            record_df["subject_id"] = grouped["subject_id"].first().values
        if "run_id" in df.columns and "run_id" not in group_cols:
            record_df["run_id"] = grouped["run_id"].first().values

        if "subject_id" in record_df.columns and "run_id" in record_df.columns:
            keys = []
            for s, r in zip(record_df["subject_id"].values, record_df["run_id"].values):
                try:
                    s_int = int(s)
                    r_int = int(r)
                    if s_int >= 0 and r_int >= 0:
                        keys.append(f"S{s_int:03d}R{r_int:02d}")
                    else:
                        keys.append(f"S{s}_R{r}")
                except Exception:
                    keys.append(f"S{s}_R{r}")
            record_df["record_key"] = keys
        elif "record_id" in record_df.columns:
            record_df["record_key"] = record_df["record_id"].astype(str)

        record_df["y_true_label"] = [labels[int(i)] if int(i) < len(labels) else str(i) for i in record_df["y_true"]]
        record_df["y_pred_label_mean"] = [
            labels[int(i)] if int(i) < len(labels) else str(i) for i in record_df["y_pred_mean"]
        ]
        if "y_pred_vote" in record_df.columns:
            record_df["y_pred_label_vote"] = [
                labels[int(i)] if int(i) < len(labels) else str(i) for i in record_df["y_pred_vote"]
            ]
        if y_pred_mean_logit is not None:
            record_df["y_pred_mean_logit"] = y_pred_mean_logit
            record_df["y_pred_label_mean_logit"] = [
                labels[int(i)] if int(i) < len(labels) else str(i) for i in y_pred_mean_logit
            ]
        if y_pred_trim is not None:
            record_df["y_pred_trimmed_mean"] = y_pred_trim
            record_df["y_pred_label_trimmed_mean"] = [
                labels[int(i)] if int(i) < len(labels) else str(i) for i in y_pred_trim
            ]

        agg = str(getattr(self.hparams.exp, "record_level_agg", "mean_prob")).lower()
        if agg in ("vote", "majority", "majority_vote") and "y_pred_vote" in record_df.columns:
            y_pred_rec = record_df["y_pred_vote"].values
        elif agg in ("mean_logit", "mean_logits"):
            y_pred_rec = y_pred_mean_logit
        elif agg in ("trimmed_mean_prob", "trimmed_mean", "trimmed"):
            y_pred_rec = y_pred_trim
        else:
            y_pred_rec = record_df["y_pred_mean"].values

        y_true_rec = record_df["y_true"].values

        agg_used = agg
        if agg in ("mean_logit", "mean_logits") and used_logit_proxy:
            agg_used = "mean_logit_proxy"
        record_df["y_pred"] = y_pred_rec
        record_df["y_pred_label"] = [
            labels[int(i)] if int(i) < len(labels) else str(i) for i in record_df["y_pred"]
        ]
        record_df["agg_used_record"] = agg_used

        try:
            from sklearn.metrics import classification_report, confusion_matrix
        except Exception as e:
            logger.warning(f"Record-level metrics skipped (sklearn missing): {e}")
            return record_df

        report_dict = classification_report(
            y_true_rec,
            y_pred_rec,
            labels=list(range(len(labels))),
            target_names=labels,
            digits=4,
            zero_division=0,
            output_dict=True,
        )

        cm = confusion_matrix(y_true_rec, y_pred_rec, labels=list(range(len(labels))))
        cm = cm.astype(float)
        total = cm.sum()
        if total > 0:
            tp = np.diag(cm)
            fp = cm.sum(axis=0) - tp
            fn = cm.sum(axis=1) - tp
            eps = 1e-12
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            macro_f1 = float(np.mean(f1))
            balanced_acc = float(np.mean(recall))
            self.log(f"{self.model_type}_{stage}_f1_macro_record", macro_f1, on_epoch=True, on_step=False, sync_dist=False)
            self.log(f"{self.model_type}_{stage}_balanced_acc_record", balanced_acc, on_epoch=True, on_step=False, sync_dist=False)
            if stage == "test":
                self.log("balanced_acc_record", balanced_acc, on_epoch=True, on_step=False, sync_dist=False)

        # Write artifacts to run dir
        out_dir = os.getcwd()
        record_csv = os.path.join(out_dir, f"predictions_{stage}_record.csv")
        report_csv = os.path.join(out_dir, f"classification_report_{stage}_record.csv")
        report_txt = os.path.join(out_dir, f"classification_report_{stage}_record.txt")
        cm_path = os.path.join(out_dir, f"confusion_matrix_{stage}_record.png")

        record_df.to_csv(record_csv, index=False)
        pd.DataFrame(report_dict).transpose().to_csv(report_csv)
        with open(report_txt, "w", encoding="utf-8") as f:
            f.write(classification_report(
                y_true_rec,
                y_pred_rec,
                labels=list(range(len(labels))),
                target_names=labels,
                digits=4,
                zero_division=0
            ))

        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        try:
            im = ax.imshow(cm, interpolation="nearest")
            ax.set_title(f"Confusion Matrix ({stage}, record)")
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
            fig.savefig(cm_path, dpi=200, bbox_inches="tight")
        finally:
            plt.close(fig)

        # MLflow artifact logging
        client, run_id = self._get_mlflow_client_and_run_id()
        if client is not None and run_id is not None:
            try:
                client.log_artifact(run_id, record_csv, artifact_path="predictions")
                client.log_artifact(run_id, report_csv, artifact_path="predictions")
                client.log_artifact(run_id, report_txt, artifact_path="predictions")
                client.log_artifact(run_id, cm_path, artifact_path="confusion_matrices")
            except Exception as e:
                logger.warning(f"Failed to log record-level artifacts: {e}")

        return record_df

    def _export_subject_level_artifacts(self, stage: str, df, labels):
        if df is None or df.empty:
            return None
        if self.hparams.exp.task != "classification":
            return None
        if "subject_id" not in df.columns:
            return None

        prob_cols = [f"p{k}" for k in range(len(labels)) if f"p{k}" in df.columns]
        if not prob_cols:
            return None

        try:
            import pandas as pd
            from sklearn.metrics import classification_report
        except Exception as e:
            logger.warning(f"Subject-level metrics skipped (sklearn/pandas missing): {e}")
            return None

        probs = df[prob_cols].to_numpy()
        y_true = df["y_true"].to_numpy()
        subject_ids = df["subject_id"].to_numpy()
        agg = str(getattr(self.hparams.exp, "subject_level_agg", "mean_prob")).lower()
        results = evaluate_subject_level(probs, y_true, subject_ids, agg=agg)

        if results["y_true"].size == 0:
            return None

        subject_df = pd.DataFrame({
            "subject_id": results["subject_ids"],
            "y_true": results["y_true"],
            "y_pred": results["y_pred"],
        })
        for i, col in enumerate(prob_cols):
            subject_df[col] = results["subject_probs"][:, i]
        subject_df["y_true_label"] = [labels[int(i)] if int(i) < len(labels) else str(i) for i in subject_df["y_true"]]
        subject_df["y_pred_label"] = [labels[int(i)] if int(i) < len(labels) else str(i) for i in subject_df["y_pred"]]
        subject_df["agg_used_subject"] = agg

        report_dict = classification_report(
            results["y_true"],
            results["y_pred"],
            labels=list(range(len(labels))),
            target_names=labels,
            digits=4,
            zero_division=0,
            output_dict=True,
        )

        out_dir = os.getcwd()
        subject_csv = os.path.join(out_dir, f"predictions_{stage}_subject.csv")
        report_csv = os.path.join(out_dir, f"classification_report_{stage}_subject.csv")
        report_txt = os.path.join(out_dir, f"classification_report_{stage}_subject.txt")
        cm_path = os.path.join(out_dir, f"confusion_matrix_{stage}_subject.png")

        subject_df.to_csv(subject_csv, index=False)
        pd.DataFrame(report_dict).transpose().to_csv(report_csv)
        with open(report_txt, "w", encoding="utf-8") as f:
            f.write(classification_report(
                results["y_true"],
                results["y_pred"],
                labels=list(range(len(labels))),
                target_names=labels,
                digits=4,
                zero_division=0
            ))

        cm = results.get("confusion_matrix")
        if cm is not None:
            fig, ax = plt.subplots(figsize=(6, 5))
            try:
                im = ax.imshow(cm, interpolation="nearest")
                ax.set_title(f"Confusion Matrix ({stage}, subject)")
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
                fig.savefig(cm_path, dpi=200, bbox_inches="tight")
            finally:
                plt.close(fig)

        if results.get("balanced_accuracy") is not None:
            self.log(f"{self.model_type}_{stage}_balanced_acc_subject",
                     float(results["balanced_accuracy"]), on_epoch=True, on_step=False, sync_dist=False)
        if results.get("macro_f1") is not None:
            self.log(f"{self.model_type}_{stage}_f1_macro_subject",
                     float(results["macro_f1"]), on_epoch=True, on_step=False, sync_dist=False)

        client, run_id = self._get_mlflow_client_and_run_id()
        if client is not None and run_id is not None:
            try:
                client.log_artifact(run_id, subject_csv, artifact_path="predictions")
                client.log_artifact(run_id, report_csv, artifact_path="predictions")
                client.log_artifact(run_id, report_txt, artifact_path="predictions")
                if os.path.exists(cm_path):
                    client.log_artifact(run_id, cm_path, artifact_path="confusion_matrices")
            except Exception as e:
                logger.warning(f"Failed to log subject-level artifacts: {e}")

        return subject_df

    def validation_step(self, batch, batch_ind):
        # Shared eval for classification/regression
        return self._shared_eval_step(batch, stage="val")

    def test_step(self, batch, batch_ind, **kwargs):
        return self._shared_eval_step(batch, stage="test")

    # Confusion matrix utilities
    def _cm_reset(self, stage: str):
        if self.hparams.exp.task != "classification":
            return
        c = int(self.dim_outcome)
        cm = torch.zeros((c, c), dtype=torch.long, device=self.device)
        if stage == "val":
            self._cm_val = cm
        elif stage == "test":
            self._cm_test = cm

    def _cm_record_reset(self, stage: str):
        if self.hparams.exp.task != "classification" or not self.multi_task:
            return
        c = int(self.dim_outcome)
        cm = torch.zeros((c, c), dtype=torch.long, device=self.device)
        if stage == "val":
            self._cm_record_val = cm
        elif stage == "test":
            self._cm_record_test = cm

    def on_train_epoch_start(self):
        self._reset_shock_errors("train")

    def on_validation_epoch_start(self):
        self._cm_reset("val")
        self._cm_record_reset("val")
        self._reset_shock_errors("val")
        self._reset_pred_samples("val")
        self._reset_pred_full("val")

    def on_test_epoch_start(self):
        self._cm_reset("test")
        self._cm_record_reset("test")
        self._reset_shock_errors("test")
        self._reset_pred_full("test")

    def on_train_epoch_end(self):
        self._warn_if_no_shock_valid("train")
        self._log_shock_t_hist("train")

    def _cm_update(self, stage: str, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
        """
        preds, targets: (B,T) long
        mask: (B,T) 0/1 or bool
        """
        if self.hparams.exp.task != "classification":
            return

        c = int(self.dim_outcome)

        preds_f = preds.reshape(-1)
        targets_f = targets.reshape(-1)
        mask_f = mask.reshape(-1).bool()

        preds_m = preds_f[mask_f]
        targets_m = targets_f[mask_f]

        # idx = true * C + pred
        idx = targets_m * c + preds_m
        cm_add = torch.bincount(idx, minlength=c * c).reshape(c, c)

        if stage == "val":
            self._cm_val += cm_add
        elif stage == "test":
            self._cm_test += cm_add

    def _cm_record_update(self, stage: str, preds: torch.Tensor, targets: torch.Tensor):
        if self.hparams.exp.task != "classification" or not self.multi_task:
            return
        c = int(self.dim_outcome)
        preds_f = preds.reshape(-1)
        targets_f = targets.reshape(-1)
        idx = targets_f * c + preds_f
        cm_add = torch.bincount(idx, minlength=c * c).reshape(c, c)
        if stage == "val":
            self._cm_record_val += cm_add
        elif stage == "test":
            self._cm_record_test += cm_add

    def _cm_finalize(self, cm: torch.Tensor) -> torch.Tensor:
        """
        Multi-GPU safe: gather then sum.
        """
        if hasattr(self, "all_gather") and self.trainer is not None and getattr(self.trainer, "world_size", 1) > 1:
            gathered = self.all_gather(cm)  # (world_size, C, C)
            return gathered.sum(dim=0)
        return cm

    @rank_zero_only
    def _cm_log_artifact(self, stage: str, cm_cpu: np.ndarray):
        """
        Log confusion matrix image into MLflow artifacts.
        """
        client, run_id = self._get_mlflow_client_and_run_id()
        if client is None or run_id is None:
            logger.warning("MLflow logger not found; confusion matrix artifact not logged.")
            return

        c = cm_cpu.shape[0]
        labels = self._get_label_names()

        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, f"confusion_matrix_{stage}_epoch_{self.current_epoch}.png")

        fig, ax = plt.subplots(figsize=(6, 5))
        try:
            im = ax.imshow(cm_cpu, interpolation="nearest")
            ax.set_title(f"Confusion Matrix ({stage})")
            fig.colorbar(im, ax=ax)
            ax.set_xticks(range(c))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticks(range(c))
            ax.set_yticklabels(labels)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

            for i in range(c):
                for j in range(c):
                    ax.text(j, i, str(cm_cpu[i, j]), ha="center", va="center")

            fig.tight_layout()
            fig.savefig(path, dpi=200, bbox_inches="tight")
        finally:
            plt.close(fig)

        # MLflow artifact
        try:
            logger.debug(f"Logging confusion matrix to MLflow run_id={run_id}: {path}")
            client.log_artifact(run_id, path, artifact_path="confusion_matrices")
        except Exception as e:
            logger.warning(f"Failed to log confusion matrix artifact: {e}")

    def _cm_log_metrics_and_artifact(self, stage: str):
        if self.hparams.exp.task != "classification":
            return

        if self.multi_task:
            cm = self._cm_record_val if stage == "val" else self._cm_record_test
        else:
            cm = self._cm_val if stage == "val" else self._cm_test
        cm = self._cm_finalize(cm)

        cm_cpu = cm.detach().cpu().numpy()
        total = cm_cpu.sum()
        if total <= 0:
            return

        # Global accuracy from confusion matrix
        acc = float(cm_cpu.diagonal().sum() / total)
        self.log(f"{self.model_type}_{stage}_acc_cm", acc, on_epoch=True, on_step=False, prog_bar=False, sync_dist=True)

        # Precision/Recall/F1 per class
        tp = cm_cpu.diagonal().astype(float)
        fp = cm_cpu.sum(axis=0).astype(float) - tp
        fn = cm_cpu.sum(axis=1).astype(float) - tp
        eps = 1e-12

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        macro_f1 = float(np.mean(f1))
        balanced_acc = float(np.mean(recall))
        self.log(f"{self.model_type}_{stage}_f1_macro", macro_f1, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f"{self.model_type}_{stage}_balanced_acc", balanced_acc, on_epoch=True, on_step=False, sync_dist=True)

        for k in range(len(tp)):
            self.log(f"{self.model_type}_{stage}_precision_c{k}", float(precision[k]), on_epoch=True, on_step=False, sync_dist=True)
            self.log(f"{self.model_type}_{stage}_recall_c{k}", float(recall[k]), on_epoch=True, on_step=False, sync_dist=True)
            self.log(f"{self.model_type}_{stage}_f1_c{k}", float(f1[k]), on_epoch=True, on_step=False, sync_dist=True)

        # Artifact image into MLflow
        self._cm_log_artifact(stage, cm_cpu)

    def _cm_record_log_metrics_and_artifact(self, stage: str):
        if self.hparams.exp.task != "classification" or not self.multi_task:
            return
        if self.trainer is not None and hasattr(self.trainer, "is_global_zero") and not self.trainer.is_global_zero:
            return

        cm = self._cm_record_val if stage == "val" else self._cm_record_test
        cm_cpu = cm.detach().cpu().numpy()
        total = cm_cpu.sum()
        if total <= 0:
            return

        acc = float(cm_cpu.diagonal().sum() / total)
        self.log(f"{self.model_type}_{stage}_acc_record_head", acc, on_epoch=True, on_step=False,
                 prog_bar=False, sync_dist=True)

        tp = cm_cpu.diagonal().astype(float)
        fp = cm_cpu.sum(axis=0).astype(float) - tp
        fn = cm_cpu.sum(axis=1).astype(float) - tp
        eps = 1e-12
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        macro_f1 = float(np.mean(f1))
        balanced_acc = float(np.mean(recall))

        self.log(f"{self.model_type}_{stage}_f1_macro_record_head", macro_f1, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f"{self.model_type}_{stage}_balanced_acc_record_head", balanced_acc, on_epoch=True, on_step=False, sync_dist=True)

        # Plot confusion matrix for record-level head
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, f"confusion_matrix_{stage}_record_head.png")
        fig, ax = plt.subplots(figsize=(6, 5))
        try:
            im = ax.imshow(cm_cpu, interpolation="nearest", cmap=plt.cm.Blues)
            fig.colorbar(im, ax=ax)
            ax.set_title(f"Confusion Matrix ({stage}, record head)")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_xticks(np.arange(cm_cpu.shape[1]))
            ax.set_yticks(np.arange(cm_cpu.shape[0]))
            for i in range(cm_cpu.shape[0]):
                for j in range(cm_cpu.shape[1]):
                    ax.text(j, i, int(cm_cpu[i, j]), ha="center", va="center", color="black")
            fig.tight_layout()
            fig.savefig(path)
        finally:
            plt.close(fig)

        client, run_id = self._get_mlflow_client_and_run_id()
        if client is not None and run_id is not None:
            try:
                client.log_artifact(run_id, path, artifact_path="confusion_matrices")
            except Exception as e:
                logger.warning(f"Failed to log record head confusion matrix: {e}")

    def _reset_shock_errors(self, stage: str):
        if not self.multi_task:
            return
        if stage not in self._shock_errors:
            self._shock_errors[stage] = []
        else:
            self._shock_errors[stage] = []
        if stage in self._shock_t_hist:
            self._shock_t_hist[stage] = []
        if stage in self._shock_valid_counts:
            self._shock_valid_counts[stage] = {"valid": 0, "total": 0}

    def _update_shock_stats(self, stage: str, valid_mask: torch.Tensor, t_shock: torch.Tensor):
        if not self.multi_task:
            return
        if stage not in self._shock_valid_counts:
            self._shock_valid_counts[stage] = {"valid": 0, "total": 0}
        total = int(valid_mask.numel())
        valid = int(valid_mask.sum().item())
        self._shock_valid_counts[stage]["total"] += total
        self._shock_valid_counts[stage]["valid"] += valid
        if valid > 0:
            vals = t_shock[valid_mask].detach().cpu().numpy()
            if stage not in self._shock_t_hist:
                self._shock_t_hist[stage] = []
            self._shock_t_hist[stage].append(vals)

    def _log_shock_error_hist(self, stage: str):
        if not self.multi_task:
            return
        if self.trainer is not None and hasattr(self.trainer, "is_global_zero") and not self.trainer.is_global_zero:
            return
        errors = self._shock_errors.get(stage, [])
        if not errors:
            return
        try:
            arr = np.concatenate(errors, axis=0)
        except Exception:
            return
        if arr.size == 0:
            return

        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, f"shock_error_hist_{stage}.png")
        fig, ax = plt.subplots(figsize=(6, 4))
        try:
            ax.hist(arr, bins=30, color="#4C72B0", alpha=0.8)
            ax.set_title(f"Shock localization error ({stage})")
            ax.set_xlabel("Absolute error (seconds)")
            ax.set_ylabel("Count")
            fig.tight_layout()
            fig.savefig(path)
        finally:
            plt.close(fig)

        client, run_id = self._get_mlflow_client_and_run_id()
        if client is not None and run_id is not None:
            try:
                client.log_artifact(run_id, path, artifact_path="shock")
            except Exception as e:
                logger.warning(f"Failed to log shock error histogram: {e}")

    def _log_shock_t_hist(self, stage: str):
        if not self.multi_task:
            return
        if self.trainer is not None and hasattr(self.trainer, "is_global_zero") and not self.trainer.is_global_zero:
            return
        values = self._shock_t_hist.get(stage, [])
        if not values:
            return
        try:
            arr = np.concatenate(values, axis=0)
        except Exception:
            return
        if arr.size == 0:
            return

        stride = float(getattr(self.hparams.dataset, "stride_seconds", 1.0))
        arr_sec = arr.astype(np.float32) * stride

        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, f"shock_t_hist_{stage}.png")
        fig, ax = plt.subplots(figsize=(6, 4))
        try:
            ax.hist(arr_sec, bins=30, color="#55A868", alpha=0.8)
            ax.set_title(f"Shock target index ({stage})")
            ax.set_xlabel("t_shock (seconds)")
            ax.set_ylabel("Count")
            fig.tight_layout()
            fig.savefig(path)
        finally:
            plt.close(fig)

        client, run_id = self._get_mlflow_client_and_run_id()
        if client is not None and run_id is not None:
            try:
                client.log_artifact(run_id, path, artifact_path="shock")
            except Exception as e:
                logger.warning(f"Failed to log shock t_shock histogram: {e}")

    def _warn_if_no_shock_valid(self, stage: str):
        if not self.multi_task:
            return
        counts = self._shock_valid_counts.get(stage, None)
        if counts is None:
            return
        total = counts.get("total", 0)
        valid = counts.get("valid", 0)
        if total > 0 and valid == 0:
            logger.warning(f"{stage} shock_valid all false for epoch (valid=0/total={total}).")

    def on_validation_epoch_end(self):
        self._cm_log_metrics_and_artifact("val")
        self._cm_record_log_metrics_and_artifact("val")
        self._log_shock_error_hist("val")
        self._log_shock_t_hist("val")
        self._warn_if_no_shock_valid("val")
        self._log_prediction_artifacts("val")
        self._log_full_prediction_artifacts("val")

    def on_test_epoch_end(self):
        self._cm_log_metrics_and_artifact("test")
        self._cm_record_log_metrics_and_artifact("test")
        self._log_shock_error_hist("test")
        self._log_shock_t_hist("test")
        self._warn_if_no_shock_valid("test")
        self._log_full_prediction_artifacts("test")

    def _shared_eval_step(self, batch, stage: str):
        if self._use_weights_ema():
            with self.ema_non_treatment.average_parameters():
                with self.ema_treatment.average_parameters():
                    treatment_pred, outcome_pred, br = self(batch)
        else:
            treatment_pred, outcome_pred, br = self(batch)

        if self.multi_task and self.hparams.exp.task == 'classification':
            (total_loss, outcome_loss, shock_loss_for_total, shock_loss_log, shock_valid_frac,
             outcome_logits, shock_logits, mask_bt, t_shock, valid) = self._compute_multi_task_losses(batch, br)

            # Record-level outcome metrics (head)
            y_outcome = batch["y_outcome"].long()
            preds_rec = torch.argmax(outcome_logits, dim=-1)
            self._cm_record_update(stage, preds_rec, y_outcome)

            self._update_shock_stats(stage, valid, t_shock)

            # Record-level accuracy (primary)
            acc_rec = (preds_rec == y_outcome).float().mean()

            # Window-level diagnostics
            targets_win = self._extract_class_targets(batch["outputs"])
            mask_win = batch["active_entries"].squeeze(-1)
            window_outcome_loss = self._classification_loss(outcome_pred, targets_win, mask_win)

            # Shock metrics
            if shock_logits.shape[1] > 0:
                pred_idx = torch.argmax(shock_logits, dim=-1)
                k = min(5, shock_logits.shape[1])
                topk_idx = torch.topk(shock_logits, k=k, dim=-1).indices
                top1 = (pred_idx == t_shock).float()
                top5 = (topk_idx == t_shock.unsqueeze(-1)).any(dim=-1).float()

                stride = float(getattr(self.hparams.dataset, "stride_seconds", 1.0))
                mae_sec = (pred_idx - t_shock).abs().float() * stride

                if valid.any():
                    v = valid.float()
                    denom = v.sum().clamp(min=1.0)
                    self.log(f'{self.model_type}_{stage}_shock_top1_acc', (top1 * v).sum() / denom,
                             on_epoch=True, on_step=False, sync_dist=True)
                    self.log(f'{self.model_type}_{stage}_shock_top5_acc', (top5 * v).sum() / denom,
                             on_epoch=True, on_step=False, sync_dist=True)
                    self.log(f'{self.model_type}_{stage}_shock_mae_sec', (mae_sec * v).sum() / denom,
                             on_epoch=True, on_step=False, sync_dist=True)

                    self._shock_errors[stage].append(mae_sec[valid].detach().cpu().numpy())

            self.log(f'{self.model_type}_{stage}_loss', total_loss, on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_{stage}_outcome_loss_record_head', outcome_loss,
                     on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_{stage}_outcome_loss', outcome_loss,
                     on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_{stage}_outcome_loss_window', window_outcome_loss,
                     on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_{stage}_shock_loss', shock_loss_log, on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_{stage}_shock_valid_frac', shock_valid_frac, on_epoch=True, on_step=False,
                     sync_dist=True)
            self.log(f'{self.model_type}_{stage}_acc', acc_rec, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)

            # Export predictions using record-head logits
            self._collect_pred_full_record_head(stage, outcome_logits, y_outcome, batch)

            return total_loss

        # BCE traitement / domain confusion (identique à train)
        balancing_mode = None if self.balancing is None else str(self.balancing).lower()
        bce_loss_raw = torch.zeros((), dtype=outcome_pred.dtype, device=outcome_pred.device)
        if balancing_mode in ('grad_reverse', 'domain_confusion'):
            if balancing_mode == 'grad_reverse':
                bce_loss_raw = self.bce_loss(treatment_pred, batch['current_treatments'].double(), kind='predict')
            elif balancing_mode == 'domain_confusion':
                bce_loss_raw = self.bce_loss(treatment_pred, batch['current_treatments'].double(), kind='confuse')
                bce_loss_raw = self.br_treatment_outcome_head.alpha * bce_loss_raw

            bce_loss_raw = (batch['active_entries'].squeeze(-1) * bce_loss_raw).sum() / batch['active_entries'].sum()

        bce_coef = 1.0
        if self.hparams.exp.task == 'classification':
            bce_coef = float(getattr(self.hparams.exp, "classification_bce_coef", 0.0))
        domain_confusion_coef = float(getattr(self.hparams.exp, "domain_confusion_coef", 1.0))
        bce_loss = bce_loss_raw * bce_coef * domain_confusion_coef

        if self.hparams.exp.task == 'classification':
            targets = self._extract_class_targets(batch['outputs'])  # (B,T)
            mask_bt = batch['active_entries'].squeeze(-1)            # (B,T)
            ce_loss = self._classification_loss(outcome_pred, targets, mask_bt)
            loss = bce_loss + ce_loss

            # Accuracy + confusion matrix
            preds = torch.argmax(outcome_pred, dim=-1)  # (B,T)
            correct = ((preds == targets) * mask_bt.bool()).sum().float()
            denom = mask_bt.sum().float().clamp(min=1.0)
            acc = (correct / denom)

            # Update confusion matrix
            self._cm_update(stage, preds, targets, mask_bt)

            self._collect_pred_samples(stage, outcome_pred, targets, mask_bt)
            self._collect_pred_full(stage, outcome_pred, targets, mask_bt, batch)

            self.log(f'{self.model_type}_{stage}_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_{stage}_bce_loss', bce_loss, on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_{stage}_bce_loss_raw', bce_loss_raw, on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_{stage}_ce_loss', ce_loss, on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_{stage}_acc', acc, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)

        else:
            mse_loss = F.mse_loss(outcome_pred, batch['outputs'], reduce=False)
            mse_loss = (batch['active_entries'] * mse_loss).sum() / batch['active_entries'].sum()
            loss = bce_loss + mse_loss

            self.log(f'{self.model_type}_{stage}_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_{stage}_bce_loss', bce_loss, on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_{stage}_bce_loss_raw', bce_loss_raw, on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_{stage}_mse_loss', mse_loss, on_epoch=True, on_step=False, sync_dist=True)

        return loss

    def predict_step(self, batch, batch_idx, dataset_idx=None):
        """
        Generates normalised output predictions
        """
        if self._use_weights_ema():
            with self.ema_non_treatment.average_parameters():
                _, outcome_pred, br = self(batch)
        else:
            _, outcome_pred, br = self(batch)
        return outcome_pred.cpu(), br.cpu()

    def get_representations(self, dataset: Dataset) -> np.array:
        logger.info(f'Balanced representations inference for {dataset.subset_name}.')
        # Creating Dataloader
        data_loader = DataLoader(dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False,
                                 **self._get_dataloader_kwargs())
        _, br = [torch.cat(arrs) for arrs in zip(*self.trainer.predict(self, data_loader))]
        return br.numpy()

    def get_predictions(self, dataset: Dataset) -> np.array:
        logger.info(f'Predictions for {dataset.subset_name}.')
        # Creating Dataloader
        data_loader = DataLoader(dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False,
                                 **self._get_dataloader_kwargs())
        outcome_pred, _ = [torch.cat(arrs) for arrs in zip(*self.trainer.predict(self, data_loader))]
        return outcome_pred.numpy()
