import numpy as np
import pytest
from omegaconf import OmegaConf

from src.data.physionet_eegmmidb.dataset import PhysioNetEEGMMIDBDatasetCollection
from runnables.train_multi import _validate_treatment_config


def _dummy_collection(treatment_dim: int = 0):
    ds = PhysioNetEEGMMIDBDatasetCollection.__new__(PhysioNetEEGMMIDBDatasetCollection)
    ds.treatment_dim = int(treatment_dim)
    ds.label_names = ["mauvais", "moyen", "excellent"]
    return ds


def test_no_treatments_build_split():
    ds = _dummy_collection(0)
    covs_list = [np.zeros((2, 3), dtype=np.float32)]
    treats_list = [np.zeros((2, 0), dtype=np.float32)]
    outs_list = [np.zeros((2, 1), dtype=np.int64)]

    data = ds._build_split(covs_list, treats_list, outs_list, max_len=2)
    assert data["current_treatments"].shape[-1] == 0
    assert data["prev_treatments"].shape[-1] == 0


def test_validate_treatment_config_none_dim_mismatch():
    cfg = OmegaConf.create({
        "dataset": {"treatment_mode": "none", "treatment_dim": 1},
        "model": {"dim_treatments": 1},
        "exp": {},
    })
    with pytest.raises(ValueError):
        _validate_treatment_config(cfg)


def test_validate_treatment_config_dim_mismatch():
    cfg = OmegaConf.create({
        "dataset": {"treatment_mode": "multiclass", "treatment_dim": 2},
        "model": {"dim_treatments": 1},
        "exp": {},
    })
    with pytest.raises(ValueError):
        _validate_treatment_config(cfg)


def test_validate_treatment_config_ok():
    cfg = OmegaConf.create({
        "dataset": {"treatment_mode": "none", "treatment_dim": 0},
        "model": {"dim_treatments": 0},
        "exp": {},
    })
    _validate_treatment_config(cfg)
