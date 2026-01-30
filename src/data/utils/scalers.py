from typing import Iterable, List, Optional, Union, Dict, Any

import numpy as np


class PerFoldStandardScaler:
    """
    Standard scaler that fits ONLY on the current fold's training data.

    This avoids leakage by never using validation/test samples when computing
    mean/std. It is intentionally lightweight and serializable for MLflow.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, arrays: Union[np.ndarray, List[np.ndarray]]) -> "PerFoldStandardScaler":
        if arrays is None:
            raise ValueError("PerFoldStandardScaler.fit received None.")
        if isinstance(arrays, list):
            if not arrays:
                raise ValueError("PerFoldStandardScaler.fit received empty list.")
            flat = np.concatenate(arrays, axis=0)
        else:
            flat = np.asarray(arrays)
        if flat.size == 0:
            raise ValueError("PerFoldStandardScaler.fit received empty data.")
        self.mean_ = flat.mean(axis=0).astype(np.float32, copy=False)
        self.std_ = (flat.std(axis=0) + self.eps).astype(np.float32, copy=False)
        return self

    def transform(self, arrays: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("PerFoldStandardScaler.transform called before fit().")
        if isinstance(arrays, list):
            return [((a - self.mean_) / self.std_).astype(np.float32, copy=False) for a in arrays]
        arr = np.asarray(arrays)
        return ((arr - self.mean_) / self.std_).astype(np.float32, copy=False)

    def fit_transform(self, arrays: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
        return self.fit(arrays).transform(arrays)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean_,
            "std": self.std_,
            "eps": self.eps,
        }

    def to_loggable(self) -> Dict[str, Any]:
        return {
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "std": self.std_.tolist() if self.std_ is not None else None,
            "eps": float(self.eps),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "PerFoldStandardScaler":
        scaler = cls(eps=float(state.get("eps", 1e-6)))
        mean = state.get("mean")
        std = state.get("std")
        if mean is not None:
            scaler.mean_ = np.asarray(mean, dtype=np.float32)
        if std is not None:
            scaler.std_ = np.asarray(std, dtype=np.float32)
        return scaler
