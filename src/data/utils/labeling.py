from typing import Iterable, Optional, Dict, Any, Tuple

import numpy as np


class QuantileLabeler:
    """
    Quantile-based labeler for 3-class quality labels.

    Fit on training scores only, then transform any split.
    Labels: 0 = low quality, 1 = medium quality, 2 = high quality.
    """

    def __init__(self, quantiles=(0.33, 0.66)):
        if quantiles is None or len(quantiles) != 2:
            raise ValueError("QuantileLabeler requires exactly two quantiles.")
        q0, q1 = float(quantiles[0]), float(quantiles[1])
        q0 = min(max(q0, 0.0), 1.0)
        q1 = min(max(q1, 0.0), 1.0)
        if q0 > q1:
            q0, q1 = q1, q0
        self.quantiles = (q0, q1)
        self.thresholds_: Optional[Tuple[float, float]] = None

    def fit(self, scores: np.ndarray) -> "QuantileLabeler":
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if scores.size == 0:
            raise ValueError("QuantileLabeler.fit received empty scores.")
        q_vals = np.quantile(scores, list(self.quantiles))
        t_low, t_high = float(q_vals[0]), float(q_vals[1])
        if t_low > t_high:
            t_low, t_high = t_high, t_low
        self.thresholds_ = (t_low, t_high)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if self.thresholds_ is None:
            raise RuntimeError("QuantileLabeler.transform called before fit().")
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        t_low, t_high = self.thresholds_
        labels = np.ones(scores.shape[0], dtype=np.int64)
        labels[scores <= t_low] = 0
        labels[scores >= t_high] = 2
        return labels

    def fit_transform(self, scores: np.ndarray) -> np.ndarray:
        return self.fit(scores).transform(scores)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "quantiles": list(self.quantiles),
            "thresholds": list(self.thresholds_) if self.thresholds_ is not None else None,
        }

    def to_loggable(self) -> Dict[str, Any]:
        return self.state_dict()
