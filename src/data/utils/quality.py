from typing import Dict

import numpy as np


def _bandpower(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return np.zeros(psd.shape[0], dtype=psd.dtype)
    return np.sum(psd[:, mask], axis=1)


def compute_quality_metrics(window: np.ndarray, sfreq: float,
                            line_freq: float = 60.0, line_width: float = 5.0,
                            saturation_sigma: float = 5.0,
                            flatline_std: float = 1e-6) -> Dict[str, float]:
    """
    Compute objective quality metrics for a single window.

    window: (C, S) array
    Returns a dict of scalar metrics.
    """
    x = np.asarray(window)
    rms = np.sqrt(np.mean(x ** 2, axis=1))
    ptp = np.ptp(x, axis=1)
    std = np.std(x, axis=1)

    # PSD-based metrics
    n = max(1, x.shape[1])
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sfreq))
    psd = (np.abs(np.fft.rfft(x, axis=1)) ** 2) / float(n)

    line_power = _bandpower(psd, freqs, line_freq - line_width, line_freq + line_width)
    signal_power = _bandpower(psd, freqs, 0.5, 45.0)
    snr = (signal_power + 1e-12) / (line_power + 1e-12)
    snr_db = 10.0 * np.log10(snr)
    line_ratio = line_power / (signal_power + 1e-12)

    # Saturation: fraction of channels with many extreme values
    med = np.median(x, axis=1)
    mad = np.median(np.abs(x - med[:, None]), axis=1) + 1e-8
    robust_std = 1.4826 * mad
    thr = saturation_sigma * robust_std
    extreme = np.abs(x - med[:, None]) > thr[:, None]
    sat_channels = np.mean(np.mean(extreme, axis=1) > 0.05)

    flat_channels = np.mean(std < flatline_std)

    return {
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "ptp_mean": float(np.mean(ptp)),
        "ptp_std": float(np.std(ptp)),
        "snr_db_mean": float(np.mean(snr_db)),
        "line_ratio": float(np.mean(line_ratio)),
        "saturation_frac": float(sat_channels),
        "flatline_frac": float(flat_channels),
    }


def compute_quality_score(window: np.ndarray, sfreq: float, feature: str = "composite") -> float:
    """
    Compute a scalar quality score for a single window.

    feature:
        - "rms": use rms_mean
        - "snr": use snr_db_mean
        - "line_ratio": negative line_ratio
        - "saturation": negative saturation_frac
        - "composite": weighted combination
    """
    metrics = compute_quality_metrics(window, sfreq)
    feat = str(feature).lower()
    if feat in ("rms", "rms_mean"):
        return float(metrics["rms_mean"])
    if feat in ("snr", "snr_db", "snr_db_mean"):
        return float(metrics["snr_db_mean"])
    if feat in ("line_ratio", "line", "line_noise"):
        return float(-metrics["line_ratio"])
    if feat in ("saturation", "sat", "saturation_frac"):
        return float(-metrics["saturation_frac"])
    # composite (higher is better)
    score = metrics["snr_db_mean"] - metrics["line_ratio"] - metrics["saturation_frac"] - metrics["flatline_frac"]
    return float(score)
