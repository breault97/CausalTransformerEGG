from typing import Optional, Dict, Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """
    Compact EEGNet-style CNN for EEG classification.
    Expects input shape (B, C, T).
    """

    def __init__(self, n_channels: int, n_samples: int, n_classes: int = 3,
                 F1: int = 8, D: int = 2, F2: Optional[int] = None,
                 kernel_length: int = 64, separable_length: int = 16,
                 dropout: float = 0.25):
        super().__init__()
        if F2 is None:
            F2 = F1 * D

        self.n_channels = int(n_channels)
        self.n_samples = int(n_samples)
        self.n_classes = int(n_classes)

        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (self.n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.sep1 = nn.Conv2d(F1 * D, F2, (1, separable_length),
                              padding=(0, separable_length // 2), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.dropout = nn.Dropout(dropout)

        # Compute feature dimension by forward with dummy
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.n_channels, self.n_samples)
            feat = self._features(dummy)
            self.classifier = nn.Linear(feat.shape[-1], self.n_classes)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.avg_pool2d(x, (1, 4))
        x = self.dropout(x)
        x = self.sep1(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = F.avg_pool2d(x, (1, 8))
        x = self.dropout(x)
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B,1,C,T)
        feats = self._features(x)
        return self.classifier(feats)


class ShallowConvNet(nn.Module):
    """
    ShallowConvNet (Schirrmeister et al.) for EEG.
    Expects input shape (B, C, T).
    """

    def __init__(self, n_channels: int, n_samples: int, n_classes: int = 3,
                 n_filters_time: int = 40, n_filters_spat: int = 40,
                 kernel_time: int = 25, pool_size: int = 75,
                 pool_stride: int = 15, dropout: float = 0.5):
        super().__init__()
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.n_channels = int(n_channels)
        self.n_samples = int(n_samples)
        self.n_classes = int(n_classes)

        self.conv_time = nn.Conv2d(1, n_filters_time, (1, kernel_time), bias=False)
        self.conv_spat = nn.Conv2d(n_filters_time, n_filters_spat, (self.n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters_spat)
        self.dropout = nn.Dropout(dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.n_channels, self.n_samples)
            feat = self._features(dummy)
            self.classifier = nn.Linear(feat.shape[-1], self.n_classes)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bn(x)
        x = torch.square(x)
        x = F.avg_pool2d(x, (1, self.pool_size), stride=(1, self.pool_stride))
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout(x)
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        feats = self._features(x)
        return self.classifier(feats)


class SimpleCNN1D(nn.Module):
    """
    Simple 1D CNN for feature sequences.
    Expects input shape (B, C, T).
    """

    def __init__(self, n_channels: int, n_samples: int, n_classes: int = 3,
                 channels: int = 32, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, channels, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(channels, channels * 2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(channels * 2, channels * 2, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return self.classifier(x)


class CSP_LDA:
    """
    CSP + LDA baseline (one-vs-rest for multi-class).
    Expects input shape (N, C, T) numpy arrays.
    """

    def __init__(self, n_components: int = 6, reg: Optional[float] = None):
        self.n_components = int(n_components)
        self.reg = reg
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        try:
            from mne.decoding import CSP
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            from sklearn.multiclass import OneVsRestClassifier
        except Exception as e:
            raise ImportError("CSP_LDA requires mne and scikit-learn.") from e

        # Use average_power to get 2D features for LDA. MNE requires log=None for 'csp_space'.
        csp = CSP(n_components=self.n_components, reg=self.reg, log=True, transform_into="average_power")
        lda = LinearDiscriminantAnalysis()
        self.model = OneVsRestClassifier(lda)
        # MNE CSP covariance needs float64; force X to float64 to avoid copy=None precision error.
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(int)
        # Fit CSP then LDA
        X_csp = csp.fit_transform(X, y)
        self.model.fit(X_csp, y)
        self.csp_ = csp
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("CSP_LDA.predict_proba called before fit().")
        X_csp = self.csp_.transform(X)
        return self.model.predict_proba(X_csp)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)
