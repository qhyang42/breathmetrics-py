from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class BreathMetricsConfig:
    fs: float  # sampling rate (Hz)
    detrend: bool = True
    smooth_win_sec: float = 0.2


class BreathMetrics:
    """Core breathing-signal parameterization (skeleton)."""

    def __init__(self, signal: np.ndarray, config: BreathMetricsConfig):
        self.signal = np.asarray(signal, dtype=float)
        self.config = config
        self.features = {}

    def preprocess(self) -> "BreathMetrics":
        x = self.signal
        if self.config.detrend:
            x = x - np.nanmean(x)
        win = max(1, int(self.config.smooth_win_sec * self.config.fs))
        if win > 1:
            # simple moving average as placeholder
            kernel = np.ones(win) / win
            x = np.convolve(x, kernel, mode="same")
        self._x = x
        return self

    def estimate_features(self) -> "BreathMetrics":
        # TODO: replace with actual inhalation/exhalation detection, etc.
        self.features["mean"] = float(np.nanmean(getattr(self, "_x", self.signal)))
        self.features["std"] = float(np.nanstd(getattr(self, "_x", self.signal)))
        return self

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame([self.features])
