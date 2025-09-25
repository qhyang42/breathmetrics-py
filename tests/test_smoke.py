# tests/test_smoke.py
import numpy as np
from breathmetrics.core import BreathMetrics, BreathMetricsConfig


def test_quick_pipeline():
    fs = 100.0
    t = np.arange(0, 5, 1 / fs)
    # simple fake breathing signal: sine wave
    x = np.sin(2 * np.pi * 0.25 * t)

    bm = BreathMetrics(x, BreathMetricsConfig(fs=fs)).preprocess().estimate_features()

    # Make sure some features were estimated
    assert "mean" in bm.features
    assert "std" in bm.features
