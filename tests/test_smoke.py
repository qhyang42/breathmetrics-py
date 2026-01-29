# tests/test_smoke.py
import numpy as np

from breathmetrics.core import Breathe


def test_basic_init():
    fs = 100.0
    t = np.arange(0, 5, 1 / fs)
    x = np.sin(2 * np.pi * 1.0 * t)

    bm = Breathe(x, fs, "humanAirflow")
    assert bm.raw_respiration is not None
    assert bm.smoothed_respiration is not None
    assert bm.supports("inhale_onsets")
