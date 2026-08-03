from __future__ import annotations

import numpy as np

from breathmetrics.kernel_onset_detection_methods import (
    find_onsets_and_pauses_legacy,
)


def test_isolated_pause_histogram_mode_does_not_collapse_range(monkeypatch) -> None:
    """Regression test for translating MATLAB's ``1:maxpausebins`` loop."""
    respiration = np.zeros(45, dtype=float)
    respiration[1:10] = -0.5
    respiration[10:20] = np.array(
        [-0.8, -0.6, -0.4, -0.2, -0.16, 0.1, 0.3, 0.5, 0.7, 0.9]
    )
    respiration[20:30] = -0.5
    respiration[30:45] = 0.5
    peaks = np.array([10, 30])
    troughs = np.array([20, 40])

    centers = np.linspace(-1.0, 1.0, 100)
    ordinary_counts = np.zeros(100, dtype=int)
    ordinary_counts[0] = 10
    isolated_mode_counts = np.zeros(100, dtype=int)
    isolated_mode_counts[39:44] = [9, 14, 123, 22, 12]
    call_count = 0

    def histogram_stub(window, custom_bins):
        nonlocal call_count
        call_count += 1
        counts = isolated_mode_counts if call_count == 3 else ordinary_counts
        return counts.copy(), centers.copy()

    monkeypatch.setattr(
        "breathmetrics.utils.hist_like_matlab",
        histogram_stub,
    )

    _, exhale_onsets, inhale_pause_onsets, _ = find_onsets_and_pauses_legacy(
        respiration,
        peaks,
        troughs,
    )

    assert exhale_onsets[0] >= peaks[0]
    assert inhale_pause_onsets[0] >= peaks[0]
