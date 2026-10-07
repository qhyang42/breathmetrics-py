# tests for the ZLP segmentation kernel + Breathe integration, and for the
# legacy onset/pause robustness guards.
from __future__ import annotations

import numpy as np
import pytest

from breathmetrics import Breathe
from breathmetrics.kernel_onset_detection_methods import find_onsets_and_pauses_legacy
from breathmetrics.kernel_zlp import (
    find_inhale_onsets,
    pair_breath_landmarks,
    prep_breath_trace,
)
from breathmetrics.utils import MISSING_EVENT

FS = 100.0


def make_nonstationary_resp(seed: int = 7) -> np.ndarray:
    """Amplitude-non-stationary synthetic breathing: three segments with
    several-fold depth and rate differences, slow baseline drift, and mild
    noise - the regime the ZLP engine was built for."""
    rng = np.random.default_rng(seed)
    segs = []
    for dur_s, rate_hz, amp in [(100, 0.25, 1.0), (100, 0.10, 5.0), (100, 0.30, 0.5)]:
        t = np.arange(int(dur_s * FS)) / FS
        # mildly asymmetric breaths: faster rise than fall
        phase = 2 * np.pi * rate_hz * t
        seg = amp * (np.sin(phase) + 0.3 * np.sin(2 * phase))
        segs.append(seg)
    resp = np.concatenate(segs)
    t_all = np.arange(resp.size) / FS
    drift = 2.0 * np.sin(2 * np.pi * t_all / 180.0) + 0.01 * t_all
    noise = 0.02 * rng.standard_normal(resp.size)
    return resp + drift + noise


EXPECTED_BREATHS = 25 + 10 + 30  # segment rate * duration


def test_prep_and_onsets_counts_and_ordering():
    resp = make_nonstationary_resp()
    det, pk0, tr0, info = prep_breath_trace(resp, FS)
    assert info["mode"] == "conservative"
    onsets, peaks, troughs = find_inhale_onsets(det, FS, pk0, tr0)

    # breath count within 15% of ground truth despite the depth changes
    assert abs(onsets.size - EXPECTED_BREATHS) <= 0.15 * EXPECTED_BREATHS

    # strict alternation of the pruned extrema
    ev = np.sort(np.concatenate([peaks, troughs]))
    kinds = np.isin(ev, peaks).astype(int)
    assert np.all(np.abs(np.diff(kinds)) == 1), "extrema must strictly alternate"

    # each onset sits inside its trough->peak pair, at or past the 20% floor
    for on in onsets:
        pk = peaks[peaks > on][0]
        tr = troughs[troughs < pk][-1]
        n_w = pk - tr + 1
        assert tr <= on < pk
        assert on - tr >= int(np.ceil(0.20 * (n_w - 1)))


def test_pair_breath_landmarks_drops_unpaired_tail():
    onsets = np.array([10, 110, 210])
    peaks = np.array([50, 150, 250])
    troughs = np.array([5, 90, 190])  # no trough after the last peak
    on, pk, tr = pair_breath_landmarks(onsets, peaks, troughs)
    assert on.size == pk.size == tr.size == 2
    assert np.all(pk > on) and np.all(tr > pk)


def test_estimate_all_features_zlp_end_to_end():
    resp = make_nonstationary_resp()
    bm = Breathe(resp, FS, "humanAirflow")
    bm.estimate_all_features_zlp()

    n = bm.inhale_onsets.size
    assert n == bm.inhale_peaks.size == bm.exhale_troughs.size
    assert abs(n - EXPECTED_BREATHS) <= 0.15 * EXPECTED_BREATHS
    assert np.all(bm.inhale_onsets < bm.inhale_peaks)
    assert bm.feature_estimations_complete

    # volumes are computed on the object's native-unit trace and should be
    # finite for the overwhelming majority of breaths
    finite = np.isfinite(np.asarray(bm.inhale_volumes, dtype=float))
    assert finite.mean() > 0.9

    # inhale pauses that survive must respect the injected onsets
    in_pause = np.asarray(bm.inhale_pause_onsets)
    for bi in range(n):
        if in_pause[bi] == MISSING_EVENT:
            continue
        assert bm.inhale_peaks[bi] < in_pause[bi] < bm.exhale_onsets[bi]

    # the QC stash exists and matches the signal length
    assert bm.zlp_detection["det"].size == resp.size


def test_estimate_all_features_zlp_rejects_non_airflow():
    resp = make_nonstationary_resp()
    bm = Breathe(resp, FS, "humanBB")
    with pytest.raises(ValueError, match="airflow"):
        bm.estimate_all_features_zlp()


def test_legacy_onsets_no_crash_on_offset_window():
    """Regression test for the empty-crossing guards: a trough->peak window
    sitting entirely above the global mean used to raise IndexError in the
    no-pause branch (and a float head/tail bound raised TypeError)."""
    fs = int(FS)
    t_high = np.arange(40 * fs) / fs
    low = np.zeros(20 * fs)
    high = 10.0 + np.sin(2 * np.pi * 0.25 * t_high)  # oscillation on a high plateau
    resp = np.concatenate([low, high])
    off = low.size
    period = 4 * fs
    # analytic extrema of the plateau oscillation (sin peaks at quarter period)
    peaks = off + period // 4 + period * np.arange(8)
    troughs = off + 3 * period // 4 + period * np.arange(7)

    inh, exh, _inh_pause, _exh_pause = find_onsets_and_pauses_legacy(
        resp, peaks, troughs, nbins=100
    )

    assert inh.size == peaks.size and exh.size == troughs.size
    # every interior inhale onset must fall in its trough->peak window
    for k in range(len(peaks) - 1):
        assert troughs[k] <= inh[k + 1] <= peaks[k + 1]
    # interior exhale onsets fall in their peak->trough windows (the final
    # one is the legacy tail special case, estimated after the LAST peak)
    for k in range(len(troughs) - 1):
        assert peaks[k] <= exh[k] <= troughs[k]
    assert exh[-1] >= peaks[-1]
