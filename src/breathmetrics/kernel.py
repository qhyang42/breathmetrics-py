from __future__ import annotations
from typing import Any
import numpy as np
from numpy.typing import ArrayLike

## core function for breathmetrics. simple functions. no classes.

## baseline correction
# kernel.py  (pure function)
from .utils import detrend_linear, fft_smooth, zscore


def correct_respiration_to_baseline(
    resp: np.ndarray,
    fs: float,
    method: str = "simple",  # "simple" | "sliding"
    zscore_out: bool = False,
    sliding_window_s: float = 60.0,
) -> np.ndarray:
    """
    Detrend → baseline correct (mean or sliding mean) → optional z-score.
    Returns a *new* array; does not mutate inputs.
    """
    x = detrend_linear(np.asarray(resp, float))

    if method == "simple":
        x_corr = x - x.mean()
    elif method == "sliding":
        win = max(1, int(np.floor(fs * sliding_window_s)))
        x_corr = x - fft_smooth(x, win)
    else:
        raise ValueError("method must be 'simple' or 'sliding'")

    if zscore_out:
        x_corr = zscore(x_corr)

    return x_corr


## find extrema
# kernel.py


def _find_respiratory_extrema_basic(
    signal: np.ndarray,
    distance: int | None = None,
    prominence: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find local maxima ('peaks') and minima ('troughs').
    Uses scipy.signal.find_peaks if available; otherwise a simple NumPy fallback.
    Returns (peaks_idx, troughs_idx) as integer arrays.
    """
    x = np.asarray(signal, dtype=float)

    try:
        from scipy.signal import find_peaks  # optional

        peaks, _ = find_peaks(x, distance=distance, prominence=prominence)
        troughs, _ = find_peaks(-x, distance=distance, prominence=prominence)
        return peaks.astype(int), troughs.astype(int)
    except Exception:
        # Fallback: strict neighbor test
        # A point i is a local max if x[i-1] < x[i] > x[i+1]; min if x[i-1] > x[i] < x[i+1]
        if x.size < 3:
            return np.array([], dtype=int), np.array([], dtype=int)
        dx1 = x[1:-1] - x[:-2]
        dx2 = x[1:-1] - x[2:]
        peaks = np.where((dx1 > 0) & (dx2 > 0))[0] + 1
        troughs = np.where((dx1 < 0) & (dx2 < 0))[0] + 1
        if distance and distance > 1:
            # crude distance enforcement: keep every 'distance'-th candidate
            peaks = peaks[:: max(1, int(distance))]
            troughs = troughs[:: max(1, int(distance))]
        return peaks.astype(int), troughs.astype(int)


## TODO debug this. needs tests.
def find_respiratory_extrema(
    resp: ArrayLike,
    srate: float,
    custom_decision_threshold: int = 0,
    sw_sizes: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find peaks and troughs in respiratory data using a sliding-window voting scheme.

    Parameters
    ----------
    resp : array-like
        1D respiratory trace.
    srate : float
        Sampling rate (Hz).
    custom_decision_threshold : int, default 0
        If > 0, overrides the auto-selected decision threshold.
    sw_sizes : list[int] | None
        Sliding-window sizes (in samples). If None, defaults to:
        floor([100, 300, 700, 1000, 5000] * (srate/1000)).

    Returns
    -------
    corrected_peaks : np.ndarray
        Indices of peak samples (after alternation cleanup, padding removed).
    corrected_troughs : np.ndarray
        Indices of trough samples (after alternation cleanup, padding removed).
    """
    x = np.asarray(resp, dtype=float).ravel()
    n = x.size
    if n < 3:
        return np.array([], dtype=int), np.array([], dtype=int)

    # Default window sizes (ms → samples), as in the MATLAB code
    if sw_sizes is None:
        srate_adjust = float(srate) / 1000.0
        sw_sizes = [
            int(np.floor(100 * srate_adjust)),
            int(np.floor(300 * srate_adjust)),
            int(np.floor(700 * srate_adjust)),
            int(np.floor(1000 * srate_adjust)),
            int(np.floor(5000 * srate_adjust)),
        ]
    sw_sizes = [max(1, int(w)) for w in sw_sizes]

    # Pad by reflecting the tail so large windows don't miss the end
    pad_ind = int(min(n - 1, max(sw_sizes) * 2))
    tail = x[-(pad_ind + 1) :][::-1]
    padded = np.concatenate([x, tail])
    Np = padded.size

    # Voting vectors over padded signal
    sw_peak_vect = np.zeros(Np, dtype=int)
    sw_trough_vect = np.zeros(Np, dtype=int)

    # Basic amplitude thresholds (same spirit as MATLAB)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    peak_threshold = mu + sd / 2.0
    trough_threshold = mu - sd / 2.0

    # Shifts to reduce boundary bias (MATLAB used SHIFTS = 1:3)
    SHIFTS = (1, 2, 3)
    n_windows = len(sw_sizes) * len(SHIFTS)

    # For each window size and phase, count window argmax/argmin "votes"
    for sw in sw_sizes:
        if sw < 1:
            continue
        for shift in SHIFTS:
            # phase offset ≈ {0, 1/3, 2/3} of window
            offset = int(np.floor((shift - 1) * sw / 3.0))
            start = offset
            while start + sw <= Np:
                seg = padded[start : start + sw]
                i_max = int(np.argmax(seg))
                i_min = int(np.argmin(seg))
                peak_idx = start + i_max
                trough_idx = start + i_min
                # Only count sufficiently large peaks / deep troughs
                if padded[peak_idx] >= peak_threshold:
                    sw_peak_vect[peak_idx] += 1
                if padded[trough_idx] <= trough_threshold:
                    sw_trough_vect[trough_idx] += 1
                start += sw

    # Auto-select decision threshold (knee of cumulative counts)
    n_peaks_found = np.array(
        [np.sum(sw_peak_vect > t) for t in range(1, n_windows + 1)]
    )
    n_troughs_found = np.array(
        [np.sum(sw_trough_vect > t) for t in range(1, n_windows + 1)]
    )

    if n_peaks_found.size > 1:
        best_peak_diff = int(np.argmax(np.diff(n_peaks_found))) + 1
    else:
        best_peak_diff = 1
    if n_troughs_found.size > 1:
        best_trough_diff = int(np.argmax(np.diff(n_troughs_found))) + 1
    else:
        best_trough_diff = 1

    if custom_decision_threshold and custom_decision_threshold > 0:
        decision_thr = int(custom_decision_threshold)
    else:
        decision_thr = int(np.floor((best_peak_diff + best_trough_diff) / 2.0))
        decision_thr = max(decision_thr, 1)

    # Provisional extrema = samples with enough votes
    peak_inds = np.where(sw_peak_vect >= decision_thr)[0]
    trough_inds = np.where(sw_trough_vect >= decision_thr)[0]

    # Enforce alternating peak–trough sequence by pruning duplicates
    # corrected_peaks: list[int] = []
    corrected_peaks: Any = []
    # corrected_troughs: list[int] = []
    corrected_troughs: Any = []
    pki, tri = 0, 0
    proceed_check = 1

    # Require lookahead of one on each list (mirrors MATLAB loop bounds)
    while pki < len(peak_inds) - 1 and tri < len(trough_inds) - 1:
        peak_trough_diff = trough_inds[tri] - peak_inds[pki]

        # If the next selected extrema is another peak before a trough,
        # keep the taller peak and drop the other
        if proceed_check == 1:
            peak_peak_diff = peak_inds[pki + 1] - peak_inds[pki]
            peak_trough_diff2 = trough_inds[tri] - peak_inds[pki]
            if peak_peak_diff < peak_trough_diff2:
                v0 = padded[peak_inds[pki]]
                v1 = padded[peak_inds[pki + 1]]
                if v1 > v0:
                    pki += 1  # take second (taller) peak
                else:
                    peak_inds = np.delete(peak_inds, pki + 1)  # drop second
                proceed_check = 0

        # If the next selected extrema is another trough before a peak,
        # keep the deeper trough and drop the other
        if proceed_check == 1:
            trough_trough_diff = trough_inds[tri + 1] - trough_inds[tri]
            trough_peak_diff2 = peak_inds[pki + 1] - trough_inds[tri]
            if trough_trough_diff < trough_peak_diff2:
                v0 = padded[trough_inds[tri]]
                v1 = padded[trough_inds[tri + 1]]
                if v1 < v0:
                    tri += 1  # take second (deeper) trough
                else:
                    trough_inds = np.delete(trough_inds, tri + 1)  # drop second
                proceed_check = 0

        # If both checks pass, we can pair this peak with this trough
        if proceed_check == 1:
            if peak_trough_diff > 0:  # peak occurs before its trough
                corrected_peaks.append(int(peak_inds[pki]))
                corrected_troughs.append(int(trough_inds[tri]))
                pki += 1
                tri += 1
            else:
                # Unexpected ordering; advance the earlier pointer to prevent stalling
                if trough_inds[tri] <= peak_inds[pki]:
                    tri += 1
                else:
                    pki += 1

        proceed_check = 1

    corrected_peaks = np.asarray(corrected_peaks, dtype=int)
    corrected_troughs = np.asarray(corrected_troughs, dtype=int)

    # Drop any extrema that fall in the reflected padding
    corrected_peaks = corrected_peaks[corrected_peaks < n]
    corrected_troughs = corrected_troughs = corrected_troughs[corrected_troughs < n]

    return corrected_peaks, corrected_troughs


## find pauses and onsets
## TODO: test this.
def find_respiratory_pauses_and_onsets(
    resp: ArrayLike,
    fs: float,
    min_pause_ms: float = 100.0,
    *,
    slope_percentile: float = 30.0,
    slope_threshold: float | None = None,
) -> dict[str, np.ndarray]:
    """
    Vectorized pause/onset detection.
      - Pauses: contiguous samples where |dx| <= threshold for >= min_pause_ms
      - Onsets: first sample index after each qualifying pause

    Parameters
    ----------
    resp : 1D signal
    fs   : sampling rate (Hz)
    min_pause_ms : minimum pause duration (ms)
    slope_percentile : percentile to set |dx| threshold when slope_threshold is None
    slope_threshold  : explicit |dx| threshold; overrides percentile if provided
    """
    x = np.asarray(resp, dtype=float).ravel()
    n = x.size
    if n < 2:
        empty = np.array([], dtype=int)
        return {"pauses": empty, "onsets": empty}

    # derivative magnitude
    dx = np.abs(np.diff(x, prepend=x[0]))

    # threshold (explicit beats percentile)
    thr = (
        float(slope_threshold)
        if slope_threshold is not None
        else float(np.percentile(dx, slope_percentile))
    )

    flat = dx <= thr  # boolean mask of "pause" samples

    # run-length encode `flat` (find starts/ends of True segments)
    # transitions: False->True (start), True->False (end)
    padded = np.pad(flat.astype(np.int8), (1, 1))
    d = np.diff(padded)
    starts = np.flatnonzero(d == 1)  # indices where run starts
    ends = np.flatnonzero(d == -1)  # indices where run ends (exclusive)
    lengths = ends - starts

    # keep runs long enough
    min_len = int(round((min_pause_ms / 1000.0) * fs))
    keep = lengths >= max(1, min_len)
    pauses = starts[keep]
    onsets = ends[keep]

    # clamp to signal bounds (defensive; ends are already exclusive)
    pauses = pauses[(pauses >= 0) & (pauses < n)]
    onsets = onsets[(onsets >= 0) & (onsets <= n)]

    return {"pauses": pauses.astype(int), "onsets": onsets.astype(int)}


## find resp offsets

## find resp volume
## calculate secondary features
## create resp ERPs
