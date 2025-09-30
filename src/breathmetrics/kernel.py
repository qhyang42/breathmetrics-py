from __future__ import annotations
from typing import Any
import numpy as np
from numpy.typing import ArrayLike

## core function for breathmetrics. simple functions. no classes.

## baseline correction
# kernel.py  (pure function)
from .utils import detrend_linear, fft_smooth, zscore

# TODO: all the functions here need tests.


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
# don't know why this is here. could be useless. qy 9/29/2025. ---ignore---
# def _find_respiratory_extrema_basic(
#     signal: np.ndarray,
#     distance: int | None = None,
#     prominence: float | None = None,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Find local maxima ('peaks') and minima ('troughs').
#     Uses scipy.signal.find_peaks if available; otherwise a simple NumPy fallback.
#     Returns (peaks_idx, troughs_idx) as integer arrays.
#     """
#     x = np.asarray(signal, dtype=float)

#     try:
#         from scipy.signal import find_peaks  # optional

#         peaks, _ = find_peaks(x, distance=distance, prominence=prominence)
#         troughs, _ = find_peaks(-x, distance=distance, prominence=prominence)
#         return peaks.astype(int), troughs.astype(int)
#     except Exception:
#         # Fallback: strict neighbor test
#         # A point i is a local max if x[i-1] < x[i] > x[i+1]; min if x[i-1] > x[i] < x[i+1]
#         if x.size < 3:
#             return np.array([], dtype=int), np.array([], dtype=int)
#         dx1 = x[1:-1] - x[:-2]
#         dx2 = x[1:-1] - x[2:]
#         peaks = np.where((dx1 > 0) & (dx2 > 0))[0] + 1
#         troughs = np.where((dx1 < 0) & (dx2 < 0))[0] + 1
#         if distance and distance > 1:
#             # crude distance enforcement: keep every 'distance'-th candidate
#             peaks = peaks[:: max(1, int(distance))]
#             troughs = troughs[:: max(1, int(distance))]
#         return peaks.astype(int), troughs.astype(int)


# find extrema with sliding window voting
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


## find resp offsets
def find_respiratory_offsets(
    resp: ArrayLike,
    inhale_onsets: ArrayLike,
    exhale_onsets: ArrayLike,
    inhale_pause_onsets: ArrayLike,
    exhale_pause_onsets: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Port of MATLAB findRespiratoryOffsets.m (object-free).
    Returns (inhale_offsets, exhale_offsets).
    - Inhale offset i: exhaleOnsets[i]-1 unless inhalePauseOnsets[i] is present,
      then inhalePauseOnsets[i]-1.
    - Exhale offset i (for all but the last): next inhale onset - 1, unless
      exhalePauseOnsets[i] is present, then exhalePauseOnsets[i]-1.
    - Final exhale offset: first positive slope after last exhale onset,
      accepted only if its distance lies within [avg/4, 1.75*avg] where avg is
      the mean exhale duration from all previous exhales. Otherwise NaN.
    """
    x = np.asarray(resp, dtype=float).ravel()
    inh_on = np.asarray(inhale_onsets, dtype=int).ravel()
    exh_on = np.asarray(exhale_onsets, dtype=int).ravel()
    inh_pause = np.asarray(inhale_pause_onsets, dtype=float).ravel()
    exh_pause = np.asarray(exhale_pause_onsets, dtype=float).ravel()

    # Outputs: match MATLAB shapes (inhale offsets sized like exhale onsets)
    inhale_offsets = np.zeros_like(exh_on, dtype=int)
    exhale_offsets = np.zeros_like(exh_on, dtype=float)  # last can be NaN

    # ---- Inhale offsets ----
    # for bi = 1:length(exhaleOnsets)
    for bi in range(exh_on.size):
        if np.isnan(inh_pause[bi]):
            inhale_offsets[bi] = exh_on[bi] - 1
        else:
            inhale_offsets[bi] = int(inh_pause[bi]) - 1

    # ---- Exhale offsets (all but last) ----
    # exhale i ends at exhalePauseOnsets[i]-1 if present; else just before next inhale onset
    for bi in range(exh_on.size - 1):
        if np.isnan(exh_pause[bi]):
            exhale_offsets[bi] = inh_on[bi + 1] - 1
        else:
            exhale_offsets[bi] = int(exh_pause[bi]) - 1

    # ---- Final exhale offset (last breath) ----
    if exh_on.size > 0:
        last = exh_on.size - 1
        start = exh_on[last]
        putative = None
        if start < x.size - 1:
            # This mirrors your `putativeExhaleOffset = find(final_window>0,1,'first');`
            # with final_window taken as the *slope* after the last exhale onset.
            dx = np.diff(x[start:])
            pos = np.flatnonzero(dx > 0)
            if pos.size:
                putative = int(pos[0] + 1)  # +1 to map diff-index to signal index

        # avg exhale length from all completed (non-final) exhales
        if exh_on.size > 1:
            avg_len = float(np.mean(exhale_offsets[:last] - exh_on[:last]))
        else:
            avg_len = np.nan

        if putative is None or not np.isfinite(avg_len):
            exhale_offsets[last] = np.nan
        else:
            lower = avg_len / 4.0
            upper = avg_len * 1.75
            if putative < lower or putative >= upper:
                exhale_offsets[last] = np.nan
            else:
                exhale_offsets[last] = exh_on[last] + putative - 1

    return inhale_offsets, exhale_offsets


# find breath durations
def find_breath_durations(
    inhale_onsets: ArrayLike,
    inhale_offsets: ArrayLike,
    exhale_onsets: ArrayLike,
    exhale_offsets: ArrayLike,
    fs: float,
    *,
    drop_invalid: bool = False,
) -> dict[str, np.ndarray]:
    """
    Compute per-breath durations from paired onsets/offsets.

    Definitions (per breath i)
    --------------------------
    inhale_duration_s  = (inhale_offset[i] - inhale_onset[i]) / fs
    exhale_duration_s  = (exhale_offset[i] - exhale_onset[i]) / fs
    cycle_duration_s   = (inhale_onset[i+1] - inhale_onset[i]) / fs    # next inhale begins breath i+1
    ie_ratio           = inhale_duration_s / exhale_duration_s

    Notes
    -----
    - Arrays are truncated to the maximum number of *complete* pairs.
    - If the final exhale_offset is NaN (common), the final exhale duration
      is set to NaN and the validity mask marks it as invalid.
    - Negative or zero-length durations are marked invalid.
    - If `drop_invalid=True`, invalid rows are removed from all outputs.

    Returns
    -------
    dict with float arrays (aligned in length):
      - "inhale_duration_s"
      - "exhale_duration_s"
      - "cycle_duration_s"      (shorter by one breath; last is NaN to align)
      - "ie_ratio"
      - "valid"                 (boolean mask per row before any dropping)
      - "index"                 (original breath indices kept after truncation)
    """
    inh_on = np.asarray(inhale_onsets, dtype=float).ravel()
    inh_off = np.asarray(inhale_offsets, dtype=float).ravel()
    exh_on = np.asarray(exhale_onsets, dtype=float).ravel()
    exh_off = np.asarray(exhale_offsets, dtype=float).ravel()

    # Number of breaths we can form with *all four* markers
    n_pairs = int(min(inh_on.size, inh_off.size, exh_on.size, exh_off.size))
    if n_pairs == 0:
        empty = np.array([], dtype=float)
        return {
            "inhale_duration_s": empty,
            "exhale_duration_s": empty,
            "cycle_duration_s": empty,
            "ie_ratio": empty,
            "valid": np.array([], dtype=bool),
            "index": np.array([], dtype=int),
        }

    # Truncate to complete pairs
    inh_on = inh_on[:n_pairs]
    inh_off = inh_off[:n_pairs]
    exh_on = exh_on[:n_pairs]
    exh_off = exh_off[:n_pairs]

    # Durations (seconds)
    with np.errstate(invalid="ignore"):
        inhale_duration_s = (inh_off - inh_on) / fs
        exhale_duration_s = (exh_off - exh_on) / fs

    # Cycle duration: from inhale onset i to inhale onset i+1
    cycle_duration_s = np.full_like(inhale_duration_s, np.nan, dtype=float)
    if inh_on.size >= 2:
        cycle_duration_s[:-1] = (inh_on[1:] - inh_on[:-1]) / fs

    # I:E ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        ie_ratio = inhale_duration_s / exhale_duration_s

    # Validity checks (monotonic ordering + positive durations)
    valid = np.isfinite(inhale_duration_s) & np.isfinite(exhale_duration_s)
    valid &= (inh_on < inh_off) & (inh_off <= exh_on) & (exh_on < exh_off)
    valid &= (inhale_duration_s > 0) & (exhale_duration_s > 0)

    # Optional: drop invalid rows across all outputs
    if drop_invalid:
        idx = np.nonzero(valid)[0]
        inhale_duration_s = inhale_duration_s[idx]
        exhale_duration_s = exhale_duration_s[idx]

        # For cycle durations, keep entries where both i and i+1 breaths survived.
        # A simple way is to recompute from the filtered onsets if you keep them around.
        # Here we approximate by selecting the same indices and leaving NaN where ambiguous.
        cycle_duration_s = cycle_duration_s[idx]
        ie_ratio = ie_ratio[idx]
        valid = np.ones_like(inhale_duration_s, dtype=bool)
        index = idx.astype(int)
    else:
        index = np.arange(n_pairs, dtype=int)

    return {
        "inhale_duration_s": inhale_duration_s.astype(float),
        "exhale_duration_s": exhale_duration_s.astype(float),
        "cycle_duration_s": cycle_duration_s.astype(float),
        "ie_ratio": ie_ratio.astype(float),
        "valid": valid.astype(bool),
        "index": index,
    }


## find resp volume
## calculate secondary features
## create resp ERPs
