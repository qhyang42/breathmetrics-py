# all the onset and pause detection methods.
# legacy, new, event-marker aided.
from __future__ import annotations

# from typing import Any
import numpy as np
from numpy.typing import ArrayLike

from breathmetrics.utils import (
    find_inflection_vectorized,
    find_inflection_from_mid3,
)


##
def find_onsets_and_pauses_legacy(
    resp: ArrayLike,
    peaks: ArrayLike,
    troughs: ArrayLike,
    nbins: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    legacy method to find resp onsets and pauses.
    from original breathmetrics matlab code.
    inputs:
        resp: respiration signal
        fs: sampling frequency
        peaks: indices of inspiratory peaks
        toughs: indices of expiratory troughs
        nbins: number of bins to use for histogramming the breath durations.
    outputs:
        onsets: indices of inspiratory onsets
        pauses: indices of expiratory pauses
        inhale pause onsets: indices of inspiratory pause onsets
        exhale pause onsets: indices of expiratory pause onsets
    """
    # steps:
    # 1. find breath onsets for the first and last breaths
    # 2. for each trough to peak segment, find if htere is a respiratory pause.
    # 3. if no pause, find where the trace crosses zero (or close to zero) before the peak.
    # 4. if pause, find the amplitude range of the pause.
    # 5. find onsets and offsets of the pause.
    # 6. find exhale onset in the peak to trough segment where it crosses zero (or close to zero).
    # 7. repeat

    from breathmetrics.utils import hist_like_matlab

    # set up variables
    resp = np.asarray(resp, dtype=float)
    peaks = np.asarray(peaks, dtype=int)
    troughs = np.asarray(troughs, dtype=int)
    nbins = int(nbins)

    # assume there is exactly one onset per breath
    inhaleonsets = np.zeros_like(peaks)
    exhaleonsets = np.zeros_like(troughs)

    # inhale pauses happen after inhale osnets, exhale pauses happen after exhale onsets
    inhalepauseonsets = np.full_like(peaks, np.nan, dtype=float)
    exhalepauseonsets = np.full_like(troughs, np.nan, dtype=float)

    # bin threshold setup
    if nbins >= 100:
        maxpausebins = 5
    else:
        maxpausebins = 2

    maximum_bin_thr = 5
    upper_thr = np.round(nbins * 0.7)
    lower_thr = np.round(nbins * 0.3)

    # if bin method fails, use zero crossing method
    simple_zero_crossing = np.mean(resp)

    # head and tail onsets are hard to estimate without lims. use average
    # breathing interval in these cases.
    tail_onset_lims = np.floor(np.mean(np.diff(peaks)))

    if peaks[0] > tail_onset_lims:
        first_zero_crossings_bound = peaks[0] - tail_onset_lims
    else:
        first_zero_crossings_bound = 1

    # do the first onset
    this_window = resp[first_zero_crossings_bound : peaks[0]]
    custom_bins = np.linspace(np.min(this_window), np.max(this_window), nbins)
    amplitude_values, window_bins = hist_like_matlab(this_window, custom_bins)
    mode_bin = np.argmax(amplitude_values)
    zero_crossing_threshold = window_bins[mode_bin]

    if mode_bin < lower_thr or mode_bin > upper_thr:
        # use simple zero crossing
        zero_crossing_threshold = simple_zero_crossing

    possible_inhale_ind = this_window < zero_crossing_threshold
    if np.sum(possible_inhale_ind) > 0:
        mask = possible_inhale_ind == 1
        inhale_onset = np.where(mask)[0][-1]
        inhaleonsets[0] = inhale_onset + first_zero_crossings_bound
    else:
        inhaleonsets[0] = first_zero_crossings_bound

    for thisbreath in range(0, len(peaks) - 1):
        inhale_window = resp[troughs[thisbreath] : peaks[thisbreath + 1]]
        custom_bins = np.linspace(np.min(inhale_window), np.max(inhale_window), nbins)
        amplitude_values, window_bins = hist_like_matlab(inhale_window, custom_bins)
        mode_bin = np.argmax(amplitude_values)
        max_bin_ratio = amplitude_values[mode_bin] / np.mean(amplitude_values)
        isexhalepause = not (
            mode_bin < lower_thr
            or mode_bin > upper_thr
            or max_bin_ratio < maximum_bin_thr
        )

        if not (isexhalepause):
            # no resp pause. use baseline crossing as inhale onset
            this_inhale_thr = simple_zero_crossing
            possible_inhale_ind = inhale_window < this_inhale_thr
            mask = possible_inhale_ind == 1
            inhale_onset = np.where(mask)[0][-1]
            exhalepauseonsets[thisbreath] = np.nan
            inhaleonsets[thisbreath + 1] = inhale_onset + troughs[thisbreath]

        else:
            min_pause_range = window_bins[mode_bin]
            max_pause_range = window_bins[mode_bin + 1]
            max_bin_total = amplitude_values[mode_bin]

            binning_threshold = 0.25

            for additional_bin in range(0, maxpausebins):
                this_bin = mode_bin - additional_bin
                nvals_added = amplitude_values[this_bin]
                if nvals_added > max_bin_total * binning_threshold:
                    min_pause_range = window_bins[this_bin]

            # add bins in negative direction
            for additional_bin in range(0, maxpausebins):
                this_bin = mode_bin + additional_bin
                nvals_added = amplitude_values[this_bin]
                if nvals_added > max_bin_total * binning_threshold:
                    max_pause_range = window_bins[this_bin]

            mask = (inhale_window > min_pause_range) & (inhale_window < max_pause_range)
            putative_pause_inds = np.where(mask)[0]
            pause_onset = putative_pause_inds[0] - 1
            inhale_onset = putative_pause_inds[-1] + 1
            exhalepauseonsets[thisbreath] = pause_onset + troughs[thisbreath]
            inhaleonsets[thisbreath + 1] = inhale_onset + troughs[thisbreath]

        # do the exhale onset
        # trough always follows peak
        exhale_window = resp[peaks[thisbreath] : troughs[thisbreath]]
        custom_bins = np.linspace(np.min(exhale_window), np.max(exhale_window), nbins)
        amplitude_values, window_bins = hist_like_matlab(exhale_window, custom_bins)
        mode_bin = np.argmax(amplitude_values)
        max_bin_ratio = amplitude_values[mode_bin] / np.mean(amplitude_values)

        isinhalepause = not (
            mode_bin < lower_thr
            or mode_bin > upper_thr
            or max_bin_ratio < maximum_bin_thr
        )

        if not (isinhalepause):
            # no resp pause. use baseline crossing as exhale onset
            this_exhale_thr = simple_zero_crossing
            possible_exhale_ind = exhale_window > this_exhale_thr
            mask = possible_exhale_ind == 1
            exhale_onset = np.where(mask)[0][-1]
            inhalepauseonsets[thisbreath] = np.nan
            exhaleonsets[thisbreath] = exhale_onset + peaks[thisbreath]

        else:
            min_pause_range = window_bins[mode_bin]
            max_pause_range = window_bins[mode_bin + 1]
            max_bin_total = amplitude_values[mode_bin]

            binning_threshold = 0.25

            for additional_bin in range(0, maxpausebins):
                this_bin = mode_bin - additional_bin
                nvals_added = amplitude_values[this_bin]
                if nvals_added > max_bin_total * binning_threshold:
                    min_pause_range = window_bins[this_bin]

            # add bins in negative direction
            for additional_bin in range(0, maxpausebins):
                this_bin = mode_bin + additional_bin
                nvals_added = amplitude_values[this_bin]
                if nvals_added > max_bin_total * binning_threshold:
                    max_pause_range = window_bins[this_bin]

            mask = (exhale_window > min_pause_range) & (exhale_window < max_pause_range)
            putative_pause_inds = np.where(mask)[0]
            pause_onset = putative_pause_inds[0] - 1
            exhale_onset = putative_pause_inds[-1] + 1
            inhalepauseonsets[thisbreath] = pause_onset + peaks[thisbreath]
            exhaleonsets[thisbreath] = exhale_onset + peaks[thisbreath]

    # do the last exhale onset because it's not in a peak to peak segment
    # treat it like the first inhale onset
    if len(resp) - peaks[-1] > tail_onset_lims:
        last_zero_crossings_bound = peaks[-1] + tail_onset_lims
    else:
        last_zero_crossings_bound = len(resp)

    exhale_window = resp[peaks[-1] : last_zero_crossings_bound]
    zero_crossing_threshold = simple_zero_crossing
    possible_exhale_ind = exhale_window > zero_crossing_threshold

    if np.sum(possible_exhale_ind) > 0:
        mask = possible_exhale_ind == 1
        exhale_best_guess = np.where(mask)[0][0]
        exhaleonsets[-1] = exhale_best_guess + peaks[-1]
    else:
        exhaleonsets[-1] = last_zero_crossings_bound

    return inhaleonsets, exhaleonsets, inhalepauseonsets, exhalepauseonsets


# Adam's breathing onset detection.
def find_onsets_new(resp: ArrayLike, fs: float, peaks: ArrayLike) -> np.ndarray:
    """detect breathing onset from resp peaks
    INPUTS
      resp  : breathing signal (vector)
      fs    : sampling rate (Hz)
      peaks : respiratory peak indices (samples)
    OUTPUTS
      inhaleOnsets : inhale-onset index for each peak (samples)
    """
    from scipy.ndimage import gaussian_filter1d

    resp = np.asarray(resp, dtype=float)
    peaks = np.asarray(peaks, dtype=int)
    nsamples = len(resp)

    inhaleonsets = np.full_like(peaks, np.nan, dtype=float)
    for i in range(len(peaks)):
        curidx = peaks[i]
        winstart = curidx - fs * 3
        winend1 = curidx

        # clamp to signal edges
        winstart = max(0, int(winstart))
        winend1 = min(nsamples - 1, int(winend1))
        insig = resp[winstart:winend1]

        # mimicking matlab smoothing behavior.
        # inSig = smoothdata(resp(winStart:winEnd1), 'gaussian', round(fs));
        win = int(round(fs))
        sigma = win / 6.0
        insig_sm = gaussian_filter1d(insig, sigma=sigma, mode="nearest")

        # find initial guess
        # [adj, _] = find_inflection(insig_sm, slope_based=True)
        [adj, _] = find_inflection_vectorized(
            insig_sm, slope_based=True
        )  # this is a vectorized version of find_inflection. If doesn't work, switch back to the original find_inflection.

        # second pass
        # winend2 = curidx + round(fs * 0.2)
        winend2 = curidx  # QY: this works better than a large window in our toy dataset. not sure if this is a generalizable solution. needs testing.
        winend2 = min(nsamples - 1, int(winend2))

        insig2 = resp[winstart:winend2]
        win = int(round(fs / 2))
        sigma = win / 6.0
        insig2_sm = gaussian_filter1d(insig2, sigma=sigma, mode="nearest")

        # second guess
        adj2 = find_inflection_from_mid3(insig2_sm, adj, fs, insig2)
        bStart = adj2 + winstart
        inhaleonsets[i] = bStart
    return inhaleonsets


# find pause based on two segment slope method. relies on an accurate inhale onset estimate.
def find_pause_slope(
    resp: ArrayLike,
    fs: float,
    inhaleonsets: ArrayLike,
    exhaletroughs: ArrayLike,
    min_edge_ms: float = 200,
    flat_frac: float = 0.5,
) -> np.ndarray:
    """
    find pause based on two segment slope method.
    inputs:
        resp: respiration signal
        fs: sampling frequency
        inhaleonsets: indices of inspiratory onsets
        exhaletroughs: indices of expiratory troughs
    outputs:
        exhale_offsets: indices of expiratory offsets (same as pause onsets)
    """
    exhale_offsets = np.full_like(exhaletroughs, np.nan, dtype=float)
    exhale_offsets = exhale_offsets[:-1]  # first breath doesn't need a pause

    x = np.asarray(resp, dtype=float)
    inhaleonsets = np.asarray(inhaleonsets, dtype=int)
    exhaletroughs = np.asarray(exhaletroughs, dtype=int)
    fs = float(fs)
    for i in range(len(exhaletroughs) - 1):  # first breath doesn't need a pause
        inhalewindow = x[exhaletroughs[i] : inhaleonsets[i + 1]]

        # slope segment based pause detection here
        y = inhalewindow[:]
        n = len(y)
        t = np.arange(n) / fs  # time in s
        y0 = y - np.mean(y)

        # steps:
        # run 1 step model
        # search through taus for 2 segment model
        # update best rss and best b2 as I go
        # compute bic for best 2 step model
        # compare bic1 and bic2
        # if bic 2 is smaller,
        # and if slope 2 is smaller than slope 1
        # its a real pause. record tauidx.
        # this exhale pause onset equals exhale trough + tauidx

        # ----- 1-step linear: y = a + b t
        X1 = np.column_stack((np.ones(n), t))
        b1, *_ = np.linalg.lstsq(X1, y0, rcond=None)
        yhat1 = X1 @ b1
        rss1 = np.sum((y0 - yhat1) ** 2)
        bic1 = n * np.log(rss1 / n) + 2 * np.log(n)

        # ----- 2-step continuous hinge: y = a + b t + c * max(0, t - tau)
        # skip two steps if not enough samples at the edge
        minEdge = max(1, int(np.floor(min_edge_ms / 1000 * fs)))
        if n - 2 * minEdge < 1:
            bic2 = np.nan
            # exhale offset of this breath is nan
            exhale_offsets[i] = np.nan
            # continue to next loop
            continue

        best_rss = np.inf
        best_beta = None
        best_tauIdx = 0
        for tauidx in range(minEdge, n - minEdge):
            tau = t[tauidx]
            hinge = np.maximum(0, t - tau)
            X2 = np.column_stack((np.ones(n), t, hinge))
            b2, *_ = np.linalg.lstsq(X2, y0, rcond=None)
            yhat2 = X2 @ b2
            rss2 = np.sum((y0 - yhat2) ** 2)

            if rss2 < best_rss:
                best_rss = rss2
                best_beta = b2
                best_tauIdx = tauidx

        bic2 = n * np.log(best_rss / n) + 4 * np.log(n)

        # ---- decision rule --- #
        if bic2 < bic1 and best_beta is not None:
            a, b, c = best_beta
            slopeEarly = b
            slopeLate = b + c
            if (c < 0) and (abs(slopeEarly) * flat_frac >= slopeLate):
                # choose 2 steps
                exhale_offsets[i] = exhaletroughs[i] + best_tauIdx
            else:
                exhale_offsets[i] = np.nan
        else:
            exhale_offsets[i] = np.nan

    return exhale_offsets


def find_pause_slope_vectorized(
    resp: ArrayLike,
    fs: float,
    inhaleonsets: ArrayLike,
    exhaletroughs: ArrayLike,
    min_edge_ms: float = 200,
    flat_frac: float = 0.5,
) -> np.ndarray:
    """
    find pause based on two segment slope method.
    outputs:
        exhale_offsets: indices of expiratory offsets (same as pause onsets)
    """
    x = np.asarray(resp, dtype=float)
    inhaleonsets = np.asarray(inhaleonsets, dtype=int)
    exhaletroughs = np.asarray(exhaletroughs, dtype=int)
    fs = float(fs)

    exhale_offsets = np.full_like(exhaletroughs, np.nan, dtype=float)[:-1]

    tiny = np.finfo(float).tiny

    def _fit_1step_bic_and_slope(
        y0: np.ndarray, t: np.ndarray
    ) -> tuple[float, float, float]:
        """
        Fit y = a + b t (OLS), return (bic1, rss1, slope b).
        Uses normal equations with sums (no lstsq).
        """
        n = y0.size
        # Sums
        S1 = float(n)
        St = float(np.sum(t))
        St2 = float(np.sum(t * t))
        Sy = float(np.sum(y0))
        Sty = float(np.sum(t * y0))
        yTy = float(np.sum(y0 * y0))

        # Solve 2x2: [[S1, St],[St, St2]] [a,b] = [Sy, Sty]
        det = S1 * St2 - St * St
        if det == 0.0:
            # Degenerate t (shouldn't happen with arange/fs), fallback consistent with "bad fit"
            rss1 = yTy
            b = 0.0
        else:
            a = (Sy * St2 - Sty * St) / det
            b = (S1 * Sty - St * Sy) / det
            # rss = y'y - beta^T X^T y  (since OLS)
            rss1 = yTy - (a * Sy + b * Sty)

        rss1 = float(max(rss1, 0.0))
        bic1 = n * np.log(max(rss1 / n, tiny)) + 2 * np.log(n)
        return float(bic1), float(rss1), float(b)

    def _best_hinge_fit(
        y0: np.ndarray, t: np.ndarray, min_edge: int
    ) -> tuple[float, np.ndarray | None, int]:
        """
        Fit y = a + b t + c * max(0, t - tau) for tau = t[k], k in [min_edge, n-min_edge-1]
        Returns (best_rss, best_beta [a,b,c], best_tauIdx).
        Uses suffix sums so each candidate is O(1) + a 3x3 solve.
        """
        n = y0.size
        tau_candidates = range(min_edge, n - min_edge)

        # Precompute global sums
        S1 = float(n)
        St = float(np.sum(t))
        St2 = float(np.sum(t * t))
        Sy = float(np.sum(y0))
        Sty = float(np.sum(t * y0))
        yTy = float(np.sum(y0 * y0))

        # Suffix sums for fast hinge-related sums over j>=k
        # (store as float arrays for speed)
        t_f = t.astype(float, copy=False)
        y_f = y0.astype(float, copy=False)

        suf1 = np.empty(n + 1, dtype=float)  # counts (as float)
        suft = np.empty(n + 1, dtype=float)  # sum t
        suft2 = np.empty(n + 1, dtype=float)  # sum t^2
        sufy = np.empty(n + 1, dtype=float)  # sum y
        sufty = np.empty(n + 1, dtype=float)  # sum t*y

        suf1[n] = 0.0
        suft[n] = 0.0
        suft2[n] = 0.0
        sufy[n] = 0.0
        sufty[n] = 0.0
        for i in range(n - 1, -1, -1):
            suf1[i] = suf1[i + 1] + 1.0
            suft[i] = suft[i + 1] + t_f[i]
            suft2[i] = suft2[i + 1] + t_f[i] * t_f[i]
            sufy[i] = sufy[i + 1] + y_f[i]
            sufty[i] = sufty[i + 1] + t_f[i] * y_f[i]

        best_rss = np.inf
        best_beta = None
        best_tauIdx = int(min_edge)

        for k in tau_candidates:
            tau = float(t_f[k])
            m = suf1[
                k
            ]  # number of points with t >= tau (since tau=t[k] and t is increasing)
            sum_t = suft[k]
            sum_t2 = suft2[k]
            sum_y = sufy[k]
            sum_ty = sufty[k]

            # hinge sums over j>=k where h = t - tau, else 0
            Sh = sum_t - m * tau
            Sth = sum_t2 - tau * sum_t
            Sh2 = sum_t2 - 2.0 * tau * sum_t + m * tau * tau
            Syh = sum_ty - tau * sum_y

            # Build XtX and XtY for X=[1, t, h]
            XtX = np.array(
                [
                    [S1, St, Sh],
                    [St, St2, Sth],
                    [Sh, Sth, Sh2],
                ],
                dtype=float,
            )
            XtY = np.array([Sy, Sty, Syh], dtype=float)

            # Solve (XtX) beta = XtY
            # If singular/ill-conditioned, skip this tau (same effect as "not best")
            try:
                beta = np.linalg.solve(XtX, XtY)
            except np.linalg.LinAlgError:
                continue

            # rss = y'y - beta^T XtY
            rss = yTy - float(beta @ XtY)
            rss = float(max(rss, 0.0))

            if rss < best_rss:
                best_rss = rss
                best_beta = beta
                best_tauIdx = int(k)

        return float(best_rss), best_beta, best_tauIdx

    for i in range(len(exhaletroughs) - 1):
        start = exhaletroughs[i]
        end = inhaleonsets[i + 1]
        y = x[start:end]
        n = y.size
        if n == 0:
            exhale_offsets[i] = np.nan
            continue

        t = np.arange(n, dtype=float) / fs
        y0 = y - float(np.mean(y))

        # ----- 1-step
        bic1, rss1, _b = _fit_1step_bic_and_slope(y0, t)

        # ----- 2-step (hinge)
        minEdge = max(1, int(np.floor(min_edge_ms / 1000.0 * fs)))
        if n - 2 * minEdge < 1:
            exhale_offsets[i] = np.nan
            continue

        best_rss, best_beta, best_tauIdx = _best_hinge_fit(y0, t, minEdge)
        bic2 = n * np.log(max(best_rss / n, tiny)) + 4 * np.log(n)

        # ---- decision rule --- #
        if (bic2 < bic1) and (best_beta is not None):
            a, b, c = best_beta
            slopeEarly = b
            slopeLate = b + c
            if (c < 0) and (abs(slopeEarly) * flat_frac >= slopeLate):
                exhale_offsets[i] = exhaletroughs[i] + best_tauIdx
            else:
                exhale_offsets[i] = np.nan
        else:
            exhale_offsets[i] = np.nan

    # no but actually this is exhale pause onset instead of exhale offset
    exhale_pause_onsets = exhale_offsets
    return exhale_pause_onsets
