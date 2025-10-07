# all the onset and pause detection methods.
# legacy, new, event-marker aided.
from __future__ import annotations

# from typing import Any
import numpy as np
from numpy.typing import ArrayLike


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


# TODO: insert new inhale onset detection method here.


# find pause based on two segment slope method. relies on an accurate inhale onset estimate.
# TODO: needs test
def find_pause_slope(
    resp: ArrayLike,
    fs: float,
    inhaleonsets: ArrayLike,
    exhaletroughs: ArrayLike,
    min_edge_ms: float = 200,
    flat_frac: float = 0.5,
) -> tuple[np.ndarray]:
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
        t = np.arange(n - 1) / fs  # time in s
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
            hinge = np.max(0, t - tau)
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

    return exhale_offsets  # type: ignore
