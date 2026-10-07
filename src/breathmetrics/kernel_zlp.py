# ZLP breath segmentation kernel.
#
# Python port of the ZLP (Zelano Lab Preprocessing) segmentation engine
# contributed to the MATLAB toolbox (breathmetrics_functions/
# prepBreathTrace_zlp.m and findInhaleOnsets_zlp.m; specification in the
# MATLAB repo's docs/zlp_segmentation.md and this repo's
# docs/zlp_segmentation.md). Designed for amplitude-non-stationary
# recordings - drifting baselines, breathing depth that changes several-fold
# between task blocks, paced or very slow breathing - where global-criterion
# extrema and pause-band onset placement struggle.
#
# Detection runs on an internally conditioned, windowed-normalized copy of
# the signal; the Breathe object's own traces are never modified, so every
# downstream feature (volumes included) stays in the signal's native units.
#
# The port replicates MATLAB semantics exactly: movmean/movstd centered
# SHRINKING windows (including the even-window asymmetry), directional
# movmax/movmin, trailing movsum, fillmissing linear + nearest edges,
# round-half-away-from-zero, and findpeaks' filter order (prominence first,
# then greedy-by-height minimum distance - scipy's find_peaks applies
# distance first). Landmark parity with the MATLAB implementation was
# verified sample-for-sample on the MATLAB repo's sample_data.mat
# (660,001 samples @ 1 kHz: 132/132 peaks and troughs, 131/131 onsets,
# detection trace max |diff| 5e-13).
#
# All indices are 0-based.

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import find_peaks

__all__ = ["find_inhale_onsets", "pair_breath_landmarks", "prep_breath_trace"]


# --------------------------------------------------------------------------
# MATLAB-semantics helpers
# --------------------------------------------------------------------------


def _mround(x: float) -> int:
    """MATLAB round(): half away from zero (Python's round is banker's)."""
    return int(np.floor(x + 0.5)) if x >= 0 else int(np.ceil(x - 0.5))


def _fillmissing_linear(x: ArrayLike) -> np.ndarray:
    """MATLAB fillmissing(x,'linear','EndValues','nearest')."""
    x = np.asarray(x, dtype=float).ravel().copy()
    ok = np.isfinite(x)
    if ok.all():
        return x
    if not ok.any():
        raise ValueError("signal contains no finite samples")
    idx = np.arange(x.size)
    x[~ok] = np.interp(idx[~ok], idx[ok], x[ok])
    return x


def _centered_bounds(n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Window bounds per MATLAB movmean/movstd: odd k = 2m+1 -> [i-m, i+m];
    even k = 2m -> [i-m, i+m-1]; truncated (shrinking) at the edges."""
    m_back = k // 2
    m_fwd = k - m_back - 1
    i = np.arange(n)
    lo = np.maximum(0, i - m_back)
    hi = np.minimum(n - 1, i + m_fwd)
    return lo, hi


def _movmean(x: np.ndarray, k: int) -> np.ndarray:
    n = x.size
    lo, hi = _centered_bounds(n, k)
    cs = np.concatenate(([0.0], np.cumsum(x)))
    return (cs[hi + 1] - cs[lo]) / (hi - lo + 1)


def _movstd(x: np.ndarray, k: int) -> np.ndarray:
    """MATLAB movstd (sample std, N-1), centered shrinking windows; a
    single-sample window reads 0."""
    n = x.size
    lo, hi = _centered_bounds(n, k)
    cs1 = np.concatenate(([0.0], np.cumsum(x)))
    cs2 = np.concatenate(([0.0], np.cumsum(x * x)))
    cnt = (hi - lo + 1).astype(float)
    s1 = cs1[hi + 1] - cs1[lo]
    s2 = cs2[hi + 1] - cs2[lo]
    var = np.zeros(n)
    mult = cnt > 1
    var[mult] = (s2[mult] - s1[mult] ** 2 / cnt[mult]) / (cnt[mult] - 1)
    np.clip(var, 0.0, None, out=var)
    return np.sqrt(var)


def _directional_bounds(n: int, kb: int, kf: int) -> tuple[np.ndarray, np.ndarray]:
    i = np.arange(n)
    return np.maximum(0, i - kb), np.minimum(n - 1, i + kf)


def _movmax_dir(x: np.ndarray, kb: int, kf: int) -> np.ndarray:
    n = x.size
    lo, hi = _directional_bounds(n, kb, kf)
    out = np.empty(n)
    for i in range(n):
        out[i] = x[lo[i] : hi[i] + 1].max()
    return out


def _movmin_dir(x: np.ndarray, kb: int, kf: int) -> np.ndarray:
    n = x.size
    lo, hi = _directional_bounds(n, kb, kf)
    out = np.empty(n)
    for i in range(n):
        out[i] = x[lo[i] : hi[i] + 1].min()
    return out


def _movsum_trailing(x: np.ndarray, k: int) -> np.ndarray:
    """MATLAB movsum(x,[k-1 0]): trailing window of length k, truncated."""
    cs = np.concatenate(([0.0], np.cumsum(x)))
    i = np.arange(x.size)
    lo = np.maximum(0, i - (k - 1))
    return cs[i + 1] - cs[lo]


def _movminmax_centered(x: np.ndarray, k: int, op: str) -> np.ndarray:
    n = x.size
    lo, hi = _centered_bounds(n, k)
    out = np.empty(n)
    fn = np.max if op == "max" else np.min
    for i in range(n):
        out[i] = fn(x[lo[i] : hi[i] + 1])
    return out


def _matlab_findpeaks(x: np.ndarray, prominence: float, distance: int) -> np.ndarray:
    """MATLAB findpeaks(...,'MinPeakProminence',p,'MinPeakDistance',d):
    prominence is applied FIRST, then peaks closer than d samples to a
    HIGHER surviving peak are removed (greedy by descending height)."""
    locs, _ = find_peaks(x, prominence=prominence)
    if locs.size == 0 or distance <= 1:
        return locs
    order = np.argsort(x[locs])[::-1]
    keep = np.ones(locs.size, dtype=bool)
    kept_locs: list[int] = []
    for j in order:
        if any(abs(int(locs[j]) - kl) < distance for kl in kept_locs):
            keep[j] = False
        else:
            kept_locs.append(int(locs[j]))
    return np.sort(locs[keep])


def _enforce_alt(
    det: np.ndarray, pk: ArrayLike, tr: ArrayLike, min_sep: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Strict trough->peak->trough alternation; of two same-type extrema in
    a row the more extreme wins."""
    pk = np.asarray(pk, dtype=int)
    tr = np.asarray(tr, dtype=int)
    if min_sep > 0:
        keep = np.ones(pk.size, dtype=bool)
        for k in range(1, pk.size):
            if pk[k] - pk[k - 1] < min_sep:
                keep[k] = False
        pk = pk[keep]
    ev = np.concatenate(
        [
            np.column_stack([pk, np.ones(pk.size, dtype=int)]),
            np.column_stack([tr, -np.ones(tr.size, dtype=int)]),
        ]
    )
    ev = ev[np.argsort(ev[:, 0], kind="stable")]
    keep = np.ones(ev.shape[0], dtype=bool)
    i = 0
    while i < ev.shape[0] - 1:
        j = i + 1
        while j < ev.shape[0] and not keep[j]:
            j += 1
        if j >= ev.shape[0]:
            break
        if ev[i, 1] == ev[j, 1]:
            if ev[i, 1] == 1:
                if det[ev[i, 0]] >= det[ev[j, 0]]:
                    keep[j] = False
                else:
                    keep[i] = False
                    i = j
            else:
                if det[ev[i, 0]] <= det[ev[j, 0]]:
                    keep[j] = False
                else:
                    keep[i] = False
                    i = j
        else:
            i = j
    ev = ev[keep]
    return ev[ev[:, 1] == 1, 0].astype(int), ev[ev[:, 1] == -1, 0].astype(int)


# --------------------------------------------------------------------------
# Stage 0-1: conditioning + validity-ruled extrema
# --------------------------------------------------------------------------


def prep_breath_trace(
    rsp: ArrayLike,
    fs: float,
    mode: str = "conservative",
    blank_below_frac: float | None = None,
    sigh_span: tuple[int, int] | None = None,
    floor_frac: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Condition the respiration trace and find validity-ruled extrema.

    Returns ``(det, peaks, troughs, info)``: the windowed-normalized
    detection trace and 0-based, strictly alternating extrema indices.
    ``mode='conservative'`` is the locked production choice; ``'pwl'`` and
    ``'twoscale'`` are kept for comparison. ``sigh_span`` marks a paced
    double-inhale block: inside it, of two peaks closer than 5 s only the
    first is kept.
    """
    rsp = _fillmissing_linear(rsp)
    base = _movmean(rsp, _mround(0.50 * fs))
    sc = _movstd(base, _mround(30 * fs))
    sc_floor = floor_frac * np.median(sc)
    x = base / np.maximum(sc, sc_floor)
    if blank_below_frac is not None:
        x = x.copy()
        x[sc < blank_below_frac * np.median(sc)] = 0.0
    n = x.size
    info: dict = {"mode": mode, "n_breakpoints": None}

    if mode == "pwl":
        ds = max(1, _mround(fs / 20))
        gi = np.arange(0, n, ds)
        g = x[gi]
        keep_bp = np.zeros(gi.size, dtype=bool)
        keep_bp[0] = keep_bp[-1] = True
        stack = [(0, gi.size - 1)]
        while stack:
            a, b = stack.pop()
            if b - a < 2:
                continue
            seg = g[a : b + 1]
            lin = np.linspace(seg[0], seg[-1], b - a + 1)
            err = np.abs(seg - lin)
            ei = int(np.argmax(err))
            if err[ei] > 0.15:
                c = a + ei
                keep_bp[c] = True
                stack.append((a, c))
                stack.append((c, b))
        bp = gi[keep_bp]
        det = np.interp(np.arange(n), bp, x[bp])
        info["n_breakpoints"] = int(bp.size)
        v = x[bp]
        pk = [
            bp[k] for k in range(1, bp.size - 1) if v[k] > v[k - 1] and v[k] > v[k + 1]
        ]
        tr = [
            bp[k] for k in range(1, bp.size - 1) if v[k] < v[k - 1] and v[k] < v[k + 1]
        ]
        peaks, troughs = _enforce_alt(det, pk, tr, min_sep=_mround(1.5 * fs))

    elif mode == "conservative":
        det = x
        pk = _matlab_findpeaks(det, 0.6, _mround(1.0 * fs))
        tr = _matlab_findpeaks(-det, 0.6, _mround(1.0 * fs))
        # peak validity: HEIGHT FLOOR (+0.5) and SOFT DESCENT (0.5 drop
        # within the next 1 s), waived for peaks >= 1.0 - kills
        # exhale-recovery crests before breathing pauses
        fwd_min = _movmin_dir(det, 0, _mround(1.0 * fs))
        sel = (det[pk] >= 1.0) | ((det[pk] >= 0.5) & (det[pk] - fwd_min[pk] >= 0.5))
        pk = pk[sel]
        peaks, troughs = _enforce_alt(det, pk, tr)
        # rise-fraction filter: a trough->peak rise under 40% of the local
        # breath amplitude is a bump, not a breath
        loc_amp = _movminmax_centered(
            det, _mround(30 * fs), "max"
        ) - _movminmax_centered(det, _mround(30 * fs), "min")
        good = np.ones(peaks.size, dtype=bool)
        for k in range(peaks.size):
            t0 = troughs[troughs < peaks[k]]
            if t0.size == 0:
                continue
            if det[peaks[k]] - det[t0[-1]] < 0.4 * 0.5 * loc_amp[peaks[k]]:
                good[k] = False
        peaks, troughs = _enforce_alt(det, peaks[good], troughs)

    elif mode == "twoscale":
        det = x
        skel = _movmean(x, _mround(0.5 * fs))
        pk = _matlab_findpeaks(skel, 0.45, _mround(2 * fs)).copy()
        tr = _matlab_findpeaks(-skel, 0.45, _mround(2 * fs)).copy()
        w = _mround(0.6 * fs)
        for k in range(pk.size):
            a = max(0, pk[k] - w)
            b = min(n - 1, pk[k] + w)
            pk[k] = a + int(np.argmax(det[a : b + 1]))
        for k in range(tr.size):
            a = max(0, tr[k] - w)
            b = min(n - 1, tr[k] + w)
            tr[k] = a + int(np.argmin(det[a : b + 1]))
        peaks, troughs = _enforce_alt(det, pk, tr)

    else:
        raise ValueError(f"unknown prep mode {mode!r}")

    # paced-sigh span: of two peaks within 5 s inside the span keep the FIRST
    if sigh_span is not None and peaks.size > 1:
        keep = np.ones(peaks.size, dtype=bool)
        last = -np.inf
        for k in range(peaks.size):
            in_span = sigh_span[0] <= peaks[k] <= sigh_span[1]
            if in_span and (peaks[k] - last) < 5 * fs:
                keep[k] = False
            else:
                last = peaks[k]
        peaks, troughs = _enforce_alt(det, peaks[keep], troughs)

    return det, peaks, troughs, info


# --------------------------------------------------------------------------
# Stage 2-3: eligibility rules + onset placement
# --------------------------------------------------------------------------


def find_inhale_onsets(
    det: ArrayLike,
    fs: float,
    peaks: ArrayLike,
    troughs: ArrayLike,
    method: str = "kneeBacktrack",
    r2_factor: float = 0.4,
    r3_factor: float = 1.25,
    dip_frac: float = 0.50,
    dip_dur: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place one inhale onset per trough->peak pair on the detection trace.

    The default arguments are the locked production configuration. May
    DELETE spurious trough/peak pairs (a window with no eligible sample is a
    false split) - always take ``peaks``/``troughs`` from the outputs. The
    thresholds are calibrated to ``prep_breath_trace``'s normalized det
    trace; feeding a raw-unit signal will not work.
    """
    resp = np.asarray(det, dtype=float).ravel()
    peaks_l = list(np.asarray(peaks, dtype=int))
    troughs = np.asarray(troughs, dtype=int)
    d = _movmean(np.concatenate(([0.0], np.diff(resp))) * fs, _mround(0.12 * fs))

    # rule 1 (not descending): x(t) - x(t-0.33s) > -0.2
    lag = _mround(0.33 * fs)
    d33 = resp - np.concatenate((np.full(lag, resp[0]), resp[:-lag]))
    elig = d33 > -0.20

    # rule 3 slope-contrast field (landing refinement only, never a gate)
    lb = _mround(0.25 * fs)
    lf = _mround(0.5 * fs)
    r3v = _movmax_dir(d, lb, lf) > r3_factor * np.maximum(_movmin_dir(d, lb, lf), 0.1)
    if r2_factor == 0.25:  # legacy remap kept from the MATLAB original
        r2_factor = 0.4

    def pair_valid(w0: int, w1: int) -> np.ndarray:
        """eligible AND rule 2 (rising ahead, pair-scaled window)."""
        seg = resp[w0 : w1 + 1]
        wlen = max(_mround(0.4 * fs), _mround(0.25 * seg.size))
        rise = _movmax_dir(seg, 0, wlen) - seg
        return elig[w0 : w1 + 1] & (rise > r2_factor)

    # spurious-pair pruning
    pruned = True
    while pruned:
        pruned = False
        for k in range(len(peaks_l)):
            t0 = troughs[troughs < peaks_l[k]]
            if t0.size == 0:
                continue
            if not pair_valid(int(t0[-1]), peaks_l[k]).any():
                troughs = troughs[troughs != t0[-1]]
                del peaks_l[k]
                pruned = True
                break
    peaks = np.asarray(peaks_l, dtype=int)

    onsets: list[int] = []
    for k in range(peaks.size):
        pk = int(peaks[k])
        tr_all = troughs[troughs < pk]
        if tr_all.size == 0:
            continue
        tr = int(tr_all[-1])
        n_w = pk - tr + 1
        if n_w < _mround(0.2 * fs):
            continue
        seg = resp[tr : pk + 1]
        dseg = d[tr : pk + 1]
        e_seg = pair_valid(tr, pk)
        rng_ = resp[pk] - resp[tr]
        lo = resp[tr] + 0.15 * rng_
        hi = resp[pk] - 0.25 * rng_
        o: int | None = None

        if method == "slopeGate":
            th = 0.20 * dseg.max()
            sustain = _mround(0.10 * fs)
            upto = n_w - sustain
            cand = np.nonzero(
                (dseg[:upto] > th)
                & (seg[:upto] >= lo)
                & (seg[:upto] <= hi)
                & e_seg[:upto]
            )[0]
            for c in cand:
                if (dseg[c : c + sustain + 1] > th * 0.5).all():
                    o = int(c)
                    break
            if o is None and cand.size:
                o = int(cand[0])

        elif method == "kneeBacktrack":
            dmax = dseg.max()
            h2 = (n_w + 1) // 2 - 1  # 0-based first index of the second half
            dmax_a = dseg[h2:].max()
            pos = np.arange(n_w)
            cand = np.nonzero((dseg >= 0.7 * dmax_a) & e_seg & (pos >= h2))[0]
            if cand.size:
                im = int(cand[-1])
            else:
                cand = np.nonzero(e_seg)[0]
                if cand.size:
                    im = int(cand[-1])
                else:
                    cand = np.nonzero(dseg >= 0.7 * dmax_a)[0]
                    im = int(cand[-1]) if cand.size else int(np.argmax(dseg))
            sus_len = max(1, _mround(dip_dur * fs))
            sus_dip = (
                _movsum_trailing((dseg < dip_frac * dmax).astype(float), sus_len)
                >= sus_len
            )
            o = im
            while o > 0 and not sus_dip[o - 1]:
                o -= 1
            if seg[o] < resp[tr] + 0.1 * rng_:
                # clean sweep: last upward crossing of the midpoint
                mid = resp[tr] + 0.5 * rng_
                cr = np.nonzero((seg[:-1] < mid) & (seg[1:] >= mid))[0]
                if cr.size:
                    o = int(cr[-1]) + 1
            else:
                if seg[o] > resp[tr] + 0.35 * rng_:
                    sus_len2 = max(1, _mround(0.15 * fs))
                    sus_dip2 = (
                        _movsum_trailing((dseg < 0.05 * dmax).astype(float), sus_len2)
                        >= sus_len2
                    )
                    o2 = o
                    while o2 > 0 and not sus_dip2[o2 - 1]:
                        o2 -= 1
                    if o2 > 0:
                        o = o2
                r3seg = r3v[tr : pk + 1]
                if not r3seg[min(o, n_w - 1)] and r3seg.any():
                    cnd = np.nonzero(r3seg)[0]
                    o = int(cnd[np.argmin(np.abs(cnd - o))])

        elif method == "changepoint":
            step = max(1, _mround(fs / 100))
            xx = np.arange(1, n_w + 1, dtype=float)
            best = np.inf
            for c in range(_mround(0.05 * n_w), _mround(0.95 * n_w) + 1, step):
                if c < 2 or c > n_w - 1:
                    continue
                pl = np.polyfit(xx[:c], seg[:c], 1)
                pr = np.polyfit(xx[c - 1 :], seg[c - 1 :], 1)
                if pr[0] <= pl[0] + 1e-9:
                    continue
                r = np.sum((seg[:c] - np.polyval(pl, xx[:c])) ** 2) + np.sum(
                    (seg[c - 1 :] - np.polyval(pr, xx[c - 1 :])) ** 2
                )
                if r < best:
                    best = r
                    o = c - 1
        else:
            raise ValueError(f"unknown method {method!r}")

        # ineligible landing -> nearest eligible sample (either direction)
        if o is not None and not e_seg[min(o, n_w - 1)]:
            cand = np.nonzero(e_seg)[0]
            o = int(cand[np.argmin(np.abs(cand - o))]) if cand.size else None
        # HARD FLOOR: never in the first 20% of the trough->peak interval;
        # forward-only eligibility snap from the floor
        if o is not None:
            floor_idx = int(np.ceil(0.20 * (n_w - 1)))
            if o < floor_idx:
                cand = np.nonzero(e_seg[floor_idx:])[0]
                o = floor_idx + int(cand[0]) if cand.size else floor_idx
        if o is not None:
            onsets.append(tr + o)

    return np.asarray(sorted(onsets), dtype=int), peaks, troughs


def pair_breath_landmarks(
    onsets: ArrayLike, peaks: ArrayLike, troughs: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-inhale landmark pairing (breathmetrics convention): peak(k) = the
    first peak after onset k; trough(k) = the first trough after that peak;
    trailing inhales with no following trough are dropped. Returns equal-
    length ``(onsets, peaks, troughs)`` suitable for injection into a
    Breathe object."""
    onsets = np.asarray(onsets, dtype=int)
    peaks = np.asarray(peaks, dtype=int)
    troughs = np.asarray(troughs, dtype=int)
    pk_a = np.full(onsets.size, -1, dtype=int)
    tr_a = np.full(onsets.size, -1, dtype=int)
    for k in range(onsets.size):
        p = peaks[peaks > onsets[k]]
        if p.size == 0:
            continue
        pk_a[k] = p[0]
        t = troughs[troughs > p[0]]
        if t.size:
            tr_a[k] = t[0]
    keep = pk_a >= 0
    on, pk_a, tr_a = onsets[keep], pk_a[keep], tr_a[keep]
    while on.size and tr_a[-1] < 0:
        on, pk_a, tr_a = on[:-1], pk_a[:-1], tr_a[:-1]
    return on, pk_a, tr_a
