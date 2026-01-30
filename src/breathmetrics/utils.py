import numpy as np
from scipy.ndimage import gaussian_filter1d
from numpy.typing import ArrayLike
import pandas as pd

## utilities for breathmetrics

MISSING_EVENT = -1


def normalize_event_array(arr: ArrayLike) -> np.ndarray:
    a = np.asarray(arr)
    if a.size == 0:
        return a.astype(int)
    if a.dtype.kind == "O":
        cleaned = []
        for v in a:
            if v is None:
                cleaned.append(float(MISSING_EVENT))
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                cleaned.append(float(MISSING_EVENT))
                continue
            if not np.isfinite(fv):
                fv = float(MISSING_EVENT)
            cleaned.append(fv)
        a = np.asarray(cleaned, dtype=float)
    if np.issubdtype(a.dtype, np.floating):
        a = np.nan_to_num(
            a, nan=MISSING_EVENT, posinf=MISSING_EVENT, neginf=MISSING_EVENT
        )
    a = a.astype(int, copy=False)
    a[a < 0] = MISSING_EVENT
    return a


def event_is_valid(x: int | float) -> bool:
    return bool(np.isfinite(x)) and x >= 0


# check inputs
def check_input(
    resp: ArrayLike, srate: float, datatype: str
) -> tuple[np.ndarray, float, str]:
    """
    check input for breathmetrics and return error message
    """
    resp = np.asarray(resp, dtype=float)
    if resp.ndim != 1:
        resp = resp.ravel()
    if resp.size < 10:
        raise ValueError("Respiration signal must be a 1D vector of length > 10.")
    if not (20 <= srate <= 5000):
        raise ValueError("Sampling rate must be between 20 Hz and 5000 Hz.")
    supported = {"humanAirflow", "humanBB", "rodentAirflow", "rodentThermocouple"}
    if datatype not in supported:
        raise ValueError(f"Unsupported data_type: {datatype}")
    return resp, srate, datatype


def detrend_linear(x: np.ndarray) -> np.ndarray:
    t = np.arange(x.size)
    A = np.vstack([t, np.ones_like(t)]).T
    m, b = np.linalg.lstsq(A, x, rcond=None)[0]
    return x - (m * t + b)


def fft_smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.astype(float)
    k = np.ones(win, float) / win
    n = x.size + k.size - 1
    nfft = 1 << (n - 1).bit_length()
    X = np.fft.rfft(x, nfft)
    K = np.fft.rfft(k, nfft)
    y = np.fft.irfft(X * K, nfft)[: x.size + k.size - 1]
    start = (k.size - 1) // 2
    return y[start : start + x.size]


def get_valid_breath_indices(is_valid, n_inhales: int, n_exhales: int):
    """Return arrays of valid inhale/exhale indices based on is_valid flags."""
    if is_valid is None or len(is_valid) == 0:
        return np.arange(n_inhales), np.arange(n_exhales)

    valid_mask = np.asarray(is_valid, dtype=bool)
    n = min(valid_mask.shape[0], n_inhales, n_exhales)
    if n == 0:
        return np.arange(n_inhales), np.arange(n_exhales)

    valid_idx = np.flatnonzero(valid_mask[:n])
    if n_inhales > n:
        valid_inhales = np.concatenate([valid_idx, np.arange(n, n_inhales)])
    else:
        valid_inhales = valid_idx
    if n_exhales > n:
        valid_exhales = np.concatenate([valid_idx, np.arange(n, n_exhales)])
    else:
        valid_exhales = valid_idx

    return valid_inhales, valid_exhales


def zscore(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    x = x.astype(float)
    return (x - x.mean()) / (x.std() + eps)


def features_to_dataframe(bm, *, include_unsupported: bool = False) -> pd.DataFrame:
    from breathmetrics.core import FEATURE_SPECS

    rows: dict[str, np.ndarray] = {}
    scalars: dict[str, float] = {}

    for spec in FEATURE_SPECS:
        name = spec.name
        if not include_unsupported and not bm.supports(name):
            continue
        if getattr(bm, "statuses", {}).get(name) not in {"computed", "edited"}:
            continue
        value = getattr(bm, name, None)
        if value is None:
            continue

        if spec.kind in {"event_index", "per_breath"}:
            arr = np.asarray(value)
            if arr.ndim != 1:
                arr = arr.ravel()
            rows[name] = arr
        elif spec.kind == "scalar":
            if isinstance(value, dict):
                for k, v in value.items():
                    scalars[f"{name}__{k}"] = float(v)
            else:
                scalars[name] = float(value)

    max_len = max((arr.size for arr in rows.values()), default=0)
    if max_len == 0:
        if not scalars:
            raise ValueError("No computed features available for export.")
        return pd.DataFrame([scalars])

    data: dict[str, np.ndarray] = {}
    data["breath_index"] = np.arange(max_len, dtype=int)

    for spec in FEATURE_SPECS:
        name = spec.name
        if name not in rows:
            continue
        arr = rows[name]
        if arr.size == max_len:
            data[name] = arr
            continue
        fill_val = MISSING_EVENT if spec.kind == "event_index" else np.nan
        padded_dtype = int if spec.kind == "event_index" else float
        padded = np.full(max_len, fill_val, dtype=padded_dtype)
        padded[: arr.size] = arr
        data[name] = padded

    for k, v in scalars.items():
        data[k] = np.full(max_len, v, dtype=float)

    return pd.DataFrame(data)


def hist_like_matlab(this_window: np.ndarray, custom_bins: np.ndarray):
    # make sure inputs are arrays
    this_window = np.asarray(this_window)
    custom_bins = np.asarray(custom_bins)

    # infer edges from bin centers
    bin_edges = np.concatenate(
        [
            [custom_bins[0] - (custom_bins[1] - custom_bins[0]) / 2],
            (custom_bins[:-1] + custom_bins[1:]) / 2,
            [custom_bins[-1] + (custom_bins[-1] - custom_bins[-2]) / 2],
        ]
    )

    amplitude_values, _ = np.histogram(this_window, bins=bin_edges)

    window_bins = custom_bins  # same as MATLAB
    return amplitude_values, window_bins


# small helpers used in updated onset detection TODO: need tests
def find_inflection(sig, slope_based=True) -> tuple[int, np.ndarray]:
    """
    Find the nearest major inflection point from the end of a signal.

    Parameters
    ----------
    sig : np.ndarray
        1D time-series array.
    slope_based : bool, optional
        If True, use slope-based method.
        If False, use sum-squared-deviation derivative method.

    Returns
    -------
    inflection : int
        Index (1-based, MATLAB-style) counting from the end of the signal.
    ss_der : np.ndarray
        Derivative of sum-squared deviation (same length as sig).
    """

    sig = np.asarray(sig).flatten()
    n = len(sig)

    SS = np.zeros(n)
    slopes = np.zeros(n)

    # Loop mirrors MATLAB: for i = 1:length(sig)-1
    for i in range(n - 1):
        x = np.arange(i, n) + 1  # MATLAB-style x indices
        y = sig[i:]

        # Linear fit
        p = np.polyfit(x, y, 1)
        yfit = np.polyval(p, x)

        SS[i] = np.sum((y - yfit) ** 2)
        slopes[i] = p[0]

    # Derivative of SS with padding
    ss_der = np.diff(SS)
    ss_der = np.concatenate(([ss_der[0]], ss_der))

    # Smooth slopes (MATLAB smoothdata gaussian)
    win = max(1, round(n / 20))
    sigma = win / 6  # MATLAB gaussian window ≈ 6σ
    slopes = gaussian_filter1d(slopes, sigma=sigma, mode="nearest")

    if slope_based:
        slopes_flipped = slopes[::-1]
        inflection = int(np.argmax(slopes_flipped)) + 1
    else:
        ss_der_flipped = ss_der[::-1]
        inflection = int(np.argmax(ss_der_flipped)) + 1

    return inflection, ss_der


def find_inflection_vectorized(
    sig: np.ndarray, slope_based: bool = True
) -> tuple[int, np.ndarray]:
    """
    Vectorized O(n) version of MATLAB findInflection (no plotting).

    Parameters
    ----------
    sig : np.ndarray
        1D time-series array.
    slope_based : bool
        If True: choose inflection by max(smoothed slopes flipped).
        If False: choose inflection by max(SS derivative flipped).

    Returns
    -------
    inflection : int
        1-based index counting from the end (MATLAB-style), same as original.
    ss_der : np.ndarray
        Derivative of SSE curve SS (padded to same length as sig).
    """
    sig = np.asarray(sig).reshape(-1)
    n = sig.size
    if n < 2:
        # Mirror “nothing to fit” behavior as sensibly as possible
        return 1, np.zeros(n, dtype=float)

    # MATLAB uses x = (i:length(sig))' where i is 1-based index into sig
    x = np.arange(1, n + 1, dtype=float)

    # Suffix sums for y, y^2, and x*y
    y = sig.astype(float)
    y2 = y * y
    xy = x * y

    suf_y = np.cumsum(y[::-1])[::-1]
    suf_y2 = np.cumsum(y2[::-1])[::-1]
    suf_xy = np.cumsum(xy[::-1])[::-1]

    # For each start i (1..n), segment is i..n with length m = n-i+1
    i = np.arange(1, n + 1, dtype=float)
    m = n - i + 1.0

    # Analytic sums for consecutive integers:
    # Sx(i..n) = sum_{t=i}^n t
    # Sxx(i..n) = sum_{t=i}^n t^2
    def sum_1_to(k):
        return k * (k + 1.0) / 2.0

    def sumsq_1_to(k):
        return k * (k + 1.0) * (2.0 * k + 1.0) / 6.0

    Sx = sum_1_to(n) - sum_1_to(i - 1.0)
    Sxx = sumsq_1_to(n) - sumsq_1_to(i - 1.0)

    Sy = suf_y
    Syy = suf_y2
    Sxy = suf_xy

    # Linear regression slope/intercept for each suffix segment:
    # b1 = (m*Sxy - Sx*Sy) / (m*Sxx - Sx^2)
    denom = m * Sxx - Sx * Sx

    # Avoid divide-by-zero in pathological cases (shouldn’t happen for m>=2, but be safe)
    safe = np.abs(denom) > np.finfo(float).eps
    b1 = np.zeros(n, dtype=float)
    b0 = np.zeros(n, dtype=float)

    b1[safe] = (m[safe] * Sxy[safe] - Sx[safe] * Sy[safe]) / denom[safe]
    b0[safe] = (Sy[safe] - b1[safe] * Sx[safe]) / m[safe]

    # SSE for each suffix segment without explicitly computing yfit:
    # SSE = Σ(y - (b1*x + b0))^2
    #     = Syy - 2*b1*Sxy - 2*b0*Sy + b1^2*Sxx + 2*b1*b0*Sx + m*b0^2
    SS = (
        Syy
        - 2.0 * b1 * Sxy
        - 2.0 * b0 * Sy
        + (b1 * b1) * Sxx
        + 2.0 * b1 * b0 * Sx
        + m * (b0 * b0)
    )

    # Match MATLAB loop: for i = 1:length(sig)-1
    # Last entry wasn't computed there; SS(n) and slopes(n) stay 0.
    SS[-1] = 0.0
    slopes = b1.copy()
    slopes[-1] = 0.0

    # SSDer = diff(SS); SSDer = [SSDer(1); SSDer]  (pad)
    ss_der = np.diff(SS)
    ss_der = np.concatenate(([ss_der[0]], ss_der))

    # slopes = smoothdata(slopes, 'gaussian', round(length(sig)/20));
    win = max(1, int(round(n / 20)))
    sigma = win / 6.0  # gaussian window ~ 6*sigma
    slopes_smooth = gaussian_filter1d(slopes, sigma=sigma, mode="nearest")

    if slope_based:
        # slopes = flipud(slopes); [~, maxidx] = max(slopes);
        inflection = int(np.argmax(slopes_smooth[::-1])) + 1  # 1-based
    else:
        inflection = int(np.argmax(ss_der[::-1])) + 1  # 1-based

    return inflection, ss_der


def find_inflection_from_mid3(
    sig: np.ndarray,
    mid_idx: int,
    fs: float = 1000,
    sig_raw: np.ndarray | None = None,
):
    """
    Refine an inflection point estimate starting from a candidate index
    measured from the end of the signal.

    Parameters
    ----------
    sig : np.ndarray
        1D processed breathing signal.
    mid_idx : int
        Candidate inflection index counted from the END (MATLAB-style, 1-based).
    fs : int, optional
        Sampling rate in Hz. Default is 1000.
    sig_raw : np.ndarray, optional
        Raw signal segment (kept for API parity; unused here since plotting
        is removed).

    Returns
    -------
    inflection : int
        Inflection index in the ORIGINAL signal (0-based Python index).
    """

    sig = np.asarray(sig).reshape(-1)
    N = sig.size

    # MATLAB:
    # starti = N - midIDX + 1 - round(fs/5)
    starti = N - mid_idx + 1 - int(round(fs / 5))
    endi = N - 5

    # Bounds checking
    starti = max(starti, 1)
    endi = min(endi, N)

    # Convert to Python slicing (0-based, end-exclusive)
    start_py = starti - 1
    end_py = endi

    # Call upward inflection finder on the subsegment
    # Assumed to return a Python index relative to sig[start_py:end_py]
    inflection_local = find_inflection_upward(sig[start_py:end_py])

    # Adjust to original signal coordinates
    inflection = inflection_local + start_py

    return inflection


def find_inflection_upward(sig: np.ndarray, sig_raw: np.ndarray | None = None) -> int:
    """
    Find an upward inflection point using slope-based linear fits,
    enforcing local monotonicity via mono_check.

    Parameters
    ----------
    sig : np.ndarray
        1D processed signal (windowed).
    sig_raw : np.ndarray, optional
        Raw signal for monotonicity checking. Defaults to sig.

    Returns
    -------
    inflection : int
        Inflection index in sig (0-based Python index).
    """

    sig = np.asarray(sig).reshape(-1)
    if sig_raw is None:
        sig_raw = sig
    else:
        sig_raw = np.asarray(sig_raw).reshape(-1)

    N = sig.size

    SS = np.zeros(N)
    slopes = np.zeros(N)

    # MATLAB: for i = 1:length(sig)-1
    for i in range(N - 1):
        x = np.arange(i, N) + 1  # MATLAB-style x
        y = sig[i:]

        p = np.polyfit(x, y, 1)
        yfit = np.polyval(p, x)

        SS[i] = np.sum((y - yfit) ** 2)
        slopes[i] = p[0]

    # Smooth slopes: smoothdata(...,'gaussian', round(N/20))
    win = max(1, int(round(N / 20)))
    sigma = win / 6.0
    slopes = gaussian_filter1d(slopes, sigma=sigma, mode="nearest")

    # Flip slopes (distance from end)
    slopes_flipped = slopes[::-1]

    # Sort candidates by descending slope
    idx_sort = np.argsort(slopes_flipped)[::-1]

    found = False
    inflection: int

    for cand in idx_sort:
        # cand is index in flipped space → convert to index in sig
        # MATLAB: peakidx = N - cand + 1 (1-based)
        peakidx = N - cand - 1  # 0-based Python

        try:
            if mono_check(sig_raw, peakidx):
                inflection = peakidx
                found = True
                break
        except Exception:
            # match MATLAB try/catch: silently skip bad edge cases
            continue

    if not found:
        # fallback to maximum slope
        peakidx = N - idx_sort[0] - 1
        inflection = peakidx

    return inflection  # type: ignore


def mono_check(vec: np.ndarray, peakidx: int) -> bool:
    """
    Check whether the signal is locally monotonic (increasing)
    around a candidate inflection point.

    Parameters
    ----------
    vec : np.ndarray
        1D signal (usually raw).
    peakidx : int
        Index of candidate inflection (0-based).

    Returns
    -------
    is_mono : bool
        True if monotonicity criterion is satisfied.
    """

    vec = np.asarray(vec).reshape(-1)
    N = vec.size

    # Guard (MATLAB: peakidx ± 5 must be valid)
    if peakidx - 5 < 0 or peakidx + 5 >= N:
        return False

    pre_val = np.min(vec[peakidx - 5 : peakidx])
    post_val = np.max(vec[peakidx + 1 : peakidx + 6])

    return post_val > pre_val
