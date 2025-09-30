import numpy as np

## utilities for breathmetrics


# utils.py
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


def zscore(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    x = x.astype(float)
    return (x - x.mean()) / (x.std() + eps)


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


## mean centering
## FFT smoothing
