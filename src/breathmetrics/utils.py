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


## mean centering
## FFT smoothing
