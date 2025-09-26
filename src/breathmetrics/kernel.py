import numpy as np


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


## find resp cycle durations
## find extrema
## find resp offsets
## find pauses and onsets
## find resp volume
## calculate secondary features
## create resp ERPs
