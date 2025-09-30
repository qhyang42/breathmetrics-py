## test all core fucntions in kernel.py
# %% import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from breathmetrics.kernel import (
    correct_respiration_to_baseline,
    find_respiratory_extrema,
)
from breathmetrics.kernel_onset_detection_methods import find_onsets_and_pauses_legacy

# %% load data
data = pd.read_csv("../data/resp_1.csv")
resp = data["resp"].values
fs = 1000
time = np.arange(resp.size) / fs  # type: ignore

# %% test baseline correction
resp_corrected = correct_respiration_to_baseline(resp, fs)  # type: ignore
# plt.plot(resp_corrected)
# plt.plot(resp)  # type: ignore

# %% test find extrema
[corrected_peaks, corrected_troughs] = find_respiratory_extrema(resp_corrected, fs)  # type: ignore

# plt.plot(resp_corrected)
# plt.plot(corrected_peaks, resp_corrected[corrected_peaks], "ro")
# plt.plot(corrected_troughs, resp_corrected[corrected_troughs], "go")


# %% test find offsets
inhaleonset, exhaleonset, *_ = find_onsets_and_pauses_legacy(resp_corrected, corrected_peaks, corrected_troughs)  # type: ignore

# %% plot
plt.plot(resp_corrected[0:20000])
plt.plot(
    inhaleonset[inhaleonset < 20000],
    resp_corrected[inhaleonset[inhaleonset < 20000]],
    "ro",
)


# %%
