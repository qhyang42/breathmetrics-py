## test all core fucntions in kernel.py
# %% import
import numpy as np
import pandas as pd

from breathmetrics.kernel_primary import (
    correct_respiration_to_baseline,
    find_respiratory_extrema,
    find_respiratory_volume,
    find_respiratory_durations,
    find_respiratory_offsets,
)
from breathmetrics.kernel_onset_detection_methods import (
    find_onsets_and_pauses_legacy,
    find_onsets_new,
    find_pause_slope,
    find_pause_slope_vectorized,
)

import matplotlib.pyplot as plt

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


# %% test find onsets
inhaleonset, exhaleonset, inhalepause_onsets, exhalepause_onsets = find_onsets_and_pauses_legacy(resp_corrected, corrected_peaks, corrected_troughs)  # type: ignore

# %% plot
# plt.plot(resp_corrected[0:20000])
# plt.plot(
#     inhaleonset[inhaleonset < 20000],
#     resp_corrected[inhaleonset[inhaleonset < 20000]],
#     "ro",
# )

## this is good. everything works.

# %% finding offsts
inhale_offset, exhale_offset = find_respiratory_offsets(
    resp_corrected, inhaleonset, exhaleonset, inhalepause_onsets, exhalepause_onsets
)

# %% test duration
result = find_respiratory_durations(
    inhaleonset,
    inhale_offset,
    exhaleonset,
    exhale_offset,
    inhalepause_onsets,
    exhalepause_onsets,
    fs,
)

# %% test volume
inhale_volume, exhale_volume = find_respiratory_volume(
    resp_corrected, inhaleonset, inhale_offset, exhaleonset, exhale_offset, fs
)

# %% test new onset detection.
inhaleonset_new = find_onsets_new(resp_corrected, fs, corrected_peaks)

# %% plot
# plt.plot(resp_corrected[0:20000])
# plt.plot(inhaleonset_new[0:4], resp_corrected[inhaleonset_new[0:4].astype(int)], "ro")

# plt.plot(inhaleonset[0:4], resp_corrected[inhaleonset[0:4].astype(int)], "bo")

# %%
# test pause slope
pause_new = find_pause_slope(resp_corrected, fs, inhaleonset_new, corrected_troughs)

# %% test vectorized pause slope
pause_new_vec = find_pause_slope_vectorized(
    resp_corrected, fs, inhaleonset_new, corrected_troughs
)

# %%
plt.plot(resp_corrected[0:20000])
plt.plot(inhaleonset_new[0:4], resp_corrected[inhaleonset_new[0:4].astype(int)], "ro")

plt.plot(pause_new_vec[0:4], resp_corrected[pause_new_vec[0:4].astype(int)], "bo")
# %%
