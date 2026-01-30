# test file for core breathmetrics class

# %%
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import breathmetrics


# %% load data
data = pd.read_csv("../data/resp_1.csv")
resp = data["resp"].values
fs = 1000
time = np.arange(resp.size) / fs  # type: ignore


# %%
bmobj = breathmetrics.Breathe(data, fs, "humanAirflow")
# %% sanity
print("raw shape:", bmobj.raw_respiration.shape)
print("srate:", bmobj.srate)
print("datatype:", bmobj.datatype)
print(
    "has smoothed:",
    hasattr(bmobj, "smoothed_respiration"),
    bmobj.smoothed_respiration is not None,
)
# PASSED


# %%
bmobj.estimate_all_features(verbose=True, compute_secondary=True)
print("complete:", bmobj.feature_estimations_complete)

# %% save test
bmobj.export_features("../_tmp_results/features.csv")
# %% GUI launch test
bmobj.inspect()

# %% alias
bmobj.behold()

# %% trough troubleshooting
plt.plot(bmobj.smoothed_respiration[0:20000])
plt.plot(
    bmobj.exhale_onsets[0:4],
    bmobj.smoothed_respiration[bmobj.exhale_onsets[0:4].astype(int)],
    "ro",
)

# %%
