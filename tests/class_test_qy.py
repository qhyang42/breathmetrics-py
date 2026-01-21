# test file for core breathmetrics class

# %%
import numpy as np

# import matplotlib.pyplot as plt
import pandas as pd

import breathmetrics

# from breathmetrics.core import bm  # or wherever bm lives


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
bmobj.estimate_all_features(verbose=True, compute_secondary=False)
print("complete:", bmobj.feature_estimations_complete)

# %% plotting
