## plotting test for breathmetrics
# %%
import numpy as np
import pandas as pd
import breathmetrics
from breathmetrics.kernel_plot import (
    plot_respiratory_features,
    plot_breathing_compositions,
)

# %% read data
data = pd.read_csv("../data/resp_1.csv")
resp = data["resp"].values
fs = 1000
time = np.arange(resp.size) / fs  # type: ignore

# %%
bmobj = breathmetrics.Breathe(data, fs, "humanAirflow")
bmobj.estimate_all_features(verbose=True, compute_secondary=False)

# %%
plot_respiratory_features(bmobj)
# %%
# plottye: 'raw', 'normalized', 'line', 'hist'
plot_breathing_compositions(bmobj, plottype="raw")
plot_breathing_compositions(bmobj, plottype="normalized")
plot_breathing_compositions(bmobj, plottype="line")
plot_breathing_compositions(bmobj, plottype="hist")

# %%
