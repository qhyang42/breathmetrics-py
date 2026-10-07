# BreathMetrics (v0.1.0)

BreathMetrics is a Python toolbox for **respiratory signal analysis**, providing automated feature estimation, visualization, manual inspection/editing via GUI, and a lightweight command-line interface.

This package is a Python reimplementation and extension of the original MATLAB BreathMetrics toolbox.

---

## 🚧 Project Status

**Version:** v0.1.0  
**Status:** Early release / API stabilizing

- Core functionality is implemented and usable
- APIs may change between minor versions
- Intended for research and exploratory analysis

---

## ✨ Features

### Core analysis
- Automated estimation of primary breathing features and summary statistics of all respiratory cycles in the recording
- Support for human respiratory signals (Test performed on human airflow data only)

### Visualization
- Breath-aligned composition plots
- Feature-level summary plots

### GUI inspection & editing
- Interactive PyQt-based GUI for:
  - Visualizing detected events
  - Manual correction of onsets/pauses
  - Inspecting individual breaths

### Command-line interface
- Estimate all features from breathing signal in csv format. 
- Inspect and summarize saved BreathMetrics objects

### Export
- Export all estimated features to CSV for downstream analysis

---

## 📦 Installation

### From source (recommended for now)

```bash
git clone https://github.com/qhyang42/breathmetrics-py.git
cd breathmetrics-py

conda env create -f environment.yml
conda activate breathmetrics-py

pip install -e . 
```

## 🚀 Jupyter notebook quickstart:
```python
import breathmetrics
import numpy as np
import pandas as pd

# load data as ndarray
data = pd.read_csv("respiration.csv")
resp = data["resp"].values
fs = 1000

# initialize object
bmobj = breathmetrics.Breathe(data, fs, "humanAirflow")

# estimate features
bmobj.estimate_all_features(verbose=True, compute_resp_stats=True)

# visualize
bmobj.plot_features()
bmobj.plot_compositions_raw()
bmobj.plot_compositions_normalized()
bmobj.plot_compositions_line()
bmobj.plot_compositions_hist()

# interactive GUI for breath inspection and editting 
bmobj.inspect()
# or if you like a bit more flourish 👀, try 
bmobj.behold()

# export estimated features to CSV
bmobj.export_features("features.csv")
```
## 🤖 Command line quickstart:
This works for human airflow data only (support for other datatypes coming soon).
```bash

breathmetrics estimate examples/human_airflow.csv --fs 1000 --datatype humanAirflow --out results/

breathmetrics info results/bm.pkl

breathmetrics inspect results/bm.pkl

```


## ZLP segmentation for difficult recordings

For amplitude-non-stationary recordings (drifting baselines, breathing depth
that changes several-fold between task blocks, paced or very slow breathing)
where the stock extrema and onset detection struggle, use the ZLP
segmentation front end, ported from the Zelano Lab preprocessing pipeline
with exact landmark parity to its MATLAB counterpart:

```python
bmobj.estimate_all_features_zlp()   # instead of estimate_all_features()
```

Segmentation (validity-ruled extrema + a slope-walk "kneeBacktrack" inhale
onset) runs on an internally conditioned copy of the signal; every
downstream feature is computed by the usual pipeline in the signal's native
units. See [docs/zlp_segmentation.md](docs/zlp_segmentation.md).
