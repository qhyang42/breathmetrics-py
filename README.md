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
- Automated estimation of primary and secondary breathing features
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
cd breathmetrics
pip install -e . 
```

## 🚀 Example usage in jupyter notebook. 
```python

import breathmetrics
import numpy as np
import pandas as pd

# load data
data = pd.read_csv("respiration.csv")
resp = data["resp"].values
fs = 1000

# initialize object
bmobj = breathmetrics.Breathe(data, fs, "humanAirflow")

# estimate features
bmobj.estimate_all_features(verbose=True, compute_secondary=True)

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

# export features to CSV
bm.export_features("features.csv")
```
## 🤖 Command line quickstart
Airflow data only.
```bash
pip install breathmetrics

breathmetrics estimate examples/human_airflow.csv --fs 1000 --datatype humanAirflow --out results/

breathmetrics inspect results/bm.pkl

breathmetrics info results/bm.pkl
```


