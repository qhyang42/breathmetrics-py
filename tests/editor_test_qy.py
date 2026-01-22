# %% test breath_editor.py
import numpy as np
import pandas as pd
import breathmetrics
import matplotlib.pyplot as plt

from breathmetrics.breath_editor import BreathEditor, EventType

# %% create breathmetrics object
data = pd.read_csv("../data/resp_1.csv")
resp = data["resp"].values
fs = 1000
time = np.arange(resp.size) / fs  # type: ignore
bm_obj = breathmetrics.Breathe(data, fs, "humanAirflow")
bm_obj.estimate_all_features(verbose=True, compute_secondary=False)

# %% init

ed = BreathEditor(bm_obj)
print("n_breaths =", ed.n_breaths)
print("dirty?", ed.is_dirty)


# %% snapshot helper
def snap(bm, i: int) -> dict[str, float]:
    """Return key editable events + the derived features you recompute."""

    def f(x):
        x = float(x)
        return x

    return {
        "inhale_onset": f(bm.inhale_onsets[i]),
        "exhale_onset": f(bm.exhale_onsets[i]),
        "inhale_pause_onset": f(bm.inhale_pause_onsets[i]),
        "exhale_pause_onset": f(bm.exhale_pause_onsets[i]),
        "inhale_time2peak": f(bm.inhale_time2peak[i]),
        "inhale_duration": f(bm.inhale_durations[i]),
        "inhale_volume": f(bm.inhale_volumes[i]),
        "exhale_time2trough": f(bm.exhale_time2trough[i]),
        "exhale_duration": f(bm.exhale_durations[i]),
        "exhale_volume": f(bm.exhale_volumes[i]),
    }


def pretty(d: dict[str, float]) -> None:
    for k, v in d.items():
        if np.isnan(v):
            print(f"{k:18s}: NaN")
        else:
            print(f"{k:18s}: {v:.3f}")


# %% move event
i = 10
before = snap(bm_obj, i)

new_sample = int(bm_obj.inhale_onsets[i]) + 25  # try a small shift
res = ed.move_event(i, EventType.INHALE_ONSET, new_sample)

after = snap(bm_obj, i)

print(res)
print("\nBefore:")
pretty(before)
print("\nAfter:")
pretty(after)

# %% test create pause if Nan
i = 10
print("Before inhale_pause_onset:", bm_obj.inhale_pause_onsets[i])

# choose something near end of inhale as a plausible pause start
target = int(bm_obj.inhale_offsets[i])  # or +10, etc.
res = ed.move_event(i, EventType.INHALE_PAUSE_ONSET, target)

print(res)
print("After inhale_pause_onset:", bm_obj.inhale_pause_onsets[i])

# %% clamping
i = 10
bad = int(bm_obj.inhale_offsets[i]) - 200  # violates order
res = ed.move_event(i, EventType.EXHALE_ONSET, bad)

print(res)
print("exhale_onset now:", bm_obj.exhale_onsets[i])
print("inhale_end (offset):", bm_obj.inhale_offsets[i])

# %%  undo
i = 10
s0 = snap(bm_obj, i)

_ = ed.move_event(i, EventType.INHALE_ONSET, int(bm_obj.inhale_onsets[i]) + 50)
s1 = snap(bm_obj, i)

ok = ed.undo_last()
s2 = snap(bm_obj, i)

print("undo ok?", ok)
print("\nOriginal:")
pretty(s0)
print("\nAfter edit:")
pretty(s1)
print("\nAfter undo:")
pretty(s2)

# %% plot


def plot_breath(
    bm, i: int, pad_s: float = 1.0, use_resp: str = "bsl_corrected_respiration"
):
    fs = float(bm.fs)
    y = np.asarray(getattr(bm, use_resp), dtype=float)

    onset = int(bm.inhale_onsets[i])
    end = int(bm.exhale_offsets[i])

    pad = int(pad_s * fs)
    lo = max(0, onset - pad)
    hi = min(len(y) - 1, end + pad)

    t = (np.arange(lo, hi + 1) - onset) / fs
    seg = y[lo : hi + 1]

    def mark(sample, label):
        if not np.isfinite(sample):
            return
        s = int(sample)
        if s < lo or s > hi:
            return
        plt.scatter([(s - onset) / fs], [y[s]], s=60, edgecolor="black")
        plt.text((s - onset) / fs, y[s], f" {label}", va="center")

    plt.figure(figsize=(10, 3.5))
    plt.plot(t, seg, linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Time relative to inhale onset (s)")
    plt.ylabel("Amplitude")

    mark(bm.inhale_onsets[i], "inh_on")
    mark(bm.exhale_onsets[i], "exh_on")
    mark(bm.inhale_pause_onsets[i], "inh_pause")
    mark(bm.exhale_pause_onsets[i], "exh_pause")
    mark(bm.inhale_peaks[i], "peak")
    mark(bm.exhale_troughs[i], "trough")

    plt.title(f"Breath {i}")
    plt.show()


# Use it:
plot_breath(bm_obj, 10)
