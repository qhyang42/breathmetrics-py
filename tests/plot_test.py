## plotting test for breathmetrics
# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

# %% normalized
# ----------------------------
# Dummy example data
# ----------------------------
MATRIX_SIZE = 1000
nBreaths = 12
# create fake "phase-coded" breath matrix (values 0–3)
rng = np.random.default_rng(42)
breathMatrix = rng.integers(0, 4, size=(nBreaths, MATRIX_SIZE))

# MATLAB-like custom colors (4x3 RGB)
custom_colors = np.array(
    [
        [0.0000, 0.4470, 0.7410],  # blue
        [0.8500, 0.3250, 0.0980],  # orange
        [0.9290, 0.6940, 0.1250],  # yellow
        [0.4940, 0.1840, 0.5560],  # purple
    ]
)

breath_phase_labels = ["Inhales", "Inhale Pauses", "Exhales", "Exhale Pauses"]

# ----------------------------
# Plot
# ----------------------------
fig, ax = plt.subplots(figsize=(10, 5))
cmap = ListedColormap(custom_colors)

# image() equivalent
im = ax.imshow(
    breathMatrix, aspect="auto", interpolation="nearest", cmap=cmap, origin="upper"
)

# x-axis percentage ticks
ax.set_xlim(0.5, MATRIX_SIZE + 0.5)
x_ticks = np.linspace(1, MATRIX_SIZE, 11)
x_tick_labels = [f"{int(round(x / MATRIX_SIZE * 100))}%" for x in x_ticks]
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_tick_labels)

# y-axis
ax.set_ylim(0.5, nBreaths + 0.5)
ax.set_xlabel("Proportion of Breathing Period")
ax.set_ylabel("Breath Number")

# legend with dummy lines
handles = [Line2D([], [], color=custom_colors[i], linewidth=2) for i in range(4)]
ax.legend(handles, breath_phase_labels, loc="upper right", frameon=False)

ax.set_title("Breath Composition (Normalized) — Dummy Example")
plt.tight_layout()
plt.show()
