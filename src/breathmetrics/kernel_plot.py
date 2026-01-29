# plotting for breathmetrics
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

# import seaborn as sns
from matplotlib.colors import ListedColormap

# from matplotlib.lines import Line2D  # ✅ import this explicitly
from matplotlib.patches import Patch
from typing import Literal


def plot_respiratory_features(
    bm,
    annotate=None,
    size_data=36,
    backend="qt",
):
    """
    Plot respiration and estimated respiratory features in an interactive
    matplotlib window (MATLAB-like).

    Parameters
    ----------
    bm : BreathMetrics-like object
        Must expose:
        - bm.time
        - bm.which_resp()
        - feature index arrays (inhale_peaks, exhale_troughs, etc.)
        - bm.datatype
    annotate : list[str] or None
        Any of: 'extrema', 'onsets', 'maxflow', 'volumes', 'pauses'
    size_data : int
        Marker size for scatter points.
    backend : {'qt', 'widget'}
        'qt' -> launches OS-level interactive window (MATLAB-like)
        'widget' -> notebook-embedded interactive plot
    """

    # --- force interactive backend BEFORE pyplot import ---
    import matplotlib

    if backend == "qt":
        matplotlib.use("QtAgg", force=True)
    elif backend == "widget":
        matplotlib.use("module://ipympl.backend_nbagg", force=True)
    else:
        raise ValueError("backend must be 'qt' or 'widget'")

    import matplotlib.pyplot as plt
    import numpy as np

    if annotate is None:
        annotate = []

    # -----------------------------------------------------
    # collect parameters to plot
    params_to_plot = []
    param_labels = []

    def _add(param, label, airflow_only=False):
        if param is None or len(param) == 0:
            return
        if airflow_only and bm.datatype not in ("humanAirflow", "rodentAirflow"):
            return
        params_to_plot.append(np.asarray(param, dtype=int))
        param_labels.append(label)

    _add(bm.inhale_peaks, "Inhale Peaks", airflow_only=True)
    _add(bm.exhale_troughs, "Exhale Troughs", airflow_only=True)
    _add(bm.inhale_onsets, "Inhale Onsets")
    _add(bm.exhale_onsets, "Exhale Onsets")

    if hasattr(bm, "inhale_pause_onsets"):
        _add(bm.inhale_pause_onsets[bm.inhale_pause_onsets > 0], "Inhale Pauses")
    if hasattr(bm, "exhale_pause_onsets"):
        _add(bm.exhale_pause_onsets[bm.exhale_pause_onsets > 0], "Exhale Pauses")

    # -----------------------------------------------------
    # main plot
    x = bm.time
    y = bm.baseline_corrected_respiration

    fig, ax = plt.subplots(figsize=(12, 4))
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title("Estimated Respiratory Features")

    (line,) = ax.plot(x, y, color="black", lw=1)
    handles = [line]
    labels = ["Respiration"]

    ax.set_xlabel("Time (seconds)")

    if bm.datatype in ("humanAirflow", "rodentAirflow"):
        ax.set_ylabel("Airflow (AU)")
    elif bm.datatype == "humanBB":
        ax.set_ylabel("Breathing Belt (AU)")
    elif bm.datatype == "rodentThermocouple":
        ax.set_ylabel("Thermocouple (AU)")

    # -----------------------------------------------------
    # scatter features
    if params_to_plot:
        colors = [
            "#4C72B0",  # Inhale Peaks   (muted blue)
            "#DD8452",  # Exhale Troughs (muted orange)
            "#55A868",  # Inhale Onsets  (muted green)
            "#C44E52",  # Exhale Onsets  (muted red)
            "#8172B3",  # Inhale Pauses  (muted purple)
            "#937860",  # Exhale Pauses  (muted brown)
        ]  # TODO this is distinct enough at lease. change if think of anything better

        for param, label, color in zip(params_to_plot, param_labels, colors):
            valid = (param > 0) & (param < len(x))
            idx = param[valid]

            sc = ax.scatter(
                x[idx],
                y[idx],
                s=size_data,
                color=color,
                edgecolor=color,
                zorder=3,
                label=label,
            )
            handles.append(sc)  # type: ignore[arg-type]
            labels.append(label)

    ax.legend(handles, labels, loc="best")

    # -----------------------------------------------------
    # annotations
    def _annotate(indices, strings):
        for i, txt in zip(indices, strings):
            ax.text(x[i], y[i], txt, fontsize=9)

    if "extrema" in annotate:
        if bm.inhale_peaks is not None:
            _annotate(
                bm.inhale_peaks,
                [f"Peak @ {x[i]:.1f}" for i in bm.inhale_peaks],
            )
        if bm.exhale_troughs is not None:
            _annotate(
                bm.exhale_troughs,
                [f"Trough @ {x[i]:.1f}" for i in bm.exhale_troughs],
            )

    if "onsets" in annotate:
        _annotate(
            bm.inhale_onsets,
            [f"Inhale @ {x[i]:.1f}" for i in bm.inhale_onsets],
        )
        _annotate(
            bm.exhale_onsets,
            [f"Exhale @ {x[i]:.1f}" for i in bm.exhale_onsets],
        )

    if "pauses" in annotate:
        if hasattr(bm, "inhale_pause_onsets"):
            real = bm.inhale_pause_onsets[bm.inhale_pause_onsets > 0]
            _annotate(
                real,
                [f"Inhale pause @ {x[i]:.1f}" for i in real],
            )
        if hasattr(bm, "exhale_pause_onsets"):
            real = bm.exhale_pause_onsets[bm.exhale_pause_onsets > 0]
            _annotate(
                real,
                [f"Exhale pause @ {x[i]:.1f}" for i in real],
            )

    plt.tight_layout()
    plt.show(block=True)

    return fig


def plot_breathing_compositions(
    bm,
    plottype: Literal["raw", "normalized", "line", "hist"],
    *,
    matrix_size: int = 1000,
):
    """
    Plot the composition of breaths in a BreathMetrics-like object.

    plottype:
      - 'raw'        : durations scaled to max breath period (pads background)
      - 'normalized' : each breath stretched to fill matrix_size columns (proportions)
      - 'line'       : per-breath phase-duration traces + mean±std (NaNs excluded)
      - 'hist'       : histograms of phase durations (NaNs excluded)

    Notes on NaNs
    -------------
    Pause durations are allowed to be np.nan (meaning: pause absent/undefined).
    - For composition images ('raw', 'normalized'): NaNs are treated as 0 for rendering.
    - For 'line' and 'hist': NaNs are treated as missing and excluded from stats/hist.
    """
    inhale_onsets = getattr(bm, "inhale_onsets", None)
    if inhale_onsets is None:
        raise ValueError(
            "bm.inhale_onsets is required. Estimate features before plotting."
        )
    inhale_onsets = np.asarray(inhale_onsets)
    if inhale_onsets.size == 0:
        raise ValueError("No breaths found (inhale_onsets is empty).")

    if plottype not in {"raw", "normalized", "line", "hist"}:
        raise ValueError("plottype must be one of {'raw','normalized','line','hist'}.")

    # Phase labels and MATLAB-ish colors
    phase_labels = ["inhale", "inhale_pause", "exhale", "exhale_pause"]
    phase_colors = np.array(
        [
            [0.0000, 0.4470, 0.7410],  # inhale (blue)
            [0.8500, 0.3250, 0.0980],  # inhale_pause (orange)
            [0.9290, 0.6940, 0.1250],  # exhale (yellow)
            [0.4940, 0.1840, 0.5560],  # exhale_pause (purple)
        ],
        dtype=float,
    )
    bg_color = np.array([[1.0, 1.0, 1.0]], dtype=float)

    def _as_float(name: str, default=None) -> np.ndarray:
        arr = getattr(bm, name, default)
        if arr is None:
            raise ValueError(f"bm.{name} is required for plottype='{plottype}'.")
        return np.asarray(arr, dtype=float)

    # Raw arrays: keep NaNs (pause missing)
    inh = _as_float("inhale_durations")
    exh = _as_float("exhale_durations")
    inhp = _as_float("inhale_pause_durations", default=np.full_like(inh, np.nan))
    exhp = _as_float("exhale_pause_durations", default=np.full_like(inh, np.nan))

    # Clamp to common length
    n_breaths = int(min(inh.size, exh.size, inhp.size, exhp.size, inhale_onsets.size))
    inh, exh, inhp, exhp = (
        inh[:n_breaths],
        exh[:n_breaths],
        inhp[:n_breaths],
        exhp[:n_breaths],
    )

    def _nan_to_zero(x: np.ndarray) -> np.ndarray:
        # for rendering compositions only
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def _allocate_by_largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
        """Convert nonnegative weights to integer counts summing to `total`."""
        w = np.clip(weights.astype(float), 0.0, np.inf)
        s = w.sum()
        if s <= 0:
            out = np.zeros_like(w, dtype=int)
            out[0] = total  # fallback: all inhale
            return out

        exact = w / s * total
        base = np.floor(exact).astype(int)
        leftover = total - int(base.sum())
        if leftover > 0:
            remainders = exact - np.floor(exact)
            add_idx = np.argsort(remainders)[::-1][:leftover]
            base[add_idx] += 1
        elif leftover < 0:
            take = -leftover
            for idx in np.argsort(base)[::-1]:
                if take == 0:
                    break
                if base[idx] > 0:
                    base[idx] -= 1
                    take -= 1
        return base

    if plottype == "normalized":
        # For compositions: treat missing pauses as 0
        inh_z, inhp_z, exh_z, exhp_z = map(_nan_to_zero, (inh, inhp, exh, exhp))

        comp = np.zeros((n_breaths, matrix_size), dtype=np.uint8)  # codes 0..3
        for b in range(n_breaths):
            counts = _allocate_by_largest_remainder(
                np.array([inh_z[b], inhp_z[b], exh_z[b], exhp_z[b]], dtype=float),
                total=matrix_size,
            )
            i_inh, i_inhp, i_exh, i_exhp = counts.tolist()

            row = np.empty(matrix_size, dtype=np.uint8)
            pos = 0
            row[pos : pos + i_inh] = 0
            pos += i_inh
            if i_inhp > 0:
                row[pos : pos + i_inhp] = 1
                pos += i_inhp
            row[pos : pos + i_exh] = 2
            pos += i_exh
            if i_exhp > 0:
                row[pos : pos + i_exhp] = 3
            comp[b, :] = row

        fig, ax = plt.subplots(figsize=(10, 5))
        cmap = ListedColormap(phase_colors, name="breath_phases_norm")
        ax.imshow(
            comp, aspect="auto", interpolation="nearest", cmap=cmap, origin="upper"
        )

        x_ticks = np.linspace(0, matrix_size, 11)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{int(round(x / matrix_size * 100))}%" for x in x_ticks])

        ax.set_xlabel("Proportion of Breathing Period")
        ax.set_ylabel("Breath Number")
        ax.set_title("Breath Composition (Normalized)")

        legend_handles = [
            Patch(facecolor=phase_colors[i], edgecolor="none") for i in range(4)
        ]
        ax.legend(legend_handles, phase_labels, loc="upper right", frameon=False)

        plt.tight_layout()
        plt.show()
        return fig, ax

    if plottype == "raw":
        # For compositions: treat missing pauses as 0
        inh_z, inhp_z, exh_z, exhp_z = map(_nan_to_zero, (inh, inhp, exh, exhp))

        srate = float(getattr(bm, "srate", 1.0))
        if inhale_onsets.size >= 2:
            max_breath_sec = float(np.ceil(np.max(np.diff(inhale_onsets)) / srate))
        else:
            max_breath_sec = 1.0

        comp = np.full((n_breaths, matrix_size), 4, dtype=np.uint8)  # 4 = background
        for b in range(n_breaths):
            raw_cols = np.array([inh_z[b], inhp_z[b], exh_z[b], exhp_z[b]], dtype=float)
            if max_breath_sec > 0:
                cols = np.round(raw_cols / max_breath_sec * matrix_size).astype(int)
            else:
                cols = np.array([matrix_size, 0, 0, 0], dtype=int)
            cols = np.clip(cols, 0, matrix_size)

            pos = 0
            k = min(cols[0], matrix_size - pos)
            comp[b, pos : pos + k] = 0
            pos += k
            k = min(cols[1], matrix_size - pos)
            comp[b, pos : pos + k] = 1
            pos += k
            k = min(cols[2], matrix_size - pos)
            comp[b, pos : pos + k] = 2
            pos += k
            k = min(cols[3], matrix_size - pos)
            comp[b, pos : pos + k] = 3

        fig, ax = plt.subplots(figsize=(10, 5))
        cmap = ListedColormap(
            np.vstack([phase_colors, bg_color]), name="breath_phases_raw"
        )
        ax.imshow(
            comp, aspect="auto", interpolation="nearest", cmap=cmap, origin="upper"
        )

        tick_step = 0.5
        sec_labels = np.arange(0.0, max_breath_sec + tick_step, tick_step)
        x_ticks = np.linspace(0, matrix_size, len(sec_labels))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{x:.1f}".rstrip("0").rstrip(".") for x in sec_labels])

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Breath Number")
        ax.set_title("Breath Composition (Raw)")

        legend_handles = [
            Patch(facecolor=phase_colors[i], edgecolor="none") for i in range(4)
        ]
        ax.legend(legend_handles, phase_labels, loc="upper right", frameon=False)

        plt.tight_layout()
        plt.show()
        return fig, ax

    if plottype == "line":
        fig, ax = plt.subplots(figsize=(9, 5))

        cmap = plt.get_cmap("viridis")
        breath_colors = cmap(np.linspace(0, 1, n_breaths))

        # Per-breath traces (NaNs = missing => skip those points)
        for b in range(n_breaths):
            xs = [1]
            ys = [float(inh[b])]

            if np.isfinite(inhp[b]) and inhp[b] > 0:
                xs.append(2)
                ys.append(float(inhp[b]))

            xs.append(3)
            ys.append(float(exh[b]))

            if np.isfinite(exhp[b]) and exhp[b] > 0:
                xs.append(4)
                ys.append(float(exhp[b]))

            ax.plot(xs, ys, color=breath_colors[b], linewidth=1.2, alpha=0.85)
            ax.scatter(
                xs,
                ys,
                marker="s",
                facecolor=breath_colors[b],
                edgecolor="none",
                s=30,
                alpha=0.9,
            )

        # Stats: exclude NaNs (treat as missing)
        def _nanmean_std(x: np.ndarray) -> tuple[float, float, float]:
            finite = x[np.isfinite(x)]
            if finite.size == 0:
                return np.nan, np.nan, np.nan
            return (
                float(np.nanmean(finite)),
                float(np.nanstd(finite)),
                float(np.nanmax(finite)),
            )

        means = []
        stds = []
        maxes = []
        for arr in (inh, inhp, exh, exhp):
            m, s, mx = _nanmean_std(arr)
            means.append(m)
            stds.append(s)
            maxes.append(mx)

        means = np.array(means, dtype=float)
        stds = np.array(stds, dtype=float)
        maxes = np.array(maxes, dtype=float)

        ax.errorbar(
            [1, 2, 3, 4],
            means,
            yerr=stds,
            color="k",
            linestyle="none",
            linewidth=2,
            capsize=3,
        )

        ax.set_xlim(0.5, 4.5)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(["Inhale", "Inhale Pause", "Exhale", "Exhale Pause"])
        ax.set_ylabel("Time (seconds)")
        ax.set_title("Breath Phase Durations (Per-Breath Lines)")

        ymax = float(np.nanmax(maxes)) if np.isfinite(np.nanmax(maxes)) else 1.0
        ax.set_ylim(0, ymax + 0.6)

        for xi in range(1, 5):
            m = means[xi - 1]
            s = stds[xi - 1]
            if np.isfinite(m):
                ax.text(xi, ymax + 0.35, f"Mean={m:.3g}s", ha="center", fontsize=9)
            if np.isfinite(s):
                ax.text(xi, ymax + 0.15, f"Std={s:.3g}s", ha="center", fontsize=9)

        plt.tight_layout()
        plt.show()
        return fig, ax

    # plottype == "hist"
    datasets = [
        (inh, "Inhale Durations (s)"),
        (inhp, "Inhale Pause Durations (s)"),
        (exh, "Exhale Durations (s)"),
        (exhp, "Exhale Pause Durations (s)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes = axes.ravel()

    for i, (data, xlabel) in enumerate(datasets):
        ax = axes[i]
        finite = data[np.isfinite(data)]
        n = int(finite.size)

        # MATLAB-ish: floor(n/5), min 10
        n_bins = int(np.floor(n / 5)) if n > 0 else 10
        if n_bins < 10:
            n_bins = 10

        counts, bins = (
            np.histogram(finite, bins=n_bins)
            if n > 0
            else (np.array([0]), np.array([0, 1]))
        )
        centers = 0.5 * (bins[:-1] + bins[1:])
        width = (bins[1] - bins[0]) if len(bins) > 1 else 1.0

        ax.bar(centers, counts, width=width, color=phase_colors[i], edgecolor="none")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(xlabel.replace(" (s)", ""))

    fig.suptitle("Breath Phase Duration Histograms (NaNs excluded)", y=1.02)
    plt.tight_layout()
    plt.show()
    return fig, axes
