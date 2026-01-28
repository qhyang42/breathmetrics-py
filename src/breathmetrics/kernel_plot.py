# plotting for breathmetrics
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D  # ✅ import this explicitly


def plot_resp_features(bm, annotate: None, sizedata: float = 36):
    """
    plots respiration signal with esitmated features
    parameters:
    bm: breathmetrics object
    annotate: features to plot. options are "extrema", "onsets", "pauses", "maxflow", "volumes"
    sizedata: size of the feature labels. helpful for visualizing features in data of different window sizes
    """

    # plot all paramters that have been calculated
    # Build parameters and labels to plot
    params_to_plot = []
    param_labels = []

    dtype = getattr(bm, "datatype", getattr(bm, "dataType", None))
    is_airflow = dtype in {"humanAirflow", "rodentAirflow"}

    def _get_array(name):
        arr = getattr(bm, name, None)
        if arr is None:
            return None
        arr = np.asarray(arr)
        return arr if arr.size > 0 else None

    # Define parameter logic
    specs = [
        ("inhalePeaks", "Inhale Peaks", lambda: is_airflow, lambda a: a),
        ("exhaleTroughs", "Exhale Troughs", lambda: is_airflow, lambda a: a),
        ("inhaleOnsets", "Inhale Onsets", lambda: True, lambda a: a),
        ("exhaleOnsets", "Exhale Onsets", lambda: True, lambda a: a),
        ("inhalePauseOnsets", "Inhale Pauses", lambda: True, lambda a: a[a > 0]),
        ("exhalePauseOnsets", "Exhale Pauses", lambda: True, lambda a: a[a > 0]),
    ]

    # Collect data to plot
    for attr, label, include_ok, transform in specs:
        if not include_ok():
            continue
        arr = _get_array(attr)
        if arr is None:
            continue
        arr = transform(arr)
        if arr is None or (hasattr(arr, "size") and arr.size == 0):
            continue
        params_to_plot.append(arr)
        param_labels.append(label)
    resptrace = getattr(bm, "baseline_corrected_respiration", None)
    if resptrace is None:
        resptrace = getattr(bm, "bsl_corrected_respiration")
    xaxis = bm.time

    # Create plot
    plt.figure(figsize=(12, 6))
    plt.title("esitmated respiratory features")
    plt.plot(xaxis, resptrace, label="Respiration Signal", color="black", linestyle="-")
    plt.xlabel("time (s)")

    # different data types have different y axis labels
    if is_airflow:
        plt.ylabel("Airflow (arb. units)")
    elif dtype == "humanBB":
        plt.ylabel("Breathing Belt Signal (arb. units)")
    elif dtype == "rodentThermocouple":
        plt.ylabel("Thermocouple Signal (arb. units)")

    # Choose a qualitative palette (muted, colorblind, deep, bright, etc.)
    palette = sns.color_palette("colorblind", n_colors=6)

    # Fixed mapping for each parameter label
    LABEL_COLOR = {
        "Inhale Peaks": palette[0],
        "Exhale Troughs": palette[1],
        "Inhale Onsets": palette[2],
        "Exhale Onsets": palette[3],
        "Inhale Pauses": palette[4],
        "Exhale Pauses": palette[5],
    }

    def color_for(label: str, i_fallback: int) -> tuple[float, float, float]:
        return LABEL_COLOR.get(label, palette[i_fallback % len(palette)])

    # Plot each feature. TODO: continue here. add color.
    if params_to_plot:
        for i, param in enumerate(params_to_plot):
            lbl = param_labels[i]
            plt.scatter(
                xaxis[param],
                resptrace[param],
                s=sizedata,
                label=lbl,
                alpha=0.7,
                color=color_for(lbl, i),
            )
    plt.legend()
    plt.show()

    # TODO: add code to plot annotations. annotation from the GUI?
    # TODO: test plotting


def plot_breathing_compositions(bm, plottype: str):
    """
    plots the composition of all breaths in bm object
    use after all features have been calculated
    parameters:
    bm: breathmetrics object
    plottype: 'raw', 'normalized', 'line', 'hist'
    """
    matrix_size = 1000
    breath_phase_labels = ["inhale", "exhale", "inhale_pause", "exhale_pause"]  # keep
    custom_colors = np.array(  # keep
        [
            [0.0000, 0.4470, 0.7410],  # blue
            [0.8500, 0.3250, 0.0980],  # orange
            [0.9290, 0.6940, 0.1250],  # yellow
            [0.4940, 0.1840, 0.5560],  # purple
            [1.0000, 1.0000, 1.0000],  # white (used in 'raw' background)
        ]
    )
    onsets = getattr(bm, "inhale_onsets", None)
    if onsets is None:
        raise ValueError("Estimated features before plotting breath compositions.")
    n_breaths = len(onsets)

    if plottype not in {"raw", "normalized", "line", "hist"}:
        raise ValueError("plottype must be one of 'raw', 'normalized', 'line', 'hist'")

    match plottype:
        case "normalized":
            breath_matrix = np.zeros((n_breaths, matrix_size))
            for b in range(n_breaths):
                ind = 1
                this_breath_comp = np.zeros(matrix_size)
                this_inhale_dur = bm.inhale_durations[b]
                this_inhale_pause_dur = bm.exhale_pause_durations[b]

                if np.isnan(this_inhale_pause_dur):
                    this_inhale_pause_dur = 0

                this_exhale_dur = bm.exhale_durations[b]
                this_exhale_pause_dur = bm.inhale_pause_durations[b]
                if np.isnan(this_exhale_dur):
                    this_exhale_dur = 0
                if np.isnan(this_exhale_pause_dur):
                    this_exhale_pause_dur = 0

                total_points = np.sum(
                    [
                        this_inhale_dur,
                        this_inhale_pause_dur,
                        this_exhale_dur,
                        this_exhale_pause_dur,
                    ]
                )
                normed_inhale_dur = int((this_inhale_dur / total_points) * matrix_size)
                normed_inhale_pause_dur = int(
                    (this_inhale_pause_dur / total_points) * matrix_size
                )
                normed_exhale_dur = int((this_exhale_dur / total_points) * matrix_size)
                normed_exhale_pause_dur = int(
                    (this_exhale_pause_dur / total_points) * matrix_size
                )

                # sum check
                sum_check = np.sum(
                    [
                        normed_exhale_dur,
                        normed_inhale_dur,
                        normed_exhale_pause_dur,
                        normed_inhale_pause_dur,
                    ]
                )
                if sum_check < matrix_size:
                    normed_exhale_dur += 1
                elif sum_check > matrix_size:
                    normed_exhale_dur -= 1

                this_breath_comp[1 : ind + normed_inhale_dur] = 1  # inhale
                ind = normed_inhale_dur
                if normed_inhale_pause_dur > 0:
                    this_breath_comp[ind + 1 : ind + normed_inhale_pause_dur] = (
                        2  # inhale pause
                    )
                    ind += normed_inhale_pause_dur
                this_breath_comp[ind + 1 : ind + normed_exhale_dur] = 3  # exhale
                ind += normed_exhale_dur
                if normed_exhale_pause_dur > 0:
                    this_breath_comp[ind + 1 : ind + normed_exhale_pause_dur] = (
                        4  # exhale pause
                    )
                breath_matrix[b, :] = this_breath_comp

            fig, ax = plt.subplots(figsize=(10, 5))

            # --- image ---
            cmap = ListedColormap(
                custom_colors[:4, :]
            )  # same as MATLAB colormap(customColors(1:4,:))
            _im = ax.imshow(
                breath_matrix,
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                origin="upper",
            )

            # x-axis limits
            ax.set_xlim(0.5, matrix_size + 0.5)

            # --- normalize x ticks to percentages ---
            x_ticks = np.linspace(1, matrix_size, 11)
            x_tick_labels = [f"{int(round(x / matrix_size * 100))}%" for x in x_ticks]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_tick_labels)

            # y limits
            ax.set_ylim(0.5, n_breaths + 0.5)

            # labels
            ax.set_xlabel("Proportion of Breathing Period")
            ax.set_ylabel("Breath Number")

            # --- custom legend with dummy line handles ---
            handles = [
                Line2D([], [], color=custom_colors[i], linewidth=2) for i in range(4)
            ]
            ax.legend(handles, breath_phase_labels)

            plt.tight_layout()
            plt.show()

            ## TODO: test this with real data but at least there is no syntax error now
        case "raw":
            MATRIX_SIZE = 1000
            nBreaths = len(bm.inhaleOnsets)

            # Compute maximum breath duration in seconds
            if len(bm.inhaleOnsets) >= 2:
                maxBreathSize = np.ceil(np.max(np.diff(bm.inhaleOnsets)) / bm.srate)
            else:
                maxBreathSize = 1  # fallback for single-breath datasets

            # Initialize breath matrix (each row = one breath)
            breathMatrix = np.zeros((nBreaths, MATRIX_SIZE), dtype=np.uint8)

            # Convert durations to arrays and handle NaNs
            inh_dur = np.nan_to_num(np.asarray(bm.inhaleDurations), nan=0.0)
            inhp_dur = np.nan_to_num(np.asarray(bm.inhalePauseDurations), nan=0.0)
            exh_dur = np.nan_to_num(np.asarray(bm.exhaleDurations), nan=0.0)
            exhp_dur = np.nan_to_num(np.asarray(bm.exhalePauseDurations), nan=0.0)

            # -----------------------------------------
            # Build composition matrix
            # -----------------------------------------
            for b in range(nBreaths):
                row = np.ones(MATRIX_SIZE, dtype=np.uint8) * 4  # base = 4 (background)
                idx = 0

                # Inhale
                thisInhaleDur = int(round((inh_dur[b] / maxBreathSize) * MATRIX_SIZE))
                row[:thisInhaleDur] = 0
                idx = thisInhaleDur

                # Inhale pause
                thisInhalePauseDur = int(
                    round((inhp_dur[b] / maxBreathSize) * MATRIX_SIZE)
                )
                row[idx : idx + thisInhalePauseDur] = 1
                idx += thisInhalePauseDur

                # Exhale
                thisExhaleDur = int(round((exh_dur[b] / maxBreathSize) * MATRIX_SIZE))
                row[idx : idx + thisExhaleDur] = 2
                idx += thisExhaleDur

                # Exhale pause
                thisExhalePauseDur = int(
                    round((exhp_dur[b] / maxBreathSize) * MATRIX_SIZE)
                )
                row[idx : idx + thisExhalePauseDur] = 3

                # Clip to matrix size (MATLAB: if length > MATRIX_SIZE)
                row = row[:MATRIX_SIZE]
                breathMatrix[b, :] = row

            # -----------------------------------------
            # Plot the image
            # -----------------------------------------
            fig, ax = plt.subplots(figsize=(10, 5))
            cmap = ListedColormap(custom_colors)

            # uint8 conversion mirrors MATLAB’s "image(uint8(...))"
            _im = ax.imshow(
                breathMatrix.astype(np.uint8),
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                origin="upper",
            )

            # Set axis limits
            ax.set_xlim(0.5, MATRIX_SIZE + 0.5)
            ax.set_ylim(0.5, nBreaths + 0.5)

            # X ticks (0:0.5:maxBreathSize)
            TICK_STEP = 0.5
            x_tick_labels = np.arange(0, maxBreathSize + TICK_STEP, TICK_STEP)
            x_ticks = np.round(np.linspace(1, MATRIX_SIZE, len(x_tick_labels))).astype(
                int
            )
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(
                [f"{x:.1f}".rstrip("0").rstrip(".") for x in x_tick_labels]
            )

            # Labels
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Breath Number")
            ax.set_title("Breath Composition (Raw)")

            # Custom legend (dummy lines)
            handles = [
                Line2D([], [], color=custom_colors[i], linewidth=2) for i in range(4)
            ]
            ax.legend(handles, breath_phase_labels, loc="upper right", frameon=False)

            plt.tight_layout()
            plt.show()
            ## TODO: not tested yet
        case "line":
            nBreaths = len(bm.inhaleDurations)

            fig, ax = plt.subplots(figsize=(8, 5))

            # parula equivalent in Matplotlib: use 'viridis'
            my_colors = sns.color_palette("viridis", n_colors=nBreaths)

            # ------------------------------------------------
            # Plot each breath’s phase durations
            # ------------------------------------------------
            for b in range(nBreaths):
                # Always an inhale
                plot_set = [[1, bm.inhaleDurations[b]]]

                # Optional inhale pause
                if not np.isnan(bm.inhalePauseDurations[b]):
                    plot_set.append([2, bm.inhalePauseDurations[b]])

                    # Always an exhale
                    plot_set.append([3, bm.exhaleDurations[b]])

                # Optional exhale pause
                if not np.isnan(bm.exhalePauseDurations[b]):
                    plot_set.append([4, bm.exhalePauseDurations[b]])

                plot_set = np.asarray(plot_set, dtype=float)

            ax.plot(
                plot_set[:, 0],  # type: ignore
                plot_set[:, 1],  # type: ignore
                color=my_colors[b],  # type: ignore
                linewidth=1.5,
            )
            ax.scatter(
                plot_set[:, 0],  # type: ignore
                plot_set[:, 1],  # type: ignore
                marker="s",
                facecolor=my_colors[b],  # type: ignore
                edgecolor="none",
                s=40,
            )

            # ------------------------------------------------
            # Compute means and standard deviations
            # ------------------------------------------------
            inh = np.asarray(bm.inhaleDurations, dtype=float)
            inhp = np.asarray(bm.inhalePauseDurations, dtype=float)
            exh = np.asarray(bm.exhaleDurations, dtype=float)
            exhp = np.asarray(bm.exhalePauseDurations, dtype=float)

            all_means = np.array(
                [
                    np.nanmean(inh),
                    np.nanmean(inhp),
                    np.nanmean(exh),
                    np.nanmean(exhp),
                ]
            )
            all_stds = np.array(
                [
                    np.nanstd(inh),
                    np.nanstd(inhp),
                    np.nanstd(exh),
                    np.nanstd(exhp),
                ]
            )

            # Add error bars (black vertical bars)
            ax.errorbar(
                [1, 2, 3, 4],
                all_means,
                yerr=all_stds,
                color="k",
                linestyle="none",
                linewidth=2,
            )

            # ------------------------------------------------
            # Axis setup
            # ------------------------------------------------
            ax.set_xlim(0, 5)
            ax.set_xticks([1, 2, 3, 4])
            ax.set_xticklabels(
                [
                    "Inhale Durations",
                    "Inhale Pause Durations",
                    "Exhale Durations",
                    "Exhale Pause Durations",
                ]
            )
            ax.set_ylabel("Time (seconds)")

            all_maxes = np.array(
                [
                    np.nanmax(inh),
                    np.nanmax(inhp),
                    np.nanmax(exh),
                    np.nanmax(exhp),
                ]
            )
            ymax = np.nanmax(all_maxes)
            ax.set_ylim(0, ymax + 0.5)

            # ------------------------------------------------
            # Text annotations (mean & std)
            # ------------------------------------------------
            topline_bump = 0.2
            bottomline_bump = 0.1
            fsize = 10

            for xi in range(1, 5):
                mean = all_means[xi - 1]
                std = all_stds[xi - 1]
                mmax = all_maxes[xi - 1]

                if np.isfinite(mmax):
                    ax.text(
                        xi,
                        mmax + topline_bump,
                        f"Mean = {mean:.3g} s",
                        ha="center",
                        fontsize=fsize,
                    )
                    ax.text(
                        xi,
                        mmax + bottomline_bump,
                        f"Std = {std:.3g} s",
                        ha="center",
                        fontsize=fsize,
                    )

            ax.set_title("Breath Phase Durations (Per-Breath Lines)")
            plt.tight_layout()
            plt.show()
        case "hist":
            inhale = np.asarray(bm.inhaleDurations, dtype=float)
            inhale_pause = np.asarray(bm.inhalePauseDurations, dtype=float)
            exhale = np.asarray(bm.exhaleDurations, dtype=float)
            exhale_pause = np.asarray(bm.exhalePauseDurations, dtype=float)

            # Replace NaNs for histogram counting
            inhale = inhale[np.isfinite(inhale)]
            inhale_pause = inhale_pause[np.isfinite(inhale_pause)]
            exhale = exhale[np.isfinite(exhale)]
            exhale_pause = exhale_pause[np.isfinite(exhale_pause)]

            # Determine number of bins (same logic as MATLAB)
            nBins = int(np.floor(len(inhale) / 5))
            if nBins < 5:
                nBins = 10

            # Get 4 distinct colors (parula-like) using seaborn
            my_colors = sns.color_palette("crest", n_colors=4)

            # --------------------------------------------------
            # Build the figure
            # --------------------------------------------------
            fig, axes = plt.subplots(2, 2, figsize=(8, 6))
            axes = axes.flatten()

            datasets = [
                (inhale, "Inhale Durations (seconds)"),
                (inhale_pause, "Inhale Pause Durations (seconds)"),
                (exhale, "Exhale Durations (seconds)"),
                (exhale_pause, "Exhale Pause Durations (seconds)"),
            ]

            for i, (data, xlabel) in enumerate(datasets):
                ax = axes[i]
                counts, bins = np.histogram(data, bins=nBins)
                centers = 0.5 * (bins[:-1] + bins[1:])
                ax.bar(
                    centers,
                    counts,
                    width=(bins[1] - bins[0]),
                    color=my_colors[i],
                    edgecolor="none",
                )
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Count")
                ax.set_title(xlabel.replace(" (seconds)", ""))

            plt.tight_layout()
            plt.show()
