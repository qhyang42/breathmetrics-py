# plotting for breathmetrics
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
    resptrace = bm.bsl_corrected_respiration
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
    breath_phase_labels = ["inhale", "exhale", "inhale_pause", "exhale_pause"]
    _custom_colors = sns.color_palette("Set2", len(breath_phase_labels))  # keep
    onsets = getattr(bm, "inhale_onsets", None)
    if onsets is None:
        raise ValueError("Estimated features before plotting breath compositions.")
    n_breaths = len(onsets)

    if plottype not in {"raw", "normalized", "line", "hist"}:
        raise ValueError("plottype must be one of 'raw', 'normalized', 'line', 'hist'")

    match plottype:
        case "normalized":
            _breath_matrix = np.zeros((n_breaths, matrix_size))  # keep

            for b in range(n_breaths):
                _ind = 1  # keep
                _this_breath_comp = np.zeros(matrix_size)  # keep
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
                ## TODO keep going. this is line 76 in plotBreathCompositions.m
