# plotting for breathmetrics
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


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
    resp_handle = plt.plot(
        xaxis, resptrace, label="Respiration Signal", color="black", linestyle="-"
    )
    plt.xlabel("time (s)")

    # different data types have different y axis labels
    if is_airflow:
        plt.ylabel("Airflow (arb. units)")
    elif dtype == "humanBB":
        plt.ylabel("Breathing Belt Signal (arb. units)")
    elif dtype == "rodentThermocouple":
        plt.ylabel("Thermocouple Signal (arb. units)")

    # Plot each feature. TODO: continue here. add color.
    if params_to_plot:
        for i, param in enumerate(params_to_plot):
            plt.scatter(
                xaxis[param],
                resptrace[param],
                s=sizedata,
                label=param_labels[i],
                alpha=0.7,
            )

    resp_handle = resp_handle[
        0
    ]  # TODO: REMOVE THIS. this is just so precommit tests pass.
