## Summary statistics of all respiratory cycles in the recording and shape calculations
from __future__ import annotations
import numpy as np
import pandas as pd

# Summary statistics of all respiratory cycles in the recording.
# breathing rate
# inter-breath interval
# coef of variation of breathing rate

# Airflow-specific summary statistics of all respiratory cycles in the recording
# mean inhale/exhale rate/volumes, tidal volumes, minute ventilations, duty cycle


def _finite_mean(values) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else np.nan


def _finite_cv(values, mean: float) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0 or not np.isfinite(mean) or mean == 0:
        return np.nan
    return float(np.std(finite) / mean)


def _pause_summary(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan, np.nan
    present = values[np.isfinite(values)]
    fraction = float(present.size / values.size)
    average = float(np.mean(present) * fraction) if present.size else 0.0
    return fraction, average


def compute_breath_timing_metrics(inhale_onsets, valid_inhales, srate: float):
    """Compute breathing rate, inter-breath interval, and coefficient of variation."""
    diffs = [
        inhale_onsets[next_] - inhale_onsets[cur]
        for cur, next_ in zip(valid_inhales[:-1], valid_inhales[1:])
        if next_ == cur + 1
    ]
    diffs = np.array(diffs, dtype=float)
    if len(diffs) == 0 or np.isnan(np.mean(diffs)):
        return {
            "breathing_rate": np.nan,
            "inter_breath_interval": np.nan,
            "cv_breathing_rate": np.nan,
        }

    mean_diff = np.nanmean(diffs)
    breathing_rate = srate / mean_diff
    inter_breath_interval = 1 / breathing_rate
    cv = np.nanstd(diffs) / mean_diff

    return {
        "breathing_rate": breathing_rate,
        "inter_breath_interval": inter_breath_interval,
        "cv_breathing_rate": cv,
    }


def exclude_outliers(values, valid_inds, n_std: float = 2.0):
    """Return valid values excluding outliers beyond ±n_std."""
    values = np.asarray(values, dtype=float)
    valid_inds = np.asarray(valid_inds, dtype=int)
    valid_inds = valid_inds[(valid_inds >= 0) & (valid_inds < values.size)]
    valid_values = values[valid_inds]
    if valid_values.size == 0:
        return np.array([])
    finite_values = valid_values[np.isfinite(valid_values)]
    if finite_values.size == 0:
        return finite_values
    mean = np.mean(finite_values)
    std = np.std(finite_values)
    if std == 0:
        return finite_values
    mask = (finite_values > mean - n_std * std) & (finite_values < mean + n_std * std)
    return finite_values[mask]


def compute_airflow_metrics(
    bm, valid_inhales, valid_exhales, inter_breath_interval: float
):
    """Compute average flow rates, volumes, tidal volume, and ventilation."""
    valid_inhale_flows = exclude_outliers(bm.peak_inspiratory_flows, valid_inhales)
    valid_exhale_flows = exclude_outliers(bm.trough_expiratory_flows, valid_exhales)
    valid_inhale_vols = exclude_outliers(bm.inhale_volumes, valid_inhales)
    valid_exhale_vols = exclude_outliers(bm.exhale_volumes, valid_exhales)

    avg_inhale_flow = _finite_mean(valid_inhale_flows)
    avg_exhale_flow = _finite_mean(valid_exhale_flows)
    avg_inhale_vol = _finite_mean(valid_inhale_vols)
    avg_exhale_vol = _finite_mean(valid_exhale_vols)
    avg_tidal = avg_inhale_vol + avg_exhale_vol
    minute_vent = (
        avg_tidal / inter_breath_interval
        if np.isfinite(inter_breath_interval) and inter_breath_interval > 0
        else np.nan
    )

    cv_tidal = _finite_cv(valid_inhale_vols, avg_inhale_vol)

    return {
        "avg_inhale_flow": avg_inhale_flow,
        "avg_exhale_flow": avg_exhale_flow,
        "avg_inhale_volume": avg_inhale_vol,
        "avg_exhale_volume": avg_exhale_vol,
        "avg_tidal_volume": avg_tidal,
        "minute_ventilation": minute_vent,
        "cv_tidal_volume": cv_tidal,
    }


def compute_duty_cycle_metrics(
    bm, valid_inhales, valid_exhales, inter_breath_interval: float
):
    """Compute duty cycles and coefficients of variation for inhale/exhale phases."""
    inhale_durations = np.asarray(bm.inhale_durations, dtype=float)[valid_inhales]
    exhale_durations = np.asarray(bm.exhale_durations, dtype=float)[valid_exhales]
    inhale_pauses = np.asarray(bm.inhale_pause_durations, dtype=float)[valid_inhales]
    exhale_pauses = np.asarray(bm.exhale_pause_durations, dtype=float)[valid_exhales]

    avg_inhale_dur = _finite_mean(inhale_durations)
    avg_exhale_dur = _finite_mean(exhale_durations)

    pct_inhale_pause, avg_inhale_pause = _pause_summary(inhale_pauses)
    pct_exhale_pause, avg_exhale_pause = _pause_summary(exhale_pauses)

    inhale_dc = avg_inhale_dur / inter_breath_interval
    exhale_dc = avg_exhale_dur / inter_breath_interval
    inhale_pause_dc = avg_inhale_pause / inter_breath_interval
    exhale_pause_dc = avg_exhale_pause / inter_breath_interval

    cv_inhale_dur = _finite_cv(inhale_durations, avg_inhale_dur)
    cv_inhale_pause = _finite_cv(inhale_pauses, avg_inhale_pause)
    cv_exhale_dur = _finite_cv(exhale_durations, avg_exhale_dur)
    cv_exhale_pause = _finite_cv(exhale_pauses, avg_exhale_pause)

    return {
        "inhale_dc": inhale_dc,
        "exhale_dc": exhale_dc,
        "inhale_pause_dc": inhale_pause_dc,
        "exhale_pause_dc": exhale_pause_dc,
        "cv_inhale_dur": cv_inhale_dur,
        "cv_inhale_pause": cv_inhale_pause,
        "cv_exhale_dur": cv_exhale_dur,
        "cv_exhale_pause": cv_exhale_pause,
        "avg_inhale_dur": avg_inhale_dur,
        "avg_exhale_dur": avg_exhale_dur,
        "avg_inhale_pause": avg_inhale_pause,
        "avg_exhale_pause": avg_exhale_pause,
        "pct_inhale_pause": pct_inhale_pause,
        "pct_exhale_pause": pct_exhale_pause,
    }


def assemble_respiratory_summary(datatype, timing, airflow, duty):
    """Assemble all metric dicts into one summary dictionary."""
    base = {
        "Breathing Rate": timing["breathing_rate"],
        "Average Inter-Breath Interval": timing["inter_breath_interval"],
        "Coefficient of Variation of Breathing Rate": timing["cv_breathing_rate"],
    }

    if datatype in {"humanAirflow", "rodentAirflow"}:
        base.update(
            {
                "Average Peak Inspiratory Flow": airflow["avg_inhale_flow"],
                "Average Peak Expiratory Flow": airflow["avg_exhale_flow"],
                "Average Inhale Volume": airflow["avg_inhale_volume"],
                "Average Exhale Volume": airflow["avg_exhale_volume"],
                "Average Tidal Volume": airflow["avg_tidal_volume"],
                "Minute Ventilation": airflow["minute_ventilation"],
                "Duty Cycle of Inhale": duty["inhale_dc"],
                "Duty Cycle of Inhale Pause": duty["inhale_pause_dc"],
                "Duty Cycle of Exhale": duty["exhale_dc"],
                "Duty Cycle of Exhale Pause": duty["exhale_pause_dc"],
                "Coefficient of Variation of Inhale Duty Cycle": duty["cv_inhale_dur"],
                "Coefficient of Variation of Inhale Pause Duty Cycle": duty[
                    "cv_inhale_pause"
                ],
                "Coefficient of Variation of Exhale Duty Cycle": duty["cv_exhale_dur"],
                "Coefficient of Variation of Exhale Pause Duty Cycle": duty[
                    "cv_exhale_pause"
                ],
                "Average Inhale Duration": duty["avg_inhale_dur"],
                "Average Exhale Duration": duty["avg_exhale_dur"],
                "Average Inhale Pause Duration": duty["avg_inhale_pause"],
                "Average Exhale Pause Duration": duty["avg_exhale_pause"],
                "Percent of Breaths With Inhale Pause": duty["pct_inhale_pause"],
                "Percent of Breaths With Exhale Pause": duty["pct_exhale_pause"],
                "Coefficient of Variation of Breath Volumes": airflow[
                    "cv_tidal_volume"
                ],
            }
        )
    return base


### Per-cycle waveform shape features from Sagar et al. 2025 Nat. Hum. Behav.
### these are calculated for breathing phases, not the whole signal.
def calculate_smoothness(signal, fs) -> float:
    """
    Compute smoothness as the negative mean absolute difference between the signal and its moving average (window size = fs/10 seconds, rounded to nearest sample).
    """
    signal = np.asarray(signal, dtype=float)
    window = max(1, int(round(fs / 10)))
    kernel = np.ones(window) / window
    smooth_sig = np.convolve(signal, kernel, mode="same")
    smoothness = -np.mean(np.abs(smooth_sig - signal))
    return smoothness  # type: ignore


def calculate_curvature(signal) -> float:
    """Compute curvature as the mean absolute second derivative of the respiration signal."""
    signal = np.asarray(signal, dtype=float)
    first_derivative = np.gradient(signal)
    second_derivative = np.gradient(first_derivative)
    curvature = np.mean(np.abs(second_derivative))
    return curvature  # type: ignore


def calculate_phase2slope(signal) -> float:
    """compute the slope of the breathing segment"""
    signal = np.asarray(signal, dtype=float)
    x = len(signal)
    p = np.polyfit(np.arange(x), signal, 1)
    slope = p[0]
    return slope  # type: ignore


def calculate_smoothness_around_point(signal, fs, point_idx) -> float:
    """Compute smoothness around a specific point in the respiration signal."""
    signal = np.asarray(signal, dtype=float)
    window_size = 0.05 * len(signal)  # 5% of total duration
    half_window_samples = int(round(window_size / 2))
    start_idx = max(0, point_idx - half_window_samples)
    end_idx = min(len(signal), point_idx + half_window_samples)
    segment = signal[start_idx:end_idx]

    window = max(1, int(round(fs / 10)))
    kernel = np.ones(window) / window
    smooth_seg = np.convolve(segment, kernel, mode="same")
    smoothness = -np.mean(np.abs(smooth_seg - segment))
    return smoothness  # type: ignore


def compute_time_symmetry(signal) -> float:
    """Compute time symmetry of the respiration signal."""
    signal = np.asarray(signal, dtype=float)
    # time flip the signal
    flipped_signal = signal[::-1]
    # remove the middle point if the length is odd
    if len(signal) % 2 != 0:
        mid_idx = len(signal) // 2
        signal = np.delete(signal, mid_idx)
        flipped_signal = np.delete(flipped_signal, mid_idx)
    # pearson's correlation between the two
    if np.std(signal) > 0 and np.std(flipped_signal) > 0:
        corr = np.corrcoef(signal, flipped_signal)[0, 1]

    return corr  # type: ignore


def compute_time_symmetry_around_point(signal, point_idx) -> float:
    """Compute time symmetry around a point of the respiration signal."""
    signal = np.asarray(signal, dtype=float)
    window_size = 0.05 * len(signal)  # 5% of total duration
    half_window_samples = int(round(window_size / 2))
    start_idx = max(0, point_idx - half_window_samples)
    end_idx = min(len(signal), point_idx + half_window_samples)
    segment = signal[start_idx:end_idx]

    # time flip the segment
    flipped_segment = segment[::-1]
    # remove the middle point if the length is odd
    if len(segment) % 2 != 0:
        mid_idx = len(segment) // 2
        segment = np.delete(segment, mid_idx)
        flipped_segment = np.delete(flipped_segment, mid_idx)
    # pearson's correlation between the two
    if np.std(segment) > 0 and np.std(flipped_segment) > 0:
        corr = np.corrcoef(segment, flipped_segment)[0, 1]

    return corr  # type: ignore


def calculate_max_flowchange(signal) -> float:
    signal = np.asarray(signal, dtype=float)

    if signal.size < 2:
        return np.nan

    return float(np.max(np.diff(signal)))


_SHAPE_FEATURE_COLUMNS = (
    "inhale_smoothness",
    "inpeak_to_extrough_smoothness",
    "exhale_smoothness",
    "inhale_curvature",
    "inpeak_to_extrough_curvature",
    "exhale_curvature",
    "inpeak_to_extrough_slope",
    "max_flowchange",
)


def _event_at(events, breath_id: int, signal_size: int) -> int | None:
    if events is None:
        return None
    values = np.asarray(events).ravel()
    if breath_id >= values.size:
        return None
    try:
        value = float(values[breath_id])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or not value.is_integer():
        return None
    index = int(value)
    if index < 0 or index >= signal_size:
        return None
    return index


def _inclusive_segment(
    signal: np.ndarray, start_events, end_events, breath_id: int
) -> np.ndarray | None:
    start = _event_at(start_events, breath_id, signal.size)
    end = _event_at(end_events, breath_id, signal.size)
    if start is None or end is None or start > end:
        return None
    segment = signal[start : end + 1]
    if segment.size == 0 or not np.all(np.isfinite(segment)):
        return None
    return segment


def _safe_shape_feature(function, *args) -> float:
    try:
        value = float(function(*args))
    except (TypeError, ValueError, FloatingPointError, np.linalg.LinAlgError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def _calculate_shape_features(
    respiration,
    srate: float,
    inhale_onsets,
    inhale_peaks=None,
    exhale_troughs=None,
    exhale_onsets=None,
    exhale_offsets=None,
) -> pd.DataFrame:
    """Assemble aligned per-breath shape features from detected landmarks."""
    signal = np.asarray(respiration, dtype=float).ravel()
    onset_values = np.asarray(inhale_onsets).ravel()
    data = {
        "breath_id": np.arange(onset_values.size, dtype=int),
        **{
            column: np.full(onset_values.size, np.nan, dtype=float)
            for column in _SHAPE_FEATURE_COLUMNS
        },
    }

    for breath_id in range(onset_values.size):
        inhale = _inclusive_segment(signal, onset_values, inhale_peaks, breath_id)
        inpeak_to_extrough = _inclusive_segment(
            signal, inhale_peaks, exhale_troughs, breath_id
        )
        exhale = _inclusive_segment(signal, exhale_onsets, exhale_offsets, breath_id)
        full_breath = _inclusive_segment(
            signal, onset_values, exhale_offsets, breath_id
        )

        if inhale is not None:
            data["inhale_smoothness"][breath_id] = _safe_shape_feature(
                calculate_smoothness, inhale, srate
            )
            data["inhale_curvature"][breath_id] = _safe_shape_feature(
                calculate_curvature, inhale
            )
        if inpeak_to_extrough is not None:
            data["inpeak_to_extrough_smoothness"][breath_id] = _safe_shape_feature(
                calculate_smoothness, inpeak_to_extrough, srate
            )
            data["inpeak_to_extrough_curvature"][breath_id] = _safe_shape_feature(
                calculate_curvature, inpeak_to_extrough
            )
            data["inpeak_to_extrough_slope"][breath_id] = _safe_shape_feature(
                calculate_phase2slope, inpeak_to_extrough
            )
        if exhale is not None:
            data["exhale_smoothness"][breath_id] = _safe_shape_feature(
                calculate_smoothness, exhale, srate
            )
            data["exhale_curvature"][breath_id] = _safe_shape_feature(
                calculate_curvature, exhale
            )
        if full_breath is not None:
            data["max_flowchange"][breath_id] = _safe_shape_feature(
                calculate_max_flowchange, full_breath
            )

    return pd.DataFrame(data)
