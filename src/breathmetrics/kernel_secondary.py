## secondary features and ERP calculations
from __future__ import annotations
import numpy as np


# secondary feature calculation.
# breathing rate
# inter-breath interval
# coef of variation of breathing rate

# airflow specific secondary features
# mean inhale/exhale rate/volumes, tidal volumes, minute ventilations, duty cycle


def get_valid_breath_indices(statuses, n_inhales: int, n_exhales: int):
    """Return arrays of valid inhale/exhale indices based on rejection statuses."""
    if statuses is None or len(statuses) == 0:
        return np.arange(n_inhales), np.arange(n_exhales)

    invalid = np.array(
        [i for i, s in enumerate(statuses) if "rejected" in str(s).lower()]
    )
    valid_inhales = np.setdiff1d(np.arange(n_inhales), invalid)
    valid_exhales = np.setdiff1d(np.arange(n_exhales), invalid)

    return valid_inhales, valid_exhales


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
    if len(values) == 0:
        return np.array([])
    mean = np.nanmean(values)
    std = np.nanstd(values)
    mask = (values > mean - n_std * std) & (values < mean + n_std * std)
    valid_mask = np.zeros_like(mask)
    valid_mask[valid_inds] = True
    return values[mask & valid_mask]


def compute_airflow_metrics(bm, valid_inhales, valid_exhales):
    """Compute average flow rates, volumes, tidal volume, and ventilation."""
    valid_inhale_flows = exclude_outliers(bm.peak_inspiratory_flows, valid_inhales)
    valid_exhale_flows = exclude_outliers(bm.trough_expiratory_flows, valid_exhales)
    valid_inhale_vols = exclude_outliers(bm.inhale_volumes, valid_inhales)
    valid_exhale_vols = exclude_outliers(bm.exhale_volumes, valid_exhales)

    avg_inhale_flow = np.nanmean(valid_inhale_flows)
    avg_exhale_flow = np.nanmean(valid_exhale_flows)
    avg_inhale_vol = np.nanmean(valid_inhale_vols)
    avg_exhale_vol = np.nanmean(valid_exhale_vols)
    avg_tidal = avg_inhale_vol + avg_exhale_vol
    minute_vent = (
        bm.srate / np.nanmean(np.diff(bm.inhale_onsets)) * avg_tidal
    )  # breathing_rate × tidal

    cv_tidal = np.nanstd(valid_inhale_vols) / np.nanmean(valid_inhale_vols)

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
    avg_inhale_dur = np.nanmean(bm.inhale_durations)
    avg_exhale_dur = np.nanmean(bm.exhale_durations)

    pct_inhale_pause = np.sum(~np.isnan(bm.inhale_pause_durations)) / len(valid_inhales)
    avg_inhale_pause = (
        np.nanmean(bm.inhale_pause_durations[valid_inhales]) * pct_inhale_pause
    )

    pct_exhale_pause = np.sum(~np.isnan(bm.exhale_pause_durations)) / len(valid_exhales)
    avg_exhale_pause = (
        np.nanmean(bm.exhale_pause_durations[valid_exhales]) * pct_exhale_pause
    )

    avg_inhale_pause = 0 if np.isnan(avg_inhale_pause) else avg_inhale_pause
    avg_exhale_pause = 0 if np.isnan(avg_exhale_pause) else avg_exhale_pause

    inhale_dc = avg_inhale_dur / inter_breath_interval
    exhale_dc = avg_exhale_dur / inter_breath_interval
    inhale_pause_dc = avg_inhale_pause / inter_breath_interval
    exhale_pause_dc = avg_exhale_pause / inter_breath_interval

    cv_inhale_dur = np.nanstd(bm.inhale_durations) / avg_inhale_dur
    cv_inhale_pause = (
        np.nanstd(bm.inhale_pause_durations) / avg_inhale_pause
        if avg_inhale_pause
        else np.nan
    )
    cv_exhale_dur = np.nanstd(bm.exhale_durations) / avg_exhale_dur
    cv_exhale_pause = (
        np.nanstd(bm.exhale_pause_durations) / avg_exhale_pause
        if avg_exhale_pause
        else np.nan
    )

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
