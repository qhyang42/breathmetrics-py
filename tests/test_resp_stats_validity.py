from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from breathmetrics.core import Breathe
from breathmetrics.kernel_resp_stats import (
    compute_airflow_metrics,
    compute_duty_cycle_metrics,
)


def test_airflow_metrics_only_use_valid_breaths() -> None:
    bm = SimpleNamespace(
        peak_inspiratory_flows=np.array([1.0, 1000.0, 3.0, 5.0]),
        trough_expiratory_flows=np.array([2.0, 2000.0, 4.0, 6.0]),
        inhale_volumes=np.array([10.0, 10000.0, 30.0, 50.0]),
        exhale_volumes=np.array([20.0, 20000.0, 40.0, 60.0]),
    )
    valid = np.array([0, 2, 3])

    result = compute_airflow_metrics(bm, valid, valid, 2.0)

    assert result["avg_inhale_flow"] == pytest.approx(3.0)
    assert result["avg_exhale_flow"] == pytest.approx(4.0)
    assert result["avg_inhale_volume"] == pytest.approx(30.0)
    assert result["avg_exhale_volume"] == pytest.approx(40.0)
    assert result["minute_ventilation"] == pytest.approx(35.0)


def test_duty_cycle_metrics_only_use_valid_breaths() -> None:
    bm = SimpleNamespace(
        inhale_durations=np.array([1.0, 100.0, 3.0]),
        exhale_durations=np.array([2.0, 200.0, 4.0]),
        inhale_pause_durations=np.array([0.5, 50.0, np.nan]),
        exhale_pause_durations=np.array([np.nan, 60.0, 1.0]),
    )
    valid = np.array([0, 2])

    result = compute_duty_cycle_metrics(bm, valid, valid, 4.0)

    assert result["avg_inhale_dur"] == pytest.approx(2.0)
    assert result["avg_exhale_dur"] == pytest.approx(3.0)
    assert result["pct_inhale_pause"] == pytest.approx(0.5)
    assert result["pct_exhale_pause"] == pytest.approx(0.5)
    assert result["avg_inhale_pause"] == pytest.approx(0.25)
    assert result["avg_exhale_pause"] == pytest.approx(0.5)


def test_refresh_after_edit_keeps_per_breath_features_for_rejected_breaths() -> None:
    signal = np.sin(np.linspace(0, 6 * np.pi, 70))
    bm = Breathe(signal, 20.0, "humanAirflow")
    bm._set_feature("baseline_corrected_respiration", signal)
    bm._set_feature("inhale_onsets", [2, 24, 46])
    bm._set_feature("inhale_peaks", [7, 29, 51])
    bm._set_feature("exhale_troughs", [13, 35, 57])
    bm._set_feature("exhale_onsets", [14, 36, 58])
    bm._set_feature("exhale_offsets", [21, 43, 66])
    bm._set_feature("inhale_durations", [1.0, 100.0, 3.0])
    bm._set_feature("exhale_durations", [2.0, 200.0, 4.0])
    bm._set_feature("inhale_pause_durations", [np.nan, 50.0, np.nan])
    bm._set_feature("exhale_pause_durations", [np.nan, 60.0, np.nan])
    bm._set_feature("inhale_volumes", [10.0, 10000.0, 30.0])
    bm._set_feature("exhale_volumes", [20.0, 20000.0, 40.0])
    bm.peak_inspiratory_flows = np.array([1.0, 1000.0, 3.0])
    bm.trough_expiratory_flows = np.array([2.0, 2000.0, 4.0])
    bm.is_valid = np.array([True, False, True])

    bm.refresh_derived_features_after_edit()

    assert len(bm.shape_features) == 3
    assert bm.shape_features["breath_id"].tolist() == [0, 1, 2]
    assert bm.inhale_durations[1] == 100.0
    assert bm.inhale_volumes[1] == 10000.0
    assert bm.resp_stats["Average Inhale Duration"] == pytest.approx(2.0)
    assert bm.resp_stats["Average Inhale Volume"] == pytest.approx(20.0)
    assert bm.statuses["resp_stats"] == "edited"
