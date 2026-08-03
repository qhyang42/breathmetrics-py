import numpy as np
import pandas as pd
import pytest

from breathmetrics.core import Breathe
from breathmetrics.kernel_secondary import (
    calculate_curvature,
    calculate_max_flowchange,
    calculate_phase2slope,
    calculate_smoothness,
)


def make_breathe(
    signal,
    *,
    inhale_onsets,
    inhale_peaks,
    exhale_troughs,
    exhale_onsets,
    exhale_offsets,
):
    breathe = Breathe(signal, 20.0, "humanAirflow")
    breathe.bsl_corrected_respiration = signal
    breathe._set_feature("inhale_onsets", inhale_onsets)
    breathe._set_feature("inhale_peaks", inhale_peaks)
    breathe._set_feature("exhale_troughs", exhale_troughs)
    breathe._set_feature("exhale_onsets", exhale_onsets)
    breathe._set_feature("exhale_offsets", exhale_offsets)
    return breathe


def test_valid_synthetic_breath_matches_kernel_helpers():
    signal = np.array(
        [
            0.0,
            0.0,
            0.1,
            0.4,
            0.7,
            1.1,
            1.4,
            1.0,
            0.6,
            0.1,
            -0.4,
            -0.9,
            -1.2,
            -1.0,
            -0.8,
            -0.5,
            -0.3,
            -0.1,
            0.0,
            0.05,
            0.0,
            0.0,
        ]
    )
    breathe = make_breathe(
        signal,
        inhale_onsets=[2],
        inhale_peaks=[6],
        exhale_troughs=[12],
        exhale_onsets=[13],
        exhale_offsets=[20],
    )

    breathe.find_shape_features()

    inhale = signal[2:7]
    transition = signal[6:13]
    exhale = signal[13:21]
    full_breath = signal[2:21]
    expected = {
        "breath_id": 0,
        "inhale_smoothness": calculate_smoothness(inhale, 20.0),
        "inpeak_to_extrough_smoothness": calculate_smoothness(transition, 20.0),
        "exhale_smoothness": calculate_smoothness(exhale, 20.0),
        "inhale_curvature": calculate_curvature(inhale),
        "inpeak_to_extrough_curvature": calculate_curvature(transition),
        "exhale_curvature": calculate_curvature(exhale),
        "inpeak_to_extrough_slope": calculate_phase2slope(transition),
        "max_flowchange": calculate_max_flowchange(full_breath),
    }
    assert breathe.shape_features is not None
    assert len(breathe.shape_features) == 1
    pd.testing.assert_series_equal(
        breathe.shape_features.iloc[0], pd.Series(expected), check_names=False
    )


def test_multiple_breaths_preserve_ordered_breath_ids():
    signal = np.sin(np.linspace(0, 6 * np.pi, 70))
    breathe = make_breathe(
        signal,
        inhale_onsets=[2, 24, 46],
        inhale_peaks=[7, 29, 51],
        exhale_troughs=[13, 35, 57],
        exhale_onsets=[14, 36, 58],
        exhale_offsets=[21, 43, 66],
    )

    breathe.find_shape_features()

    assert len(breathe.shape_features) == len(breathe.inhale_onsets)
    assert breathe.shape_features["breath_id"].tolist() == [0, 1, 2]


def test_incomplete_final_breath_is_preserved():
    signal = np.sin(np.linspace(0, 4 * np.pi, 50))
    breathe = make_breathe(
        signal,
        inhale_onsets=[2, 25],
        inhale_peaks=[6, 29],
        exhale_troughs=[12, 35],
        exhale_onsets=[13, -1],
        exhale_offsets=[20, -1],
    )

    breathe.find_shape_features()

    final = breathe.shape_features.iloc[1]
    assert len(breathe.shape_features) == 2
    assert final["breath_id"] == 1
    assert (
        final[["exhale_smoothness", "exhale_curvature", "max_flowchange"]].isna().all()
    )
    assert (
        final[
            [
                "inhale_smoothness",
                "inpeak_to_extrough_smoothness",
                "inhale_curvature",
                "inpeak_to_extrough_curvature",
                "inpeak_to_extrough_slope",
            ]
        ]
        .notna()
        .all()
    )


def test_invalid_landmarks_are_isolated_to_affected_features():
    signal = np.sin(np.linspace(0, 8 * np.pi, 90))
    breathe = make_breathe(
        signal,
        inhale_onsets=[5, 28, 51, 74],
        inhale_peaks=[3, 33, 56, 79],
        exhale_troughs=[14, -1, 64, 85],
        exhale_onsets=[15, 38, 65, 86],
        exhale_offsets=[22, 45, 95, 89],
    )

    breathe.find_shape_features()

    features = breathe.shape_features
    assert features.loc[0, ["inhale_smoothness", "inhale_curvature"]].isna().all()
    assert (
        features.loc[
            1,
            [
                "inpeak_to_extrough_smoothness",
                "inpeak_to_extrough_curvature",
                "inpeak_to_extrough_slope",
            ],
        ]
        .isna()
        .all()
    )
    assert (
        features.loc[2, ["exhale_smoothness", "exhale_curvature", "max_flowchange"]]
        .isna()
        .all()
    )
    assert features.loc[3, "exhale_smoothness"] == pytest.approx(
        calculate_smoothness(signal[86:90], 20.0)
    )

    breathe.exhale_offsets = None
    breathe.find_shape_features()
    assert (
        breathe.shape_features[
            ["exhale_smoothness", "exhale_curvature", "max_flowchange"]
        ]
        .isna()
        .all()
        .all()
    )


def test_repeated_calls_replace_the_dataframe():
    signal = np.sin(np.linspace(0, 4 * np.pi, 50))
    breathe = make_breathe(
        signal,
        inhale_onsets=[2, 25],
        inhale_peaks=[6, 29],
        exhale_troughs=[12, 35],
        exhale_onsets=[13, 36],
        exhale_offsets=[20, 44],
    )

    breathe.find_shape_features()
    first = breathe.shape_features.copy()
    breathe.find_shape_features()

    pd.testing.assert_frame_equal(breathe.shape_features, first)


def test_find_shape_features_requires_signal_and_inhale_onsets():
    breathe = Breathe(np.linspace(-1, 1, 20), 20.0, "humanAirflow")

    with pytest.raises(ValueError, match="baseline_corrected_respiration"):
        breathe.find_shape_features()

    breathe.bsl_corrected_respiration = breathe.raw_respiration
    with pytest.raises(ValueError, match="inhale_onsets"):
        breathe.find_shape_features()
