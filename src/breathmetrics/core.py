from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Mapping, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

# import breathmetrics.kernel
# import breathmetrics.kernel_onset_detection_methods
import breathmetrics.kernel_onset_detection_methods
import breathmetrics.kernel_primary
import breathmetrics.utils

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


CAPABILITY_AIRFLOW = "AIRFLOW"
CAPABILITY_NON_AIRFLOW = "NON_AIRFLOW"

CAPABILITY_BY_DATATYPE: dict[str, str] = {
    "humanAirflow": CAPABILITY_AIRFLOW,
    "rodentAirflow": CAPABILITY_AIRFLOW,
    "humanBB": CAPABILITY_NON_AIRFLOW,
    "rodentThermocouple": CAPABILITY_NON_AIRFLOW,
}


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: str  # "event_index", "per_breath", "per_sample", "scalar"
    supported_by: frozenset[str]
    dtype: type
    missing_value: Any


FEATURE_SPECS: list[FeatureSpec] = [
    FeatureSpec(
        name="raw_respiration",
        kind="per_sample",
        supported_by=frozenset({CAPABILITY_AIRFLOW, CAPABILITY_NON_AIRFLOW}),
        dtype=float,
        missing_value=np.nan,
    ),
    FeatureSpec(
        name="smoothed_respiration",
        kind="per_sample",
        supported_by=frozenset({CAPABILITY_AIRFLOW, CAPABILITY_NON_AIRFLOW}),
        dtype=float,
        missing_value=np.nan,
    ),
    FeatureSpec(
        name="baseline_corrected_respiration",
        kind="per_sample",
        supported_by=frozenset({CAPABILITY_AIRFLOW, CAPABILITY_NON_AIRFLOW}),
        dtype=float,
        missing_value=np.nan,
    ),
    FeatureSpec(
        name="signal_peaks",
        kind="event_index",
        supported_by=frozenset({CAPABILITY_AIRFLOW, CAPABILITY_NON_AIRFLOW}),
        dtype=int,
        missing_value=breathmetrics.utils.MISSING_EVENT,
    ),
    FeatureSpec(
        name="signal_troughs",
        kind="event_index",
        supported_by=frozenset({CAPABILITY_AIRFLOW, CAPABILITY_NON_AIRFLOW}),
        dtype=int,
        missing_value=breathmetrics.utils.MISSING_EVENT,
    ),
    FeatureSpec(
        name="inhale_onsets",
        kind="event_index",
        supported_by=frozenset({CAPABILITY_AIRFLOW, CAPABILITY_NON_AIRFLOW}),
        dtype=int,
        missing_value=breathmetrics.utils.MISSING_EVENT,
    ),
    FeatureSpec(
        name="exhale_onsets",
        kind="event_index",
        supported_by=frozenset({CAPABILITY_AIRFLOW, CAPABILITY_NON_AIRFLOW}),
        dtype=int,
        missing_value=breathmetrics.utils.MISSING_EVENT,
    ),
    FeatureSpec(
        name="inhale_offsets",
        kind="event_index",
        supported_by=frozenset({CAPABILITY_AIRFLOW}),
        dtype=int,
        missing_value=breathmetrics.utils.MISSING_EVENT,
    ),
    FeatureSpec(
        name="exhale_offsets",
        kind="event_index",
        supported_by=frozenset({CAPABILITY_AIRFLOW}),
        dtype=int,
        missing_value=breathmetrics.utils.MISSING_EVENT,
    ),
    FeatureSpec(
        name="inhale_pause_onsets",
        kind="event_index",
        supported_by=frozenset({CAPABILITY_AIRFLOW}),
        dtype=int,
        missing_value=breathmetrics.utils.MISSING_EVENT,
    ),
    FeatureSpec(
        name="exhale_pause_onsets",
        kind="event_index",
        supported_by=frozenset({CAPABILITY_AIRFLOW}),
        dtype=int,
        missing_value=breathmetrics.utils.MISSING_EVENT,
    ),
    FeatureSpec(
        name="inhale_peaks",
        kind="event_index",
        supported_by=frozenset({CAPABILITY_AIRFLOW}),
        dtype=int,
        missing_value=breathmetrics.utils.MISSING_EVENT,
    ),
    FeatureSpec(
        name="exhale_troughs",
        kind="event_index",
        supported_by=frozenset({CAPABILITY_AIRFLOW}),
        dtype=int,
        missing_value=breathmetrics.utils.MISSING_EVENT,
    ),
    FeatureSpec(
        name="inhale_durations",
        kind="per_breath",
        supported_by=frozenset({CAPABILITY_AIRFLOW}),
        dtype=float,
        missing_value=np.nan,
    ),
    FeatureSpec(
        name="exhale_durations",
        kind="per_breath",
        supported_by=frozenset({CAPABILITY_AIRFLOW}),
        dtype=float,
        missing_value=np.nan,
    ),
    FeatureSpec(
        name="inhale_pause_durations",
        kind="per_breath",
        supported_by=frozenset({CAPABILITY_AIRFLOW}),
        dtype=float,
        missing_value=np.nan,
    ),
    FeatureSpec(
        name="exhale_pause_durations",
        kind="per_breath",
        supported_by=frozenset({CAPABILITY_AIRFLOW}),
        dtype=float,
        missing_value=np.nan,
    ),
    FeatureSpec(
        name="inhale_volumes",
        kind="per_breath",
        supported_by=frozenset({CAPABILITY_AIRFLOW}),
        dtype=float,
        missing_value=np.nan,
    ),
    FeatureSpec(
        name="exhale_volumes",
        kind="per_breath",
        supported_by=frozenset({CAPABILITY_AIRFLOW}),
        dtype=float,
        missing_value=np.nan,
    ),
    FeatureSpec(
        name="inhale_time2peak",
        kind="per_breath",
        supported_by=frozenset({CAPABILITY_AIRFLOW}),
        dtype=float,
        missing_value=np.nan,
    ),
    FeatureSpec(
        name="exhale_time2trough",
        kind="per_breath",
        supported_by=frozenset({CAPABILITY_AIRFLOW}),
        dtype=float,
        missing_value=np.nan,
    ),
    FeatureSpec(
        name="secondary_features",
        kind="scalar",
        supported_by=frozenset({CAPABILITY_AIRFLOW, CAPABILITY_NON_AIRFLOW}),
        dtype=float,
        missing_value=None,
    ),
]

FEATURE_REGISTRY: dict[str, FeatureSpec] = {spec.name: spec for spec in FEATURE_SPECS}


class Breathe:  # this might be too cute. Consider changing back to class BreathMetrics:
    """
    Stateful breathing-analysis pipeline.

    Initializes with a respiration signal and incrementally
    detects, parameterizes, and edits breathing features.

    """

    # core
    srate: float
    datatype: str
    capability: str
    time: FloatArray

    supported_features: set[str]
    statuses: dict[str, str]
    computed_features: set[str]

    feature_estimations_complete: bool
    features_manually_edited: bool
    is_valid: Optional[BoolArray]

    # per-sample
    raw_respiration: Optional[FloatArray]
    smoothed_respiration: Optional[FloatArray]
    baseline_corrected_respiration: Optional[FloatArray]

    # event indices
    signal_peaks: Optional[IntArray]
    signal_troughs: Optional[IntArray]
    inhale_onsets: Optional[IntArray]
    exhale_onsets: Optional[IntArray]
    inhale_offsets: Optional[IntArray]
    exhale_offsets: Optional[IntArray]
    inhale_pause_onsets: Optional[IntArray]
    exhale_pause_onsets: Optional[IntArray]
    inhale_peaks: Optional[IntArray]
    exhale_troughs: Optional[IntArray]

    # per-breath floats
    inhale_durations: Optional[FloatArray]
    exhale_durations: Optional[FloatArray]
    inhale_pause_durations: Optional[FloatArray]
    exhale_pause_durations: Optional[FloatArray]
    inhale_volumes: Optional[FloatArray]
    exhale_volumes: Optional[FloatArray]
    inhale_time2peak: Optional[FloatArray]
    exhale_time2trough: Optional[FloatArray]

    # optional derived vectors
    peak_inspiratory_flows: Optional[FloatArray]
    trough_expiratory_flows: Optional[FloatArray]

    # secondary (note: not float!)
    secondary_features: Optional[Mapping[str, float]]

    # GUI handle
    _inspect_window: Any
    ## method definition starts here

    def __init__(self, signal: ArrayLike, srate: float, datatype: str):
        signal = np.asarray(signal, dtype=float)
        resp, srate, datatype = breathmetrics.utils.check_input(signal, srate, datatype)
        self.srate = srate
        self.datatype = datatype
        self.capability = CAPABILITY_BY_DATATYPE[self.datatype]

        self.supported_features = {
            spec.name for spec in FEATURE_SPECS if self.capability in spec.supported_by
        }
        self.statuses: dict[str, str] = {}
        self.computed_features: set[str] = set()

        for spec in FEATURE_SPECS:
            setattr(self, spec.name, None)
            if spec.name in self.supported_features:
                self.statuses[spec.name] = "missing"
            else:
                self.statuses[spec.name] = "unsupported"

        self.feature_estimations_complete = False
        self.features_manually_edited = False
        self.is_valid = None

        # seed core signals
        self._set_feature("raw_respiration", resp, status="computed")

        # get smooth window
        if self.datatype == "humanAirflow":
            smoothwin = 50
        elif self.datatype == "humanBB":
            smoothwin = 50
            print(
                "Notice: Only certain features can be derived from breathing belt data"
            )
        elif self.datatype == "rodentThermocouple":
            smoothwin = 10
            print(
                "Notice: Only certain features can be derived from rodent thermocouple data"
            )
        elif self.datatype == "rodentAirflow":
            smoothwin = 10
        else:
            smoothwin = 50

        corrected_smooth_window = np.floor((self.srate / 1000) * smoothwin)
        smoothed = breathmetrics.utils.fft_smooth(
            resp, corrected_smooth_window.astype(int)
        )
        self._set_feature("smoothed_respiration", smoothed, status="computed")
        self.time = np.arange(
            1, len(resp) / self.srate, 1 / self.srate, dtype=np.float64
        )

    # ----- feature registry helpers -----
    def supports(self, feature_name: str) -> bool:
        return feature_name in self.supported_features

    def available_features(self) -> list[str]:
        return [
            spec.name
            for spec in FEATURE_SPECS
            if self.statuses.get(spec.name) in {"computed", "edited"}
        ]

    def missing_features(self) -> list[str]:
        return [
            spec.name
            for spec in FEATURE_SPECS
            if self.statuses.get(spec.name) == "missing"
        ]

    def _set_feature(self, name: str, value: Any, status: str = "computed") -> None:
        if name not in FEATURE_REGISTRY:
            raise KeyError(f"Unknown feature: {name}")
        if status not in {"unsupported", "missing", "computed", "edited"}:
            raise ValueError(f"Invalid status: {status}")

        if not self.supports(name):
            setattr(self, name, None)
            self.statuses[name] = "unsupported"
            self.computed_features.discard(name)
            return

        if value is None:
            setattr(self, name, None)
            self.statuses[name] = "missing"
            self.computed_features.discard(name)
            return

        spec = FEATURE_REGISTRY[name]
        if spec.kind == "event_index":
            val = breathmetrics.utils.normalize_event_array(value)
        elif spec.kind in {"per_breath", "per_sample"}:
            val = np.asarray(value, dtype=float)
        else:
            val = value
            if spec.dtype is float and not isinstance(value, dict):
                try:
                    val = float(value)
                except (TypeError, ValueError):
                    val = value

        setattr(self, name, val)
        self.statuses[name] = status
        if status in {"computed", "edited"}:
            self.computed_features.add(name)
        else:
            self.computed_features.discard(name)

    def check_feature_estimations(
        self, *, verbose: bool = False, ignore: set[str] | None = None
    ) -> bool:
        ignore = ignore or set()
        missing = [
            name
            for name in self.supported_features
            if self.statuses.get(name) == "missing" and name not in ignore
        ]
        if missing and verbose:
            print(f"Missing supported features: {missing}")
        return len(missing) == 0

    @property
    def bsl_corrected_respiration(self) -> np.ndarray | None:
        return self.baseline_corrected_respiration

    @bsl_corrected_respiration.setter
    def bsl_corrected_respiration(self, value: Any) -> None:
        self._set_feature("baseline_corrected_respiration", value, status="computed")

    def correct_resp_to_baseline(self):
        resp = cast(
            FloatArray, self.smoothed_respiration
        )  # This is runtime redundant but keeps type checkers happy

        baseline = breathmetrics.kernel_primary.correct_respiration_to_baseline(
            resp, self.srate
        )
        self._set_feature("baseline_corrected_respiration", baseline, status="computed")

    ## feature extraction
    def find_extrema(self):
        if self.baseline_corrected_respiration is None:
            raise ValueError("baseline_corrected_respiration is missing.")
        peaks, troughs = breathmetrics.kernel_primary.find_respiratory_extrema(
            self.baseline_corrected_respiration, self.srate, datatype=self.datatype
        )
        self._set_feature("signal_peaks", peaks)
        self._set_feature("signal_troughs", troughs)
        if self.capability == CAPABILITY_AIRFLOW:
            self._set_feature("inhale_peaks", peaks)
            self._set_feature("exhale_troughs", troughs)

    def find_onsets_and_pauses(self):
        self._require_features(
            "baseline_corrected_respiration",
            "inhale_peaks",
            "exhale_troughs",
        )
        (
            inhale_onsets,
            exhale_onsets,
            inhale_pause_onsets,
            exhale_pause_onsets,
        ) = breathmetrics.kernel_onset_detection_methods.find_onsets_and_pauses_legacy(
            cast(FloatArray, self.baseline_corrected_respiration),
            cast(IntArray, self.inhale_peaks),
            cast(IntArray, self.exhale_troughs),
        )
        self._set_feature("inhale_onsets", inhale_onsets)
        self._set_feature("exhale_onsets", exhale_onsets)
        self._set_feature("inhale_pause_onsets", inhale_pause_onsets)
        self._set_feature("exhale_pause_onsets", exhale_pause_onsets)

    # new method for inhale onset and exhale onset detection. rely on previous output.
    def find_inhale_onsets_new(self):
        self._require_features("baseline_corrected_respiration", "inhale_peaks")
        inhale_onsets = breathmetrics.kernel_onset_detection_methods.find_onsets_new(
            cast(FloatArray, self.baseline_corrected_respiration),
            self.srate,
            cast(IntArray, self.inhale_peaks),
        )
        self._set_feature("inhale_onsets", inhale_onsets)

    def find_pause_slope(self):
        self._require_features(
            "baseline_corrected_respiration", "inhale_onsets", "exhale_troughs"
        )
        exhale_pause_onsets = (
            breathmetrics.kernel_onset_detection_methods.find_pause_slope_vectorized(
                cast(FloatArray, self.baseline_corrected_respiration),
                self.srate,
                cast(IntArray, self.inhale_onsets),
                cast(IntArray, self.exhale_troughs),
            )
        )
        self._set_feature("exhale_pause_onsets", exhale_pause_onsets)

    def find_time2peaktrough(self):
        self._require_features(
            "inhale_onsets", "inhale_peaks", "exhale_onsets", "exhale_troughs"
        )
        inhale_time2peak, exhale_time2trough = (
            breathmetrics.kernel_primary.find_time_to_peak_trough(
                cast(IntArray, self.inhale_onsets),
                cast(IntArray, self.inhale_peaks),
                cast(IntArray, self.exhale_onsets),
                cast(IntArray, self.exhale_troughs),
                self.srate,
            )
        )
        self._set_feature("inhale_time2peak", inhale_time2peak)
        self._set_feature("exhale_time2trough", exhale_time2trough)

    def find_respiratory_offsets(self):
        self._require_features(
            "baseline_corrected_respiration",
            "inhale_onsets",
            "exhale_onsets",
            "inhale_pause_onsets",
            "exhale_pause_onsets",
        )
        inhale_offsets, exhale_offsets = (
            breathmetrics.kernel_primary.find_respiratory_offsets(
                cast(FloatArray, self.baseline_corrected_respiration),
                cast(IntArray, self.inhale_onsets),
                cast(IntArray, self.exhale_onsets),
                cast(IntArray, self.inhale_pause_onsets),
                cast(IntArray, self.exhale_pause_onsets),
            )
        )
        self._set_feature("inhale_offsets", inhale_offsets)
        self._set_feature("exhale_offsets", exhale_offsets)

    def find_resp_durations(self):
        self._require_features(
            "inhale_onsets",
            "inhale_offsets",
            "exhale_onsets",
            "exhale_offsets",
            "inhale_pause_onsets",
            "exhale_pause_onsets",
        )
        (
            inhale_durations,
            exhale_durations,
            inhale_pause_durations,
            exhale_pause_durations,
        ) = breathmetrics.kernel_primary.find_respiratory_durations(
            cast(IntArray, self.inhale_onsets),
            cast(IntArray, self.inhale_offsets),
            cast(IntArray, self.exhale_onsets),
            cast(IntArray, self.exhale_offsets),
            cast(IntArray, self.inhale_pause_onsets),
            cast(IntArray, self.exhale_pause_onsets),
            self.srate,
        )
        self._set_feature("inhale_durations", inhale_durations)
        self._set_feature("exhale_durations", exhale_durations)
        self._set_feature("inhale_pause_durations", inhale_pause_durations)
        self._set_feature("exhale_pause_durations", exhale_pause_durations)

    def find_resp_volume(self):
        self._require_features(
            "baseline_corrected_respiration",
            "inhale_onsets",
            "inhale_offsets",
            "exhale_onsets",
            "exhale_offsets",
        )
        inhale_volumes, exhale_volumes = (
            breathmetrics.kernel_primary.find_respiratory_volume(
                cast(FloatArray, self.baseline_corrected_respiration),
                cast(IntArray, self.inhale_onsets),
                cast(IntArray, self.inhale_offsets),
                cast(IntArray, self.exhale_onsets),
                cast(IntArray, self.exhale_offsets),
                self.srate,
            )
        )
        self._set_feature("inhale_volumes", inhale_volumes)
        self._set_feature("exhale_volumes", exhale_volumes)

    def _infer_onsets_from_extrema(self) -> None:
        if self.signal_peaks is None or self.signal_troughs is None:
            raise ValueError("signal_peaks/troughs are missing. Run find_extrema().")
        peaks = breathmetrics.utils.normalize_event_array(self.signal_peaks)
        troughs = breathmetrics.utils.normalize_event_array(self.signal_troughs)

        if self.datatype == "humanBB":
            inhale_onsets = troughs
            exhale_onsets = peaks
        elif self.datatype == "rodentThermocouple":
            inhale_onsets = peaks
            exhale_onsets = troughs
        else:
            raise ValueError(f"Non-airflow inference not supported for {self.datatype}")

        inhale_onsets = inhale_onsets[inhale_onsets >= 0]
        exhale_onsets = exhale_onsets[exhale_onsets >= 0]

        paired_inh: list[int] = []
        paired_exh: list[int] = []
        j = 0
        for inh in inhale_onsets.tolist():
            while j < exhale_onsets.size and exhale_onsets[j] <= inh:
                j += 1
            if j >= exhale_onsets.size:
                break
            paired_inh.append(int(inh))
            paired_exh.append(int(exhale_onsets[j]))
            j += 1

        inhale_onsets = np.asarray(paired_inh, dtype=int)
        exhale_onsets = np.asarray(paired_exh, dtype=int)

        self._set_feature("inhale_onsets", inhale_onsets)
        self._set_feature("exhale_onsets", exhale_onsets)

    ## secondary features TODO: the whole thing needs test

    def get_secondary_respiratory_features(
        self, verbose: bool = False
    ) -> dict[str, float]:
        """
        Compute secondary (summary-level) respiratory statistics from a BreathMetrics object.

        Parameters
        ----------
        bm : BreathMetrics
            Object containing basic breathing features (onsets, volumes, durations, etc.)
        verbose : bool, optional
            Whether to print summary output.

        Returns
        -------
        dict[str, float]
            Dictionary of derived secondary features.
        """

        from breathmetrics.kernel_secondary import (
            compute_breath_timing_metrics,
            compute_airflow_metrics,
            compute_duty_cycle_metrics,
            assemble_respiratory_summary,
        )
        from breathmetrics.utils import get_valid_breath_indices

        if self.inhale_onsets is None or self.exhale_onsets is None:
            raise ValueError("inhale_onsets/exhale_onsets are missing.")

        # 1. Identify valid breaths
        valid_inhales, valid_exhales = get_valid_breath_indices(
            self.is_valid, len(self.inhale_onsets), len(self.exhale_onsets)
        )

        # 2. Compute breathing rate, IBI, variability
        timing = compute_breath_timing_metrics(
            self.inhale_onsets, valid_inhales, self.srate
        )

        # 3. Optional airflow features
        airflow: dict[str, float] = {}
        duty: dict[str, float] = {}
        if self.capability == CAPABILITY_AIRFLOW:
            airflow_keys = [
                "avg_inhale_flow",
                "avg_exhale_flow",
                "avg_inhale_volume",
                "avg_exhale_volume",
                "avg_tidal_volume",
                "minute_ventilation",
                "cv_tidal_volume",
            ]
            duty_keys = [
                "inhale_dc",
                "exhale_dc",
                "inhale_pause_dc",
                "exhale_pause_dc",
                "cv_inhale_dur",
                "cv_inhale_pause",
                "cv_exhale_dur",
                "cv_exhale_pause",
                "avg_inhale_dur",
                "avg_exhale_dur",
                "avg_inhale_pause",
                "avg_exhale_pause",
                "pct_inhale_pause",
                "pct_exhale_pause",
            ]

            can_airflow = (
                self.inhale_volumes is not None and self.exhale_volumes is not None
            )
            can_duty = (
                self.inhale_durations is not None
                and self.exhale_durations is not None
                and self.inhale_pause_durations is not None
                and self.exhale_pause_durations is not None
            )

            if can_airflow and can_duty:
                if getattr(self, "peak_inspiratory_flows", None) is None:
                    if (
                        self.baseline_corrected_respiration is not None
                        and self.inhale_peaks is not None
                    ):
                        resp = self.baseline_corrected_respiration
                        peaks = breathmetrics.utils.normalize_event_array(
                            self.inhale_peaks
                        )
                        flows = np.full(peaks.shape[0], np.nan, dtype=float)
                        valid = (peaks >= 0) & (peaks < resp.shape[0])
                        flows[valid] = resp[peaks[valid]]
                        self.peak_inspiratory_flows = flows
                    else:
                        self.peak_inspiratory_flows = np.full(
                            len(self.inhale_onsets), np.nan, dtype=float
                        )
                if getattr(self, "trough_expiratory_flows", None) is None:
                    if (
                        self.baseline_corrected_respiration is not None
                        and self.exhale_troughs is not None
                    ):
                        resp = self.baseline_corrected_respiration
                        troughs = breathmetrics.utils.normalize_event_array(
                            self.exhale_troughs
                        )
                        flows = np.full(troughs.shape[0], np.nan, dtype=float)
                        valid = (troughs >= 0) & (troughs < resp.shape[0])
                        flows[valid] = resp[troughs[valid]]
                        self.trough_expiratory_flows = flows
                    else:
                        self.trough_expiratory_flows = np.full(
                            len(self.exhale_onsets), np.nan, dtype=float
                        )

                airflow = compute_airflow_metrics(self, valid_inhales, valid_exhales)
                duty = compute_duty_cycle_metrics(
                    self, valid_inhales, valid_exhales, timing["inter_breath_interval"]
                )
            else:
                airflow = {k: np.nan for k in airflow_keys}
                duty = {k: np.nan for k in duty_keys}

        # 4. Assemble everything into a summary dict
        summary = assemble_respiratory_summary(self.datatype, timing, airflow, duty)

        # 5. Optional printing
        if verbose:
            print("\nSecondary Respiratory Features:")
            for k, v in summary.items():
                print(f"{k:40s}: {v:.5g}")

        return summary

    ## ERPS
    def compute_erp(
        self,
        event_array: ArrayLike | None = None,
        *,
        pre_s: float = 1.0,
        post_s: float = 1.0,
        append_nans: bool = False,
        verbose: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute an ERP-like matrix aligned to respiratory events.

        Parameters
        ----------
        event_array : array-like | None
            Event indices (sample indices) to align to. If None, uses inhale_onsets.
        pre_s : float
            Seconds before each event to include.
        post_s : float
            Seconds after each event to include.
        append_nans : bool
            If True, pad out-of-bounds segments with NaN instead of rejecting.
        verbose : bool
            If True, prints details for rejected events.
        """
        if self.baseline_corrected_respiration is None:
            raise ValueError("baseline_corrected_respiration is missing.")
        if event_array is None:
            if self.inhale_onsets is None:
                raise ValueError("event_array is None and inhale_onsets is missing.")
            event_array = self.inhale_onsets

        pre_samples = int(round(float(pre_s) * float(self.srate)))
        post_samples = int(round(float(post_s) * float(self.srate)))
        if pre_samples < 0 or post_samples < 0:
            raise ValueError("pre_s and post_s must be non-negative.")

        (
            ERPMatrix,
            trial_events,
            rejected_events,
            trial_event_inds,
            rejected_event_inds,
        ) = breathmetrics.kernel_primary.create_respiratory_erp_matrix(
            self.baseline_corrected_respiration,
            np.asarray(event_array),
            pre_samples,
            post_samples,
            append_nans=append_nans,
            verbose=verbose,
        )

        self.ERPMatrix = ERPMatrix
        self.ERPxAxis = np.arange(-pre_samples, post_samples + 1, dtype=float) / float(
            self.srate
        )
        self.ERPtrialEvents = trial_events
        self.ERPrejectedEvents = rejected_events
        self.ERPtrialEventInds = trial_event_inds
        self.ERPrejectedEventInds = rejected_event_inds

        return (
            ERPMatrix,
            trial_events,
            rejected_events,
            trial_event_inds,
            rejected_event_inds,
        )

    ## esitmate all features. call all methods in order.
    def estimate_all_features(
        self, *, compute_secondary: bool = False, verbose: bool = False
    ) -> None:
        """
        Run the full BreathMetrics feature estimation pipeline in the correct order.
        Sets `feature_estimations_complete` when finished.

        Parameters
        ----------
        compute_secondary : bool
            If True, also compute and store secondary summary features.
        verbose : bool
            If True, print simple progress messages.
        """
        if verbose:
            print("BreathMetrics: estimating all features...")

        # 0) Preconditions
        if (
            not hasattr(self, "smoothed_respiration")
            or self.smoothed_respiration is None
        ):
            raise ValueError(
                "smoothed_respiration is missing. Did __init__ finish preprocessing?"
            )

        # 1) Baseline correction
        self.correct_resp_to_baseline()

        # 2) Extrema
        self.find_extrema()

        if self.capability == CAPABILITY_AIRFLOW:
            # 3) Onsets/offsets/pauses
            self.find_onsets_and_pauses()
            self.find_inhale_onsets_new()
            self.find_pause_slope()
            self.find_respiratory_offsets()

            self.is_valid = (
                np.ones(len(self.inhale_onsets), dtype=bool)
                if self.inhale_onsets is not None
                else None
            )

            # 4) Durations
            self.find_resp_durations()
            self.find_time2peaktrough()

            # 5) Volume
            self.find_resp_volume()
        else:
            # Non-airflow: infer onsets from extrema only
            self._infer_onsets_from_extrema()
            self.is_valid = (
                np.ones(len(self.inhale_onsets), dtype=bool)
                if self.inhale_onsets is not None
                else None
            )

        # 6) Secondary summary (optional)
        if compute_secondary:
            secondary = self.get_secondary_respiratory_features(verbose=verbose)
            self._set_feature("secondary_features", secondary, status="computed")

        ignore = set()
        if not compute_secondary:
            ignore.add("secondary_features")
        self.feature_estimations_complete = self.check_feature_estimations(
            ignore=ignore
        )
        if verbose:
            print("BreathMetrics: done.")

    def inspect(self) -> None:
        """
        Launch the BreathMetrics inspection / editing GUI for this object.

        Notes
        -----
        - Expects all primary breathing features to be estimated already.
        - Intended for notebook / interactive usage.
        """
        self._ensure_qt_event_loop()

        # --- lightweight validation ---
        required_attrs = (
            "srate",
            "baseline_corrected_respiration",
            "inhale_onsets",
            "exhale_onsets",
        )
        missing = [a for a in required_attrs if not hasattr(self, a)]
        missing = [a for a in missing if self.supports(a)]
        if missing:
            raise RuntimeError(
                "BreathMetrics.inspect() cannot launch the GUI because required "
                f"features are missing: {missing}\n\n"
                "Did you forget to run estimate_all_features()?"
            )

        # --- import here to avoid hard GUI dependency on core ---
        from PyQt6.QtWidgets import QApplication
        from breathmetrics.breathmetrics_gui import BreathMetricsMainWindow

        app = QApplication.instance()

        if app is None:
            app = QApplication([])

        win = BreathMetricsMainWindow(self)
        self._inspect_window = win  # type: ignore[attr-defined]

        win.show()

    def behold(self) -> None:
        """Alias for inspect(), with a tiny flourish. Thank you for using BreathMetrics!"""
        try:
            from IPython.core.getipython import get_ipython

            in_ipython = get_ipython() is not None
        except Exception:
            in_ipython = False

        import sys

        if in_ipython or sys.stdout.isatty():
            print("Tada! 👀✨\nThank you for using BreathMetrics!")

        self.inspect()

    # ----------
    # Internal. 🦊🤖
    # ----------
    def _ensure_qt_event_loop(self) -> None:
        """
        Try to enable Qt event loop integration when running inside IPython.
        Safe no-op in non-IPython contexts.
        """
        try:
            from IPython.core.getipython import get_ipython

            ip = get_ipython()
            if ip is None:
                return

            # Only run if GUI integration isn't already active
            # (running it twice is usually harmless, but be polite)
            ip.run_line_magic("gui", "qt6")
        except Exception:
            # Never let GUI setup crash the user's session
            pass

    def _require_features(self, *names: str) -> None:
        missing = [n for n in names if getattr(self, n) is None]
        if missing:
            raise ValueError(
                "Missing required features: "
                + ", ".join(missing)
                + ". Did you run earlier pipeline steps?"
            )
