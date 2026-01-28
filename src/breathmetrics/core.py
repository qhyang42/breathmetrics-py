from __future__ import annotations
from tabnanny import verbose
import numpy as np
from numpy.typing import ArrayLike
from dataclasses import dataclass

# import breathmetrics.kernel
# import breathmetrics.kernel_onset_detection_methods
import breathmetrics.kernel_primary
import breathmetrics.kernel_onset_detection_methods
import breathmetrics.utils


@dataclass
class Breathe:  # this might be too cute. Consider changing back to class BreathMetrics:
    """
    Stateful breathing-analysis pipeline.

    Initializes with a respiration signal and incrementally
    detects, parameterizes, and edits breathing features.

    """

    # signal properties:
    datatype: str
    srate: float  # sampling rate (Hz)
    time: np.ndarray

    # breathing signal
    raw_respiration: np.ndarray
    smoothed_respiration: np.ndarray
    bsl_corrected_respiration: np.ndarray

    # calculated features
    inhale_peaks: np.ndarray
    exhale_troughs: np.ndarray

    peak_inspiratory_flows: np.ndarray
    trough_expiratory_flows: np.ndarray

    inhale_onsets: np.ndarray
    exhale_onsets: np.ndarray

    inhale_offsets: np.ndarray
    exhale_offsets: np.ndarray

    inhale_time2peak: np.ndarray
    exhale_time2trough: np.ndarray

    inhale_volumes: np.ndarray
    exhale_volumes: np.ndarray

    inhale_durations: np.ndarray
    exhale_durations: np.ndarray

    inhale_pause_onsets: np.ndarray
    exhale_pause_onsets: np.ndarray

    inhale_pause_durations: np.ndarray
    exhale_pause_durations: np.ndarray

    # secondaryFeatures: TODO figure out what type this is

    respiratory_phase: np.ndarray

    ##  ERP parameters
    # ERPMatrix
    # ERPxAxis

    # resampledERPMatrix
    # resampledERPxAxis

    # ERPtrialEvents
    # ERPrejectedEvents

    # ERPtrialEventInds
    # ERPrejectedEventInds

    is_valid: np.ndarray | None
    # notes

    feature_estimations_complete: bool
    features_manually_edited: bool

    ## method definition starts here

    def __init__(self, signal: ArrayLike, srate: float, datatype: str):
        signal = np.asarray(signal, dtype=float)
        self.raw_respiration, self.srate, self.datatype = (
            breathmetrics.utils.check_input(signal, srate, datatype)
        )

        self.feature_estimations_complete = False
        self.features_manually_edited = False
        self.is_valid = None

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
        elif (
            self.datatype == "rodentAirflow"
        ):  # gets the same features as human airflow
            smoothwin = 10

        else:
            smoothwin = 50  # TODO this is not quite right. if datatype is illegal, give warning and quit?

        corrected_smooth_window = np.floor((self.srate / 1000) * smoothwin)
        self.smoothed_respiration = breathmetrics.utils.fft_smooth(
            self.raw_respiration, corrected_smooth_window.astype(int)
        )
        self.time = np.arange(1, len(signal) / self.srate, 1 / self.srate)

        # correct resp to baseline

    def correct_resp_to_baseline(self):
        self.bsl_corrected_respiration = (
            breathmetrics.kernel_primary.correct_respiration_to_baseline(
                self.smoothed_respiration, self.srate
            )
        )

    ## feature extraction
    def find_extrema(self):
        self.inhale_peaks, self.exhale_troughs = (
            breathmetrics.kernel_primary.find_respiratory_extrema(
                self.bsl_corrected_respiration, self.srate, datatype=self.datatype
            )
        )

    def find_onsets_and_pauses(self):
        (
            self.inhale_onsets,
            self.exhale_onsets,
            self.inhale_pause_onsets,
            self.exhale_pause_onsets,
        ) = breathmetrics.kernel_onset_detection_methods.find_onsets_and_pauses_legacy(
            self.bsl_corrected_respiration, self.inhale_peaks, self.exhale_troughs
        )

    # new method for inhale onset and exhale onset detection. rely on previous output.
    def find_inhale_onsets_new(self):
        self.inhale_onsets = (
            breathmetrics.kernel_onset_detection_methods.find_onsets_new(
                self.bsl_corrected_respiration, self.srate, self.inhale_peaks
            )
        )

    def find_pause_slope(self):
        self.exhale_pause_onsets = (
            breathmetrics.kernel_onset_detection_methods.find_pause_slope_vectorized(
                self.bsl_corrected_respiration,
                self.srate,
                self.inhale_onsets,
                self.exhale_troughs,
            )
        )

    def find_time2peaktrough(self):
        self.inhale_time2peak, self.exhale_time2trough = (
            breathmetrics.kernel_primary.find_time_to_peak_trough(
                self.inhale_onsets,
                self.inhale_peaks,
                self.exhale_onsets,
                self.exhale_troughs,
                self.srate,
            )
        )

    def find_respiratory_offsets(self):
        (self.inhale_offsets, self.exhale_offsets) = (
            breathmetrics.kernel_primary.find_respiratory_offsets(
                self.bsl_corrected_respiration,
                self.inhale_onsets,
                self.exhale_onsets,
                self.inhale_pause_onsets,
                self.exhale_pause_onsets,
            )
        )

    def find_resp_durations(self):
        (
            self.inhale_durations,
            self.exhale_durations,
            self.inhale_pause_durations,
            self.exhale_pause_durations,
        ) = breathmetrics.kernel_primary.find_respiratory_durations(
            self.inhale_onsets,
            self.inhale_offsets,
            self.exhale_onsets,
            self.exhale_offsets,
            self.inhale_pause_onsets,
            self.exhale_pause_onsets,
            self.srate,
        )

    def find_resp_volume(self):
        self.inhale_volumes, self.exhale_volumes = (
            breathmetrics.kernel_primary.find_respiratory_volume(
                self.bsl_corrected_respiration,
                self.inhale_onsets,
                self.inhale_offsets,
                self.exhale_onsets,
                self.exhale_offsets,
                self.srate,
            )
        )

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

        # 1. Identify valid breaths
        valid_inhales, valid_exhales = get_valid_breath_indices(
            self.is_valid, len(self.inhale_onsets), len(self.exhale_onsets)
        )

        # 2. Compute breathing rate, IBI, variability
        timing = compute_breath_timing_metrics(
            self.inhale_onsets, valid_inhales, self.srate
        )

        # 3. Optional airflow features
        airflow = {}
        duty = {}
        if self.datatype in {"humanAirflow", "rodentAirflow"}:
            airflow = compute_airflow_metrics(self, valid_inhales, valid_exhales)
            duty = compute_duty_cycle_metrics(
                self, valid_inhales, valid_exhales, timing["inter_breath_interval"]
            )

        # 4. Assemble everything into a summary dict
        summary = assemble_respiratory_summary(self.datatype, timing, airflow, duty)

        # 5. Optional printing
        if verbose:
            print("\nSecondary Respiratory Features:")
            for k, v in summary.items():
                print(f"{k:40s}: {v:.5g}")

        return summary

    ## ERPS
    def compute_erp(self):
        # check how the matlab toolbox handle this.
        return None

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

        # 3) Onsets/offsets/pauses
        self.find_onsets_and_pauses()

        # call the new methods here.
        self.find_inhale_onsets_new()
        self.find_pause_slope()

        self.find_respiratory_offsets()
        self.inhale_peaks = breathmetrics.utils.normalize_event_array(self.inhale_peaks)
        self.exhale_troughs = breathmetrics.utils.normalize_event_array(
            self.exhale_troughs
        )
        self.inhale_onsets = breathmetrics.utils.normalize_event_array(
            self.inhale_onsets
        )
        self.exhale_onsets = breathmetrics.utils.normalize_event_array(
            self.exhale_onsets
        )
        self.inhale_offsets = breathmetrics.utils.normalize_event_array(
            self.inhale_offsets
        )
        self.exhale_offsets = breathmetrics.utils.normalize_event_array(
            self.exhale_offsets
        )
        self.inhale_pause_onsets = breathmetrics.utils.normalize_event_array(
            self.inhale_pause_onsets
        )
        self.exhale_pause_onsets = breathmetrics.utils.normalize_event_array(
            self.exhale_pause_onsets
        )
        self.is_valid = np.ones(len(self.inhale_onsets), dtype=bool)

        # 4) Durations
        self.find_resp_durations()
        self.find_time2peaktrough()

        # 5) Volume (if applicable / desired)
        # Some datatypes may not support it—up to you if you want guards here.
        self.find_resp_volume()

        # 6) Secondary summary (optional)
        if compute_secondary:
            self.secondary_features = self.get_secondary_respiratory_features(
                verbose=verbose
            )

        self.feature_estimations_complete = True

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
            "bsl_corrected_respiration",
            "inhale_onsets",
            "exhale_onsets",
            "exhale_offsets",
            "inhale_pause_onsets",
            "exhale_pause_onsets",
        )

        missing = [a for a in required_attrs if not hasattr(self, a)]
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
    # Internal. For ipython qt event loop integration 🦊🤖
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
