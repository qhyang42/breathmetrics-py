from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from dataclasses import dataclass

# import breathmetrics.kernel
# import breathmetrics.kernel_onset_detection_methods
import breathmetrics.kernel
import breathmetrics.kernel_onset_detection_methods
import breathmetrics.utils


## TODO this needs to be built block by block. no clusterfuck dumping. keep things clean.

## required data attributes. same as orig breathmetrics
#  % signal properties
#         dataType
#         srate
#         time

#         % breathing signal
#         rawRespiration
#         smoothedRespiration
#         baselineCorrectedRespiration

#         % calculated features
#         inhalePeaks
#         exhaleTroughs

#         peakInspiratoryFlows
#         troughExpiratoryFlows

#         inhaleOnsets
#         exhaleOnsets

#         inhaleOffsets
#         exhaleOffsets

#         inhaleTimeToPeak
#         exhaleTimeToTrough

#         inhaleVolumes
#         exhaleVolumes

#         inhaleDurations
#         exhaleDurations

#         inhalePauseOnsets
#         exhalePauseOnsets

#         inhalePauseDurations
#         exhalePauseDurations

#         secondaryFeatures

#         respiratoryPhase

#         % erp params
#         ERPMatrix
#         ERPxAxis

#         resampledERPMatrix
#         resampledERPxAxis

#         ERPtrialEvents
#         ERPrejectedEvents

#         ERPtrialEventInds
#         ERPrejectedEventInds

#         statuses
#         notes


#         featureEstimationsComplete
#         featuresManuallyEdited


@dataclass
class bm:
    """
    Breathmetrics class.

    """

    # signal properties:
    datatype: str
    fs: float  # sampling rate (Hz)
    time: np.ndarray

    # breathing signal
    raw_respiration: np.ndarray
    smoothed_respitation: np.ndarray
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

    statuses: None
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

            corrected_smooth_window = np.floor((self.srate / 1000) * smoothwin)
            self.smoothed_respitation = breathmetrics.utils.fft_smooth(
                self.raw_respiration, corrected_smooth_window
            )
            self.time = np.arange(1, len(signal) / self.srate, 1 / self.srate)

        # correct resp to baseline

    def correct_resp_to_baseline(self):
        self.bsl_corrected_respiration = (
            breathmetrics.kernel.correct_respiration_to_baseline(
                self.smoothed_respitation, self.fs
            )
        )

    ## feature extraction
    def find_extrema(self):
        self.inhale_peaks, self.exhale_troughs = (
            breathmetrics.kernel.find_respiratory_extrema(
                self.bsl_corrected_respiration, self.srate
            )
        )

    def find_onsets_and_pauses(self):
        (
            self.inhale_onsets,
            self.exhale_onsets,
            self.inhale_offsets,
            self.exhale_offsets,
        ) = breathmetrics.kernel_onset_detection_methods.find_onsets_and_pauses_legacy(
            self.bsl_corrected_respiration, self.inhale_peaks, self.exhale_troughs
        )

    def find_resp_durations(self):
        (
            self.inhale_durations,
            self.exhale_durations,
            self.inhale_pause_durations,
            self.exhale_pause_durations,
        ) = breathmetrics.kernel.find_respiratory_durations(
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
            breathmetrics.kernel.find_respiratory_volume(
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
            get_valid_breath_indices,
            compute_breath_timing_metrics,
            compute_airflow_metrics,
            compute_duty_cycle_metrics,
            assemble_respiratory_summary,
        )

        # 1. Identify valid breaths
        valid_inhales, valid_exhales = get_valid_breath_indices(
            self.statuses, len(self.inhale_onsets), len(self.exhale_onsets)
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
