from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from dataclasses import dataclass


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
class breathmetrics:
    # signal properties:
    datatype: None
    fs: float  # sampling rate (Hz)
    time: float

    # breathing signal
    rawRespiration: ArrayLike
    smoothedRespiration: np.ndarray
    baselineCorrectedRespiration: np.ndarray

    # calculated features
    inhalePeaks: np.ndarray
    exhaleTroughs: np.ndarray

    peakInspiratoryFlows: np.ndarray
    troughExpiratoryFlows: np.ndarray

    inhaleOnsets: np.ndarray
    exhaleOnsets: np.ndarray

    inhaleOffsets: np.ndarray
    exhaleOffsets: np.ndarray

    #

    # class BreathMetrics:
    """Core breathing-signal parameterization (skeleton)."""

    # def __init__(self, signal: np.ndarray, config: BreathMetricsConfig):
    #     self.signal = np.asarray(signal, dtype=float)
    #     self.config = config
    #     self.features = {}

    # def preprocess(self) -> "BreathMetrics":
    #     x = self.signal
    #     if self.config.detrend:
    #         x = x - np.nanmean(x)
    #     win = max(1, int(self.config.smooth_win_sec * self.config.fs))
    #     if win > 1:
    #         # simple moving average as placeholder
    #         kernel = np.ones(win) / win
    #         x = np.convolve(x, kernel, mode="same")
    #     self._x = x
    #     return self

    # def estimate_features(self) -> "BreathMetrics":
    #     # TODO: replace with actual inhalation/exhalation detection, etc.
    #     self.features["mean"] = float(np.nanmean(getattr(self, "_x", self.signal)))
    #     self.features["std"] = float(np.nanstd(getattr(self, "_x", self.signal)))
    #     return self

    # def to_frame(self):
    #     import pandas as pd

    #     return pd.DataFrame([self.features])
