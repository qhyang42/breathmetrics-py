from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, cast

import numpy as np
from numpy.typing import NDArray

from breathmetrics.utils import MISSING_EVENT, normalize_event_array


# ---------- typing ----------
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


class EventType(str, Enum):
    INHALE_ONSET = "inhale_onset"
    EXHALE_ONSET = "exhale_onset"
    INHALE_PAUSE_ONSET = "inhale_pause_onset"
    EXHALE_PAUSE_ONSET = "exhale_pause_onset"


@dataclass(frozen=True, slots=True)
class EditResult:
    """
    What the GUI/controller needs after an edit.
    """

    breath_i: int
    event: EventType
    requested_sample: int
    applied_sample: int
    was_clamped: bool
    created: bool  # True if the event was missing and now set
    changed_fields: tuple[str, ...]  # bm attributes changed by this call


@dataclass(frozen=True, slots=True)
class _UndoRecord:
    breath_i: int
    event: EventType
    # old values (needed to restore)
    old_inhale_onset: float
    old_exhale_onset: float
    old_inhale_pause_onset: float
    old_exhale_pause_onset: float
    # also restore recomputed features (optional but makes undo exact)
    old_inhale_time2peak: float
    old_inhale_duration: float
    old_inhale_volume: float
    old_exhale_time2trough: float
    old_exhale_duration: float
    old_exhale_volume: float


def _as_float_array(x: np.ndarray) -> FloatArray:
    # Pylance-friendly cast helper
    return cast(FloatArray, np.asarray(x, dtype=np.float64))


def _as_int_array(x: np.ndarray) -> IntArray:
    return cast(IntArray, np.asarray(x, dtype=np.int64))


def _finite(x: float) -> bool:
    return bool(np.isfinite(x)) and x >= 0


def _safe_int(x: float) -> int:
    # round to nearest sample index
    return int(np.rint(x))


def _clamp_int(x: int, lo: int, hi: int) -> int:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _nan_to_none(x: float) -> Optional[float]:
    return None if (not np.isfinite(x)) or x < 0 else float(x)


class BreathEditor:
    """
    BreathEditor mutates a bm instance in a controlled way:
    - clamps event moves to valid ranges
    - prevents event ordering violations
    - recomputes dependent features (local breath, optionally neighbors)
    - maintains undo history
    """

    def __init__(
        self,
        model: object,
        *,
        resp_attr: str = "baseline_corrected_respiration",
        fs_attr: str = "srate",
        # optional: if your model ever uses a different attribute name later
        is_valid_attr: str = "is_valid",
    ) -> None:
        self.bm = model
        self._resp_attr = resp_attr
        self._fs_attr = fs_attr
        self._is_valid_attr = is_valid_attr

        self._undo_stack: list[_UndoRecord] = []
        self._dirty_breaths: set[int] = set()

        self._ensure_is_valid()

    # ----------------------------
    # Public API
    # ----------------------------

    @property
    def n_breaths(self) -> int:
        inhale_onsets = self._get_event_array("inhale_onsets")
        return int(inhale_onsets.shape[0])

    @property
    def is_dirty(self) -> bool:
        return bool(self._dirty_breaths)

    def dirty_breaths(self) -> tuple[int, ...]:
        return tuple(sorted(self._dirty_breaths))

    def reject(self, breath_i: int, rejected: bool = True) -> None:
        is_valid = self._get_is_valid()
        self._check_breath_i(breath_i)
        is_valid[breath_i] = not rejected
        self._dirty_breaths.add(breath_i)

    def toggle_reject(self, breath_i: int) -> None:
        is_valid = self._get_is_valid()
        self._check_breath_i(breath_i)
        is_valid[breath_i] = ~is_valid[breath_i]
        self._dirty_breaths.add(breath_i)

    def move_event(
        self, breath_i: int, event: EventType, new_sample: int
    ) -> EditResult:
        """
        Core API for GUI dot-dragging:
        - clamps move
        - writes into bm arrays
        - recomputes dependent features

        new_sample should be a sample index (int).
        """
        self._check_breath_i(breath_i)
        self._check_sample_int(new_sample)

        if not self._supports_event(event):
            return EditResult(
                breath_i=breath_i,
                event=event,
                requested_sample=new_sample,
                applied_sample=new_sample,
                was_clamped=False,
                created=False,
                changed_fields=(),
            )

        # snapshot for undo
        rec = self._make_undo_record(breath_i, event)
        self._undo_stack.append(rec)

        # apply move with clamping + ordering constraints
        applied, was_clamped, created = self._apply_event_move(
            breath_i, event, new_sample
        )

        # recompute dependent features (only what user specified)
        changed_fields = self._recompute_for_event(breath_i, event)

        self._dirty_breaths.add(breath_i)

        self._mark_edited(event, changed_fields)

        return EditResult(
            breath_i=breath_i,
            event=event,
            requested_sample=new_sample,
            applied_sample=applied,
            was_clamped=was_clamped,
            created=created,
            changed_fields=changed_fields,
        )

    def undo_last(self) -> bool:
        """
        Undo the most recent edit (any breath).
        Returns True if something was undone.
        """
        if not self._undo_stack:
            return False
        rec = self._undo_stack.pop()
        self._restore_undo_record(rec)
        self._dirty_breaths.add(rec.breath_i)
        return True

    def undo_breath(self, breath_i: int) -> int:
        """
        Undo all edits for one breath.
        Returns number of undone edits.
        """
        self._check_breath_i(breath_i)
        n = 0
        # pop all records for this breath and restore from the earliest state
        # simplest: undo in reverse order until no more for this breath
        kept: list[_UndoRecord] = []
        while self._undo_stack:
            rec = self._undo_stack.pop()
            if rec.breath_i == breath_i:
                self._restore_undo_record(rec)
                n += 1
            else:
                kept.append(rec)
        # restore other breath edits back onto stack in original order
        self._undo_stack.extend(reversed(kept))

        if n:
            self._dirty_breaths.add(breath_i)
        return n

    def commit(self) -> None:
        """
        Marks the current model state as committed from the editor's POV.
        Does NOT save to disk; it just clears dirty + undo state.
        """
        self._dirty_breaths.clear()
        self._undo_stack.clear()

        # optional: could set a bm flag for your pipeline
        if hasattr(self.bm, "features_manually_edited"):
            setattr(self.bm, "features_manually_edited", True)

    # ----------------------------
    # Internals: model access
    # ----------------------------

    def _ensure_is_valid(self) -> None:
        # Attach is_valid to bm if missing.
        if (
            not hasattr(self.bm, self._is_valid_attr)
            or getattr(self.bm, self._is_valid_attr) is None
        ):
            n = self.n_breaths
            is_valid: BoolArray = np.ones((n,), dtype=np.bool_)
            setattr(self.bm, self._is_valid_attr, is_valid)

    def _get_is_valid(self) -> BoolArray:
        arr = getattr(self.bm, self._is_valid_attr)
        if arr is None:
            self._ensure_is_valid()
            arr = getattr(self.bm, self._is_valid_attr)
        return cast(BoolArray, arr)

    def _get_fs(self) -> float:
        fs = float(getattr(self.bm, self._fs_attr))
        if fs <= 0:
            raise ValueError(f"Invalid fs={fs}")
        return fs

    def _get_resp(self) -> FloatArray:
        resp = getattr(self.bm, self._resp_attr)
        if resp is None:
            raise ValueError(f"bm.{self._resp_attr} is None")
        return _as_float_array(resp)

    def _event_length(self) -> int:
        arr = getattr(self.bm, "inhale_onsets", None)
        if arr is None:
            return 0
        return int(np.asarray(arr).shape[0])

    def _supports(self, feature: str) -> bool:
        supports = getattr(self.bm, "supports", None)
        if callable(supports):
            try:
                return bool(supports(feature))
            except Exception:
                return False
        return getattr(self.bm, feature, None) is not None

    def _supports_event(self, event: EventType) -> bool:
        mapping = {
            EventType.INHALE_ONSET: "inhale_onsets",
            EventType.EXHALE_ONSET: "exhale_onsets",
            EventType.INHALE_PAUSE_ONSET: "inhale_pause_onsets",
            EventType.EXHALE_PAUSE_ONSET: "exhale_pause_onsets",
        }
        return self._supports(mapping[event])

    def _mark_edited(self, event: EventType, changed_fields: tuple[str, ...]) -> None:
        statuses = getattr(self.bm, "statuses", None)
        if not isinstance(statuses, dict):
            return
        mapping = {
            EventType.INHALE_ONSET: "inhale_onsets",
            EventType.EXHALE_ONSET: "exhale_onsets",
            EventType.INHALE_PAUSE_ONSET: "inhale_pause_onsets",
            EventType.EXHALE_PAUSE_ONSET: "exhale_pause_onsets",
        }
        to_mark = [mapping[event], *changed_fields]
        computed = getattr(self.bm, "computed_features", None)
        for name in to_mark:
            if not self._supports(name):
                continue
            statuses[name] = "edited"
            if isinstance(computed, set):
                computed.add(name)

    def _get_event_array(self, attr: str) -> IntArray:
        arr = getattr(self.bm, attr, None)
        if arr is None:
            if not self._supports(attr):
                return cast(
                    IntArray,
                    np.full(self._event_length(), MISSING_EVENT, dtype=np.int64),
                )
            arr = np.full(self._event_length(), MISSING_EVENT, dtype=np.int64)
            setattr(self.bm, attr, arr)
            return cast(IntArray, arr)

        arr = np.asarray(arr)
        if arr.dtype.kind != "i":
            arr = normalize_event_array(arr)
            setattr(self.bm, attr, arr)
        return cast(IntArray, arr)

    # ----------------------------
    # Validation + bounds
    # ----------------------------

    def _check_breath_i(self, breath_i: int) -> None:
        n = self.n_breaths
        if not (0 <= breath_i < n):
            raise IndexError(f"breath_i out of range: {breath_i} (n={n})")

    def _check_sample_int(self, s: int) -> None:
        if not isinstance(s, int):
            raise TypeError("new_sample must be an int sample index")

    def _signal_bounds(self) -> tuple[int, int]:
        n = int(self._get_resp().shape[0])
        return 0, max(0, n - 1)

    def _breath_neighbor_bounds(self, i: int) -> tuple[int, int]:
        """
        Conservative neighbor constraints to avoid cross-breath scrambling:
        inhale_onset[i] should not move before exhale_offset[i-1],
        and should not move after inhale_onset[i+1] (if defined).
        """
        inhale_onsets = self._get_event_array("inhale_onsets")
        if (
            self._supports("exhale_offsets")
            and getattr(self.bm, "exhale_offsets", None) is not None
        ):
            exhale_offsets = self._get_event_array("exhale_offsets")
        else:
            exhale_offsets = self._get_event_array("exhale_onsets")

        sig_lo, sig_hi = self._signal_bounds()

        lo = sig_lo
        if i > 0 and _finite(exhale_offsets[i - 1]):
            lo = max(lo, _safe_int(exhale_offsets[i - 1]) + 1)

        hi = sig_hi
        if i + 1 < inhale_onsets.shape[0] and _finite(inhale_onsets[i + 1]):
            hi = min(hi, _safe_int(inhale_onsets[i + 1]) - 1)

        return lo, hi

    def _phase_effective_offsets(self, i: int) -> tuple[int, int]:
        """
        Effective phase offsets:
        - inhale end is inhale_pause_onset if finite else inhale_offsets[i]
        - exhale end is exhale_pause_onset if finite else exhale_offsets[i]
        """
        inhale_offsets = self._get_event_array("inhale_offsets")
        exhale_offsets = self._get_event_array("exhale_offsets")
        inhale_pause_onsets = self._get_event_array("inhale_pause_onsets")
        exhale_pause_onsets = self._get_event_array("exhale_pause_onsets")

        inhale_end = inhale_offsets[i]
        if _finite(inhale_pause_onsets[i]):
            inhale_end = inhale_pause_onsets[i]

        exhale_end = exhale_offsets[i]
        if _finite(exhale_pause_onsets[i]):
            exhale_end = exhale_pause_onsets[i]

        # Fallback safety if these are NaN (shouldn't be, per your spec)
        if not _finite(inhale_end):
            inhale_end = inhale_offsets[i]
        if not _finite(exhale_end):
            exhale_end = exhale_offsets[i]

        return _safe_int(inhale_end), _safe_int(exhale_end)

    def _ordering_constraints(self, i: int) -> dict[EventType, tuple[int, int]]:
        """
        Returns allowable [lo, hi] for each editable event given current model state.

        Constraints enforced (no scrambling):
        inhale_onset < inhale_peak < exhale_onset < exhale_trough < next_inhale_onset

        Pause rules:
        - inhale_pause_onset between inhale_onset and exhale_onset
        - exhale_pause_onset between exhale_trough and next_inhale_onset
        """
        inhale_onsets = self._get_event_array("inhale_onsets")
        exhale_onsets = self._get_event_array("exhale_onsets")
        if (
            self._supports("inhale_peaks")
            and getattr(self.bm, "inhale_peaks", None) is not None
        ):
            inhale_peaks = self._get_event_array("inhale_peaks")
        elif (
            self._supports("signal_peaks")
            and getattr(self.bm, "signal_peaks", None) is not None
        ):
            inhale_peaks = self._get_event_array("signal_peaks")
        else:
            inhale_peaks = inhale_onsets

        if (
            self._supports("exhale_troughs")
            and getattr(self.bm, "exhale_troughs", None) is not None
        ):
            exhale_troughs = self._get_event_array("exhale_troughs")
        elif (
            self._supports("signal_troughs")
            and getattr(self.bm, "signal_troughs", None) is not None
        ):
            exhale_troughs = self._get_event_array("signal_troughs")
        else:
            exhale_troughs = exhale_onsets

        sig_lo, sig_hi = self._signal_bounds()
        neigh_lo, neigh_hi = self._breath_neighbor_bounds(i)

        inhale_peak = (
            _safe_int(inhale_peaks[i])
            if _finite(inhale_peaks[i])
            else _safe_int(inhale_onsets[i])
        )
        exhale_trough = (
            _safe_int(exhale_troughs[i])
            if _finite(exhale_troughs[i])
            else _safe_int(exhale_onsets[i])
        )

        # current exhale onset might be NaN; treat as inhale_peak+1 minimum
        cur_exhale_onset = exhale_onsets[i]
        exhale_onset_min = inhale_peak + 1
        exhale_onset = (
            _safe_int(cur_exhale_onset)
            if _finite(cur_exhale_onset)
            else exhale_onset_min
        )

        # next inhale onset bound if present
        next_inhale_onset = (
            _safe_int(inhale_onsets[i + 1])
            if (i + 1) < inhale_onsets.shape[0] and _finite(inhale_onsets[i + 1])
            else sig_hi
        )

        # inhale onset bounds: after previous breath end, before inhale_peak-1
        lo_inh_on = max(sig_lo, neigh_lo)
        hi_inh_on = min(
            sig_hi,
            neigh_hi,
            inhale_peak - 1,
            exhale_onset - 1,
            exhale_trough - 1,
            next_inhale_onset - 1,
        )

        # exhale onset bounds: after inhale_peak+1, before exhale_trough-1
        lo_exh_on = max(sig_lo, inhale_peak + 1)
        hi_exh_on = min(sig_hi, exhale_trough - 1, next_inhale_onset - 1)

        # inhale pause onset bounds: between inhale_onset and exhale_onset
        lo_inh_pause = max(sig_lo, _safe_int(inhale_onsets[i]) + 1)
        hi_inh_pause = min(sig_hi, exhale_onset - 1)

        # exhale pause onset bounds: between exhale_trough and next inhale onset
        lo_exh_pause = max(sig_lo, exhale_trough + 1)
        hi_exh_pause = min(sig_hi, next_inhale_onset - 1)

        # clamp any inverted bounds to something safe (so we "clamp to nearest available point")
        def safe_bounds(lo: int, hi: int) -> tuple[int, int]:
            if hi < lo:
                # collapse to a single valid point (nearest available)
                return lo, lo
            return lo, hi

        return {
            EventType.INHALE_ONSET: safe_bounds(lo_inh_on, hi_inh_on),
            EventType.EXHALE_ONSET: safe_bounds(lo_exh_on, hi_exh_on),
            EventType.INHALE_PAUSE_ONSET: safe_bounds(lo_inh_pause, hi_inh_pause),
            EventType.EXHALE_PAUSE_ONSET: safe_bounds(lo_exh_pause, hi_exh_pause),
        }

    # ----------------------------
    # Apply edit + recompute
    # ----------------------------

    def _apply_event_move(
        self, i: int, event: EventType, new_sample: int
    ) -> tuple[int, bool, bool]:
        """
        Returns (applied_sample, was_clamped, created)
        """
        bounds = self._ordering_constraints(i)[event]
        lo, hi = bounds

        sig_lo, sig_hi = self._signal_bounds()
        lo = _clamp_int(lo, sig_lo, sig_hi)
        hi = _clamp_int(hi, sig_lo, sig_hi)

        applied = _clamp_int(new_sample, lo, hi)
        was_clamped = applied != new_sample

        created = False

        if event == EventType.INHALE_ONSET:
            arr = self._get_event_array("inhale_onsets")
            arr[i] = int(applied)
            return applied, was_clamped, created

        if event == EventType.EXHALE_ONSET:
            arr = self._get_event_array("exhale_onsets")
            arr[i] = int(applied)
            return applied, was_clamped, created

        if event == EventType.INHALE_PAUSE_ONSET:
            arr = self._get_event_array("inhale_pause_onsets")
            if not _finite(arr[i]):
                created = True
            arr[i] = int(applied)
            return applied, was_clamped, created

        if event == EventType.EXHALE_PAUSE_ONSET:
            arr = self._get_event_array("exhale_pause_onsets")
            if not _finite(arr[i]):
                created = True
            arr[i] = int(applied)
            return applied, was_clamped, created

        raise ValueError(f"Unsupported event: {event}")

    def _recompute_for_event(self, i: int, event: EventType) -> tuple[str, ...]:
        """
        Recompute only what you specified.

        Inhale recompute triggers:
          - inhale_time2peak, inhale_duration, inhale_volume
        Exhale recompute triggers:
          - exhale_time2trough, exhale_duration, exhale_volume
        """
        if event in (EventType.INHALE_ONSET, EventType.INHALE_PAUSE_ONSET):
            self._recompute_inhale(i)
            return tuple(
                name
                for name in ("inhale_time2peak", "inhale_durations", "inhale_volumes")
                if self._supports(name) and getattr(self.bm, name, None) is not None
            )

        if event in (EventType.EXHALE_ONSET, EventType.EXHALE_PAUSE_ONSET):
            self._recompute_exhale(i)
            return tuple(
                name
                for name in ("exhale_time2trough", "exhale_durations", "exhale_volumes")
                if self._supports(name) and getattr(self.bm, name, None) is not None
            )

        return ()

    def _recompute_inhale(self, i: int) -> None:
        if not (
            self._supports("inhale_time2peak")
            or self._supports("inhale_durations")
            or self._supports("inhale_volumes")
        ):
            return
        fs = self._get_fs()
        resp = self._get_resp()

        inhale_onsets = self._get_event_array("inhale_onsets")
        if (
            self._supports("inhale_peaks")
            and getattr(self.bm, "inhale_peaks", None) is not None
        ):
            inhale_peaks = self._get_event_array("inhale_peaks")
        else:
            inhale_peaks = inhale_onsets
        if (
            self._supports("inhale_offsets")
            and getattr(self.bm, "inhale_offsets", None) is not None
        ):
            inhale_offsets = self._get_event_array("inhale_offsets")
        else:
            inhale_offsets = inhale_onsets
        if (
            self._supports("inhale_pause_onsets")
            and getattr(self.bm, "inhale_pause_onsets", None) is not None
        ):
            inhale_pause_onsets = self._get_event_array("inhale_pause_onsets")
        else:
            inhale_pause_onsets = np.full_like(inhale_onsets, MISSING_EVENT)

        inhale_time2peak = (
            _as_float_array(getattr(self.bm, "inhale_time2peak"))
            if getattr(self.bm, "inhale_time2peak", None) is not None
            else None
        )
        inhale_durations = (
            _as_float_array(getattr(self.bm, "inhale_durations"))
            if getattr(self.bm, "inhale_durations", None) is not None
            else None
        )
        inhale_volumes = (
            _as_float_array(getattr(self.bm, "inhale_volumes"))
            if getattr(self.bm, "inhale_volumes", None) is not None
            else None
        )

        onset = _safe_int(inhale_onsets[i])

        # effective inhale end: pause onset if present else inhale_offsets
        end_f = inhale_offsets[i]
        if _finite(inhale_pause_onsets[i]):
            end_f = inhale_pause_onsets[i]
        end = _safe_int(end_f) if _finite(end_f) else onset

        # duration (sec)
        if inhale_durations is not None:
            inhale_durations[i] = float(max(0, end - onset) / fs)

        # time to peak (sec)
        peak = _safe_int(inhale_peaks[i]) if _finite(inhale_peaks[i]) else onset
        if inhale_time2peak is not None:
            inhale_time2peak[i] = float(max(0, peak - onset) / fs)

        # volume (simple trapezoid integral over corrected resp)
        lo, hi = sorted((onset, end))
        lo = max(0, lo)
        hi = min(resp.shape[0] - 1, hi)
        if inhale_volumes is not None:
            if hi <= lo:
                inhale_volumes[i] = 0.0
            else:
                seg = resp[lo : hi + 1]
                inhale_volumes[i] = float(np.trapz(seg, dx=1.0 / fs))

    def _recompute_exhale(self, i: int) -> None:
        if not (
            self._supports("exhale_time2trough")
            or self._supports("exhale_durations")
            or self._supports("exhale_volumes")
        ):
            return
        fs = self._get_fs()
        resp = self._get_resp()

        exhale_onsets = self._get_event_array("exhale_onsets")
        if (
            self._supports("exhale_troughs")
            and getattr(self.bm, "exhale_troughs", None) is not None
        ):
            exhale_troughs = self._get_event_array("exhale_troughs")
        else:
            exhale_troughs = exhale_onsets
        if (
            self._supports("exhale_offsets")
            and getattr(self.bm, "exhale_offsets", None) is not None
        ):
            exhale_offsets = self._get_event_array("exhale_offsets")
        else:
            exhale_offsets = exhale_onsets
        if (
            self._supports("exhale_pause_onsets")
            and getattr(self.bm, "exhale_pause_onsets", None) is not None
        ):
            exhale_pause_onsets = self._get_event_array("exhale_pause_onsets")
        else:
            exhale_pause_onsets = np.full_like(exhale_onsets, MISSING_EVENT)

        exhale_time2trough = (
            _as_float_array(getattr(self.bm, "exhale_time2trough"))
            if getattr(self.bm, "exhale_time2trough", None) is not None
            else None
        )
        exhale_durations = (
            _as_float_array(getattr(self.bm, "exhale_durations"))
            if getattr(self.bm, "exhale_durations", None) is not None
            else None
        )
        exhale_volumes = (
            _as_float_array(getattr(self.bm, "exhale_volumes"))
            if getattr(self.bm, "exhale_volumes", None) is not None
            else None
        )

        onset = _safe_int(exhale_onsets[i]) if _finite(exhale_onsets[i]) else 0

        # effective exhale end: pause onset if present else exhale_offsets
        end_f = exhale_offsets[i]
        if _finite(exhale_pause_onsets[i]):
            end_f = exhale_pause_onsets[i]
        end = _safe_int(end_f) if _finite(end_f) else onset

        if exhale_durations is not None:
            exhale_durations[i] = float(max(0, end - onset) / fs)

        trough = _safe_int(exhale_troughs[i]) if _finite(exhale_troughs[i]) else onset
        if exhale_time2trough is not None:
            exhale_time2trough[i] = float(max(0, trough - onset) / fs)

        lo, hi = sorted((onset, end))
        lo = max(0, lo)
        hi = min(resp.shape[0] - 1, hi)
        if exhale_volumes is not None:
            if hi <= lo:
                exhale_volumes[i] = 0.0
            else:
                seg = resp[lo : hi + 1]
                exhale_volumes[i] = float(np.trapz(seg, dx=1.0 / fs))

    # ----------------------------
    # Undo support
    # ----------------------------

    def _make_undo_record(self, i: int, event: EventType) -> _UndoRecord:
        # event is stored mostly for debugging / future (we restore all the tracked fields anyway)
        def _event_val(name: str) -> float:
            if not self._supports(name) or getattr(self.bm, name, None) is None:
                return float(MISSING_EVENT)
            arr = self._get_event_array(name)
            if i >= arr.shape[0]:
                return float(MISSING_EVENT)
            return float(arr[i])

        def _float_val(name: str) -> float:
            if not self._supports(name) or getattr(self.bm, name, None) is None:
                return float("nan")
            arr = _as_float_array(getattr(self.bm, name))
            if i >= arr.shape[0]:
                return float("nan")
            return float(arr[i])

        return _UndoRecord(
            breath_i=i,
            event=event,
            old_inhale_onset=_event_val("inhale_onsets"),
            old_exhale_onset=_event_val("exhale_onsets"),
            old_inhale_pause_onset=_event_val("inhale_pause_onsets"),
            old_exhale_pause_onset=_event_val("exhale_pause_onsets"),
            old_inhale_time2peak=_float_val("inhale_time2peak"),
            old_inhale_duration=_float_val("inhale_durations"),
            old_inhale_volume=_float_val("inhale_volumes"),
            old_exhale_time2trough=_float_val("exhale_time2trough"),
            old_exhale_duration=_float_val("exhale_durations"),
            old_exhale_volume=_float_val("exhale_volumes"),
        )

    def _restore_undo_record(self, rec: _UndoRecord) -> None:
        i = rec.breath_i

        def _restore_event(name: str, value: float) -> None:
            if not self._supports(name) or getattr(self.bm, name, None) is None:
                return
            arr = self._get_event_array(name)
            if i >= arr.shape[0]:
                return
            if np.isfinite(value) and value >= 0:
                arr[i] = int(value)
            else:
                arr[i] = MISSING_EVENT

        def _restore_float(name: str, value: float) -> None:
            if not self._supports(name) or getattr(self.bm, name, None) is None:
                return
            arr = _as_float_array(getattr(self.bm, name))
            if i >= arr.shape[0]:
                return
            arr[i] = value

        _restore_event("inhale_onsets", rec.old_inhale_onset)
        _restore_event("exhale_onsets", rec.old_exhale_onset)
        _restore_event("inhale_pause_onsets", rec.old_inhale_pause_onset)
        _restore_event("exhale_pause_onsets", rec.old_exhale_pause_onset)

        _restore_float("inhale_time2peak", rec.old_inhale_time2peak)
        _restore_float("inhale_durations", rec.old_inhale_duration)
        _restore_float("inhale_volumes", rec.old_inhale_volume)

        _restore_float("exhale_time2trough", rec.old_exhale_time2trough)
        _restore_float("exhale_durations", rec.old_exhale_duration)
        _restore_float("exhale_volumes", rec.old_exhale_volume)
