# tests for the ZLP onset refinement and the incomplete-exhale breath merge.
#
# The reference (tests/zlp_reference/zlp_refine_merge_reference.npz) holds a
# deterministic synthetic trace (float32 copy, exactly what MATLAB saw) and the
# outputs of the MATLAB toolbox functions on it; see
# tests/zlp_reference/make_zlp_refine_merge_reference.py. The trace holds regular
# breaths with end-expiratory pauses and, interleaved, patterns that fire
# every rule: refinement rules foot / far / level, and the merge's pause blips,
# lead bumps, double inhales and a trough rebound (keep rule). MATLAB samples
# are one-based in the reference; the tests subtract 1.
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from breathmetrics import Breathe
from breathmetrics.kernel_zlp import (
    apply_onset_refinement,
    find_inhale_onsets,
    merge_incomplete_exhales,
    merge_paired_landmarks,
    pair_breath_landmarks,
    prep_breath_trace,
    refine_inhale_onsets,
)

REF_FILE = Path(__file__).parent / "zlp_reference" / "zlp_refine_merge_reference.npz"


@pytest.fixture(scope="module")
def ref() -> dict[str, np.ndarray]:
    with np.load(REF_FILE) as z:
        return {k: z[k] for k in z.files}


def zero_based(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.int64) - 1


def rules_of(text: np.ndarray) -> list[str]:
    return re.findall(r"<([^>]*)>", str(text[0]))


@pytest.fixture(scope="module")
def trace(ref: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    return ref["resp"].astype(float), float(ref["fs"][0])


@pytest.fixture(scope="module")
def detection(trace: tuple[np.ndarray, float]):
    resp, fs = trace
    det, pk0, tr0, _ = prep_breath_trace(resp, fs)
    on, pk, tr = find_inhale_onsets(det, fs, pk0, tr0)
    return on, pk, tr


def test_detection_landmarks_match_matlab(ref, detection):
    on, pk, tr = detection
    np.testing.assert_array_equal(on, zero_based(ref["engOn"]))
    np.testing.assert_array_equal(pk, zero_based(ref["engPk"]))
    np.testing.assert_array_equal(tr, zero_based(ref["engTr"]))


def test_refine_inhale_onsets_matches_matlab(ref, trace, detection):
    resp, fs = trace
    on, pk, tr = detection
    new_on, info = refine_inhale_onsets(resp, fs, on, pk, tr)
    np.testing.assert_array_equal(new_on, zero_based(ref["refOn"]))
    assert info["rule"] == rules_of(ref["refRule"])
    np.testing.assert_allclose(info["shift_ms"], ref["refShiftMs"], atol=1e-9)
    # the synthetic trace exercises all three rules
    counts = info["counts"]
    assert counts["n_foot"] > 100 and counts["n_far"] >= 1 and counts["n_level"] >= 1
    assert [counts["n_foot"], counts["n_far"], counts["n_level"]] == list(
        ref["refCounts"][:3]
    )
    assert (
        counts["n_foot"] + counts["n_far"] + counts["n_level"] + counts["n_unchanged"]
        == on.size
    )
    # peaks and troughs are not touched; one refined onset per input onset
    assert new_on.size == on.size


def test_refine_inhale_onsets_option_variant_matches_matlab(ref, trace, detection):
    resp, fs = trace
    on, pk, tr = detection
    opts = {"level_guard": "none", "level_trough_frac": 0}
    new_on, info = refine_inhale_onsets(resp, fs, on, pk, tr, opts)
    np.testing.assert_array_equal(new_on, zero_based(ref["refVarOn"]))
    assert info["rule"] == rules_of(ref["refVarRule"])
    assert info["opts"]["level_guard"] == "none"


def test_refine_inhale_onsets_rejects_unknown_option(trace, detection):
    resp, fs = trace
    on, pk, tr = detection
    with pytest.raises(ValueError, match="unknown option"):
        refine_inhale_onsets(resp, fs, on, pk, tr, {"no_such_option": 1})


def test_apply_onset_refinement_checks_its_result(ref, trace, detection):
    resp, fs = trace
    on, pk, tr = detection
    new_on, rec = apply_onset_refinement(resp, fs, on, pk, tr, None)
    np.testing.assert_array_equal(new_on, zero_based(ref["refOn"]))
    assert rec["n_on_in"] == rec["n_on_out"] == on.size
    assert len(rec["rule"]) == on.size
    # an onset with no peak after it cannot be refined consistently
    with pytest.raises(ValueError, match="onset refinement result rejected"):
        apply_onset_refinement(resp, fs, on, pk[:-3], tr, None)


def merge_input(ref):
    return (
        zero_based(ref["mergeInOn"]),
        zero_based(ref["mergeInPk"]),
        zero_based(ref["mergeInTr"]),
    )


@pytest.mark.parametrize(("tag", "ex_frac"), [("mg1", 0.20), ("mg2", 0.50)])
def test_merge_incomplete_exhales_matches_matlab(ref, trace, tag, ex_frac):
    resp, fs = trace
    on, pk, tr = merge_input(ref)
    merged, info = merge_incomplete_exhales(resp, fs, on, pk, tr, ex_frac)
    np.testing.assert_array_equal(merged["onsets"], zero_based(ref[tag + "On"]))
    np.testing.assert_array_equal(merged["ends"], zero_based(ref[tag + "End"]))
    np.testing.assert_array_equal(merged["peaks"], zero_based(ref[tag + "Pk"]))
    np.testing.assert_array_equal(merged["troughs"], zero_based(ref[tag + "Tr"]))
    np.testing.assert_array_equal(merged["kept_row"], zero_based(ref[tag + "Kept"]))
    np.testing.assert_array_equal(merged["n_pulses"], ref[tag + "NPulses"])
    np.testing.assert_array_equal(merged["n_rows"], ref[tag + "NRows"])
    np.testing.assert_allclose(merged["inhale_area"], ref[tag + "InArea"], rtol=1e-9)
    np.testing.assert_allclose(merged["exhale_area"], ref[tag + "ExArea"], rtol=1e-9)
    flat = np.concatenate(merged["orig_rows"])
    np.testing.assert_array_equal(flat, zero_based(ref[tag + "OrigFlat"]))
    np.testing.assert_array_equal(
        [len(r) for r in merged["orig_rows"]], ref[tag + "OrigCount"]
    )
    for key, name in [
        ("is_minor", "IsMinor"),
        ("is_lead_bump", "IsLeadBump"),
        ("is_tail", "IsTail"),
        ("is_trough_head", "IsTroughHead"),
    ]:
        np.testing.assert_array_equal(info[key].astype(int), ref[tag + name])
    counts = [
        info[k]
        for k in (
            "n_before",
            "n_after",
            "n_minor",
            "n_lead_bumps",
            "n_merged",
            "n_trough_heads",
        )
    ]
    assert counts == list(ref[tag + "Counts"])


def test_merge_record_counts_add_up(ref, trace):
    resp, fs = trace
    on, pk, tr = merge_input(ref)
    merged, info = merge_incomplete_exhales(resp, fs, on, pk, tr)
    n_after = merged["onsets"].size
    # every input breath belongs to exactly one merged breath
    rows = np.sort(np.concatenate(merged["orig_rows"]))
    np.testing.assert_array_equal(rows, np.arange(on.size - 1))
    assert int(np.sum(merged["n_rows"])) == info["n_before"]
    assert info["n_after"] == n_after
    assert info["n_before"] == n_after + info["n_minor"] + info["n_merged"]
    # the merge fires every step on this trace
    assert info["n_minor"] > 0 and info["n_lead_bumps"] > 0
    assert info["n_merged"] > 0 and info["n_trough_heads"] > 0
    # each merged breath keeps the first of its member rows
    assert all(
        r[0] == k for r, k in zip(merged["orig_rows"], merged["kept_row"], strict=True)
    )


def test_merge_incomplete_exhales_rejects_bad_input(trace):
    resp, fs = trace
    with pytest.raises(ValueError, match="unknown option"):
        merge_incomplete_exhales(
            resp, fs, [100, 400, 700], [200, 500], [300, 600], opts={"bogus": 1}
        )
    with pytest.raises(ValueError, match="at least two onsets"):
        merge_incomplete_exhales(resp, fs, [100], [], [])
    with pytest.raises(ValueError, match="strictly ascending"):
        merge_incomplete_exhales(resp, fs, [400, 100], [200], [300])


def landmarks_of(bm: Breathe) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(bm.inhale_onsets, dtype=int),
        np.asarray(bm.inhale_peaks, dtype=int),
        np.asarray(bm.exhale_troughs, dtype=int),
    )


def test_breathe_refine_off_reproduces_the_detection(ref, trace, detection):
    """refine_onsets=False is the segmentation as it was before the stages."""
    resp, fs = trace
    on, pk, tr = detection
    on0, pk0, tr0 = pair_breath_landmarks(on, pk, tr)
    bm = Breathe(resp, fs, "humanAirflow")
    bm.estimate_all_features_zlp(refine_onsets=False)
    got = landmarks_of(bm)
    np.testing.assert_array_equal(got[0], on0)
    np.testing.assert_array_equal(got[1], pk0)
    np.testing.assert_array_equal(got[2], tr0)
    np.testing.assert_array_equal(got[0], zero_based(ref["drv0On"]))
    assert bm.zlp_detection["refine"] is None
    assert bm.zlp_detection["merge"] is None


def test_breathe_default_refines_onsets_like_matlab(ref, trace):
    resp, fs = trace
    bm = Breathe(resp, fs, "humanAirflow")
    bm.estimate_all_features_zlp()
    got = landmarks_of(bm)
    np.testing.assert_array_equal(got[0], zero_based(ref["drv1On"]))
    np.testing.assert_array_equal(got[1], zero_based(ref["drv1Pk"]))
    np.testing.assert_array_equal(got[2], zero_based(ref["drv1Tr"]))
    assert bm.zlp_detection["merge"] is None
    rec = bm.zlp_detection["refine"]
    assert rec["n_on_in"] == rec["n_on_out"]
    assert rec["onsets_out"].size == rec["shift_ms"].size == len(rec["rule"])
    assert bm.feature_estimations_complete


def test_breathe_merge_matches_matlab_and_is_consistent(ref, trace):
    resp, fs = trace
    bm = Breathe(resp, fs, "humanAirflow")
    bm.estimate_all_features_zlp(merge_incomplete_exhales=True)
    on, pk, tr = landmarks_of(bm)
    np.testing.assert_array_equal(on, zero_based(ref["drv2On"]))
    np.testing.assert_array_equal(pk, zero_based(ref["drv2Pk"]))
    np.testing.assert_array_equal(tr, zero_based(ref["drv2Tr"]))

    # one breath per merged breath, in landmark order
    assert on.size == pk.size == tr.size == bm.inhale_onsets.size
    assert np.all(on[:-1] < pk[:-1]) and np.all(pk[:-1] < tr[:-1])
    assert np.all(tr[:-1] < on[1:])
    assert np.all(np.asarray(bm.inhale_onsets) < np.asarray(bm.inhale_peaks))

    # the records: merged breaths = the merge's output + the last paired inhale
    rec = bm.zlp_detection["merge"]
    assert rec["merged"]["onsets"].size == on.size - 1
    assert rec["info"]["n_after"] == on.size - 1
    assert rec["n_trough_guard"] == int(ref["nTroughGuard"][0])
    # features are recomputed for the merged breaths
    assert bm.feature_estimations_complete
    assert np.asarray(bm.inhale_volumes).size == on.size
    finite = np.isfinite(np.asarray(bm.inhale_volumes, dtype=float))
    assert finite.mean() > 0.9


def test_breathe_merge_fewer_breaths_than_unmerged(ref, trace):
    resp, fs = trace
    plain = Breathe(resp, fs, "humanAirflow")
    plain.estimate_all_features_zlp()
    merged = Breathe(resp, fs, "humanAirflow")
    merged.estimate_all_features_zlp(merge_incomplete_exhales=True)
    assert merged.inhale_onsets.size < plain.inhale_onsets.size
    # every merged onset is an onset of the unmerged segmentation
    assert set(merged.inhale_onsets.tolist()) <= set(plain.inhale_onsets.tolist())


def test_merge_paired_landmarks_trough_guard_and_order(ref, trace):
    resp, fs = trace
    on, pk, tr = (
        zero_based(ref["drv1On"]),
        zero_based(ref["drv1Pk"]),
        zero_based(ref["drv1Tr"]),
    )
    on_m, _pk_m, _tr_m, rec = merge_paired_landmarks(resp, fs, on, pk, tr)
    np.testing.assert_array_equal(on_m, zero_based(ref["drv2On"]))
    assert rec["n_trough_guard"] == int(ref["nTroughGuard"][0])
    with pytest.raises(ValueError, match="at least two"):
        merge_paired_landmarks(resp, fs, on[:1], pk[:1], tr[:1])
