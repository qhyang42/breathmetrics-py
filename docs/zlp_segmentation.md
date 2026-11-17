# ZLP breath segmentation

An alternative segmentation front end for the `Breathe` class, ported from
the Zelano Lab preprocessing pipeline (ZLP), where it is the production
segmentation behind every breathmetrics-based output across seven task
pipelines (intracranial + scalp-EEG sessions; belt, pressure and nasal
cannula sensors). The same engine has been contributed to the MATLAB
toolbox (`zelanolab/breathmetrics` PR #13 / `qhyang42/breathmetrics` PR #1);
the full rule-by-rule algorithm specification with every constant lives in
that PR's `docs/zlp_segmentation.md` and applies verbatim here.

## Usage

```python
from breathmetrics import Breathe

bm = Breathe(resp, srate, "humanAirflow")
bm.estimate_all_features_zlp()          # instead of estimate_all_features()

bm = Breathe(resp, srate, "humanAirflow")
bm.estimate_all_features_zlp(merge_incomplete_exhales=True)   # one breath per exhale
bm.zlp_detection["merge"]["info"]       # what the merge did

bm = Breathe(resp, srate, "humanAirflow")
bm.estimate_all_features_zlp(refine_onsets=False)   # onsets as before the refinement
```

Options: `floor_frac` (amplitude-normalization floor, default 0.05),
`blank_below_frac` (zero hardware-attenuated stretches), `sigh_span`
(a `[start, end]` sample pair marking a paced double-inhale block, merged at
the segmentation level), `refine_onsets` (default `True`: re-place every
inhale onset on the stored trace, see below) with `refine_opts` (a dict of
overrides of the refinement's options), `merge_incomplete_exhales` (default
`False`: join inhales that have no exhale since the previous one) with
`merge_ex_frac` (default 0.20) and `merge_opts` (a dict of overrides), plus
`compute_resp_stats` / `verbose` as in `estimate_all_features`. Option names
are the snake_case spellings of the MATLAB options; an unknown name raises
`ValueError`.

**Change of the default output:** the onset refinement is on by default, as
in the MATLAB driver, so `estimate_all_features_zlp()` places onsets
differently from the first release of this method. `refine_onsets=False`
reproduces that earlier behaviour exactly.

Lower-level access mirrors the MATLAB signatures
(`breathmetrics.kernel_zlp`): `prep_breath_trace` (conditioning +
validity-ruled extrema; `'conservative'` locked, `'pwl'`/`'twoscale'` kept
for comparison), `find_inhale_onsets` (eligibility rules + the
`'kneeBacktrack'` slope-walk placement; may prune spurious extrema pairs),
`pair_breath_landmarks` (equal-length per-inhale arrays),
`refine_inhale_onsets` (onset refinement on the stored trace; returns
`(onsets, info)`), `merge_incomplete_exhales` (returns `(merged, info)`),
and the two wrappers `apply_onset_refinement` (refinement plus the
consistency checks of the driver) and `merge_paired_landmarks` (the merge on
the paired landmarks, with the trough guard described below). All sample
indices are 0-based.

## Why

On amplitude-non-stationary recordings — drifting baselines, breathing depth
changing several-fold between task blocks, paced or very slow breathing —
global-criterion extrema and pause-band onset placement put inhale onsets on
exhale troughs, on peaks, or mid-rise, and either split single breaths or
miss quiet ones. The ZLP engine conditions a detection copy (500-ms
smoothing + 30-s moving-std amplitude normalization), finds validity-ruled
extrema on it, and places one onset per trough→peak pair with a slope-walk
algorithm whose every rule came out of a reviewed failure case (~20
live-reviewed QC generations on an ≈80-session corpus).

## How it integrates

`estimate_all_features_zlp` runs the same pipeline order as
`estimate_all_features`, with segmentation swapped:

1. baseline correction (unchanged);
2. **ZLP extrema + inhale onsets** on an internally conditioned copy of
   `raw_respiration`; then the **onset refinement** (on by default), and,
   after the landmarks are paired into equal-length per-inhale arrays, the
   optional **merge**. The detection trace and the records of both stages are
   stashed on `bm.zlp_detection` for QC (`det`, `info`, `prep_peaks`,
   `prep_troughs`, `refine`, `merge`; `refine` / `merge` are `None` when the
   stage is off);
3. the legacy onset/pause pass runs on the ZLP extrema to estimate exhale
   onsets and pauses — its inhale onsets are then replaced by the ZLP ones,
   and any legacy inhale-pause onset the ZLP onsets contradict is dropped;
4. slope-based exhale pauses, offsets, durations, volumes and shape features
   as usual.

**Units note (difference from the MATLAB integration):** here the `Breathe`
object keeps the signal in its native units — the ZLP windowed normalization
is internal to detection only — so volumes and flows come out in raw signal
units directly. (In the MATLAB integration the object is built on the
normalized detection trace, which is why that PR adds separate raw-unit
volume twins.)

## Onset refinement

The detection finds each onset on the 500-ms-smoothed, 30-s-normalized
trace. On the stored trace (raw units) that onset is often early - the trace
is still flat and the rise starts 0.1-0.3 s later -, sometimes sits on a small
bump well before a quick rise, and sometimes on a small upward inflection
riding high on a larger inhale. `refine_inhale_onsets` looks at the stored
trace itself (lightly smoothed) and moves an onset only when the trace shows a
definitively better one. Peaks, troughs and the number of breaths are not
changed. Slope scales are moving means of 40, 80, 160 and 300 ms; per breath
the finest scale whose steepest slope in the search range exceeds 5 times the
local slope noise is used, and the foot of a rise is the last sample before
the steepest rise (weighted towards the peak) where the slope has stayed
below max(5 % of the steepest slope, 2 times the noise) for 50 ms. Rules, in
order (each sees the result of the earlier ones):

1. `foot` - early by smoothing. The foot of the steepest rise within
   [onset - 200 ms, peak - 100 ms] replaces the onset when it moved by at
   least 20 ms. A forward move may step over a flat stretch or a small bump,
   never over part of the rise itself (refused above 0.15 x the breath
   amplitude of level, or above 0.05 x while the stepped-over stretch is
   steep).
2. `far` - a bump long before a quick rise. An onset farther from its peak
   than max(0.8 s, 1.5 x the median onset-to-peak time of its 8 + 8
   neighbours) is searched again in the last 1.5 x that median before the
   peak (forward moves only).
3. `level` - an onset riding high on a larger inhale. The reference is the
   median onset level of the whole trace handed to the function, the margin
   half the width of the middle 30 % of the raw trace values; an onset above
   reference + margin is searched again for the foot of the rise (the earlier
   rise if that foot is still above the band) and accepted when it is lower by
   at least 0.10 x the amplitude. Guards: an onset after a flat pause near the
   usual onset level is not moved, a pause before the detected onset is a
   floor, and a new onset with a deep exhale trough within 1 s before it is
   refused (a rebound, not an inhale start).

Every search starts no earlier than max(previous peak + 50 ms, the pair's
trough), and a final pass keeps the onsets strictly ascending and between the
previous peak and their own peak. `apply_onset_refinement` (used by
`estimate_all_features_zlp`) checks the result and raises `ValueError` rather
than repair it: onsets finite and inside the trace, strictly ascending, one
per detected onset, a peak after each onset, no two onsets sharing a peak. The
record (`bm.zlp_detection["refine"]`) holds the landmarks before and after,
the rule that moved each onset (`''`, `'foot'`, `'far'`, `'level'`, with a
`+clamp` tag) and the shift in ms. All constants are in
`kernel_zlp._REFINE_DEFAULTS` (seconds / milliseconds, not samples).

On the MATLAB toolbox's `sample_data.mat` (660 s at 1 kHz) the refinement
moves 130 of 131 onsets, all by rule `foot` and all later (median shift
263 ms, maximum 911 ms).

## Joining inhales that have no exhale between them (optional)

An inhale is not a new breath unless there has been an exhale since the last
one. `merge_incomplete_exhales(resp, fs, onsets, peaks, troughs, ex_frac)`
takes the landmarks of the breaths (breath `k` = `[onsets[k], onsets[k+1])`,
one peak and one trough each; the last onset only closes the last breath)
and joins an inhale to the breath before it when the exhale since the
previous inhale is under `ex_frac` (0.20) of that inhale, both as signal
areas of the trace above / below zero flow. Three steps, in order:

1. Pause blips. An onset whose own window holds neither an inhale nor an
   exhale (each area under 10 % of the median of the windows within 10 rows,
   beyond a noise band of 5 % of the local swing) joins the breath before it.
2. Lead bumps. A small pulse (peak under half of the highest pulse of the run
   the next inhale starts) with no exhale after it and at least 3 s near zero
   flow before the next inhale is the overshoot of the previous exhale's
   recovery: it joins the breath before, and the breath starts at the next
   inhale.
3. Merge. For each remaining inhale `k`, over `[onset(k-1), onset(k))`: I =
   the area above zero flow up to the first crossing from above to below zero
   flow (to the onset when there is none), E = the area below zero flow from
   that crossing to the onset (0 without a crossing). When E < `ex_frac` x I,
   inhale `k` joins the breath of inhale `k-1`; consecutive joins form one
   breath, and each inhale is judged on its own window, so a later complete
   exhale always ends the breath.

Keep rule: a joined breath keeps its first onset, except when that onset has
an exhale trough immediately before it and the joining inhale a flat pause
immediately before it - the first onset is then the sensor's rebound off the
trough; it and the rows up to the joining inhale become the end of the breath
before, and the joining inhale starts the breath. The merged breath ends where
its last member ends, its peak is the highest of its own inhale pulses, its
trough the lowest member trough.

Zero flow is the higher of the level the trace dwells at between breaths
(its most frequent level within 10 s, accepted only when at least 15 % of
the window sits within one histogram bin of it and at least 10 % lies above
and below it) and the 60-s moving mean. Areas are signal areas of the
200-ms-smoothed trace, not volumes; nasal pressure under-reads slow flow, so a
very slow, shallow exhale after a fast inhale can fall under 20 % and the next
inhale then merges.

In `estimate_all_features_zlp(merge_incomplete_exhales=True)` the merge runs
on the paired breaths `0..n-2` (the last paired inhale has no closing onset
and is kept as it is) and the merged landmarks, plus that last breath,
replace the per-inhale arrays, so the object holds one breath per merged
breath and every feature is recomputed for it. A merged breath's trough is the
lowest member trough that lies after its merged peak (`merge_paired_landmarks`
counts the replacements in `n_trough_guard`). The merge changes the number of
breaths: match breaths between two runs by onset sample, not by index.
`bm.zlp_detection["merge"]` holds `merged` (onsets, ends, peaks, troughs,
`kept_row`, `n_pulses`, `n_rows`, areas, `orig_rows`), `info` (the rule record:
`n_before`, `n_after`, `n_minor` = blips + lead bumps, `n_merged`,
`n_trough_heads`, per-breath arrays) and `n_trough_guard`.

On `sample_data.mat` the merge joins nothing (130 -> 130 breaths): the
recording has no double inhales. On the deterministic synthetic trace of
`tests/zlp_reference/make_zlp_refine_merge_reference.py` (regular breaths with
double inhales, an exhale rebound, lead bumps, pause blips, quick inhales behind
a slow bump and two-stage inhales; 638 s at 100 Hz) the refinement fires every
rule (140 `foot`, 1 `far`, 4 `level`) and the merge takes 149 breaths to 131.

## Parity with MATLAB

`kernel_zlp.py` replicates MATLAB semantics exactly (`movmean`/`movstd`
centered shrinking windows including the even-window asymmetry, directional
`movmax`/`movmin`, trailing `movsum`, `fillmissing` linear + nearest edges,
round-half-away-from-zero, and `findpeaks`' prominence-then-greedy-distance
filter order). Landmark parity with the MATLAB implementation was verified
sample-for-sample on the MATLAB repo's `sample_data.mat` (660,001 samples @
1 kHz): detection trace max |diff| 5×10⁻¹³; 132/132 prep peaks and troughs,
131/131 onsets, 132/132 final extrema exactly identical.

The refinement and the merge were transcribed from the MATLAB functions
with 1-based sample numbers inside and are checked against them in
`tests/test_zlp_refine_merge.py` (detection landmarks, refined onsets, rule
labels and shifts, merged landmarks, areas, per-breath merge flags, and the
landmarks of the `Breathe` integration against the MATLAB driver, on the
deterministic synthetic trace and reference stored in `tests/zlp_reference/`). The same
comparison on `sample_data.mat` agrees exactly in every stage. Where MATLAB
and numpy round differently in the last digit, the port emulates MATLAB: the
colon operator (histogram bin edges), `prctile`, and the accumulation order of
`conv(c, [1 1 1]/3, 'same')`, which decides ties between equal bin counts when
the zero-flow level is picked from the histogram. The last one was measured
against one MATLAB release (R2026a) and is an emulation of its internal
order; a different release or CPU could break a tie differently, which can
move a zero-flow level by one histogram bin.

All indices are 0-based. Requires the Signal Processing stack already in the
package dependencies (numpy, scipy).
