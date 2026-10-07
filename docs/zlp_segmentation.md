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
```

Options: `floor_frac` (amplitude-normalization floor, default 0.05),
`blank_below_frac` (zero hardware-attenuated stretches), `sigh_span`
(a `[start, end]` sample pair marking a paced double-inhale block, merged at
the segmentation level), plus `compute_resp_stats` / `verbose` as in
`estimate_all_features`.

Lower-level access mirrors the MATLAB signatures
(`breathmetrics.kernel_zlp`): `prep_breath_trace` (conditioning +
validity-ruled extrema; `'conservative'` locked, `'pwl'`/`'twoscale'` kept
for comparison), `find_inhale_onsets` (eligibility rules + the
`'kneeBacktrack'` slope-walk placement; may prune spurious extrema pairs),
`pair_breath_landmarks` (equal-length per-inhale arrays).

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
   `raw_respiration` (landmarks are paired into equal-length per-inhale
   arrays; the detection trace is stashed on `bm.zlp_detection` for QC);
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

## Parity with MATLAB

`kernel_zlp.py` replicates MATLAB semantics exactly (`movmean`/`movstd`
centered shrinking windows including the even-window asymmetry, directional
`movmax`/`movmin`, trailing `movsum`, `fillmissing` linear + nearest edges,
round-half-away-from-zero, and `findpeaks`' prominence-then-greedy-distance
filter order). Landmark parity with the MATLAB implementation was verified
sample-for-sample on the MATLAB repo's `sample_data.mat` (660,001 samples @
1 kHz): detection trace max |diff| 5×10⁻¹³; 132/132 prep peaks and troughs,
131/131 onsets, 132/132 final extrema exactly identical.

All indices are 0-based. Requires the Signal Processing stack already in the
package dependencies (numpy, scipy).
