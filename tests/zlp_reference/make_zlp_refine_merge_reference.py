"""Build ``zlp_refine_merge_reference.npz``, the MATLAB reference of
``tests/test_zlp_refine_merge.py``.

The reference is the output of the MATLAB toolbox functions
``refineInhaleOnsets_zlp`` / ``mergeIncompleteExhales_zlp`` (and the
``zlpEstimateAllFeatures`` driver) on a deterministic synthetic trace.

Steps (run from this folder)::

    python make_zlp_refine_merge_reference.py trace trace.bin
    # MATLAB, toolbox on the path:  run zlp_refine_merge_reference.m
    #   (reads trace.bin, writes ref.mat)
    python make_zlp_refine_merge_reference.py convert ref.mat \\
        zlp_refine_merge_reference.npz trace.bin

The trace is rounded to float32 before it is handed to MATLAB, so the
stored float32 copy is exactly the signal both implementations see.
"""

from __future__ import annotations

import sys

import numpy as np

FS = 100.0

# patterns: rows [start, duration, amplitude] relative to the pattern start,
# and the time the pattern occupies (s)
PATTERNS = {
    "regular": ([[0, 0.8, 1], [0.8, 1.8, -0.6]], 4.0),
    "far": ([[0, 1.8, -0.6], [3.8, 1.0, 0.25], [5.8, 0.8, 1], [6.6, 1.8, -0.6]], 8.9),
    "level": ([[0, 1.0, 0.7], [0.65, 1.5, 0.9], [2.15, 1.8, -0.6]], 5.5),
    "double": ([[0, 1.0, 1], [0.3, 2.4, 0.35], [1.5, 1.0, 0.8], [3.0, 1.8, -0.6]], 6.0),
    "rebound": (
        [[0, 1.2, -0.9], [1.2, 0.9, 0.55], [3.6, 1.3, 1], [4.9, 1.8, -0.6]],
        7.5,
    ),
    "lead": ([[0, 1.8, -0.6], [2.0, 0.8, 0.45], [6.0, 1.3, 1], [7.3, 1.8, -0.6]], 10.0),
    "blip": ([[0, 1.8, -0.6], [3.0, 0.5, 0.4], [7.0, 1.3, 1], [8.3, 1.8, -0.6]], 11.0),
}
FAR_AMP = [0.25, 0.22, 0.28, 0.25, 0.30, 0.20, 0.26, 0.24]
ORDER = ["far", "level", "double", "rebound", "lead", "blip"]


def synthetic_trace(
    fs: float = FS,
    repeats: int = 4,
    regular_between: int = 4,
    noise: float = 0.4,
    seed: int = 12345,
) -> np.ndarray:
    """A nasal-pressure-like trace (inhale positive, ~100 units peak to peak
    per breath) of raised-cosine humps: regular breaths with an end-expiratory
    pause, interleaved with short blocks that exercise the rules of the onset
    refinement and the merge (a small slow bump ahead of a quick inhale; a
    two-stage inhale; a double inhale; an exhale rebound before a pause; a
    lead bump and a pause blip before a long rest)."""
    comps: list[list[float]] = []
    t = 5.0
    for _ in range(12):
        comps += [[a + t, b, c] for a, b, c in PATTERNS["regular"][0]]
        t += PATTERNS["regular"][1]
    for name in ORDER:
        for rep in range(repeats):
            rows, span = PATTERNS[name]
            rows = [list(r) for r in rows]
            if name == "far":
                rows[1][2] = FAR_AMP[rep % len(FAR_AMP)]
            comps += [[a + t, b, c] for a, b, c in rows]
            t += span
            for _ in range(regular_between):
                comps += [[a + t, b, c] for a, b, c in PATTERNS["regular"][0]]
                t += PATTERNS["regular"][1]
    n = int(np.floor((t + 5) * fs + 0.5))
    tt = np.arange(n) / fs
    x = np.zeros(n)
    for a, d, amp in comps:
        m = (tt >= a) & (tt < a + d)
        x[m] += amp * 0.5 * (1 - np.cos(2 * np.pi * (tt[m] - a) / d))
    x = 100 * x + 2 * np.sin(2 * np.pi * 1.15 * tt) + 5 * np.sin(2 * np.pi * tt / 37)
    x = x + noise * np.random.default_rng(seed).standard_normal(n)
    return x.astype(np.float32)


def convert(mat_file: str, npz_file: str, trace_bin: str) -> None:
    from scipy.io import loadmat

    ref = loadmat(mat_file, squeeze_me=True)
    out: dict[str, np.ndarray] = {}
    for key, val in ref.items():
        if key.startswith("__"):
            continue
        out[key] = np.atleast_1d(np.asarray(val))
    out["resp"] = np.fromfile(trace_bin, dtype="<f8").astype(np.float32)
    np.savez_compressed(npz_file, **out)


if __name__ == "__main__":
    if sys.argv[1] == "trace":
        # float64 copy of the float32-rounded trace, for MATLAB
        synthetic_trace().astype("<f8").tofile(sys.argv[2])
    elif sys.argv[1] == "convert":
        convert(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        raise SystemExit(__doc__)
