from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import breathmetrics


def _fail(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(2)


def _ensure_csv(path: Path) -> None:
    if path.suffix.lower() != ".csv":
        _fail("Input must be a .csv file containing a respiration column.")


def _ensure_pkl(path: Path) -> None:
    if path.suffix.lower() != ".pkl":
        _fail("Input must be a .pkl file saved by `breathmetrics estimate`.")


def _read_csv_column(path: Path, column: str) -> np.ndarray:
    _ensure_csv(path)
    if not path.exists():
        _fail(f"Input file not found: {path}")
    df = pd.read_csv(path)
    if column not in df.columns:
        cols = ", ".join(df.columns.astype(str))
        _fail(
            f"Column '{column}' not found in CSV. Available columns: {cols if cols else 'none'}"
        )
    return df[column].to_numpy(dtype=float)


def _dump_pickle(obj: Any, path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        _fail(f"Output already exists: {path} (use --overwrite to replace).")
    with path.open("wb") as f:
        pickle.dump(obj, f)


def _load_pickle(path: Path) -> Any:
    _ensure_pkl(path)
    if not path.exists():
        _fail(f"Pickle not found: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def _supports(obj: Any, name: str) -> bool:
    supports = getattr(obj, "supports", None)
    if callable(supports):
        try:
            return bool(supports(name))
        except Exception:
            return False
    return getattr(obj, name, None) is not None


def _format_feature_status(obj: Any, names: Iterable[str]) -> str:
    parts = []
    for name in names:
        if not _supports(obj, name):
            status = "unsupported"
        else:
            val = getattr(obj, name, None)
            status = "present" if val is not None else "missing"
        parts.append(f"{name}:{status}")
    return ", ".join(parts)


def _estimate(args: argparse.Namespace) -> None:
    data = _read_csv_column(args.input, args.column)
    bm = breathmetrics.Breathe(data, float(args.fs), str(args.datatype))
    bm.estimate_all_features(verbose=bool(args.verbose))

    out_dir = args.out
    if out_dir.exists() and not out_dir.is_dir():
        _fail(f"Output path is not a directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bm.pkl"
    _dump_pickle(bm, out_path, overwrite=bool(args.overwrite))

    if args.verbose:
        print(f"Wrote: {out_path}")


def _inspect(args: argparse.Namespace) -> None:
    obj = _load_pickle(args.path)
    required = ("baseline_corrected_respiration", "inhale_onsets", "exhale_onsets")
    missing = [
        name for name in required if _supports(obj, name) and getattr(obj, name) is None
    ]
    if missing:
        _fail(
            "This object has not been fully estimated.\nRun `breathmetrics estimate ...` first."
        )
    if not hasattr(obj, "inspect"):
        _fail("Loaded object does not support inspection.")
    obj.inspect()


def _info(args: argparse.Namespace) -> None:
    obj = _load_pickle(args.path)
    raw = getattr(obj, "raw_respiration", None)
    if raw is None:
        raw = getattr(obj, "baseline_corrected_respiration", None)
    if raw is None:
        _fail("Object is missing respiration signal data.")

    n_samples = int(np.asarray(raw).shape[0])
    fs = float(getattr(obj, "srate", float("nan")))
    duration_s = float(n_samples / fs) if fs > 0 else float("nan")
    datatype = getattr(obj, "datatype", "unknown")

    inhale_onsets = getattr(obj, "inhale_onsets", None)
    n_breaths = None
    if inhale_onsets is not None:
        arr = np.asarray(inhale_onsets, dtype=float)
        n_breaths = int(np.sum(arr >= 0))

    primary = [
        "signal_peaks",
        "signal_troughs",
        "inhale_onsets",
        "exhale_onsets",
        "inhale_offsets",
        "exhale_offsets",
        "inhale_pause_onsets",
        "exhale_pause_onsets",
    ]

    print(f"datatype: {datatype}")
    print(f"sampling_rate_hz: {fs:g}")
    print(f"signal_duration: {n_samples} samples ({duration_s:.3f} s)")
    if n_breaths is not None:
        print(f"breaths_detected: {n_breaths}")
    else:
        print("breaths_detected: unknown")
    print(f"primary_features: {_format_feature_status(obj, primary)}")

    if args.verbose:
        extra = [
            "inhale_peaks",
            "exhale_troughs",
            "inhale_time2peak",
            "exhale_time2trough",
            "inhale_durations",
            "exhale_durations",
            "inhale_volumes",
            "exhale_volumes",
        ]
        print(f"secondary_features: {_format_feature_status(obj, extra)}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="breathmetrics",
        description="BreathMetrics CLI (v0.1)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    estimate = subparsers.add_parser(
        "estimate", help="Estimate features from raw respiration CSV."
    )
    estimate.add_argument("input", type=Path, help="Path to CSV input file.")
    estimate.add_argument("--fs", type=float, required=True, help="Sampling rate (Hz).")
    estimate.add_argument(
        "--datatype",
        type=str,
        required=True,
        help="Respiration datatype (e.g., humanAirflow).",
    )
    estimate.add_argument("--out", type=Path, required=True, help="Output directory.")
    estimate.add_argument(
        "--column",
        type=str,
        default="resp",
        help="CSV column name for respiration data.",
    )
    estimate.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    estimate.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress messages.",
    )
    estimate.set_defaults(func=_estimate)

    inspect = subparsers.add_parser(
        "inspect", help="Launch GUI for a saved BreathMetrics object."
    )
    inspect.add_argument("path", type=Path, help="Path to .pkl BreathMetrics object.")
    inspect.add_argument("--verbose", action="store_true", help="Verbose output.")
    inspect.set_defaults(func=_inspect)

    info = subparsers.add_parser(
        "info", help="Print summary for a saved BreathMetrics object."
    )
    info.add_argument("path", type=Path, help="Path to .pkl BreathMetrics object.")
    info.add_argument("--verbose", action="store_true", help="Verbose output.")
    info.set_defaults(func=_info)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        _fail("Interrupted.")
    except Exception as exc:
        _fail(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
