from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from breathmetrics import cli


def test_estimate_creates_output(tmp_path) -> None:
    fs = 200.0
    t = np.arange(0, 10, 1 / fs)
    resp = np.sin(2 * np.pi * 1.0 * t)
    csv_path = tmp_path / "resp.csv"
    pd.DataFrame({"resp": resp}).to_csv(csv_path, index=False)

    out_dir = tmp_path / "results"
    cli.main(
        [
            "estimate",
            str(csv_path),
            "--fs",
            str(fs),
            "--datatype",
            "humanAirflow",
            "--out",
            str(out_dir),
            "--overwrite",
        ]
    )

    assert (out_dir / "bm.pkl").exists()


def test_inspect_refuses_non_pkl() -> None:
    with pytest.raises(SystemExit):
        cli.main(["inspect", "not_a_pickle.txt"])


def test_info_refuses_non_pkl() -> None:
    with pytest.raises(SystemExit):
        cli.main(["info", "not_a_pickle.txt"])
