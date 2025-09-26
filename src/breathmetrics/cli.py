import typer
import numpy as np
from .core import BreathMetrics, BreathMetricsConfig

app = typer.Typer(help="BreathMetrics CLI")


@app.command()
def quickstart(fs: float = 100.0, seconds: float = 10.0):
    """Run a tiny demo on synthetic data."""
    t = np.arange(0, seconds, 1 / fs)
    # toy breath-like waveform
    x = np.sin(2 * np.pi * 0.25 * t) + 0.1 * np.random.randn(t.size)

    bm = BreathMetrics(x, BreathMetricsConfig(fs=fs)).preprocess().estimate_features()
    print("Estimated features:", bm.features)


if __name__ == "__main__":
    app()
