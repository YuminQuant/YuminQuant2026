from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _sample_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [20250102, 20250103] * 3,
            "factor_id": ["sample"] * 6,
            "portfolio": ["group_1", "group_1", "group_2", "group_2", "long_short", "long_short"],
            "return": [0.01, -0.02, 0.02, 0.01, 0.01, 0.03],
            "excess_return": [0.005, -0.01, 0.01, 0.004, None, None],
            "turnover": [0.2, 0.1, 0.3, 0.2, None, None],
        }
    )


def test_plot_return_summary_returns_figure() -> None:
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    from yq_analysis.plots import plot_return_summary

    fig = plot_return_summary(_sample_returns(), groups=2, save=False)
    assert isinstance(fig, Figure)


def test_plot_return_summary_can_save_jpg(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    from yq_analysis.plots import plot_return_summary

    fig = plot_return_summary(_sample_returns(), groups=2, save=True, save_dir=tmp_path)
    assert (tmp_path / "sample.jpg").exists()
    assert not plt.fignum_exists(fig.number)
