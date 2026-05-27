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


def _sample_barra_exposure() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [20250102, 20250103, 20250102, 20250103, None, None],
            "factor_id": ["sample"] * 6,
            "barra_factor": ["SIZE", "SIZE", "VALUE", "VALUE", "SIZE", "VALUE"],
            "metric": [
                "long_group_exposure",
                "long_group_exposure",
                "long_group_exposure",
                "long_group_exposure",
                "barra_ic_mean",
                "barra_ic_mean",
            ],
            "pair_count": [100, 100, 100, 100, 2, 2],
            "rank_ic_sign": [-1.0, -1.0, -1.0, -1.0, None, None],
            "selected_group": ["group_1", "group_1", "group_1", "group_1", None, None],
            "value": [0.1, 0.2, -0.1, -0.2, 0.12, -0.08],
        }
    )


def _sample_index_group_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [20250102, 20250103] * 3,
            "factor_id": ["sample"] * 6,
            "index_id": ["000300.SH", "000300.SH", "000905.SH", "000905.SH", "000852.SH", "000852.SH"],
            "portfolio": ["group_5"] * 6,
            "excess_return": [0.01, 0.02, -0.01, 0.01, 0.0, 0.005],
        }
    )


def _sample_ic() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "factor_id": ["sample"] * 20,
            "factor_date": [20250102] * 20,
            "horizon": list(range(1, 21)),
            "ic": [0.01] * 20,
            "rank_ic": [0.02] * 20,
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


def test_plot_return_summary_accepts_barra_exposure() -> None:
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    from yq_analysis.plots import plot_return_summary

    fig = plot_return_summary(
        _sample_returns(),
        groups=2,
        barra_exposure=_sample_barra_exposure(),
        index_group_returns=_sample_index_group_returns(),
        ic=_sample_ic(),
        save=False,
    )

    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 9
    titles = {axis.get_title() for axis in fig.axes}
    assert "Rolling 10-period long group Barra exposure" in titles
    assert "Mean factor-Barra Pearson IC" in titles
    assert "Index long group cumulative excess" in titles
    assert "IC decay" in titles
    ic_decay_axis = next(axis for axis in fig.axes if axis.get_title() == "IC decay")
    ic_decay_ticks = [tick.get_text() for tick in ic_decay_axis.get_xticklabels()]
    assert ic_decay_ticks[-2:] == ["5D", "20D"]
    assert ic_decay_axis.get_legend() is None
    assert len(ic_decay_axis.texts) == 6
    labels = [text.get_text() for axis in fig.axes for text in axis.texts]
    assert "0.12" in labels
    assert any(label.endswith("%") for label in labels)
