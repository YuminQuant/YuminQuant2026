from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yq_analysis.io import load_backtest_result


def test_load_backtest_result_reads_optional_barra_exposure(tmp_path: Path) -> None:
    factor_id = "sample"
    factor_root = tmp_path / factor_id
    factor_root.mkdir()

    pd.DataFrame({"factor_id": [factor_id], "portfolio": ["group_1"], "return": [0.01]}).to_parquet(
        factor_root / "returns.parquet"
    )
    pd.DataFrame({"factor_id": [factor_id], "metric": ["barra_ic_mean"], "value": [0.1]}).to_parquet(
        factor_root / "barra_exposure.parquet"
    )
    pd.DataFrame({"factor_id": [factor_id], "index_id": ["000300.SH"], "excess_return": [0.02]}).to_parquet(
        factor_root / "index_group_returns.parquet"
    )

    result = load_backtest_result(tmp_path, factor_id)

    assert result["returns"] is not None
    assert result["ic"] is None
    assert result["factor_stats"] is None
    assert result["barra_exposure"] is not None
    assert result["barra_exposure"]["metric"].tolist() == ["barra_ic_mean"]
    assert result["index_group_returns"] is not None
    assert result["index_group_returns"]["index_id"].tolist() == ["000300.SH"]


def test_load_backtest_result_does_not_fallback_to_metric_first_layout(tmp_path: Path) -> None:
    factor_id = "sample"
    (tmp_path / "returns").mkdir()
    pd.DataFrame({"factor_id": [factor_id], "portfolio": ["group_1"], "return": [0.01]}).to_parquet(
        tmp_path / "returns" / f"{factor_id}.parquet"
    )

    result = load_backtest_result(tmp_path, factor_id)

    assert result["returns"] is None


def test_load_backtest_result_keeps_missing_barra_exposure_optional(tmp_path: Path) -> None:
    result = load_backtest_result(tmp_path, "missing")

    assert result["returns"] is None
    assert result["ic"] is None
    assert result["factor_stats"] is None
    assert result["barra_exposure"] is None
    assert result["index_group_returns"] is None
