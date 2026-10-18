import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yq_analysis.io import load_strategy_result, load_index_returns
from yq_analysis.strategy import make_strategy_report, diagnose_stock_cash, POSITION_FIELDS, TRADE_FIELDS
from yq_analysis.plots import plot_strategy_summary, plot_strategy_comparison


def snapshots(pnls=(-10., -5., 5.), dates=(20250102, 20250103, 20260102)):
    rows = []
    for date, pnl in zip(dates, pnls):
        row = dict(trade_date=date, trade_time="daily", strategy_id="sample", asset_class="stock",
                   account_pnl=pnl, cash=100 + pnl, net_market_value=0., gross_market_value=0.,
                   unrealized_pnl=0., realized_pnl_cum=pnl, net_realized_pnl_cum=pnl,
                   margin_required=0., position_count=0, buy_count=0, sell_count=0,
                   short_count=0, cover_count=0)
        row.update({key: "[]" for key in [*POSITION_FIELDS, *TRADE_FIELDS]})
        rows.append(row)
    return pd.DataFrame(rows)


def test_initial_loss_and_year_boundary_returns():
    report = make_strategy_report(snapshots(), initial_cash=100)
    daily = report["daily"]
    assert daily["return"].iloc[0] == pytest.approx(-.1)
    assert daily.nav.iloc[-1] == pytest.approx(1.05)
    assert report["portfolio_total"]["max_drawdown(%)"].iloc[0] == pytest.approx(-10)
    assert report["drawdowns"].iloc[0].recovery == pd.Timestamp("2026-01-02")
    assert report["portfolio_by_year"].iloc[1]["cumulative_return(%)"] == pytest.approx((105/95-1)*100)


def test_date_slice_uses_previous_equity_not_initial_cash():
    daily = make_strategy_report(snapshots(), initial_cash=100, start_date=20250103)["daily"]
    assert daily.previous_equity.iloc[0] == 90
    assert daily.nav.iloc[-1] == pytest.approx(105/90)


def test_minute_aggregation_and_unavailable_turnover():
    frame = snapshots(pnls=(0, 1, 2), dates=(20250102,)*3)
    frame.trade_time = ["09:30", "09:31", "15:00"]
    frame.buy_count = [1, 2, 0]
    report = make_strategy_report(frame, initial_cash=100)
    assert len(report["daily"]) == 1
    assert report["daily"].fill_count.iloc[0] == 3
    assert report["daily"].equity.iloc[0] == 102
    assert np.isnan(report["daily"].turnover.iloc[0])


def test_stock_turnover_and_cost_include_both_sides():
    frame = snapshots(pnls=(-1,), dates=(20250102,))
    frame.loc[0, ["buy_count", "sell_count"]] = [1, 1]
    frame.loc[0, "realized_pnl_cum"] = 0
    values = [["A", "B"], [1, -2], [2, 3], [10, 20], [0, 0], [-.4, -.6]]
    for column, value in zip(TRADE_FIELDS, values):
        frame.loc[0, column] = json.dumps(value)
    report = make_strategy_report(frame, initial_cash=100)
    assert report["daily"].turnover.iloc[0] == .8
    assert report["activity"].cost_total.iloc[0] == 1
    frame.loc[0, "trade_prices_json"] = '[null, 20]'
    incomplete = make_strategy_report(frame, initial_cash=100)
    assert np.isnan(incomplete["daily"].turnover.iloc[0])


def test_futures_equity_uses_unrealized_pnl_not_notional():
    frame = snapshots(pnls=(20,), dates=(20250102,))
    frame.asset_class = "future"
    frame.cash = 100
    frame.unrealized_pnl = 20
    frame.realized_pnl_cum = frame.net_realized_pnl_cum = 0
    frame.net_market_value = 1000
    report = make_strategy_report(frame, initial_cash=100)
    assert report["daily"].equity_reconciliation_error.iloc[0] == 0
    assert report["daily"].turnover.isna().all()


def test_benchmark_alignment_and_missing_coverage():
    frame = snapshots()
    benchmark = pd.DataFrame({"trade_date": frame.trade_date, "return": [.01, -.02, .03]})
    report = make_strategy_report(frame, initial_cash=100, benchmark=benchmark)
    assert report["daily"].relative_nav.iloc[-1] == pytest.approx(1.05/(1.01*.98*1.03))
    with pytest.warns(RuntimeWarning, match="incomplete"):
        partial = make_strategy_report(frame, initial_cash=100, benchmark=benchmark.iloc[1:])
    assert "relative_nav" not in partial["daily"]


def test_invalid_inputs_are_not_silently_repaired():
    frame = snapshots()
    with pytest.raises(ValueError, match="Duplicate"):
        make_strategy_report(pd.concat([frame, frame]), initial_cash=100)
    frame.loc[0, "trade_symbols_json"] = '["A"]'
    with pytest.raises(ValueError, match="Mismatched"):
        make_strategy_report(frame, initial_cash=100)
    with pytest.raises(ValueError, match="Non-positive"):
        make_strategy_report(snapshots(pnls=(-101,), dates=(20250102,)), initial_cash=100)


def test_io_and_plot(tmp_path):
    folder = tmp_path / "sample"
    folder.mkdir()
    snapshots().to_parquet(folder / "holdings.parquet")
    report = make_strategy_report(load_strategy_result(tmp_path, "sample"), initial_cash=100)
    output = tmp_path / "strategy.png"
    fig = plot_strategy_summary(report, rolling_window=2, save=output)
    assert output.stat().st_size > 1000
    assert len(fig.axes) == 7
    plt.close(fig)
    index = tmp_path / "000905_SH"
    index.mkdir()
    pd.DataFrame({"trade_date": [20250102], "close": [110], "pre_close": [100]}).to_parquet(index / "2025.parquet")
    assert load_index_returns(tmp_path, "000905.SH", 20250101, 20251231)["return"].iloc[0] == pytest.approx(.1)


def test_comparison_layout_has_shared_nav_drawdown_and_no_benchmark():
    frame = snapshots()
    benchmark = pd.DataFrame({"trade_date": frame.trade_date, "return": [.01, .02, .03]})
    report = make_strategy_report(frame, initial_cash=100, benchmark=benchmark)
    fig = plot_strategy_comparison({"Top 10": report, "Top 20": report, "Top 50": report})
    assert fig.axes[0].get_shared_x_axes().joined(fig.axes[0], fig.axes[1])
    assert fig.axes[2].get_title().startswith("Account PnL")
    assert len(fig.axes) == 9  # five account axes, three heatmaps, one colorbar
    labels = [line.get_label() for line in fig.axes[0].lines]
    assert "Benchmark" not in labels and "Relative wealth" not in labels
    assert not any("year" in ax.get_title().lower() for ax in fig.axes)
    plt.close(fig)


def test_cash_diagnostic_distinguishes_limit_up_and_lot_residual(tmp_path):
    frame = snapshots()
    report = make_strategy_report(frame, initial_cash=100)
    (tmp_path / "calendar").mkdir()
    pd.DataFrame({"cal_date": [20241231, *frame.trade_date], "is_open": [1]*4}).to_parquet(tmp_path / "calendar/trade_cal_SSE.parquet")
    signal_root = tmp_path / "models"
    (signal_root / "2024").mkdir(parents=True)
    pd.DataFrame({"ts_code": ["A.SZ"], "score": [1.0]}).to_parquet(signal_root / "2024/20241231.parquet")
    pv_root = tmp_path / "chn_stock_data/daily/pv/2025"
    pv_root.mkdir(parents=True)
    pd.DataFrame({"ts_code": ["A.SZ"], "open": [100.], "high": [100.], "low": [100.], "close": [100.]}).to_parquet(pv_root / "20250102.parquet")
    flag_root = tmp_path / "chn_stock_data/daily/trade_filter/2025"
    flag_root.mkdir(parents=True)
    flags = pd.DataFrame({"ts_code": ["A.SZ"], "is_limit_up": [True]})
    flags.to_parquet(flag_root / "20250102.parquet")
    kwargs = dict(data_root=tmp_path, signal_root=signal_root, signal_id="score", top_n=1)
    assert diagnose_stock_cash(report, **kwargs).reason.iloc[0] == "all selected stocks limit-up"
    flags.is_limit_up = False
    flags.to_parquet(flag_root / "20250102.parquet")
    assert diagnose_stock_cash(report, **kwargs).reason.iloc[0] == "cash below cheapest executable lot"


def test_dividend_receivable_is_equity_but_not_spendable_cash():
    frame = snapshots(pnls=(0,), dates=(20250102,))
    frame.cash = 40
    frame.net_market_value = frame.gross_market_value = 55
    frame["dividend_receivable"] = 5
    frame["dividend_income_cum"] = 5
    frame.realized_pnl_cum = frame.net_realized_pnl_cum = 5
    frame.unrealized_pnl = -5
    report = make_strategy_report(frame, initial_cash=100)
    assert report["daily"].equity_reconciliation_error.iloc[0] == 0
    assert report["daily"].pnl_reconciliation_error.iloc[0] == 0
    assert report["daily"].cash_ratio.iloc[0] == .4


def test_seven_strategy_comparison_includes_every_curve_and_monthly_panel():
    report = make_strategy_report(snapshots(), initial_cash=100)
    labels = [f"Top {n}" for n in [10, 20, 50, 100, 200, 300, 500]]
    fig = plot_strategy_comparison(dict.fromkeys(labels, report))
    assert len(fig.axes) == 13  # 5 account axes + 7 monthly panels + shared colorbar
    assert set(labels) <= {line.get_label() for line in fig.axes[0].lines}
    assert set(labels) <= {line.get_label() for line in fig.axes[2].lines}
    assert sum("monthly return" in axis.get_title() for axis in fig.axes) == 7
    plt.close(fig)
