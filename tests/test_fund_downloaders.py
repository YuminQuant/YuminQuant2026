from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from data_manager.downloader.fund.fund_downloader import (
    FundPortfolioDownloader, FundBasicDownloader, PORTFOLIO_FIELDS, BASIC_FIELDS,
    atomic_write, writer_lock,
)


def downloader(tmp_path, cls=FundPortfolioDownloader):
    obj = cls.__new__(cls)
    obj.endpoint = "fund_portfolio" if cls is FundPortfolioDownloader else "fund_basic"
    obj.save_dir = tmp_path
    obj.page_limit = 1000
    obj.max_pages = 50
    obj.retries = 2
    obj.lookback_days = 7
    obj.safe_sleep = Mock()
    obj.logger = Mock()
    obj.pro = Mock()
    return obj


def row(**changes):
    value = dict(zip(PORTFOLIO_FIELDS, ["001753.OF", "20260420", "20260331", "000001.SZ", 100., 10., None, None]))
    value.update(changes)
    return value


def api(obj, pages):
    getattr(obj.pro, obj.endpoint).side_effect = pages


def test_short_pages_offset_actual_and_duplicates(tmp_path):
    obj = downloader(tmp_path)
    api(obj, [pd.DataFrame([row(), row()]), pd.DataFrame([row(symbol="000002.SZ")]), pd.DataFrame()])
    obj.sync(mode="incremental", target_date="20260331")
    assert [c.kwargs["offset"] for c in obj.pro.fund_portfolio.call_args_list] == [0, 2, 3]
    assert obj.safe_sleep.call_count == 3
    result = pd.read_parquet(tmp_path / "2026.parquet")
    assert len(result) == 2
    assert result.stk_float_ratio.isna().all()
    assert str(result.mkv.dtype) == "float64"


def test_repeated_page_and_retry_do_not_publish(tmp_path):
    obj = downloader(tmp_path)
    api(obj, [pd.DataFrame([row()]), pd.DataFrame([row()])])
    with pytest.raises(RuntimeError, match="no progress"):
        obj.sync(mode="incremental", target_date="20260331")
    assert not list(tmp_path.glob("*.parquet"))
    api(obj, [pd.DataFrame([row()]), RuntimeError("failure"), RuntimeError("failure")])
    with patch("data_manager.downloader.fund.fund_downloader.time.sleep"), pytest.raises(RuntimeError, match="failure"):
        obj.sync(mode="incremental", target_date="20260331")
    assert not list(tmp_path.glob("*.parquet"))
    assert not list(tmp_path.glob(".query-*"))
    assert obj.safe_sleep.call_count == 5


def test_conflicting_query_fails_before_any_year_is_published(tmp_path):
    obj = downloader(tmp_path)
    api(obj, [pd.DataFrame([row(end_date="20251231", ann_date="20260101"),
                             row(end_date="20251231", ann_date="20260101", mkv=200)]) , pd.DataFrame()])
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(ValueError, match="Conflicting"):
        obj._query_and_save("20260424", period="20251231")
    assert not list(tmp_path.glob("*.parquet"))


@pytest.mark.parametrize("reverse", [False, True])
def test_portfolio_rounding_duplicates_keep_precise_value_across_pages(tmp_path, reverse):
    obj = downloader(tmp_path)
    rows = [row(ts_code="001875.OF", symbol="00981.HK", end_date="20170630",
                ann_date="20170828", mkv=value, amount=305000., stk_mkv_ratio=3.86)
            for value in [2395676.18, 2395676.00]]
    if reverse:
        rows.reverse()
    api(obj, [pd.DataFrame([rows[0]]), pd.DataFrame([rows[1]]), pd.DataFrame()])
    obj._query_and_save("20260424", period="20170630")
    result = pd.read_parquet(tmp_path / "2017.parquet")
    assert len(result) == 1
    assert result.mkv.tolist() == [2395676.18]
    assert result.amount.tolist() == [305000.]
    assert "rounded mkv duplicate" in obj.logger.warning.call_args.args[0]


@pytest.mark.parametrize("changes", [
    {"mkv": 2395677.}, {"mkv": 2395676., "amount": 11.},
    {"mkv": 2395676., "stk_mkv_ratio": 0.5}, {"mkv": None},
    {"mkv": 2395676.19},
])
def test_rounding_rule_does_not_hide_real_conflicts(tmp_path, changes):
    obj = downloader(tmp_path)
    frame = obj._normalize(pd.DataFrame([row(mkv=2395676.18), row(**changes)]))
    with pytest.raises(ValueError, match="Conflicting.*001753.OF"):
        obj._deduplicate_query(frame)


def test_versions_share_classes_revisions_and_idempotency(tmp_path):
    obj = downloader(tmp_path)
    initial = obj._normalize(pd.DataFrame([row(), row(ann_date="20260421"), row(ts_code="001754.OF")]))
    obj._merge_year(initial, 2026)
    before = pd.read_parquet(tmp_path / "2026.parquet")
    obj._merge_year(initial, 2026)
    after = pd.read_parquet(tmp_path / "2026.parquet")
    assert len(after) == 3
    assert after.first_seen_at.equals(before.first_seen_at)
    assert not (tmp_path / "revisions").exists()
    obj._merge_year(obj._normalize(pd.DataFrame([row(mkv=300.)])), 2026)
    assert len(pd.read_parquet(tmp_path / "2026.parquet")) == 3
    assert pd.read_parquet(tmp_path / "revisions/2026.parquet").mkv.tolist() == [100.]


def test_cutoff_and_ann_year(tmp_path):
    obj = downloader(tmp_path)
    api(obj, [pd.DataFrame([row(end_date="20251231", ann_date="20260201"),
                             row(end_date="20251231", ann_date="20260501")]), pd.DataFrame()])
    obj._query_and_save("20260424", period="20251231")
    assert pd.read_parquet(tmp_path / "2026.parquet").ann_date.tolist() == [20260201]
    assert not (tmp_path / "2025.parquet").exists()


def test_incremental_report_periods_and_history_periods(tmp_path):
    obj = downloader(tmp_path)
    obj._query_and_save = Mock()
    obj.sync(mode="incremental", target_date="20260420")
    obj._query_and_save.assert_not_called()
    obj.sync(mode="incremental", target_date="20260331")
    obj._query_and_save.assert_called_once_with(None, period="20260331")
    obj._query_and_save.reset_mock()
    obj.sync(start_year=2025, end_date="20260424")
    assert [c.kwargs["period"] for c in obj._query_and_save.call_args_list] == ["20250331", "20250630", "20250930", "20251231", "20260331"]


def test_basic_all_markets_status_and_schema(tmp_path):
    obj = downloader(tmp_path, FundBasicDownloader)
    def fetch(**query):
        if query["offset"]:
            return pd.DataFrame()
        record = dict.fromkeys(BASIC_FIELDS)
        record.update(ts_code=f'{query["market"]}{query["status"]}', market=query["market"], status=query["status"])
        return pd.DataFrame([record])
    obj.pro.fund_basic.side_effect = fetch
    obj.sync()
    frame = pd.read_parquet(tmp_path / "fund_basic.parquet")
    assert len(frame) == 6
    assert set(frame.status) == {"D", "I", "L"}
    assert set(BASIC_FIELDS).issubset(frame.columns)
    assert frame.exp_return.isna().all()
    assert len(list((tmp_path / "snapshots").glob("*.parquet"))) == 1


def test_schema_dates_filter_validation(tmp_path):
    obj = downloader(tmp_path)
    with pytest.raises(ValueError, match="Missing fund response"):
        obj._normalize(pd.DataFrame([row()]).drop(columns="mkv"))
    with pytest.raises(ValueError, match="Invalid fund date"):
        obj._normalize(pd.DataFrame([row(ann_date=None)]))
    api(obj, [pd.DataFrame([row(end_date="20251231")]), pd.DataFrame()])
    with pytest.raises(ValueError, match="does not match"):
        obj.sync(mode="incremental", target_date="20260331")


def test_atomic_failure_and_lock(tmp_path):
    path = tmp_path / "test.parquet"
    atomic_write(pd.DataFrame({"x": [1]}), path)
    with patch("data_manager.downloader.fund.fund_downloader.os.replace", side_effect=OSError("fail")):
        with pytest.raises(OSError):
            atomic_write(pd.DataFrame({"x": [2]}), path)
    assert pd.read_parquet(path).x.tolist() == [1]
    assert not list(tmp_path.glob("*.tmp"))
    with writer_lock(tmp_path):
        with pytest.raises(FileExistsError):
            with writer_lock(tmp_path):
                pass
    assert not (tmp_path / ".download.lock").exists()


def test_none_response_and_max_pages_are_errors(tmp_path):
    obj = downloader(tmp_path)
    api(obj, [None, None])
    with patch("data_manager.downloader.fund.fund_downloader.time.sleep"), pytest.raises(RuntimeError, match="None"):
        list(obj._pages(period="20260331"))
    obj.max_pages = 1
    api(obj, [pd.DataFrame([row()])])
    with pytest.raises(RuntimeError, match="maximum page"):
        list(obj._pages(period="20260331"))


def test_basic_failure_preserves_previous_table(tmp_path):
    obj = downloader(tmp_path, FundBasicDownloader)
    path = tmp_path / "fund_basic.parquet"
    atomic_write(pd.DataFrame({"sentinel": [42]}), path)
    record = dict.fromkeys(BASIC_FIELDS)
    record.update(ts_code="001753.OF", market="E", status="D")
    api(obj, [pd.DataFrame([record]), pd.DataFrame(), RuntimeError("failed"), RuntimeError("failed")])
    with patch("data_manager.downloader.fund.fund_downloader.time.sleep"), pytest.raises(RuntimeError):
        obj.sync()
    assert pd.read_parquet(path).sentinel.tolist() == [42]


def test_incremental_groups_defaults_and_dispatch_without_network():
    from types import SimpleNamespace
    from scripts import update_incremental as incremental
    assert "fund_basic" in incremental.GROUPS
    assert "fund_portfolio" in incremental.GROUPS
    assert {g for g in incremental.DEFAULT_GROUPS if g.startswith("fund_")} == {"fund_basic", "fund_portfolio"}
    args = SimpleNamespace(start_date="20260330", end_date="20260401", fund_ann_lookback_days=7)
    with patch.object(incremental, "FundBasicDownloader") as basic, patch.object(incremental, "FundPortfolioDownloader") as portfolio:
        incremental.update_fund_basic(args, Mock())
        incremental.update_fund_portfolio(args, Mock())
        basic.return_value.sync.assert_called_once_with()
        portfolio.return_value.sync.assert_called_once_with(
            mode="incremental", target_date="20260331")


def test_period_refresh_keeps_later_disclosures_and_compresses(tmp_path):
    import pyarrow.parquet as pq
    obj = downloader(tmp_path)
    api(obj, [pd.DataFrame([row(ann_date="20260430")]), pd.DataFrame()])
    obj.sync(mode="incremental", target_date="20260331")
    path = tmp_path / "2026.parquet"
    assert pd.read_parquet(path).ann_date.tolist() == [20260430]
    assert obj.pro.fund_portfolio.call_args_list[0].kwargs["period"] == "20260331"
    assert "ann_date" not in obj.pro.fund_portfolio.call_args_list[0].kwargs
    with pq.ParquetFile(path) as parquet:
        assert all(parquet.metadata.row_group(0).column(i).compression == "ZSTD"
                   for i in range(parquet.metadata.num_columns))


def test_progress_logs_include_pages_empty_and_period_completion(tmp_path):
    obj = downloader(tmp_path)
    api(obj, [pd.DataFrame([row()]), pd.DataFrame()])
    obj.sync(start_year=2026, end_date="20260424")
    messages = "\n".join(c.args[0] for c in obj.logger.info.call_args_list)
    assert "[1/1] period=20260331 start completed=0/1" in messages
    assert "page=1 offset=0 limit=1000 fetched_rows=0 requesting" in messages
    assert "page=1 rows=1 fetched_rows=1" in messages
    assert "page=2 empty result; pagination complete fetched_rows=1" in messages
    assert "retained_rows=1 unique_rows=1" in messages
    assert "period=20260331 complete completed=1/1" in messages


def test_empty_query_logs_and_does_not_write(tmp_path):
    obj = downloader(tmp_path)
    api(obj, [pd.DataFrame()])
    obj.sync(start_year=2026, end_date="20260424")
    messages = "\n".join(c.args[0] for c in obj.logger.info.call_args_list)
    assert "empty result" in messages
    assert "nothing written" in messages
    assert "completed=1/1" in messages
    assert not list(tmp_path.glob("*.parquet"))
