import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_manager.downloader.chn_stock.main_business_downloader import (  # noqa: E402
    MAINBZ_FIELDS,
    MAINBZ_TYPES,
    MainBusinessDownloader,
)
from scripts.update_incremental import mainbz_periods_for_check_date  # noqa: E402


class FakeLogger:
    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


def make_downloader(fake_pro=None, save_dir=None):
    downloader = object.__new__(MainBusinessDownloader)
    downloader.pro = fake_pro
    downloader.page_limit = 100
    downloader.save_dir = str(save_dir) if save_dir is not None else ""
    downloader.task_name = "main business"
    downloader.fields = MAINBZ_FIELDS
    downloader.bz_types = MAINBZ_TYPES
    downloader.sleep_time = 0
    downloader.logger = FakeLogger()
    return downloader


def mainbz_frame(period: str, bz_type: str, count: int, all_empty_code: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": [f"{idx:06d}.SZ" for idx in range(count)],
            "end_date": [period] * count,
            "bz_item": [f"item_{bz_type}_{idx}" for idx in range(count)],
            "bz_code": [None if all_empty_code else f"{bz_type}{idx}" for idx in range(count)],
            "bz_sales": [float(idx) for idx in range(count)],
            "bz_profit": [float(idx) / 10.0 for idx in range(count)],
            "bz_cost": [float(idx) / 2.0 for idx in range(count)],
            "curr_type": ["CNY"] * count,
            "update_flag": [0] * count,
        }
    )


class FakeMainbzPro:
    def __init__(self):
        self.calls = []

    def fina_mainbz_vip(self, **kwargs):
        self.calls.append(kwargs)
        bz_type = kwargs["type"]
        offset = kwargs["offset"]
        period = kwargs["period"]
        if bz_type == "P" and offset == 0:
            return mainbz_frame(period, bz_type, 100, all_empty_code=True)
        if bz_type == "P" and offset == 100:
            return mainbz_frame(period, bz_type, 1, all_empty_code=True)
        if bz_type in {"D", "I"} and offset == 0:
            return mainbz_frame(period, bz_type, 1)
        return pd.DataFrame()


def test_mainbz_checkpoint_period_mapping():
    assert mainbz_periods_for_check_date("20260430") == ["20251231", "20260331"]
    assert mainbz_periods_for_check_date("20260831") == ["20260630"]
    assert mainbz_periods_for_check_date("20261031") == ["20260930"]
    assert mainbz_periods_for_check_date("20260501") == []


def test_mainbz_fetches_three_types_and_paginates():
    fake_pro = FakeMainbzPro()
    downloader = make_downloader(fake_pro=fake_pro)

    frame = downloader._fetch_period("20260331")

    assert set(frame["bz_type"]) == {"P", "D", "I"}
    assert len(frame[frame["bz_type"] == "P"]) == 101
    assert all(call["limit"] == 100 for call in fake_pro.calls)
    assert [(call["type"], call["offset"]) for call in fake_pro.calls] == [
        ("P", 0),
        ("P", 100),
        ("D", 0),
        ("I", 0),
    ]


def test_mainbz_save_preserves_all_empty_columns_and_partitions_by_end_year(tmp_path):
    downloader = make_downloader(save_dir=tmp_path)
    frame = pd.concat(
        [
            mainbz_frame("20260331", "P", 2, all_empty_code=True).assign(bz_type="P"),
            mainbz_frame("20260331", "I", 1, all_empty_code=True).assign(bz_type="I"),
        ],
        ignore_index=True,
    )

    downloader._process_and_save(frame)

    path = tmp_path / "2026.parquet"
    assert path.exists()
    saved = pd.read_parquet(path)
    assert "bz_code" in saved.columns
    assert saved["bz_code"].isna().all()
    assert set(saved["bz_type"]) == {"P", "I"}
