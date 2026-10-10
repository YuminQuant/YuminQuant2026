"""Fund sources: bounded queries, disclosure versions and schema-preserving writes."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import time
import uuid

import numpy as np
import pandas as pd

from data_manager.core import BaseDownloader, ConfigManager
from data_manager.downloader.chn_stock.fin_statement_downloader import (
    QUARTER_SUFFIXES, _concat_preserve_schema,
)

PORTFOLIO_FIELDS = "ts_code,ann_date,end_date,symbol,mkv,amount,stk_mkv_ratio,stk_float_ratio".split(",")
PORTFOLIO_KEY = ["ts_code", "symbol", "end_date", "ann_date"]
BASIC_FIELDS = ("ts_code,name,management,custodian,fund_type,found_date,due_date,list_date,"
                "issue_date,delist_date,issue_amount,m_fee,c_fee,duration_year,p_value,"
                "min_amount,exp_return,benchmark,status,invest_type,type,trustee,"
                "purc_startdate,redm_startdate,market").split(",")
BASIC_NUMBERS = "issue_amount,m_fee,c_fee,duration_year,p_value,min_amount,exp_return".split(",")
OBSERVED = ["first_seen_at", "last_seen_at"]


def parse_date(value):
    text = str(value)
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"Expected YYYYMMDD, got {value!r}")
    return datetime.strptime(text, "%Y%m%d")


def today():
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d")


def normalize(frame, fields, numeric, required_dates=(), required_text=()):
    missing = set(fields) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing fund response columns: {sorted(missing)}")
    result = frame[fields].copy()
    for column in fields:
        if column.endswith("date"):
            source = result[column].astype("string").str.strip().str.replace("-", "", regex=False)
            source = source.str.replace(r"\.0$", "", regex=True)
            dates = pd.to_datetime(source, format="%Y%m%d", errors="coerce")
            invalid = source.notna() & source.ne("") & (dates.isna() | ~source.str.fullmatch(r"\d{8}", na=False))
            if invalid.any() or (column in required_dates and dates.isna().any()):
                raise ValueError(f"Invalid fund date: {column}")
            result[column] = pd.to_numeric(dates.dt.strftime("%Y%m%d")).astype("Int32")
        elif column in numeric:
            result[column] = pd.to_numeric(result[column], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")
        else:
            result[column] = result[column].astype("string").str.strip().replace("", pd.NA)
            if column in required_text and result[column].isna().any():
                raise ValueError(f"Missing fund key: {column}")
    return result


def unique_query(frame, key, fields):
    result = frame.drop_duplicates(subset=fields)
    if result.duplicated(key).any():
        raise ValueError(f"Conflicting fund rows within one query: {key}")
    return result


@contextmanager
def writer_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".download.lock"
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, str(os.getpid()).encode("ascii"))
        os.close(fd)
        fd = None
        yield
    finally:
        if fd is not None:
            os.close(fd)
        path.unlink()


def atomic_write(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_parquet(tmp, index=False)
        import pyarrow.parquet as pq
        with pq.ParquetFile(tmp) as saved:
            if saved.metadata.num_rows != len(frame) or saved.schema_arrow.names != list(frame.columns):
                raise ValueError(f"Parquet validation failed: {path}")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class _FundDownloader(BaseDownloader):
    def __init__(self, endpoint, path_key, default_path, page_default, rate_default):
        config = ConfigManager().config
        super().__init__(rate_limit=config.get("api", {}).get("rate_limits", {}).get(endpoint, rate_default))
        self.endpoint = endpoint
        self.page_limit = int(config.get("api", {}).get("page_limits", {}).get(endpoint, page_default))
        if self.page_limit <= 0:
            raise ValueError("Fund page limit must be positive")
        settings = config.get("fund_download", {})
        self.max_pages = int(settings.get("max_pages", 100000))
        self.retries = int(settings.get("retries", 3))
        self.lookback_days = int(settings.get("ann_lookback_days", 7))
        if self.max_pages <= 0 or self.retries <= 0 or self.lookback_days < 0:
            raise ValueError("Invalid fund retry/page/lookback settings")
        self.save_dir = Path(self.base_data_dir) / config["paths"].get(path_key, default_path)

    def _pages(self, **query):
        offset, seen = 0, set()
        for _ in range(self.max_pages):
            for attempt in range(self.retries):
                self.safe_sleep()  # Every request, including retries, consumes the rate budget.
                try:
                    page = getattr(self.pro, self.endpoint)(**query, fields=",".join(self.fields),
                                                             limit=self.page_limit, offset=offset)
                    if page is None:
                        raise RuntimeError("Fund API returned None, not an empty data frame")
                    break
                except Exception:
                    if attempt + 1 == self.retries:
                        raise
                    time.sleep(min(2 ** attempt, 8))
            if page.empty:
                return
            normalized = self._normalize(page)
            hashes = set(pd.util.hash_pandas_object(normalized, index=False).tolist())
            if not hashes - seen:
                raise RuntimeError(f"{self.endpoint}: pagination made no progress at offset={offset}")
            seen.update(hashes)
            yield normalized
            offset += len(page)  # Short pages are not proof of completion.
        raise RuntimeError(f"{self.endpoint}: maximum page count exceeded")


class FundBasicDownloader(_FundDownloader):
    fields = BASIC_FIELDS

    def __init__(self):
        super().__init__("fund_basic", "fund_basic_dir", "fund_data/basic", 15000, 180)
        self.page_limit = min(self.page_limit, 15000)

    def _normalize(self, frame):
        return normalize(frame, self.fields, BASIC_NUMBERS, required_text=("ts_code", "market", "status"))

    def sync(self):
        with writer_lock(self.save_dir):
            frames = []
            for market in ("E", "O"):
                for status in ("D", "I", "L"):
                    for page in self._pages(market=market, status=status):
                        if not (page.market.eq(market) & page.status.eq(status)).all():
                            raise ValueError("fund_basic response does not match market/status")
                        frames.append(page)
            if not frames:
                raise RuntimeError("Empty fund universe: refusing to overwrite basic data")
            result = unique_query(_concat_preserve_schema(frames), ["ts_code"], self.fields)
            result = result.sort_values("ts_code").reset_index(drop=True)
            result["fetch_date"] = np.int32(today())
            atomic_write(result, self.save_dir / "snapshots" / f"{today()}.parquet")
            atomic_write(result, self.save_dir / "fund_basic.parquet")
            self.logger.info(f"fund_basic saved {len(result)} rows")


class FundPortfolioDownloader(_FundDownloader):
    fields = PORTFOLIO_FIELDS

    def __init__(self):
        super().__init__("fund_portfolio", "fund_portfolio_dir", "fund_data/portfolio", 1000, 180)

    def _normalize(self, frame):
        result = normalize(frame, self.fields, self.fields[4:], ("ann_date", "end_date"), ("ts_code", "symbol"))
        if result.ann_date.lt(result.end_date).any():
            raise ValueError("Fund disclosure precedes report period")
        return result

    def _merge_year(self, incoming, year):
        path = self.save_dir / f"{year}.parquet"
        stamp = datetime.now(timezone.utc).isoformat()
        incoming = incoming.copy()
        incoming["first_seen_at"] = stamp
        incoming["last_seen_at"] = stamp
        if path.exists():
            old = pd.read_parquet(path)
            if not set(self.fields + OBSERVED).issubset(old.columns):
                raise ValueError(f"Invalid existing fund schema: {path}")
            old_index = old.set_index(PORTFOLIO_KEY, verify_integrity=True)
            new_index = incoming.set_index(PORTFOLIO_KEY, verify_integrity=True)
            common = old_index.index.intersection(new_index.index)
            values = [c for c in self.fields if c not in PORTFOLIO_KEY]
            left, right = old_index.loc[common, values], new_index.loc[common, values]
            equal = (left.eq(right) | (left.isna() & right.isna())).all(axis=1)
            unchanged = common[equal.to_numpy()]
            new_index.loc[unchanged, "first_seen_at"] = old_index.loc[unchanged, "first_seen_at"]
            revised = old_index.loc[common[~equal.to_numpy()]].reset_index()
            if not revised.empty:
                audit_path = self.save_dir / "revisions" / f"{year}.parquet"
                if audit_path.exists():
                    revised = _concat_preserve_schema([pd.read_parquet(audit_path), revised])
                revised = revised.drop_duplicates(self.fields + ["first_seen_at"])
                atomic_write(revised, audit_path)
                self.logger.warning(f"fund_portfolio revised existing keys in {year}; archived old observations")
            incoming = _concat_preserve_schema([old_index.drop(new_index.index, errors="ignore").reset_index(),
                                               new_index.reset_index()])
        incoming = incoming[self.fields + OBSERVED].sort_values(PORTFOLIO_KEY).reset_index(drop=True)
        atomic_write(incoming, path)
        self.logger.info(f"fund_portfolio saved {len(incoming)} rows: {path}")

    def _query_and_save(self, cutoff, **query):
        # Stage a complete query before publishing; only one announcement year is merged at a time.
        with TemporaryDirectory(prefix=".query-", dir=self.save_dir) as folder:
            staged = Path(folder)
            for page_idx, page in enumerate(self._pages(**query)):
                for key in ("period", "ann_date"):
                    column = "end_date" if key == "period" else key
                    if key in query and not page[column].eq(int(query[key])).all():
                        raise ValueError(f"fund_portfolio response does not match {key}")
                page = page[page.ann_date.le(int(cutoff)) & page.end_date.le(int(cutoff))]
                for year, rows in page.groupby(page.ann_date // 10000):
                    directory = staged / str(year)
                    directory.mkdir(exist_ok=True)
                    rows.to_parquet(directory / f"{page_idx}.parquet", index=False)
            # Validate every partition before modifying any destination.
            for directory in sorted(staged.iterdir()):
                frame = _concat_preserve_schema([pd.read_parquet(p) for p in directory.glob("*.parquet")])
                frame = unique_query(frame, PORTFOLIO_KEY, self.fields)
                frame.to_parquet(staged / f"{directory.name}.validated", index=False)
            for path in sorted(staged.glob("*.validated")):
                self._merge_year(pd.read_parquet(path), int(path.stem))

    def sync(self, mode="historical", start_year=2009, start_date=None, end_date=None, lookback_days=None):
        end = parse_date(end_date or today())
        with writer_lock(self.save_dir):
            if mode == "historical":
                if not 1900 <= int(start_year) <= end.year:
                    raise ValueError("Invalid history start year")
                for year in range(int(start_year), end.year + 1):
                    for suffix in QUARTER_SUFFIXES:
                        period = f"{year}{suffix}"
                        if parse_date(period) <= end:
                            self._query_and_save(end.strftime("%Y%m%d"), period=period)
            elif mode == "incremental":
                begin = parse_date(start_date or end.strftime("%Y%m%d"))
                days = self.lookback_days if lookback_days is None else int(lookback_days)
                if begin > end or days < 0:
                    raise ValueError("Invalid incremental date range/lookback")
                begin -= timedelta(days=days)
                while begin <= end:
                    self._query_and_save(end.strftime("%Y%m%d"), ann_date=begin.strftime("%Y%m%d"))
                    begin += timedelta(days=1)
            else:
                raise ValueError(f"Unknown fund download mode: {mode}")
