"""Quarterly analyst/stock memberships and chunked binary attention masks."""
from __future__ import annotations

from bisect import bisect_right
import json
from pathlib import Path

import numpy as np
import pandas as pd

from yq_ml_alpha.data.stores import daily_path
from yq_ml_alpha.output import artifact_io as data_io


VERSION = "analyst_coverage_v1"


def _quarter(date):
    return (date // 10000, (date // 100 % 100 - 1) // 3)


class AnalystCoverage:
    """Use the calendar's expected snapshot; missing new quarters never fall back."""

    def __init__(self, root, window_months=6):
        if isinstance(window_months, bool) or int(window_months) != window_months or window_months <= 0:
            raise ValueError("coverage_window_months must be a positive integer")
        self.root = Path(root)
        self.window_months = int(window_months)
        self.manifest = None
        self.dates = []
        self.snapshot_dates = []
        self._cached_snapshot = None
        self._cached_memberships = None
        self._cached_symbols = None

    def identity(self):
        if self.manifest is None:
            path = self.root / "manifest.json"
            if not data_io.exists(path):
                raise FileNotFoundError(f"missing analyst coverage: {path}; run derive-analyst-coverage")
            manifest = json.loads(data_io.read_text(path, encoding="utf-8"))
            expected = dict(
                version=VERSION, relations=["analyst"], window_months=self.window_months,
                snapshot_frequency="quarterly_end", snapshot_anchor="last_open_session",
                availability="report_date_lte_snapshot", analyst_identity="institution_split_author",
            )
            if any(manifest.get(key) != value for key, value in expected.items()):
                raise ValueError("analyst coverage dataset version/window/snapshot policy mismatch")
            self.manifest = manifest
        return self.manifest

    def set_calendar_dates(self, dates):
        dates = list(dates)
        if not dates or dates != sorted(set(dates)):
            raise ValueError("analyst coverage requires a sorted, unique trading calendar")
        # A date in the next quarter confirms that this is a completed quarter.
        # Never infer an endpoint from the end of a request or file inventory.
        self.dates = dates
        self.snapshot_dates = [left for left, right in zip(dates, dates[1:])
                               if _quarter(left) != _quarter(right)]
        self._cached_snapshot = None
        self._cached_memberships = None
        self._cached_symbols = None

    def snapshot_date(self, date):
        date = int(date)
        if not self.dates or not self.dates[0] <= date <= self.dates[-1]:
            raise ValueError(f"trading calendar does not cover analyst snapshot request {date}")
        index = bisect_right(self.snapshot_dates, date) - 1
        if index < 0:
            raise ValueError(f"no completed quarterly analyst snapshot before {date}; extend calendar history")
        return self.snapshot_dates[index]

    def _memberships(self, date):
        self.identity()
        snapshot = self.snapshot_date(date)
        if self._cached_snapshot != snapshot:
            path = daily_path(self.root, snapshot)
            if not data_io.exists(path):
                raise FileNotFoundError(
                    f"missing analyst coverage snapshot {snapshot} for {date}: {path}; "
                    "run derive-analyst-coverage including the preceding quarter"
                )
            frame = data_io.read_parquet(path, columns=["trade_date", "relation", "group_id", "ts_code"])
            if (frame.isna().any().any() or not frame.trade_date.eq(snapshot).all()
                    or not frame.relation.eq("analyst").all()
                    or frame.duplicated(["group_id", "ts_code"]).any()):
                raise ValueError(f"invalid analyst coverage memberships: {path}")
            for column in ("group_id", "ts_code"):
                if not frame[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
                    raise ValueError(f"invalid analyst coverage {column}: {path}")
            self._cached_snapshot = snapshot
            self._cached_memberships = frame
            self._cached_symbols = frozenset(frame.ts_code)
        return self._cached_memberships

    def covered_symbols(self, date):
        self._memberships(date)
        return set(self._cached_symbols)

    def graph(self, date, symbols):
        frame = self._memberships(date)
        symbols = list(symbols)
        if len(set(symbols)) != len(symbols):
            raise ValueError("duplicate analyst coverage graph nodes")
        missing = set(symbols).difference(self._cached_symbols)
        if missing:
            raise ValueError(f"analyst graph contains uncovered stocks on {date}: {sorted(missing)[:5]}")
        return BinaryCoverageGraph(frame, symbols)


class BinaryCoverageGraph:
    """Union shared-analyst neighbors without storing stock-pair edges."""

    def __init__(self, memberships, symbols):
        self.symbols = list(symbols)
        self.node_count = len(self.symbols)
        if len(set(self.symbols)) != self.node_count:
            raise ValueError("duplicate analyst coverage graph nodes")
        positions = {symbol: index for index, symbol in enumerate(self.symbols)}
        selected = memberships.loc[memberships.ts_code.isin(positions)]
        members_by_group = []
        groups_by_node = [[] for _ in self.symbols]
        for _, rows in selected.groupby("group_id", sort=True):
            members = rows.ts_code.map(positions).to_numpy(dtype="int64")
            group = len(members_by_group)
            members_by_group.append(members)
            for node in members:
                groups_by_node[node].append(group)
        self._members_by_group = tuple(members_by_group)
        self._groups_by_node = tuple(tuple(groups) for groups in groups_by_node)

    def mask(self, start, stop):
        if not 0 <= start <= stop <= self.node_count:
            raise ValueError("invalid analyst attention query range")
        mask = np.zeros((stop - start, self.node_count), dtype=np.bool_)
        for row, node in enumerate(range(start, stop)):
            for group in self._groups_by_node[node]:
                mask[row, self._members_by_group[group]] = True
            # Covered nodes remain valid even if all their other stocks were filtered.
            mask[row, node] = True
        return mask
