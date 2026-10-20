"""Local analyst-report events; forecasts are labels, never text-model inputs."""
from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd

from yq_ml_alpha.data.filters import TradeFilters
from yq_ml_alpha.data.universe import Universe


EVENT_KEY = ["report_date", "ts_code", "org_name", "report_title"]
TEXT_COLUMNS = [*EVENT_KEY, "name", "report_type"]


class AnalystReportDataset:
    """Read annual report_rc files without writing a derived table or disk cache."""

    feature_columns = ["report_title"]

    def __init__(self, config, calendar):
        self.config = config
        self.calendar = calendar
        self.feature_provider = self
        self.universe = Universe(config.universe.id, config.data_root)
        self.filters = TradeFilters(config.filters, config.data_root)
        self.files = {int(p.stem): p for p in config.features.root.glob("*.parquet")
                      if p.stem.isdigit()}
        if not self.files:
            raise FileNotFoundError(f"no annual analyst reports: {config.features.root}")
        self.revision_months = int(config.features.params.get("revision_lookback_months", 12))
        if self.revision_months <= 0:
            raise ValueError("revision_lookback_months must be positive")

    def dataset_identity(self):
        return {
            "format": "report_rc_title_v1",
            "availability": "strict_next_open_after_report_date",
            "label": "current_calendar_year_Q4_np_vs_previous_distinct_report_day",
            "revision_lookback_months": self.revision_months,
            "ambiguous_previous_day": "missing_label",
            "event_key": EVENT_KEY,
        }

    def load(self, dates, *, include_label, split_name):
        started = perf_counter()
        if not dates:
            return pd.DataFrame(columns=[*TEXT_COLUMNS, "trade_date", self.config.label.id])
        # The previous open itself is included: its reports become visible on dates[0].
        lower = self.calendar.previous_open(min(dates))
        if lower is None:
            raise ValueError("report events require calendar history before the first sample")
        read_lower = pd.to_datetime(str(lower))
        if include_label:
            read_lower -= pd.DateOffset(months=self.revision_months)
        columns = TEXT_COLUMNS + (["quarter", "np"] if include_label else [])
        parts = []
        rows = 0
        for year in range(max(read_lower.year, min(self.files)), max(dates) // 10000 + 1):
            if year not in self.files:
                raise FileNotFoundError(f"missing analyst report year: {year}")
            table = pd.read_parquet(self.files[year], columns=columns)
            table = table.loc[
                (table.report_date >= int(read_lower.strftime("%Y%m%d")))
                & (table.report_date < max(dates))
            ].copy()
            parts.append(table)
            rows += len(table)
            print(f"ml-alpha reports {split_name} year={year} rows={len(table)} "
                  f"loaded_rows={rows} elapsed_s={perf_counter() - started:.1f}", flush=True)
        if not parts:
            raise ValueError(f"no source files cover {split_name}")
        frame = pd.concat(parts, ignore_index=True)
        for column in TEXT_COLUMNS:
            if column != "report_date":
                frame[column] = frame[column].fillna("").astype(str).str.strip()
        frame = frame.loc[
            frame.ts_code.ne("") & frame.org_name.ne("") & frame.report_title.ne("")
            & frame.report_type.ne("非个股")
        ].copy()
        if include_label:
            frame = self._attach_revision(frame)
        else:
            frame = frame.drop_duplicates(EVENT_KEY)
        # Publication dates have no intraday timestamps. A weekend/holiday report
        # maps to the next session too; no reports are lost to daily sampling.
        opens = np.asarray(self.calendar.dates, dtype=np.int64)
        positions = np.searchsorted(opens, frame.report_date.to_numpy(), side="right")
        frame = frame.loc[positions < len(opens)].copy()
        frame["trade_date"] = opens[positions[positions < len(opens)]]
        frame = frame.loc[frame.trade_date.isin(dates)].copy()
        if include_label:
            frame = frame.loc[np.isfinite(frame[self.config.label.id])].copy()
            filtered = []
            for date, daily in frame.groupby("trade_date", sort=True):
                filtered.append(self.filter(daily, int(date)))
            if filtered:
                frame = pd.concat(filtered, ignore_index=True)
        frame = frame.sort_values(["trade_date", *EVENT_KEY], kind="mergesort").reset_index(drop=True)
        print(f"ml-alpha reports {split_name} events={len(frame)} "
              f"elapsed_s={perf_counter() - started:.1f}", flush=True)
        return frame

    def _attach_revision(self, frame):
        events = frame.drop_duplicates(EVENT_KEY)[TEXT_COLUMNS].copy()
        forecasts = frame.loc[
            frame.quarter.fillna("").str.match(r"^\d{4}Q4$")
        ].copy()
        forecasts["forecast_year"] = forecasts.quarter.str[:4].astype(int)
        forecasts["np"] = pd.to_numeric(forecasts["np"], errors="coerce")
        forecasts.loc[~np.isfinite(forecasts["np"]), "np"] = np.nan
        keys = [*EVENT_KEY, "forecast_year"]
        conflicts = forecasts.groupby(keys)["np"].nunique()
        if (conflicts > 1).any():
            raise ValueError(f"conflicting forecasts for the same report: {conflicts[conflicts > 1].index[0]}")
        forecasts = forecasts.sort_values("np", na_position="last").drop_duplicates(keys)
        group = ["ts_code", "org_name", "forecast_year"]
        day_key = [*group, "report_date"]
        # Several reports on the same date have no reliable publication order.
        # Only an unambiguous preceding day may supply the comparison forecast.
        daily = forecasts.groupby(day_key)["np"].agg(["first", "nunique"]).reset_index()
        daily.loc[daily["nunique"] != 1, "first"] = np.nan
        daily = daily.sort_values(day_key)
        prior = daily.groupby(group)
        daily["previous_np"] = prior["first"].shift(1)
        daily["previous_date"] = prior["report_date"].shift(1)
        forecasts = forecasts.merge(daily[day_key + ["previous_np", "previous_date"]],
                                    on=day_key, how="left", validate="many_to_one")
        lower = pd.to_datetime(forecasts.report_date.astype(str)) - pd.DateOffset(months=self.revision_months)
        prior_dates = pd.to_datetime(
            forecasts.previous_date.fillna(0).astype("int64").astype(str),
            format="%Y%m%d", errors="coerce",
        )
        usable = ((forecasts.forecast_year == forecasts.report_date // 10000)
                  & forecasts.previous_np.notna() & forecasts.previous_np.ne(0)
                  & prior_dates.ge(lower))
        forecasts[self.config.label.id] = np.nan
        forecasts.loc[usable, self.config.label.id] = (
            (forecasts.loc[usable, "np"] - forecasts.loc[usable, "previous_np"])
            / forecasts.loc[usable, "previous_np"].abs()
        )
        current = forecasts.loc[
            forecasts.forecast_year == forecasts.report_date // 10000,
            [*EVENT_KEY, self.config.label.id],
        ]
        return events.merge(current, on=EVENT_KEY, how="left", validate="one_to_one")

    def filter(self, frame, date):
        return self.filters.apply(self.universe.filter(frame, date), date)


class MonthlyReportScores:
    """Keep each report's original OOS score, then latest broker coverage in 3m."""

    def __init__(self, calendar, start_date, lookback_months=3):
        self.calendar = calendar
        self.start_date = start_date
        self.lookback_months = lookback_months
        self.events = pd.DataFrame(columns=[*EVENT_KEY, "trade_date", "score"])
        self.held = pd.DataFrame(columns=["ts_code", "score"])
        self.month_ends = set()
        for index, date in enumerate(calendar.dates[:-1]):
            if date // 100 != calendar.dates[index + 1] // 100:
                self.month_ends.add(date)

    def add(self, events, scores):
        new = events[[*EVENT_KEY, "trade_date"]].copy()
        new["score"] = np.asarray(scores, dtype=np.float32)
        new = new.loc[np.isfinite(new.score)]
        self.events = pd.concat([self.events, new], ignore_index=True).drop_duplicates(EVENT_KEY, keep="last")

    def at(self, date):
        lower = pd.Timestamp(str(date)) - pd.DateOffset(months=self.lookback_months)
        lower_int = int(lower.strftime("%Y%m%d"))
        if date in self.month_ends:
            eligible = self.events.loc[
                (self.events.report_date > lower_int) & (self.events.trade_date <= date)
            ]
            # Require a complete OOS history, never rescore training reports with
            # a later model simply to populate the first three months.
            if lower_int >= self.start_date and not eligible.empty:
                latest = eligible.groupby(["ts_code", "org_name"]).report_date.transform("max")
                eligible = eligible.loc[eligible.report_date.eq(latest)]
                broker = eligible.groupby(["ts_code", "org_name"]).score.mean()
                self.held = broker.groupby(level="ts_code").mean().rename("score").reset_index()
            else:
                self.held = pd.DataFrame(columns=["ts_code", "score"])
        self.events = self.events.loc[self.events.report_date > lower_int].copy()
        output = self.held.copy()
        output["trade_date"] = date
        return output[["trade_date", "ts_code", "score"]]
