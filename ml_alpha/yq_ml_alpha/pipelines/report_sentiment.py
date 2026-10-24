"""Report-event adapter using shared windows, artifacts and daily factor output."""
from __future__ import annotations
from yq_ml_alpha.output import artifact_io as data_io

from dataclasses import replace

import pandas as pd

from yq_ml_alpha.calendar import TradingCalendar
from yq_ml_alpha.data.report_events import AnalystReportDataset, MonthlyReportScores
from yq_ml_alpha.output.artifacts import save_manifest, validate_manifest, window_artifact_path
from yq_ml_alpha.output.daily_wide_writer import DailyWideWriter
from yq_ml_alpha.output.factor_metadata import write_factor_metadata
from yq_ml_alpha.pipelines.runtime import _context, _model_class, build_windows


def execute(config, *, train, predict, resume=False):
    _validate(config)
    model_class = _model_class(config.model.class_path)
    check = getattr(model_class, "check_dependencies", None)
    if callable(check):
        check()
    calendar = TradingCalendar.load(config.data_root)
    dataset = AnalystReportDataset(config, calendar)
    # Replaying prior OOS windows reconstructs both the three-month report book
    # and the previous month's held factor for a partial prediction request.
    # It never scores those old reports with the new year's fitted model.
    extended = _history_config(config)
    windows = build_windows(extended, calendar)
    if not windows:
        raise ValueError("no report training/prediction windows")
    writer = DailyWideWriter(
        config.output.root, config.output.id, layout="standard",
        asset=config.output.asset, frequency=config.output.frequency,
        base_root=config.output.base_root, write_workers=config.output.write_workers,
    ) if predict else None
    output_dates = calendar.between(*config.dates.predict)
    first_event_date = calendar.previous_open(windows[0].predict_dates[0])
    book = MonthlyReportScores(calendar, first_event_date,
                               int(config.features.params.get("aggregation_months", 3)))
    paths = []
    if predict:
        metadata = write_factor_metadata(config)
        if metadata is not None:
            paths.append(metadata)
    for index, window in enumerate(windows, 1):
        print(f"ml-alpha reports [{index}/{len(windows)}] {window.window_id} "
              f"cutoff={window.training_cutoff_date}", flush=True)
        path = window_artifact_path(config.model.artifact_dir, window.window_id)
        context = _context(config, window.window_id, dataset.feature_columns)
        if train and not (resume and data_io.exists(path)):
            train_frame = dataset.load(window.train_dates, include_label=True, split_name="train")
            valid_frame = dataset.load(window.valid_dates, include_label=True, split_name="valid")
            if train_frame.empty or valid_frame.empty:
                raise ValueError(f"{window.window_id}: empty train or validation report events")
            model = model_class()
            model.fit(train_frame, valid_frame, context)
            del train_frame, valid_frame
            model.save(path)
            save_manifest(path, config, window, dataset)
            paths.append(path)
            if config.diagnostics.enabled:
                paths.extend(model.write_diagnostics(context))
        else:
            validate_manifest(path, config, window, dataset)
            model = model_class.load(path) if predict else None
        if predict:
            # One projected annual read, then bounded in-memory score/write batches.
            # Forecast fields are never read for prediction.
            events = dataset.load(window.predict_dates, include_label=False, split_name="predict")
            size = config.materialize.predict_batch_size
            for start in range(0, len(window.predict_dates), size):
                dates = window.predict_dates[start:start + size]
                batch = events.loc[events.trade_date.isin(dates)]
                if not batch.empty:
                    book.add(batch, model.predict(batch, context))
                outputs = []
                coverage = []
                for date in dates:
                    daily = book.at(date)
                    if config.dates.predict[0] <= date <= config.dates.predict[1]:
                        daily = dataset.filter(daily, date)
                        # A covered but delisted stock must not be resurrected in
                        # a daily output just because its report is still recent.
                        base_path = config.output.base_root / str(date // 10000) / f"{date}.parquet"
                        codes = data_io.read_parquet(base_path, columns=["ts_code"])["ts_code"]
                        daily = daily.loc[daily.ts_code.isin(codes)]
                        outputs.append(daily)
                        coverage.append(date)
                if coverage:
                    paths.extend(writer.write(pd.concat(outputs, ignore_index=True),
                                              coverage_dates=coverage, schema_dates=output_dates))
                print(f"ml-alpha reports predict {dates[0]}..{dates[-1]} "
                      f"events={len(batch)} output_dates={len(coverage)}", flush=True)
            del events
        del model
    return paths


def _history_config(config):
    # Whole-year identities are stable even if dates.predict is narrowed later.
    start = pd.Timestamp(str(config.dates.predict[0]))
    months = int(config.features.params.get("aggregation_months", 3))
    first = pd.Timestamp(str(config.dates.train[0])) + pd.DateOffset(
        months=int(config.train_scheme.train_lookback[:-1]))
    history = start - pd.DateOffset(months=months + 1)
    history_year = max(first.year, history.year)
    lower = history_year * 10000 + 101
    return replace(config, dates=replace(config.dates, predict=(lower, config.dates.predict[1])))


def _validate(config):
    if (config.output.kind != "factor" or config.output.asset != "stock"
            or config.output.frequency != "daily"):
        raise ValueError("report sentiment currently requires a daily stock factor output")
    if config.sample.train_frequency != "daily" or config.sample.predict_frequency != "daily":
        raise ValueError("report events require daily availability dates; do not subsample publications")
    if (config.train_scheme.type != "rolling"
            or config.train_scheme.validation_lookback is None
            or not str(config.train_scheme.train_lookback).endswith("m")
            or config.train_scheme.validation_unit != "dates"):
        raise ValueError("report events require annual calendar-month train/validation windows")
    if config.dates.predict is None or config.label.lookahead_days != 0:
        raise ValueError("report revision is a publication-time label with lookahead_days=0")
    if config.label.id != "report_np_revision_current_fy":
        raise ValueError("unsupported report revision label")
    if config.materialize.cache_samples or config.preprocess.cross_section_transform != "none":
        raise ValueError("report events require cache_samples=false and transform=none")
    if list(config.features.columns) != ["report_title"]:
        raise ValueError("the title-only RPTF adapter requires columns=['report_title']")
    if int(config.features.params.get("aggregation_months", 3)) <= 0:
        raise ValueError("aggregation_months must be positive")
