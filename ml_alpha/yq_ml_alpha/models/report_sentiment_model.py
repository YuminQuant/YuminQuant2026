"""DFZQ report 86 RPTF: title log term-frequency -> earnings revision."""
from __future__ import annotations

from yq_ml_alpha.output import artifact_io

from collections import Counter
from dataclasses import replace
from importlib.metadata import version
import json
import re

import numpy as np
import pandas as pd

from yq_ml_alpha.data.report_events import EVENT_KEY, AnalystReportDataset
from yq_ml_alpha.models.base import AlphaModel


_POS = frozenset(("d", "vn", "t", "vd", "an", "ad", "v", "a", "nz", "nr", "n"))
_TOKENIZER = None


def _tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            import jieba
            import jieba.posseg
        except ImportError as exc:
            raise ImportError("RPTF requires jieba==0.42.1; install the ml_alpha 'text' extra") from exc
        if version("jieba") != "0.42.1":
            raise RuntimeError("RPTF tokenizer is pinned to jieba==0.42.1")
        tokenizer = jieba.Tokenizer()
        # Jieba.initialize normally writes a pickle under the OS temp directory.
        # Build the prefix dictionary in memory instead; no cache anywhere.
        with tokenizer.get_dict_file() as dictionary:
            tokenizer.FREQ, tokenizer.total = tokenizer.gen_pfdict(dictionary)
        tokenizer.initialized = True
        _TOKENIZER = jieba.posseg.POSTokenizer(tokenizer)
    return _TOKENIZER


def _tokens(frame):
    tokenizer = _tokenizer()
    result = []
    for title, name, code in frame[["report_title", "name", "ts_code"]].itertuples(index=False, name=None):
        text = str(title)
        if name:
            text = text.replace(str(name), "")
        text = text.replace(str(code), "").replace(str(code).split(".")[0], "")
        text = re.sub(r"[\x00-\x1f]+", " ", text).strip(" ：:，,（）()")
        result.append([token.word for token in tokenizer.cut(text)
                       if token.flag in _POS and token.word.strip()])
    return result


class ReportTermFrequencyAlphaModel(AlphaModel):
    @classmethod
    def run_factor_workflow(cls, config, *, train, predict, resume=False, window_ids=None):
        return _execute(cls, config, train=train, predict=predict, resume=resume, window_ids=window_ids)

    @staticmethod
    def planning_config(config):
        return _history_config(config)

    @staticmethod
    def materialize_error():
        return "report events are read directly; factor-materialize is not supported"

    @classmethod
    def check_dependencies(cls):
        # Fail before reading years of source data, without initializing a
        # dictionary or creating an OS-temp cache.
        from importlib.util import find_spec

        missing = [name for name in ("jieba", "scipy", "xgboost") if find_spec(name) is None]
        if missing:
            raise ImportError(f"RPTF requires ml_alpha text dependencies; missing: {', '.join(missing)}")
        if version("jieba") != "0.42.1":
            raise RuntimeError("RPTF tokenizer is pinned to jieba==0.42.1")

    def fit(self, train_data, valid_data, context):
        import xgboost as xgb
        from scipy.stats import norm, rankdata

        if context.model_search:
            raise ValueError("RPTF uses fixed parameters, not a search configuration")
        tokens = _tokens(train_data)
        keep = np.asarray([bool(words) for words in tokens])
        target = train_data[context.label_column].to_numpy(dtype=np.float64)
        keep &= np.isfinite(target)
        tokens = [words for words, usable in zip(tokens, keep) if usable]
        target = target[keep]
        if len(target) < 2:
            raise ValueError("RPTF requires at least two labeled, nonempty report titles")
        self.target_values = np.unique(target)
        self.target_quantiles = (rankdata(target, method="average") - 0.5) / len(target)
        order = np.argsort(target, kind="mergesort")
        _, first = np.unique(target[order], return_index=True)
        self.target_quantiles = self.target_quantiles[order][first]
        counts = Counter(word for words in tokens for word in words)
        params = dict(context.model_params)
        size = int(params.pop("max_title_features", 1000))
        self.prediction_batch_size = int(params.pop("prediction_batch_size", 4096))
        if self.prediction_batch_size <= 0:
            raise ValueError("prediction_batch_size must be positive")
        if size <= 0:
            raise ValueError("max_title_features must be positive")
        self.vocabulary = {word: index for index, (word, _) in enumerate(
            sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:size])}
        if not self.vocabulary:
            raise ValueError("empty training vocabulary")
        self.tokenizer_version = version("jieba")
        self.estimator = xgb.XGBRegressor(**params)
        x = self._matrix(tokens)
        y = norm.ppf(np.interp(target, self.target_values, self.target_quantiles))
        print(f"ml-alpha rptf fit reports={len(y)} vocabulary={x.shape[1]} "
              f"matrix_mib={x.nbytes / 2**20:.1f}", flush=True)
        self.estimator.fit(x, y, verbose=False)
        self.info = {"train_reports": len(y), "vocabulary_size": len(self.vocabulary),
                     "tokenizer": f"jieba=={self.tokenizer_version}",
                     "xgboost_version": xgb.__version__, "params": params,
                     "train_mse": float(np.mean((self.estimator.predict(x) - y) ** 2))}
        del x
        if not valid_data.empty:
            valid_tokens = _tokens(valid_data)
            valid_keep = np.asarray([bool(words) for words in valid_tokens])
            raw = valid_data[context.label_column].to_numpy(dtype=np.float64)
            valid_keep &= np.isfinite(raw)
            if valid_keep.any():
                vy = norm.ppf(np.interp(raw[valid_keep], self.target_values, self.target_quantiles))
                self.info["valid_reports"] = int(valid_keep.sum())
                prediction = self._predict_tokens(
                    [words for words, usable in zip(valid_tokens, valid_keep) if usable])
                self.info["valid_mse"] = float(np.mean((prediction - vy) ** 2))

    def _matrix(self, tokens):
        # Numeric zero is a word's absence, not an XGBoost missing value.
        # Fill directly to avoid an additional CSR and corpus-size Python lists.
        matrix = np.zeros((len(tokens), len(self.vocabulary)), dtype=np.float32)
        for row, words in enumerate(tokens):
            counts = Counter(self.vocabulary[word] for word in words if word in self.vocabulary)
            if counts:
                matrix[row, list(counts)] = np.log1p(list(counts.values()))
        return matrix

    def _predict_tokens(self, tokens):
        parts = []
        for start in range(0, len(tokens), self.prediction_batch_size):
            parts.append(self.estimator.predict(
                self._matrix(tokens[start:start + self.prediction_batch_size])))
        return np.concatenate(parts)

    def predict(self, data, context):
        if version("jieba") != self.tokenizer_version:
            raise ValueError("jieba version changed since training; retrain or restore the tokenizer")
        result = np.full(len(data), np.nan, dtype=np.float32)
        if not data.empty:
            tokens = _tokens(data)
            keep = np.asarray([bool(words) for words in tokens])
            if keep.any():
                result[keep] = self._predict_tokens(
                    [words for words, usable in zip(tokens, keep) if usable])
        return pd.Series(result, index=data.index)

    def write_diagnostics(self, context):
        artifact_io.mkdir(context.artifact_dir, parents=True, exist_ok=True)
        info = context.artifact_dir / "model_info.json"
        vocab = context.artifact_dir / "vocabulary.json"
        artifact_io.write_text(info, json.dumps(self.info, ensure_ascii=False, indent=2), encoding="utf-8")
        artifact_io.write_text(vocab, json.dumps(self.vocabulary, ensure_ascii=False, indent=2), encoding="utf-8")
        return [info, vocab]


def _execute(model_class, config, *, train, predict, resume=False, window_ids=None):
    from yq_ml_alpha.calendar import TradingCalendar
    from yq_ml_alpha.output.artifacts import save_manifest, validate_manifest, window_artifact_path
    from yq_ml_alpha.output.daily_wide_writer import DailyWideWriter
    from yq_ml_alpha.output.factor_metadata import write_factor_metadata
    from yq_ml_alpha.pipelines.runtime import _context, build_windows
    from yq_ml_alpha.pipelines.training_tasks import select_windows

    if window_ids is not None and predict:
        raise ValueError("Window selection is training-only; report prediction requires sequential replay")
    _validate(config)
    check = getattr(model_class, "check_dependencies", None)
    if callable(check):
        check()
    calendar = TradingCalendar.for_config(config)
    dataset = AnalystReportDataset(config, calendar)
    # Replaying prior OOS windows reconstructs both the three-month report book
    # and the previous month's held factor for a partial prediction request.
    # It never scores those old reports with the new year's fitted model.
    extended = _history_config(config)
    windows = build_windows(extended, calendar)
    windows = select_windows(windows, window_ids)
    if not windows:
        raise ValueError("no report training/prediction windows")
    writer = DailyWideWriter(
        config.output.root, config.output.id, layout="standard",
        asset=config.output.asset, frequency=config.output.frequency,
        base_root=config.output.base_root, write_workers=config.output.write_workers,
        file_suffix="_ml",
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
        if train and not (resume and artifact_io.exists(path)):
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
                        codes = artifact_io.read_parquet(base_path, columns=["ts_code"])["ts_code"]
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
