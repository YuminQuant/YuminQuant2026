"""Bounded multi-model inference using unchanged refit and input batch identities."""
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import sys

import pandas as pd

from yq_ml_alpha.calendar import TradingCalendar
from yq_ml_alpha.data.dataset import DatasetBuilder
from yq_ml_alpha.output import artifact_io
from yq_ml_alpha.output.artifacts import validate_manifest, window_artifact_path
from yq_ml_alpha.output.factor_metadata import write_factor_metadata_many
from yq_ml_alpha.output.parquet_reader import projected_inputs
from yq_ml_alpha.pipelines.execution import ExecutionOptions, check_memory, execution_scope
from yq_ml_alpha.pipelines.task_plan import load_configs, output_group, prediction_units
from yq_ml_alpha.pipelines.runtime import (
    _context, _load_bundle, _model_class, _predict_model, _prediction_frame,
    _empty_prediction_frame, _predict_dates,
)


def predict_many(config_paths, *, options=None, target_dates=None):
    return predict_many_configs(load_configs(config_paths, 'factor'), options=options, target_dates=target_dates)


def _bytes(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, pd.DataFrame):
        return int(value.memory_usage(index=True, deep=True).sum())
    if hasattr(value, 'nbytes'):
        return int(value.nbytes)
    if callable(getattr(value, 'numel', None)):
        return int(value.numel() * value.element_size())
    if callable(getattr(value, 'state_dict', None)):
        return _bytes(value.state_dict(), seen)
    if isinstance(value, dict):
        return sys.getsizeof(value) + sum(_bytes(k, seen) + _bytes(v, seen) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return sys.getsizeof(value) + sum(_bytes(v, seen) for v in value)
    if hasattr(value, '__dict__'):
        return sys.getsizeof(value) + _bytes(vars(value), seen)
    return sys.getsizeof(value)


def _input_key(config, dates, index):
    cls = _model_class(config.model.class_path)
    if not getattr(cls, 'prediction_inputs_readonly', False):
        return ('isolated', index, tuple(dates))
    from yq_ml_alpha.pipelines.runtime import _uses_sequence_dataset
    identity = dict(features=asdict(config.features), preprocess=asdict(config.preprocess),
        filters=asdict(config.filters), universe=asdict(config.universe), label=asdict(config.label),
        data_root=str(config.data_root), data_version=config.data_version,
        tensor_storage=config.materialize.tensor_storage, dates=dates,
        sequence=(config.model.params.get('sequence_length', 6),
                  config.model.params.get('sequence_frequency', config.sample.train_frequency))
                  if _uses_sequence_dataset(config) else None)
    return json.dumps(identity, sort_keys=True, default=str)


class _Models:
    def __init__(self, configs, options):
        self.configs, self.options = configs, options
        self.entries = OrderedDict()
        self.bytes = 0

    def remove(self, key):
        model, session, cost = self.entries.pop(key)
        session.__exit__(None, None, None)
        self.bytes -= cost

    def get(self, index, window, dataset):
        key = index, window.window_id
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key][0]
        for old in list(self.entries):
            if old[0] == index:
                self.remove(old)
        config = self.configs[index]
        path = window_artifact_path(config.model.artifact_dir, window.window_id)
        validate_manifest(path, config, window, dataset)
        model = _model_class(config.model.class_path).load(path)
        cost = max(_bytes(model), artifact_io.stat(path).st_size)
        if cost > self.options.model_cache_mib * 1024**2:
            raise MemoryError(f'{config.output.id}: model exceeds model_cache_mib')
        while self.entries and (len(self.entries) >= self.options.max_resident_models or
                self.bytes + cost > self.options.model_cache_mib * 1024**2):
            candidate = next((k for k, (m, _, _) in self.entries.items()
                              if getattr(m, 'prediction_replay_safe', False)), None)
            if candidate is None:
                raise MemoryError('Active stateful models cannot be evicted safely; increase model budget '
                                  'or reduce models in this prediction group')
            self.remove(candidate)
        session = model.prediction_session()
        session.__enter__()
        self.entries[key] = model, session, cost
        self.bytes += cost
        check_memory(self.options, 'model load')
        return model

    def close(self):
        for key in list(self.entries):
            self.remove(key)


def _projection_requests(configs, units):
    requests = defaultdict(list)
    for index, _, dates in units:
        config = configs[index]
        if config.features.type not in {'factor_frame', 'raw_panel'}:
            continue
        columns = config.features.columns
        if not isinstance(columns, list) or any(c in {'*', 'all', '__all__'} for c in columns):
            continue
        for date in dates:
            path = Path(config.features.root) / str(date // 10000) / f'{date}.parquet'
            requests[str(artifact_io.read_path(path).absolute())].extend(['trade_date', 'ts_code', *columns])
    return requests


def predict_many_configs(configs, *, options=None, target_dates=None):
    options = options or ExecutionOptions()
    if not configs:
        raise ValueError('At least one prediction config is required')
    groups = defaultdict(list)
    for config in configs:
        groups[output_group(config)].append(config)
    written = []
    with execution_scope(options):
        before = artifact_io.parquet_cache_info()
        if artifact_io.remote_enabled() and any(c.materialize.cache_samples for c in configs):
            raise ValueError('Grouped remote prediction must not materialize samples on node disk')
        if os.environ.get('YQ_METADATA_MANAGED') != '1':
            written.extend(write_factor_metadata_many(configs))
        for group in groups.values():
            written.extend(_predict_group(group, options, target_dates))
        after = artifact_io.parquet_cache_info()
        print('grouped-predict-io ' + json.dumps({name: after[name] - before[name]
              for name in ('opens', 'footer_reads', 'hits', 'misses', 'evictions')}), flush=True)
    return written


def _predict_group(configs, options, target_dates):
    if len({c.output.id for c in configs}) != len(configs):
        raise ValueError('Duplicate model output in one prediction group')
    first = configs[0]
    if any(c.features.type == 'analyst_report_events' for c in configs):
        raise ValueError('Report events require the existing sequential replay workflow')
    if len(configs) > 1 and any(
            not getattr(_model_class(c.model.class_path), 'prediction_replay_safe', False)
            and c.model.params.get('device', 'cpu') != 'cpu' for c in configs):
        raise ValueError('Unreviewed GPU/stateful models retain the single-model workflow; '
                         'do not interleave their device sessions')
    calendar = TradingCalendar.load(first.data_root)
    coverage = {c.output.id: set(_predict_dates(c, calendar)) for c in configs}
    all_dates = sorted(set.union(*coverage.values()))
    owned = set(all_dates if target_dates is None else target_dates) & set(all_dates)
    units = [(i, window, dates) for i, config in enumerate(configs)
             for window, dates in prediction_units(config, calendar) if owned.intersection(dates)]
    units.sort(key=lambda u: (u[2][-1], tuple(u[2]), u[0]))
    unique = set()
    for i, window, dates in units:
        for date in dates:
            if (i, date) in unique:
                raise ValueError(f'Overlapping model windows on {date}: {configs[i].output.id}')
            unique.add((i, date))
    expected = {date: {c.output.id for c in configs if date in coverage[c.output.id]} for date in owned}
    pending = defaultdict(dict)
    computed = {i: {d for mi, _, dates in units if mi == i for d in dates} for i in range(len(configs))}
    for i, config in enumerate(configs):
        for date in (coverage[config.output.id] & owned) - computed[i]:
            pending[date][config.output.id] = _empty_prediction_frame()
    if options.model_predict_workers > 1:
        for config in configs:
            cls = _model_class(config.model.class_path)
            if not getattr(cls, 'prediction_cpu_parallel', False) or config.model.params.get('device', 'cpu') != 'cpu':
                raise ValueError('Concurrent inference requires CPU-parallel-safe models; '
                                 'use model_predict_workers=1 for neural/GPU models')
        if options.model_predict_workers > min(options.threads, options.max_resident_models):
            raise ValueError('model_predict_workers must not exceed threads or max_resident_models')
    from yq_ml_alpha.pipelines.factor import _new_writer as factor_writer
    from yq_ml_alpha.pipelines.model import _new_writer as signal_writer
    writer = factor_writer(first) if first.output.kind == 'factor' else signal_writer(first).writer
    writer.write_workers = options.write_workers
    writer._known_columns.update(c.output.id for c in configs)
    models = _Models(configs, options)
    written = []
    peak = 0
    shared = 0
    requests = _projection_requests(configs, units)

    def flush_ready():
        ready = sorted(d for d in pending if expected[d] <= set(pending[d]))
        if not ready:
            return
        values = defaultdict(list)
        covered = defaultdict(list)
        for date in ready:
            for name, frame in pending[date].items():
                values[name].append(frame)
                covered[name].append(date)
        frames = {name: pd.concat(parts, ignore_index=True) for name, parts in values.items()}
        if sum(_bytes(v) for v in frames.values()) > options.score_buffer_mib * 1024**2:
            raise MemoryError('Ready scores exceed score_buffer_mib; reduce configured prediction batch sizes')
        check_memory(options, 'before daily merge')
        written.extend(writer.write_many(frames, coverage_dates=covered, schema_dates=sorted(owned)))
        for date in ready:
            del pending[date]

    try:
        position = 0
        while position < len(units):
            end = position + 1
            while end < len(units) and units[end][2] == units[position][2]:
                end += 1
            wave = units[position:end]
            by_input = defaultdict(list)
            for unit in wave:
                by_input[_input_key(configs[unit[0]], unit[2], unit[0])].append(unit)
            with projected_inputs(requests):
                for compatible in by_input.values():
                    index, _, dates = compatible[0]
                    dataset = DatasetBuilder(configs[index])
                    bundle = None
                    try:
                        bundle = _load_bundle(configs[index], dataset, calendar, dates, include_label=False)
                        check_memory(options, 'prediction input')
                        shared += max(0, len(compatible) - 1)
                        for start in range(0, len(compatible), options.model_predict_workers):
                            jobs = []
                            for mi, mw, md in compatible[start:start+options.model_predict_workers]:
                                other = dataset if mi == index else DatasetBuilder(configs[mi])
                                try:
                                    if bundle.frame.empty:
                                        path = window_artifact_path(configs[mi].model.artifact_dir, mw.window_id)
                                        validate_manifest(path, configs[mi], mw, other)
                                    model = None if bundle.frame.empty else models.get(mi, mw, other)
                                finally:
                                    if other is not dataset:
                                        other.close(clear_read_cache=False)
                                jobs.append((mi, mw, md, model))
                            def infer(job):
                                mi, mw, md, model = job
                                if model is None:
                                    return mi, _empty_prediction_frame()
                                context = _context(configs[mi], mw.window_id, bundle.feature_columns)
                                score = _predict_model(model, bundle, context)
                                if len(score) != len(bundle.frame) or not score.index.equals(bundle.frame.index):
                                    raise ValueError('Prediction scores must preserve the input row index')
                                result = _prediction_frame(bundle.frame, score)
                                if getattr(model, 'prediction_replay_safe', False):
                                    model.suspend_prediction_device()
                                return mi, result
                            if options.model_predict_workers == 1:
                                results = [infer(job) for job in jobs]
                            else:
                                try:
                                    from threadpoolctl import threadpool_limits
                                    budget = threadpool_limits(limits=max(1, options.threads // options.model_predict_workers))
                                except ImportError:
                                    if any(os.environ.get(name) != '1' for name in
                                           ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):
                                        raise ValueError('Install threadpoolctl or start with BLAS/OMP threads=1 for concurrent inference')
                                    budget = nullcontext()
                                with budget:
                                    with ThreadPoolExecutor(max_workers=options.model_predict_workers) as pool:
                                        results = list(pool.map(infer, jobs))
                            for mi, prediction in results:
                                name = configs[mi].output.id
                                prediction = prediction.loc[prediction.trade_date.isin(owned)].copy()
                                prediction['ts_code'] = prediction.ts_code.astype('category')
                                grouped = {int(d): frame for d, frame in prediction.groupby('trade_date', observed=True, sort=False)}
                                for date in dates:
                                    if date in owned:
                                        pending[date][name] = grouped.get(date, _empty_prediction_frame())
                            del results, jobs
                    finally:
                        if bundle is not None:
                            bundle.close()
                        dataset.close(clear_read_cache=False)
                    peak = max(peak, check_memory(options, 'prediction scores'))
            flush_ready()
            if sum(_bytes(frame) for columns in pending.values() for frame in columns.values()) > options.score_buffer_mib * 1024**2:
                raise MemoryError('Unequal prediction batches exceed score_buffer_mib; increase budget or use compatible batch sizes')
            print(f'grouped-predict batches={end}/{len(units)} shared_bundles={shared} '
                  f'rss_mib={peak/1024**2:.1f}', flush=True)
            position = end
        flush_ready()
        if pending:
            raise ValueError('Incomplete prediction coverage; no silent forward-fill across model windows')
    finally:
        models.close()
        gc.collect()
    return written
