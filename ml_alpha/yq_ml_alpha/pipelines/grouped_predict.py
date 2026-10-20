"""Sequential model inference, coalesced daily writes; never train a model."""
import gc

import pandas as pd

from yq_ml_alpha.calendar import TradingCalendar
from yq_ml_alpha.data.dataset import DatasetBuilder
from yq_ml_alpha.output.artifacts import validate_manifest, window_artifact_path
from yq_ml_alpha.output.factor_metadata import write_factor_metadata
from yq_ml_alpha.pipelines.factor import _load_factor_config, _new_writer
from yq_ml_alpha.pipelines.runtime import (
    _chunks, _context, _load_bundle, _model_class, _predict_model,
    _prediction_frame, _release_accelerator_memory, build_windows,
)


def predict_many(config_paths):
    if not config_paths:
        raise ValueError("grouped prediction needs at least one factor config")
    configs = [_load_factor_config(path) for path in config_paths]
    if len({config.output.id for config in configs}) != len(configs):
        raise ValueError("grouped prediction requires distinct factor ids")
    first = configs[0]
    for config in configs:
        if config.features.type == "analyst_report_events":
            raise ValueError("report-event factors require their event replay prediction workflow")
        if (config.output.root, config.output.asset, config.output.frequency, config.output.base_root,
                config.data_root, config.dates.predict) != (
                first.output.root, first.output.asset, first.output.frequency, first.output.base_root,
                first.data_root, first.dates.predict):
            raise ValueError("grouped factors must share output layout, calendar and prediction range")
    calendar = TradingCalendar.load(first.data_root)
    schedules = [build_windows(config, calendar) for config in configs]
    expected = [window.predict_dates for window in schedules[0]]
    if any([window.predict_dates for window in schedule] != expected for schedule in schedules[1:]):
        raise ValueError("grouped factors must have matching prediction-window boundaries")
    writer = _new_writer(first)
    written = []
    for config in configs:
        path = write_factor_metadata(config)
        if path is not None:
            written.append(path)
    all_dates = [date for window in schedules[0] for date in window.predict_dates]
    for index, first_window in enumerate(schedules[0]):
        predictions = {}
        # Only one model occupies the GPU. Hold only scalar scores for this
        # window on CPU, not models' tensors or encoded feature matrices.
        for config, schedule in zip(configs, schedules):
            window = schedule[index]
            dataset = DatasetBuilder(config)
            model = None
            parts = []
            try:
                path = window_artifact_path(config.model.artifact_dir, window.window_id)
                validate_manifest(path, config, window, dataset)
                model = _model_class(config.model.class_path).load(path)
                with model.prediction_session():
                    for dates in _chunks(window.predict_dates, config.materialize.predict_batch_size):
                        bundle = _load_bundle(config, dataset, calendar, dates, include_label=False)
                        try:
                            if not bundle.frame.empty:
                                context = _context(config, window.window_id, bundle.feature_columns)
                                parts.append(_prediction_frame(bundle.frame, _predict_model(model, bundle, context)))
                        finally:
                            bundle.close()
                predictions[config.output.id] = (
                    pd.concat(parts, ignore_index=True) if parts
                    else pd.DataFrame(columns=["trade_date", "ts_code", "score"])
                )
                predictions[config.output.id]["ts_code"] = predictions[config.output.id].ts_code.astype("category")
                del parts
                print(f"grouped-predict window={window.window_id} factor={config.output.id} "
                      f"rows={len(predictions[config.output.id])}", flush=True)
            finally:
                dataset.close()
                del model, dataset
                gc.collect()
                _release_accelerator_memory()
        for dates in _chunks(first_window.predict_dates, first.materialize.predict_batch_size):
            batch = {name: frame.loc[frame.trade_date.isin(dates)] for name, frame in predictions.items()}
            written.extend(writer.write_many(batch, coverage_dates={name: dates for name in batch},
                                              schema_dates=all_dates))
        del predictions
    return written
