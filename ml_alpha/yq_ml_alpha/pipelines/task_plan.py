"""Plan task ownership without editing scientific configs or refit windows."""
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from yq_ml_alpha.calendar import TradingCalendar
from yq_ml_alpha.pipelines.runtime import build_windows, _predict_dates, _chunks


def load_configs(paths, kind):
    from yq_ml_alpha.pipelines import factor, model
    loader = factor._load_factor_config if kind == 'factor' else model._load_model_config
    return [loader(path) for path in paths]


def output_group(config):
    return (str(config.output.root), config.output.kind, config.output.asset,
            config.output.frequency, str(config.output.base_root), str(config.data_root))


def output_path(config, date):
    root = Path(config.output.root)
    if config.output.kind == 'factor':
        root = root / config.output.asset / config.output.frequency
    suffix = '_ml' if config.output.kind == 'factor' else ''
    return root / str(date // 10000) / f'{date}{suffix}.parquet'


def prediction_units(config, calendar):
    return [(window, dates) for window in build_windows(config, calendar)
            for dates in _chunks(window.predict_dates, config.materialize.predict_batch_size)]


def plan_tasks(configs, action, options):
    if not configs:
        raise ValueError('At least one model config is required')
    artifacts = set()
    outputs = set()
    layouts = {}
    tasks = []
    groups = defaultdict(list)
    for index, config in enumerate(configs):
        key = output_group(config)[:4] + (config.output.id,)
        if key in outputs:
            raise ValueError(f'Duplicate output ID in one destination: {config.output.id}')
        outputs.add(key)
        if action == 'train':
            from yq_ml_alpha.pipelines.training_tasks import training_plan
            plan = training_plan(config)
            from yq_ml_alpha.pipelines.runtime import _model_class
            sequential = getattr(_model_class(config.model.class_path), 'training_requires_previous_window', False)
            for wi, window in enumerate(plan['windows']):
                path = str(Path(plan['artifact_dir']) / window['window_id'])
                if path in artifacts:
                    raise ValueError(f'Training artifact collision: {path}')
                artifacts.add(path)
                if not sequential:
                    tasks.append(dict(id=f'train-{index:03d}-{wi:04d}', config_indices=[index],
                        window_id=window['window_id'], artifact_dir=path,
                        cache_dir=str(Path(plan['cache_dir']) / 'training_windows' / window['window_id']),
                        window=window))
            if sequential:
                tasks.append(dict(id=f'train-{index:03d}-sequential', config_indices=[index],
                    window_id=None, artifact_dir=plan['artifact_dir'],cache_dir=plan['cache_dir'],
                    windows=plan['windows']))
        elif action == 'predict':
            destination = output_group(config)[:4]
            inputs = output_group(config)[4:]
            if destination in layouts and layouts[destination] != inputs:
                raise ValueError('Models writing the same daily files must share the calendar and base universe')
            layouts[destination] = inputs
            if config.features.type == 'analyst_report_events':
                raise ValueError('Report-event replay must use its existing single-model workflow; '
                                 'it cannot be partitioned by the generic prediction planner')
            groups[output_group(config)].append(index)
        else:
            raise ValueError(f'Unsupported task action: {action}')
    for gi, indices in enumerate(groups.values()):
        calendar = TradingCalendar.load(configs[indices[0]].data_root)
        dates = sorted({d for i in indices for d in _predict_dates(configs[i], calendar)})
        # Arbitrary model extensions may carry state across prediction batches.
        # Only declared replay-safe models can be independently date-sharded.
        from yq_ml_alpha.pipelines.runtime import _model_class
        safe = all(getattr(_model_class(configs[i].model.class_path), 'prediction_replay_safe', False) for i in indices)
        chunks = _chunks(dates, options.predict_task_days) if safe else [dates]
        for ti, target in enumerate(chunks):
            if target:
                tasks.append(dict(id=f'predict-{gi:03d}-{ti:04d}', config_indices=indices,
                    target_dates=target, output_paths=[str(output_path(configs[indices[0]], d)) for d in target],
                    replay_safe=safe))
    if not tasks:
        raise ValueError('No model tasks match the configured calendar and prediction dates')
    diagnostics = [str(Path(config.model.artifact_dir).parent / 'diagnostics') for config in configs
                   if config.diagnostics.enabled and
                   (config.diagnostics.write_loss_history or config.diagnostics.write_window_summary)]
    return dict(action=action, execution=asdict(options), tasks=tasks, diagnostics=diagnostics)
