"""Shared local/cluster task executor. Cluster scheduling lives outside the repo."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import os
import subprocess
import sys

from yq_ml_alpha.pipelines.execution import check_memory, execution_scope
from yq_ml_alpha.pipelines.task_plan import plan_tasks


def execute_task(configs, action, task, options, *, resume=False):
    selected = [configs[index] for index in task['config_indices']]
    if action == 'predict':
        from yq_ml_alpha.pipelines.grouped_predict import predict_many_configs
        return predict_many_configs(selected, options=options, target_dates=task['target_dates'])
    config = selected[0]
    from yq_ml_alpha.pipelines import factor, model
    with execution_scope(options):
        check_memory(options, 'before training')
        function = factor.train_config if config.output.kind == 'factor' else model.train_config
        extra = {'resume': resume} if config.output.kind == 'factor' else {}
        result = function(config, window_ids=[task['window_id']] if task['window_id'] else None, **extra)
        check_memory(options, 'after training')
        return result


def execute_plan(configs, action, options, *, task_id=None, config_paths=None,
                 execution_path=None, command=None, resume=False):
    plan = plan_tasks(configs, action, options)
    tasks = plan['tasks']
    if task_id is not None:
        selected = [task for task in tasks if task['id'] == task_id]
        if len(selected) != 1:
            raise ValueError(f'Unknown task ID: {task_id}')
        return execute_task(configs, action, selected[0], options, resume=resume)
    parallel = options.train_max_parallel if action == 'train' else options.predict_max_parallel
    print('ml-execution ' + json.dumps(dict(action=action, tasks=len(tasks),
        max_parallel=parallel, per_task=asdict(options)), sort_keys=True), flush=True)
    if action == 'predict':
        from yq_ml_alpha.output.factor_metadata import write_factor_metadata_many
        if os.environ.get('YQ_METADATA_MANAGED') != '1':
            write_factor_metadata_many(configs)
    env = dict(os.environ, **options.environment(), YQ_METADATA_MANAGED='1')
    if parallel == 1:
        old = os.environ.get('YQ_METADATA_MANAGED')
        os.environ['YQ_METADATA_MANAGED'] = '1'
        try:
            result = []
            for task in tasks:
                result.extend(execute_task(configs, action, task, options, resume=resume))
        finally:
            if old is None:
                os.environ.pop('YQ_METADATA_MANAGED', None)
            else:
                os.environ['YQ_METADATA_MANAGED'] = old
    else:
        if not config_paths or not execution_path or not command:
            raise ValueError('Parallel local tasks require config paths and an execution TOML')
        if os.environ.get('YQ_STORAGE_BACKEND') == 'hdfs':
            raise ValueError('Remote parallel tasks must be submitted by the deployment scheduler')
        def launch(task):
            argv = [sys.executable, '-B', '-m', 'yq_ml_alpha', command,
                    '--execution-config', str(execution_path), '--task-id', task['id']]
            for path in config_paths:
                argv += ['--config', str(path)]
            if resume:
                argv.append('--resume')
            subprocess.run(argv, env=env, check=True)
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            list(pool.map(launch, tasks))
        result = []
    if action == 'train':
        from yq_ml_alpha.pipelines.runtime import _aggregate_diagnostics
        for config in configs:
            result.extend(_aggregate_diagnostics(config))
    return result
