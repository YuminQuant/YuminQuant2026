"""Cross-platform computation entrypoint; deployment adapters remain external."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ML_COMMANDS = {f'{kind}-{action}' for kind in ('model','factor') for action in ('train','predict','run','materialize')}
ML_COMMANDS |= {'factor-metadata','factor-metadata-all'}
PRODUCE = {'run','label-run','barra-run','backtest','strategy-run'}

def option(arguments, name, default=None):
    for index, value in enumerate(arguments):
        if value == name:
            if index+1 >= len(arguments): raise ValueError(f'Missing value for {name}')
            return arguments[index+1]
        if value.startswith(name+'='): return value.split('=',1)[1]
    return default

def set_option(arguments, name, value):
    result = list(arguments)
    for index, item in enumerate(result):
        if item == name:
            result[index+1] = str(value)
            return result
        if item.startswith(name+'='):
            result[index] = name+'='+str(value)
            return result
    return result+[name,str(value)]

def split_tasks(command, arguments, dates, single_task=False):
    if single_task or command not in {'run','label-run','barra-run'}:
        return [{'id':'task-0000','command':command,'arguments':list(arguments)}]
    size = int(option(arguments,'--date-batch-size','120'))
    if size <= 0: raise ValueError('--date-batch-size must be positive')
    tasks=[]
    for index in range(0,len(dates),size):
        target=dates[index:index+size]
        args=set_option(set_option(arguments,'--start-date',target[0]),'--end-date',target[-1])
        tasks.append({'id':f'task-{len(tasks):04d}','command':command,'arguments':args,'target_dates':target})
    return tasks


def split_training_tasks(command, arguments, plan):
    if command not in {'factor-train','model-train'}:
        raise ValueError('Training window splitting requires factor-train or model-train')
    if '--aggregate-diagnostics-only' in arguments or '--list-windows' in arguments:
        raise ValueError('Use --dry-run to inspect a split training plan')
    base = []
    index = 0
    while index < len(arguments):
        value = arguments[index]
        if value == '--window-id':
            index += 2
            continue
        if not value.startswith('--window-id='):
            base.append(value)
        index += 1
    windows = plan['windows']
    ids = [window['window_id'] for window in windows]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Training plan must contain unique nonempty windows')
    return [dict(id=f'train-{index:04d}', command=command,
                 arguments=base+['--window-id',window['window_id']], window=window)
            for index,window in enumerate(windows)]


def training_aggregate_task(command, arguments, plan):
    if not plan.get('aggregate_diagnostics'):
        return None
    config = option(arguments,'--config')
    return dict(id='training-diagnostics',command=command,
                arguments=['--config',config,'--aggregate-diagnostics-only'])

def metadata_commands(command):
    if command in {'run','plan','list','backtest','strategy-run'}: return ['metadata']
    if command.startswith('label-') and command != 'label-metadata': return ['label-metadata']
    if command.startswith('barra-') and command != 'barra-metadata': return ['barra-metadata']
    return []

def project_arguments(command, arguments):
    if command in ML_COMMANDS: return []
    path=option(arguments,'--project-config') or option(arguments,'--config')
    return ['--config',path] if path else []

def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    plugin=os.environ.get('YQ_DEPLOYMENT_MODULE')
    if not plugin:
        project=Path(option(argv,'--project-config',str(ROOT/'config.toml')))
        if project.is_file():
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib
            execution=tomllib.loads(project.read_text(encoding='utf-8')).get('execution',{})
            plugin=execution.get('deployment_module')
            if plugin:
                sys.path.insert(0,str(execution['module_root']))
                os.environ.setdefault('YQ_DEPLOYMENT_CONFIG',str(execution['deployment_config']))
    if plugin:
        return importlib.import_module(plugin).main(argv)
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--single-task',action='store_true')
    parser.add_argument('--split-training-windows',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--binary',type=Path)
    parser.add_argument('--no-build',action='store_true')
    options,rest=parser.parse_known_args(argv)
    if not rest:
        parser.error('Provide a command, e.g. run --asset stock --frequency daily --start-date ... --end-date ... --factors ...')
    command,arguments=rest[0],rest[1:]
    if options.split_training_windows and (options.single_task or command not in {'model-train','factor-train'}):
        parser.error('--split-training-windows requires a training command without --single-task')
    if command in ML_COMMANDS:
        env=dict(os.environ)
        env['PYTHONPATH']=os.pathsep.join([str(ROOT/'ml_alpha'),env.get('PYTHONPATH','')])
        config_count=sum(value == '--config' or value.startswith('--config=') for value in arguments)
        multi_mode=(command in {'model-train','factor-train','model-predict','factor-predict'} and
                    (config_count > 1 or option(arguments,'--execution-config') is not None
                     or '--list-tasks' in arguments or option(arguments,'--task-id') is not None))
        execution=option(arguments,'--execution-config')
        if execution:
            import tomllib
            operational=tomllib.loads(Path(execution).read_text(encoding='utf-8')).get('execution',{})
            threads=str(max(1,int(operational.get('threads',1))//int(operational.get('model_predict_workers',1))))
            env.update({name:threads for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
        tasks = [dict(command=command,arguments=arguments)]
        aggregate = None
        if options.split_training_windows and not multi_mode:
            plan=json.loads(subprocess.check_output([sys.executable,'-B','-m','yq_ml_alpha',command,
                            *arguments,'--list-windows'],cwd=ROOT,env=env,text=True))
            tasks=split_training_tasks(command,arguments,plan)
            aggregate=training_aggregate_task(command,arguments,plan)
        if options.dry_run:
            if multi_mode:
                return subprocess.call([sys.executable,'-B','-m','yq_ml_alpha',command,*arguments,'--list-tasks'],cwd=ROOT,env=env)
            if options.split_training_windows:
                print(json.dumps({'executor':'local','tasks':tasks,'after_success':aggregate},indent=2))
                return 0
            print(json.dumps({'command':command,'arguments':arguments,'executor':'local'}))
            return 0
        if command.startswith('factor-') and command not in {'factor-metadata','factor-metadata-all'}:
            configs=[]
            for index,value in enumerate(arguments):
                if value == '--config': configs+=['--config',arguments[index+1]]
                elif value.startswith('--config='): configs+=['--config',value.split('=',1)[1]]
            subprocess.run([sys.executable,'-B','-m','yq_ml_alpha','factor-metadata',*configs],cwd=ROOT,env=env,check=True)
            env['YQ_METADATA_MANAGED']='1'
        for task in tasks + ([aggregate] if aggregate else []):
            result=subprocess.run([sys.executable,'-B','-m','yq_ml_alpha',task['command'],*task['arguments']],cwd=ROOT,env=env)
            if result.returncode:
                return result.returncode
        return 0
    if command in {'run','label-run','barra-run'} and option(arguments,'--date-batch-size') is None:
        arguments+=['--date-batch-size','120']
    binary=options.binary or Path(os.environ.get('CARGO_TARGET_DIR',ROOT/'factor_engine/target'))/'release'/('yq-factor-engine.exe' if os.name=='nt' else 'yq-factor-engine')
    if not options.no_build:
        subprocess.run(['cargo','build','--release','--manifest-path',str(ROOT/'factor_engine/Cargo.toml')],cwd=ROOT,check=True)
    dates=[]
    if not options.single_task and command in {'run','label-run','barra-run'}:
        exchange='SSE' if option(arguments,'--asset','stock')=='stock' else 'SHFE'
        dates=json.loads(subprocess.check_output([str(binary),'calendar-dates',
            '--exchange',exchange,'--start-date',option(arguments,'--start-date'),
            '--end-date',option(arguments,'--end-date'),*project_arguments(command,arguments)],cwd=ROOT,text=True))
    tasks=split_tasks(command,arguments,dates,options.single_task)
    if options.dry_run:
        print(json.dumps({'metadata':metadata_commands(command),'tasks':tasks},ensure_ascii=False,indent=2))
        return 0
    for refresh in metadata_commands(command):
        subprocess.run([str(binary),refresh,*project_arguments(command,arguments)],cwd=ROOT,check=True)
    if command == 'run' and len(tasks) > 1:
        subprocess.run([str(binary),'raw-run',*arguments],cwd=ROOT,check=True)
        for task in tasks:
            task['arguments']=set_option(task['arguments'],'--refresh-minute-cache','false')
    for task in tasks:
        subprocess.run([str(binary),task['command'],*task['arguments']],cwd=ROOT,check=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
