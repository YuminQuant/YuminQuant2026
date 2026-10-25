"""Operational budgets; deliberately excluded from model artifact identities."""
from contextlib import contextmanager
from dataclasses import dataclass, fields
import os
import sys
from pathlib import Path


@dataclass(frozen=True)
class ExecutionOptions:
    train_max_parallel: int = 1
    predict_max_parallel: int = 1
    model_predict_workers: int = 1
    threads: int = 1
    predict_task_days: int = 120
    write_workers: int = 1
    input_cache_mib: int = 64
    metadata_cache_mib: int = 8
    score_buffer_mib: int = 128
    model_cache_mib: int = 512
    max_resident_models: int = 2
    memory_budget_mib: int = 2048

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            minimum = 0 if field.name in {'input_cache_mib', 'metadata_cache_mib'} else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f'{field.name} must be an integer >= {minimum}')
        if self.input_cache_mib + self.metadata_cache_mib + self.score_buffer_mib + self.model_cache_mib >= self.memory_budget_mib:
            raise ValueError('Cache and score budgets must leave room for inputs, models and the interpreter')

    def environment(self):
        inner = str(max(1, self.threads // self.model_predict_workers))
        return {'YQ_COMPUTE_THREADS': str(self.threads), 'OMP_NUM_THREADS': inner,
                'OPENBLAS_NUM_THREADS': inner, 'MKL_NUM_THREADS': inner,
                'NUMEXPR_NUM_THREADS': inner, 'NUMBA_NUM_THREADS': inner}


def load_execution(path=None):
    if path is None:
        return ExecutionOptions()
    from yq_ml_alpha.config import _tomllib
    raw = _tomllib.loads(Path(path).read_text(encoding='utf-8'))
    if set(raw) != {'execution'}:
        raise ValueError('Execution TOML must contain only [execution]; model configs remain separate')
    return ExecutionOptions(**raw['execution'])


def resident_bytes():
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        value = Counters(); value.cb = ctypes.sizeof(value)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb):
            raise OSError(ctypes.get_last_error(), 'GetProcessMemoryInfo')
        return value.WorkingSetSize
    with open('/proc/self/statm') as stream:
        return int(stream.read().split()[1]) * os.sysconf('SC_PAGE_SIZE')


def check_memory(options, stage):
    current = resident_bytes()
    if current > options.memory_budget_mib * 1024**2:
        raise MemoryError(f'{stage}: resident RAM {current / 1024**2:.1f} MiB exceeds '
                          f'{options.memory_budget_mib} MiB; reduce parallelism/batches or increase the budget')
    return current


@contextmanager
def execution_scope(options):
    from yq_ml_alpha.output.parquet_reader import cache
    reader = cache()
    environment = options.environment()
    previous_env = {name: os.environ.get(name) for name in environment}
    os.environ.update(environment)
    torch = sys.modules.get('torch')
    torch_threads = torch.get_num_threads() if torch is not None else None
    if torch is not None:
        torch.set_num_threads(max(1, options.threads // options.model_predict_workers))
    old = reader.table_limit, reader.metadata_limit
    reader.invalidate()
    reader.table_limit = options.input_cache_mib * 1024**2
    reader.metadata_limit = options.metadata_cache_mib * 1024**2
    try:
        try:
            from threadpoolctl import threadpool_limits
            native = threadpool_limits(limits=max(1, options.threads // options.model_predict_workers))
        except ImportError:
            native = None
        import pyarrow as pa
        previous = pa.cpu_count()
        pa.set_cpu_count(options.threads)
        yield
    finally:
        if 'native' in locals() and native is not None:
            native.restore_original_limits()
        reader.invalidate()
        reader.table_limit, reader.metadata_limit = old
        if 'previous' in locals():
            pa.set_cpu_count(previous)
        if torch is not None:
            torch.set_num_threads(torch_threads)
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
