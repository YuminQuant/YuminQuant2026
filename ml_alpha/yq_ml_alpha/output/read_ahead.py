"""Remote-only ordered IO; provider state and cache updates stay on the caller."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from threading import Lock, local

SUPPORTS_FIXED_IO_POLICY = True

_lock = Lock()
_pools = {}
_worker = local()


def options(root):
    from . import artifact_io as io
    store = io.backend(root)
    if store is None:
        return {"workers": 1}
    configure = getattr(store, "io_options", None)
    values = configure() if configure else {}
    workers = int(values.get("workers", 50))
    if workers < 1:
        raise ValueError("remote IO workers must be positive")
    return {"workers": workers, "fixed": values.get("fixed", False)}


def ordered_map(function, items, *, root, workers=None):
    settings = options(root)
    fixed = settings.get("fixed", False)
    width = settings["workers"]
    if not fixed and workers is not None:
        if int(workers) < 1: raise ValueError("IO workers must be positive")
        width = min(width, int(workers))
    if width == 1 or getattr(_worker, "active", False):
        for item in items:
            yield function(item)
        return
    with _lock:
        pool = _pools.get(settings["workers"])
        if pool is None:
            pool = ThreadPoolExecutor(max_workers=settings["workers"], thread_name_prefix="remote-io")
            _pools[settings["workers"]] = pool
    def run(item):
        _worker.active = True
        try:
            return function(item)
        finally:
            _worker.active = False
    iterator = iter(items)
    pending = deque()
    exhausted = False
    try:
        while pending or not exhausted:
            while not exhausted and len(pending) < width:
                try:
                    item = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                context = copy_context()
                pending.append(pool.submit(context.run, run, item))
            if pending:
                yield pending.popleft().result()
    finally:
        for future in pending: future.cancel()
        for future in pending:
            try: future.result()
            except BaseException: pass
        pending.clear()
