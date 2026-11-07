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
        return {"workers": 1, "prefetch_bytes": 0}
    configure = getattr(store, "io_options", None)
    values = configure() if configure else {}
    workers = int(values.get("workers", 50))
    budget = values.get("prefetch_bytes", 128 * 1024**2)
    if budget is not None:
        budget = int(budget)
    if workers < 1 or (budget is not None and budget < 1):
        raise ValueError("remote IO workers and prefetch budget must be positive")
    return {"workers": workers, "prefetch_bytes": budget,
            "fixed": values.get("fixed", False)}


def ordered_map(function, items, *, root, path_for=None, workers=None,
                prefetch_bytes=None, estimate_bytes=None):
    from . import artifact_io as io
    settings = options(root)
    fixed = settings.get("fixed", False)
    if not fixed and prefetch_bytes is not None:
        if type(prefetch_bytes) is not int or prefetch_bytes < 1:
            raise ValueError("prefetch_bytes must be a positive integer")
        settings["prefetch_bytes"] = prefetch_bytes
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
    def reserve(item):
        if settings["prefetch_bytes"] is None:
            return 0
        if estimate_bytes is not None:
            size = estimate_bytes(item)
            if size is not None:
                if type(size) is not int or size < 1:
                    raise ValueError("estimated item bytes must be a positive integer")
                return size
        if path_for is None: return 8 * 1024**2
        try:
            # Conservative compressed-to-pandas estimate, not a hard RSS bound.
            return max(1024**2, int(io.stat(path_for(item)).st_size) * 8)
        except FileNotFoundError:
            return 1024**2
    iterator = iter(items)
    pending = deque()
    held = 0
    waiting = None
    exhausted = False
    try:
        while pending or not exhausted or waiting is not None:
            while len(pending) < width:
                if waiting is None:
                    try: item = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    waiting = (item, reserve(item))
                item, size = waiting
                budget = settings["prefetch_bytes"]
                if budget is not None and pending and held + size > budget: break
                context = copy_context()
                pending.append((pool.submit(context.run, run, item), size))
                held += size
                waiting = None
            if pending:
                future, size = pending.popleft()
                result = future.result()
                try:
                    yield result
                finally:
                    held -= size
                    del result, future
    finally:
        for future, _ in pending: future.cancel()
        for future, _ in pending:
            try: future.result()
            except BaseException: pass
        pending.clear()
