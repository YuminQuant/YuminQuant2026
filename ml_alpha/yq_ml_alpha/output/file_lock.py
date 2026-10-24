"""Cross-process read/modify/write lock for a daily output file."""
from yq_ml_alpha.output import artifact_io as data_io
from contextlib import contextmanager
import os
from pathlib import Path
import threading
import time
import weakref
from yq_ml_alpha.output import artifact_io

_locks = weakref.WeakValueDictionary()
_guard = threading.Lock()


@contextmanager
def daily_file_lock(path, timeout=60.0):
    path = Path(path).resolve()
    store = artifact_io.backend(path)
    if store:
        with store.lock(path):
            yield
        return
    data_io.mkdir(path.parent, parents=True, exist_ok=True)
    with _guard:
        lock = _locks.setdefault(str(path), threading.RLock())
    with lock:
        # Keep the lock inode: unlinking it could let a third writer bypass a
        # second writer already holding/opening the original lock file.
        with data_io.open_file(path.with_name(path.name + ".lock"), "a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            started = time.monotonic()
            while True:
                handle.seek(0)
                try:
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() - started >= timeout:
                        raise TimeoutError(f"timed out waiting for daily output lock: {path}")
                    time.sleep(0.05)
            try:
                yield
            finally:
                handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle, fcntl.LOCK_UN)
