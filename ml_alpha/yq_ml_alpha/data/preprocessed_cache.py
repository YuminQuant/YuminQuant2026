"""Run-scoped cache for repeated labeled dates in rolling windows."""
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


class PreprocessedDateCache:
    def __init__(self, date_groups):
        counts = Counter(date for dates in date_groups for date in dates)
        self.remaining = {date: count for date, count in counts.items() if count > 1}
        self.directory = None

    def take(self, date):
        if self.directory is None or date not in self.remaining:
            return None
        path = Path(self.directory.name) / f"{date}.pkl"
        if not path.exists():
            return None
        frame = pd.read_pickle(path)
        self.remaining[date] -= 1
        if self.remaining[date] == 0:
            path.unlink()
            del self.remaining[date]
            if not self.remaining:
                self.close()
        return frame

    def put(self, date, frame):
        if date not in self.remaining:
            return
        if self.directory is None:
            self.directory = TemporaryDirectory(prefix="yq-preprocessed-")
        frame.to_pickle(Path(self.directory.name) / f"{date}.pkl")
        self.remaining[date] -= 1

    def close(self):
        if self.directory is not None:
            self.directory.cleanup()
            self.directory = None
