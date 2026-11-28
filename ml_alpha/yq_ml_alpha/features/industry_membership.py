"""Point-in-time industry membership shared by models and neutralization."""
from bisect import bisect_right

import numpy as np
import pandas as pd


class IndustryMembership:
    """Same inclusive interval and latest-effective-row convention as the Rust reader."""

    def __init__(self, frame, codes):
        self.groups = {}
        positions = {code: i for i, code in enumerate(codes)}
        columns = ["ts_code", "l1_code", "in_date", "out_date"]
        for symbol, code, start, stop in frame[columns].itertuples(index=False, name=None):
            if pd.isna(symbol) or pd.isna(code) or pd.isna(start) or not str(code).strip():
                continue
            if code not in positions:
                raise ValueError(f"unknown CITIC level-one code {code}; check the configured industry schema")
            start = int(start)
            stop = 99991231 if pd.isna(stop) or not str(stop).strip() else int(stop)
            if stop < start:
                raise ValueError(f"invalid CITIC industry interval: {symbol}/{start}/{stop}")
            self.groups.setdefault(symbol, []).append((start, stop, positions[code]))
        for stock, rows in self.groups.items():
            rows.sort(key=lambda row: row[0])
            self.groups[stock] = tuple(zip(*rows))

    def indices(self, symbols, date):
        result = np.full(len(symbols), -1, dtype=np.intp)
        for i, stock in enumerate(symbols):
            history = self.groups.get(stock)
            if history is None:
                continue
            starts, ends, codes = history
            for j in range(bisect_right(starts, date) - 1, -1, -1):
                if date <= ends[j]:
                    result[i] = codes[j]
                    break
        return result
