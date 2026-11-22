"""Normalize observed Tushare minimum opening quantities without inventing history."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data_manager.core.config_manager import ConfigManager


SCHEMA = pa.schema([
    ("trade_date", pa.int32()), ("ts_code", pa.string()),
    ("min_open_lots", pa.int32()), ("min_close_lots", pa.int32()),
    ("source_ts_code", pa.string()), ("source", pa.string()),
])


class FutureOrderLimitsBuilder:
    def __init__(self):
        paths = ConfigManager().config["paths"]
        root = Path(paths["base_data_dir"])
        self.input = root / paths.get("fut_trade_param_dir", "future_data/basic/trade_param")
        self.output = root / paths.get("fut_order_limits_dir", "future_data/basic/order_limits")

    def sync(self, start_date=None, target_end_date=None):
        end = target_end_date or datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d")
        start = start_date or end
        if start > end:
            raise ValueError("order-limit start date is after end date")
        for year in range(int(start[:4]), int(end[:4]) + 1):
            frames = []
            for path in sorted((self.input / "CZCE" / str(year)).glob("*.parquet")):
                raw = pd.read_parquet(path, columns=["trade_date", "ts_code", "min_order_qty"])
                if raw.empty:
                    continue
                qty = pd.to_numeric(raw.min_order_qty, errors="raise")
                if not (qty.ge(1) & qty.mod(1).eq(0)).all():
                    raise ValueError(f"Invalid minimum opening quantity: {path}")
                frame = pd.DataFrame({
                    "trade_date": pd.to_numeric(raw.trade_date, errors="raise"),
                    "ts_code": raw.ts_code,
                    "min_open_lots": qty, "min_close_lots": 1,
                    "source_ts_code": raw.ts_code,
                    "source": "tushare:fut_trade_param.min_order_qty",
                })
                if not frame.trade_date.eq(int(path.stem)).all():
                    raise ValueError(f"Unexpected trade date: {path}")
                frames.append(frame)
            result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SCHEMA.names)
            if result.duplicated(["trade_date", "ts_code"]).any():
                raise ValueError(f"Duplicate order-limit keys for {year}")
            table = pa.Table.from_pydict({name: result[name].tolist() for name in SCHEMA.names}, schema=SCHEMA)
            self.output.mkdir(parents=True, exist_ok=True)
            path = self.output / f"{year}.parquet"
            temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
            try:
                pq.write_table(table, temporary)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
