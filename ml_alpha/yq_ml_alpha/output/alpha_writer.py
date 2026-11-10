from __future__ import annotations

from pathlib import Path

import pandas as pd

from yq_ml_alpha.data.stores import KEY_COLUMNS
from yq_ml_alpha.output.daily_wide_writer import DailyWideWriter


class AlphaWriter:
    def __init__(
        self,
        output_root: str | Path,
        alpha_id: str,
        *,
        base_root: str | Path | None = None,
        write_workers: int = 4,
        key_columns: list[str] = KEY_COLUMNS,
    ) -> None:
        self.alpha_id = alpha_id
        self.writer = DailyWideWriter(
            output_root,
            alpha_id,
            layout="direct",
            base_root=base_root,
            write_workers=write_workers,
            key_columns=key_columns,
        )

    def write(
        self,
        predictions: pd.DataFrame,
        *,
        coverage_dates: list[int] | None = None,
        schema_dates: list[int] | None = None,
    ) -> list[Path]:
        return self.writer.write(predictions, coverage_dates=coverage_dates, schema_dates=schema_dates)

    def _path(self, trade_date: int) -> Path:
        return self.writer._path(trade_date)
