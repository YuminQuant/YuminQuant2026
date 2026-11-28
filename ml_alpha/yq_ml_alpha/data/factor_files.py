"""Logical daily factors over independent manual and ML Parquet partitions."""
import json
from pathlib import Path

from yq_ml_alpha.output import artifact_io as io

KEYS = ["trade_date", "ts_code"]
PARTITIONS = {"manual", "ml"}


def factor_file_date(path):
    path = Path(path)
    stem = path.stem[:-3] if path.stem.endswith("_ml") else path.stem
    return int(stem) if path.suffix == ".parquet" and len(stem) == 8 and stem.isdigit() else None


def factor_path(root, date, partition="manual"):
    if partition not in PARTITIONS:
        raise ValueError(f"invalid factor storage partition: {partition}")
    suffix = "_ml" if partition == "ml" else ""
    return Path(root) / str(date // 10000) / f"{date}{suffix}.parquet"


def factor_dates(root):
    paths = list(io.glob(Path(root), "*/*.parquet")) + list(io.glob(Path(root), "*.parquet"))
    return sorted({date for path in paths
                   if (date := factor_file_date(path)) is not None})


def metadata_path(root):
    root = Path(root)
    for parent in (root, *list(root.parents)[:2]):
        path = parent / "factor_metadata.parquet"
        if io.exists(path):
            return path
    return None


def normalize_metadata(frame):
    """Old metadata is readable; every newly published row has explicit ownership."""
    import pandas as pd

    frame = frame.copy()
    inferred = frame.tags_json.map(
        lambda value: "ml" if "model_generated" in json.loads("[]" if pd.isna(value) else value or "[]") else "manual")
    if "storage_partition" not in frame:
        frame["storage_partition"] = inferred
    else:
        current = frame.storage_partition
        frame["storage_partition"] = current.where(current.notna() & current.ne(""), inferred)
    if not frame.storage_partition.isin(PARTITIONS).all():
        raise ValueError("factor metadata storage_partition must be manual or ml")
    if not frame.storage_partition.eq(inferred).all():
        raise ValueError("factor ownership conflicts with model_generated metadata")
    scope = [name for name in ("asset_class", "frequency") if name in frame]
    identity = [*scope, "output_column"]
    owners = frame.groupby(identity, dropna=False).storage_partition.nunique()
    if (owners > 1).any() or frame.duplicated(identity).any():
        raise ValueError("factor metadata output columns must have one owner")
    return frame


class FactorFiles:
    """Project only requested columns; ML presence wins over legacy mixed files.

    A present ML column owns its whole cross-section, including explicit nulls.
    Values are never filled from old scores for stocks absent in the ML file.
    """
    def __init__(self, root):
        self.root = Path(root)
        path = metadata_path(root)
        metadata = normalize_metadata(io.read_parquet(path)) if path is not None else None
        if metadata is not None:
            asset, frequency = self.root.parent.name, self.root.name
            if ("asset_class" in metadata and "frequency" in metadata
                    and metadata.asset_class.eq(asset).any() and metadata.frequency.eq(frequency).any()):
                metadata = metadata.loc[metadata.asset_class.eq(asset) & metadata.frequency.eq(frequency)]
            if metadata.groupby("output_column").storage_partition.nunique().gt(1).any():
                raise ValueError("factor metadata ownership is ambiguous; use an asset/frequency factor root")
        self.owners = {} if metadata is None else dict(zip(metadata.output_column, metadata.storage_partition))

    def exists(self, date):
        return bool(self._paths(date))

    def _paths(self, date):
        # Direct daily roots remain supported alongside the standard year layout.
        paths = {}
        for part in ("manual", "ml"):
            path = factor_path(self.root, date, part)
            if io.exists(path):
                paths[part] = path
            else:
                flat = self.root / path.name
                if io.exists(flat):
                    paths[part] = flat
        return paths

    def read(self, date, columns=None, *, require_file=False):
        requested = None if columns is None else list(dict.fromkeys(c for c in columns if c not in KEYS))
        paths = self._paths(date)
        if not paths:
            if require_file:
                raise FileNotFoundError(f"missing factor history for {date} under {self.root}")
            import pandas as pd

            return pd.DataFrame(columns=[*KEYS, *(requested or [])])
        if requested:
            return self._read_requested(date, paths, requested)
        schemas = {}
        if "ml" in paths:
            schemas["ml"] = set(io.parquet_columns(paths["ml"]))
        if "manual" in paths:
            schemas["manual"] = set(io.parquet_columns(paths["manual"]))
        names = requested if requested is not None else sorted(set.union(set(), *schemas.values()) - set(KEYS))
        groups = {part: [] for part in schemas}
        for name in names:
            owner = self.owners.get(name)
            if owner != "manual" and name in schemas.get("ml", set()):
                groups["ml"].append(name)
            elif name in schemas.get("manual", set()):
                groups["manual"].append(name)
        frames = []
        for part, selected in groups.items():
            if not selected:
                continue
            frame = io.read_parquet(paths[part], columns=[*KEYS, *selected])
            _validate_keys(frame, date, paths[part])
            frames.append(frame)
        if not frames:
            path = paths.get("manual", paths.get("ml"))
            frames = [io.read_parquet(path, columns=KEYS)]
            _validate_keys(frames[0], date, path)
        return _merge_frames(frames)

    def _read_requested(self, date, paths, requested):
        # Project existing columns while opening each partition once. The ML
        # projection determines which legacy mixed-file columns can fall back.
        frames = {}
        ml_columns = [name for name in requested if self.owners.get(name) != "manual"]
        ml_names = set()
        if ml_columns and "ml" in paths:
            frames["ml"] = io.read_projected_frame(paths["ml"], [*KEYS, *ml_columns],
                                                   required=KEYS, pre_buffer=True)
            ml_names = set(frames["ml"].columns)
        manual_columns = [name for name in requested if name not in ml_names]
        if manual_columns and "manual" in paths:
            frames["manual"] = io.read_projected_frame(paths["manual"], [*KEYS, *manual_columns],
                                                       required=KEYS, pre_buffer=True)
        selected = [part for part, frame in frames.items() if len(frame.columns) > len(KEYS)]
        if not selected:
            part = "manual" if "manual" in paths else "ml"
            if part not in frames:
                frames[part] = io.read_parquet(paths[part], columns=KEYS)
            selected = [part]
        for part in selected:
            _validate_keys(frames[part], date, paths[part])
        return _merge_frames([frames[part] for part in selected])


def _merge_frames(frames):
    result = frames[0]
    if len(frames) == 1:
        return result
    for frame in frames[1:]:
        result = result.merge(frame, on=KEYS, how="outer", validate="one_to_one", sort=False)
    # Keep absent columns absent: callers can distinguish missing output from null output.
    return result.sort_values(KEYS).reset_index(drop=True)


def _validate_keys(frame, date, path):
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any() or not frame.trade_date.eq(date).all():
        raise ValueError(f"invalid daily factor keys in {path}; expected unique ({date}, ts_code)")
