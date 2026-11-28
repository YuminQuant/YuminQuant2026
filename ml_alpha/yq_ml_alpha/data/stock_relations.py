"""Sparse stock/group incidence; weighted stock pairs are never stored densely."""
from yq_ml_alpha.output import artifact_io as data_io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from yq_ml_alpha.data.stores import daily_path
from yq_ml_alpha.output.parquet_reader import transient_parquet_tables


RELATIONS = ("industry", "fund", "analyst")
VERSION = "stock_relations_v1"


class StockRelations:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = None
        self._memberships = {}

    def identity(self):
        if self.manifest is None:
            path = self.root / "manifest.json"
            if not data_io.exists(path):
                raise FileNotFoundError(f"Missing stock relations: {path}; run derive-stock-relations")
            manifest = json.loads(data_io.read_text(path, encoding="utf-8"))
            if manifest.get("version") != VERSION or tuple(manifest.get("relations", [])) != RELATIONS:
                raise ValueError("stock relation dataset version/schema mismatch")
            self.manifest = manifest
        return self.manifest

    def load(self, date, symbols, device):
        self.identity()
        keys = tuple(symbols)
        cached = self._memberships.get(date)
        if cached is None or cached[0] != keys:
            path = daily_path(self.root, date)
            try:
                with transient_parquet_tables():
                    frame = data_io.read_parquet(path, columns=["trade_date", "relation", "group_id", "ts_code"])
            except FileNotFoundError as error:
                raise FileNotFoundError(f"Missing stock relations for {date}: {path}") from error
            if (frame.isna().any().any() or not frame.trade_date.eq(date).all()
                    or frame.duplicated(["relation", "group_id", "ts_code"]).any()
                    or not set(frame.relation).issubset(RELATIONS)):
                raise ValueError(f"invalid stock relation memberships: {path}")
            cached = keys, _relation_indices(frame, symbols)
            self._memberships[date] = cached
        return _graphs(cached[1], len(symbols), device)


def build_relations(frame, symbols, device):
    return _graphs(_relation_indices(frame, symbols), len(symbols), device)


def _relation_indices(frame, symbols):
    if len(set(symbols)) != len(symbols):
        raise ValueError("duplicate graph nodes")
    positions = {symbol: i for i, symbol in enumerate(symbols)}
    selected = frame.loc[frame.ts_code.isin(positions)]
    result = []
    for relation in RELATIONS:
        rows = selected.loc[selected.relation.eq(relation)]
        if rows.duplicated(["group_id", "ts_code"]).any():
            raise ValueError("duplicate stock/group membership")
        group_codes, groups = pd.factorize(rows.group_id, sort=True)
        node_codes = rows.ts_code.map(positions).to_numpy(dtype="int64")
        result.append((np.array([node_codes, group_codes], dtype=np.int32), len(groups)))
    return result


def _graphs(memberships, count, device):
    import torch

    result = []
    for rows, groups in memberships:
        indices = torch.as_tensor(rows, dtype=torch.long, device=device)
        incidence = torch.sparse_coo_tensor(
            indices, torch.ones(rows.shape[1], device=device), (count, groups), device=device,
        ).coalesce()
        result.append(IncidenceGraph(incidence))
    return result


class IncidenceGraph:
    """D^-1/2 (B B' - diag(B 1)) D^-1/2, without self-loops."""
    def __init__(self, incidence):
        import torch

        self.incidence = incidence
        self.transpose = incidence.transpose(0, 1).coalesce()
        ones = torch.ones((incidence.shape[0], 1), device=incidence.device)
        self.counts = torch.sparse.sum(incidence, dim=1).to_dense().reshape(-1, 1)
        degree = torch.sparse.mm(incidence, torch.sparse.mm(self.transpose, ones)) - self.counts
        self.connected = degree > 0
        self.scale = torch.where(self.connected, degree.clamp(min=1).rsqrt(), torch.zeros_like(degree))

    def aggregate(self, features):
        import torch

        scaled = features * self.scale
        summed = torch.sparse.mm(self.incidence, torch.sparse.mm(self.transpose, scaled))
        return (summed - self.counts * scaled) * self.scale
