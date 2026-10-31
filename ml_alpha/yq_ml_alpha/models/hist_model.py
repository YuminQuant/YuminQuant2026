"""DFQ-HIST: concept correction, hidden concepts and double residual learning."""
from __future__ import annotations

from yq_ml_alpha.output import artifact_io

from collections import deque
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from yq_ml_alpha.config import _project_path
from yq_ml_alpha.data.hist_concepts import HistConcepts
from yq_ml_alpha.models.mlp_model import (
    MLPAlphaModel, _cpu_state_dict, _device, _set_seed, _torch_modules,
)


def _params(raw):
    params = dict(
        sequence_length=60, hidden_size=128, num_layers=2, top_k=20,
        dropout=0.1, epochs=100, patience=20, average_epochs=5,
        batch_size=3000, graph_chunk_size=512,
        lr=0.001, weight_decay=0.00001, min_delta=1e-8,
        seed=42, device="auto", gradient_checkpointing=True,
        concept_root="data/derived/stock/hist_concepts",
        basic_root="data/chn_stock_data/daily/basic",
    )
    unknown = set(raw).difference(params)
    if unknown:
        raise ValueError(f"unknown HIST parameters: {sorted(unknown)}")
    params.update(raw)
    for key in ("sequence_length", "hidden_size", "num_layers", "top_k", "epochs",
                "patience", "average_epochs", "batch_size", "graph_chunk_size"):
        params[key] = int(params[key])
        if params[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if (not 0 <= params["dropout"] < 1 or params["lr"] <= 0
            or params["weight_decay"] < 0 or params["min_delta"] < 0):
        raise ValueError("invalid HIST optimization parameters")
    return params


def build_hist(nn, input_size, params):
    import torch
    import torch.nn.functional as functional
    from torch.utils.checkpoint import checkpoint

    hidden = params["hidden_size"]
    chunk_size = params["graph_chunk_size"]

    class ConceptModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.mix = nn.Linear(hidden, hidden)
            self.backcast = nn.Linear(hidden, hidden)
            self.forecast = nn.Sequential(nn.Linear(hidden, hidden), nn.LeakyReLU())

        def forward(self, shared):
            mixed = self.mix(shared)
            return self.backcast(mixed), self.forecast(mixed)

    class HIST(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(
                input_size, hidden, params["num_layers"], batch_first=True,
                dropout=params["dropout"] if params["num_layers"] > 1 else 0.0,
            )
            self.predefined = ConceptModule()
            self.hidden_concepts = ConceptModule()
            self.individual = nn.Sequential(nn.Linear(hidden, hidden), nn.LeakyReLU())
            self.output = nn.Linear(hidden, 1)
            for layer in self.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

        def encode(self, host_values, device):
            # Host inputs stay on CPU across checkpointed forward/backward.
            # A device token makes reentrant checkpoint track GRU parameter
            # gradients even though raw input features require no gradient.
            def run(values, token):
                return self.gru(values.to(device))[0][:, -1]

            if self.training and params["gradient_checkpointing"]:
                token = torch.zeros((), device=device, requires_grad=True)
                return checkpoint(run, host_values, token)
            return run(host_values, None)

        def attend(self, queries, keys, values):
            """Cosine attention, globally normalized over every key, in blocks."""
            def block(q, k, v):
                logits = functional.normalize(q, dim=1, eps=1e-8) @ functional.normalize(
                    k, dim=1, eps=1e-8).T
                return torch.softmax(logits, dim=1) @ v

            parts = []
            for start in range(0, len(queries), chunk_size):
                q = queries[start:start + chunk_size]
                if self.training and params["gradient_checkpointing"]:
                    parts.append(checkpoint(block, q, keys, values))
                else:
                    parts.append(block(q, keys, values))
            return torch.cat(parts)

        def predefined_shared(self, x, incidence):
            stocks, concepts, count, market_cap = incidence
            if count == 0:
                return torch.zeros_like(x)
            # Sparse edges, with each concept's market-cap weights summing to one.
            weights = market_cap[stocks]
            denominator = x.new_zeros(count).index_add(0, concepts, weights)
            initial = x.new_zeros((count, hidden)).index_add(
                0, concepts, x[stocks] * (weights / denominator[concepts]).unsqueeze(1))
            # First softmax is over ALL stocks for each concept, not vice versa.
            corrected = self.attend(initial, x, x)
            # The second softmax is over ALL corrected concepts for each stock.
            return self.attend(x, corrected, corrected)

        def hidden_shared(self, residual):
            count = len(residual)
            if count < 2:
                return None
            normalized = functional.normalize(residual, dim=1, eps=1e-8)
            concepts = torch.zeros_like(residual)
            connected = torch.zeros(count, dtype=torch.bool, device=residual.device)
            k = min(params["top_k"], count - 1)
            for start in range(0, count, chunk_size):
                stop = min(start + chunk_size, count)
                similarity = normalized[start:stop] @ normalized.T
                local = torch.arange(stop - start, device=residual.device)
                similarity[local, local + start] = -torch.inf
                weights, neighbors = similarity.topk(k, dim=1)
                # Keep the selected cosine weights, including negative values.
                contributions = residual[start:stop, None, :] * weights[:, :, None]
                concepts = concepts.index_add(0, neighbors.reshape(-1),
                                              contributions.reshape(-1, hidden))
                nonzero = weights.detach().ne(0)
                connected[neighbors[nonzero]] = True
            # A selected anchor contributes its own stock as in the HIST
            # reference construction. Unselected anchors are removed.
            self_similarity = normalized.square().sum(dim=1)
            concepts = concepts + residual * (connected * self_similarity).unsqueeze(1)
            keep = connected & concepts.detach().norm(dim=1).gt(1e-8)
            if not keep.any():
                return None
            concepts = concepts[keep]
            return self.attend(residual, concepts, concepts)

        def forward(self, encoded, incidence):
            back_pre, fore_pre = self.predefined(self.predefined_shared(encoded, incidence))
            # With no available concepts, omit that branch including learned bias.
            if incidence[2] == 0:
                back_pre, fore_pre = torch.zeros_like(encoded), torch.zeros_like(encoded)
            residual = encoded - back_pre
            shared = self.hidden_shared(residual)
            if shared is None:
                back_hidden, fore_hidden = torch.zeros_like(encoded), torch.zeros_like(encoded)
            else:
                back_hidden, fore_hidden = self.hidden_concepts(shared)
            individual = self.individual(residual - back_hidden)
            return self.output(fore_pre + fore_hidden + individual).squeeze(-1)

    return HIST()


def _rank_target(rows, label):
    raw = pd.to_numeric(rows[label], errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(raw)
    target = np.full(len(raw), np.nan, dtype=np.float32)
    if finite.sum() >= 2:
        ranks = pd.Series(raw[finite]).rank(pct=True).to_numpy()
        scale = ranks.std()
        if scale > 1e-8:
            target[finite] = (ranks - ranks.mean()) / scale
    return target


class HISTAlphaModel(MLPAlphaModel):
    def __init__(self):
        super().__init__()
        self.feature_columns = []

    def _source(self, bundle):
        source = bundle.tensor_source
        if (source is None or source.columns != self.feature_columns
                or source.length != self.params["sequence_length"]):
            raise ValueError("HIST requires matching daily factor sequence columns and length")
        return source

    def _score(self, date, rows, source, device):
        torch, _ = _torch_modules()
        symbols = rows.ts_code.tolist()
        if sorted(symbols) != source.symbols(date):
            raise ValueError("HIST requires the complete eligible date cross-section, including unlabeled nodes")
        encoded = []
        for start in range(0, len(symbols), self.params["batch_size"]):
            host = torch.from_numpy(source.get(date, symbols[start:start + self.params["batch_size"]]))
            encoded.append(self.model.encode(host, device))
        return self.model(torch.cat(encoded), self.concepts.load(date, symbols, device))

    def _evaluate(self, bundle, device):
        torch, _ = _torch_modules()
        source = self._source(bundle)
        self.model.eval()
        values = []
        with torch.no_grad():
            for date, rows in bundle.frame.groupby("trade_date", sort=True):
                target = _rank_target(rows, bundle.label_column)
                keep = np.isfinite(target)
                if keep.sum() < 2:
                    continue
                score = self._score(int(date), rows, source, device).cpu().numpy()[keep]
                if not np.isfinite(score).all():
                    raise ValueError(f"nonfinite HIST validation output on {date}")
                values.append(float(np.corrcoef(score, target[keep])[0, 1]) if score.std() > 1e-8 else 0.0)
        if not values:
            raise ValueError("HIST validation has no usable labeled cross-sections")
        return float(np.mean(values))

    def fit_bundle(self, train, valid, context):
        torch, nn = _torch_modules()
        self.params = _params(context.model_params)
        if context.model_search:
            raise ValueError("HIST does not use model.search")
        self.feature_columns = list(context.feature_columns)
        self.input_dim = len(self.feature_columns)
        source = self._source(train)
        self._source(valid)
        if train.frame.empty or valid.frame.empty:
            raise ValueError("HIST requires nonempty separate train and validation data")
        if train.frame.trade_date.max() >= valid.frame.trade_date.min():
            raise ValueError("HIST train/validation dates must be chronological and disjoint")
        self.concepts = HistConcepts(_project_path(self.params["concept_root"]),
                                     _project_path(self.params["basic_root"]))
        self.concepts.identity()
        _set_seed(torch, self.params["seed"])
        device = _device(torch, self.params["device"])
        self.model = build_hist(nn, self.input_dim, self.params).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"],
                                     weight_decay=self.params["weight_decay"])
        grouped = train.frame.groupby("trade_date", sort=True)
        days = np.asarray(sorted(grouped.groups))
        recent = deque(maxlen=self.params["average_epochs"])
        best, best_state, stale, best_epoch = -np.inf, None, 0, 0
        self.loss_history = []
        started = perf_counter()
        for epoch in range(1, self.params["epochs"] + 1):
            self.model.train()
            # Every epoch traverses all eligible training dates in calendar order.
            selected = days
            loss_sum, used = 0.0, 0
            for date in selected:
                rows = grouped.get_group(date)
                y = _rank_target(rows, context.label_column)
                keep = np.isfinite(y)
                if keep.sum() < 2:
                    continue
                optimizer.zero_grad(set_to_none=True)
                score = self._score(int(date), rows, source, device)
                mask = torch.as_tensor(keep, device=device)
                target = torch.as_tensor(y[keep], device=device)
                loss = (score[mask] - target).square().mean()
                if not torch.isfinite(loss):
                    raise ValueError(f"nonfinite HIST loss on {date}")
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                loss_sum += float(loss.detach())
                used += 1
                del score, loss
            if not used:
                raise ValueError("HIST epoch has no usable label cross-section")
            live = _cpu_state_dict(self.model)
            recent.append(live)
            averaged = {name: sum(state[name] for state in recent) / len(recent) for name in live}
            self.model.load_state_dict(averaged)
            try:
                metric = self._evaluate(valid, device)
            finally:
                # Evaluation must never replace the optimizer's live trajectory.
                self.model.load_state_dict(live)
            improved = metric > best + self.params["min_delta"]
            if improved:
                best, best_state, stale, best_epoch = metric, averaged, 0, epoch
            else:
                stale += 1
            self.loss_history.append(dict(
                window_id=context.artifact_dir.name, epoch=epoch, train_loss=loss_sum / used,
                valid_ic=metric, best_ic=best, is_best=improved, stale_epochs=stale,
                train_date_count=used, averaged_epochs=len(recent),
                elapsed_seconds=perf_counter() - started,
            ))
            if context.diagnostics.get("print_epoch", False):
                print(f"hist epoch={epoch} dates={used} loss={loss_sum / used:.6f} "
                      f"valid_ic={metric:.6f} average={len(recent)} stale={stale}", flush=True)
            if stale >= self.params["patience"]:
                break
        self.model.load_state_dict(best_state)
        self.model.eval()
        self.model_info = dict(
            best_epoch=best_epoch, best_valid_ic=best, feature_count=self.input_dim,
            params=self.params, concepts=self.concepts.identity(),
            loss="date_rank_pct_zscore_mse", validation="date_mean_pearson_ic",
            checkpoint="best_validation_trailing_epoch_parameter_mean",
        )

    def predict_bundle(self, bundle, context):
        torch, _ = _torch_modules()
        if list(context.feature_columns) != self.feature_columns:
            raise ValueError("HIST prediction feature order changed")
        source = self._source(bundle)
        device = _device(torch, self.params["device"])
        self.model.to(device).eval()
        self.concepts = HistConcepts(_project_path(self.params["concept_root"]),
                                     _project_path(self.params["basic_root"]))
        frame = bundle.frame.reset_index(drop=True)
        result = np.empty(len(frame), dtype=np.float32)
        with torch.no_grad():
            for date, rows in frame.groupby("trade_date", sort=True):
                scores = self._score(int(date), rows, source, device).cpu().numpy()
                if not np.isfinite(scores).all():
                    raise ValueError(f"nonfinite HIST predictions on {date}")
                result[rows.index.to_numpy()] = scores
        return pd.Series(result, index=bundle.frame.index)

    def save(self, path):
        torch, _ = _torch_modules()
        path = Path(path)
        artifact_io.mkdir(path.parent, parents=True, exist_ok=True)
        artifact_io.torch_save(dict(version=1, params=self.params, feature_columns=self.feature_columns,
                        state_dict=_cpu_state_dict(self.model), loss_history=self.loss_history,
                        model_info=self.model_info), path)

    @classmethod
    def load(cls, path):
        torch, nn = _torch_modules()
        saved = artifact_io.torch_load(path, map_location="cpu")
        if saved.get("version") != 1:
            raise ValueError("unsupported HIST checkpoint")
        model = cls()
        model.params = _params(saved["params"])
        model.feature_columns = saved["feature_columns"]
        model.input_dim = len(model.feature_columns)
        model.model = build_hist(nn, model.input_dim, model.params)
        model.model.load_state_dict(saved["state_dict"])
        model.model.eval()
        model.loss_history, model.model_info = saved["loss_history"], saved["model_info"]
        return model
