"""DFZQ spatial/temporal factor combination, with two second-stage rankers."""
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from yq_ml_alpha.config import _project_path
from yq_ml_alpha.data.stock_relations import StockRelations
from yq_ml_alpha.models.mlp_model import MLPAlphaModel, _cpu_state_dict, _device, _set_seed, _torch_modules
from yq_ml_alpha.models.multivariate_gru_model import _correlation_penalty
from yq_ml_alpha.models.graph_ranker import fit_ranker, predict_ranker


def standardize(values):
    values = np.asarray(values, dtype="float32")
    std = values.std()
    return (values - values.mean()) / std if std > 1e-8 else np.zeros_like(values)


def rank_ic(values, target):
    valid = np.isfinite(values) & np.isfinite(target)
    if valid.sum() < 2:
        return None
    x = pd.Series(values[valid]).rank().to_numpy()
    y = pd.Series(target[valid]).rank().to_numpy()
    if x.std() < 1e-8 or y.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _params(raw):
    params = dict(hidden_size=32, base_factors=32, gnn_layers=3, gru_layers=2,
                  sequence_length=40, dropout=0.5, orthogonal_lambda=0.05,
                  epochs=100, patience=20, lr=0.001, weight_decay=0.0,
                  batch_size=3000, seed=42, device="auto",
                  relation_root="data/derived/stock/relations",
                  ranker_trials=20, ranker_early_stopping=20, ranker_n_jobs=4,
                  ranker_device="auto")
    params.update(raw)
    for key in ("hidden_size", "base_factors", "gnn_layers", "gru_layers", "sequence_length",
                "epochs", "patience", "batch_size", "ranker_trials", "ranker_early_stopping", "ranker_n_jobs"):
        params[key] = int(params[key])
        if params[key] < 1:
            raise ValueError(f"{key} must be positive")
    if params["base_factors"] < 2 or params["batch_size"] < 3:
        raise ValueError("base_factors must be >= 2 and batch_size must be >= 3")
    if not 0 <= params["dropout"] < 1 or params["lr"] <= 0 or params["orthogonal_lambda"] < 0:
        raise ValueError("invalid graph/temporal optimizer parameters")
    return params


def build_encoder(nn, input_size, params, branch):
    import torch

    hidden = params["hidden_size"]

    class RelationLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.projections = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(3)])
            self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(3)])
            self.output_norm = nn.LayerNorm(hidden)
            self.dropout = nn.Dropout(params["dropout"])

        def forward(self, x, relations):
            messages = []
            for graph, projection, norm in zip(relations, self.projections, self.norms):
                # Mask after LayerNorm so its learned bias cannot create isolated neighbors.
                value = norm(torch.relu(graph.aggregate(projection(x))))
                messages.append(value * graph.connected)
            neighbor = sum(messages) / 3.0
            return self.output_norm(0.5 * x + 0.5 * self.dropout(neighbor))

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            if branch == "gnn":
                self.input = nn.Linear(input_size, hidden)
                self.layers = nn.ModuleList([RelationLayer() for _ in range(params["gnn_layers"])])
            elif branch == "gru":
                self.gru = nn.GRU(input_size, hidden, num_layers=params["gru_layers"], batch_first=True,
                                  dropout=params["dropout"] if params["gru_layers"] > 1 else 0.0)
            else:
                raise ValueError(f"unknown encoder: {branch}")
            self.factor_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                             nn.Dropout(params["dropout"]),
                                             nn.Linear(hidden, params["base_factors"]))
            self.factor_norm = nn.BatchNorm1d(params["base_factors"])
            self.score_norm = nn.BatchNorm1d(1)
            self.register_buffer("equal_weights", torch.full((params["base_factors"],), 1 / params["base_factors"]))

        def factors(self, x, relations=None):
            if branch == "gnn":
                x = self.input(x)
                for layer in self.layers:
                    x = layer(x, relations)
            else:
                x = self.gru(x)[0][:, -1]
            return self.factor_norm(self.factor_head(x))

        def forward(self, x, relations=None):
            base = self.factors(x, relations)
            return base, self.score_norm((base @ self.equal_weights).reshape(-1, 1)).reshape(-1)

    return Encoder()


def _batches(count, size):
    # Avoid a singleton BatchNorm batch without increasing the requested batch size.
    start = 0
    while start < count:
        stop = min(start + size, count)
        if count - stop == 1 and stop - start > 2:
            stop -= 1
        yield slice(start, stop)
        start = stop


class GraphTemporalAlphaModel(MLPAlphaModel):
    """Full-date GNN, stock-batched GRU, frozen factors -> two Optuna rankers."""
    def __init__(self):
        super().__init__()
        self.encoders = {}
        self.rankers = {}
        self.feature_columns = []

    def _check_source(self, bundle):
        source = bundle.tensor_source
        if source is None or source.columns != self.feature_columns or source.length != self.params["sequence_length"]:
            raise ValueError("graph model requires the checkpoint's factor columns and daily history length")
        return source

    def _inputs(self, branch, date, rows, source, device):
        torch, _ = _torch_modules()
        symbols = rows.ts_code.tolist()
        if branch == "gnn":
            values = source.day(date).loc[symbols].to_numpy(dtype="float32")
            return torch.from_numpy(values).to(device), self.relations.load(date, symbols, device)
        values = source.get(date, symbols)
        return torch.from_numpy(values).to(device), None

    def _date_chunks(self, branch, rows):
        return [slice(0, len(rows))] if branch == "gnn" else _batches(len(rows), self.params["batch_size"])

    def _evaluate(self, branch, bundle, device, extract=False):
        torch, _ = _torch_modules()
        source = self._check_source(bundle)
        encoder = self.encoders[branch]
        encoder.eval()
        metrics, features, labels, groups = [], [], [], []
        with torch.no_grad():
            for date, rows in bundle.frame.groupby("trade_date", sort=True):
                outputs, bases = [], []
                for chunk in self._date_chunks(branch, rows):
                    x, relations = self._inputs(branch, int(date), rows.iloc[chunk], source, device)
                    base, score = encoder(x, relations)
                    outputs.append(score.cpu().numpy())
                    if extract:
                        bases.append(base.cpu().numpy())
                target = pd.to_numeric(rows[bundle.label_column], errors="coerce").to_numpy(dtype="float32")
                metric = rank_ic(np.concatenate(outputs), target)
                if metric is not None:
                    metrics.append(metric)
                mask = np.isfinite(target)
                if extract and mask.sum() >= 2:
                    features.append(np.concatenate(bases)[mask])
                    labels.append(target[mask])
                    groups.append(int(mask.sum()))
        if not metrics:
            raise ValueError(f"{branch}: no labeled validation cross-sections")
        if extract:
            if not features:
                raise ValueError("no labeled ranker groups")
            return np.concatenate(features), np.concatenate(labels), groups
        return float(np.mean(metrics))

    def _fit_encoder(self, branch, train, valid, context, device):
        torch, nn = _torch_modules()
        source = self._check_source(train)
        encoder = build_encoder(nn, len(self.feature_columns), self.params, branch).to(device)
        self.encoders[branch] = encoder
        optimizer = torch.optim.Adam(encoder.parameters(), lr=self.params["lr"], weight_decay=self.params["weight_decay"])
        best, best_state, best_epoch, stale = -np.inf, None, 0, 0
        started = perf_counter()
        for epoch in range(1, self.params["epochs"] + 1):
            encoder.train()
            loss_sum, count = 0.0, 0
            # Chronological traversal gives adjacent samples reusable daily history.
            for date, rows in train.frame.groupby("trade_date", sort=True):
                target = pd.to_numeric(rows[context.label_column], errors="coerce").to_numpy(dtype="float32")
                finite = np.isfinite(target)
                if finite.sum() < 2 or target[finite].std() < 1e-8:
                    continue
                target[finite] = standardize(target[finite])
                for chunk in self._date_chunks(branch, rows):
                    if len(rows.iloc[chunk]) < 2 or np.isfinite(target[chunk]).sum() < 2:
                        continue
                    x, relations = self._inputs(branch, int(date), rows.iloc[chunk], source, device)
                    y = torch.from_numpy(target[chunk]).to(device)
                    keep = torch.isfinite(y)
                    optimizer.zero_grad()
                    base, score = encoder(x, relations)
                    loss = (score[keep] - y[keep]).square().mean()
                    loss = loss + self.params["orthogonal_lambda"] * _correlation_penalty(torch, base[keep])
                    if not torch.isfinite(loss):
                        raise ValueError(f"nonfinite {branch} loss on {date}")
                    loss.backward()
                    nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
                    optimizer.step()
                    n = int(keep.sum())
                    loss_sum += float(loss.detach()) * n
                    count += n
            if not count:
                raise ValueError(f"{branch}: no usable training labels")
            metric = self._evaluate(branch, valid, device)
            improved = metric > best + 1e-8
            if improved:
                best, best_state, best_epoch, stale = metric, _cpu_state_dict(encoder), epoch, 0
            else:
                stale += 1
            row = dict(window_id=context.artifact_dir.name, branch=branch, epoch=epoch,
                       train_loss=loss_sum / count, valid_rank_ic=metric, best_rank_ic=best,
                       is_best=improved, stale_epochs=stale, elapsed_seconds=perf_counter() - started)
            self.loss_history.append(row)
            if context.diagnostics.get("print_epoch", True):
                print(f"graph-temporal branch={branch} epoch={epoch} loss={row['train_loss']:.6f} "
                      f"valid_rank_ic={metric:.6f} stale={stale}", flush=True)
            if stale >= self.params["patience"]:
                break
        encoder.load_state_dict(best_state)
        encoder.eval()
        self.model_info[branch] = dict(best_epoch=best_epoch, best_valid_rank_ic=best)

    def fit_bundle(self, train, valid, context):
        torch, _ = _torch_modules()
        self.params = _params(context.model_params)
        self.feature_columns = list(context.feature_columns)
        self._check_source(train)
        self._check_source(valid)
        if train.frame.empty or valid.frame.empty:
            raise ValueError("graph/temporal fitting requires separate train and validation dates")
        if int(train.frame.trade_date.max()) >= int(valid.frame.trade_date.min()):
            raise ValueError("graph/temporal train and validation dates must be disjoint and chronological")
        self.relations = StockRelations(_project_path(self.params["relation_root"]))
        self.relations.identity()
        device = _device(torch, self.params["device"])
        self.loss_history, self.model_info = [], {}
        for branch in ("gnn", "gru"):
            _set_seed(torch, self.params["seed"])
            self._fit_encoder(branch, train, valid, context, device)
            print(f"graph-temporal branch={branch} step=extract_frozen_factors", flush=True)
            train_matrix = self._evaluate(branch, train, device, extract=True)
            valid_matrix = self._evaluate(branch, valid, device, extract=True)
            # XGBoost owns the device during stage two. The frozen encoder stays on CPU.
            self.encoders[branch].cpu()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            booster, details = fit_ranker(train_matrix, valid_matrix, self.params, branch, device.type)
            self.rankers[branch] = booster
            self.model_info[branch]["ranker"] = details
            del train_matrix, valid_matrix
        self.model_info.update(feature_count=len(self.feature_columns), params=self.params,
                               relations=self.relations.identity(), output="zscore(gnn_ranker)+zscore(gru_ranker)",
                               loss="date_zscore_mse_plus_factor_correlation", seed=self.params["seed"])

    def predict_bundle(self, bundle, context):
        torch, _ = _torch_modules()
        source = self._check_source(bundle)
        if list(context.feature_columns) != self.feature_columns:
            raise ValueError("graph model feature order mismatch")
        device = _device(torch, self.params["device"])
        self.relations = StockRelations(_project_path(self.params["relation_root"]))
        result = np.empty(len(bundle.frame), dtype="float32")
        frame = bundle.frame.reset_index(drop=True)
        for branch in ("gnn", "gru"):
            self.encoders[branch].to(device).eval()
        with torch.no_grad():
            for date, rows in frame.groupby("trade_date", sort=True):
                if sorted(rows.ts_code.tolist()) != source.symbols(int(date)):
                    raise ValueError("graph prediction requires a complete eligible cross-section")
                scores = []
                for branch in ("gnn", "gru"):
                    values = []
                    for chunk in self._date_chunks(branch, rows):
                        x, relations = self._inputs(branch, int(date), rows.iloc[chunk], source, device)
                        values.append(self.encoders[branch].factors(x, relations).cpu().numpy())
                    score = predict_ranker(self.rankers[branch], np.concatenate(values),
                                           self.model_info[branch]["ranker"]["best_rounds"],
                                           self.params["ranker_n_jobs"])
                    scores.append(standardize(score))
                result[rows.index.to_numpy()] = scores[0] + scores[1]
        return pd.Series(result, index=bundle.frame.index)

    def save(self, path):
        torch, _ = _torch_modules()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(version=1, params=self.params, feature_columns=self.feature_columns,
                        encoders={key: _cpu_state_dict(model) for key, model in self.encoders.items()},
                        rankers={key: bytes(model.save_raw()) for key, model in self.rankers.items()},
                        loss_history=self.loss_history, model_info=self.model_info), path)

    @classmethod
    def load(cls, path):
        import xgboost as xgb
        torch, nn = _torch_modules()
        saved = torch.load(path, map_location="cpu")
        if saved.get("version") != 1:
            raise ValueError("unsupported graph/temporal checkpoint version")
        model = cls()
        model.params, model.feature_columns = saved["params"], saved["feature_columns"]
        model.loss_history, model.model_info = saved["loss_history"], saved["model_info"]
        for branch in ("gnn", "gru"):
            encoder = build_encoder(nn, len(model.feature_columns), model.params, branch)
            encoder.load_state_dict(saved["encoders"][branch])
            model.encoders[branch] = encoder.eval()
            booster = xgb.Booster()
            booster.load_model(bytearray(saved["rankers"][branch]))
            model.rankers[branch] = booster
        return model
