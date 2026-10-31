"""DFQ-FactorGCL with historical hypergraphs and training-only future supervision."""
from __future__ import annotations

from yq_ml_alpha.output import artifact_io

from collections import deque
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from yq_ml_alpha.config import _project_path
from yq_ml_alpha.data.hist_concepts import HistConcepts
from yq_ml_alpha.features.transforms import _zscore_series
from yq_ml_alpha.models.bar_gru_model import _release_torch_memory
from yq_ml_alpha.models.contrastive import residual_contrastive_loss
from yq_ml_alpha.models.hypergraph import propagate_hypergraph, subset_binary_incidence
from yq_ml_alpha.models.mlp_model import MLPAlphaModel, _cpu_state_dict, _device, _set_seed, _torch_modules


def _params(raw):
    params = dict(sequence_length=30, future_look=10, hidden_size=128, num_layers=2,
                  num_factor=48, epochs=200, patience=20, average_epochs=5,
                  batch_size=3000, contrastive_chunk_size=512, lr=0.00009, weight_decay=0.0,
                  gamma=1.0, tau=0.25, min_delta=1e-8, negative_slope=0.01,
                  seed=42, device="auto", gradient_checkpointing=True,
                  concept_root="data/derived/stock/hist_concepts")
    unknown = set(raw).difference(params)
    if unknown:
        raise ValueError(f"unknown FactorGCL parameters: {sorted(unknown)}")
    params.update(raw)
    for key in ("sequence_length", "future_look", "hidden_size", "num_layers", "num_factor",
                "epochs", "patience", "average_epochs", "batch_size", "contrastive_chunk_size"):
        value = params[key]
        params[key] = int(value)
        if params[key] <= 0 or params[key] != value:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("lr", "weight_decay", "gamma", "tau", "min_delta", "negative_slope"):
        params[key] = float(params[key])
        if not np.isfinite(params[key]) or params[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if not params["lr"] or not params["tau"]:
        raise ValueError("lr and tau must be positive")
    if not isinstance(params["gradient_checkpointing"], bool):
        raise ValueError("gradient_checkpointing must be boolean")
    params["seed"] = int(params["seed"])
    return params


def build_factor_gcl(nn, input_size, params):
    import torch
    from torch.utils.checkpoint import checkpoint

    width = params["hidden_size"]

    class HyperGCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.transform = nn.Linear(width, width, bias=False)
            self.activation = nn.LeakyReLU(params["negative_slope"])

        def forward(self, x, incidence):
            return self.activation(propagate_hypergraph(self.transform(x), incidence))

    class FactorGCL(nn.Module):
        def __init__(self):
            super().__init__()
            self.past_encoder = nn.GRU(input_size, width, params["num_layers"], batch_first=True)
            self.future_encoder = nn.GRU(input_size, width, params["num_layers"], batch_first=True)
            self.prior = HyperGCN()
            self.hidden = HyperGCN()
            self.prototypes = nn.Parameter(torch.randn(params["num_factor"], width) / width ** 0.5)
            self.alpha = nn.Sequential(nn.Linear(width, width), nn.LeakyReLU(params["negative_slope"]))
            self.head = nn.Linear(3 * width, 1)
            self.projection = nn.Sequential(nn.Linear(width, width), nn.LeakyReLU(params["negative_slope"]),
                                            nn.Linear(width, width))

        def encode(self, host, device, future=False):
            encoder = self.future_encoder if future else self.past_encoder

            def run(values, token):
                return encoder(values.to(device))[1][-1]

            if self.training and params["gradient_checkpointing"]:
                token = torch.zeros((), device=device, requires_grad=True)
                return checkpoint(run, host, token)
            return run(host, None)

        def historical(self, encoded, incidence):
            prior = self.prior(encoded, incidence)
            residual = encoded - prior
            beta = torch.sigmoid(residual @ self.prototypes.T)
            hidden = self.hidden(residual, beta)
            alpha = self.alpha(residual - hidden)
            score = self.head(torch.cat([prior, hidden, alpha], dim=1)).squeeze(-1)
            return score, alpha, beta

        def future_alpha(self, encoded, incidence, historical_beta):
            residual = encoded - self.prior(encoded, incidence)
            return self.alpha(residual - self.hidden(residual, historical_beta))

    return FactorGCL()


def _target(rows, label):
    raw = pd.to_numeric(rows[label], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return _zscore_series(raw.rank(method="average", pct=True)).to_numpy(dtype=np.float32)


class FactorGCLAlphaModel(MLPAlphaModel):
    def __init__(self):
        super().__init__()
        self.feature_columns = []

    def fit(self, train_data, valid_data, context):
        raise ValueError("FactorGCL requires concept_factor_sequence bundles")

    def predict(self, data, context):
        raise ValueError("FactorGCL requires complete-date predict_bundle")

    def _source(self, bundle):
        source = bundle.tensor_source
        if (source is None or list(bundle.feature_columns) != self.feature_columns
                or source.columns != self.feature_columns or source.length != self.params["sequence_length"]
                or getattr(source, "history_transform", None) != "zscore_clip3"):
            raise ValueError("FactorGCL requires matching zscore_clip3 factor histories")
        return source

    def _encode(self, date, symbols, source, device, future=False):
        torch, _ = _torch_modules()
        chunks = []
        for start in range(0, len(symbols), self.params["batch_size"]):
            host = torch.from_numpy(source.get(date, symbols[start:start + self.params["batch_size"]]))
            chunks.append(self.model.encode(host, device, future=future))
        return torch.cat(chunks)

    def _historical(self, date, rows, source, device):
        symbols = rows.ts_code.tolist()
        if len(set(symbols)) != len(symbols) or sorted(symbols) != source.symbols(date):
            raise ValueError("FactorGCL must retain complete historical date nodes, including unlabeled stocks")
        incidence = self.concepts.load_binary(date, symbols, device)
        score, alpha, beta = self.model.historical(self._encode(date, symbols, source, device), incidence)
        return score, alpha, beta, incidence

    def _future_ends(self, train, valid, source):
        if train.information_cutoff is None or train.supervision_lookahead_days < self.params["future_look"]:
            raise ValueError("FactorGCL requires a shared training cutoff and declared auxiliary supervision horizon")
        first_valid = int(valid.frame.trade_date.min())
        ends = {}
        for date in train.frame.trade_date.unique():
            position = source.positions[int(date)] + self.params["future_look"]
            if position >= len(source.dates):
                raise ValueError(f"missing forward supervision calendar on {date}")
            end = source.dates[position]
            if end > train.information_cutoff or end >= first_valid:
                raise ValueError(f"future supervision on {date} crosses training information/split boundary")
            ends[int(date)] = end
        return ends

    def _evaluate(self, bundle, device):
        torch, _ = _torch_modules()
        source = self._source(bundle)
        self.model.eval()
        metrics = []
        with torch.no_grad():
            for date, rows in bundle.frame.groupby("trade_date", sort=True):
                y = _target(rows, bundle.label_column)
                keep = np.isfinite(y)
                if keep.sum() < 2 or y[keep].std() <= 1e-8:
                    continue
                score = self._historical(int(date), rows, source, device)[0].cpu().numpy()[keep]
                if not np.isfinite(score).all():
                    raise ValueError(f"nonfinite FactorGCL validation score on {date}")
                metrics.append(float(np.corrcoef(score, y[keep])[0, 1]) if score.std() > 1e-8 else 0.0)
        if not metrics:
            raise ValueError("FactorGCL validation has no usable labeled dates")
        return float(np.mean(metrics))

    def fit_bundle(self, train, valid, context):
        self.params = _params(context.model_params)
        if context.model_search:
            raise ValueError("FactorGCL does not use model.search")
        self.feature_columns = list(context.feature_columns)
        self.input_dim = len(self.feature_columns)
        source = self._source(train)
        self._source(valid)
        if not self.input_dim or train.frame.empty or valid.frame.empty:
            raise ValueError("FactorGCL requires factors and nonempty chronological train/validation partitions")
        if train.frame.trade_date.max() >= valid.frame.trade_date.min():
            raise ValueError("FactorGCL train/validation dates must be chronological and disjoint")
        if train.label_column != context.label_column or valid.label_column != context.label_column:
            raise ValueError("FactorGCL label columns must match context")
        ends = self._future_ends(train, valid, source)
        self.concepts = HistConcepts(_project_path(self.params["concept_root"]))
        self.concepts.identity()
        grouped = train.frame.groupby("trade_date", sort=True)
        days = np.asarray(sorted(grouped.groups))
        targets = {int(day): _target(grouped.get_group(day), context.label_column) for day in days}
        future_source = source.forward_view(self.params["future_look"])
        torch, nn = _torch_modules()
        _set_seed(torch, self.params["seed"])
        device = _device(torch, self.params["device"])
        self.model = build_factor_gcl(nn, self.input_dim, self.params).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"], weight_decay=self.params["weight_decay"])
        recent = deque(maxlen=self.params["average_epochs"])
        best, best_state, best_epoch, stale = -np.inf, None, 0, 0
        self.loss_history = []
        started = perf_counter()
        try:
            for epoch in range(1, self.params["epochs"] + 1):
                self.model.train()
                selected = days
                totals = np.zeros(4)
                used, paired, missing, unpaired_dates = 0, 0, 0, 0
                for date in selected:
                    date = int(date)
                    rows = grouped.get_group(date)
                    y = targets[date]
                    keep = np.isfinite(y)
                    if keep.sum() < 2 or y[keep].std() <= 1e-8:
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    score, past_alpha, beta, incidence = self._historical(date, rows, source, device)
                    mask = torch.as_tensor(keep, device=device)
                    mse = (score[mask] - torch.as_tensor(y[keep], device=device)).square().mean()
                    projected = self.model.projection(past_alpha)
                    kwargs = dict(temperature=self.params["tau"], chunk_size=self.params["contrastive_chunk_size"],
                                  checkpointing=self.params["gradient_checkpointing"])
                    cross = residual_contrastive_loss(projected, projected, **kwargs)
                    end = ends[date]
                    available = set(future_source.symbols(end))
                    symbols = rows.ts_code.tolist()
                    pair_rows = [i for i, symbol in enumerate(symbols) if symbol in available]
                    missing += len(symbols) - len(pair_rows)
                    temporal = projected.sum() * 0
                    if len(pair_rows) >= 2:
                        positions = torch.as_tensor(pair_rows, dtype=torch.long, device=device)
                        codes = [symbols[i] for i in pair_rows]
                        future_encoded = self._encode(end, codes, future_source, device, future=True)
                        future_incidence = subset_binary_incidence(incidence, positions, len(symbols))
                        # Restrict historical beta rows; never re-estimate beta using future features.
                        future_alpha = self.model.future_alpha(future_encoded, future_incidence, beta[positions])
                        temporal = residual_contrastive_loss(projected[positions], self.model.projection(future_alpha),
                                                             squared=True, **kwargs)
                        paired += len(pair_rows)
                        del future_encoded, future_alpha
                    else:
                        unpaired_dates += 1
                    loss = mse + self.params["gamma"] * (temporal + cross)
                    if not bool(torch.isfinite(loss).item()):
                        raise ValueError(f"nonfinite FactorGCL loss on {date}")
                    loss.backward()
                    finite = torch.stack([torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None])
                    if not bool(finite.all().item()):
                        raise ValueError(f"nonfinite FactorGCL gradients on {date}")
                    optimizer.step()
                    totals += [value.detach().item() for value in (loss, mse, temporal, cross)]
                    used += 1
                    del score, past_alpha, beta, projected, loss, mse, temporal, cross
                if not used:
                    raise ValueError("FactorGCL epoch has no usable labeled dates")
                live = _cpu_state_dict(self.model)
                recent.append(live)
                averaged = {name: sum(state[name] for state in recent) / len(recent) for name in live}
                self.model.load_state_dict(averaged)
                try:
                    metric = self._evaluate(valid, device)
                finally:
                    self.model.load_state_dict(live)
                if not np.isfinite(metric):
                    raise ValueError("nonfinite FactorGCL validation IC")
                improved = metric > best + self.params["min_delta"]
                if improved:
                    best, best_state, best_epoch, stale = metric, averaged, epoch, 0
                else:
                    stale += 1
                losses = (totals / used).tolist()
                self.loss_history.append(dict(
                    window_id=context.artifact_dir.name, epoch=epoch, train_loss=losses[0], mse=losses[1],
                    temporal_nce=losses[2], cross_section_nce=losses[3], valid_ic=metric, best_ic=best,
                    is_best=improved, stale_epochs=stale, train_date_count=used, averaged_epochs=len(recent),
                    temporal_pairs=paired, missing_future_stocks=missing, dates_without_temporal_pairs=unpaired_dates,
                    elapsed_seconds=perf_counter() - started,
                ))
                if context.diagnostics.get("print_epoch", False):
                    print(f"factorgcl window={context.artifact_dir.name} epoch={epoch} loss={losses[0]:.6g} "
                          f"mse={losses[1]:.6g} temporal={losses[2]:.6g} cross={losses[3]:.6g} "
                          f"valid_ic={metric:.6g} pairs={paired} missing_future={missing} stale={stale}", flush=True)
                if stale >= self.params["patience"]:
                    break
            self.model.load_state_dict(best_state)
        finally:
            future_source.close()
            self.model.cpu().eval()
            del optimizer
            _release_torch_memory(torch)
        self.model_info = dict(
            model_class=self.__class__.__name__, alpha_id=context.alpha_id, window_id=context.artifact_dir.name,
            feature_columns=self.feature_columns, params=self.params, best_epoch=best_epoch, best_valid_ic=best,
            epochs_run=len(self.loss_history), train_rows=len(train.frame), valid_rows=len(valid.frame),
            information_cutoff=train.information_cutoff, last_future_feature=max(ends.values()),
            declared_supervision_lookahead_days=train.supervision_lookahead_days, device=str(device),
            label_transform="cross_section_rank_pct_zscore", incidence="binary_prior_and_sigmoid_historical_prototypes",
            temporal_loss="signed_square_positive_square_denominator_mean", cross_loss="cosine_infonce_mean",
            inference="historical_only_complete_date", checkpoint="best_validation_trailing_epoch_parameter_mean",
        )

    def predict_bundle(self, bundle, context):
        if self.model is None:
            raise RuntimeError("FactorGCL is not fitted")
        if list(context.feature_columns) != self.feature_columns:
            raise ValueError("FactorGCL prediction feature order changed")
        source = self._source(bundle)
        torch, _ = _torch_modules()
        device = _device(torch, self.params["device"])
        self.model.to(device).eval()
        frame = bundle.frame.reset_index(drop=True)
        result = np.empty(len(frame), dtype=np.float32)
        try:
            with torch.no_grad():
                for date, rows in frame.groupby("trade_date", sort=True):
                    score = self._historical(int(date), rows, source, device)[0].cpu().numpy()
                    if not np.isfinite(score).all():
                        raise ValueError(f"nonfinite FactorGCL predictions on {date}")
                    result[rows.index.to_numpy()] = score
            return pd.Series(result, index=bundle.frame.index)
        finally:
            if not getattr(self, "_prediction_active", False):
                self.model.cpu()
                _release_torch_memory(torch)

    def save(self, path):
        if self.model is None:
            raise RuntimeError("cannot save unfitted FactorGCL")
        torch, _ = _torch_modules()
        path = Path(path)
        artifact_io.mkdir(path.parent, parents=True, exist_ok=True)
        artifact_io.torch_save(dict(version=1, model_kind="factor_gcl", params=self.params, feature_columns=self.feature_columns,
                        state_dict=_cpu_state_dict(self.model), loss_history=self.loss_history, model_info=self.model_info), path)

    @classmethod
    def load(cls, path):
        torch, nn = _torch_modules()
        saved = artifact_io.torch_load(path, map_location="cpu")
        if saved.get("version") != 1 or saved.get("model_kind") != "factor_gcl":
            raise ValueError("unsupported FactorGCL checkpoint")
        model = cls()
        model.params = _params(saved["params"])
        model.feature_columns = list(saved["feature_columns"])
        model.input_dim = len(model.feature_columns)
        model.model = build_factor_gcl(nn, model.input_dim, model.params)
        model.model.load_state_dict(saved["state_dict"])
        model.model.eval()
        model.concepts = HistConcepts(_project_path(model.params["concept_root"]))
        model.loss_history, model.model_info = saved["loss_history"], saved["model_info"]
        return model
