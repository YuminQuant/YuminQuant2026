"""Huatai AI 58: single-head GAT with binary analyst coverage and a linear skip."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from yq_ml_alpha.features.transforms import _zscore_series
from yq_ml_alpha.models.bar_gru_model import _cpu_state_dict, _device, _release_torch_memory, _set_seed, _torch_modules
from yq_ml_alpha.models.mlp_model import MLPAlphaModel, _diagnostics
from yq_ml_alpha.output import artifact_io


def _params(raw):
    params = dict(hidden_size=5, sequence_length=1, negative_slope=0.2,
                  epochs=100, patience=10, lr=0.0005, weight_decay=0.0,
                  batch_size=3000, graph_chunk_size=512, gradient_checkpointing=True,
                  min_delta=1e-12, seed=42, device="auto")
    unknown = set(raw).difference(params)
    if unknown:
        raise ValueError(f"unknown analyst coverage GAT parameters: {sorted(unknown)}")
    params.update(raw)
    for key in ("hidden_size", "sequence_length", "epochs", "patience", "batch_size", "graph_chunk_size", "seed"):
        value = params[key]
        params[key] = int(value)
        if isinstance(value, bool) or params[key] != value or (key != "seed" and params[key] <= 0):
            raise ValueError(f"invalid integer GAT parameter {key}")
    if params["sequence_length"] != 1:
        raise ValueError("analyst coverage GAT uses only the current factor snapshot")
    if params["batch_size"] > 3000 or params["graph_chunk_size"] > params["batch_size"]:
        raise ValueError("GAT query chunks must not exceed stock chunks, capped at 3000")
    if not isinstance(params["gradient_checkpointing"], bool):
        raise ValueError("gradient_checkpointing must be boolean")
    for key in ("negative_slope", "lr", "weight_decay", "min_delta"):
        params[key] = float(params[key])
        if not np.isfinite(params[key]) or params[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if params["lr"] <= 0:
        raise ValueError("lr must be positive")
    return params


def build_analyst_gat(nn, input_size, params):
    import torch
    import torch.nn.functional as functional
    from torch.utils.checkpoint import checkpoint

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            width = params["hidden_size"]
            # 3 input factors and width=5 give the report's 51 parameters.
            self.projection = nn.Linear(input_size, width, bias=False)
            self.attention_source = nn.Parameter(torch.empty(width))
            self.attention_target = nn.Parameter(torch.empty(width))
            self.skip = nn.Linear(input_size, width, bias=True)
            self.head = nn.Linear(width, 1)
            for layer in (self.projection, self.skip, self.head):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            nn.init.xavier_uniform_(self.attention_source.view(1, -1))
            nn.init.xavier_uniform_(self.attention_target.view(1, -1))

        def _messages(self, encoded, start, graph):
            stop = min(start + params["graph_chunk_size"], len(encoded))
            source = encoded[start:stop] @ self.attention_source
            target = encoded @ self.attention_target
            logits = functional.leaky_relu(source[:, None] + target[None, :], params["negative_slope"])
            # Rebuild this block on backward as well: no retained N x N mask.
            mask = graph.mask(start, stop)
            if mask.shape != (stop - start, len(encoded)):
                raise ValueError("analyst coverage graph block shape differs from model nodes")
            mask = torch.from_numpy(mask).to(encoded.device)
            attention = torch.softmax(logits.masked_fill(~mask, -torch.inf), dim=1)
            return attention @ encoded

        def forward(self, host_values, graph, device):
            if host_values.ndim != 2 or host_values.shape[1] != input_size or len(host_values) != graph.node_count:
                raise ValueError("analyst GAT expects a complete [stocks, factors] snapshot matching graph nodes")
            projections, skips = [], []
            for start in range(0, len(host_values), params["batch_size"]):
                values = host_values[start:start + params["batch_size"]].to(device)
                projections.append(self.projection(values))
                skips.append(self.skip(values))
            encoded, residual = torch.cat(projections), torch.cat(skips)
            messages = []
            for start in range(0, len(encoded), params["graph_chunk_size"]):
                # The explicit start argument avoids late-bound loop closures.
                if self.training and params["gradient_checkpointing"]:
                    messages.append(checkpoint(self._messages, encoded, start, graph))
                else:
                    messages.append(self._messages(encoded, start, graph))
            hidden = residual + functional.leaky_relu(torch.cat(messages), params["negative_slope"])
            return self.head(hidden).squeeze(-1)

    return Model()


def _target(rows, label):
    raw = pd.to_numeric(rows[label], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return _zscore_series(raw).to_numpy(dtype="float32")


class AnalystCoverageGATAlphaModel(MLPAlphaModel):
    prediction_replay_safe = True
    prediction_inputs_readonly = True

    def __init__(self):
        super().__init__()
        self.feature_columns = []

    def fit(self, train_data, valid_data, context):
        raise ValueError("analyst coverage GAT requires analyst_factor_sequence and fit_bundle")

    def predict(self, data, context):
        raise ValueError("analyst coverage GAT requires complete-date predict_bundle")

    def _source(self, bundle):
        source = bundle.tensor_source
        if (source is None or not hasattr(source, "coverage") or source.length != 1
                or source.columns != self.feature_columns or list(bundle.feature_columns) != self.feature_columns
                or source.history_transform != "mad5_zscore_rank_label"):
            raise ValueError("analyst GAT requires matching single-date covered-stock factor snapshots")
        return source

    def _forward_date(self, date, rows, source, device):
        torch, _ = _torch_modules()
        symbols = rows.ts_code.tolist()
        expected = source.symbols(int(date))
        if len(symbols) != len(set(symbols)) or len(symbols) != len(expected) or set(symbols) != set(expected):
            raise ValueError(f"analyst GAT requires the complete eligible covered-stock cross-section on {date}")
        graph = source.coverage.graph(int(date), symbols)
        host_values = torch.from_numpy(np.ascontiguousarray(source.get(int(date), symbols)[:, 0]))
        return self.model(host_values, graph, device)

    def _evaluate(self, bundle, context, device):
        torch, _ = _torch_modules()
        source = self._source(bundle)
        self.model.eval()
        losses = []
        with torch.no_grad():
            for date, rows in bundle.frame.groupby("trade_date", sort=True):
                y = _target(rows, context.label_column)
                keep = np.isfinite(y)
                if keep.sum() < 2 or y[keep].std() <= 1e-8:
                    continue
                prediction = self._forward_date(date, rows, source, device).cpu().numpy()
                if not np.isfinite(prediction).all():
                    raise ValueError(f"nonfinite analyst GAT validation score on {date}")
                losses.append(float(np.mean((prediction[keep] - y[keep]) ** 2)))
        if not losses:
            raise ValueError("analyst GAT validation has no usable labeled cross-sections")
        return float(np.mean(losses))

    def fit_bundle(self, train, valid, context):
        if context.model_search:
            raise ValueError("analyst coverage GAT does not use model.search")
        if context.label_columns and context.label_columns != [context.label_column]:
            raise ValueError("analyst coverage GAT supports one return label")
        self.params = _params(context.model_params)
        self.feature_columns = list(context.feature_columns)
        self.input_dim = len(self.feature_columns)
        source = self._source(train)
        self._source(valid)
        if not self.input_dim or train.frame.empty or valid.frame.empty:
            raise ValueError("analyst GAT needs factors and nonempty train/validation partitions")
        if train.label_column != context.label_column or valid.label_column != context.label_column:
            raise ValueError("analyst GAT label columns must match context")
        if train.frame.trade_date.max() >= valid.frame.trade_date.min():
            raise ValueError("analyst GAT validation dates must follow all training dates")
        groups = []
        for date, rows in train.frame.groupby("trade_date", sort=True):
            y = _target(rows, context.label_column)
            keep = np.isfinite(y)
            if keep.sum() >= 2 and y[keep].std() > 1e-8:
                groups.append((int(date), rows, y, keep))
        if not groups:
            raise ValueError("analyst GAT training has no usable labeled cross-sections")
        quarter_groups = {}
        for index, (date, _, _, _) in enumerate(groups):
            quarter_groups.setdefault(source.coverage.snapshot_date(date), []).append(index)
        quarters = list(quarter_groups)
        torch, nn = _torch_modules()
        _set_seed(torch, self.params["seed"])
        device = _device(torch, self.params["device"])
        self.model = build_analyst_gat(nn, self.input_dim, self.params).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"],
                                     weight_decay=self.params["weight_decay"])
        best_loss, best_state, best_epoch, stale = np.inf, None, 0, 0
        started = perf_counter()
        diagnostics = _diagnostics(context)
        self.loss_history = []
        try:
            for epoch in range(1, self.params["epochs"] + 1):
                self.model.train()
                losses = []
                # Reuse each quarterly membership snapshot across adjacent dates.
                quarter_order = np.random.default_rng(self.params["seed"] + epoch).permutation(len(quarters))
                order = [index for quarter in quarter_order for index in quarter_groups[quarters[quarter]]]
                for index in order:
                    date, rows, y, keep = groups[index]
                    optimizer.zero_grad(set_to_none=True)
                    prediction = self._forward_date(date, rows, source, device)
                    if not bool(torch.isfinite(prediction).all().item()):
                        raise ValueError(f"nonfinite analyst GAT training score on {date}")
                    mask = torch.from_numpy(keep).to(device)
                    target = torch.from_numpy(y[keep]).to(device)
                    loss = (prediction[mask] - target).square().mean()
                    if not bool(torch.isfinite(loss).item()):
                        raise ValueError(f"nonfinite analyst GAT loss on {date}")
                    loss.backward()
                    gradients = [torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None]
                    if not gradients or not bool(torch.stack(gradients).all().item()):
                        raise ValueError(f"nonfinite or absent analyst GAT gradients on {date}")
                    optimizer.step()
                    losses.append(loss.item())
                    del prediction, target, mask, loss
                valid_loss = self._evaluate(valid, context, device)
                improved = valid_loss < best_loss - self.params["min_delta"]
                if improved:
                    best_loss, best_state, best_epoch, stale = valid_loss, _cpu_state_dict(self.model), epoch, 0
                else:
                    stale += 1
                row = dict(window_id=context.artifact_dir.name, epoch=epoch, train_loss=float(np.mean(losses)),
                           valid_loss=valid_loss, best_loss=best_loss, is_best=improved, stale_epochs=stale,
                           elapsed_seconds=perf_counter() - started)
                self.loss_history.append(row)
                if diagnostics["print_epoch"]:
                    print(f"analyst-gat window={context.artifact_dir.name} epoch={epoch} "
                          f"mse={row['train_loss']:.6g} valid_mse={valid_loss:.6g} "
                          f"stale={stale}/{self.params['patience']}", flush=True)
                if stale >= self.params["patience"]:
                    break
            self.model.load_state_dict(best_state)
        finally:
            self.model.cpu().eval()
            del optimizer
            _release_torch_memory(torch)
        self.model_info = dict(
            model_class=self.__class__.__name__, alpha_id=context.alpha_id, window_id=context.artifact_dir.name,
            feature_columns=self.feature_columns, params=self.params, device=str(device),
            train_rows=len(train.frame), valid_rows=len(valid.frame), train_dates=len(groups),
            best_epoch=best_epoch, best_valid_mse=best_loss, epochs_run=len(self.loss_history),
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            graph="binary_shared_analyst_plus_self_loop_complete_covered_cross_section",
            coverage_identity=source.coverage.identity(), target="date_return_zscore", loss="date_mean_mse",
            graph_attention="single_head_additive_leakyrelu_global_neighbor_softmax",
            skip="independent_linear_projection_with_bias", prediction="raw_scalar_no_label_input",
        )

    def predict_bundle(self, bundle, context):
        if self.model is None:
            raise RuntimeError("analyst coverage GAT is not fitted")
        if list(context.feature_columns) != self.feature_columns:
            raise ValueError("analyst GAT prediction feature order changed")
        source = self._source(bundle)
        torch, _ = _torch_modules()
        device = _device(torch, self.params["device"])
        self.model.to(device).eval()
        frame = bundle.frame.reset_index(drop=True)
        scores = np.empty(len(frame), dtype="float32")
        try:
            with torch.no_grad():
                for date, rows in frame.groupby("trade_date", sort=True):
                    prediction = self._forward_date(date, rows, source, device).cpu().numpy()
                    if not np.isfinite(prediction).all():
                        raise ValueError(f"nonfinite analyst GAT prediction on {date}")
                    scores[rows.index.to_numpy()] = prediction
            return pd.Series(scores, index=bundle.frame.index, dtype="float32")
        finally:
            if not getattr(self, "_prediction_active", False):
                self.model.cpu()
                _release_torch_memory(torch)

    def save(self, path):
        if self.model is None:
            raise RuntimeError("cannot save an unfitted analyst coverage GAT")
        torch, _ = _torch_modules()
        path = Path(path)
        artifact_io.mkdir(path.parent, parents=True, exist_ok=True)
        artifact_io.torch_save(dict(version=1, model_kind="analyst_coverage_gat", params=self.params,
                                   feature_columns=self.feature_columns, state_dict=_cpu_state_dict(self.model),
                                   loss_history=self.loss_history, model_info=self.model_info), path)

    @classmethod
    def load(cls, path):
        torch, nn = _torch_modules()
        saved = artifact_io.torch_load(Path(path), map_location="cpu")
        if saved.get("version") != 1 or saved.get("model_kind") != "analyst_coverage_gat":
            raise ValueError("unsupported analyst coverage GAT checkpoint")
        model = cls()
        model.params = _params(saved["params"])
        model.feature_columns = list(saved["feature_columns"])
        model.input_dim = len(model.feature_columns)
        model.model = build_analyst_gat(nn, model.input_dim, model.params)
        model.model.load_state_dict(saved["state_dict"])
        model.model.eval()
        model.loss_history, model.model_info = saved["loss_history"], saved["model_info"]
        return model
