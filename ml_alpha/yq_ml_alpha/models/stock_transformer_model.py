"""Huatai AI69: full-date stock-token Transformer with 10/20-day heads."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from yq_ml_alpha.features.transforms import _zscore_series
from yq_ml_alpha.models.adaptive_gate_model import _rank_weights
from yq_ml_alpha.models.bar_gru_model import _cpu_state_dict, _device, _release_torch_memory, _set_seed, _torch_modules
from yq_ml_alpha.models.mlp_model import MLPAlphaModel, _diagnostics
from yq_ml_alpha.models.multi_task_model import _dwa_weights, _label_columns, _params as _multi_task_params
from yq_ml_alpha.models.stock_transformer_network import build_stock_transformer
from yq_ml_alpha.output import artifact_io


def _params(raw):
    common = dict(raw)
    network = dict(sequence_length=1, num_layers=2, attention_heads=2, feedforward_size=64,
                   dropout=0.1, attention_chunk_size=512, gradient_checkpointing=True)
    for key in network:
        network[key] = common.pop(key, network[key])
    params = {**_multi_task_params(common), **network}
    if params["weighting"] != "dwa":
        raise ValueError("stock Transformer uses DWA task weights")
    for key in ("sequence_length", "num_layers", "attention_heads", "feedforward_size", "attention_chunk_size"):
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"stock Transformer {key} must be a positive integer")
    if params["sequence_length"] != 1:
        raise ValueError("stock Transformer uses a single current-date snapshot")
    if params["batch_size"] > 3000 or params["attention_chunk_size"] > params["batch_size"]:
        raise ValueError("stock Transformer query chunks must not exceed stock chunks, capped at 3000")
    if not isinstance(params["gradient_checkpointing"], bool):
        raise ValueError("gradient_checkpointing must be boolean")
    dropout = params["dropout"]
    if isinstance(dropout, bool) or not np.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("stock Transformer dropout must be finite and in [0,1)")
    params["dropout"] = float(dropout)
    return params


def _supervision(bundle, labels, params):
    """Mask supervision without dropping attention nodes with unknown returns."""
    missing = set(labels).difference(bundle.frame.columns)
    if missing:
        raise ValueError(f"missing stock Transformer label columns: {sorted(missing)}")
    groups, skipped = [], 0
    weight_params = dict(loss="rank_exp_mse", rank_weight_half_life=params["rank_weight_half_life"])
    for date, rows in bundle.frame.groupby("trade_date", sort=True):
        raw = rows[labels].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
        selected = np.flatnonzero(np.isfinite(raw).all(axis=1))
        if len(selected) < 2 or (np.ptp(raw[selected], axis=0) <= 0).any():
            skipped += 1
            continue
        target = np.empty((len(selected), len(labels)), dtype="float32")
        weights = np.empty_like(target)
        for task in range(len(labels)):
            values = raw[selected, task]
            target[:, task] = _zscore_series(pd.Series(values).rank(method="average")).to_numpy(dtype="float32")
            weights[:, task] = _rank_weights(values, weight_params)
        groups.append((int(date), rows, selected, target, weights))
    return groups, skipped


class StockTransformerAlphaModel(MLPAlphaModel):
    prediction_replay_safe = True
    prediction_inputs_readonly = True

    def __init__(self):
        super().__init__()
        self.feature_columns = []
        self.label_columns = []

    def fit(self, train_data, valid_data, context):
        raise ValueError("stock Transformer requires factor_sequence and complete-date fit_bundle")

    def predict(self, data, context):
        raise ValueError("stock Transformer requires complete-date predict_bundle")

    def _source(self, bundle):
        source = bundle.tensor_source
        if (source is None or source.length != 1 or source.columns != self.feature_columns
                or list(bundle.feature_columns) != self.feature_columns
                or source.history_transform != "mad5_zscore_rank_label"):
            raise ValueError("stock Transformer requires matching single-date MAD5/zscore factor snapshots")
        return source

    def _forward_date(self, date, rows, source, device):
        torch, _ = _torch_modules()
        symbols = rows.ts_code.tolist()
        expected = source.symbols(int(date))
        if len(symbols) != len(set(symbols)) or len(symbols) != len(expected) or set(symbols) != set(expected):
            raise ValueError(f"stock Transformer requires the complete eligible cross-section on {date}")
        values = np.ascontiguousarray(source.get(int(date), symbols)[:, 0])
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite stock Transformer inputs on {date}")
        return self.model(torch.from_numpy(values), device)

    def _task_losses(self, group, source, device):
        torch, _ = _torch_modules()
        date, rows, selected, targets, weights = group
        prediction = self._forward_date(date, rows, source, device)
        if prediction.shape != (len(rows), len(self.label_columns)) or not bool(torch.isfinite(prediction).all().item()):
            raise ValueError(f"invalid stock Transformer predictions on {date}")
        selected = torch.from_numpy(selected).to(device)
        target = torch.from_numpy(targets).to(device)
        weight = torch.from_numpy(weights).to(device)
        return (weight * (prediction[selected] - target).square()).mean(dim=0)

    def _evaluate(self, groups, source, device):
        torch, _ = _torch_modules()
        self.model.eval()
        totals = []
        with torch.no_grad():
            for group in groups:
                losses = self._task_losses(group, source, device).cpu().numpy()
                if not np.isfinite(losses).all():
                    raise ValueError(f"nonfinite stock Transformer validation loss on {group[0]}")
                totals.append(losses)
        return np.mean(totals, axis=0)

    def fit_bundle(self, train, valid, context):
        if context.model_search:
            raise ValueError("stock Transformer does not use model.search")
        self.params = _params(context.model_params)
        self.feature_columns = list(context.feature_columns)
        self.label_columns = _label_columns(context)
        self.input_dim = len(self.feature_columns)
        train_source, valid_source = self._source(train), self._source(valid)
        if not self.input_dim or train.frame.empty or valid.frame.empty:
            raise ValueError("stock Transformer needs factors and nonempty chronological partitions")
        if train.label_column != context.label_column or valid.label_column != context.label_column:
            raise ValueError("stock Transformer bundle labels must match the model context")
        if train.frame.trade_date.max() >= valid.frame.trade_date.min():
            raise ValueError("stock Transformer validation must follow training dates")
        train_groups, train_skipped = _supervision(train, self.label_columns, self.params)
        valid_groups, valid_skipped = _supervision(valid, self.label_columns, self.params)
        if not train_groups or not valid_groups:
            raise ValueError("stock Transformer needs dates with usable nonconstant targets for all tasks")
        torch, nn = _torch_modules()
        _set_seed(torch, self.params["seed"])
        device = _device(torch, self.params["device"])
        task_count = len(self.label_columns)
        self.model = build_stock_transformer(nn, self.input_dim, task_count, self.params).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"],
                                     weight_decay=self.params["weight_decay"])
        diagnostics = _diagnostics(context)
        started = perf_counter()
        history = []
        best_loss, best_state, best_epoch, stale = np.inf, None, 0, 0
        self.loss_history = []
        try:
            for epoch in range(1, self.params["epochs"] + 1):
                self.model.train()
                dwa = _dwa_weights(history, task_count, self.params["dwa_temperature"])
                coefficients = torch.from_numpy(dwa).to(device)
                totals = np.zeros(task_count, dtype="float64")
                objectives = []
                order = np.random.default_rng(self.params["seed"] + epoch).permutation(len(train_groups))
                for index in order:
                    group = train_groups[index]
                    optimizer.zero_grad(set_to_none=True)
                    task_losses = self._task_losses(group, train_source, device)
                    objective = (coefficients * task_losses).sum()
                    if not bool(torch.isfinite(objective).item()):
                        raise ValueError(f"nonfinite stock Transformer objective on {group[0]}")
                    objective.backward()
                    gradients = [torch.isfinite(parameter.grad).all() for parameter in self.model.parameters()
                                 if parameter.grad is not None]
                    if not gradients or not bool(torch.stack(gradients).all().item()):
                        raise ValueError(f"nonfinite or absent stock Transformer gradients on {group[0]}")
                    optimizer.step()
                    totals += task_losses.detach().cpu().numpy()
                    objectives.append(objective.item())
                    del task_losses, objective
                train_tasks = totals / len(train_groups)
                history.append(train_tasks.copy())
                valid_tasks = self._evaluate(valid_groups, valid_source, device)
                score = float(valid_tasks.mean())
                improved = score < best_loss - self.params["min_delta"]
                if improved:
                    best_loss, best_state, best_epoch, stale = score, _cpu_state_dict(self.model), epoch, 0
                else:
                    stale += 1
                row = dict(window_id=context.artifact_dir.name, epoch=epoch,
                           train_loss=float(np.mean(objectives)), valid_loss=score,
                           best_loss=best_loss, is_best=improved, stale_epochs=stale,
                           elapsed_seconds=perf_counter() - started)
                for task, label in enumerate(self.label_columns):
                    row[f"train_wmse_{label}"] = float(train_tasks[task])
                    row[f"valid_wmse_{label}"] = float(valid_tasks[task])
                    row[f"end_task_weight_{label}"] = float(dwa[task])
                self.loss_history.append(row)
                if diagnostics["print_epoch"]:
                    print(f"stock-transformer window={context.artifact_dir.name} epoch={epoch} "
                          f"train_loss={row['train_loss']:.6g} valid_wmse={score:.6g} "
                          f"task_weights={dwa.tolist()} stale={stale}/{self.params['patience']}", flush=True)
                if stale >= self.params["patience"]:
                    break
            self.model.load_state_dict(best_state)
        finally:
            self.model.cpu().eval()
            del optimizer
            _release_torch_memory(torch)
        self.model_info = dict(
            model_class=self.__class__.__name__, alpha_id=context.alpha_id, window_id=context.artifact_dir.name,
            params=self.params, feature_columns=self.feature_columns, label_columns=self.label_columns,
            train_rows=len(train.frame), valid_rows=len(valid.frame), train_dates=len(train_groups),
            valid_dates=len(valid_groups), train_supervised_rows=sum(len(group[2]) for group in train_groups),
            valid_supervised_rows=sum(len(group[2]) for group in valid_groups),
            train_skipped_dates=train_skipped, valid_skipped_dates=valid_skipped,
            best_epoch=best_epoch, best_valid_wmse=best_loss, epochs_run=len(self.loss_history), device=str(device),
            model_dimension=self.model.model_dim, zero_padding_features=self.model.padding_features,
            parameter_count=sum(parameter.numel() for parameter in self.model.parameters()),
            attention="full_current_date_stock_tokens_no_position_encoding",
            missing_labels="retain_attention_nodes_mask_only_supervision_joint_labels",
            target="date_rank_zscore", sample_weights="return_rank_exponential_half_life",
            training_batch="one_complete_date_per_update_chunked_queries_global_keys_values",
            task_weights="DWA_previous_two_epoch_losses_initial_two_epochs_equal",
            validation="date_mean_task_mean_wmse_without_DWA_coefficients",
            prediction="arithmetic_mean_of_task_heads",
        )

    def predict_bundle(self, bundle, context):
        if self.model is None:
            raise RuntimeError("stock Transformer is not fitted")
        if list(context.feature_columns) != self.feature_columns or _label_columns(context) != self.label_columns:
            raise ValueError("stock Transformer prediction feature/label order changed")
        source = self._source(bundle)
        torch, _ = _torch_modules()
        device = _device(torch, self.params["device"])
        self.model.to(device).eval()
        frame = bundle.frame.reset_index(drop=True)
        scores = np.empty(len(frame), dtype="float32")
        try:
            with torch.no_grad():
                for date, indices in frame.groupby("trade_date", sort=True).indices.items():
                    prediction = self._forward_date(date, frame.iloc[indices], source, device)
                    scores[indices] = prediction.mean(dim=1).cpu().numpy()
            if not np.isfinite(scores).all():
                raise ValueError("nonfinite stock Transformer scores")
            return pd.Series(scores, index=bundle.frame.index)
        finally:
            if not getattr(self, "_prediction_active", False):
                self.model.cpu()
                _release_torch_memory(torch)

    def save(self, path):
        if self.model is None:
            raise RuntimeError("cannot save an unfitted stock Transformer")
        path = Path(path)
        artifact_io.mkdir(path.parent, parents=True, exist_ok=True)
        artifact_io.torch_save(dict(version=1, model_kind="huatai_stock_transformer", params=self.params,
                                   feature_columns=self.feature_columns, label_columns=self.label_columns,
                                   state_dict=_cpu_state_dict(self.model), loss_history=self.loss_history,
                                   model_info=self.model_info), path)

    @classmethod
    def load(cls, path):
        torch, nn = _torch_modules()
        saved = artifact_io.torch_load(Path(path), map_location="cpu")
        if saved.get("version") != 1 or saved.get("model_kind") != "huatai_stock_transformer":
            raise ValueError("unsupported stock Transformer checkpoint")
        model = cls()
        model.params = _params(saved["params"])
        model.feature_columns = list(saved["feature_columns"])
        model.label_columns = list(saved["label_columns"])
        model.input_dim = len(model.feature_columns)
        model.model = build_stock_transformer(nn, model.input_dim, len(model.label_columns), model.params)
        model.model.load_state_dict(saved["state_dict"])
        model.model.eval()
        model.loss_history, model.model_info = saved["loss_history"], saved["model_info"]
        return model
