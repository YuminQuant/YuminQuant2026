"""Northeast Securities eLSTM: temporal OHLCV normalization and MSE."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from yq_ml_alpha.models.daily_sequence_model import DailySequenceAlphaModel
from yq_ml_alpha.models.exponential_lstm import eLSTM
from yq_ml_alpha.models.mlp_model import _cpu_state_dict, _device, _set_seed, _torch_modules
from yq_ml_alpha.output import artifact_io


def _params(raw):
    integers = {"sequence_length", "input_size", "hidden_size", "num_layers", "batch_size", "epochs", "patience"}
    scalars = {"dropout", "norm_eps", "eps", "lr", "weight_decay", "min_delta"}
    keys = integers | scalars | {"seed", "device"}
    if set(raw) != keys:
        raise ValueError(f"eLSTM parameters: missing={sorted(keys - set(raw))}, unknown={sorted(set(raw) - keys)}")
    params = dict(raw)
    for key in integers:
        if type(params[key]) is not int or params[key] < 1:
            raise ValueError(f"eLSTM {key} must be a positive integer")
    for key in scalars:
        params[key] = float(params[key])
        if not np.isfinite(params[key]) or params[key] < 0:
            raise ValueError(f"eLSTM {key} must be finite and nonnegative")
    if any(params[k] == 0 for k in ("eps", "norm_eps", "lr")) or params["dropout"] >= 1:
        raise ValueError("eLSTM requires positive numerical scales/lr and dropout in [0,1)")
    if type(params["seed"]) is not int or not isinstance(params["device"], str):
        raise ValueError("eLSTM seed/device must be integer/string")
    return params


def build_network(nn, params):
    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.recurrent = eLSTM(nn, params["input_size"], params["hidden_size"],
                                   num_layers=params["num_layers"], eps=params["eps"])
            self.norm = nn.LayerNorm(params["hidden_size"], eps=params["norm_eps"])
            self.dropout = nn.Dropout(params["dropout"])
            self.head = nn.Linear(params["hidden_size"], 1)
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def forward(self, values):
            _, last = self.recurrent(values)
            return self.head(self.dropout(self.norm(last))).squeeze(-1)

    return Network()


def _labels(frame, column):
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")


class RawBarELSTMAlphaModel(DailySequenceAlphaModel):
    def __init__(self):
        super().__init__()
        self.label_mean = 0.0
        self.label_std = 1.0

    def _source(self, bundle):
        from yq_ml_alpha.features.rawbar_sequence import RawBarHistory
        source = super()._source(bundle)
        if not isinstance(source, RawBarHistory):
            raise ValueError("rawbar eLSTM requires per-stock temporal z-score histories")
        return source

    def _scores(self, torch, bundle, date, frame, device, *, training):
        values = self._inputs(bundle, date, frame)
        return torch.cat([self.model(torch.from_numpy(values[start:start + self.params["batch_size"]]).to(device))
                          for start in range(0, len(values), self.params["batch_size"])])

    def _epoch(self, torch, bundle, context, device, rng, optimizer=None):
        training = optimizer is not None
        self.model.train(training)
        squared_error, count, dates = 0.0, 0, 0
        # Fixed dates[::5] come from the public sampler once per window. Every
        # epoch traverses all of them; only stock minibatch order is shuffled.
        with torch.set_grad_enabled(training):
            for date, frame in bundle.frame.groupby("trade_date", sort=True):
                labels = _labels(frame, context.label_column)
                keep = np.isfinite(labels)
                if not keep.any():
                    continue
                frame = frame.loc[keep]
                values = self._inputs(bundle, date, frame)
                target = ((labels[keep] - self.label_mean) / self.label_std).astype("float32")
                order = rng.permutation(len(frame)) if training else np.arange(len(frame))
                for start in range(0, len(order), self.params["batch_size"]):
                    rows = order[start:start + self.params["batch_size"]]
                    inputs = torch.from_numpy(values[rows]).to(device)
                    truth = torch.from_numpy(target[rows]).to(device)
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                    prediction = self.model(inputs)
                    loss = (prediction - truth).square().mean()
                    if not torch.isfinite(loss):
                        raise ValueError(f"nonfinite eLSTM loss on {date}")
                    if training:
                        loss.backward()
                        optimizer.step()
                    squared_error += float(loss.detach()) * len(rows)
                    count += len(rows)
                dates += 1
        if not count:
            raise ValueError("eLSTM requires finite labeled samples in each train/validation split")
        return squared_error / count, count, dates

    def fit_bundle(self, train, valid, context):
        torch, nn = _torch_modules()
        if train.frame.empty or valid.frame.empty or context.model_search:
            raise ValueError("eLSTM requires separate nonempty train/valid bundles and no model.search")
        if train.label_column != context.label_column or valid.label_column != context.label_column:
            raise ValueError("eLSTM bundle labels must match the configured target")
        if train.frame.trade_date.max() >= valid.frame.trade_date.min():
            raise ValueError("eLSTM training and validation must be chronological and disjoint")
        self.params = _params(context.model_params)
        self.feature_columns = list(context.feature_columns)
        self.input_dim = len(self.feature_columns)
        if self.input_dim != self.params["input_size"]:
            raise ValueError("eLSTM input_size must match rawbar feature columns")
        source = self._source(train)
        self._source(valid)
        labels = _labels(train.frame, context.label_column)
        labels = labels[np.isfinite(labels)]
        if len(labels) < 2:
            raise ValueError("eLSTM needs at least two finite training labels")
        self.label_mean, self.label_std = float(labels.mean()), float(labels.std(ddof=0))
        if not np.isfinite(self.label_std) or self.label_std <= self.params["eps"]:
            raise ValueError("eLSTM training labels have zero or invalid variance")
        _set_seed(torch, self.params["seed"])
        rng = np.random.default_rng(self.params["seed"])
        device = _device(torch, self.params["device"])
        self.model = build_network(nn, self.params).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"], weight_decay=self.params["weight_decay"])
        best, best_state, best_epoch, stale = np.inf, None, 0, 0
        self.loss_history = []
        started = perf_counter()
        for epoch in range(1, self.params["epochs"] + 1):
            epoch_started, before = perf_counter(), source.cache_info()
            train_loss, train_rows, train_dates = self._epoch(torch, train, context, device, rng, optimizer)
            valid_loss, valid_rows, valid_dates = self._epoch(torch, valid, context, device, rng)
            improved = valid_loss < best - self.params["min_delta"]
            if improved:
                best, best_state, best_epoch, stale = valid_loss, _cpu_state_dict(self.model), epoch, 0
            else:
                stale += 1
            self.loss_history.append(dict(window_id=context.artifact_dir.name, epoch=epoch,
                train_loss=train_loss, valid_loss=valid_loss, best_loss=best, is_best=improved, stale_epochs=stale,
                train_rows=train_rows, valid_rows=valid_rows, train_date_count=train_dates, valid_date_count=valid_dates,
                epoch_elapsed_seconds=perf_counter() - epoch_started, elapsed_seconds=perf_counter() - started))
            if context.diagnostics.get("enabled", False) and context.diagnostics.get("print_epoch", False):
                print(f"rawbar-elstm window={context.artifact_dir.name} epoch={epoch} dates={train_dates}/{valid_dates} "
                      f"train_mse={train_loss:.6g} valid_mse={valid_loss:.6g} best={best:.6g} "
                      f"stale={stale}/{self.params['patience']} epoch_elapsed_s={perf_counter() - epoch_started:.2f}", flush=True)
                source.log_cache(f"rawbar_elstm_epoch_{epoch}", before=before)
            if stale >= self.params["patience"]:
                break
        self.model.load_state_dict(best_state)
        self.model.to("cpu").eval()
        self.model_info = dict(window_id=context.artifact_dir.name, params=self.params, feature_columns=self.feature_columns,
            best_epoch=best_epoch, best_valid_mse=best, epochs_run=len(self.loss_history),
            train_rows=train_rows, valid_rows=valid_rows, train_date_count=train_dates, valid_date_count=valid_dates,
            label_mean=self.label_mean, label_std=self.label_std, label_transform="train_fitted_mean_std_ddof0",
            date_selection="all_fixed_sampled_dates_every_epoch", prediction="standardized_return_score_no_labels")

    def save(self, path):
        if self.model is None:
            raise RuntimeError("cannot save an unfitted rawbar eLSTM")
        path = Path(path)
        artifact_io.mkdir(path.parent, parents=True, exist_ok=True)
        artifact_io.torch_save(dict(version=1, model_kind="rawbar_elstm", params=self.params,
            feature_columns=self.feature_columns, label_mean=self.label_mean, label_std=self.label_std,
            state_dict=_cpu_state_dict(self.model), loss_history=self.loss_history, model_info=self.model_info), path)

    @classmethod
    def load(cls, path):
        _, nn = _torch_modules()
        saved = artifact_io.torch_load(Path(path), map_location="cpu")
        if saved.get("version") != 1 or saved.get("model_kind") != "rawbar_elstm":
            raise ValueError("unsupported rawbar eLSTM checkpoint")
        model = cls()
        model.params = _params(saved["params"])
        model.feature_columns = list(saved["feature_columns"])
        model.input_dim = len(model.feature_columns)
        model.label_mean, model.label_std = float(saved["label_mean"]), float(saved["label_std"])
        if (model.input_dim != model.params["input_size"] or not np.isfinite(model.label_mean)
                or not np.isfinite(model.label_std) or model.label_std <= model.params["eps"]):
            raise ValueError("invalid rawbar eLSTM checkpoint dimensions or label statistics")
        model.model = build_network(nn, model.params)
        model.model.load_state_dict(saved["state_dict"])
        model.model.eval()
        model.loss_history = list(saved.get("loss_history", []))
        model.model_info = dict(saved.get("model_info", {}))
        return model
