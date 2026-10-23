"""DFQ-TimesNet: fixed 5/60-session periods and equally averaged 2D convolutions."""
from __future__ import annotations

from yq_ml_alpha.output import artifact_io

from collections import deque
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from yq_ml_alpha.features.transforms import _zscore_series
from yq_ml_alpha.models.bar_gru_model import _release_torch_memory
from yq_ml_alpha.models.mlp_model import MLPAlphaModel, _cpu_state_dict, _device, _set_seed, _torch_modules


def _params(raw):
    params = dict(sequence_length=60, hidden_size=128, feedforward_size=128,
                  num_blocks=1, periods=[5, 60], num_kernels=3, epochs=200,
                  patience=20, average_epochs=5, dates_per_epoch=100, batch_size=3000,
                  lr=0.00009, weight_decay=0.0, min_delta=1e-8, seed=42, device="auto")
    unknown = set(raw).difference(params)
    if unknown:
        raise ValueError(f"unknown TimesNet parameters: {sorted(unknown)}")
    params.update(raw)
    for key in ("sequence_length", "hidden_size", "feedforward_size", "num_blocks", "num_kernels",
                "epochs", "patience", "average_epochs", "dates_per_epoch", "batch_size"):
        value = params[key]
        params[key] = int(value)
        if params[key] <= 0 or params[key] != value:
            raise ValueError(f"{key} must be a positive integer")
    periods = params["periods"]
    if not isinstance(periods, (list, tuple)) or not periods:
        raise ValueError("periods must be a nonempty sequence")
    if any(int(p) != p or not 1 <= int(p) <= params["sequence_length"] for p in periods):
        raise ValueError("periods must be integer session counts within sequence_length")
    params["periods"] = [int(p) for p in periods]
    if len(set(params["periods"])) != len(params["periods"]):
        raise ValueError("periods must be unique")
    for key in ("lr", "weight_decay", "min_delta"):
        params[key] = float(params[key])
        if not np.isfinite(params[key]) or params[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if params["lr"] == 0:
        raise ValueError("lr must be positive")
    params["seed"] = int(params["seed"])
    return params


def build_timesnet(nn, input_size, params):
    import torch.nn.functional as functional

    class Inception(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.convs = nn.ModuleList([
                nn.Conv2d(in_channels, out_channels, 2 * i + 1, padding=i)
                for i in range(params["num_kernels"])
            ])

        def forward(self, x):
            # Accumulate rather than stack all kernel outputs in an extra dimension.
            result = self.convs[0](x)
            for conv in self.convs[1:]:
                result = result + conv(x)
            return result / len(self.convs)

    class TimesBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.convolution = nn.Sequential(
                Inception(params["hidden_size"], params["feedforward_size"]), nn.GELU(),
                Inception(params["feedforward_size"], params["hidden_size"]),
            )

        def forward(self, x):
            batch, length, channels = x.shape
            merged = None
            for period in params["periods"]:
                padding = (-length) % period
                padded = functional.pad(x, (0, 0, 0, padding)) if padding else x
                image = padded.reshape(batch, -1, period, channels).permute(0, 3, 1, 2).contiguous()
                # Same convolution weights for every period in this TimesBlock.
                image = self.convolution(image)
                sequence = image.permute(0, 2, 3, 1).reshape(batch, length + padding, channels)[:, :length]
                merged = sequence if merged is None else merged + sequence
            return x + merged / len(params["periods"])

    class TimesNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.token = nn.Conv1d(input_size, params["hidden_size"], 3, padding=1, bias=False)
            self.blocks = nn.Sequential(*[TimesBlock() for _ in range(params["num_blocks"])])
            self.head = nn.Linear(params["hidden_size"], 1)
            for module in self.modules():
                if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def forward(self, x):
            if x.ndim != 3 or tuple(x.shape[1:]) != (params["sequence_length"], input_size):
                raise ValueError("TimesNet expects [stocks, configured sessions, input factors]")
            encoded = self.token(x.transpose(1, 2)).transpose(1, 2)
            return self.head(self.blocks(encoded)[:, -1]).squeeze(-1)

    return TimesNet()


def _target(rows, column):
    raw = pd.to_numeric(rows[column], errors="coerce")
    return _zscore_series(raw).to_numpy(dtype=np.float32)


class TimesNetAlphaModel(MLPAlphaModel):
    def __init__(self):
        super().__init__()
        self.feature_columns = []

    def fit(self, train_data, valid_data, context):
        raise ValueError("TimesNet requires features.type=factor_sequence and fit_bundle")

    def predict(self, data, context):
        raise ValueError("TimesNet requires predict_bundle with factor histories")

    def _source(self, bundle):
        source = bundle.tensor_source
        if (source is None or list(bundle.feature_columns) != self.feature_columns
                or source.columns != self.feature_columns or source.length != self.params["sequence_length"]
                or getattr(source, "history_transform", None) != "zscore_clip3"):
            raise ValueError("TimesNet requires matching factor histories with history_transform=zscore_clip3")
        return source

    def _scores(self, date, rows, source, device):
        torch, _ = _torch_modules()
        symbols = rows.ts_code.tolist()
        result = np.empty(len(rows), dtype=np.float32)
        for start in range(0, len(rows), self.params["batch_size"]):
            selected = symbols[start:start + self.params["batch_size"]]
            x = torch.from_numpy(source.get(int(date), selected)).to(device)
            result[start:start + len(selected)] = self.model(x).cpu().numpy()
            del x
        if not np.isfinite(result).all():
            raise ValueError(f"nonfinite TimesNet scores on {date}")
        return result

    def _evaluate(self, bundle, device):
        torch, _ = _torch_modules()
        source = self._source(bundle)
        self.model.eval()
        metrics = []
        with torch.no_grad():
            for date, rows in bundle.frame.groupby("trade_date", sort=True):
                target = _target(rows, bundle.label_column)
                keep = np.isfinite(target)
                if keep.sum() < 2 or target[keep].std() <= 1e-8:
                    continue
                score = self._scores(date, rows.loc[keep], source, device)
                metrics.append(float(np.corrcoef(score, target[keep])[0, 1]) if score.std() > 1e-8 else 0.0)
        if not metrics:
            raise ValueError("TimesNet validation has no usable labeled cross-sections")
        return float(np.mean(metrics))

    def fit_bundle(self, train, valid, context):
        self.params = _params(context.model_params)
        if context.model_search:
            raise ValueError("TimesNet does not use model.search")
        self.feature_columns = list(context.feature_columns)
        self.input_dim = len(self.feature_columns)
        source = self._source(train)
        self._source(valid)
        if not self.input_dim or train.frame.empty or valid.frame.empty:
            raise ValueError("TimesNet requires factors and separate nonempty train/validation partitions")
        if train.label_column != context.label_column or valid.label_column != context.label_column:
            raise ValueError("TimesNet label columns must match context")
        if train.frame.trade_date.max() >= valid.frame.trade_date.min():
            raise ValueError("TimesNet train/validation dates must be chronological and disjoint")
        # Precompute date-local labels, not expanded feature sequence tensors.
        groups = []
        for date, rows in train.frame.groupby("trade_date", sort=True):
            y = _target(rows, context.label_column)
            keep = np.isfinite(y)
            if keep.sum() >= 2 and y[keep].std() > 1e-8:
                groups.append((int(date), rows.ts_code.to_numpy()[keep].tolist(), y[keep]))
        if not groups:
            raise ValueError("TimesNet training has no usable labeled cross-sections")
        torch, nn = _torch_modules()
        _set_seed(torch, self.params["seed"])
        device = _device(torch, self.params["device"])
        self.model = build_timesnet(nn, self.input_dim, self.params).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"],
                                     weight_decay=self.params["weight_decay"])
        rng = np.random.RandomState(self.params["seed"])
        recent = deque(maxlen=self.params["average_epochs"])
        best, best_state, stale, best_epoch = -np.inf, None, 0, 0
        self.loss_history = []
        started = perf_counter()
        try:
            for epoch in range(1, self.params["epochs"] + 1):
                self.model.train()
                selected = np.sort(rng.choice(len(groups), min(len(groups), self.params["dates_per_epoch"]), replace=False))
                total = 0.0
                for index in selected:
                    date, symbols, target = groups[index]
                    optimizer.zero_grad(set_to_none=True)
                    for start in range(0, len(symbols), self.params["batch_size"]):
                        codes = symbols[start:start + self.params["batch_size"]]
                        x = torch.from_numpy(source.get(date, codes)).to(device)
                        y = torch.from_numpy(target[start:start + len(codes)]).to(device)
                        # Sum/N gives the complete-date MSE gradient, including a short final chunk.
                        loss = (self.model(x) - y).square().sum() / len(symbols)
                        if not bool(torch.isfinite(loss).item()):
                            raise ValueError(f"nonfinite TimesNet loss on {date}")
                        loss.backward()
                        total += loss.item()
                        del x, y, loss
                    finite_gradients = torch.stack([torch.isfinite(p.grad).all()
                                                    for p in self.model.parameters() if p.grad is not None])
                    if not bool(finite_gradients.all().item()):
                        raise ValueError(f"nonfinite TimesNet gradient on {date}")
                    optimizer.step()
                live = _cpu_state_dict(self.model)
                recent.append(live)
                averaged = {name: sum(state[name] for state in recent) / len(recent) for name in live}
                self.model.load_state_dict(averaged)
                try:
                    metric = self._evaluate(valid, device)
                finally:
                    self.model.load_state_dict(live)
                if not np.isfinite(metric):
                    raise ValueError("nonfinite TimesNet validation IC")
                improved = metric > best + self.params["min_delta"]
                if improved:
                    best, best_state, stale, best_epoch = metric, averaged, 0, epoch
                else:
                    stale += 1
                self.loss_history.append(dict(
                    window_id=context.artifact_dir.name, epoch=epoch, train_loss=total / len(selected),
                    valid_ic=metric, best_ic=best, is_best=improved, stale_epochs=stale,
                    sampled_dates=len(selected), averaged_epochs=len(recent),
                    elapsed_seconds=perf_counter() - started,
                ))
                if context.diagnostics.get("print_epoch", False):
                    print(f"timesnet window={context.artifact_dir.name} epoch={epoch} "
                          f"mse={total / len(selected):.6g} valid_ic={metric:.6g} "
                          f"dates={len(selected)} average={len(recent)} stale={stale}", flush=True)
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
            best_epoch=best_epoch, best_valid_ic=best, epochs_run=len(self.loss_history),
            train_rows=len(train.frame), valid_rows=len(valid.frame), train_usable_dates=len(groups),
            loss="date_mean_mse_zscore_returns", input_transform="daily_zscore_clip3_fill0",
            prediction="last_step_scalar_no_labels", periods="fixed_equal_weight_shared_convolution",
            checkpoint="best_validation_trailing_epoch_parameter_mean",
        )

    def predict_bundle(self, bundle, context):
        if self.model is None:
            raise RuntimeError("TimesNet is not fitted")
        if list(context.feature_columns) != self.feature_columns:
            raise ValueError("TimesNet prediction feature order changed")
        torch, _ = _torch_modules()
        source = self._source(bundle)
        device = _device(torch, self.params["device"])
        self.model.to(device).eval()
        frame = bundle.frame.reset_index(drop=True)
        result = np.empty(len(frame), dtype=np.float32)
        try:
            with torch.no_grad():
                for date, rows in frame.groupby("trade_date", sort=True):
                    result[rows.index.to_numpy()] = self._scores(date, rows, source, device)
            return pd.Series(result, index=bundle.frame.index)
        finally:
            if not getattr(self, "_prediction_active", False):
                self.model.cpu()
                _release_torch_memory(torch)

    def save(self, path):
        if self.model is None:
            raise RuntimeError("cannot save an unfitted TimesNet")
        torch, _ = _torch_modules()
        path = Path(path)
        artifact_io.mkdir(path.parent, parents=True, exist_ok=True)
        artifact_io.torch_save(dict(version=1, model_kind="dfq_timesnet", params=self.params,
                        feature_columns=self.feature_columns, state_dict=_cpu_state_dict(self.model),
                        loss_history=self.loss_history, model_info=self.model_info), path)

    @classmethod
    def load(cls, path):
        torch, nn = _torch_modules()
        saved = artifact_io.torch_load(path, map_location="cpu")
        if saved.get("version") != 1 or saved.get("model_kind") != "dfq_timesnet":
            raise ValueError("unsupported TimesNet checkpoint")
        model = cls()
        model.params = _params(saved["params"])
        model.feature_columns = list(saved["feature_columns"])
        model.input_dim = len(model.feature_columns)
        model.model = build_timesnet(nn, model.input_dim, model.params)
        model.model.load_state_dict(saved["state_dict"])
        model.model.eval()
        model.loss_history, model.model_info = saved["loss_history"], saved["model_info"]
        return model
