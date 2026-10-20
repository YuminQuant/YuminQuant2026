"""Fixed CNN/GRU chip-age factor with whole-date Pearson IC optimization."""
import json
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from yq_ml_alpha.models.base import AlphaModel


def _network(torch, params):
    nn = torch.nn

    class ChipNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(4, 32, 5, padding=2)
            self.conv2 = nn.Conv1d(32, 32, 3, padding=1)
            self.gru = nn.GRU(1024, 64, num_layers=2, batch_first=True,
                              dropout=float(params["dropout"]))
            self.linear = nn.Linear(64, 1)
            self.norm = nn.BatchNorm1d(1)

        def encode(self, x):
            batch, steps, _, _ = x.shape
            z = torch.relu(self.conv1(x.reshape(batch * steps, 4, 32)))
            z = torch.relu(self.conv2(z)).reshape(batch, steps, 1024)
            z, _ = self.gru(z)
            return self.linear(z[:, -1])

    return ChipNetwork()


def _negative_ic(torch, scores, labels, eps):
    x, y = scores.reshape(-1), labels.reshape(-1)
    x, y = x - x.mean(), y - y.mean()
    denominator = x.square().sum().clamp_min(eps).sqrt() * y.square().sum().clamp_min(eps).sqrt()
    return -(x * y).sum() / denominator


def _params(raw):
    defaults = dict(seeds=[42], epochs=100, patience=10, lr=0.001, weight_decay=0.0001,
                    dropout=0.1, device="cuda", micro_batch_size=256, min_stocks=20,
                    ic_eps=1e-8, gradient_clip=1.0, min_improvement=1e-5)
    unknown = set(raw) - set(defaults)
    if unknown:
        raise ValueError(f"unknown chip model parameters: {sorted(unknown)}")
    result = {**defaults, **raw}
    if not result["seeds"] or len(set(result["seeds"])) != len(result["seeds"]):
        raise ValueError("chip seeds must be nonempty and unique")
    for key in ("epochs", "micro_batch_size", "min_stocks"):
        if int(result[key]) < 1:
            raise ValueError(f"{key} must be positive")
    if result["min_stocks"] < 2 or result["patience"] < 1:
        raise ValueError("chip model requires min_stocks >= 2 and patience >= 1")
    return result


class ChipAgeCNNGRUAlphaModel(AlphaModel):
    def __init__(self):
        self.params = {}
        self.states = []
        self.loss_history = []
        self.model_info = {}

    def fit(self, train_data, valid_data, context):
        raise ValueError("ChipAgeCNNGRUAlphaModel requires the lazy tensor bundle interface")

    def predict(self, data, context):
        raise ValueError("ChipAgeCNNGRUAlphaModel requires the lazy tensor bundle interface")

    def _device(self, torch):
        device = self.params["device"]
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("chip_age_alpha requires CUDA; GPU is unavailable")
        return torch.device(device)

    def _scores(self, torch, network, source, date, symbols, device, training):
        from torch.utils.checkpoint import checkpoint
        pieces = []
        size = int(self.params["micro_batch_size"])
        for start in range(0, len(symbols), size):
            array = source.get(date, symbols[start:start + size])
            x = torch.from_numpy(array).to(device)
            if training:
                # Recompute activations in backward; retain all stocks for ONE
                # date-wide loss and ONE BatchNorm update. No local-batch IC.
                x.requires_grad_(True)
                pieces.append(checkpoint(network.encode, x, preserve_rng_state=True))
            else:
                pieces.append(network.encode(x))
        return network.norm(torch.cat(pieces, dim=0)).reshape(-1)

    def _epoch(self, torch, network, bundle, context, device, optimizer=None):
        training = optimizer is not None
        network.train(training)
        losses = []
        skipped = 0
        # Chronological dates permit bounded daily-buffer reuse across samples.
        for date, frame in bundle.frame.groupby("trade_date", sort=True):
            labels = frame[context.label_column].to_numpy(dtype="float32")
            if len(frame) < self.params["min_stocks"] or np.std(labels) <= self.params["ic_eps"]:
                skipped += 1
                continue
            symbols = frame.ts_code.tolist()
            with torch.set_grad_enabled(training):
                if training:
                    optimizer.zero_grad()
                scores = self._scores(torch, network, bundle.tensor_source, int(date), symbols, device, training)
                y = torch.from_numpy(labels).to(device)
                loss = _negative_ic(torch, scores, y, float(self.params["ic_eps"]))
                if not torch.isfinite(loss).item():
                    raise ValueError(f"nonfinite IC loss on {date}")
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), float(self.params["gradient_clip"]))
                    optimizer.step()
                losses.append(float(loss.detach().cpu()))
        if not losses:
            raise ValueError("no valid daily cross sections for IC loss")
        return float(np.mean(losses)), skipped

    def fit_bundle(self, train_bundle, valid_bundle, context):
        import torch
        if train_bundle.frame.empty or valid_bundle.frame.empty:
            raise ValueError("chip fixed training requires nonempty training and validation sets")
        if train_bundle.tensor_source is None or valid_bundle.tensor_source is None:
            raise ValueError("missing chip tensor source")
        self.params = _params(context.model_params)
        self.states, self.loss_history = [], []
        device = self._device(torch)
        started = perf_counter()
        seed_info = []
        for seed in self.params["seeds"]:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            network = _network(torch, self.params).to(device)
            optimizer = torch.optim.AdamW(network.parameters(), lr=float(self.params["lr"]),
                                          weight_decay=float(self.params["weight_decay"]))
            best, stale, best_state, best_epoch = float("inf"), 0, None, 0
            for epoch in range(1, int(self.params["epochs"]) + 1):
                train_loss, train_skipped = self._epoch(torch, network, train_bundle, context, device, optimizer)
                valid_loss, valid_skipped = self._epoch(torch, network, valid_bundle, context, device)
                improved = valid_loss < best - float(self.params["min_improvement"])
                if improved:
                    best, stale, best_epoch = valid_loss, 0, epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
                else:
                    stale += 1
                row = dict(window_id=context.artifact_dir.name, seed=seed, epoch=epoch,
                           train_loss=train_loss, valid_loss=valid_loss, best_loss=best,
                           is_best=improved, stale_epochs=stale, train_skipped_dates=train_skipped,
                           valid_skipped_dates=valid_skipped, elapsed_seconds=perf_counter() - started)
                self.loss_history.append(row)
                if context.diagnostics.get("print_epoch", True):
                    print(f"chip-age seed={seed} epoch={epoch} train_ic={-train_loss:.6f} "
                          f"valid_ic={-valid_loss:.6f} patience={stale}/{self.params['patience']}", flush=True)
                if stale >= self.params["patience"]:
                    break
            self.states.append(best_state)
            seed_info.append(dict(seed=seed, best_epoch=best_epoch, best_valid_ic=-best))
            del network, optimizer
            if device.type == "cuda":
                torch.cuda.empty_cache()
        self.model_info = dict(window_id=context.artifact_dir.name, seeds=seed_info,
                               train_rows=len(train_bundle.frame), valid_rows=len(valid_bundle.frame),
                               params=self.params, loss="negative_mean_datewise_pearson_ic",
                               ensemble="arithmetic_mean_of_eval_scores", input_shape=[30, 4, 32])

    def predict_bundle(self, bundle, context):
        import torch
        if not self.states:
            raise RuntimeError("chip model has not been fitted")
        device = self._device(torch)
        output = np.zeros(len(bundle.frame), dtype="float32")
        network = getattr(self, "_prediction_network", None)
        if network is None:
            network = _network(torch, self.params).to(device)
            if getattr(self, "_prediction_active", False):
                self._prediction_network = network
        for state_index, state in enumerate(self.states):
            if not getattr(self, "_prediction_active", False) or getattr(self, "_prediction_state_index", None) != state_index:
                network.load_state_dict(state)
                if getattr(self, "_prediction_active", False):
                    self._prediction_state_index = state_index
            network.eval()
            with torch.no_grad():
                for date, indices in bundle.frame.groupby("trade_date", sort=True).indices.items():
                    frame = bundle.frame.iloc[indices]
                    scores = self._scores(torch, network, bundle.tensor_source, int(date), frame.ts_code.tolist(), device, False)
                    output[indices] += scores.cpu().numpy() / len(self.states)
        return pd.Series(output, index=bundle.frame.index, dtype="float32")

    def save(self, path):
        import torch
        if not self.states:
            raise RuntimeError("cannot save unfitted chip model")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(params=self.params, states=self.states, loss_history=self.loss_history,
                        model_info=self.model_info), path)

    @classmethod
    def load(cls, path):
        import torch
        try:
            saved = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            saved = torch.load(path, map_location="cpu")
        model = cls()
        model.params = _params(saved["params"])
        model.states = saved["states"]
        model.loss_history = saved["loss_history"]
        model.model_info = saved["model_info"]
        return model

    def write_diagnostics(self, context):
        paths = []
        context.artifact_dir.mkdir(parents=True, exist_ok=True)
        if context.diagnostics.get("write_loss_history", False):
            path = context.artifact_dir / "loss_history.parquet"
            pd.DataFrame(self.loss_history).to_parquet(path, index=False)
            paths.append(path)
        if context.diagnostics.get("write_model_info", False):
            path = context.artifact_dir / "model_info.json"
            path.write_text(json.dumps(self.model_info, indent=2), encoding="utf-8")
            paths.append(path)
        return paths
