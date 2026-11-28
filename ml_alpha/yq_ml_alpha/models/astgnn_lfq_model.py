"""Fundamental ASTGNN with separate stock and risk encoders."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from yq_ml_alpha.features.fundamental_lfq import FundamentalLfqProvider
from yq_ml_alpha.models.daily_sequence_model import DailySequenceAlphaModel
from yq_ml_alpha.models.torch_support import _cpu_state_dict, _device, _set_seed, _torch_modules
from yq_ml_alpha.output import artifact_io


def _params(raw):
    params = dict(raw)
    for key in ("input_size", "sequence_length", "hidden_size", "risk_size", "num_layers",
                "batch_size", "graph_chunk_size", "epochs", "patience", "base_factors"):
        if type(params[key]) is not int or params[key] < 1:
            raise ValueError(f"ASTGNN LFQ {key} must be a positive integer")
    for key in ("lr", "ridge", "grad_clip", "risk_weight", "orthogonal_weight", "alpha_orthogonal_weight"):
        if not np.isfinite(params[key]) or params[key] <= 0:
            raise ValueError(f"ASTGNN LFQ {key} must be positive and finite")
    if not 0 <= params["dropout"] < 1 or not 0 < params["horizon_decay"] <= 1:
        raise ValueError("invalid ASTGNN LFQ dropout or horizon decay")
    return params


def _network(torch, nn, params):
    from torch.utils.checkpoint import checkpoint

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            width = params["hidden_size"]
            gru = dict(input_size=params["input_size"], hidden_size=width, batch_first=True,
                       num_layers=params["num_layers"],
                       dropout=params["dropout"] if params["num_layers"] > 1 else 0.0)
            self.alpha_encoder = nn.GRU(**gru)
            self.risk_encoder = nn.GRU(**gru)
            self.risk_head = nn.Linear(width, params["risk_size"])
            self.value_head = nn.Linear(width, width, bias=False)
            self.factor_head = nn.Linear(width, params["base_factors"])
            self.factor_norm = nn.BatchNorm1d(params["base_factors"])

        def encode(self, host, token):
            values = host.to(token.device)
            _, alpha = self.alpha_encoder(values)
            _, risk = self.risk_encoder(values)
            return alpha[-1] + token * 0, self.risk_head(risk[-1])

        def factors(self, alpha, risk):
            def aggregate(query, keys, values):
                weights = (query @ keys.T).relu().softmax(dim=1)
                return weights @ values

            values = self.value_head(alpha)
            messages = []
            for query in risk.split(params["graph_chunk_size"]):
                if self.training and params["gradient_checkpointing"]:
                    message = checkpoint(aggregate, query, risk, values, use_reentrant=False)
                else:
                    message = aggregate(query, risk, values)
                messages.append(message)
            return self.factor_norm(self.factor_head(values - torch.cat(messages)))

    return Network()


class AstgnnLfqAlphaModel(DailySequenceAlphaModel):
    @staticmethod
    def make_feature_provider(config):
        if config.features.type != "fundamental_lfq":
            raise ValueError("ASTGNN LFQ requires fundamental_lfq features")
        return FundamentalLfqProvider(config)

    def _forward(self, torch, bundle, date, frame, device, *, training):
        from torch.utils.checkpoint import checkpoint

        host = torch.from_numpy(self._inputs(bundle, date, frame))
        token = torch.zeros((), device=device, requires_grad=training)
        encode = self.model.encode
        encoded = []
        for values in host.split(self.params["batch_size"]):
            if training and self.params["gradient_checkpointing"]:
                encoded.append(checkpoint(encode, values, token, use_reentrant=True))
            else:
                encoded.append(encode(values, token))
        alpha, risk = (torch.cat(parts) for parts in zip(*encoded))
        return self.model.factors(alpha, risk), risk

    def _scores(self, torch, bundle, date, frame, device, *, training):
        factors, _ = self._forward(torch, bundle, date, frame, device, training=training)
        return factors.mean(dim=1)

    def predict_components_bundle(self, bundle, context):
        return self._predict_bundle(bundle, with_components=True)

    def _scores_and_components(self, torch, bundle, date, frame, device):
        factors, _ = self._forward(torch, bundle, date, frame, device, training=False)
        return factors.mean(dim=1), factors

    @staticmethod
    def _orthogonal_penalty(values):
        centered = values - values.mean(0)
        unit = centered / centered.norm(dim=0).clamp_min(1e-6)
        corr = unit.T @ unit
        return (corr - corr.diag().diag()).square().mean()

    def _risk_loss(self, torch, risk, targets):
        orthogonal = self._orthogonal_penalty(risk)
        losses, weights = [], []
        # Primary target is last here, giving horizons 1/5/10 in that order.
        for index, target in enumerate(targets):
            observed = torch.isfinite(target)
            y = target[observed]
            if len(y) <= risk.shape[1] or y.std(unbiased=False) <= 1e-8:
                continue
            x = risk[observed]
            x = x - x.mean(0)
            x = x / x.norm(dim=0).clamp_min(1e-6)
            y = y - y.mean()
            # Solve a small K x K ridge system. This equals unexplained
            # variance in the zero-ridge, full-rank limit.
            identity = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
            beta = torch.linalg.solve(x.T @ x + self.params["ridge"] * identity, x.T @ y)
            losses.append((y - x @ beta).square().sum() / y.square().sum())
            weights.append(self.params["horizon_decay"] ** index)
        fit = sum(loss * weight for loss, weight in zip(losses, weights)) / sum(weights) if weights else 0
        return self.params["risk_weight"] * fit + self.params["orthogonal_weight"] * orthogonal

    def _epoch(self, torch, bundle, sections, device, optimizer=None):
        training = optimizer is not None
        self.model.train(training)
        total_alpha, total_loss, count = 0.0, 0.0, 0
        with torch.set_grad_enabled(training):
            for date, frame, targets, observed in sections:
                if training:
                    optimizer.zero_grad(set_to_none=True)
                factors, risk = self._forward(torch, bundle, date, frame, device, training=training)
                mask = torch.from_numpy(observed).to(device)
                primary = torch.from_numpy(targets[0, observed]).to(device)
                alpha_loss = (factors[mask].mean(dim=1) - primary).square().mean()
                loss = alpha_loss
                if training:
                    risk_targets = torch.from_numpy(targets[[1, 2, 0]]).to(device)
                    loss = (loss + self._risk_loss(torch, risk, risk_targets)
                            + self.params["alpha_orthogonal_weight"] * self._orthogonal_penalty(factors))
                if not bool(torch.isfinite(loss)):
                    raise ValueError(f"nonfinite ASTGNN LFQ loss on {date}")
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params["grad_clip"],
                                                   error_if_nonfinite=True)
                    optimizer.step()
                total_alpha += float(alpha_loss.detach())
                total_loss += float(loss.detach())
                count += 1
        return total_alpha / count, total_loss / count

    @staticmethod
    def _sections(bundle, labels):
        sections = []
        for date, frame in bundle.frame.groupby("trade_date", sort=True):
            targets = frame[labels].to_numpy(dtype=np.float32, copy=True).T
            observed = np.isfinite(targets[0])
            primary = targets[0, observed]
            if len(primary) > 1 and primary.std() > 1e-8:
                targets[0, observed] = (primary - primary.mean()) / primary.std()
                sections.append((int(date), frame[["ts_code"]], targets, observed))
        if not sections:
            raise ValueError("ASTGNN LFQ requires usable labeled dates in both splits")
        return sections

    def fit_bundle(self, train, valid, context):
        torch, nn = _torch_modules()
        self.params = _params(context.model_params)
        self.feature_columns = list(context.feature_columns)
        self.input_dim = len(self.feature_columns)
        if self.input_dim != self.params["input_size"] or context.model_search:
            raise ValueError("ASTGNN LFQ feature dimension mismatch or unsupported model search")
        source = self._source(train)
        self._source(valid)
        labels = ["future_vwap_return_10d", "future_vwap_return_1d", "future_vwap_return_5d"]
        if context.label_column != labels[0] or set(context.label_columns) != set(labels):
            raise ValueError("ASTGNN LFQ requires 10-day VWAP and 1/5-day auxiliary labels")
        train_dates, valid_dates = self._sections(train, labels), self._sections(valid, labels)
        if train_dates[-1][0] >= valid_dates[0][0]:
            raise ValueError("ASTGNN LFQ requires chronological train and validation splits")
        _set_seed(torch, self.params["seed"])
        device = _device(torch, self.params["device"])
        self.model = _network(torch, nn, self.params).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"])
        best, state, best_epoch, stale = np.inf, None, 0, 0
        self.loss_history = []
        for epoch in range(1, self.params["epochs"] + 1):
            start = perf_counter()
            before = source.cache_info()
            order = np.random.default_rng(self.params["seed"] + epoch).permutation(len(train_dates))
            train_mse, train_loss = self._epoch(torch, train, [train_dates[i] for i in order], device, optimizer)
            valid_mse, _ = self._epoch(torch, valid, valid_dates, device)
            improved = valid_mse < best
            if improved:
                best, state, best_epoch, stale = valid_mse, _cpu_state_dict(self.model), epoch, 0
            else:
                stale += 1
            self.loss_history.append(dict(window_id=context.artifact_dir.name, epoch=epoch, train_loss=train_loss,
                train_mse=train_mse, valid_loss=valid_mse, best_loss=best, is_best=improved, stale_epochs=stale,
                train_dates=len(train_dates), valid_dates=len(valid_dates), epoch_elapsed_seconds=perf_counter() - start))
            if context.diagnostics.get("enabled") and context.diagnostics.get("print_epoch"):
                print(f"astgnn_lfq epoch={epoch} train_dates={len(train_dates)} train_loss={train_loss:.6g} "
                      f"train_mse={train_mse:.6g} valid_mse={valid_mse:.6g} best={best:.6g} "
                      f"epoch_elapsed_s={perf_counter() - start:.2f}", flush=True)
                source.log_cache(f"astgnn_lfq_epoch_{epoch}", before=before)
            if stale >= self.params["patience"]:
                break
        self.model.load_state_dict(state)
        self.model.to("cpu").eval()
        self.model_info = dict(best_epoch=best_epoch, best_valid_mse=best, epochs_run=len(self.loss_history),
                               params=self.params, features=self.feature_columns, labels=labels,
                               graph="full_date_subtractive", selection="mean_datewise_alpha_mse",
                               aggregation="equal_weight_component_mean")

    def save(self, path):
        path = Path(path)
        artifact_io.mkdir(path.parent, parents=True, exist_ok=True)
        artifact_io.torch_save(dict(model_kind="astgnn_lfq", version=2, params=self.params,
            feature_columns=self.feature_columns, state_dict=_cpu_state_dict(self.model),
            loss_history=self.loss_history, model_info=self.model_info), path)

    @classmethod
    def load(cls, path):
        torch, nn = _torch_modules()
        saved = artifact_io.torch_load(Path(path), map_location="cpu")
        if saved.get("model_kind") != "astgnn_lfq" or saved.get("version") != 2:
            raise ValueError("ASTGNN LFQ requires a 64-component checkpoint; train it with the current config")
        model = cls()
        model.params = _params(saved["params"])
        model.feature_columns = list(saved["feature_columns"])
        model.input_dim = len(model.feature_columns)
        if model.input_dim != model.params["input_size"]:
            raise ValueError("ASTGNN LFQ checkpoint feature dimension mismatch")
        model.model = _network(torch, nn, model.params)
        model.model.load_state_dict(saved["state_dict"])
        model.model.eval()
        model.loss_history, model.model_info = saved["loss_history"], saved["model_info"]
        return model
