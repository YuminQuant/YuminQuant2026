"""Risk-factor queries and keys over daily GRU value embeddings."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from yq_ml_alpha.features.risk_attention_sequence import RiskAttentionProvider
from yq_ml_alpha.models.daily_sequence_model import DailySequenceAlphaModel
from yq_ml_alpha.models.torch_support import _cpu_state_dict, _device, _diagnostics, _set_seed, _torch_modules
from yq_ml_alpha.output import artifact_io


def _params(raw):
    params = dict(raw)
    for key in ("sequence_length", "market_width", "risk_width", "gru_width", "gru_layers",
                "attention_width", "head_width", "batch_size", "query_block_size",
                "epochs", "patience", "lr_patience"):
        if type(params[key]) is not int or params[key] < 1:
            raise ValueError(f"risk attention {key} must be a positive integer")
    if params["market_width"] != 5 or params["sequence_length"] != 30:
        raise ValueError("risk attention requires 30-day OHLCV")
    for key in ("lr", "grad_clip", "eps"):
        if not np.isfinite(params[key]) or params[key] <= 0:
            raise ValueError(f"risk attention {key} must be positive and finite")
    if not 0 < params["lr_factor"] < 1 or type(params["gradient_checkpointing"]) is not bool:
        raise ValueError("invalid risk attention learning-rate factor or checkpointing option")
    return params


def _network(torch, nn, params):
    from torch.nn.functional import scaled_dot_product_attention
    from torch.utils.checkpoint import checkpoint

    class RiskAttention(nn.Module):
        def __init__(self):
            super().__init__()
            hidden, width = params["gru_width"], params["attention_width"]
            self.gru = nn.GRU(params["market_width"], hidden, num_layers=params["gru_layers"], batch_first=True)
            self.query = nn.Linear(params["risk_width"], width)
            self.key = nn.Linear(params["risk_width"], width)
            self.value = nn.Linear(hidden, width)
            self.head = nn.Sequential(nn.Linear(hidden + width, params["head_width"]), nn.ReLU(),
                                      nn.Linear(params["head_width"], 1))

        def _encode(self, market):
            return self.gru(market)[1][-1]

        def forward(self, market, risk):
            recompute = self.training and params["gradient_checkpointing"]
            parts = []
            for block in market.split(params["batch_size"]):
                parts.append(checkpoint(self._encode, block, use_reentrant=False) if recompute else self._encode(block))
            hidden = torch.cat(parts, dim=0)
            query = self.query(risk)
            keys = self.key(risk)[None, None]
            values = self.value(hidden)[None, None]

            def aggregate(block, all_keys, all_values):
                return scaled_dot_product_attention(block[None, None], all_keys, all_values)[0, 0]

            messages = []
            for block in query.split(params["query_block_size"]):
                # Every query sees the complete date's keys and values.
                messages.append(checkpoint(aggregate, block, keys, values, use_reentrant=False)
                                if recompute else aggregate(block, keys, values))
            return self.head(torch.cat((hidden, torch.cat(messages, dim=0)), dim=1)).squeeze(1)

    return RiskAttention()


def ccc_loss(predicted, target, eps):
    pred_mean, target_mean = predicted.mean(), target.mean()
    pred_centered, target_centered = predicted - pred_mean, target - target_mean
    covariance = (pred_centered * target_centered).mean()
    denom = pred_centered.square().mean() + target_centered.square().mean() + (pred_mean - target_mean).square()
    return 1 - 2 * covariance / denom.clamp_min(eps)


def _rank_unit(values):
    ranks = pd.Series(values).rank(method="average").to_numpy(dtype=np.float64, copy=True)
    ranks -= ranks.mean()
    norm = np.linalg.norm(ranks)
    return ranks / norm if norm else ranks


class RiskAttentionAlphaModel(DailySequenceAlphaModel):
    @staticmethod
    def make_feature_provider(config):
        if config.features.type != "risk_attention_sequence":
            raise ValueError("risk attention requires risk_attention_sequence inputs")
        return RiskAttentionProvider(config)

    def _source(self, bundle):
        source = super()._source(bundle)
        if len(source.barra_columns) + len(source.industry_codes) != self.params["risk_width"]:
            raise ValueError("risk attention requires the configured Barra and industry width")
        return source

    def _scores(self, torch, bundle, date, frame, device, *, training):
        market, risk = self._source(bundle).inputs(int(date), frame.ts_code.tolist())
        return self.model(torch.from_numpy(market).to(device), torch.from_numpy(risk).to(device))

    def _prepare(self, torch, bundle, label, *, validation):
        source = self._source(bundle)
        sections = []
        for date, frame in bundle.frame.groupby("trade_date", sort=True):
            target = frame[label].to_numpy(dtype=np.float32, na_value=np.nan)
            observed = np.flatnonzero(np.isfinite(target))
            if len(observed) < 2:
                continue
            target = target[observed]
            std = target.std(dtype=np.float64)
            if not np.isfinite(std) or std <= self.params["eps"]:
                continue
            target = ((target - target.mean(dtype=np.float64)) / std).astype(np.float32)
            market, risk = source.inputs(int(date), frame.ts_code.tolist())
            ranks = _rank_unit(target) if validation else None
            sections.append((torch.from_numpy(market), torch.from_numpy(risk),
                             torch.from_numpy(observed), torch.from_numpy(target), ranks))
        if not sections:
            raise ValueError("risk attention split contains no usable labeled cross-sections")
        return sections

    def _epoch(self, torch, sections, device, optimizer=None):
        training = optimizer is not None
        self.model.train(training)
        total = torch.zeros((), device=device)
        rank_ics = []
        with torch.set_grad_enabled(training):
            for market, risk, rows, target, ranks in sections:
                if training:
                    optimizer.zero_grad(set_to_none=True)
                prediction = self.model(market.to(device), risk.to(device))
                prediction = prediction.index_select(0, rows.to(device))
                loss = ccc_loss(prediction, target.to(device), self.params["eps"])
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params["grad_clip"],
                                                   error_if_nonfinite=True)
                    optimizer.step()
                else:
                    values = prediction.detach().cpu().numpy()
                    if not np.isfinite(values).all():
                        raise ValueError("nonfinite risk attention validation scores")
                    rank_ics.append(float(_rank_unit(values) @ ranks))
                total += loss.detach()
        return float(total / len(sections)), float(np.mean(rank_ics)) if rank_ics else None

    def fit_bundle(self, train, valid, context):
        torch, nn = _torch_modules()
        params = self.params = _params(context.model_params)
        self.feature_columns = list(context.feature_columns)
        self.input_dim = len(self.feature_columns)
        if (train.frame.empty or valid.frame.empty or train.frame.trade_date.max() >= valid.frame.trade_date.min()
                or self.input_dim != params["market_width"] or context.model_search):
            raise ValueError("risk attention requires chronological train/validation and fixed OHLCV inputs")
        diagnostics = _diagnostics(context)
        started = perf_counter()
        training = self._prepare(torch, train, context.label_column, validation=False)
        validation = self._prepare(torch, valid, context.label_column, validation=True)
        if diagnostics["enabled"]:
            print(f"risk-attention prepared train_dates={len(training)} valid_dates={len(validation)} "
                  f"elapsed_s={perf_counter() - started:.2f}", flush=True)
        _set_seed(torch, params["seed"])
        device = _device(torch, params["device"])
        self.model = _network(torch, nn, params).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=params["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=params["lr_factor"],
                                                               patience=params["lr_patience"] - 1, threshold=0)
        best, state, best_epoch, stale = -np.inf, None, 0, 0
        self.loss_history = []
        for epoch in range(1, params["epochs"] + 1):
            order = np.random.default_rng(params["seed"] + epoch).permutation(len(training))
            train_loss, _ = self._epoch(torch, [training[i] for i in order], device, optimizer)
            valid_loss, rank_ic = self._epoch(torch, validation, device)
            if not np.isfinite([train_loss, valid_loss, rank_ic]).all():
                raise ValueError(f"nonfinite risk attention metrics at epoch {epoch}")
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step(valid_loss)
            improved = rank_ic > best
            if improved:
                best, state, best_epoch, stale = rank_ic, _cpu_state_dict(self.model), epoch, 0
            else:
                stale += 1
            self.loss_history.append(dict(window_id=context.artifact_dir.name, epoch=epoch, train_loss=train_loss,
                valid_loss=valid_loss, valid_rank_ic=rank_ic, best_rank_ic=best, is_best=improved,
                stale_epochs=stale, learning_rate=lr, elapsed_seconds=perf_counter() - started))
            if diagnostics["print_epoch"]:
                print(f"risk-attention epoch={epoch} train_ccc_loss={train_loss:.6g} "
                      f"valid_ccc_loss={valid_loss:.6g} valid_rank_ic={rank_ic:.6g} "
                      f"lr={lr:.6g} stale={stale}/{params['patience']}", flush=True)
            if stale >= params["patience"]:
                break
        self.model.load_state_dict(state)
        self.model.to("cpu").eval()
        self.model_info = dict(params=params, feature_columns=self.feature_columns, best_epoch=best_epoch,
                               best_valid_rank_ic=best, epochs_run=len(self.loss_history),
                               train_dates=len(training), valid_dates=len(validation), loss="1_minus_ccc",
                               graph="full_cross_section_risk_qk_market_v")

    def save(self, path):
        path = Path(path)
        artifact_io.mkdir(path.parent, parents=True, exist_ok=True)
        artifact_io.torch_save(dict(kind="risk_attention", version=1, params=self.params,
            feature_columns=self.feature_columns, state_dict=_cpu_state_dict(self.model),
            model_info=self.model_info, loss_history=self.loss_history), path)

    @classmethod
    def load(cls, path):
        torch, nn = _torch_modules()
        saved = artifact_io.torch_load(Path(path), map_location="cpu")
        if saved.get("kind") != "risk_attention" or saved.get("version") != 1:
            raise ValueError("unsupported risk attention checkpoint")
        model = cls()
        model.params = _params(saved["params"])
        model.feature_columns = list(saved["feature_columns"])
        model.input_dim = len(model.feature_columns)
        model.model = _network(torch, nn, model.params)
        model.model.load_state_dict(saved["state_dict"])
        model.model.eval()
        model.loss_history, model.model_info = saved["loss_history"], saved["model_info"]
        return model
