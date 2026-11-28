"""ADWM Model3: generate weak alphas and learn stock-level weights in one model."""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from yq_ml_alpha.data.sampler import sample_dates
from yq_ml_alpha.features.fundamental_lfq import FundamentalStyleProvider
from yq_ml_alpha.features.industry_membership import IndustryMembership
from yq_ml_alpha.features.neutralization import industry_size_residual
from yq_ml_alpha.models.daily_sequence_model import DailySequenceAlphaModel
from yq_ml_alpha.models.torch_support import _cpu_state_dict, _device, _diagnostics, _set_seed, _torch_modules
from yq_ml_alpha.output import artifact_io


def _params(raw):
    params = dict(raw)
    for key in ("input_size", "sequence_length", "hidden_size", "gru_layers", "base_factors", "risk_size",
                "batch_size", "graph_block_size", "rank_block_size", "top_k", "abcm_months", "gate_months",
                "epochs", "patience"):
        if type(params[key]) is not int or params[key] < 1:
            raise ValueError(f"ADWM {key} must be a positive integer")
    if params["top_k"] > params["base_factors"]:
        raise ValueError("ADWM top_k exceeds the number of weak factors")
    for key in ("lr", "gate_lr", "grad_clip", "ridge", "eps", "risk_weight", "orthogonal_weight",
                "rank_weight", "router_z_weight"):
        if not np.isfinite(params[key]) or params[key] <= 0:
            raise ValueError(f"ADWM {key} must be positive and finite")
    if type(params["gradient_checkpointing"]) is not bool:
        raise ValueError("ADWM gradient_checkpointing must be boolean")
    if params["label_neutralize"] != "citic_l1+barra:SIZE":
        raise ValueError("ADWM gate labels require CITIC level-one plus Barra SIZE neutralization")
    return params


def _boundaries(start, params):
    origin = pd.Timestamp(str(start))
    return tuple(int((origin + pd.DateOffset(months=months)).strftime("%Y%m%d"))
                 for months in (params["abcm_months"], params["abcm_months"] + params["gate_months"]))


def _network(torch, nn, params):
    import torch.nn.functional as functional
    from torch.utils.checkpoint import checkpoint

    class ABCM(nn.Module):
        def __init__(self):
            super().__init__()
            hidden, risk = params["hidden_size"], params["risk_size"]
            self.gru = nn.GRU(params["input_size"], hidden, params["gru_layers"], batch_first=True)
            self.alpha = nn.Linear(hidden, params["base_factors"])
            self.risk_input = nn.Linear(hidden, risk)
            self.query = nn.Linear(risk, risk)
            self.key = nn.Linear(risk, risk)
            self.value = nn.Linear(risk, risk)
            self.risk_output = nn.Linear(risk, risk)

        def encode(self, host, device):
            def encode_block(values):
                return self.gru(values.to(device))[1][-1]

            blocks = host.split(params["batch_size"])
            recompute = self.training and params["gradient_checkpointing"]
            return torch.cat([checkpoint(encode_block, block, use_reentrant=False, preserve_rng_state=False)
                              if recompute else encode_block(block) for block in blocks])

        def risk_graph(self, hidden):
            risk = self.risk_input(hidden)
            queries, keys, values = self.query(risk), self.key(risk), self.value(risk)
            nodes = torch.arange(len(risk), device=risk.device)

            def aggregate(query, key, value, rows):
                adj = (query @ key.T).relu().masked_fill(rows[:, None] == nodes, 0)
                return adj.softmax(dim=1) @ value

            size = params["graph_block_size"]
            recompute = self.training and params["gradient_checkpointing"]
            messages = [checkpoint(aggregate, query, keys, values, rows, use_reentrant=False,
                                   preserve_rng_state=False)
                        if recompute else aggregate(query, keys, values, rows)
                        for query, rows in zip(queries.split(size), nodes.split(size))]
            return risk + torch.cat(messages)

    class StatusGate(nn.Module):
        def __init__(self):
            super().__init__()
            dimensions = (params["risk_size"], params["base_factors"])
            self.logits = nn.Linear(*dimensions)
            self.noise = nn.Linear(*dimensions)

        def forward(self, alpha, risk):
            logits = self.logits(risk)
            if self.training:
                logits = logits + functional.softplus(self.noise(risk)) * torch.randn_like(logits)
            best, indices = logits.topk(params["top_k"], dim=1)
            score = (best.softmax(dim=1) * alpha.gather(1, indices)).sum(dim=1)
            router_z = logits.logsumexp(dim=1).square().mean() if self.training else None
            return score, router_z

    model = nn.Module()
    model.abcm = ABCM()
    model.gate = StatusGate()
    return model


def _ranknet(torch, score, target, block_size):
    """Exact all-pair RankNet gradient, without retaining a pair matrix for backward."""
    from torch.nn import functional

    with torch.no_grad():
        gradient = torch.empty_like(score)
        loss = score.new_zeros(())
        detached = score.detach()
        sorted_target = target.sort().values
        comparison_sum = (torch.searchsorted(sorted_target, target)
                          + torch.searchsorted(sorted_target, target, right=True)).to(score.dtype) * 0.5
        for start in range(0, len(score), block_size):
            difference = detached[start:start + block_size, None] - detached
            gradient[start:start + block_size] = difference.sigmoid().sum(1) - comparison_sum[start:start + block_size]
            loss += functional.softplus(difference).sum()
        count = len(score) * (len(score) - 1)
        comparison_loss = detached @ (2 * comparison_sum - len(score))
        loss = (loss - comparison_loss - len(score) * np.log(2.0)) / count
    return loss + ((score - detached) * gradient).sum() * (2 / count)


def _risk_losses(torch, factors, rows, target, ridge, eps):
    x = factors.index_select(0, rows)
    x = x / x.norm(dim=0).clamp_min(eps)
    gram = x.T @ x
    cholesky = torch.linalg.cholesky_ex(gram + ridge * torch.eye(gram.shape[0], device=x.device)).L
    coef = torch.cholesky_solve((x.T @ target)[:, None], cholesky).squeeze(1)
    residual = (target - x @ coef).square().sum() / target.square().sum().clamp_min(eps)
    centered = factors - factors.mean(0)
    unit = centered / centered.norm(dim=0).clamp_min(eps)
    corr = unit.T @ unit
    return residual, (corr - corr.diag().diag()).square().mean()


class ADWMAlphaModel(DailySequenceAlphaModel):
    @staticmethod
    def make_feature_provider(config):
        return FundamentalStyleProvider(config)

    @staticmethod
    def adjust_training_window(config, calendar, window):
        if window.training_start_date is None or window.activation_date is None:
            raise ValueError("ADWM requires calendar-month rolling training windows")
        gate_start, valid_start = _boundaries(window.training_start_date, config.model.params)
        cutoff = window.training_cutoff_date
        positions = {day: i for i, day in enumerate(calendar.dates)}
        gate_bound = bisect_left(calendar.dates, gate_start)
        valid_bound = bisect_left(calendar.dates, valid_start)
        primary = config.label.lookahead_days
        horizon = max(primary, *(label.lookahead_days for label in config.label.auxiliary))
        frequency = config.sample.train_frequency
        # Keep the public five-session sampling phase across the train pool.
        pool = sample_dates(calendar, (window.training_start_date, calendar.previous_open(valid_start)), frequency)
        abcm = [day for day in pool if day < gate_start and positions[day] + horizon < gate_bound]
        gate = [day for day in pool if day >= gate_start and positions[day] + primary < valid_bound]
        valid = [day for day in sample_dates(calendar, (valid_start, cutoff), frequency)
                 if positions[day] + primary <= positions[cutoff]]
        if not abcm or not gate or not valid:
            raise ValueError(f"{window.window_id}: ADWM has an empty split after label maturity filtering")
        return replace(window, train_dates=abcm + gate, valid_dates=valid)

    def _weak_factors(self, torch, source, date, device, previous):
        generator = self.model.abcm
        position = source.positions[date]
        # H at each lag is evaluated on that day's full stock graph.
        # Frozen weights allow adjacent dates to reuse their last two H tensors.
        if previous is not None:
            for day in list(previous):
                if source.positions[day] < position - 2 or day > date:
                    del previous[day]
        current_symbols = source.symbols(date)
        hidden = generator.encode(torch.from_numpy(source.get(date, current_symbols)), device)
        current = generator.risk_graph(hidden)
        smoothed = current
        for day in source.dates[max(0, position - 2):position]:
            cached = previous.get(day) if previous is not None else None
            if cached is None:
                symbols = source.symbols(day)
                if not symbols:
                    continue
                encoded = generator.encode(torch.from_numpy(source.get(day, symbols)), device)
                lag = generator.risk_graph(encoded)
                cached = pd.Index(symbols), lag
                if previous is not None:
                    previous[day] = cached
            symbols, lag = cached
            aligned = symbols.get_indexer(current_symbols)
            rows = np.flatnonzero(aligned >= 0)
            if len(rows):
                smoothed = smoothed.index_add(0, torch.as_tensor(rows, device=device),
                                               lag[torch.as_tensor(aligned[rows], device=device)])
        if previous is not None:
            previous[date] = (pd.Index(current_symbols), current)
        return pd.Index(current_symbols), generator.alpha(hidden), generator.risk_output(smoothed)

    def _scores(self, torch, bundle, date, frame, device, *, training):
        symbols, alpha, risk = self._weak_factors(torch, self._source(bundle), int(date), device,
                                                  self._predict_previous)
        scores, _ = self.model.gate(alpha, risk)
        return scores[torch.as_tensor(symbols.get_indexer(frame.ts_code), device=device)]

    def predict_bundle(self, bundle, context):
        self._predict_previous = {}
        try:
            return super().predict_bundle(bundle, context)
        finally:
            self._predict_previous.clear()

    def _abcm_sections(self, frame, primary, auxiliary):
        sections = []
        for date, rows in frame.groupby("trade_date", sort=True):
            alpha = rows[primary].to_numpy(dtype=np.float32, na_value=np.nan)
            risk = rows[auxiliary].to_numpy(dtype=np.float32, na_value=np.nan)
            alpha_rows = np.flatnonzero(np.isfinite(alpha))
            risk_rows = np.flatnonzero(np.isfinite(risk))
            if (len(alpha_rows) >= 2 and len(risk_rows) > self.params["base_factors"] + self.params["risk_size"]
                    and risk[risk_rows].dot(risk[risk_rows]) > self.params["eps"]):
                sections.append((int(date), pd.Index(rows.ts_code), alpha_rows, alpha[alpha_rows],
                                 risk_rows, risk[risk_rows]))
        if not sections:
            raise ValueError("ADWM requires usable ABCM training cross-sections")
        return sections

    def _gate_sections(self, torch, source, frame, label, device, industry):
        previous, sections = {}, []
        with torch.no_grad():
            for date, rows in frame.groupby("trade_date", sort=True):
                date = int(date)
                symbols = pd.Index(rows.ts_code)
                source.symbols(date)
                size = source.size_exposures[date].reindex(symbols).to_numpy()
                target = industry_size_residual(rows[label].to_numpy(dtype=np.float64, na_value=np.nan),
                                                size, industry.indices(symbols, date))
                observed = np.isfinite(target)
                if observed.sum() < 2:
                    continue
                universe, alpha, risk = self._weak_factors(torch, source, date, device, previous)
                indices = torch.as_tensor(universe.get_indexer(symbols[observed]), device=device)
                sections.append((alpha[indices].cpu(), risk[indices].cpu(), torch.from_numpy(target[observed])))
        if not sections:
            raise ValueError("ADWM requires usable neutralized gate cross-sections")
        return sections

    def _fit_stage(self, torch, module, stage, optimizer, context, train_epoch, validate=None):
        diagnostics = _diagnostics(context)
        best, stale, state, best_epoch = np.inf, 0, None, 0
        for epoch in range(1, self.params["epochs"] + 1):
            started = perf_counter()
            module.train()
            metrics = train_epoch(optimizer)
            module.eval()
            objective = metrics["train_loss"] if validate is None else validate()
            if not np.isfinite(objective):
                raise ValueError(f"nonfinite ADWM {stage} objective at epoch {epoch}")
            improved = objective < best
            if improved:
                best, stale, state, best_epoch = objective, 0, _cpu_state_dict(module), epoch
            else:
                stale += 1
            self.loss_history.append(dict(window_id=context.artifact_dir.name, stage=stage, epoch=epoch,
                **metrics, valid_loss=objective if validate is not None else None,
                best_loss=best, stale_epochs=stale, is_best=improved, epoch_elapsed_seconds=perf_counter() - started))
            if diagnostics["print_epoch"]:
                details = " ".join(f"{name}={value:.7g}" for name, value in metrics.items())
                print(f"adwm stage={stage} epoch={epoch} {details} selection_loss={objective:.7g} "
                      f"stale={stale} epoch_elapsed_s={perf_counter() - started:.2f}", flush=True)
            if stale >= self.params["patience"]:
                break
        module.load_state_dict(state)
        return best_epoch, best

    def fit_bundle(self, train, valid, context):
        torch, nn = _torch_modules()
        params = self.params = _params(context.model_params)
        self.feature_columns = list(context.feature_columns)
        self.input_dim = len(self.feature_columns)
        source, valid_source = self._source(train), self._source(valid)
        primary, auxiliary = "future_vwap_return_10d", "future_vwap_return_20d"
        if (self.input_dim != params["input_size"] or context.model_search
                or context.label_column != primary or auxiliary not in context.label_columns
                or train.training_start_date is None):
            raise ValueError("ADWM requires matching sequence inputs and 10/20-day VWAP labels")
        gate_start, valid_start = _boundaries(train.training_start_date, params)
        if (train.frame.empty or valid.frame.empty or train.frame.trade_date.max() >= valid_start
                or valid.frame.trade_date.min() < valid_start):
            raise ValueError("ADWM requires chronological and disjoint training/validation cuts")
        abcm = self._abcm_sections(train.frame[train.frame.trade_date < gate_start], primary, auxiliary)
        _set_seed(torch, params["seed"])
        device = _device(torch, params["device"])
        self.model = _network(torch, nn, params).to(device)
        self.loss_history = []

        def step(loss, optimizer, module):
            loss.backward()
            nn.utils.clip_grad_norm_(module.parameters(), params["grad_clip"], error_if_nonfinite=True)
            optimizer.step()

        def abcm_epoch(optimizer):
            totals = torch.zeros(5, device=device)
            for date, rows, alpha_rows, alpha_target, risk_rows, risk_target in abcm:
                optimizer.zero_grad(set_to_none=True)
                symbols, alpha, risk = self._weak_factors(torch, source, date, device, None)
                ids = symbols.get_indexer(rows)
                x = alpha[torch.as_tensor(ids[alpha_rows], device=device)]
                y = torch.as_tensor(alpha_target, device=device)
                mse = (x - y[:, None]).square().mean()
                rank = _ranknet(torch, x.mean(1), y, params["rank_block_size"])
                residual, corr = _risk_losses(torch, torch.cat((alpha, risk), dim=1),
                    torch.as_tensor(ids[risk_rows], device=device), torch.as_tensor(risk_target, device=device),
                    params["ridge"], params["eps"])
                loss = mse + params["rank_weight"] * rank + params["risk_weight"] * residual + params["orthogonal_weight"] * corr
                step(loss, optimizer, self.model.abcm)
                totals += torch.stack((loss, mse, rank, residual, corr)).detach()
            return dict(zip(("train_loss", "train_mse", "ranknet_loss", "unexplained_risk", "correlation_loss"),
                            (totals / len(abcm)).tolist()))

        abcm_epoch_id, abcm_loss = self._fit_stage(torch, self.model.abcm, "abcm",
            torch.optim.Adam(self.model.abcm.parameters(), lr=params["lr"]), context, abcm_epoch)
        self.model.abcm.requires_grad_(False)
        membership = artifact_io.read_parquet(source.industry_path, columns=["ts_code", "l1_code", "in_date", "out_date"])
        industry = IndustryMembership(membership, sorted(membership.l1_code.dropna().unique()))
        gate = self._gate_sections(torch, source, train.frame[train.frame.trade_date >= gate_start], primary, device, industry)
        validation = self._gate_sections(torch, valid_source, valid.frame, primary, device, industry)

        def gate_epoch(optimizer):
            totals = torch.zeros(3, device=device)
            for alpha, risk, target in gate:
                optimizer.zero_grad(set_to_none=True)
                score, router_z = self.model.gate(alpha.to(device), risk.to(device))
                mse = (score - target.to(device)).square().mean()
                loss = mse + params["router_z_weight"] * router_z
                step(loss, optimizer, self.model.gate)
                totals += torch.stack((loss, mse, router_z)).detach()
            return dict(zip(("train_loss", "train_mse", "router_z_loss"), (totals / len(gate)).tolist()))

        def validate():
            with torch.no_grad():
                losses = [(self.model.gate(alpha.to(device), risk.to(device))[0] - target.to(device)).square().mean()
                          for alpha, risk, target in validation]
            return float(torch.stack(losses).mean())

        gate_epoch_id, gate_loss = self._fit_stage(torch, self.model.gate, "gate",
            torch.optim.Adam(self.model.gate.parameters(), lr=params["gate_lr"]), context, gate_epoch, validate)
        self.model.to("cpu").eval()
        self.model_info = dict(params=params, feature_columns=self.feature_columns,
            abcm_dates=len(abcm), gate_dates=len(gate), valid_dates=len(validation), gate_start=gate_start,
            validation_start=valid_start, abcm_best_epoch=abcm_epoch_id, abcm_best_loss=abcm_loss,
            best_epoch=gate_epoch_id, best_valid_mse=gate_loss)

    def save(self, path):
        path = Path(path)
        artifact_io.mkdir(path.parent, parents=True, exist_ok=True)
        artifact_io.torch_save(dict(kind="adwm_model3", version=1, params=self.params,
            feature_columns=self.feature_columns, state_dict=_cpu_state_dict(self.model),
            model_info=self.model_info, loss_history=self.loss_history), path)

    @classmethod
    def load(cls, path):
        torch, nn = _torch_modules()
        saved = artifact_io.torch_load(Path(path), map_location="cpu")
        if saved.get("kind") != "adwm_model3" or saved.get("version") != 1:
            raise ValueError("unsupported ADWM checkpoint")
        instance = cls()
        instance.params = _params(saved["params"])
        instance.feature_columns = saved["feature_columns"]
        instance.input_dim = len(instance.feature_columns)
        instance.model = _network(torch, nn, instance.params)
        instance.model.load_state_dict(saved["state_dict"])
        instance.model.eval()
        instance.model_info, instance.loss_history = saved["model_info"], saved["loss_history"]
        return instance
