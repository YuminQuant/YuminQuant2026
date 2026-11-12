"""Characteristic-managed momentum and reversal on the raw weekly bar panel."""
from __future__ import annotations

import math

import numpy as np

from yq_ml_alpha.features.neutralization import apply_rust_neutralization
from yq_ml_alpha.features.resnet_panel import ResNetPanelProvider, _read_required
from yq_ml_alpha.features.transforms import _zscore_series, apply_cross_section_transform
from yq_ml_alpha.models.bar_panel_model import BarPanelAlphaModel, normalize_bar_params
from yq_ml_alpha.models.bar_training import fit_bar_panel
from yq_ml_alpha.models.resnet_gru_model import _ResNetGRURegressor


class CMMWeekAlphaModel(BarPanelAlphaModel):
    @classmethod
    def make_feature_provider(cls, config):
        features = config.features
        if features.type != "resnet_panel":
            return None
        preprocess = features.params.get("week_preprocess", "none")
        if preprocess == "none":
            return None
        if preprocess != "mad5_zscore":
            raise ValueError("CMM week_preprocess must be none or mad5_zscore")
        return CMMWeekPanelProvider(features.root, features.columns, features.params)

    def _normalize_params(self, raw):
        params = normalize_bar_params(raw)
        if (params["sequence_length"], params["input_size"]) != (150, 7):
            raise ValueError("CMM week requires 150 sessions, six bar features and one daily return")
        params["return_lookback"] = int(params["return_lookback"])
        params["half_life"] = float(params["half_life"])
        params["conv_channels"] = int(params["conv_channels"])
        params["kernel_size"] = int(params["kernel_size"])
        if not 1 <= params["return_lookback"] <= params["sequence_length"]:
            raise ValueError("CMM return_lookback must fit the bar history")
        if not math.isfinite(params["half_life"]) or params["half_life"] <= 0:
            raise ValueError("CMM half_life must be finite and positive")
        if params["conv_channels"] < 1 or params["kernel_size"] not in (1, 3, 5):
            raise ValueError("CMM requires positive conv_channels and kernel_size of 1, 3 or 5")
        return params

    def _build_regressor(self, nn):
        import torch

        params = self.params
        lookback = params["return_lookback"]

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = _ResNetGRURegressor(
                    nn, {**params, "input_size": 6, "patch_width": 5}, encoder_only=True,
                )
                self.head = nn.Linear(params["hidden_size"], 3)
                ages = torch.arange(lookback - 1, -1, -1, dtype=torch.float32)
                self.register_buffer("decay", torch.exp2(-ages / params["half_life"]))

            def return_weights(self, inputs):
                returns = inputs[:, -lookback:, 6]
                hidden = self.encoder.encode(inputs[:, :, :6])
                momentum, reversal, gate_logit = self.head(hidden).split(1, dim=-1)
                gate = torch.sigmoid(gate_logit)
                positive = torch.softmax(momentum * returns + self.decay, dim=-1)
                negative = torch.softmax(reversal * returns + self.decay, dim=-1)
                return gate * positive - (1.0 - gate) * negative

            def forward(self, inputs):
                return (self.return_weights(inputs) * inputs[:, -lookback:, 6]).sum(dim=-1)

        return Model()

    @staticmethod
    def _mse_loss(torch, prediction, target):
        valid = torch.isfinite(target)
        if int(valid.sum()) < 2:
            return None
        loss = (prediction[valid] - target[valid]).square().mean()
        if not torch.isfinite(loss):
            raise ValueError("CMM produced a nonfinite MSE")
        return loss

    def _training_loss(self, torch, inputs, target):
        return self._mse_loss(torch, self.model(inputs), target)

    def _fit(self, train_data, valid_data, context, *, train_tensor, valid_tensor):
        spec = str(context.model_params.get("label_neutralize", "none"))
        if valid_data.empty:
            raise ValueError("CMM requires validation data for MSE weight selection")

        def prepare(frame):
            frame = frame.copy()
            label = context.label_column
            frame[label] = apply_rust_neutralization(frame, frame[label], spec)
            return apply_cross_section_transform(frame, "zscore", [], [label])

        train_data, valid_data = prepare(train_data), prepare(valid_data)
        valid_label_counts = valid_data.groupby("trade_date")[context.label_column].count()
        if not valid_label_counts.ge(2).any():
            raise ValueError("CMM requires a validation date with at least two usable labels")
        fit_bar_panel(
            self, train_data, valid_data, context,
            train_tensor=train_tensor, valid_tensor=valid_tensor, validation_loss=self._mse_loss,
            loss_name="cross_section_standardized_label_mse", train_loss_log_key="cmm_train_mse",
            missing_loss_error="CMM could not compute finite MSE",
        )
        self.model_info.update(validation_loss="datewise_mse",
                               label_neutralize=spec,
                               return_lookback=self.params["return_lookback"],
                               half_life=self.params["half_life"])


class CMMWeekPanelProvider(ResNetPanelProvider):
    """Adjusted week bars with CMM's cross-sectional preprocessing."""

    def __init__(self, root, columns, params):
        super().__init__(root, columns, params)
        if self.dataset != "week" or not self.include_daily_return:
            raise ValueError("CMM requires week bars with daily returns")

    def _prepare_week_window(self, values, symbols):
        valid_anchor = np.isfinite(values[:, -1, 3]) & (values[:, -1, 3] > 0)
        values, symbols = super()._prepare_week_window(values[valid_anchor], symbols[valid_anchor])
        values[:, :, :6] = _mad5_zscore_panel(values[:, :, :6])
        # The return branch is already standardized at each historical date.
        return np.where(np.isfinite(values), values, 0.0), symbols

    def _week_session(self, date):
        columns = ["open", "high", "low", "close", "vol", "amount", "pre_close"]
        frame = _read_required(self.root, date, columns)
        close = frame.close.astype("float64")
        previous = frame.pre_close.astype("float64")
        returns = close.where(close > 0) / previous.where(previous > 0) - 1.0
        frame = frame.assign(daily_return_zscore=_zscore_series(returns))
        adjustment = _read_required(self.daily_root / "adj_factor", date, ["adj_factor"])
        basic = _read_required(self.daily_root / "basic", date, ["turnover_rate_f"])
        frame = frame.merge(adjustment, on=["trade_date", "ts_code"], how="left", validate="one_to_one")
        frame = frame.merge(basic, on=["trade_date", "ts_code"], how="left", validate="one_to_one").set_index("ts_code")
        numeric = frame[["open", "high", "low", "close", "vol", "amount", "adj_factor", "turnover_rate_f"]].astype("float64")
        numeric = numeric.where(np.isfinite(numeric))
        numeric.iloc[:, :7] = numeric.iloc[:, :7].where(numeric.iloc[:, :7] > 0)
        numeric["turnover_rate_f"] = numeric.turnover_rate_f.where(numeric.turnover_rate_f >= 0)
        numeric["vwap"] = numeric["amount"] * 10.0 / numeric["vol"]  # Daily source: thousand yuan / lots.
        numeric.loc[:, self.output_features[:5]] = numeric[self.output_features[:5]].mul(numeric.adj_factor, axis=0)
        numeric["daily_return_zscore"] = frame["daily_return_zscore"]
        return numeric.index, numeric[self.output_features].to_numpy(dtype="float64")[:, None, :]


def _mad5_zscore_panel(values):
    """Winsorize then standardize each historical slot and channel across stocks."""
    if not len(values):
        return values
    flat = np.where(np.isfinite(values), values, np.nan).reshape(len(values), -1)
    observed = np.isfinite(flat).any(axis=0)
    data = flat[:, observed]
    median = np.nanmedian(data, axis=0)
    mad = np.nanmedian(np.abs(data - median), axis=0)
    data = np.clip(data, median - 5.0 * mad, median + 5.0 * mad)
    mean, scale = np.nanmean(data, axis=0), np.nanstd(data, axis=0)
    flat.fill(0.0)
    flat[:, observed] = np.divide(data - mean, scale, out=np.zeros_like(data),
                                 where=(scale > 0) & np.isfinite(data))
    return flat.reshape(values.shape)
