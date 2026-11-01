"""Shared lazy market/style inputs and label-free date-wise prediction."""
from __future__ import annotations

from yq_ml_alpha.features.market_style_sequence import normalize_market_style
from yq_ml_alpha.models.daily_sequence_model import DailySequenceAlphaModel


def full_date_ic_loss(prediction, target, eps):
    if prediction.ndim != 1 or target.shape != prediction.shape or prediction.numel() < 2:
        raise ValueError("full-date IC requires equally sized one-dimensional vectors with at least two stocks")
    prediction, target = prediction - prediction.mean(), target - target.mean()
    denominator = prediction.norm() * target.norm()
    return 1 - (prediction * target).sum() / denominator.clamp_min(eps)


class MarketStyleAlphaModel(DailySequenceAlphaModel):
    def _source(self, bundle):
        source = super()._source(bundle)
        if not hasattr(source, "normalization"):
            raise ValueError(f"{type(self).__name__} requires matching observed market/style histories")
        return source

    def _inputs(self, bundle, date, frame):
        source = self._source(bundle)
        symbols = frame.ts_code.tolist()
        if sorted(symbols) != source.symbols(int(date)):
            raise ValueError("market/style normalization requires the complete eligible date cross-section")
        return normalize_market_style(source.get(int(date), symbols), mode=source.normalization, eps=self.params["eps"])
