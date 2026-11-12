from __future__ import annotations

from yq_ml_alpha.models.rawbar_training import RawBarRecurrentRegressor, _RawBarAlphaModel


class RawBarLSTMAlphaModel(_RawBarAlphaModel):
    """Single-output LSTM over a normalized daily rawbar panel."""

    architecture = "lstm"

    def _build_regressor(self, nn):
        def readout(network, output, state):
            return state[0][-1]

        return RawBarRecurrentRegressor(nn, self.params, nn.LSTM, readout, getattr(self, "_panel_stats", None))
