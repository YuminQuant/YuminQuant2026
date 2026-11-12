from __future__ import annotations

from yq_ml_alpha.models.sequence_training import _SequenceAlphaModel


class LSTMAlphaModel(_SequenceAlphaModel):
    rnn_type = "LSTM"
