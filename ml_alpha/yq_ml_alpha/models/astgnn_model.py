"""Standalone week and msnew ASTGNN factor."""
from yq_ml_alpha.models.astgnn_temporal_trainer import ASTGNNTemporalTrainer


class ASTGNNAlphaModel(ASTGNNTemporalTrainer):
    """Train, save and predict the standalone factor."""
