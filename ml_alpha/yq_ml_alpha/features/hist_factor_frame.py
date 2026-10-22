"""HIST factor histories; label availability never defines graph membership."""
from yq_ml_alpha.config import _project_path
from yq_ml_alpha.data.hist_concepts import HistConcepts
from yq_ml_alpha.data.stores import daily_path
from yq_ml_alpha.features.graph_factor_frame import FactorSequenceProvider


class HistFactorFrameProvider(FactorSequenceProvider):
    def __init__(self, config, provider):
        super().__init__(config, provider)
        self.concepts = HistConcepts(
            _project_path(config.model.params.get("concept_root", "data/derived/stock/hist_concepts")),
            _project_path(config.model.params.get("basic_root", config.data_root / "chn_stock_data/daily/basic")),
        )

    def dataset_identity(self):
        return {"factor_history": super().dataset_identity()["factor_history"], "concepts": self.concepts.identity(),
                "market_cap": "daily_basic_total_mv"}

    def load(self, trade_date):
        self.concepts.identity()
        frame = super().load(trade_date)
        if not frame.empty and not daily_path(self.concepts.root, trade_date).exists():
            raise FileNotFoundError(f"missing HIST concepts on {trade_date}; run derive-hist-concepts")
        return frame
