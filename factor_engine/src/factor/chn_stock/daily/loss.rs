use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::gfzq_behavioral;
use crate::factor::Factor;

const VERSION: &str = "0.1.0";

pub struct StockDailyLoss;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyLoss)
}

impl Factor for StockDailyLoss {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "loss".to_string(),
            aliases: vec!["Loss".to_string()],
            name: "Loss Selling Disposition".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: ["GFZQ", "behavioral", "chip", "turnover", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "GFZQ loss selling disposition factor from 100-day surviving turnover weighted unrealized loss versus current daily VWAP.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["amount", "vol"]),
                DataRequest::new(DatasetId::StockDailyBasic, &["turnover_rate_f"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: gfzq_behavioral::LOSS_WINDOW,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let (_panel, factor) = gfzq_behavioral::loss_from_data(data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}
