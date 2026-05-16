use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::gfzq_behavioral;
use crate::factor::Factor;

const VERSION: &str = "0.1.0";

pub struct StockDailyCgo;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyCgo)
}

impl Factor for StockDailyCgo {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "cgo".to_string(),
            aliases: vec!["CGO".to_string()],
            name: "Capital Gains Overhang".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: ["GFZQ", "behavioral", "chip", "turnover", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "GFZQ capital gains overhang from 100-day surviving turnover weighted daily VWAP reference price.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["amount", "vol", "close"]),
                DataRequest::new(DatasetId::StockDailyBasic, &["turnover_rate_f"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: gfzq_behavioral::CGO_WINDOW,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let (_panel, factor) = gfzq_behavioral::cgo_from_data(data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}
