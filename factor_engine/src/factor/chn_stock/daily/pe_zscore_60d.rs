use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::compute_daily_by_instrument;
use crate::factor::Factor;
use crate::operators::ts_zscore;

pub struct StockDailyPeZscore60d;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyPeZscore60d)
}

impl Factor for StockDailyPeZscore60d {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "pe_zscore_60d".to_string(),
            aliases: Vec::new(),
            name: "Stock 60-day PE z-score".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["valuation", "pe", "zscore", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "Rolling 60 trading-day z-score of daily PE.".to_string(),
            dependencies: vec![DataRequest::new(DatasetId::StockDailyBasic, &["pe"])],
            lookback: Lookback { trading_days: 60 },
        }
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        compute_daily_by_instrument(
            self.spec(),
            context,
            data.daily(DatasetId::StockDailyBasic)?,
            |series| Ok(ts_zscore(series.column("pe")?, 60, 60)),
        )
    }
}
