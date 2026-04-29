use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
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
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 60 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyBasic)?;
        let factor = panel.column("pe")?.ts(|values| ts_zscore(values, 60, 60))?;
        Ok(factor.to_factor_series(self.spec()))
    }
}
