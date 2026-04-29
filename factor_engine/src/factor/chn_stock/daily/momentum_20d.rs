use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::Factor;
use crate::operators::ts_pctchg;

pub struct StockDailyMomentum20d;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyMomentum20d)
}

impl Factor for StockDailyMomentum20d {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "momentum_20d".to_string(),
            aliases: vec!["stock.daily.pv.momentum_20d".to_string()],
            name: "Stock 20-day close return".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["price_volume", "momentum", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "Close-to-close percentage change over 20 trading days.".to_string(),
            dependencies: vec![DataRequest::new(DatasetId::StockDailyPv, &["close"])],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 20 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let factor = panel.column("close")?.ts(|values| ts_pctchg(values, 20))?;
        Ok(factor.to_factor_series(self.spec()))
    }
}
