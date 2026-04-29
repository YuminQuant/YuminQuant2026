use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::Factor;
use crate::operators::{ts_pctchg, ts_std_dev};

pub struct StockDailyVolatility20d;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyVolatility20d)
}

impl Factor for StockDailyVolatility20d {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "volatility_20d".to_string(),
            aliases: vec!["stock.daily.pv.volatility_20d".to_string()],
            name: "Stock 20-day return volatility".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["price_volume", "volatility", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "Standard deviation of the latest 20 close-to-close daily returns."
                .to_string(),
            dependencies: vec![DataRequest::new(DatasetId::StockDailyPv, &["close"])],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 21 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let factor = panel
            .column("close")?
            .ts(|values| ts_pctchg(values, 1))?
            .ts(|values| ts_std_dev(values, 20, 20))?;
        Ok(factor.to_factor_series(self.spec()))
    }
}
