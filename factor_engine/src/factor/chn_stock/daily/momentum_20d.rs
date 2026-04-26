use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::compute_daily_by_instrument;
use crate::factor::common::vector::map_binary;
use crate::factor::Factor;
use crate::operators::ts_mean;

pub struct StockDailyMomentum20d;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyMomentum20d)
}

impl Factor for StockDailyMomentum20d {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "momentum_20d".to_string(),
            aliases: vec!["stock.daily.pv.momentum_20d".to_string()],
            name: "Stock close over trailing 20-day mean".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["price_volume", "momentum", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "Close divided by trailing 20 trading-day mean close, minus one."
                .to_string(),
            dependencies: vec![DataRequest::new(DatasetId::StockDailyPv, &["close"])],
            lookback: Lookback { trading_days: 20 },
        }
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        compute_daily_by_instrument(
            self.spec(),
            context,
            data.daily(DatasetId::StockDailyPv)?,
            |series| {
                let close = series.column("close")?;
                let mean_20 = ts_mean(close, 20, 20);
                Ok(map_binary(close, &mean_20, |close, mean| {
                    (mean.abs() > f64::EPSILON).then_some(close / mean - 1.0)
                }))
            },
        )
    }
}
