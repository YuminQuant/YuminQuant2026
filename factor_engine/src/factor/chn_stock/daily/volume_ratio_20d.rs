use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::compute_daily_by_instrument;
use crate::factor::common::vector::map_binary;
use crate::factor::Factor;
use crate::operators::ts_delay;
use crate::operators::ts_mean;

pub struct StockDailyVolumeRatio20d;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyVolumeRatio20d)
}

impl Factor for StockDailyVolumeRatio20d {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "stock.daily.pv.volume_ratio_20d".to_string(),
            name: "Stock volume over trailing 20-day average".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["price_volume", "volume", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description:
                "Current volume divided by the average volume of the previous 20 observations."
                    .to_string(),
            dependencies: vec![DataRequest::new(DatasetId::StockDailyPv, &["vol"])],
            lookback: Lookback { trading_days: 20 },
        }
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        compute_daily_by_instrument(
            self.spec(),
            context,
            data.daily(DatasetId::StockDailyPv)?,
            |series| {
                let volume = series.column("vol")?;
                let prev_volume = ts_delay(volume, 1);
                let mean_prev_20 = ts_mean(&prev_volume, 20, 20);
                Ok(map_binary(volume, &mean_prev_20, |volume, mean| {
                    (mean.abs() > f64::EPSILON).then_some(volume / mean)
                }))
            },
        )
    }
}
