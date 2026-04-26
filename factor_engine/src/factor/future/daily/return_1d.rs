use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::compute_daily_by_instrument;
use crate::factor::common::vector::map_binary;
use crate::factor::Factor;

pub struct FutureDailyReturn1d;

pub fn create() -> Box<dyn Factor> {
    Box::new(FutureDailyReturn1d)
}

impl Factor for FutureDailyReturn1d {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "return_1d".to_string(),
            aliases: vec!["future.daily.pv.return_1d".to_string()],
            name: "Future daily close/pre_close return".to_string(),
            asset_class: AssetClass::Future,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["price_volume", "return", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "Daily futures contract return computed from close and pre_close."
                .to_string(),
            dependencies: vec![DataRequest::new(
                DatasetId::FutureDaily,
                &["close", "pre_close"],
            )],
            lookback: Lookback { trading_days: 0 },
        }
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        compute_daily_by_instrument(
            self.spec(),
            context,
            data.daily(DatasetId::FutureDaily)?,
            |series| {
                Ok(map_binary(
                    series.column("close")?,
                    series.column("pre_close")?,
                    |close, pre_close| {
                        (pre_close.abs() > f64::EPSILON).then_some(close / pre_close - 1.0)
                    },
                ))
            },
        )
    }
}
