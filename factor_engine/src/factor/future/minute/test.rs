use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::compute_rolling_minute_by_instrument;
use crate::factor::Factor;
use crate::operators::ts_mean;

const SHORT_WINDOW: usize = 5;
const LONG_WINDOW: usize = 10;

struct MovingAverageRatio;

pub fn create() -> Box<dyn Factor> {
    Box::new(MovingAverageRatio)
}

impl Factor for MovingAverageRatio {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "test".to_string(),
            aliases: Vec::new(),
            name: "Futures minute moving average ratio".to_string(),
            asset_class: AssetClass::Future,
            frequency: Frequency::Minute1,
            version: "0.1.0".to_string(),
            tags: ["price_volume", "momentum", "minute"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            description: "MA(5) / MA(10) - 1 of one-minute closes per contract, seeded from the previous trading day."
                .to_string(),
            dependencies: vec![DataRequest::new(DatasetId::FutureMinute1m, &["close"])],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 1 },
        }
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        compute_rolling_minute_by_instrument(
            self.spec(),
            context,
            data,
            DatasetId::FutureMinute1m,
            "close",
            LONG_WINDOW - 1,
            |closes| {
                let short = ts_mean(closes, SHORT_WINDOW, SHORT_WINDOW);
                let long = ts_mean(closes, LONG_WINDOW, LONG_WINDOW);
                short
                    .into_iter()
                    .zip(long)
                    .map(|(short, long)| {
                        let long = long.filter(|value| *value != 0.0)?;
                        Some(short? / long - 1.0).filter(|value| value.is_finite())
                    })
                    .collect()
            },
        )
    }
}
