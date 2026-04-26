use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::compute_daily_by_instrument;
use crate::factor::Factor;
use crate::operators::ts_pctchg;
use crate::operators::ts_std_dev;

pub struct FutureDailyVolatility20d;

pub fn create() -> Box<dyn Factor> {
    Box::new(FutureDailyVolatility20d)
}

impl Factor for FutureDailyVolatility20d {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "volatility_20d".to_string(),
            aliases: vec!["future.daily.pv.volatility_20d".to_string()],
            name: "Future 20-day return volatility".to_string(),
            asset_class: AssetClass::Future,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["price_volume", "volatility", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "Standard deviation of the latest 20 close-to-close daily returns."
                .to_string(),
            dependencies: vec![DataRequest::new(DatasetId::FutureDaily, &["close"])],
            lookback: Lookback { trading_days: 21 },
        }
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        compute_daily_by_instrument(
            self.spec(),
            context,
            data.daily(DatasetId::FutureDaily)?,
            |series| {
                let returns = ts_pctchg(series.column("close")?, 1);
                Ok(ts_std_dev(&returns, 20, 20))
            },
        )
    }
}
