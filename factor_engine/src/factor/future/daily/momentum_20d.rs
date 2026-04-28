use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::DailyPanel;
use crate::factor::Factor;
use crate::operators::ts_pctchg;

pub struct FutureDailyMomentum20d;

pub fn create() -> Box<dyn Factor> {
    Box::new(FutureDailyMomentum20d)
}

impl Factor for FutureDailyMomentum20d {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "momentum_20d".to_string(),
            aliases: vec!["future.daily.pv.momentum_20d".to_string()],
            name: "Future 20-day close return".to_string(),
            asset_class: AssetClass::Future,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["price_volume", "momentum", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "Close-to-close percentage change over 20 trading days.".to_string(),
            dependencies: vec![DataRequest::new(DatasetId::FutureDaily, &["close"])],
            lookback: Lookback { trading_days: 20 },
        }
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = DailyPanel::from_table(data.daily(DatasetId::FutureDaily)?, context)?;
        let factor = panel.column("close")?.ts(|values| ts_pctchg(values, 20))?;
        Ok(factor.to_factor_series(self.spec()))
    }
}
