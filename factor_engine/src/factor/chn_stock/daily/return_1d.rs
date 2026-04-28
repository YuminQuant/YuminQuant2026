use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::map_binary;
use crate::factor::common::DailyPanel;
use crate::factor::Factor;

pub struct StockDailyReturn1d;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyReturn1d)
}

impl Factor for StockDailyReturn1d {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "return_1d".to_string(),
            aliases: vec!["stock.daily.pv.return_1d".to_string()],
            name: "Stock daily close/pre_close return".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["price_volume", "return", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "Daily stock return computed from close and pre_close.".to_string(),
            dependencies: vec![DataRequest::new(
                DatasetId::StockDailyPv,
                &["close", "pre_close"],
            )],
            lookback: Lookback { trading_days: 0 },
        }
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = DailyPanel::from_table(data.daily(DatasetId::StockDailyPv)?, context)?;
        let close = panel.column("close")?;
        let pre_close = panel.column("pre_close")?;
        let factor = close.ts_binary(&pre_close, |close, pre_close| {
            map_binary(close, pre_close, |close, pre_close| {
                (pre_close.abs() > f64::EPSILON).then_some(close / pre_close - 1.0)
            })
        })?;
        Ok(factor.to_factor_series(self.spec()))
    }
}
