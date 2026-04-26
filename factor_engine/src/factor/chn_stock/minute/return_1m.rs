use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::compute_minute_by_instrument;
use crate::factor::Factor;
use crate::operators::ts_pctchg;

pub struct StockMinuteReturn1m;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockMinuteReturn1m)
}

impl Factor for StockMinuteReturn1m {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "return_1m".to_string(),
            aliases: vec!["stock.minute_1m.pv.return_1m".to_string()],
            name: "Stock one-minute return".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Minute1,
            version: "0.1.0".to_string(),
            tags: ["price_volume", "return", "minute"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "One-minute close-to-close return per stock within each trading day."
                .to_string(),
            dependencies: vec![DataRequest::new(DatasetId::StockMinute1m, &["close"])],
            lookback: Lookback { trading_days: 0 },
        }
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        compute_minute_by_instrument(
            self.spec(),
            context,
            data,
            DatasetId::StockMinute1m,
            |series| Ok(ts_pctchg(series.column("close")?, 1)),
        )
    }
}
