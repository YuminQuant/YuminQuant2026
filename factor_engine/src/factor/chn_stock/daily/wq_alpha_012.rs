use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::Factor;
use crate::operators::ts_diff;

pub struct StockDailyWQAlpha012;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyWQAlpha012)
}

impl Factor for StockDailyWQAlpha012 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "WQAlpha012".to_string(),
            aliases: Vec::new(),
            name: "WQAlpha012".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["worldquant101alpha", "price_volume", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "sign(delta(volume,1)) * -delta(close,1)".to_string(),
            dependencies: vec![DataRequest::new(DatasetId::StockDailyPv, &["close", "vol"])],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 1 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let delta_volume = panel.column("vol")?.ts(|values| ts_diff(values, 1))?;
        let delta_close = panel.column("close")?.ts(|values| ts_diff(values, 1))?;
        let factor = delta_volume.ts_binary(&delta_close, |delta_volume, delta_close| {
            delta_volume
                .iter()
                .zip(delta_close)
                .map(|(delta_volume, delta_close)| {
                    match (clean(*delta_volume), clean(*delta_close)) {
                        (Some(delta_volume), Some(delta_close)) => {
                            Some(delta_volume.signum() * -delta_close)
                        }
                        _ => None,
                    }
                })
                .collect()
        })?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| !value.is_nan())
}
