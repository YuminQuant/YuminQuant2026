use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::Factor;
use crate::operators::{ts_diff, ts_max, ts_min};

pub struct StockDailyWQAlpha009;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyWQAlpha009)
}

impl Factor for StockDailyWQAlpha009 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "WQAlpha009".to_string(),
            aliases: Vec::new(),
            name: "WQAlpha009".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["worldquant101alpha", "price_volume", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "if 0 < ts_min(delta(close,1),5) then delta(close,1) else if ts_max(delta(close,1),5) < 0 then delta(close,1) else -delta(close,1)".to_string(),
            dependencies: vec![DataRequest::new(DatasetId::StockDailyPv, &["close"])],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 5 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let delta_close = panel.column("close")?.ts(|values| ts_diff(values, 1))?;
        let min_delta = delta_close.ts(|values| ts_min(values, 5, 5))?;
        let max_delta = delta_close.ts(|values| ts_max(values, 5, 5))?;
        let factor = delta_close.ts_ternary(&min_delta, &max_delta, |delta, min, max| {
            delta
                .iter()
                .zip(min)
                .zip(max)
                .map(
                    |((delta, min), max)| match (clean(*delta), clean(*min), clean(*max)) {
                        (Some(delta), Some(min), Some(max)) => Some(if 0.0 < min || max < 0.0 {
                            delta
                        } else {
                            -delta
                        }),
                        _ => None,
                    },
                )
                .collect()
        })?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| !value.is_nan())
}
