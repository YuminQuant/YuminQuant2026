use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::Factor;
use crate::operators::{cs_pctrank, ts_delay, ts_diff, ts_sum};

pub struct StockDailyWQAlpha019;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyWQAlpha019)
}

impl Factor for StockDailyWQAlpha019 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "WQAlpha019".to_string(),
            aliases: Vec::new(),
            name: "WQAlpha019".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["worldquant101alpha", "price_volume", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "-sign((close - delay(close,7)) + delta(close,7)) * (1 + rank(1 + sum(returns,250)))".to_string(),
            dependencies: vec![DataRequest::new(
                DatasetId::StockDailyPv,
                &["close", "pre_close"],
            )],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 249 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let close = panel.column("close")?;
        let delayed_close = close.ts(|values| ts_delay(values, 7))?;
        let delta_close = close.ts(|values| ts_diff(values, 7))?;
        let signal = close.ts_ternary(&delayed_close, &delta_close, |close, delayed, delta| {
            close
                .iter()
                .zip(delayed)
                .zip(delta)
                .map(|((close, delayed), delta)| {
                    match (clean(*close), clean(*delayed), clean(*delta)) {
                        (Some(close), Some(delayed), Some(delta)) => {
                            Some(-((close - delayed) + delta).signum())
                        }
                        _ => None,
                    }
                })
                .collect()
        })?;
        let returns = close.ts_binary(&panel.column("pre_close")?, |close, pre_close| {
            close
                .iter()
                .zip(pre_close)
                .map(
                    |(close, pre_close)| match (clean(*close), clean(*pre_close)) {
                        (Some(close), Some(pre_close)) if pre_close.abs() > f64::EPSILON => {
                            Some(close / pre_close - 1.0)
                        }
                        _ => None,
                    },
                )
                .collect()
        })?;
        let ranked_sum = returns
            .ts(|values| ts_sum(values, 250, 250))?
            .cs(|values| {
                let shifted = values
                    .iter()
                    .map(|value| clean(*value).map(|value| value + 1.0))
                    .collect::<Vec<_>>();
                cs_pctrank(&shifted, true)
            })?;
        let factor = signal.ts_binary(&ranked_sum, |signal, ranked_sum| {
            signal
                .iter()
                .zip(ranked_sum)
                .map(
                    |(signal, ranked_sum)| match (clean(*signal), clean(*ranked_sum)) {
                        (Some(signal), Some(ranked_sum)) => Some(signal * (1.0 + ranked_sum)),
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
