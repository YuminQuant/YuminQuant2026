use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::Factor;
use crate::operators::{ts_diff, ts_mean, ts_pctrank};

pub struct StockDailyWQAlpha007;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyWQAlpha007)
}

impl Factor for StockDailyWQAlpha007 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "WQAlpha007".to_string(),
            aliases: Vec::new(),
            name: "WQAlpha007".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["worldquant101alpha", "price_volume", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "if adv20 < volume then -ts_rank(abs(delta(close,7)),60) * sign(delta(close,7)) else -1".to_string(),
            dependencies: vec![DataRequest::new(
                DatasetId::StockDailyPv,
                &["close", "vol"],
            )],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 66 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let close = panel.column("close")?;
        let volume = panel.column("vol")?;
        let adv20 = volume.ts(|values| ts_mean(values, 20, 20))?;
        let delta_close = close.ts(|values| ts_diff(values, 7))?;
        let abs_delta_rank = delta_close.ts(|values| {
            let abs_delta = values
                .iter()
                .map(|value| clean(*value).map(f64::abs))
                .collect::<Vec<_>>();
            ts_pctrank(&abs_delta, 60, 60)
        })?;
        let true_value = abs_delta_rank.ts_binary(&delta_close, |ranked, delta| {
            ranked
                .iter()
                .zip(delta)
                .map(|(ranked, delta)| match (clean(*ranked), clean(*delta)) {
                    (Some(ranked), Some(delta)) => Some(-ranked * delta.signum()),
                    _ => None,
                })
                .collect()
        })?;
        let factor = volume.ts_ternary(&adv20, &true_value, |volume, adv20, true_value| {
            volume
                .iter()
                .zip(adv20)
                .zip(true_value)
                .map(
                    |((volume, adv20), true_value)| match (clean(*volume), clean(*adv20)) {
                        (Some(volume), Some(adv20)) if adv20 < volume => clean(*true_value),
                        (Some(_), Some(_)) => Some(-1.0),
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
