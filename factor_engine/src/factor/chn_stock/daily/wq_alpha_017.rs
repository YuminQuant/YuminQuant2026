use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::Factor;
use crate::operators::{cs_pctrank, ts_diff, ts_mean, ts_pctrank};

pub struct StockDailyWQAlpha017;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyWQAlpha017)
}

impl Factor for StockDailyWQAlpha017 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "WQAlpha017".to_string(),
            aliases: Vec::new(),
            name: "WQAlpha017".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["worldquant101alpha", "price_volume", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "-rank(ts_rank(close,10)) * rank(delta(delta(close,1),1)) * rank(ts_rank(volume / adv20,5))".to_string(),
            dependencies: vec![DataRequest::new(
                DatasetId::StockDailyPv,
                &["close", "vol"],
            )],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 23 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let close = panel.column("close")?;
        let volume = panel.column("vol")?;
        let part1 = close
            .ts(|values| ts_pctrank(values, 10, 10))?
            .cs(|values| {
                cs_pctrank(values, true)
                    .into_iter()
                    .map(|value| value.map(|value| -value))
                    .collect()
            })?;
        let part2 = close
            .ts(|values| ts_diff(values, 1))?
            .ts(|values| ts_diff(values, 1))?
            .cs(|values| cs_pctrank(values, true))?;
        let adv20 = volume.ts(|values| ts_mean(values, 20, 20))?;
        let volume_adv = volume.ts_binary(&adv20, |volume, adv20| {
            volume
                .iter()
                .zip(adv20)
                .map(|(volume, adv20)| match (clean(*volume), clean(*adv20)) {
                    (Some(volume), Some(adv20)) if adv20.abs() > f64::EPSILON => {
                        Some(volume / adv20)
                    }
                    _ => None,
                })
                .collect()
        })?;
        let part3 = volume_adv
            .ts(|values| ts_pctrank(values, 5, 5))?
            .cs(|values| cs_pctrank(values, true))?;
        let left = part1.ts_binary(&part2, |part1, part2| {
            part1
                .iter()
                .zip(part2)
                .map(|(part1, part2)| match (clean(*part1), clean(*part2)) {
                    (Some(part1), Some(part2)) => Some(part1 * part2),
                    _ => None,
                })
                .collect()
        })?;
        let factor = left.ts_binary(&part3, |left, part3| {
            left.iter()
                .zip(part3)
                .map(|(left, part3)| match (clean(*left), clean(*part3)) {
                    (Some(left), Some(part3)) => Some(left * part3),
                    _ => None,
                })
                .collect()
        })?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| !value.is_nan())
}
