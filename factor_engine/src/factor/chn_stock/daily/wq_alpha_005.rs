use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::Factor;
use crate::operators::{cs_pctrank, ts_sum};

pub struct StockDailyWQAlpha005;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyWQAlpha005)
}

impl Factor for StockDailyWQAlpha005 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "WQAlpha005".to_string(),
            aliases: Vec::new(),
            name: "WQAlpha005".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["worldquant101alpha", "price_volume", "daily", "deprecated"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "rank(open - sum(vwap,10)/10) * -abs(rank(close - vwap))".to_string(),
            dependencies: vec![DataRequest::new(
                DatasetId::StockDailyPv,
                &["amount", "close", "open", "vol"],
            )],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 9 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let amount = panel.column("amount")?;
        let volume = panel.column("vol")?;
        let vwap = amount.ts_binary(&volume, |amount, volume| {
            amount
                .iter()
                .zip(volume)
                .map(|(amount, volume)| match (clean(*amount), clean(*volume)) {
                    (Some(amount), Some(volume)) if volume.abs() > f64::EPSILON => {
                        Some(amount * 10.0 / volume)
                    }
                    _ => None,
                })
                .collect()
        })?;
        let avg_vwap_10 = vwap.ts(|values| {
            ts_sum(values, 10, 10)
                .into_iter()
                .map(|value| value.map(|value| value * 0.1))
                .collect()
        })?;
        let left = panel
            .column("open")?
            .ts_binary(&avg_vwap_10, |open, avg_vwap| {
                open.iter()
                    .zip(avg_vwap)
                    .map(|(open, avg_vwap)| match (clean(*open), clean(*avg_vwap)) {
                        (Some(open), Some(avg_vwap)) => Some(open - avg_vwap),
                        _ => None,
                    })
                    .collect()
            })?
            .cs(|values| cs_pctrank(values, true))?;
        let right = panel
            .column("close")?
            .ts_binary(&vwap, |close, vwap| {
                close
                    .iter()
                    .zip(vwap)
                    .map(|(close, vwap)| match (clean(*close), clean(*vwap)) {
                        (Some(close), Some(vwap)) => Some(close - vwap),
                        _ => None,
                    })
                    .collect()
            })?
            .cs(|values| {
                cs_pctrank(values, true)
                    .into_iter()
                    .map(|value| value.map(|value| -value.abs()))
                    .collect()
            })?;
        let factor = left.ts_binary(&right, |left, right| {
            left.iter()
                .zip(right)
                .map(|(left, right)| match (clean(*left), clean(*right)) {
                    (Some(left), Some(right)) => Some(left * right),
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
