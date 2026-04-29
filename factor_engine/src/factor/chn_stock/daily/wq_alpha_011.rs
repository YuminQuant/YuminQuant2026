use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::Factor;
use crate::operators::{cs_pctrank, ts_diff, ts_max, ts_min};

pub struct StockDailyWQAlpha011;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyWQAlpha011)
}

impl Factor for StockDailyWQAlpha011 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "WQAlpha011".to_string(),
            aliases: Vec::new(),
            name: "WQAlpha011".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["worldquant101alpha", "price_volume", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "(rank(ts_max(vwap - close,3)) + rank(ts_min(vwap - close,3))) * rank(delta(volume,3))".to_string(),
            dependencies: vec![DataRequest::new(
                DatasetId::StockDailyPv,
                &["amount", "close", "vol"],
            )],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 3 },
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
        let vwap_close = vwap.ts_binary(&panel.column("close")?, |vwap, close| {
            vwap.iter()
                .zip(close)
                .map(|(vwap, close)| match (clean(*vwap), clean(*close)) {
                    (Some(vwap), Some(close)) => Some(vwap - close),
                    _ => None,
                })
                .collect()
        })?;
        let left = vwap_close
            .ts(|values| ts_max(values, 3, 3))?
            .cs(|values| cs_pctrank(values, true))?
            .ts_binary(
                &vwap_close
                    .ts(|values| ts_min(values, 3, 3))?
                    .cs(|values| cs_pctrank(values, true))?,
                |max_rank, min_rank| {
                    max_rank
                        .iter()
                        .zip(min_rank)
                        .map(
                            |(max_rank, min_rank)| match (clean(*max_rank), clean(*min_rank)) {
                                (Some(max_rank), Some(min_rank)) => Some(max_rank + min_rank),
                                _ => None,
                            },
                        )
                        .collect()
                },
            )?;
        let ranked_delta_volume = volume
            .ts(|values| ts_diff(values, 3))?
            .cs(|values| cs_pctrank(values, true))?;
        let factor = left.ts_binary(&ranked_delta_volume, |left, right| {
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
