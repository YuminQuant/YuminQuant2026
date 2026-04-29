use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::Factor;
use crate::operators::{cs_pctrank, ts_delay};

pub struct StockDailyWQAlpha020;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyWQAlpha020)
}

impl Factor for StockDailyWQAlpha020 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "WQAlpha020".to_string(),
            aliases: Vec::new(),
            name: "WQAlpha020".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["worldquant101alpha", "price_volume", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "-rank(open - delay(high,1)) * rank(open - delay(close,1)) * rank(open - delay(low,1))".to_string(),
            dependencies: vec![DataRequest::new(
                DatasetId::StockDailyPv,
                &["close", "high", "low", "open"],
            )],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 1 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let open = panel.column("open")?;
        let part1 = open
            .ts_binary(
                &panel.column("high")?.ts(|values| ts_delay(values, 1))?,
                |open, high| {
                    open.iter()
                        .zip(high)
                        .map(|(open, high)| match (clean(*open), clean(*high)) {
                            (Some(open), Some(high)) => Some(open - high),
                            _ => None,
                        })
                        .collect()
                },
            )?
            .cs(|values| {
                cs_pctrank(values, true)
                    .into_iter()
                    .map(|value| value.map(|value| -value))
                    .collect()
            })?;
        let part2 = open
            .ts_binary(
                &panel.column("close")?.ts(|values| ts_delay(values, 1))?,
                |open, close| {
                    open.iter()
                        .zip(close)
                        .map(|(open, close)| match (clean(*open), clean(*close)) {
                            (Some(open), Some(close)) => Some(open - close),
                            _ => None,
                        })
                        .collect()
                },
            )?
            .cs(|values| cs_pctrank(values, true))?;
        let part3 = open
            .ts_binary(
                &panel.column("low")?.ts(|values| ts_delay(values, 1))?,
                |open, low| {
                    open.iter()
                        .zip(low)
                        .map(|(open, low)| match (clean(*open), clean(*low)) {
                            (Some(open), Some(low)) => Some(open - low),
                            _ => None,
                        })
                        .collect()
                },
            )?
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
