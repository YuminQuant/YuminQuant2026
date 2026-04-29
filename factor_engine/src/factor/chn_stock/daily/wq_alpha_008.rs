use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::Factor;
use crate::operators::{cs_pctrank, ts_delay, ts_sum};

pub struct StockDailyWQAlpha008;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyWQAlpha008)
}

impl Factor for StockDailyWQAlpha008 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "WQAlpha008".to_string(),
            aliases: Vec::new(),
            name: "WQAlpha008".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["worldquant101alpha", "price_volume", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description:
                "-rank(sum(open,5) * sum(returns,5) - delay(sum(open,5) * sum(returns,5),10))"
                    .to_string(),
            dependencies: vec![DataRequest::new(
                DatasetId::StockDailyPv,
                &["close", "open", "pre_close"],
            )],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 14 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let close = panel.column("close")?;
        let pre_close = panel.column("pre_close")?;
        let returns = close.ts_binary(&pre_close, |close, pre_close| {
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
        let sum_open = panel.column("open")?.ts(|values| ts_sum(values, 5, 5))?;
        let sum_returns = returns.ts(|values| ts_sum(values, 5, 5))?;
        let product = sum_open.ts_binary(&sum_returns, |sum_open, sum_returns| {
            sum_open
                .iter()
                .zip(sum_returns)
                .map(
                    |(sum_open, sum_returns)| match (clean(*sum_open), clean(*sum_returns)) {
                        (Some(sum_open), Some(sum_returns)) => Some(sum_open * sum_returns),
                        _ => None,
                    },
                )
                .collect()
        })?;
        let delayed = product.ts(|values| ts_delay(values, 10))?;
        let factor = product
            .ts_binary(&delayed, |current, delayed| {
                current
                    .iter()
                    .zip(delayed)
                    .map(
                        |(current, delayed)| match (clean(*current), clean(*delayed)) {
                            (Some(current), Some(delayed)) => Some(current - delayed),
                            _ => None,
                        },
                    )
                    .collect()
            })?
            .cs(|values| {
                cs_pctrank(values, true)
                    .into_iter()
                    .map(|value| value.map(|value| -value))
                    .collect()
            })?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| !value.is_nan())
}
