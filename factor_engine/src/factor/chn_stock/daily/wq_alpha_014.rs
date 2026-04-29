use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::Factor;
use crate::operators::{cs_pctrank, ts_corr, ts_diff};

pub struct StockDailyWQAlpha014;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyWQAlpha014)
}

impl Factor for StockDailyWQAlpha014 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "WQAlpha014".to_string(),
            aliases: Vec::new(),
            name: "WQAlpha014".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["worldquant101alpha", "price_volume", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "-rank(delta(returns,3)) * correlation(open, volume, 10)".to_string(),
            dependencies: vec![DataRequest::new(
                DatasetId::StockDailyPv,
                &["close", "open", "pre_close", "vol"],
            )],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 9 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let returns =
            panel
                .column("close")?
                .ts_binary(&panel.column("pre_close")?, |close, pre_close| {
                    close
                        .iter()
                        .zip(pre_close)
                        .map(
                            |(close, pre_close)| match (clean(*close), clean(*pre_close)) {
                                (Some(close), Some(pre_close))
                                    if pre_close.abs() > f64::EPSILON =>
                                {
                                    Some(close / pre_close - 1.0)
                                }
                                _ => None,
                            },
                        )
                        .collect()
                })?;
        let left = returns.ts(|values| ts_diff(values, 3))?.cs(|values| {
            cs_pctrank(values, true)
                .into_iter()
                .map(|value| value.map(|value| -value))
                .collect()
        })?;
        let right = panel
            .column("open")?
            .ts_binary(&panel.column("vol")?, |open, volume| {
                ts_corr(open, volume, 10, 10)
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
