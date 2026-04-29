use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::Factor;
use crate::operators::{cs_pctrank, ts_argmax, ts_std_dev};

pub struct StockDailyWQAlpha001;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyWQAlpha001)
}

impl Factor for StockDailyWQAlpha001 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "WQAlpha001".to_string(),
            aliases: Vec::new(),
            name: "WQAlpha001".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["worldquant101alpha", "price_volume", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "rank(ts_argmax((if returns < 0 then stddev(returns,20) else close)^2 signed, 5)) - 0.5".to_string(),
            dependencies: vec![DataRequest::new(
                DatasetId::StockDailyPv,
                &["close", "pre_close"],
            )],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 23 },
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
        let std_returns = returns.ts(|values| ts_std_dev(values, 20, 20))?;
        let base = returns.ts_ternary(&std_returns, &close, |returns, std_returns, close| {
            returns
                .iter()
                .zip(std_returns)
                .zip(close)
                .map(|((ret, std_ret), close)| match clean(*ret) {
                    Some(ret) if ret < 0.0 => clean(*std_ret),
                    Some(_) => clean(*close),
                    None => None,
                })
                .collect()
        })?;
        let powered = base.ts(|values| {
            values
                .iter()
                .map(|value| clean(*value).map(|value| value.signum() * value.abs().powf(2.0)))
                .collect()
        })?;
        let argmax = powered.ts(|values| ts_argmax(values, 5, 5))?;
        let factor = argmax.cs(|values| {
            cs_pctrank(values, true)
                .into_iter()
                .map(|value| value.map(|value| value - 0.5))
                .collect()
        })?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| !value.is_nan())
}
