use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::Factor;
use crate::operators::{cs_pctrank, ts_corr, ts_std_dev};

pub struct StockDailyWQAlpha018;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyWQAlpha018)
}

impl Factor for StockDailyWQAlpha018 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "WQAlpha018".to_string(),
            aliases: Vec::new(),
            name: "WQAlpha018".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["worldquant101alpha", "price_volume", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description:
                "-rank(stddev(abs(close - open),5) + (close - open) + correlation(close, open,10))"
                    .to_string(),
            dependencies: vec![DataRequest::new(
                DatasetId::StockDailyPv,
                &["close", "open"],
            )],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 9 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let close = panel.column("close")?;
        let open = panel.column("open")?;
        let close_open = close.ts_binary(&open, |close, open| {
            close
                .iter()
                .zip(open)
                .map(|(close, open)| match (clean(*close), clean(*open)) {
                    (Some(close), Some(open)) => Some(close - open),
                    _ => None,
                })
                .collect()
        })?;
        let std_abs_close_open = close_open.ts(|values| {
            let abs_values = values
                .iter()
                .map(|value| clean(*value).map(f64::abs))
                .collect::<Vec<_>>();
            ts_std_dev(&abs_values, 5, 5)
        })?;
        let corr_close_open = close.ts_binary(&open, |close, open| ts_corr(close, open, 10, 10))?;
        let raw = std_abs_close_open.ts_ternary(
            &close_open,
            &corr_close_open,
            |std_abs, close_open, corr| {
                std_abs
                    .iter()
                    .zip(close_open)
                    .zip(corr)
                    .map(|((std_abs, close_open), corr)| {
                        match (clean(*std_abs), clean(*close_open), clean(*corr)) {
                            (Some(std_abs), Some(close_open), Some(corr)) => {
                                Some(std_abs + close_open + corr)
                            }
                            _ => None,
                        }
                    })
                    .collect()
            },
        )?;
        let factor = raw.cs(|values| {
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
