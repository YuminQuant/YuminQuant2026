use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::Factor;
use crate::operators::{cs_pctrank, ts_corr, ts_diff};

pub struct StockDailyWQAlpha002;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyWQAlpha002)
}

impl Factor for StockDailyWQAlpha002 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "WQAlpha002".to_string(),
            aliases: Vec::new(),
            name: "WQAlpha002".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["worldquant101alpha", "price_volume", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description:
                "-correlation(rank(delta(log(volume), 2)), rank((close - open) / open), 6)"
                    .to_string(),
            dependencies: vec![DataRequest::new(
                DatasetId::StockDailyPv,
                &["close", "open", "vol"],
            )],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 7 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let volume = panel.column("vol")?;
        let log_volume = volume.ts(|values| {
            values
                .iter()
                .map(|value| clean(*value).and_then(|value| (value > 0.0).then_some(value.ln())))
                .collect()
        })?;
        let ranked_delta = log_volume
            .ts(|values| ts_diff(values, 2))?
            .cs(|values| cs_pctrank(values, true))?;
        let open = panel.column("open")?;
        let close = panel.column("close")?;
        let ranked_return = close
            .ts_binary(&open, |close, open| {
                close
                    .iter()
                    .zip(open)
                    .map(|(close, open)| match (clean(*close), clean(*open)) {
                        (Some(close), Some(open)) if open.abs() > f64::EPSILON => {
                            Some((close - open) / open)
                        }
                        _ => None,
                    })
                    .collect()
            })?
            .cs(|values| cs_pctrank(values, true))?;
        let factor = ranked_delta.ts_binary(&ranked_return, |left, right| {
            ts_corr(left, right, 6, 6)
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
