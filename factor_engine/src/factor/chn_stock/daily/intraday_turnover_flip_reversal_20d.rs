use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::Factor;
use crate::operators::{cs_mean, ts_diff, ts_mean};

pub struct StockDailyIntradayTurnoverFlipReversal20d;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyIntradayTurnoverFlipReversal20d)
}

impl Factor for StockDailyIntradayTurnoverFlipReversal20d {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "intraday_turnover_flip_reversal_20d".to_string(),
            aliases: Vec::new(),
            name: "Intraday Turnover Flip Reversal 20D".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: [
                "price_volume",
                "intraday_return",
                "reversal",
                "turnover",
                "daily",
                "FZZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description:
                "20-day mean of close/open intraday returns flipped by cross-sectional turnover change."
                    .to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close", "open"]),
                DataRequest::new(DatasetId::StockDailyBasic, &["turnover_rate_f"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 20 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let returns = panel
            .column("close")?
            .zip_binary(&panel.column("open")?, ret)?;
        let turnover = panel
            .column_from_table(data.daily(DatasetId::StockDailyBasic)?, "turnover_rate_f")?
            .map_values(|value| clean(value).map(|value| value / 100.0));
        let turnover_delta = turnover.ts(|values| ts_diff(values, 1))?;
        let turnover_delta_mean = turnover_delta.cs(cs_mean)?;
        let flipped = returns.zip_binary(
            &turnover_delta.zip_binary(&turnover_delta_mean, less_than)?,
            |ret, flip| match (clean(ret), clean(flip)) {
                (Some(ret), Some(flip)) => Some(if flip > 0.0 { -ret } else { ret }),
                _ => None,
            },
        )?;
        let factor = flipped.ts(|values| ts_mean(values, 20, 20))?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn ret(close: Option<f64>, open: Option<f64>) -> Option<f64> {
    match (clean(close), clean(open)) {
        (Some(close), Some(open)) if open.abs() > f64::EPSILON => Some(close / open - 1.0),
        _ => None,
    }
}

fn less_than(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some((left < right) as i32 as f64),
        _ => None,
    }
}
