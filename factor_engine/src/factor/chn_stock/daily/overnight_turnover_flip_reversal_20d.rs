use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::Factor;
use crate::operators::{cs_mean, ts_delay, ts_diff, ts_mean};

pub struct StockDailyOvernightTurnoverFlipReversal20d;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyOvernightTurnoverFlipReversal20d)
}

impl Factor for StockDailyOvernightTurnoverFlipReversal20d {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "overnight_turnover_flip_reversal_20d".to_string(),
            aliases: Vec::new(),
            name: "Overnight Turnover Flip Reversal 20D".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: [
                "price_volume",
                "overnight_return",
                "reversal",
                "turnover",
                "daily",
                "FZZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "20-day mean of overnight distance flipped by lagged turnover distance."
                .to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close", "open"]),
                DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
                DataRequest::new(DatasetId::StockDailyBasic, &["turnover_rate_f"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 21 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let adj_factor =
            panel.column_from_table(data.daily(DatasetId::StockAdjFactor)?, "adj_factor")?;
        let adj_open = panel.column("open")?.zip_binary(&adj_factor, mul)?;
        let adj_close = panel.column("close")?.zip_binary(&adj_factor, mul)?;
        let prev_adj_close = adj_close.ts(|values| ts_delay(values, 1))?;
        let overnight_return = adj_open.zip_binary(&prev_adj_close, ret)?;
        let overnight_mean = overnight_return.cs(cs_mean)?;
        let overnight_distance = overnight_return.zip_binary(&overnight_mean, abs_diff)?;

        let turnover = panel
            .column_from_table(data.daily(DatasetId::StockDailyBasic)?, "turnover_rate_f")?
            .map_values(|value| clean(value).map(|value| value / 100.0));
        let turnover_delta_lag1 = turnover
            .ts(|values| ts_diff(values, 1))?
            .ts(|values| ts_delay(values, 1))?;
        let turnover_delta_lag1_mean = turnover_delta_lag1.cs(cs_mean)?;
        let turnover_distance =
            turnover_delta_lag1.zip_binary(&turnover_delta_lag1_mean, abs_diff)?;
        let turnover_distance_mean = turnover_distance.cs(cs_mean)?;

        let flipped = overnight_distance.zip_binary(
            &turnover_distance.zip_binary(&turnover_distance_mean, less_than)?,
            |distance, flip| match (clean(distance), clean(flip)) {
                (Some(distance), Some(flip)) => Some(if flip > 0.0 { -distance } else { distance }),
                _ => None,
            },
        )?;
        let factor = flipped.ts(|values| ts_mean(values, 20, 20))?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn mul(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some(left * right),
        _ => None,
    }
}

fn ret(open: Option<f64>, prev_close: Option<f64>) -> Option<f64> {
    match (clean(open), clean(prev_close)) {
        (Some(open), Some(prev_close)) if prev_close.abs() > f64::EPSILON => {
            Some(open / prev_close - 1.0)
        }
        _ => None,
    }
}

fn abs_diff(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some((left - right).abs()),
        _ => None,
    }
}

fn less_than(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some((left < right) as i32 as f64),
        _ => None,
    }
}
