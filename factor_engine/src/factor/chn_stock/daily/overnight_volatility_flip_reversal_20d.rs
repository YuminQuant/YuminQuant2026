use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::Factor;
use crate::operators::{cs_mean, ts_delay, ts_mean, ts_std_dev};

pub struct StockDailyOvernightVolatilityFlipReversal20d;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyOvernightVolatilityFlipReversal20d)
}

impl Factor for StockDailyOvernightVolatilityFlipReversal20d {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "overnight_volatility_flip_reversal_20d".to_string(),
            aliases: Vec::new(),
            name: "Overnight Volatility Flip Reversal 20D".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: [
                "price_volume",
                "overnight_return",
                "reversal",
                "volatility",
                "daily",
                "FZZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "20-day overnight distance mean flipped by overnight distance volatility."
                .to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close", "open"]),
                DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 20 },
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
        let distance_mean20 = overnight_distance.ts(|values| ts_mean(values, 20, 20))?;
        let distance_std20 = overnight_distance.ts(|values| ts_std_dev(values, 20, 20))?;
        let distance_std20_mean = distance_std20.cs(cs_mean)?;
        let factor = distance_mean20.zip_binary(
            &distance_std20.zip_binary(&distance_std20_mean, less_than)?,
            |mean20, flip| match (clean(mean20), clean(flip)) {
                (Some(mean20), Some(flip)) => Some(if flip > 0.0 { -mean20 } else { mean20 }),
                _ => None,
            },
        )?;
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
