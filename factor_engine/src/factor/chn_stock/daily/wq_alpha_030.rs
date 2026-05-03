use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::Factor;
use crate::operators::*;

pub struct StockDailyWQAlpha030;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyWQAlpha030)
}

impl Factor for StockDailyWQAlpha030 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "WQAlpha030".to_string(),
            aliases: Vec::new(),
            name: "WQAlpha030".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["worldquant101alpha", "price_volume", "daily", "deprecated"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "WQAlpha030".to_string(),
            dependencies: vec![DataRequest::new(DatasetId::StockDailyPv, &["close", "vol"])],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 20 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let close = panel.column("close")?;
        let volume = panel.column("vol")?;
        let d1 = close.ts(|values| ts_delay(values, 1))?;
        let d2 = close.ts(|values| ts_delay(values, 2))?;
        let d3 = close.ts(|values| ts_delay(values, 3))?;
        let first_two = close.zip_ternary(&d1, &d2, |close, d1, d2| {
            match (sign_value(sub(close, d1)), sign_value(sub(d1, d2))) {
                (Some(left), Some(right)) => Some(left + right),
                _ => None,
            }
        })?;
        let signs =
            first_two.zip_binary(&d2.zip_binary(&d3, |d2, d3| sign_value(sub(d2, d3)))?, add)?;
        let ranked = signs.cs(|values| cs_pctrank(values, true))?;
        let volume_ratio = volume
            .ts(|values| ts_sum(values, 5, 5))?
            .zip_binary(&volume.ts(|values| ts_sum(values, 20, 20))?, div)?;
        let factor = ranked
            .map_values(|value| clean(value).map(|value| 1.0 - value))
            .zip_binary(&volume_ratio, mul)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

#[allow(dead_code)]
fn div(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (clean(left), clean(right)) {
        (Some(left), Some(right)) if right.abs() > f64::EPSILON => Some(left / right),
        _ => None,
    }
}

#[allow(dead_code)]
fn mul(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some(left * right),
        _ => None,
    }
}

#[allow(dead_code)]
fn add(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some(left + right),
        _ => None,
    }
}

#[allow(dead_code)]
fn sub(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some(left - right),
        _ => None,
    }
}

#[allow(dead_code)]
fn ret(close: Option<f64>, pre_close: Option<f64>) -> Option<f64> {
    div(close, pre_close).map(|value| value - 1.0)
}

#[allow(dead_code)]
fn vwap_value(amount: Option<f64>, volume: Option<f64>) -> Option<f64> {
    match (clean(amount), clean(volume)) {
        (Some(amount), Some(volume)) if volume.abs() > f64::EPSILON => Some(amount * 10.0 / volume),
        _ => None,
    }
}

#[allow(dead_code)]
fn sign_value(value: Option<f64>) -> Option<f64> {
    clean(value).map(|value| value.signum())
}
