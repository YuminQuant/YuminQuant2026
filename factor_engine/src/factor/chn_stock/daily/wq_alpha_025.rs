use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::Factor;
use crate::operators::*;

pub struct StockDailyWQAlpha025;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyWQAlpha025)
}

impl Factor for StockDailyWQAlpha025 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "WQAlpha025".to_string(),
            aliases: Vec::new(),
            name: "WQAlpha025".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["worldquant101alpha", "price_volume", "daily", "deprecated"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "WQAlpha025".to_string(),
            dependencies: vec![DataRequest::new(
                DatasetId::StockDailyPv,
                &["close", "pre_close", "high", "vol", "amount"],
            )],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 20 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let close = panel.column("close")?;
        let pre_close = panel.column("pre_close")?;
        let high = panel.column("high")?;
        let volume = panel.column("vol")?;
        let amount = panel.column("amount")?;
        let returns = close.zip_binary(&pre_close, ret)?;
        let adv20 = volume.ts(|values| ts_mean(values, 20, 20))?;
        let vwap = amount.zip_binary(&volume, vwap_value)?;
        let raw = returns
            .map_values(|value| clean(value).map(|value| -value))
            .zip_binary(&adv20, mul)?
            .zip_binary(&vwap, mul)?
            .zip_binary(&high.zip_binary(&close, sub)?, mul)?;
        let factor = raw.cs(|values| cs_pctrank(values, true))?;
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
