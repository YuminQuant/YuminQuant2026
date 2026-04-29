use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::Factor;

pub struct StockDailyWQAlpha054;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyWQAlpha054)
}

impl Factor for StockDailyWQAlpha054 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "WQAlpha054".to_string(),
            aliases: Vec::new(),
            name: "WQAlpha054".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["worldquant101alpha", "price_volume", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "WQAlpha054".to_string(),
            dependencies: vec![DataRequest::new(
                DatasetId::StockDailyPv,
                &["low", "close", "open", "high"],
            )],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 0 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let low = panel.column("low")?;
        let close = panel.column("close")?;
        let open = panel.column("open")?;
        let high = panel.column("high")?;
        let numerator = low.zip_binary(&close, sub)?.zip_binary(
            &open.map_values(|value| clean(value).map(|value| value.powi(5))),
            mul,
        )?;
        let denominator = low.zip_binary(&high, sub)?.zip_binary(
            &close.map_values(|value| clean(value).map(|value| value.powi(5))),
            mul,
        )?;
        let factor = numerator
            .zip_binary(&denominator, div)?
            .map_values(|value| clean(value).map(|value| -value));
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
