use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::Factor;
use crate::operators::*;

pub struct StockDailyWQAlpha055;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyWQAlpha055)
}

impl Factor for StockDailyWQAlpha055 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "WQAlpha055".to_string(),
            aliases: Vec::new(),
            name: "WQAlpha055".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["worldquant101alpha", "price_volume", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "WQAlpha055".to_string(),
            dependencies: vec![DataRequest::new(
                DatasetId::StockDailyPv,
                &["close", "low", "high", "vol"],
            )],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 17 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let close = panel.column("close")?;
        let low_min = panel.column("low")?.ts(|values| ts_min(values, 12, 12))?;
        let high_max = panel.column("high")?.ts(|values| ts_max(values, 12, 12))?;
        let normalized = close
            .zip_binary(&low_min, sub)?
            .zip_binary(&high_max.zip_binary(&low_min, sub)?, div)?
            .cs(|values| cs_pctrank(values, true))?;
        let rank_volume = panel.column("vol")?.cs(|values| cs_pctrank(values, true))?;
        let factor = normalized
            .ts_binary(&rank_volume, |left, right| ts_corr(left, right, 6, 6))?
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
