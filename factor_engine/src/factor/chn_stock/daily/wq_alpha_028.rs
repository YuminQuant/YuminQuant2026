use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::Factor;
use crate::operators::*;

pub struct StockDailyWQAlpha028;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyWQAlpha028)
}

impl Factor for StockDailyWQAlpha028 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "WQAlpha028".to_string(),
            aliases: Vec::new(),
            name: "WQAlpha028".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["worldquant101alpha", "price_volume", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "WQAlpha028".to_string(),
            dependencies: vec![DataRequest::new(
                DatasetId::StockDailyPv,
                &["close", "high", "low", "vol"],
            )],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 24 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let close = panel.column("close")?;
        let low = panel.column("low")?;
        let high = panel.column("high")?;
        let adv20 = panel.column("vol")?.ts(|values| ts_mean(values, 20, 20))?;
        let corr = adv20.ts_binary(&low, |adv20, low| ts_corr(adv20, low, 5, 5))?;
        let mid = high.zip_binary(&low, |high, low| add(high, low).map(|value| value / 2.0))?;
        let factor = corr
            .zip_binary(&mid, add)?
            .zip_binary(&close, sub)?
            .cs(cs_scale)?;
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
