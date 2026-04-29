use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::Factor;
use crate::operators::*;

pub struct StockDailyWQAlpha029;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyWQAlpha029)
}

impl Factor for StockDailyWQAlpha029 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "WQAlpha029".to_string(),
            aliases: Vec::new(),
            name: "WQAlpha029".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["worldquant101alpha", "price_volume", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "WQAlpha029".to_string(),
            dependencies: vec![DataRequest::new(
                DatasetId::StockDailyPv,
                &["close", "pre_close"],
            )],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 11 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let close = panel.column("close")?;
        let returns = close.zip_binary(&panel.column("pre_close")?, ret)?;
        let left = close
            .ts(|values| ts_diff(values, 5))?
            .cs(|values| cs_pctrank(values, true))?
            .map_values(|value| clean(value).map(|value| -value))
            .cs(|values| cs_pctrank(values, true))?
            .cs(|values| cs_pctrank(values, true))?
            .ts(|values| ts_min(values, 2, 2))?
            .map_values(|value| clean(value).and_then(|value| (value > 0.0).then_some(value.ln())))
            .cs(cs_scale)?
            .cs(|values| cs_pctrank(values, true))?
            .cs(|values| cs_pctrank(values, true))?
            .ts(|values| ts_min(values, 5, 5))?;
        let right = returns
            .map_values(|value| clean(value).map(|value| -value))
            .ts(|values| ts_delay(values, 6))?
            .ts(|values| ts_pctrank(values, 5, 5))?;
        let factor = left.zip_binary(&right, add)?;
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
