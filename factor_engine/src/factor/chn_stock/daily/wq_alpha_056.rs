use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::Factor;
use crate::operators::*;

pub struct StockDailyWQAlpha056;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyWQAlpha056)
}

impl Factor for StockDailyWQAlpha056 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "WQAlpha056".to_string(),
            aliases: Vec::new(),
            name: "WQAlpha056".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["worldquant101alpha", "price_volume", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "WQAlpha056".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close", "pre_close"]),
                DataRequest::new(DatasetId::StockDailyBasic, &["circ_mv"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 10 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let pv_panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let basic_panel = data.daily_panel(DatasetId::StockDailyBasic)?;
        let close = pv_panel.column("close")?;
        let returns = close.zip_binary(&pv_panel.column("pre_close")?, ret)?;
        let left = returns
            .ts(|values| ts_sum(values, 10, 10))?
            .zip_binary(
                &returns
                    .ts(|values| ts_sum(values, 2, 2))?
                    .ts(|values| ts_sum(values, 3, 3))?,
                div,
            )?
            .cs(|values| cs_pctrank(values, true))?;
        let right = returns
            .zip_binary(&basic_panel.column("circ_mv")?, mul)?
            .cs(|values| cs_pctrank(values, true))?;
        let factor = left
            .zip_binary(&right, mul)?
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
