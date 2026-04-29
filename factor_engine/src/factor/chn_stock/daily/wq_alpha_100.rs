use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::common::{ClassificationLevel, ClassificationMap};
use crate::factor::Factor;
use crate::operators::*;

pub struct StockDailyWQAlpha100;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyWQAlpha100)
}

impl Factor for StockDailyWQAlpha100 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "WQAlpha100".to_string(),
            aliases: Vec::new(),
            name: "WQAlpha100".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["worldquant101alpha", "price_volume", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "WQAlpha100".to_string(),
            dependencies: vec![
                DataRequest::new(
                    DatasetId::StockDailyPv,
                    &["close", "low", "high", "vol", "amount"],
                ),
                DataRequest::new(DatasetId::StockSwClassification, &["l3_code"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 30 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let sector_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockSwClassification)?,
            ClassificationLevel::Subindustry,
        )?;
        let close = panel.column("close")?;
        let low = panel.column("low")?;
        let high = panel.column("high")?;
        let volume = panel.column("vol")?;
        let adv20 = volume.ts(|values| ts_mean(values, 20, 20))?;
        let left_rank = close
            .zip_binary(&low, sub)?
            .zip_binary(&high.zip_binary(&close, sub)?, sub)?
            .zip_binary(&high.zip_binary(&low, sub)?, div)?
            .zip_binary(&volume, mul)?
            .cs(|values| cs_pctrank(values, true))?;
        let left_neutral = left_rank
            .cs_by_group(
                |date, codes| sector_map.groups_for(date, codes),
                cs_neutralize,
            )?
            .cs_by_group(
                |date, codes| sector_map.groups_for(date, codes),
                cs_neutralize,
            )?
            .cs(cs_scale)?
            .map_values(|value| clean(value).map(|value| value * 1.5));
        let right = close
            .ts_binary(
                &adv20.cs(|values| cs_pctrank(values, true))?,
                |close, adv| ts_corr(close, adv, 5, 5),
            )?
            .zip_binary(
                &close
                    .ts(|values| ts_argmin(values, 30, 30))?
                    .cs(|values| cs_pctrank(values, true))?,
                sub,
            )?
            .cs_by_group(
                |date, codes| sector_map.groups_for(date, codes),
                cs_neutralize,
            )?
            .cs(cs_scale)?;
        let factor = left_neutral
            .zip_binary(&right, sub)?
            .zip_binary(&volume.zip_binary(&adv20, div)?, mul)?
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
