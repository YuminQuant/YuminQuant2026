use std::collections::BTreeMap;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawAuxiliaryRequest, IntradayDailyRawRequest,
    IntradayDailyRawSeries, IntradayDailyRawSpec, Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::umr;
use crate::factor::common::{
    clean_intraday_value, intraday_time_in_range, stock_derived_bar_raw_spec, DailyPanel,
    PanelColumn,
};
use crate::factor::Factor;

const RAW_ID: &str = "daily_umr_minute_skewness";
const RAW_VERSION: &str = "0.2.0";
const VERSION: &str = "0.1.0";
const FIVE_MINUTE_BAR_SIZE: usize = 5;
const FIVE_MINUTE_BARS: usize = 48;

pub struct StockDailyUmrMinuteSkewness;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyUmrMinuteSkewness)
}

fn raw_spec() -> IntradayDailyRawSpec {
    stock_derived_bar_raw_spec(RAW_ID, RAW_VERSION, FIVE_MINUTE_BAR_SIZE, &["close"], 1)
}

impl Factor for StockDailyUmrMinuteSkewness {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "umr_minute_skewness".to_string(),
            aliases: vec![
                "UMR_MINUTE_SKEWNESS".to_string(),
                "Minute Skewness-Adjusted UMR".to_string(),
            ],
            name: "Minute Skewness-Adjusted UMR".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "Unified momentum and reversal factor adjusted by intraday 5-minute close-to-close return skewness, neutralized by Barra SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close", "pre_close"]),
                umr::market_close_return_request(),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(
                RAW_ID,
                umr::UMR_LOOKBACK,
            )],
            lookback: Lookback {
                trading_days: umr::UMR_LOOKBACK,
            },
        }
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        vec![raw_spec()]
    }

    fn intraday_raw_auxiliary_requirements(
        &self,
        raw_ids: &[String],
    ) -> Vec<IntradayDailyRawAuxiliaryRequest> {
        if raw_ids.iter().any(|raw_id| raw_id == RAW_ID) {
            vec![IntradayDailyRawAuxiliaryRequest::new(
                DataRequest::new(DatasetId::StockMinute1m, &["close"]),
                0,
            )]
        } else {
            Vec::new()
        }
    }

    fn minute_compute(
        &self,
        raw_id: &str,
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Option<IntradayDailyRawSeries>> {
        if raw_id != RAW_ID {
            return Ok(None);
        }
        let mut values = Vec::new();
        for trade_date in &context.target_dates {
            let daily_values = match data.derived_bar(FIVE_MINUTE_BAR_SIZE, *trade_date) {
                Some(table) => match raw_values_from_derived_bar(table) {
                    Ok(Some(values)) => values,
                    Ok(None) | Err(_) => {
                        eprintln!(
                            "warning: umr_minute_skewness falling back to 1m minute data for {trade_date}; derived 5m bar is incomplete or incompatible"
                        );
                        raw_values_from_minute(data, *trade_date)?
                    }
                },
                None => {
                    eprintln!(
                        "warning: umr_minute_skewness falling back to 1m minute data for {trade_date}; derived 5m bar is missing"
                    );
                    raw_values_from_minute(data, *trade_date)?
                }
            };
            for (ts_code, value) in daily_values {
                values.push(FactorValue {
                    key: FactorRowKey::Daily {
                        trade_date: *trade_date,
                        ts_code,
                    },
                    value,
                });
            }
        }
        Ok(Some(IntradayDailyRawSeries {
            spec: raw_spec(),
            values,
        }))
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(RAW_ID)?;
        let risk = panel.column(RAW_ID)?;
        let ex_ret = excess_return_from_raw_panel(&panel, data)?;
        let raw = umr::umr_raw(&risk, &ex_ret, true)?;
        let factor = umr::neutralize_size_sector(&raw, &panel, data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn tags() -> Vec<String> {
    [
        "price_volume",
        "return",
        "skewness",
        "intraday",
        "minute_agg",
        "momentum",
        "reversal",
        "neutralize",
        "barra",
        "size",
        "sector",
        "daily",
        "GXZQ",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect()
}

fn excess_return_from_raw_panel(panel: &DailyPanel, data: &DataPool) -> Result<PanelColumn> {
    let close = panel.column_from_table(data.daily(DatasetId::StockDailyPv)?, "close")?;
    let pre_close = panel.column_from_table(data.daily(DatasetId::StockDailyPv)?, "pre_close")?;
    let stock_ret = close.zip_binary(&pre_close, umr::ret)?;
    let market_ret = umr::expanded_market_return(panel, data, false)?;
    stock_ret.zip_binary(&market_ret, umr::subtract)
}

fn raw_values_from_minute(data: &DataPool, trade_date: i32) -> Result<Vec<(String, Option<f64>)>> {
    let Some(table) = data.minute(DatasetId::StockMinute1m, trade_date) else {
        return Ok(Vec::new());
    };
    let ts_codes = table.required_utf8("ts_code")?;
    let trade_times = table.required_utf8("trade_time")?;
    let close = table.required_f64_cast("close")?;

    let mut grouped = BTreeMap::<String, Vec<usize>>::new();
    for idx in 0..table.len {
        let Some(ts_code) = ts_codes[idx].clone() else {
            continue;
        };
        let Some(trade_time) = trade_times[idx].as_deref() else {
            continue;
        };
        if intraday_time_in_range(trade_time, "09:31:00", "15:00:00") {
            grouped.entry(ts_code).or_default().push(idx);
        }
    }

    let mut output = Vec::new();
    for (ts_code, mut indices) in grouped {
        indices.sort_by(|left, right| trade_times[*left].cmp(&trade_times[*right]));
        output.push((ts_code, five_minute_return_skew(&indices, &close)));
    }
    Ok(output)
}

fn raw_values_from_derived_bar(
    table: &crate::data::Table,
) -> Result<Option<Vec<(String, Option<f64>)>>> {
    if !["ts_code", "bar_index", "minute_count", "close"]
        .iter()
        .all(|column| table.columns.contains_key(*column))
    {
        return Ok(None);
    }
    let ts_codes = table.required_utf8("ts_code")?;
    let bar_indices = table.required_i32("bar_index")?;
    let minute_counts = table.required_i32("minute_count")?;
    let close = table.required_f64_cast("close")?;
    let mut grouped = BTreeMap::<String, [Option<f64>; FIVE_MINUTE_BARS]>::new();
    for idx in 0..table.len {
        let Some(ts_code) = ts_codes[idx].clone() else {
            continue;
        };
        let Some(slot) = bar_indices[idx].and_then(|value| usize::try_from(value).ok()) else {
            continue;
        };
        if slot >= FIVE_MINUTE_BARS || minute_counts[idx] != Some(FIVE_MINUTE_BAR_SIZE as i32) {
            continue;
        }
        grouped.entry(ts_code).or_insert([None; FIVE_MINUTE_BARS])[slot] =
            clean_intraday_value(close[idx]);
    }
    Ok(Some(
        grouped
            .into_iter()
            .map(|(ts_code, closes)| {
                let returns = five_minute_close_returns_from_bar_closes(&closes);
                let value = (returns.len() >= 2).then(|| skew(&returns)).flatten();
                (ts_code, value)
            })
            .collect(),
    ))
}

fn five_minute_return_skew(indices: &[usize], close: &[Option<f64>]) -> Option<f64> {
    let returns = five_minute_close_returns(indices, close);
    if returns.len() < 2 {
        return None;
    }
    skew(&returns)
}

fn five_minute_close_returns(indices: &[usize], close: &[Option<f64>]) -> Vec<f64> {
    let bar_closes = indices
        .iter()
        .enumerate()
        .filter_map(|(pos, idx)| ((pos + 1) % 5 == 0).then(|| clean_intraday_value(close[*idx]))?)
        .collect::<Vec<_>>();
    let mut returns = Vec::new();
    for pair in bar_closes.windows(2) {
        let (prev, curr) = (pair[0], pair[1]);
        if prev.abs() > f64::EPSILON {
            returns.push(curr / prev - 1.0);
        }
    }
    returns
}

fn five_minute_close_returns_from_bar_closes(closes: &[Option<f64>; FIVE_MINUTE_BARS]) -> Vec<f64> {
    let bar_closes = closes
        .iter()
        .filter_map(|value| clean_intraday_value(*value))
        .collect::<Vec<_>>();
    let mut returns = Vec::new();
    for pair in bar_closes.windows(2) {
        let (prev, curr) = (pair[0], pair[1]);
        if prev.abs() > f64::EPSILON {
            returns.push(curr / prev - 1.0);
        }
    }
    returns
}

fn skew(values: &[f64]) -> Option<f64> {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64;
    let std = variance.sqrt();
    if std <= f64::EPSILON {
        return None;
    }
    let third = values
        .iter()
        .map(|value| (value - mean).powi(3))
        .sum::<f64>()
        / values.len() as f64;
    Some(third / std.powi(3))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skew_rejects_constant_returns() {
        assert_eq!(skew(&[1.0, 1.0, 1.0]), None);
    }

    #[test]
    fn umr_minute_skewness_raw_spec_prefers_derived_bar_5m() {
        let spec = raw_spec();
        assert_eq!(spec.source_dataset, DatasetId::StockDerivedBar);
        assert_eq!(spec.source_bar_size, Some(FIVE_MINUTE_BAR_SIZE));
        assert_eq!(spec.columns, vec!["close"]);

        let factor = StockDailyUmrMinuteSkewness;
        let aux = factor.intraday_raw_auxiliary_requirements(&[RAW_ID.to_string()]);
        assert_eq!(aux.len(), 1);
        assert_eq!(aux[0].request.dataset, DatasetId::StockMinute1m);
        assert_eq!(aux[0].request.columns, vec!["close"]);
    }
}
