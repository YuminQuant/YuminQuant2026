use std::collections::BTreeMap;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::{
    clean_intraday_value, intraday_time_in_range, stock_minute_raw_spec, ClassificationLevel,
    ClassificationMap, PanelColumn,
};
use crate::factor::Factor;
use crate::operators::{cs_zscore, ts_delay, ts_mean, ts_std_dev};

pub const DAILY_JUMP_ERROR_RAW_ID: &str = "daily_jump_error";

const RAW_VERSION: &str = "0.1.0";
const WINDOW: usize = 20;

pub struct StockDailyMothToFlame;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyMothToFlame)
}

fn raw_spec() -> IntradayDailyRawSpec {
    stock_minute_raw_spec(DAILY_JUMP_ERROR_RAW_ID, RAW_VERSION, &["close"], 1)
}

impl Factor for StockDailyMothToFlame {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "moth_to_flame".to_string(),
            aliases: Vec::new(),
            name: "Moth to Flame".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: [
                "price_volume",
                "return",
                "volatility",
                "jump",
                "intraday",
                "minute_agg",
                "composite",
                "neutralize",
                "barra",
                "size",
                "sector",
                "daily",
                "FZZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Moth to Flame factor from intraday jump-error raw and daily high/low/close Taylor reversal branches, neutralized by Barra SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close", "high", "low"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(
                DAILY_JUMP_ERROR_RAW_ID,
                WINDOW - 1,
            )],
            lookback: Lookback {
                trading_days: WINDOW - 1,
            },
        }
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        vec![raw_spec()]
    }

    fn minute_compute(
        &self,
        raw_id: &str,
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Option<IntradayDailyRawSeries>> {
        if raw_id != DAILY_JUMP_ERROR_RAW_ID {
            return Ok(None);
        }

        let mut values = Vec::new();
        for trade_date in &context.target_dates {
            let Some(table) = data.minute(DatasetId::StockMinute1m, *trade_date) else {
                continue;
            };
            let ts_codes = table.required_utf8("ts_code")?;
            let trade_times = table.required_utf8("trade_time")?;
            let close = table.required_f64_cast("close")?;
            let mut grouped = BTreeMap::<String, Vec<usize>>::new();
            for idx in 0..table.len {
                let Some(ts_code) = ts_codes[idx].clone() else {
                    continue;
                };
                if trade_times[idx].is_none() {
                    continue;
                }
                grouped.entry(ts_code).or_default().push(idx);
            }

            for (ts_code, mut indices) in grouped {
                indices.sort_by(|left, right| trade_times[*left].cmp(&trade_times[*right]));
                values.push(FactorValue {
                    key: FactorRowKey::Daily {
                        trade_date: *trade_date,
                        ts_code,
                    },
                    value: jump_error_from_rows(&indices, trade_times, &close),
                });
            }
        }

        Ok(Some(IntradayDailyRawSeries {
            spec: raw_spec(),
            values,
        }))
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let sector_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockSwClassification)?,
            ClassificationLevel::Sector,
        )?;
        let panel = data.intraday_daily_raw_panel(DAILY_JUMP_ERROR_RAW_ID)?;
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;
        let daily_jump = panel.column(DAILY_JUMP_ERROR_RAW_ID)?;
        let close = panel.column_from_table(data.daily(DatasetId::StockDailyPv)?, "close")?;
        let high = panel.column_from_table(data.daily(DatasetId::StockDailyPv)?, "high")?;
        let low = panel.column_from_table(data.daily(DatasetId::StockDailyPv)?, "low")?;

        let jump_mean20 = daily_jump.ts(|values| ts_mean(values, WINDOW, 1))?;
        let jump_std20 = daily_jump.ts(|values| ts_std_dev(values, WINDOW, 1))?;
        let monthly_jump =
            average_pair(&jump_mean20.cs(cs_zscore)?, &jump_std20.cs(cs_zscore)?)?.cs(cs_zscore)?;

        let prev_close = close.ts(|values| ts_delay(values, 1))?;
        let applitude = high.zip_ternary(&low, &prev_close, applitude)?;
        let modified_jump1 = applitude
            .zip_binary(&daily_jump.cs(sign_by_cross_mean)?, mul)?
            .ts(|values| ts_mean(values, WINDOW, 1))?
            .cs(cs_zscore)?;

        let prev_low = low.ts(|values| ts_delay(values, 1))?;
        let taylor = high.zip_binary(&prev_low, taylor_value)?;
        let modified_jump2 = applitude
            .zip_binary(&taylor.cs(sign_by_cross_mean)?, mul)?
            .ts(|values| ts_mean(values, WINDOW, 1))?
            .cs(cs_zscore)?;

        let modified_jump = average_pair(&modified_jump1, &modified_jump2)?.cs(cs_zscore)?;
        let raw_factor = monthly_jump
            .zip_binary(&modified_jump, add)?
            .cs(cs_zscore)?
            .map_values(|value| clean(value).map(|value| 0.5 * value));
        let neutralized = raw_factor.cs_neutralize_regression_by_group(
            &[&size],
            None,
            |trade_date, ts_codes| sector_map.groups_for(trade_date, ts_codes),
        )?;

        Ok(neutralized.to_factor_series(self.spec()))
    }
}

fn jump_error_from_rows(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
) -> Option<f64> {
    let close_series = indices
        .iter()
        .filter(|idx| {
            trade_times[**idx]
                .as_deref()
                .is_some_and(|time| intraday_time_in_range(time, "09:31:00", "14:57:00"))
        })
        .map(|idx| clean_intraday_value(close[*idx]))
        .collect::<Vec<_>>();
    jump_error_from_close_series(&close_series)
}

fn jump_error_from_close_series(close: &[Option<f64>]) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for idx in 1..close.len() {
        let (Some(current), Some(previous)) = (clean(close[idx]), clean(close[idx - 1])) else {
            continue;
        };
        let log_return = safe_log_ratio(current, previous)?;
        let simple_return = current / previous - 1.0;
        sum += 2.0 * (simple_return - log_return) - log_return.powi(2);
        count += 1;
    }
    (count > 0).then_some(sum / count as f64)
}

fn sign_by_cross_mean(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let valid = values
        .iter()
        .filter_map(|value| clean(*value))
        .collect::<Vec<_>>();
    if valid.is_empty() {
        return vec![None; values.len()];
    }
    let mean = valid.iter().sum::<f64>() / valid.len() as f64;
    values
        .iter()
        .map(|value| clean(*value).map(|value| if value < mean { -1.0 } else { 1.0 }))
        .collect()
}

fn applitude(high: Option<f64>, low: Option<f64>, prev_close: Option<f64>) -> Option<f64> {
    let (Some(high), Some(low), Some(prev_close)) = (clean(high), clean(low), clean(prev_close))
    else {
        return None;
    };
    if prev_close.abs() <= f64::EPSILON {
        return None;
    }
    Some((high - low) / prev_close)
}

fn taylor_value(high: Option<f64>, prev_low: Option<f64>) -> Option<f64> {
    let (Some(high), Some(prev_low)) = (clean(high), clean(prev_low)) else {
        return None;
    };
    let log_return = safe_log_ratio(high, prev_low)?;
    let simple_return = high / prev_low - 1.0;
    Some(2.0 * (simple_return - log_return) - log_return.powi(2))
}

fn safe_log_ratio(numerator: f64, denominator: f64) -> Option<f64> {
    if numerator <= 0.0 || denominator <= 0.0 {
        return None;
    }
    Some((numerator / denominator).ln())
}

fn average_pair(left: &PanelColumn, right: &PanelColumn) -> Result<PanelColumn> {
    left.zip_binary(right, |left, right| match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some((left + right) / 2.0),
        _ => None,
    })
}

fn mul(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some(left * right),
        _ => None,
    }
}

fn add(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some(left + right),
        _ => None,
    }
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| !value.is_nan())
}

#[cfg(test)]
mod tests {
    use super::{jump_error_from_close_series, safe_log_ratio, sign_by_cross_mean, taylor_value};

    #[test]
    fn jump_error_uses_simple_and_log_returns() {
        let close = vec![Some(100.0), Some(102.0), Some(101.0)];
        let first_log = (102.0_f64 / 100.0).ln();
        let first_simple = 102.0 / 100.0 - 1.0;
        let second_log = (101.0_f64 / 102.0).ln();
        let second_simple = 101.0 / 102.0 - 1.0;
        let expected = (2.0 * (first_simple - first_log) - first_log.powi(2)
            + 2.0 * (second_simple - second_log)
            - second_log.powi(2))
            / 2.0;

        let actual = jump_error_from_close_series(&close).expect("jump error");
        assert!((actual - expected).abs() < 1e-12);
    }

    #[test]
    fn jump_error_rejects_missing_or_non_positive_pairs() {
        assert_eq!(
            jump_error_from_close_series(&[Some(100.0), None, Some(101.0)]),
            None
        );
        assert_eq!(
            jump_error_from_close_series(&[Some(100.0), Some(0.0), Some(101.0)]),
            None
        );
        assert_eq!(safe_log_ratio(1.0, 0.0), None);
    }

    #[test]
    fn sign_by_cross_mean_sets_below_mean_to_negative_one() {
        let output = sign_by_cross_mean(&[Some(1.0), Some(3.0), Some(2.0), None]);
        assert_eq!(output, vec![Some(-1.0), Some(1.0), Some(1.0), None]);
    }

    #[test]
    fn taylor_value_matches_formula() {
        let log_return = (110.0_f64 / 100.0).ln();
        let simple_return = 110.0 / 100.0 - 1.0;
        let expected = 2.0 * (simple_return - log_return) - log_return.powi(2);
        let actual = taylor_value(Some(110.0), Some(100.0)).expect("taylor");
        assert!((actual - expected).abs() < 1e-12);
    }
}
