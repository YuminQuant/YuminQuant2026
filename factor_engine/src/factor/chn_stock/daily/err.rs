use std::collections::{BTreeMap, BTreeSet};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::neutralize_size_sector;
use crate::factor::common::{clean_intraday_value, stock_minute_raw_spec};
use crate::factor::Factor;
use crate::operators::{cs_pctrank, ts_mean};

const VERSION: &str = "0.1.0";
const RAW_VERSION: &str = "0.1.0";
const PROVIDER_KEY: &str = "kyzq_err_provider";
const WINDOW: usize = 20;
const RAW_WINDOW_DAYS: usize = 1;

const EXTREME_RETURN_RAW_ID: &str = "daily_kyzq_err_extreme_return_raw";
const PREV_RETURN_RAW_ID: &str = "daily_kyzq_err_prev_return_raw";

pub struct StockDailyErr;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyErr)
}

impl Factor for StockDailyErr {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "err".to_string(),
            aliases: vec!["ERR".to_string()],
            name: "err".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "KYZQ ERR factor from daily extreme 1-minute return and previous-minute return cross-sectional percentile ranks, 20-day mean, neutralized by Barra SIZE and SW sector. Negative report direction is preserved.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: all_raw_ids()
                .iter()
                .map(|raw_id| IntradayDailyRawRequest::new(raw_id, WINDOW - 1))
                .collect(),
            lookback: Lookback {
                trading_days: WINDOW - 1,
            },
        }
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        all_raw_ids()
            .iter()
            .map(|raw_id| raw_spec(raw_id))
            .collect()
    }

    fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
        PROVIDER_KEY.to_string()
    }

    fn minute_compute_many(
        &self,
        raw_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<IntradayDailyRawSeries>> {
        let requested = raw_ids
            .iter()
            .map(String::as_str)
            .filter(|raw_id| all_raw_ids().contains(raw_id))
            .collect::<BTreeSet<_>>();
        if requested.is_empty() {
            return Ok(Vec::new());
        }

        let mut values = all_raw_ids()
            .iter()
            .map(|raw_id| (*raw_id, Vec::<FactorValue>::new()))
            .collect::<BTreeMap<_, _>>();

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
                let raw = err_daily_raw_from_indices(&indices, &trade_times, &close);
                let key = FactorRowKey::Daily {
                    trade_date: *trade_date,
                    ts_code,
                };
                push_requested(
                    &mut values,
                    &requested,
                    EXTREME_RETURN_RAW_ID,
                    &key,
                    raw.extreme_return,
                );
                push_requested(
                    &mut values,
                    &requested,
                    PREV_RETURN_RAW_ID,
                    &key,
                    raw.previous_return,
                );
            }
        }

        let mut output = Vec::new();
        for raw_id in all_raw_ids() {
            if requested.contains(raw_id) {
                output.push(IntradayDailyRawSeries {
                    spec: raw_spec(raw_id),
                    values: values.remove(raw_id).unwrap_or_default(),
                });
            }
        }
        Ok(output)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(EXTREME_RETURN_RAW_ID)?;
        let extreme = panel.column(EXTREME_RETURN_RAW_ID)?;
        let previous = panel.column(PREV_RETURN_RAW_ID)?;
        let extreme_rank = extreme.cs(|values| cs_pctrank(values, true))?;
        let previous_rank = previous.cs(|values| cs_pctrank(values, true))?;
        let daily_signal = extreme_rank.zip_binary(&previous_rank, add_pair)?;
        let raw = daily_signal.ts(|series| ts_mean(series, WINDOW, 1))?;
        let factor = neutralize_size_sector(&raw, &panel, data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct ErrRaw {
    extreme_return: Option<f64>,
    previous_return: Option<f64>,
}

fn all_raw_ids() -> [&'static str; 2] {
    [EXTREME_RETURN_RAW_ID, PREV_RETURN_RAW_ID]
}

fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
    stock_minute_raw_spec(raw_id, RAW_VERSION, &["close"], RAW_WINDOW_DAYS)
}

fn tags() -> Vec<String> {
    [
        "KYZQ",
        "return",
        "extreme",
        "intraday",
        "minute_agg",
        "neutralize",
        "barra",
        "size",
        "sector",
        "daily",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect()
}

fn err_daily_raw_from_indices(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
) -> ErrRaw {
    let returns = minute_returns_from_indices(indices, trade_times, close);
    err_daily_raw_from_returns(&returns)
}

fn minute_returns_from_indices(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
) -> [Option<f64>; 240] {
    let mut output = [None; 240];
    let mut previous_close = None;
    for idx in indices {
        let current_close = clean_intraday_value(close[*idx]).filter(|value| *value > 0.0);
        if let Some(trade_time) = trade_times[*idx].as_deref() {
            if let Some(minute_idx) = minute_index(trade_time) {
                output[minute_idx] = minute_return(previous_close, current_close);
            }
        }
        if current_close.is_some() {
            previous_close = current_close;
        }
    }
    output
}

fn err_daily_raw_from_returns(returns: &[Option<f64>; 240]) -> ErrRaw {
    let valid = returns
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| clean_intraday_value(*value).map(|value| (idx, value)))
        .collect::<Vec<_>>();
    if valid.is_empty() {
        return ErrRaw::default();
    }
    let median = median(valid.iter().map(|(_, value)| *value).collect::<Vec<_>>());
    let Some((min_idx, min_value)) = valid
        .iter()
        .min_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.0.cmp(&right.0))
        })
        .copied()
    else {
        return ErrRaw::default();
    };
    let Some((max_idx, max_value)) = valid
        .iter()
        .max_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| right.0.cmp(&left.0))
        })
        .copied()
    else {
        return ErrRaw::default();
    };
    let min_distance = (min_value - median).abs();
    let max_distance = (max_value - median).abs();
    let (extreme_idx, extreme_return) = if max_distance > min_distance {
        (max_idx, max_value)
    } else if min_distance > max_distance {
        (min_idx, min_value)
    } else if min_idx <= max_idx {
        (min_idx, min_value)
    } else {
        (max_idx, max_value)
    };
    let previous_return = extreme_idx.checked_sub(1).and_then(|idx| returns[idx]);
    ErrRaw {
        extreme_return: Some(extreme_return),
        previous_return,
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    }
}

fn minute_return(previous_close: Option<f64>, current_close: Option<f64>) -> Option<f64> {
    let (Some(previous), Some(current)) = (previous_close, current_close) else {
        return None;
    };
    if previous.abs() <= f64::EPSILON {
        return None;
    }
    let value = current / previous - 1.0;
    value.is_finite().then_some(value)
}

fn minute_index(trade_time: &str) -> Option<usize> {
    let minutes = time_to_minutes(trade_time)?;
    let morning_start = 9 * 60 + 31;
    let morning_end = 11 * 60 + 30;
    let afternoon_start = 13 * 60 + 1;
    let afternoon_end = 15 * 60;
    if (morning_start..=morning_end).contains(&minutes) {
        return Some((minutes - morning_start) as usize);
    }
    if (afternoon_start..=afternoon_end).contains(&minutes) {
        return Some(120 + (minutes - afternoon_start) as usize);
    }
    None
}

fn time_to_minutes(value: &str) -> Option<i32> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let time = value
        .rsplit_once(' ')
        .map(|(_, right)| right)
        .or_else(|| value.rsplit_once('T').map(|(_, right)| right))
        .unwrap_or(value)
        .trim();
    if time.len() < 5 {
        return None;
    }
    let hour = time.get(0..2)?.parse::<i32>().ok()?;
    let minute = time.get(3..5)?.parse::<i32>().ok()?;
    Some(hour * 60 + minute)
}

fn add_pair(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (clean_intraday_value(left), clean_intraday_value(right)) {
        (Some(left), Some(right)) => Some(left + right),
        _ => None,
    }
}

fn push_requested(
    values: &mut BTreeMap<&'static str, Vec<FactorValue>>,
    requested: &BTreeSet<&str>,
    raw_id: &'static str,
    key: &FactorRowKey,
    value: Option<f64>,
) {
    if requested.contains(raw_id) {
        values
            .get_mut(raw_id)
            .expect("raw id initialized")
            .push(FactorValue {
                key: key.clone(),
                value,
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("value");
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn kyzq_err_selects_extreme_by_distance_to_median() {
        let mut returns = [None; 240];
        returns[0] = Some(-0.02);
        returns[1] = Some(0.0);
        returns[2] = Some(0.03);

        let raw = err_daily_raw_from_returns(&returns);

        assert_close(raw.extreme_return, 0.03);
        assert_close(raw.previous_return, 0.0);
    }

    #[test]
    fn kyzq_err_first_extreme_minute_has_no_previous_return() {
        let mut returns = [None; 240];
        returns[0] = Some(-0.10);
        returns[1] = Some(0.01);
        returns[2] = Some(0.02);

        let raw = err_daily_raw_from_returns(&returns);

        assert_close(raw.extreme_return, -0.10);
        assert_eq!(raw.previous_return, None);
    }

    #[test]
    fn kyzq_err_minute_index_uses_regular_session_numbering() {
        assert_eq!(minute_index("09:31:00"), Some(0));
        assert_eq!(minute_index("11:30:00"), Some(119));
        assert_eq!(minute_index("13:01:00"), Some(120));
        assert_eq!(minute_index("15:00:00"), Some(239));
        assert_eq!(minute_index("09:30:00"), None);
    }

    #[test]
    fn kyzq_err_factor_spec_has_kyzq_tag() {
        let spec = StockDailyErr.spec();
        assert_eq!(spec.id, "err");
        assert!(spec.tags.iter().any(|tag| tag == "KYZQ"));
    }
}
