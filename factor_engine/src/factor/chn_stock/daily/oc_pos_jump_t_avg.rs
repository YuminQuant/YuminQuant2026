use std::collections::BTreeMap;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawAuxiliaryRequest, IntradayDailyRawRequest,
    IntradayDailyRawSeries, IntradayDailyRawSpec, Lookback,
};
use crate::data::{DataPool, Table};
use crate::error::Result;
use crate::factor::common::stock_daily_ops::neutralize_size_sector;
use crate::factor::common::{clean_intraday_value, stock_minute_raw_spec};
use crate::factor::Factor;
use crate::operators::ts_ewm;

const VERSION: &str = "0.1.0";
const RAW_VERSION: &str = "0.1.0";
const RAW_ID: &str = "daily_zszq_oc_pos_jump_t_raw";
const PROVIDER_KEY: &str = "zszq_oc_pos_jump_t_provider";

const INTERVALS: usize = 49;
const ENDPOINTS: usize = 49;
const EMA_SPAN: usize = 20;
const MIN_PERIODS: usize = 1;
const JS_THRESHOLD: f64 = 1.96;
const EPS: f64 = 1e-12;

pub struct StockDailyOcPosJumpTAvg;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyOcPosJumpTAvg)
}

impl Factor for StockDailyOcPosJumpTAvg {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "oc_pos_jump_t_avg".to_string(),
            aliases: vec!["OC_Pos_JumpT_Avg".to_string()],
            name: "oc_pos_jump_t_avg".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "ZSZQ average turnover-weighted positive intraday jump return factor. A 49-interval swap-variance jump test identifies overnight plus 5-minute intraday jumps; positive intraday log jump returns are summed, multiplied by free-float turnover, EMA20 smoothed, and neutralized by Barra SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(RAW_ID, EMA_SPAN - 1)],
            lookback: Lookback {
                trading_days: EMA_SPAN - 1,
            },
        }
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        vec![raw_spec()]
    }

    fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
        PROVIDER_KEY.to_string()
    }

    fn intraday_raw_auxiliary_requirements(
        &self,
        raw_ids: &[String],
    ) -> Vec<IntradayDailyRawAuxiliaryRequest> {
        if !raw_ids.iter().any(|raw_id| raw_id == RAW_ID) {
            return Vec::new();
        }
        vec![
            IntradayDailyRawAuxiliaryRequest::new(
                DataRequest::new(DatasetId::StockDailyPv, &["pre_close"]),
                0,
            ),
            IntradayDailyRawAuxiliaryRequest::new(
                DataRequest::new(DatasetId::StockDailyBasic, &["turnover_rate_f"]),
                0,
            ),
        ]
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

        let pre_close = daily_lookup(data.daily(DatasetId::StockDailyPv)?, "pre_close")?;
        let turnover = daily_lookup(data.daily(DatasetId::StockDailyBasic)?, "turnover_rate_f")?;

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

            for (ts_code, indices) in grouped {
                let key = (*trade_date, ts_code.clone());
                let value = match (pre_close.get(&key).copied(), turnover.get(&key).copied()) {
                    (Some(Some(pre_close)), Some(Some(turnover))) => {
                        let endpoints = endpoints_from_rows(&indices, trade_times, &close);
                        oc_pos_jump_turnover_value(pre_close, &endpoints, turnover)
                    }
                    _ => None,
                };
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
        let raw = panel.column(RAW_ID)?;
        let smoothed = raw.ts(|series| ts_ewm(series, EMA_SPAN, MIN_PERIODS))?;
        let factor = neutralize_size_sector(&smoothed, &panel, data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn raw_spec() -> IntradayDailyRawSpec {
    stock_minute_raw_spec(RAW_ID, RAW_VERSION, &["close"], 1)
}

fn tags() -> Vec<String> {
    [
        "price_volume",
        "ZSZQ",
        "jump",
        "swap_variance",
        "positive_jump",
        "turnover",
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

fn daily_lookup(table: &Table, column: &str) -> Result<BTreeMap<(i32, String), Option<f64>>> {
    let trade_dates = table.required_i32("trade_date")?;
    let ts_codes = table.required_utf8("ts_code")?;
    let values = table.required_f64_cast(column)?;
    let mut output = BTreeMap::new();
    for idx in 0..table.len {
        let (Some(trade_date), Some(ts_code)) = (trade_dates[idx], ts_codes[idx].clone()) else {
            continue;
        };
        output.insert((trade_date, ts_code), clean_intraday_value(values[idx]));
    }
    Ok(output)
}

fn endpoints_from_rows(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
) -> [Option<f64>; ENDPOINTS] {
    let mut endpoints = [None; ENDPOINTS];
    for idx in indices {
        let Some(trade_time) = trade_times[*idx].as_deref() else {
            continue;
        };
        let Some(slot) = endpoint_slot(trade_time) else {
            continue;
        };
        endpoints[slot] = clean_intraday_value(close[*idx]).filter(|value| *value > 0.0);
    }
    endpoints
}

fn endpoint_slot(trade_time: &str) -> Option<usize> {
    if time_to_minutes(trade_time) == Some(9 * 60 + 30) {
        return Some(0);
    }
    let minute_idx = minute_index(trade_time)?;
    if (minute_idx + 1) % 5 == 0 {
        Some((minute_idx + 1) / 5)
    } else {
        None
    }
}

fn minute_index(trade_time: &str) -> Option<usize> {
    let total = time_to_minutes(trade_time)?;
    let morning_start = 9 * 60 + 31;
    let morning_end = 11 * 60 + 30;
    let afternoon_start = 13 * 60 + 1;
    let afternoon_end = 15 * 60;
    if (morning_start..=morning_end).contains(&total) {
        Some((total - morning_start) as usize)
    } else if (afternoon_start..=afternoon_end).contains(&total) {
        Some(120 + (total - afternoon_start) as usize)
    } else {
        None
    }
}

fn time_to_minutes(value: &str) -> Option<i32> {
    let value = value.trim();
    let value = value
        .rsplit_once(' ')
        .map(|(_, right)| right)
        .or_else(|| value.rsplit_once('T').map(|(_, right)| right))
        .unwrap_or(value)
        .trim();
    if value.len() < 5 {
        return None;
    }
    let hour = value.get(0..2)?.parse::<i32>().ok()?;
    let minute = value.get(3..5)?.parse::<i32>().ok()?;
    let total = hour * 60 + minute;
    Some(total)
}

fn oc_pos_jump_turnover_value(
    pre_close: f64,
    endpoints: &[Option<f64>; ENDPOINTS],
    turnover_rate_f: f64,
) -> Option<f64> {
    if !pre_close.is_finite() || pre_close <= EPS {
        return None;
    }
    let turnover = clean_intraday_value(Some(turnover_rate_f))?;
    let returns = interval_returns(pre_close, endpoints)?;
    let positive_intraday_jump = positive_intraday_jump_return(&returns);
    Some(positive_intraday_jump * turnover / 100.0)
}

fn interval_returns(
    pre_close: f64,
    endpoints: &[Option<f64>; ENDPOINTS],
) -> Option<[(f64, f64); INTERVALS]> {
    let mut prices = [0.0; ENDPOINTS + 1];
    prices[0] = pre_close;
    for (idx, value) in endpoints.iter().enumerate() {
        let value = (*value)?;
        if !value.is_finite() || value <= EPS {
            return None;
        }
        prices[idx + 1] = value;
    }

    let mut returns = [(0.0, 0.0); INTERVALS];
    for idx in 0..INTERVALS {
        let previous = prices[idx];
        let current = prices[idx + 1];
        if previous <= EPS || current <= EPS {
            return None;
        }
        let simple = current / previous - 1.0;
        let log_ret = (current / previous).ln();
        if !simple.is_finite() || !log_ret.is_finite() {
            return None;
        }
        returns[idx] = (simple, log_ret);
    }
    Some(returns)
}

fn positive_intraday_jump_return(raw_returns: &[(f64, f64); INTERVALS]) -> f64 {
    let mut current = *raw_returns;
    let mut output = 0.0;
    for _ in 0..INTERVALS {
        let parts = JumpStatsParts::from_returns(&current);
        let js0 = parts.js();
        if !js0.is_finite() || js0.abs() <= JS_THRESHOLD {
            break;
        }
        let simple_median = median_simple(&current);
        let log_median = median_log(&current);
        let Some((jump_idx, _)) =
            best_jump_replacement(&current, &parts, js0, simple_median, log_median)
        else {
            break;
        };
        if jump_idx > 0 {
            let raw_log_ret = raw_returns[jump_idx].1;
            if raw_log_ret > 0.0 && raw_log_ret.is_finite() {
                output += raw_log_ret;
            }
        }
        current[jump_idx] = (simple_median, log_median);
    }
    output
}

fn best_jump_replacement(
    returns: &[(f64, f64); INTERVALS],
    parts: &JumpStatsParts,
    js0: f64,
    simple_median: f64,
    log_median: f64,
) -> Option<(usize, f64)> {
    let mut best = None;
    for idx in 0..INTERVALS {
        let js_i = parts
            .with_replacement(returns, idx, simple_median, log_median)
            .js();
        if !js_i.is_finite() {
            continue;
        }
        let contribution = js0.abs() - js_i.abs();
        if !contribution.is_finite() {
            continue;
        }
        let better = match best {
            Some((_, best_value)) => contribution > best_value + EPS,
            None => true,
        };
        if better {
            best = Some((idx, contribution));
        }
    }
    best
}


#[derive(Clone, Copy, Debug)]
struct JumpStatsParts {
    rv: f64,
    swv: f64,
    bipower_adjacent_sum: f64,
    omega_six_product_sum: f64,
}

impl JumpStatsParts {
    fn from_returns(returns: &[(f64, f64); INTERVALS]) -> Self {
        let mut parts = Self {
            rv: 0.0,
            swv: 0.0,
            bipower_adjacent_sum: 0.0,
            omega_six_product_sum: 0.0,
        };
        for (simple, log_ret) in returns {
            parts.rv += log_ret * log_ret;
            parts.swv += 2.0 * (simple - log_ret);
        }
        for start in 0..INTERVALS.saturating_sub(1) {
            parts.bipower_adjacent_sum += adjacent_abs_product(returns, start);
        }
        for start in 0..=INTERVALS - 6 {
            parts.omega_six_product_sum += six_abs_product(returns, start);
        }
        parts
    }

    fn with_replacement(
        &self,
        returns: &[(f64, f64); INTERVALS],
        idx: usize,
        new_simple: f64,
        new_log: f64,
    ) -> Self {
        let (old_simple, old_log) = returns[idx];
        let mut next = *self;
        next.rv += new_log * new_log - old_log * old_log;
        next.swv += 2.0 * ((new_simple - new_log) - (old_simple - old_log));

        let adjacent_start = idx.saturating_sub(1);
        let adjacent_end = idx.min(INTERVALS - 2);
        for start in adjacent_start..=adjacent_end {
            next.bipower_adjacent_sum -= adjacent_abs_product(returns, start);
            next.bipower_adjacent_sum +=
                adjacent_abs_product_with_replacement(returns, start, idx, new_log);
        }

        let six_start = idx.saturating_sub(5);
        let six_end = idx.min(INTERVALS - 6);
        for start in six_start..=six_end {
            next.omega_six_product_sum -= six_abs_product(returns, start);
            next.omega_six_product_sum +=
                six_abs_product_with_replacement(returns, start, idx, new_log);
        }
        next
    }

    fn js(&self) -> f64 {
        let bipower = self.bipower_adjacent_sum / mu1_square();
        let omega = omega_multiplier() * self.omega_six_product_sum;
        if self.swv <= EPS
            || omega <= EPS
            || !self.rv.is_finite()
            || !self.swv.is_finite()
            || !bipower.is_finite()
        {
            return 0.0;
        }
        let value = INTERVALS as f64 * bipower / omega.sqrt() * (1.0 - self.rv / self.swv);
        value.is_finite().then_some(value).unwrap_or(0.0)
    }
}

fn adjacent_abs_product(returns: &[(f64, f64); INTERVALS], start: usize) -> f64 {
    returns[start].1.abs() * returns[start + 1].1.abs()
}

fn adjacent_abs_product_with_replacement(
    returns: &[(f64, f64); INTERVALS],
    start: usize,
    replacement_idx: usize,
    replacement_log: f64,
) -> f64 {
    let left = if start == replacement_idx {
        replacement_log
    } else {
        returns[start].1
    };
    let right_idx = start + 1;
    let right = if right_idx == replacement_idx {
        replacement_log
    } else {
        returns[right_idx].1
    };
    left.abs() * right.abs()
}

fn six_abs_product(returns: &[(f64, f64); INTERVALS], start: usize) -> f64 {
    (start..start + 6)
        .map(|idx| returns[idx].1.abs())
        .product::<f64>()
}

fn six_abs_product_with_replacement(
    returns: &[(f64, f64); INTERVALS],
    start: usize,
    replacement_idx: usize,
    replacement_log: f64,
) -> f64 {
    (start..start + 6)
        .map(|idx| {
            if idx == replacement_idx {
                replacement_log
            } else {
                returns[idx].1
            }
            .abs()
        })
        .product::<f64>()
}

fn mu1_square() -> f64 {
    2.0 / std::f64::consts::PI
}

fn omega_multiplier() -> f64 {
    let n = INTERVALS as f64;
    let mu6 = 15.0;
    let mu1_inv6 = (std::f64::consts::PI / 2.0).powi(3);
    mu6 / 9.0 * n.powi(3) * mu1_inv6 / (n - 5.0)
}



fn median_simple(returns: &[(f64, f64); INTERVALS]) -> f64 {
    let mut values = [0.0; INTERVALS];
    for (idx, (simple, _)) in returns.iter().enumerate() {
        values[idx] = *simple;
    }
    median_array(values)
}

fn median_log(returns: &[(f64, f64); INTERVALS]) -> f64 {
    let mut values = [0.0; INTERVALS];
    for (idx, (_, log_ret)) in returns.iter().enumerate() {
        values[idx] = *log_ret;
    }
    median_array(values)
}

fn median_array(mut values: [f64; INTERVALS]) -> f64 {
    values.sort_by(|left, right| left.total_cmp(right));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}
