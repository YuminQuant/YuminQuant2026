use std::collections::{BTreeMap, BTreeSet};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::neutralize_size_sector;
use crate::factor::common::stock_daily_raw_ids::{
    FLASH_CRASH_PROB_RAW_ID, HIGH_STD_RTN_MEAN_RAW_ID, RTN_COND_VAR_RAW_ID, RTN_DW_RAW_ID,
    RTN_FOC_RAW_ID, RTN_LBQ_RAW_ID, RTN_RHO_RAW_ID, VOL_DW_RAW_ID, VOL_FOC_RAW_ID, VOL_LBQ_RAW_ID,
    VOL_RHO_RAW_ID,
};
use crate::factor::common::{clean_intraday_value, intraday_time_in_range, stock_minute_raw_spec};
use crate::operators::{cs_pctrank, ts_mean, ts_std_dev};

pub const RAW_VERSION: &str = "0.1.0";
pub const VERSION: &str = "0.1.0";

const RAW_WINDOW_DAYS: usize = 1;
const SMOOTH_WINDOW: usize = 15;
const SHARED_RAW_LOOKBACK: usize = SMOOTH_WINDOW - 1;
const MIN_PERIODS: usize = 1;
const START_TIME: &str = "09:41:00";
const END_TIME: &str = "14:50:00";
const HIGH_STD_WINDOW: usize = 30;
const LBQ_MAX_LAG: usize = 10;
const EPS: f64 = f64::EPSILON;

#[derive(Clone, Copy, Debug)]
pub enum XyzqSerialAggregation {
    Mean,
    Std,
}

#[derive(Clone, Copy, Debug)]
pub struct XyzqSerialFactorDef {
    pub id: &'static str,
    pub alias: &'static str,
    pub name: &'static str,
    pub raw_id: &'static str,
    pub aggregation: XyzqSerialAggregation,
}

#[derive(Clone, Copy, Debug, Default)]
struct SerialStats {
    rtn_foc: Option<f64>,
    vol_foc: Option<f64>,
    rtn_dw: Option<f64>,
    vol_dw: Option<f64>,
    rtn_rho: Option<f64>,
    vol_rho: Option<f64>,
    rtn_lbq: Option<f64>,
    vol_lbq: Option<f64>,
    high_std_rtn_mean: Option<f64>,
    rtn_cond_var: Option<f64>,
    lambda_pos: Option<f64>,
    lambda_neg: Option<f64>,
    flash_crash_prob: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
struct MinutePoint {
    in_window: bool,
    close: Option<f64>,
    vol: Option<f64>,
}

pub fn all_raw_ids() -> [&'static str; 11] {
    [
        RTN_FOC_RAW_ID,
        VOL_FOC_RAW_ID,
        RTN_DW_RAW_ID,
        VOL_DW_RAW_ID,
        RTN_RHO_RAW_ID,
        VOL_RHO_RAW_ID,
        RTN_LBQ_RAW_ID,
        VOL_LBQ_RAW_ID,
        HIGH_STD_RTN_MEAN_RAW_ID,
        RTN_COND_VAR_RAW_ID,
        FLASH_CRASH_PROB_RAW_ID,
    ]
}

pub fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
    stock_minute_raw_spec(raw_id, RAW_VERSION, &["close", "vol"], RAW_WINDOW_DAYS)
}

pub fn raw_specs() -> Vec<IntradayDailyRawSpec> {
    all_raw_ids()
        .iter()
        .map(|raw_id| raw_spec(raw_id))
        .collect()
}

pub fn factor_spec(def: XyzqSerialFactorDef) -> FactorSpec {
    FactorSpec {
        id: def.id.to_string(),
        aliases: vec![def.alias.to_string()],
        name: def.name.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: tags(),
        description: format!(
            "{} from intraday serial-structure raw, rolling aggregation, cross-sectional percentile rank, and SIZE/SW-sector neutralization.",
            def.name
        ),
        dependencies: dependencies(),
        intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(
            def.raw_id,
            SHARED_RAW_LOOKBACK,
        )],
        lookback: Lookback {
            trading_days: SHARED_RAW_LOOKBACK,
        },
    }
}

pub fn compute_factor(def: XyzqSerialFactorDef, data: &DataPool) -> Result<FactorSeries> {
    let panel = data.intraday_daily_raw_panel(def.raw_id)?;
    let raw = panel.column(def.raw_id)?;
    let aggregated = match def.aggregation {
        XyzqSerialAggregation::Mean => {
            raw.ts(|values| ts_mean(values, SMOOTH_WINDOW, MIN_PERIODS))?
        }
        XyzqSerialAggregation::Std => {
            raw.ts(|values| ts_std_dev(values, SMOOTH_WINDOW, MIN_PERIODS))?
        }
    };
    let ranked = aggregated.cs(|values| cs_pctrank(values, true))?;
    let factor = neutralize_size_sector(&ranked, &panel, data)?;
    Ok(factor.to_factor_series(factor_spec(def)))
}

#[macro_export]
macro_rules! define_xyzq_serial_structure_factor {
    ($struct_name:ident, $id:expr, $alias:expr, $name:expr, $raw_id:expr, $aggregation:ident) => {
        const DEF: $crate::factor::common::xyzq_serial_structure::XyzqSerialFactorDef =
            $crate::factor::common::xyzq_serial_structure::XyzqSerialFactorDef {
                id: $id,
                alias: $alias,
                name: $name,
                raw_id: $raw_id,
                aggregation:
                    $crate::factor::common::xyzq_serial_structure::XyzqSerialAggregation::$aggregation,
            };

        pub struct $struct_name;

        pub fn create() -> Box<dyn $crate::factor::Factor> {
            Box::new($struct_name)
        }

        impl $crate::factor::Factor for $struct_name {
            fn spec(&self) -> $crate::core::FactorSpec {
                $crate::factor::common::xyzq_serial_structure::factor_spec(DEF)
            }

            fn compute(
                &self,
                _context: &$crate::core::FactorContext,
                data: &$crate::data::DataPool,
            ) -> $crate::error::Result<$crate::core::FactorSeries> {
                $crate::factor::common::xyzq_serial_structure::compute_factor(DEF, data)
            }
        }
    };
}

pub fn minute_compute_many(
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
        let vol = table.required_f64_cast("vol")?;

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

        let mut per_stock = Vec::<(FactorRowKey, SerialStats)>::new();
        for (ts_code, mut indices) in grouped {
            indices.sort_by(|left, right| trade_times[*left].cmp(&trade_times[*right]));
            let points = minute_points_from_indices(&indices, &trade_times, &close, &vol);
            let stats = serial_stats(&points);
            let key = FactorRowKey::Daily {
                trade_date: *trade_date,
                ts_code,
            };
            per_stock.push((key, stats));
        }

        fill_flash_crash_probabilities(&mut per_stock);

        for (key, stats) in per_stock {
            push_requested(&mut values, &requested, RTN_FOC_RAW_ID, &key, stats.rtn_foc);
            push_requested(&mut values, &requested, VOL_FOC_RAW_ID, &key, stats.vol_foc);
            push_requested(&mut values, &requested, RTN_DW_RAW_ID, &key, stats.rtn_dw);
            push_requested(&mut values, &requested, VOL_DW_RAW_ID, &key, stats.vol_dw);
            push_requested(&mut values, &requested, RTN_RHO_RAW_ID, &key, stats.rtn_rho);
            push_requested(&mut values, &requested, VOL_RHO_RAW_ID, &key, stats.vol_rho);
            push_requested(&mut values, &requested, RTN_LBQ_RAW_ID, &key, stats.rtn_lbq);
            push_requested(&mut values, &requested, VOL_LBQ_RAW_ID, &key, stats.vol_lbq);
            push_requested(
                &mut values,
                &requested,
                HIGH_STD_RTN_MEAN_RAW_ID,
                &key,
                stats.high_std_rtn_mean,
            );
            push_requested(
                &mut values,
                &requested,
                RTN_COND_VAR_RAW_ID,
                &key,
                stats.rtn_cond_var,
            );
            push_requested(
                &mut values,
                &requested,
                FLASH_CRASH_PROB_RAW_ID,
                &key,
                stats.flash_crash_prob,
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

fn tags() -> Vec<String> {
    [
        "price_volume",
        "return",
        "volume",
        "intraday",
        "minute_agg",
        "serial_correlation",
        "neutralize",
        "barra",
        "size",
        "sector",
        "daily",
        "XYZQ",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect()
}

fn dependencies() -> Vec<DataRequest> {
    vec![
        DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
        DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
    ]
}

fn push_requested(
    values: &mut BTreeMap<&'static str, Vec<FactorValue>>,
    requested: &BTreeSet<&str>,
    raw_id: &'static str,
    key: &FactorRowKey,
    value: Option<f64>,
) {
    if requested.contains(raw_id) {
        values.entry(raw_id).or_default().push(FactorValue {
            key: key.clone(),
            value,
        });
    }
}

fn minute_points_from_indices(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
    vol: &[Option<f64>],
) -> Vec<MinutePoint> {
    indices
        .iter()
        .map(|idx| {
            let in_window = trade_times[*idx]
                .as_deref()
                .is_some_and(|time| intraday_time_in_range(time, START_TIME, END_TIME));
            MinutePoint {
                in_window,
                close: clean_intraday_value(close[*idx]).filter(|value| *value > 0.0),
                vol: clean_intraday_value(vol[*idx]).filter(|value| *value >= 0.0),
            }
        })
        .collect()
}

fn serial_stats(points: &[MinutePoint]) -> SerialStats {
    let returns = simple_returns(points);
    let selected_positions = points
        .iter()
        .enumerate()
        .filter_map(|(idx, point)| point.in_window.then_some(idx))
        .collect::<Vec<_>>();
    let selected_returns = selected_positions
        .iter()
        .map(|idx| returns[*idx])
        .collect::<Vec<_>>();
    let selected_close = selected_positions
        .iter()
        .map(|idx| points[*idx].close)
        .collect::<Vec<_>>();
    let selected_vol = selected_positions
        .iter()
        .map(|idx| points[*idx].vol)
        .collect::<Vec<_>>();
    let vol_pct = volume_pct(&selected_vol);

    let (lambda_pos, lambda_neg) = run_length_lambdas(&selected_returns);
    SerialStats {
        rtn_foc: lag_corr(&selected_returns),
        vol_foc: lag_corr(&vol_pct),
        rtn_dw: durbin_watson(&selected_returns),
        vol_dw: durbin_watson(&vol_pct),
        rtn_rho: residual_ar1_rho(&selected_returns),
        vol_rho: residual_ar1_rho(&vol_pct),
        rtn_lbq: lbq_std(&selected_returns),
        vol_lbq: lbq_std(&vol_pct),
        high_std_rtn_mean: high_std_rtn_mean(points, &returns),
        rtn_cond_var: cond_var(&selected_close),
        lambda_pos,
        lambda_neg,
        flash_crash_prob: None,
    }
}

fn simple_returns(points: &[MinutePoint]) -> Vec<Option<f64>> {
    let mut returns = vec![None; points.len()];
    for idx in 1..points.len() {
        returns[idx] = match (points[idx].close, points[idx - 1].close) {
            (Some(current), Some(previous)) if previous.abs() > EPS => {
                finite_value(current / previous - 1.0)
            }
            _ => None,
        };
    }
    returns
}

fn volume_pct(volumes: &[Option<f64>]) -> Vec<Option<f64>> {
    let total = volumes.iter().filter_map(|value| *value).sum::<f64>();
    if total <= EPS {
        return vec![None; volumes.len()];
    }
    volumes
        .iter()
        .map(|value| value.and_then(|vol| finite_value(vol / total)))
        .collect()
}

fn lag_corr(values: &[Option<f64>]) -> Option<f64> {
    let pairs = lag_pairs(values);
    pearson_pairs(&pairs)
}

fn durbin_watson(values: &[Option<f64>]) -> Option<f64> {
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    let mut count = 0usize;
    for idx in 1..values.len() {
        if let (Some(current), Some(previous)) = (values[idx], values[idx - 1]) {
            numerator += (current - previous).powi(2);
            denominator += current.powi(2);
            count += 1;
        }
    }
    if count == 0 || denominator <= EPS {
        return None;
    }
    finite_value(numerator / denominator)
}

fn residual_ar1_rho(values: &[Option<f64>]) -> Option<f64> {
    let pairs = lag_pairs(values);
    let (_, _, residuals) = ols_with_intercept(&pairs)?;
    let residual_pairs = residuals
        .windows(2)
        .map(|pair| (pair[0], pair[1]))
        .collect::<Vec<_>>();
    let (_, slope, _) = ols_with_intercept(&residual_pairs)?;
    finite_value(slope)
}

fn lbq_std(values: &[Option<f64>]) -> Option<f64> {
    let series = values.iter().filter_map(|value| *value).collect::<Vec<_>>();
    if series.len() <= LBQ_MAX_LAG {
        return None;
    }
    let n = series.len() as f64;
    let mut cumulative = 0.0;
    let mut q_values = Vec::with_capacity(LBQ_MAX_LAG);
    for lag in 1..=LBQ_MAX_LAG {
        let rho = autocorr(&series, lag)?;
        cumulative += rho.powi(2) / (n - lag as f64);
        q_values.push(n * (n + 2.0) * cumulative);
    }
    std_dev(&q_values)
}

fn high_std_rtn_mean(points: &[MinutePoint], _returns: &[Option<f64>]) -> Option<f64> {
    let mut rtn5 = vec![None; points.len()];
    for idx in 5..points.len() {
        rtn5[idx] = match (points[idx].close, points[idx - 5].close) {
            (Some(current), Some(previous)) if previous.abs() > EPS => {
                finite_value(current / previous - 1.0)
            }
            _ => None,
        };
    }
    let selected = points
        .iter()
        .enumerate()
        .filter_map(|(idx, point)| point.in_window.then_some(rtn5[idx]))
        .collect::<Vec<_>>();

    let mut std30 = vec![None; selected.len()];
    for idx in 0..selected.len() {
        if idx + 1 < HIGH_STD_WINDOW {
            continue;
        }
        let window = &selected[idx + 1 - HIGH_STD_WINDOW..=idx];
        let Some(values) = all_valid(window) else {
            continue;
        };
        std30[idx] = std_dev(&values);
    }
    let std_values = std30.iter().filter_map(|value| *value).collect::<Vec<_>>();
    let q80 = quantile(&std_values, 0.80)?;
    let high_returns = selected
        .iter()
        .zip(std30.iter())
        .filter_map(|(ret, std)| match (*ret, *std) {
            (Some(ret), Some(std)) if std > q80 => Some(ret),
            _ => None,
        })
        .collect::<Vec<_>>();
    mean(&high_returns)
}

fn cond_var(close: &[Option<f64>]) -> Option<f64> {
    let logs = close
        .iter()
        .map(|value| value.map(f64::ln))
        .collect::<Vec<_>>();
    let valid_logs = logs.iter().filter_map(|value| *value).collect::<Vec<_>>();
    let mu = mean(&valid_logs)?;
    let std = std_dev(&valid_logs)?;
    let rho = lag_corr(&logs)?;
    let last_log = logs.iter().rev().find_map(|value| *value)?;
    let one_minus_rho2 = 1.0 - rho * rho;
    if one_minus_rho2 < -EPS {
        return None;
    }
    finite_value((1.0 - rho) * (mu - last_log) - 1.96 * std * one_minus_rho2.max(0.0).sqrt())
}

fn fill_flash_crash_probabilities(values: &mut [(FactorRowKey, SerialStats)]) {
    let pos_lambdas = values
        .iter()
        .filter_map(|(_, stats)| stats.lambda_pos)
        .collect::<Vec<_>>();
    let neg_lambdas = values
        .iter()
        .filter_map(|(_, stats)| stats.lambda_neg)
        .collect::<Vec<_>>();
    let Some(x) = median(&pos_lambdas) else {
        return;
    };
    let Some(q75) = quantile(&neg_lambdas, 0.75) else {
        return;
    };
    let pos_threshold = x.floor().max(0.0) as u64;
    let neg_threshold = q75.ceil().max(0.0) as u64;

    for (_, stats) in values.iter_mut() {
        let value = match (stats.lambda_pos, stats.lambda_neg) {
            (Some(lambda_pos), Some(lambda_neg)) => poisson_cdf(pos_threshold, lambda_pos)
                .and_then(|p_pos| {
                    poisson_survival_at_least(neg_threshold, lambda_neg)
                        .and_then(|p_neg| finite_value(p_pos * p_neg))
                }),
            _ => None,
        };
        stats.flash_crash_prob = value;
    }
}

fn run_length_lambdas(values: &[Option<f64>]) -> (Option<f64>, Option<f64>) {
    let pos = run_lengths(values, |value| value > 0.0);
    let neg = run_lengths(values, |value| value < 0.0);
    (
        mean(&pos.iter().map(|value| *value as f64).collect::<Vec<_>>()),
        mean(&neg.iter().map(|value| *value as f64).collect::<Vec<_>>()),
    )
}

fn run_lengths<F>(values: &[Option<f64>], predicate: F) -> Vec<usize>
where
    F: Fn(f64) -> bool,
{
    let mut lengths = Vec::new();
    let mut current = 0usize;
    for value in values {
        if let Some(value) = value {
            if predicate(*value) {
                current += 1;
                continue;
            }
        }
        if current > 0 {
            lengths.push(current);
            current = 0;
        }
    }
    if current > 0 {
        lengths.push(current);
    }
    lengths
}

fn poisson_cdf(k: u64, lambda: f64) -> Option<f64> {
    if !lambda.is_finite() || lambda < 0.0 {
        return None;
    }
    if lambda == 0.0 {
        return Some(1.0);
    }
    let mut term = (-lambda).exp();
    let mut sum = term;
    for i in 1..=k {
        term *= lambda / i as f64;
        sum += term;
    }
    finite_value(sum.clamp(0.0, 1.0))
}

fn poisson_survival_at_least(k: u64, lambda: f64) -> Option<f64> {
    if k == 0 {
        return Some(1.0);
    }
    poisson_cdf(k - 1, lambda).and_then(|cdf| finite_value((1.0 - cdf).clamp(0.0, 1.0)))
}

fn lag_pairs(values: &[Option<f64>]) -> Vec<(f64, f64)> {
    values
        .windows(2)
        .filter_map(|pair| Some((pair[0]?, pair[1]?)))
        .collect()
}

fn pearson_pairs(pairs: &[(f64, f64)]) -> Option<f64> {
    if pairs.len() < 2 {
        return None;
    }
    let mean_x = pairs.iter().map(|(x, _)| *x).sum::<f64>() / pairs.len() as f64;
    let mean_y = pairs.iter().map(|(_, y)| *y).sum::<f64>() / pairs.len() as f64;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for (x, y) in pairs {
        let dx = x - mean_x;
        let dy = y - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x <= EPS || var_y <= EPS {
        return None;
    }
    finite_value(cov / (var_x.sqrt() * var_y.sqrt()))
}

fn ols_with_intercept(pairs: &[(f64, f64)]) -> Option<(f64, f64, Vec<f64>)> {
    if pairs.len() < 2 {
        return None;
    }
    let mean_x = pairs.iter().map(|(x, _)| *x).sum::<f64>() / pairs.len() as f64;
    let mean_y = pairs.iter().map(|(_, y)| *y).sum::<f64>() / pairs.len() as f64;
    let var_x = pairs.iter().map(|(x, _)| (x - mean_x).powi(2)).sum::<f64>();
    if var_x <= EPS {
        return None;
    }
    let cov_xy = pairs
        .iter()
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum::<f64>();
    let slope = cov_xy / var_x;
    let intercept = mean_y - slope * mean_x;
    let residuals = pairs
        .iter()
        .map(|(x, y)| y - intercept - slope * x)
        .collect::<Vec<_>>();
    Some((intercept, slope, residuals))
}

fn autocorr(values: &[f64], lag: usize) -> Option<f64> {
    if lag == 0 || values.len() <= lag {
        return None;
    }
    let mean = mean(values)?;
    let denominator = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>();
    if denominator <= EPS {
        return None;
    }
    let numerator = (lag..values.len())
        .map(|idx| (values[idx] - mean) * (values[idx - lag] - mean))
        .sum::<f64>();
    finite_value(numerator / denominator)
}

fn all_valid(values: &[Option<f64>]) -> Option<Vec<f64>> {
    let output = values.iter().filter_map(|value| *value).collect::<Vec<_>>();
    (output.len() == values.len()).then_some(output)
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    finite_value(values.iter().sum::<f64>() / values.len() as f64)
}

fn std_dev(values: &[f64]) -> Option<f64> {
    let mean = mean(values)?;
    finite_value(
        (values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / values.len() as f64)
            .sqrt(),
    )
}

fn median(values: &[f64]) -> Option<f64> {
    quantile(values, 0.5)
}

fn quantile(values: &[f64], q: f64) -> Option<f64> {
    let mut values = values.to_vec();
    crate::factor::common::quantile_linear(&mut values, q)
}

fn finite_value(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Option<f64>, expected: Option<f64>) {
        match (actual, expected) {
            (Some(actual), Some(expected)) => assert!(
                (actual - expected).abs() < 1e-10,
                "expected {expected}, got {actual}"
            ),
            (None, None) => {}
            _ => panic!("expected {:?}, got {:?}", expected, actual),
        }
    }

    fn point(close: f64, vol: f64, in_window: bool) -> MinutePoint {
        MinutePoint {
            in_window,
            close: Some(close),
            vol: Some(vol),
        }
    }

    #[test]
    fn xyzq_serial_returns_are_computed_before_window_filter() {
        let points = [
            point(100.0, 1.0, false),
            point(110.0, 1.0, true),
            point(121.0, 1.0, true),
        ];
        let stats = serial_stats(&points);

        assert_close(stats.rtn_dw, Some(0.0));
        assert_close(simple_returns(&points)[1], Some(0.1));
    }

    #[test]
    fn xyzq_serial_volume_pct_uses_selected_window_denominator() {
        let pct = volume_pct(&[Some(1.0), Some(3.0), Some(6.0)]);

        assert_eq!(pct, vec![Some(0.1), Some(0.3), Some(0.6)]);
    }

    #[test]
    fn xyzq_serial_foc_and_dw_match_small_sample() {
        let values = [Some(1.0), Some(2.0), Some(4.0)];

        assert_close(lag_corr(&values), Some(1.0));
        assert_close(durbin_watson(&values), Some(5.0 / 20.0));
    }

    #[test]
    fn xyzq_serial_residual_ar1_rho_uses_two_intercept_regressions() {
        let values = [Some(1.0), Some(2.0), Some(1.5), Some(2.5), Some(2.0)];
        let rho = residual_ar1_rho(&values);

        assert!(rho.is_some());
    }

    #[test]
    fn xyzq_serial_lbq_returns_std_of_ten_q_values() {
        let values = (0..40)
            .map(|idx| Some((idx as f64 / 3.0).sin()))
            .collect::<Vec<_>>();
        let lbq = lbq_std(&values);

        assert!(lbq.is_some());
    }

    #[test]
    fn xyzq_serial_high_std_rtn_uses_rolling_five_and_thirty() {
        let points = (0..40)
            .map(|idx| point(100.0 + idx as f64 + (idx % 7) as f64, 1.0, true))
            .collect::<Vec<_>>();
        let returns = simple_returns(&points);
        let value = high_std_rtn_mean(&points, &returns);

        assert!(value.is_some());
    }

    #[test]
    fn xyzq_serial_cond_var_uses_last_selected_close() {
        let close = [Some(10.0), Some(11.0), Some(12.0), Some(11.5)];
        let value = cond_var(&close);

        assert!(value.is_some());
    }

    #[test]
    fn xyzq_serial_run_lengths_and_poisson_probability_are_directional() {
        let values = [
            Some(1.0),
            Some(1.0),
            Some(-1.0),
            Some(-1.0),
            Some(-1.0),
            Some(0.0),
        ];
        let (lambda_pos, lambda_neg) = run_length_lambdas(&values);

        assert_close(lambda_pos, Some(2.0));
        assert_close(lambda_neg, Some(3.0));
        assert!(poisson_cdf(2, 2.0).unwrap() > 0.0);
        assert!(poisson_survival_at_least(3, 3.0).unwrap() > 0.0);
    }
}
