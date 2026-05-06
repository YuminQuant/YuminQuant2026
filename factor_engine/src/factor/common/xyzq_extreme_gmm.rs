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
    EX_RTN_MAX_FRE_RAW_ID, EX_RTN_MAX_VAL_RAW_ID, EX_RTN_MIN_FRE_RAW_ID, EX_RTN_MIN_VAL_RAW_ID,
    GMM_MEAN2WGT_RAW_ID, GMM_MEANDIF2WGTDIF_RAW_ID, GMM_MEANDIF_RAW_ID, GMM_MEAN_RAW_ID,
};
use crate::factor::common::{clean_intraday_value, stock_minute_raw_spec};
use crate::operators::{cs_pctrank, ts_mean};

pub const RAW_VERSION: &str = "0.1.0";
pub const VERSION: &str = "0.1.0";

const RAW_WINDOW_DAYS: usize = 1;
const DEFAULT_SMOOTH_WINDOW: usize = 20;
const GMM_MEAN2WGT_SMOOTH_WINDOW: usize = 15;
const MIN_PERIODS: usize = 1;
const EPS: f64 = f64::EPSILON;
const EXTREME_Z: f64 = 1.96;
const GMM_MAX_ITER: usize = 50;
const GMM_TOL: f64 = 0.01;
const GMM_MIN_SAMPLES: usize = 10;
const VAR_FLOOR_ABS: f64 = 1e-12;
const VAR_FLOOR_REL: f64 = 1e-6;
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

#[derive(Clone, Copy, Debug)]
pub struct XyzqExtremeGmmFactorDef {
    pub id: &'static str,
    pub alias: &'static str,
    pub name: &'static str,
    pub raw_id: &'static str,
    pub smooth_window: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum XyzqExtremeGmmRawFamily {
    ExtremeReturn,
    GmmReturn,
}

#[derive(Clone, Copy, Debug, Default)]
struct ExtremeGmmStats {
    ex_rtn_max_val: Option<f64>,
    ex_rtn_max_fre: Option<f64>,
    ex_rtn_min_val: Option<f64>,
    ex_rtn_min_fre: Option<f64>,
    gmm_mean: Option<f64>,
    gmm_mean2wgt: Option<f64>,
    gmm_meandif: Option<f64>,
    gmm_meandif2wgtdif: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
struct GmmFit {
    mu_s: f64,
    mu_j: f64,
    w_s: f64,
    w_j: f64,
}

pub const fn default_smooth_window() -> usize {
    DEFAULT_SMOOTH_WINDOW
}

pub const fn gmm_mean2wgt_smooth_window() -> usize {
    GMM_MEAN2WGT_SMOOTH_WINDOW
}

pub fn all_raw_ids() -> [&'static str; 8] {
    [
        EX_RTN_MAX_VAL_RAW_ID,
        EX_RTN_MAX_FRE_RAW_ID,
        EX_RTN_MIN_VAL_RAW_ID,
        EX_RTN_MIN_FRE_RAW_ID,
        GMM_MEAN_RAW_ID,
        GMM_MEAN2WGT_RAW_ID,
        GMM_MEANDIF_RAW_ID,
        GMM_MEANDIF2WGTDIF_RAW_ID,
    ]
}

pub fn extreme_raw_ids() -> [&'static str; 4] {
    [
        EX_RTN_MAX_VAL_RAW_ID,
        EX_RTN_MAX_FRE_RAW_ID,
        EX_RTN_MIN_VAL_RAW_ID,
        EX_RTN_MIN_FRE_RAW_ID,
    ]
}

pub fn gmm_raw_ids() -> [&'static str; 4] {
    [
        GMM_MEAN_RAW_ID,
        GMM_MEAN2WGT_RAW_ID,
        GMM_MEANDIF_RAW_ID,
        GMM_MEANDIF2WGTDIF_RAW_ID,
    ]
}

fn raw_ids_for_family(family: XyzqExtremeGmmRawFamily) -> Vec<&'static str> {
    match family {
        XyzqExtremeGmmRawFamily::ExtremeReturn => extreme_raw_ids().to_vec(),
        XyzqExtremeGmmRawFamily::GmmReturn => gmm_raw_ids().to_vec(),
    }
}

pub fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
    stock_minute_raw_spec(raw_id, RAW_VERSION, &["close"], RAW_WINDOW_DAYS)
}

pub fn raw_specs() -> Vec<IntradayDailyRawSpec> {
    all_raw_ids()
        .iter()
        .map(|raw_id| raw_spec(raw_id))
        .collect()
}

pub fn extreme_raw_specs() -> Vec<IntradayDailyRawSpec> {
    extreme_raw_ids()
        .iter()
        .map(|raw_id| raw_spec(raw_id))
        .collect()
}

pub fn gmm_raw_specs() -> Vec<IntradayDailyRawSpec> {
    gmm_raw_ids()
        .iter()
        .map(|raw_id| raw_spec(raw_id))
        .collect()
}

pub fn factor_spec(def: XyzqExtremeGmmFactorDef) -> FactorSpec {
    let raw_lookback = def.smooth_window - 1;
    let intraday_raw_dependencies = vec![IntradayDailyRawRequest::new(def.raw_id, raw_lookback)];
    FactorSpec {
        id: def.id.to_string(),
        aliases: vec![def.alias.to_string()],
        name: def.name.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: tags(),
        description: format!(
            "{} from rolling 5-minute log return extreme/GMM raw, {}-day mean, cross-sectional percentile rank, and SIZE/SW-sector neutralization.",
            def.name, def.smooth_window
        ),
        dependencies: dependencies(),
        intraday_raw_dependencies,
        lookback: Lookback {
            trading_days: raw_lookback,
        },
    }
}

pub fn compute_factor(def: XyzqExtremeGmmFactorDef, data: &DataPool) -> Result<FactorSeries> {
    let panel = data.intraday_daily_raw_panel(def.raw_id)?;
    let raw = panel.column(def.raw_id)?;
    let smoothed = raw.ts(|values| ts_mean(values, def.smooth_window, MIN_PERIODS))?;
    let ranked = smoothed.cs(|values| cs_pctrank(values, true))?;
    let factor = neutralize_size_sector(&ranked, &panel, data)?;
    Ok(factor.to_factor_series(factor_spec(def)))
}

#[macro_export]
macro_rules! define_xyzq_extreme_gmm_factor {
    ($struct_name:ident, $id:expr, $alias:expr, $name:expr, $raw_id:expr, $smooth_window:expr) => {
        const DEF: $crate::factor::common::xyzq_extreme_gmm::XyzqExtremeGmmFactorDef =
            $crate::factor::common::xyzq_extreme_gmm::XyzqExtremeGmmFactorDef {
                id: $id,
                alias: $alias,
                name: $name,
                raw_id: $raw_id,
                smooth_window: $smooth_window,
            };

        pub struct $struct_name;

        pub fn create() -> Box<dyn $crate::factor::Factor> {
            Box::new($struct_name)
        }

        impl $crate::factor::Factor for $struct_name {
            fn spec(&self) -> $crate::core::FactorSpec {
                $crate::factor::common::xyzq_extreme_gmm::factor_spec(DEF)
            }

            fn compute(
                &self,
                _context: &$crate::core::FactorContext,
                data: &$crate::data::DataPool,
            ) -> $crate::error::Result<$crate::core::FactorSeries> {
                $crate::factor::common::xyzq_extreme_gmm::compute_factor(DEF, data)
            }
        }
    };
}

pub fn minute_compute_many(
    raw_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
) -> Result<Vec<IntradayDailyRawSeries>> {
    minute_compute_many_for(
        raw_ids,
        context,
        data,
        XyzqExtremeGmmRawFamily::ExtremeReturn,
    )
}

pub fn minute_compute_many_for(
    raw_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
    family: XyzqExtremeGmmRawFamily,
) -> Result<Vec<IntradayDailyRawSeries>> {
    let family_raw_ids = raw_ids_for_family(family);
    let requested = raw_ids
        .iter()
        .map(String::as_str)
        .filter(|raw_id| family_raw_ids.contains(raw_id))
        .collect::<BTreeSet<_>>();
    if requested.is_empty() {
        return Ok(Vec::new());
    }
    let needs_extreme = requested
        .iter()
        .any(|raw_id| extreme_raw_ids().contains(raw_id));
    let needs_gmm = requested
        .iter()
        .any(|raw_id| gmm_raw_ids().contains(raw_id));

    let mut values = family_raw_ids
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

        for (ts_code, indices) in grouped {
            let closes = sorted_intraday_closes(&indices, trade_times, &close);
            let returns = rolling_five_minute_log_returns(&closes);
            let stats = extreme_gmm_stats(&returns, needs_extreme, needs_gmm);
            let key = FactorRowKey::Daily {
                trade_date: *trade_date,
                ts_code,
            };

            push_requested(
                &mut values,
                &requested,
                EX_RTN_MAX_VAL_RAW_ID,
                &key,
                stats.ex_rtn_max_val,
            );
            push_requested(
                &mut values,
                &requested,
                EX_RTN_MAX_FRE_RAW_ID,
                &key,
                stats.ex_rtn_max_fre,
            );
            push_requested(
                &mut values,
                &requested,
                EX_RTN_MIN_VAL_RAW_ID,
                &key,
                stats.ex_rtn_min_val,
            );
            push_requested(
                &mut values,
                &requested,
                EX_RTN_MIN_FRE_RAW_ID,
                &key,
                stats.ex_rtn_min_fre,
            );
            push_requested(
                &mut values,
                &requested,
                GMM_MEAN_RAW_ID,
                &key,
                stats.gmm_mean,
            );
            push_requested(
                &mut values,
                &requested,
                GMM_MEAN2WGT_RAW_ID,
                &key,
                stats.gmm_mean2wgt,
            );
            push_requested(
                &mut values,
                &requested,
                GMM_MEANDIF_RAW_ID,
                &key,
                stats.gmm_meandif,
            );
            push_requested(
                &mut values,
                &requested,
                GMM_MEANDIF2WGTDIF_RAW_ID,
                &key,
                stats.gmm_meandif2wgtdif,
            );
        }
    }

    let mut output = Vec::new();
    for raw_id in family_raw_ids {
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
        "intraday",
        "minute_agg",
        "extreme_return",
        "gmm",
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

fn sorted_intraday_closes(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
) -> Vec<f64> {
    let mut by_second = BTreeMap::new();
    for idx in indices {
        let (Some(second), Some(close_value)) = (
            trade_times[*idx].as_deref().and_then(time_to_seconds),
            clean_intraday_value(close[*idx]),
        ) else {
            continue;
        };
        if !(TRADING_START..=TRADING_END).contains(&second) || close_value <= 0.0 {
            continue;
        }
        by_second.insert(second, close_value);
    }
    by_second.into_values().collect()
}

fn rolling_five_minute_log_returns(closes: &[f64]) -> Vec<f64> {
    let mut output = Vec::new();
    for idx in 5..closes.len() {
        let previous = closes[idx - 5];
        let current = closes[idx];
        if previous <= 0.0 || current <= 0.0 {
            continue;
        }
        let value = (current / previous).ln();
        if value.is_finite() {
            output.push(value);
        }
    }
    output
}

const TRADING_START: i32 = 9 * 3600 + 30 * 60;
const TRADING_END: i32 = 15 * 3600;

fn time_to_seconds(value: &str) -> Option<i32> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let value = value
        .rsplit_once(' ')
        .map(|(_, right)| right)
        .or_else(|| value.rsplit_once('T').map(|(_, right)| right))
        .unwrap_or(value)
        .trim();
    if value.contains(':') {
        let parts = value.split(':').collect::<Vec<_>>();
        if parts.len() < 2 {
            return None;
        }
        let hour = parts[0].parse::<i32>().ok()?;
        let minute = parts[1].parse::<i32>().ok()?;
        let second = parts
            .get(2)
            .and_then(|value| value.get(..2))
            .and_then(|value| value.parse::<i32>().ok())
            .unwrap_or(0);
        return Some(hour * 3600 + minute * 60 + second);
    }
    if value.len() >= 6 {
        let hour = value.get(0..2)?.parse::<i32>().ok()?;
        let minute = value.get(2..4)?.parse::<i32>().ok()?;
        let second = value.get(4..6)?.parse::<i32>().ok()?;
        return Some(hour * 3600 + minute * 60 + second);
    }
    if value.len() == 4 {
        let hour = value.get(0..2)?.parse::<i32>().ok()?;
        let minute = value.get(2..4)?.parse::<i32>().ok()?;
        return Some(hour * 3600 + minute * 60);
    }
    None
}

fn extreme_gmm_stats(returns: &[f64], needs_extreme: bool, needs_gmm: bool) -> ExtremeGmmStats {
    let (ex_rtn_max_val, ex_rtn_max_fre, ex_rtn_min_val, ex_rtn_min_fre) = if needs_extreme {
        extreme_stats(returns)
    } else {
        (None, None, None, None)
    };
    let (gmm_mean, gmm_mean2wgt, gmm_meandif, gmm_meandif2wgtdif) = if needs_gmm {
        gmm_stats_from_fit(fit_two_gaussian_mixture(returns))
    } else {
        (None, None, None, None)
    };
    ExtremeGmmStats {
        ex_rtn_max_val,
        ex_rtn_max_fre,
        ex_rtn_min_val,
        ex_rtn_min_fre,
        gmm_mean,
        gmm_mean2wgt,
        gmm_meandif,
        gmm_meandif2wgtdif,
    }
}

fn extreme_stats(values: &[f64]) -> (Option<f64>, Option<f64>, Option<f64>, Option<f64>) {
    let Some((avg, std)) = mean_and_sample_std(values) else {
        return (None, None, None, None);
    };
    if std <= EPS {
        return (None, None, None, None);
    }
    let up_threshold = avg + EXTREME_Z * std;
    let down_threshold = avg - EXTREME_Z * std;

    let up_values = values
        .iter()
        .copied()
        .filter(|value| *value > up_threshold)
        .collect::<Vec<_>>();
    let down_values = values
        .iter()
        .copied()
        .filter(|value| *value < down_threshold)
        .collect::<Vec<_>>();

    let max_fre = Some(up_values.len() as f64);
    let min_fre = Some(down_values.len() as f64);
    let max_val = if up_values.is_empty() || up_threshold.abs() <= EPS {
        None
    } else {
        mean(&up_values).and_then(|value| finite_value(value / up_threshold))
    };
    let min_val = if down_values.is_empty() || down_threshold.abs() <= EPS {
        None
    } else {
        mean(&down_values).and_then(|value| finite_value((value / down_threshold).abs()))
    };
    (max_val, max_fre, min_val, min_fre)
}

fn gmm_stats_from_fit(fit: Option<GmmFit>) -> (Option<f64>, Option<f64>, Option<f64>, Option<f64>) {
    let Some(fit) = fit else {
        return (None, None, None, None);
    };
    let gmm_mean = finite_value(fit.mu_j);
    let gmm_mean2wgt = if fit.w_j.abs() <= EPS {
        None
    } else {
        finite_value(fit.mu_j / fit.w_j)
    };
    let mean_dif = fit.mu_s - fit.mu_j;
    let gmm_meandif = finite_value(mean_dif);
    let weight_dif = fit.w_s - fit.w_j;
    let gmm_meandif2wgtdif = if weight_dif.abs() <= EPS {
        None
    } else {
        finite_value(mean_dif / weight_dif)
    };
    (gmm_mean, gmm_mean2wgt, gmm_meandif, gmm_meandif2wgtdif)
}

fn fit_two_gaussian_mixture(values: &[f64]) -> Option<GmmFit> {
    if values.len() < GMM_MIN_SAMPLES {
        return None;
    }
    let total_var = population_variance(values)?;
    if total_var <= EPS {
        return None;
    }
    let variance_floor = (total_var * VAR_FLOOR_REL).max(VAR_FLOOR_ABS);
    let median = median(values)?;
    let tail_count = ((values.len() as f64) * 0.2).ceil() as usize;
    let tail_count = tail_count.clamp(1, values.len() - 1);

    let mut by_distance = values
        .iter()
        .enumerate()
        .map(|(idx, value)| (idx, (value - median).abs()))
        .collect::<Vec<_>>();
    by_distance.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    let mut is_tail = vec![false; values.len()];
    for (idx, _) in by_distance.into_iter().take(tail_count) {
        is_tail[idx] = true;
    }

    let mut mu_j = group_mean(values, &is_tail, true)?;
    let mut mu_s = group_mean(values, &is_tail, false)?;
    let mut var_j = group_variance(values, &is_tail, true, mu_j)?.max(variance_floor);
    let mut var_s = group_variance(values, &is_tail, false, mu_s)?.max(variance_floor);
    let mut w_j = tail_count as f64 / values.len() as f64;
    let mut w_s = 1.0 - w_j;
    let mut previous_ll = f64::NEG_INFINITY;

    let mut resp_j = vec![0.0; values.len()];
    for _ in 0..GMM_MAX_ITER {
        let mut ll = 0.0;
        for (slot, value) in resp_j.iter_mut().zip(values) {
            let p_s = w_s * gaussian_pdf(*value, mu_s, var_s);
            let p_j = w_j * gaussian_pdf(*value, mu_j, var_j);
            let denom = p_s + p_j;
            if denom <= 0.0 || !denom.is_finite() {
                return None;
            }
            *slot = (p_j / denom).clamp(0.0, 1.0);
            ll += denom.ln();
        }
        if (ll - previous_ll).abs() / (values.len() as f64) < GMM_TOL {
            break;
        }
        previous_ll = ll;

        let sum_j = resp_j.iter().sum::<f64>();
        let sum_s = values.len() as f64 - sum_j;
        if sum_j <= EPS || sum_s <= EPS {
            return None;
        }
        w_j = sum_j / values.len() as f64;
        w_s = 1.0 - w_j;
        mu_j = values
            .iter()
            .zip(&resp_j)
            .map(|(value, resp)| value * resp)
            .sum::<f64>()
            / sum_j;
        mu_s = values
            .iter()
            .zip(&resp_j)
            .map(|(value, resp)| value * (1.0 - resp))
            .sum::<f64>()
            / sum_s;
        var_j = values
            .iter()
            .zip(&resp_j)
            .map(|(value, resp)| resp * (value - mu_j).powi(2))
            .sum::<f64>()
            / sum_j;
        var_s = values
            .iter()
            .zip(&resp_j)
            .map(|(value, resp)| (1.0 - resp) * (value - mu_s).powi(2))
            .sum::<f64>()
            / sum_s;
        if !var_j.is_finite() || !var_s.is_finite() {
            return None;
        }
        var_j = var_j.max(variance_floor);
        var_s = var_s.max(variance_floor);
    }

    if ![mu_s, mu_j, w_s, w_j, var_s, var_j]
        .iter()
        .all(|value| value.is_finite())
    {
        return None;
    }

    if w_j <= w_s {
        Some(GmmFit {
            mu_s,
            mu_j,
            w_s,
            w_j,
        })
    } else {
        Some(GmmFit {
            mu_s: mu_j,
            mu_j: mu_s,
            w_s: w_j,
            w_j: w_s,
        })
    }
}

fn gaussian_pdf(value: f64, mean: f64, variance: f64) -> f64 {
    if variance <= 0.0 || !variance.is_finite() {
        return 0.0;
    }
    INV_SQRT_2PI / variance.sqrt() * (-0.5 * (value - mean).powi(2) / variance).exp()
}

fn group_mean(values: &[f64], is_tail: &[bool], tail_value: bool) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for (value, is_tail) in values.iter().zip(is_tail) {
        if *is_tail == tail_value {
            sum += value;
            count += 1;
        }
    }
    if count == 0 {
        return None;
    }
    finite_value(sum / count as f64)
}

fn group_variance(values: &[f64], is_tail: &[bool], tail_value: bool, mean: f64) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for (value, is_tail) in values.iter().zip(is_tail) {
        if *is_tail == tail_value {
            sum += (value - mean).powi(2);
            count += 1;
        }
    }
    if count == 0 {
        return None;
    }
    finite_value(sum / count as f64)
}

fn mean_and_sample_std(values: &[f64]) -> Option<(f64, f64)> {
    if values.len() < 2 {
        return None;
    }
    let mean = mean(values)?;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    if variance <= EPS || !variance.is_finite() {
        return None;
    }
    Some((mean, variance.sqrt()))
}

fn population_variance(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mean = mean(values)?;
    finite_value(
        values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / values.len() as f64,
    )
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        Some(sorted[mid])
    } else {
        finite_value((sorted[mid - 1] + sorted[mid]) / 2.0)
    }
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    finite_value(values.iter().sum::<f64>() / values.len() as f64)
}

fn finite_value(value: f64) -> Option<f64> {
    if value.is_finite() {
        Some(value)
    } else {
        None
    }
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

    #[test]
    fn xyzq_extreme_time_parser_accepts_common_formats() {
        assert_eq!(time_to_seconds("09:30:00"), Some(TRADING_START));
        assert_eq!(time_to_seconds("2026-04-24 15:00:00"), Some(TRADING_END));
        assert_eq!(time_to_seconds("093500"), Some(9 * 3600 + 35 * 60));
    }

    #[test]
    fn xyzq_extreme_rolling_log_returns_use_full_day_sequence() {
        let closes = vec![100.0, 101.0, 102.0, 103.0, 104.0, 110.0, 121.0];
        let returns = rolling_five_minute_log_returns(&closes);

        assert_eq!(returns.len(), 2);
        assert!((returns[0] - (110.0_f64 / 100.0).ln()).abs() < 1e-12);
        assert!((returns[1] - (121.0_f64 / 101.0).ln()).abs() < 1e-12);
    }

    #[test]
    fn xyzq_extreme_stats_handle_counts_values_and_zero_denominators() {
        let mut values = vec![0.0; 18];
        values.push(-10.0);
        values.push(10.0);
        let (max_val, max_fre, min_val, min_fre) = extreme_stats(&values);

        assert_eq!(max_fre, Some(1.0));
        assert_eq!(min_fre, Some(1.0));
        assert!(max_val.expect("max val") > 1.0);
        assert!(min_val.expect("min val") > 1.0);

        let no_extreme = vec![-1.0, 0.0, 1.0];
        let (max_val, max_fre, min_val, min_fre) = extreme_stats(&no_extreme);
        assert_eq!(max_fre, Some(0.0));
        assert_eq!(min_fre, Some(0.0));
        assert_eq!(max_val, None);
        assert_eq!(min_val, None);
    }

    #[test]
    fn xyzq_extreme_gmm_identifies_low_weight_component_as_jump() {
        let mut values = Vec::new();
        for idx in 0..40 {
            values.push((idx as f64 - 20.0) * 0.001);
        }
        for value in [0.09, 0.10, 0.11, 0.12] {
            values.push(value);
        }

        let fit = fit_two_gaussian_mixture(&values).expect("fit");

        assert!(fit.w_j <= 0.5);
        assert!(fit.mu_j > fit.mu_s);
    }

    #[test]
    fn xyzq_extreme_gmm_stats_match_output_formulas() {
        let fit = GmmFit {
            mu_s: 0.01,
            mu_j: 0.05,
            w_s: 0.8,
            w_j: 0.2,
        };
        let (mean, mean2wgt, meandif, meandif2wgtdif) = gmm_stats_from_fit(Some(fit));

        assert_close(mean, Some(0.05));
        assert_close(mean2wgt, Some(0.25));
        assert_close(meandif, Some(-0.04));
        assert_close(meandif2wgtdif, Some(-0.04 / 0.6));
    }

    #[test]
    fn xyzq_extreme_gmm_factor_spec_uses_factor_window_lookback() {
        let def = XyzqExtremeGmmFactorDef {
            id: "gmm_mean2wgt",
            alias: "gmm_mean2wgt",
            name: "gmm_mean2wgt",
            raw_id: GMM_MEAN2WGT_RAW_ID,
            smooth_window: GMM_MEAN2WGT_SMOOTH_WINDOW,
        };
        let spec = factor_spec(def);

        assert_eq!(spec.lookback.trading_days, GMM_MEAN2WGT_SMOOTH_WINDOW - 1);
        assert_eq!(spec.intraday_raw_dependencies.len(), 1);
        assert_eq!(
            spec.intraday_raw_dependencies[0].daily_lookback,
            GMM_MEAN2WGT_SMOOTH_WINDOW - 1
        );
    }
}
