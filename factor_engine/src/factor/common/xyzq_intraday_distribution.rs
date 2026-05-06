use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Mutex, OnceLock};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::neutralize_size_sector;
use crate::factor::common::stock_daily_raw_ids::{
    CPR_SW_RAW_ID, NOS_GS_RAW_ID, NOS_SW_RAW_ID, REAL_VAR_RAW_ID, RTN5_MEAN_RAW_ID,
    RTN_KURT_RAW_ID, RTN_SKEW_RAW_ID, RV_DOWN_RAW_ID, RV_UMD_RAW_ID, RV_UP_RAW_ID,
};
use crate::factor::common::{clean_intraday_value, stock_minute_raw_spec};
use crate::operators::{cs_pctrank, ts_mean};

pub const RAW_VERSION: &str = "0.1.0";
pub const VERSION: &str = "0.1.0";

const RAW_WINDOW_DAYS: usize = 1;
const SMOOTH_WINDOW: usize = 15;
const MIN_PERIODS: usize = 1;
const EPS: f64 = f64::EPSILON;
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
const KDE_GRID_POINTS: usize = 100;

#[derive(Clone, Copy, Debug)]
pub struct XyzqFactorDef {
    pub id: &'static str,
    pub alias: &'static str,
    pub name: &'static str,
    pub raw_id: &'static str,
}

#[derive(Clone, Copy, Debug)]
pub enum XyzqDistributionRawFamily {
    MinuteReturnDistribution,
    FiveMinuteNoise,
}

#[derive(Clone, Copy, Debug, Default)]
struct DailyDistributionStats {
    rtn5_mean: Option<f64>,
    real_var: Option<f64>,
    rtn_skew: Option<f64>,
    rtn_kurt: Option<f64>,
    rv_up: Option<f64>,
    rv_down: Option<f64>,
    rv_umd: Option<f64>,
    nos_sw: Option<f64>,
    nos_gs: Option<f64>,
    cpr_sw: Option<f64>,
}

pub fn all_raw_ids() -> [&'static str; 10] {
    [
        RTN5_MEAN_RAW_ID,
        REAL_VAR_RAW_ID,
        RTN_SKEW_RAW_ID,
        RTN_KURT_RAW_ID,
        RV_UP_RAW_ID,
        RV_DOWN_RAW_ID,
        RV_UMD_RAW_ID,
        NOS_SW_RAW_ID,
        NOS_GS_RAW_ID,
        CPR_SW_RAW_ID,
    ]
}

pub fn minute_return_distribution_raw_ids() -> [&'static str; 7] {
    [
        REAL_VAR_RAW_ID,
        RTN_SKEW_RAW_ID,
        RTN_KURT_RAW_ID,
        RV_UP_RAW_ID,
        RV_DOWN_RAW_ID,
        RV_UMD_RAW_ID,
        CPR_SW_RAW_ID,
    ]
}

pub fn five_minute_noise_raw_ids() -> [&'static str; 3] {
    [RTN5_MEAN_RAW_ID, NOS_SW_RAW_ID, NOS_GS_RAW_ID]
}

fn raw_ids_for_family(family: XyzqDistributionRawFamily) -> Vec<&'static str> {
    match family {
        XyzqDistributionRawFamily::MinuteReturnDistribution => {
            minute_return_distribution_raw_ids().to_vec()
        }
        XyzqDistributionRawFamily::FiveMinuteNoise => five_minute_noise_raw_ids().to_vec(),
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

pub fn minute_return_distribution_raw_specs() -> Vec<IntradayDailyRawSpec> {
    minute_return_distribution_raw_ids()
        .iter()
        .map(|raw_id| raw_spec(raw_id))
        .collect()
}

pub fn five_minute_noise_raw_specs() -> Vec<IntradayDailyRawSpec> {
    five_minute_noise_raw_ids()
        .iter()
        .map(|raw_id| raw_spec(raw_id))
        .collect()
}

pub fn factor_spec(def: XyzqFactorDef) -> FactorSpec {
    FactorSpec {
        id: def.id.to_string(),
        aliases: vec![def.alias.to_string()],
        name: def.name.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: tags(),
        description: format!(
            "{} from intraday return distribution raw, 15-day mean, cross-sectional percentile rank, and SIZE/SW-sector neutralization.",
            def.name
        ),
        dependencies: dependencies(),
        intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(
            def.raw_id,
            SMOOTH_WINDOW - 1,
        )],
        lookback: Lookback {
            trading_days: SMOOTH_WINDOW - 1,
        },
    }
}

pub fn compute_factor(def: XyzqFactorDef, data: &DataPool) -> Result<FactorSeries> {
    let panel = data.intraday_daily_raw_panel(def.raw_id)?;
    let raw = panel.column(def.raw_id)?;
    let smoothed = raw.ts(|values| ts_mean(values, SMOOTH_WINDOW, MIN_PERIODS))?;
    let ranked = smoothed.cs(|values| cs_pctrank(values, true))?;
    let factor = neutralize_size_sector(&ranked, &panel, data)?;
    Ok(factor.to_factor_series(factor_spec(def)))
}

#[macro_export]
macro_rules! define_xyzq_distribution_factor {
    ($struct_name:ident, $id:expr, $alias:expr, $name:expr, $raw_id:expr) => {
        const DEF: $crate::factor::common::xyzq_intraday_distribution::XyzqFactorDef =
            $crate::factor::common::xyzq_intraday_distribution::XyzqFactorDef {
                id: $id,
                alias: $alias,
                name: $name,
                raw_id: $raw_id,
            };

        pub struct $struct_name;

        pub fn create() -> Box<dyn $crate::factor::Factor> {
            Box::new($struct_name)
        }

        impl $crate::factor::Factor for $struct_name {
            fn spec(&self) -> $crate::core::FactorSpec {
                $crate::factor::common::xyzq_intraday_distribution::factor_spec(DEF)
            }

            fn compute(
                &self,
                _context: &$crate::core::FactorContext,
                data: &$crate::data::DataPool,
            ) -> $crate::error::Result<$crate::core::FactorSeries> {
                $crate::factor::common::xyzq_intraday_distribution::compute_factor(DEF, data)
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
        XyzqDistributionRawFamily::FiveMinuteNoise,
    )
}

pub fn minute_compute_many_for(
    raw_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
    family: XyzqDistributionRawFamily,
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
            let close_by_second = close_by_second(&indices, trade_times, &close);
            let stats = daily_distribution_stats_for(&close_by_second, family);
            let key = FactorRowKey::Daily {
                trade_date: *trade_date,
                ts_code,
            };

            push_requested(
                &mut values,
                &requested,
                RTN5_MEAN_RAW_ID,
                &key,
                stats.rtn5_mean,
            );
            push_requested(
                &mut values,
                &requested,
                REAL_VAR_RAW_ID,
                &key,
                stats.real_var,
            );
            push_requested(
                &mut values,
                &requested,
                RTN_SKEW_RAW_ID,
                &key,
                stats.rtn_skew,
            );
            push_requested(
                &mut values,
                &requested,
                RTN_KURT_RAW_ID,
                &key,
                stats.rtn_kurt,
            );
            push_requested(&mut values, &requested, RV_UP_RAW_ID, &key, stats.rv_up);
            push_requested(&mut values, &requested, RV_DOWN_RAW_ID, &key, stats.rv_down);
            push_requested(&mut values, &requested, RV_UMD_RAW_ID, &key, stats.rv_umd);
            push_requested(&mut values, &requested, NOS_SW_RAW_ID, &key, stats.nos_sw);
            push_requested(&mut values, &requested, NOS_GS_RAW_ID, &key, stats.nos_gs);
            push_requested(&mut values, &requested, CPR_SW_RAW_ID, &key, stats.cpr_sw);
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
        "distribution",
        "normality",
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

fn close_by_second(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
) -> BTreeMap<i32, f64> {
    let mut output = BTreeMap::new();
    for idx in indices {
        let (Some(second), Some(close_value)) = (
            trade_times[*idx].as_deref().and_then(time_to_seconds),
            clean_intraday_value(close[*idx]),
        ) else {
            continue;
        };
        output.insert(second, close_value);
    }
    output
}

fn one_minute_returns(close_by_second: &BTreeMap<i32, f64>) -> Vec<f64> {
    let mut output = Vec::new();
    append_close_to_close_returns(
        close_by_second,
        &minute_anchors(MORNING_START, MORNING_END, 60),
        &mut output,
    );
    append_close_to_close_returns(
        close_by_second,
        &minute_anchors(AFTERNOON_START, AFTERNOON_END, 60),
        &mut output,
    );
    output
}

fn five_minute_returns(close_by_second: &BTreeMap<i32, f64>) -> Vec<f64> {
    let mut anchors = minute_anchors(MORNING_START, MORNING_END, 300);
    anchors.extend(five_minute_afternoon_anchors());
    let mut output = Vec::new();
    append_close_to_close_returns(close_by_second, &anchors[..25], &mut output);
    append_close_to_close_returns(close_by_second, &anchors[25..], &mut output);
    output
}

const MORNING_START: i32 = 9 * 3600 + 30 * 60;
const MORNING_END: i32 = 11 * 3600 + 30 * 60;
const AFTERNOON_START: i32 = 13 * 3600 + 60;
const AFTERNOON_END: i32 = 15 * 3600;

fn minute_anchors(start: i32, end: i32, step: i32) -> Vec<i32> {
    let mut output = Vec::new();
    let mut second = start;
    while second <= end {
        output.push(second);
        second += step;
    }
    output
}

fn five_minute_afternoon_anchors() -> Vec<i32> {
    let mut output = vec![AFTERNOON_START];
    let mut second = 13 * 3600 + 5 * 60;
    while second <= AFTERNOON_END {
        output.push(second);
        second += 300;
    }
    output
}

fn append_close_to_close_returns(
    close_by_second: &BTreeMap<i32, f64>,
    anchors: &[i32],
    output: &mut Vec<f64>,
) {
    for pair in anchors.windows(2) {
        let (Some(previous), Some(current)) =
            (close_by_second.get(&pair[0]), close_by_second.get(&pair[1]))
        else {
            continue;
        };
        if previous.abs() <= EPS {
            continue;
        }
        let value = current / previous - 1.0;
        if value.is_finite() {
            output.push(value);
        }
    }
}

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

fn daily_distribution_stats_for(
    close_by_second: &BTreeMap<i32, f64>,
    family: XyzqDistributionRawFamily,
) -> DailyDistributionStats {
    match family {
        XyzqDistributionRawFamily::MinuteReturnDistribution => {
            let minute_returns = one_minute_returns(close_by_second);
            let real_var = realized_variance(&minute_returns);
            let rv_up = signed_realized_variance(&minute_returns, |value| value > 0.0);
            let rv_down = signed_realized_variance(&minute_returns, |value| value < 0.0);
            DailyDistributionStats {
                real_var,
                rtn_skew: skewness(&minute_returns),
                rtn_kurt: kurtosis(&minute_returns),
                rv_up,
                rv_down,
                rv_umd: rv_umd(rv_up, rv_down, real_var),
                cpr_sw: cumulative_return_skewness(&minute_returns),
                ..DailyDistributionStats::default()
            }
        }
        XyzqDistributionRawFamily::FiveMinuteNoise => {
            let five_minute_returns = five_minute_returns(close_by_second);
            DailyDistributionStats {
                rtn5_mean: mean(&five_minute_returns),
                nos_sw: nos_sw(&five_minute_returns),
                nos_gs: nos_gs(&five_minute_returns),
                ..DailyDistributionStats::default()
            }
        }
    }
}

fn realized_variance(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    finite_value(values.iter().map(|value| value * value).sum())
}

fn signed_realized_variance<F>(values: &[f64], mut predicate: F) -> Option<f64>
where
    F: FnMut(f64) -> bool,
{
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        if predicate(*value) {
            sum += value * value;
            count += 1;
        }
    }
    if count < 2 {
        return Some(0.0);
    }
    finite_value(sum)
}

fn rv_umd(rv_up: Option<f64>, rv_down: Option<f64>, real_var: Option<f64>) -> Option<f64> {
    let Some(real_var) = real_var else {
        return None;
    };
    if real_var.abs() <= EPS {
        return Some(0.0);
    }
    match (rv_up, rv_down) {
        (Some(up), Some(down)) => finite_value((up - down) / real_var),
        _ => None,
    }
}

fn cumulative_return_skewness(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut cumulative = Vec::with_capacity(values.len());
    let mut sum = 0.0;
    for value in values {
        sum += value;
        cumulative.push(sum);
    }
    skewness(&cumulative)
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    finite_value(values.iter().sum::<f64>() / values.len() as f64)
}

fn central_moments(values: &[f64]) -> Option<(f64, f64, f64)> {
    if values.len() < 2 {
        return None;
    }
    let mean = mean(values)?;
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    let mut m4 = 0.0;
    for value in values {
        let deviation = value - mean;
        let d2 = deviation * deviation;
        m2 += d2;
        m3 += d2 * deviation;
        m4 += d2 * d2;
    }
    let n = values.len() as f64;
    let m2 = m2 / n;
    if m2 <= EPS || !m2.is_finite() {
        return None;
    }
    Some((m2, m3 / n, m4 / n))
}

fn skewness(values: &[f64]) -> Option<f64> {
    let (m2, m3, _) = central_moments(values)?;
    finite_value(m3 / m2.powf(1.5))
}

fn kurtosis(values: &[f64]) -> Option<f64> {
    let (m2, _, m4) = central_moments(values)?;
    finite_value(m4 / m2.powi(2))
}

fn standardized_sample(values: &[f64]) -> Option<Vec<f64>> {
    if values.len() < 3 {
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
    let std = variance.sqrt();
    Some(values.iter().map(|value| (value - mean) / std).collect())
}

fn nos_sw(values: &[f64]) -> Option<f64> {
    let mut standardized = standardized_sample(values)?;
    standardized.sort_by(f64::total_cmp);
    let weights = shapiro_francia_weights(standardized.len())?;
    let numerator = weights
        .iter()
        .zip(&standardized)
        .map(|(weight, value)| weight * value)
        .sum::<f64>()
        .powi(2);
    let mean = mean(&standardized)?;
    let denominator = standardized
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>();
    if denominator <= EPS {
        return None;
    }
    finite_value((numerator / denominator).clamp(0.0, 1.0))
}

fn shapiro_francia_weights(n: usize) -> Option<Vec<f64>> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Vec<f64>>>> = OnceLock::new();
    if n < 3 {
        return None;
    }
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(weights) = cache.lock().ok()?.get(&n).cloned() {
        return Some(weights);
    }

    let mut scores = Vec::with_capacity(n);
    for idx in 1..=n {
        let p = (idx as f64 - 0.375) / (n as f64 + 0.25);
        scores.push(inverse_standard_normal(p)?);
    }
    let norm = scores.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm <= EPS {
        return None;
    }
    let weights = scores
        .into_iter()
        .map(|value| value / norm)
        .collect::<Vec<_>>();
    cache.lock().ok()?.insert(n, weights.clone());
    Some(weights)
}

fn nos_gs(values: &[f64]) -> Option<f64> {
    let standardized = standardized_sample(values)?;
    let n = standardized.len() as f64;
    let h = 1.06 * n.powf(-0.2);
    if h <= EPS || !h.is_finite() {
        return None;
    }
    let min = standardized.iter().copied().reduce(f64::min)?;
    let max = standardized.iter().copied().reduce(f64::max)?;
    let step = (max - min) / KDE_GRID_POINTS as f64;

    let mut sum = 0.0;
    for idx in 0..KDE_GRID_POINTS {
        let point = min + step * idx as f64;
        let density = standardized
            .iter()
            .map(|value| normal_pdf((point - value) / h))
            .sum::<f64>()
            / (n * h);
        let diff = density - normal_pdf(point);
        sum += diff * diff;
    }
    finite_value(sum)
}

fn normal_pdf(value: f64) -> f64 {
    INV_SQRT_2PI * (-0.5 * value * value).exp()
}

// Acklam's rational approximation for the inverse standard normal CDF.
fn inverse_standard_normal(p: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&p) || p <= 0.0 || p >= 1.0 {
        return None;
    }

    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    let value = if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    finite_value(value)
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
    fn xyzq_time_parser_accepts_colon_and_compact_times() {
        assert_eq!(time_to_seconds("09:30:00"), Some(MORNING_START));
        assert_eq!(
            time_to_seconds("2026-04-24 13:01:00"),
            Some(AFTERNOON_START)
        );
        assert_eq!(time_to_seconds("150000"), Some(AFTERNOON_END));
    }

    #[test]
    fn xyzq_minute_returns_use_0930_anchor_and_do_not_bridge_lunch() {
        let mut close = BTreeMap::new();
        close.insert(MORNING_START, 100.0);
        close.insert(MORNING_START + 60, 101.0);
        close.insert(MORNING_END, 200.0);
        close.insert(AFTERNOON_START, 300.0);
        close.insert(AFTERNOON_START + 60, 303.0);

        let returns = one_minute_returns(&close);

        assert_eq!(returns.len(), 2);
        assert!((returns[0] - 0.01).abs() < 1e-12);
        assert!((returns[1] - 0.01).abs() < 1e-12);
    }

    #[test]
    fn xyzq_five_minute_returns_use_session_anchors() {
        let mut close = BTreeMap::new();
        close.insert(MORNING_START, 100.0);
        close.insert(MORNING_START + 300, 105.0);
        close.insert(AFTERNOON_START, 200.0);
        close.insert(13 * 3600 + 5 * 60, 210.0);

        let returns = five_minute_returns(&close);

        assert_eq!(returns.len(), 2);
        assert!((returns[0] - 0.05).abs() < 1e-12);
        assert!((returns[1] - 0.05).abs() < 1e-12);
    }

    #[test]
    fn xyzq_distribution_stats_match_small_sample_formulas() {
        let returns = vec![1.0, -2.0, 3.0, -4.0];

        assert_close(realized_variance(&returns), Some(30.0));
        assert_close(
            signed_realized_variance(&returns, |value| value > 0.0),
            Some(10.0),
        );
        assert_close(
            signed_realized_variance(&returns, |value| value < 0.0),
            Some(20.0),
        );
        assert_close(rv_umd(Some(10.0), Some(20.0), Some(30.0)), Some(-1.0 / 3.0));
        assert_close(rv_umd(Some(0.0), Some(0.0), Some(0.0)), Some(0.0));
    }

    #[test]
    fn xyzq_signed_variance_returns_zero_with_less_than_two_samples() {
        let returns = vec![1.0, -2.0, -3.0];

        assert_close(
            signed_realized_variance(&returns, |value| value > 0.0),
            Some(0.0),
        );
        assert_close(
            signed_realized_variance(&returns, |value| value < 0.0),
            Some(13.0),
        );
    }

    #[test]
    fn xyzq_skewness_and_kurtosis_use_population_moments() {
        let values = vec![-1.0, 0.0, 1.0];

        assert_close(skewness(&values), Some(0.0));
        assert_close(kurtosis(&values), Some(1.5));
    }

    #[test]
    fn xyzq_cumulative_return_skewness_uses_running_sum() {
        let returns = vec![1.0, -1.0, 2.0];
        let cumulative = vec![1.0, 0.0, 2.0];

        assert_close(cumulative_return_skewness(&returns), skewness(&cumulative));
    }

    #[test]
    fn xyzq_nos_sw_is_bounded_and_rejects_zero_std() {
        assert_eq!(nos_sw(&[1.0, 1.0, 1.0]), None);
        let value = nos_sw(&[-2.0, -1.0, 0.0, 1.0, 2.0]).expect("W");
        assert!((0.0..=1.0).contains(&value));
    }

    #[test]
    fn xyzq_inverse_normal_is_symmetric() {
        let left = inverse_standard_normal(0.025).expect("left");
        let right = inverse_standard_normal(0.975).expect("right");

        assert!((left + right).abs() < 1e-9);
    }

    #[test]
    fn xyzq_nos_gs_uses_silverman_bandwidth_and_grid() {
        assert_eq!(nos_gs(&[1.0, 1.0, 1.0]), None);
        let value = nos_gs(&[-2.0, -1.0, 0.0, 1.0, 2.0]).expect("kde distance");
        assert!(value >= 0.0);
    }
}
