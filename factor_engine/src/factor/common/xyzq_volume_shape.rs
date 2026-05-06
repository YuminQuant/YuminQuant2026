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
    CUMSUMVOL_MEAN_RAW_ID, CUMSUMVOL_STD_RAW_ID, LOGVOL_10TAIL_RAW_ID, LOGVOL_90TAIL_RAW_ID,
    LOGVOL_SKEW_RAW_ID, VOLROC_KURT_RAW_ID, VOLROC_SKEW_RAW_ID, VOL_ENTROPY_SHAPE_RAW_ID,
    VOL_MAXMEAN_RAW_ID, VOL_MAXSTD_RAW_ID, VSA_HIGH2MIN_RAW_ID, VSA_LOW2MAX_RAW_ID,
    VSA_RATIO_RAW_ID,
};
use crate::factor::common::{clean_intraday_value, intraday_time_in_range, stock_minute_raw_spec};
use crate::operators::{cs_pctrank, ts_mean, ts_std_dev};

pub const RAW_VERSION: &str = "0.1.0";
pub const VERSION: &str = "0.1.0";

const RAW_WINDOW_DAYS: usize = 1;
const DEFAULT_WINDOW: usize = 15;
const ENTROPY_WINDOW: usize = 20;
const SHARED_RAW_LOOKBACK: usize = ENTROPY_WINDOW - 1;
const MIN_PERIODS: usize = 1;
const START_TIME: &str = "09:31:00";
const END_TIME: &str = "15:00:00";
const FIFTEEN_MINUTE_BUCKETS: usize = 16;
const FIFTEEN_MINUTE_BUCKET_SIZE: usize = 15;
const ENTROPY_BINS: usize = 10;
const EPS: f64 = f64::EPSILON;

#[derive(Clone, Copy, Debug)]
pub enum XyzqVolumeAggregation {
    Mean,
    Std,
}

#[derive(Clone, Copy, Debug)]
pub enum XyzqVolumeRawFamily {
    LogvolShape,
    VolrocShape,
    CumsumvolShape,
    VolEntropy,
    VolBootstrapMax,
    Vsa,
}

#[derive(Clone, Copy, Debug)]
pub struct XyzqVolumeFactorDef {
    pub id: &'static str,
    pub alias: &'static str,
    pub name: &'static str,
    pub raw_id: &'static str,
    pub window: usize,
    pub aggregation: XyzqVolumeAggregation,
}

#[derive(Clone, Copy, Debug, Default)]
struct VolumeShapeStats {
    logvol_skew: Option<f64>,
    logvol_90tail: Option<f64>,
    logvol_10tail: Option<f64>,
    volroc_skew: Option<f64>,
    volroc_kurt: Option<f64>,
    cumsumvol_mean: Option<f64>,
    cumsumvol_std: Option<f64>,
    vol_entropy: Option<f64>,
    vol_maxmean: Option<f64>,
    vol_maxstd: Option<f64>,
    vsa_ratio: Option<f64>,
    vsa_low2max: Option<f64>,
    vsa_high2min: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default)]
struct RequestedGroups {
    logvol: bool,
    volroc: bool,
    cumsum: bool,
    entropy: bool,
    bootstrap_max: bool,
    vsa: bool,
}

pub const fn default_window() -> usize {
    DEFAULT_WINDOW
}

pub const fn entropy_window() -> usize {
    ENTROPY_WINDOW
}

pub fn all_raw_ids() -> [&'static str; 13] {
    [
        LOGVOL_SKEW_RAW_ID,
        LOGVOL_90TAIL_RAW_ID,
        LOGVOL_10TAIL_RAW_ID,
        VOLROC_SKEW_RAW_ID,
        VOLROC_KURT_RAW_ID,
        CUMSUMVOL_MEAN_RAW_ID,
        CUMSUMVOL_STD_RAW_ID,
        VOL_ENTROPY_SHAPE_RAW_ID,
        VOL_MAXMEAN_RAW_ID,
        VOL_MAXSTD_RAW_ID,
        VSA_RATIO_RAW_ID,
        VSA_LOW2MAX_RAW_ID,
        VSA_HIGH2MIN_RAW_ID,
    ]
}

pub fn raw_ids_for_family(family: XyzqVolumeRawFamily) -> &'static [&'static str] {
    match family {
        XyzqVolumeRawFamily::LogvolShape => &[
            LOGVOL_SKEW_RAW_ID,
            LOGVOL_90TAIL_RAW_ID,
            LOGVOL_10TAIL_RAW_ID,
        ],
        XyzqVolumeRawFamily::VolrocShape => &[VOLROC_SKEW_RAW_ID, VOLROC_KURT_RAW_ID],
        XyzqVolumeRawFamily::CumsumvolShape => &[CUMSUMVOL_MEAN_RAW_ID, CUMSUMVOL_STD_RAW_ID],
        XyzqVolumeRawFamily::VolEntropy => &[VOL_ENTROPY_SHAPE_RAW_ID],
        XyzqVolumeRawFamily::VolBootstrapMax => &[VOL_MAXMEAN_RAW_ID, VOL_MAXSTD_RAW_ID],
        XyzqVolumeRawFamily::Vsa => &[VSA_RATIO_RAW_ID, VSA_LOW2MAX_RAW_ID, VSA_HIGH2MIN_RAW_ID],
    }
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

pub fn raw_specs_for_family(family: XyzqVolumeRawFamily) -> Vec<IntradayDailyRawSpec> {
    raw_ids_for_family(family)
        .iter()
        .map(|raw_id| raw_spec(raw_id))
        .collect()
}

pub fn logvol_raw_specs() -> Vec<IntradayDailyRawSpec> {
    raw_specs_for_family(XyzqVolumeRawFamily::LogvolShape)
}

pub fn volroc_raw_specs() -> Vec<IntradayDailyRawSpec> {
    raw_specs_for_family(XyzqVolumeRawFamily::VolrocShape)
}

pub fn cumsumvol_raw_specs() -> Vec<IntradayDailyRawSpec> {
    raw_specs_for_family(XyzqVolumeRawFamily::CumsumvolShape)
}

pub fn vol_entropy_raw_specs() -> Vec<IntradayDailyRawSpec> {
    raw_specs_for_family(XyzqVolumeRawFamily::VolEntropy)
}

pub fn vol_bootstrap_max_raw_specs() -> Vec<IntradayDailyRawSpec> {
    raw_specs_for_family(XyzqVolumeRawFamily::VolBootstrapMax)
}

pub fn vsa_raw_specs() -> Vec<IntradayDailyRawSpec> {
    raw_specs_for_family(XyzqVolumeRawFamily::Vsa)
}

pub fn factor_spec(def: XyzqVolumeFactorDef) -> FactorSpec {
    FactorSpec {
        id: def.id.to_string(),
        aliases: vec![def.alias.to_string()],
        name: def.name.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: tags(),
        description: format!(
            "{} from intraday volume-shape raw, rolling aggregation, cross-sectional percentile rank, and SIZE/SW-sector neutralization.",
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

pub fn compute_factor(def: XyzqVolumeFactorDef, data: &DataPool) -> Result<FactorSeries> {
    let panel = data.intraday_daily_raw_panel(def.raw_id)?;
    let raw = panel.column(def.raw_id)?;
    let aggregated = match def.aggregation {
        XyzqVolumeAggregation::Mean => raw.ts(|values| ts_mean(values, def.window, MIN_PERIODS))?,
        XyzqVolumeAggregation::Std => {
            raw.ts(|values| ts_std_dev(values, def.window, MIN_PERIODS))?
        }
    };
    let ranked = aggregated.cs(|values| cs_pctrank(values, true))?;
    let factor = neutralize_size_sector(&ranked, &panel, data)?;
    Ok(factor.to_factor_series(factor_spec(def)))
}

#[macro_export]
macro_rules! define_xyzq_volume_shape_factor {
    ($struct_name:ident, $id:expr, $alias:expr, $name:expr, $raw_id:expr, $window:expr, $aggregation:ident) => {
        const DEF: $crate::factor::common::xyzq_volume_shape::XyzqVolumeFactorDef =
            $crate::factor::common::xyzq_volume_shape::XyzqVolumeFactorDef {
                id: $id,
                alias: $alias,
                name: $name,
                raw_id: $raw_id,
                window: $window,
                aggregation:
                    $crate::factor::common::xyzq_volume_shape::XyzqVolumeAggregation::$aggregation,
            };

        pub struct $struct_name;

        pub fn create() -> Box<dyn $crate::factor::Factor> {
            Box::new($struct_name)
        }

        impl $crate::factor::Factor for $struct_name {
            fn spec(&self) -> $crate::core::FactorSpec {
                $crate::factor::common::xyzq_volume_shape::factor_spec(DEF)
            }

            fn compute(
                &self,
                _context: &$crate::core::FactorContext,
                data: &$crate::data::DataPool,
            ) -> $crate::error::Result<$crate::core::FactorSeries> {
                $crate::factor::common::xyzq_volume_shape::compute_factor(DEF, data)
            }
        }
    };
}

pub fn minute_compute_many(
    raw_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
) -> Result<Vec<IntradayDailyRawSeries>> {
    minute_compute_many_impl(raw_ids, context, data, &all_raw_ids())
}

pub fn minute_compute_many_for(
    raw_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
    family: XyzqVolumeRawFamily,
) -> Result<Vec<IntradayDailyRawSeries>> {
    minute_compute_many_impl(raw_ids, context, data, raw_ids_for_family(family))
}

fn minute_compute_many_impl(
    raw_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
    family_raw_ids: &[&'static str],
) -> Result<Vec<IntradayDailyRawSeries>> {
    let requested = raw_ids
        .iter()
        .map(String::as_str)
        .filter(|raw_id| family_raw_ids.contains(raw_id))
        .collect::<BTreeSet<_>>();
    if requested.is_empty() {
        return Ok(Vec::new());
    }
    let groups = requested_groups(&requested);

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
            let Some(time) = trade_times[idx].as_deref() else {
                continue;
            };
            if intraday_time_in_range(time, START_TIME, END_TIME) {
                grouped.entry(ts_code).or_default().push(idx);
            }
        }

        for (ts_code, mut indices) in grouped {
            indices.sort_by(|left, right| trade_times[*left].cmp(&trade_times[*right]));
            let minute_data = minute_data_from_indices(&indices, &close, &vol);
            let stats = volume_shape_stats(&minute_data, groups);
            let key = FactorRowKey::Daily {
                trade_date: *trade_date,
                ts_code,
            };

            push_requested(
                &mut values,
                &requested,
                LOGVOL_SKEW_RAW_ID,
                &key,
                stats.logvol_skew,
            );
            push_requested(
                &mut values,
                &requested,
                LOGVOL_90TAIL_RAW_ID,
                &key,
                stats.logvol_90tail,
            );
            push_requested(
                &mut values,
                &requested,
                LOGVOL_10TAIL_RAW_ID,
                &key,
                stats.logvol_10tail,
            );
            push_requested(
                &mut values,
                &requested,
                VOLROC_SKEW_RAW_ID,
                &key,
                stats.volroc_skew,
            );
            push_requested(
                &mut values,
                &requested,
                VOLROC_KURT_RAW_ID,
                &key,
                stats.volroc_kurt,
            );
            push_requested(
                &mut values,
                &requested,
                CUMSUMVOL_MEAN_RAW_ID,
                &key,
                stats.cumsumvol_mean,
            );
            push_requested(
                &mut values,
                &requested,
                CUMSUMVOL_STD_RAW_ID,
                &key,
                stats.cumsumvol_std,
            );
            push_requested(
                &mut values,
                &requested,
                VOL_ENTROPY_SHAPE_RAW_ID,
                &key,
                stats.vol_entropy,
            );
            push_requested(
                &mut values,
                &requested,
                VOL_MAXMEAN_RAW_ID,
                &key,
                stats.vol_maxmean,
            );
            push_requested(
                &mut values,
                &requested,
                VOL_MAXSTD_RAW_ID,
                &key,
                stats.vol_maxstd,
            );
            push_requested(
                &mut values,
                &requested,
                VSA_RATIO_RAW_ID,
                &key,
                stats.vsa_ratio,
            );
            push_requested(
                &mut values,
                &requested,
                VSA_LOW2MAX_RAW_ID,
                &key,
                stats.vsa_low2max,
            );
            push_requested(
                &mut values,
                &requested,
                VSA_HIGH2MIN_RAW_ID,
                &key,
                stats.vsa_high2min,
            );
        }
    }

    let mut output = Vec::new();
    for &raw_id in family_raw_ids {
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
        "volume",
        "intraday",
        "minute_agg",
        "distribution",
        "vsa",
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

fn requested_groups(requested: &BTreeSet<&str>) -> RequestedGroups {
    RequestedGroups {
        logvol: requested.contains(LOGVOL_SKEW_RAW_ID)
            || requested.contains(LOGVOL_90TAIL_RAW_ID)
            || requested.contains(LOGVOL_10TAIL_RAW_ID),
        volroc: requested.contains(VOLROC_SKEW_RAW_ID) || requested.contains(VOLROC_KURT_RAW_ID),
        cumsum: requested.contains(CUMSUMVOL_MEAN_RAW_ID)
            || requested.contains(CUMSUMVOL_STD_RAW_ID),
        entropy: requested.contains(VOL_ENTROPY_SHAPE_RAW_ID),
        bootstrap_max: requested.contains(VOL_MAXMEAN_RAW_ID)
            || requested.contains(VOL_MAXSTD_RAW_ID),
        vsa: requested.contains(VSA_RATIO_RAW_ID)
            || requested.contains(VSA_LOW2MAX_RAW_ID)
            || requested.contains(VSA_HIGH2MIN_RAW_ID),
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
        values.entry(raw_id).or_default().push(FactorValue {
            key: key.clone(),
            value,
        });
    }
}

#[derive(Clone, Copy, Debug)]
struct MinutePoint {
    close: Option<f64>,
    vol: Option<f64>,
}

fn minute_data_from_indices(
    indices: &[usize],
    close: &[Option<f64>],
    vol: &[Option<f64>],
) -> Vec<MinutePoint> {
    indices
        .iter()
        .map(|idx| MinutePoint {
            close: clean_intraday_value(close[*idx]).filter(|value| *value > 0.0),
            vol: clean_intraday_value(vol[*idx]).filter(|value| *value >= 0.0),
        })
        .collect()
}

fn volume_shape_stats(data: &[MinutePoint], groups: RequestedGroups) -> VolumeShapeStats {
    let mut stats = VolumeShapeStats::default();
    let volumes = valid_volumes(data);

    if groups.logvol {
        let (skew, tail90, tail10) = logvol_stats(data);
        stats.logvol_skew = skew;
        stats.logvol_90tail = tail90;
        stats.logvol_10tail = tail10;
    }
    if groups.volroc {
        let (skew, kurt) = volroc_stats(data);
        stats.volroc_skew = skew;
        stats.volroc_kurt = kurt;
    }
    if groups.cumsum {
        let (mean, std) = cumsumvol_stats(data);
        stats.cumsumvol_mean = mean;
        stats.cumsumvol_std = std;
    }
    if groups.entropy {
        stats.vol_entropy = volume_entropy(&volumes);
    }
    if groups.bootstrap_max {
        let (mean, std) = bootstrap_max_adjusted_stats(&volumes);
        stats.vol_maxmean = mean;
        stats.vol_maxstd = std;
    }
    if groups.vsa {
        let (ratio, low2max, high2min) = vsa_stats(data);
        stats.vsa_ratio = ratio;
        stats.vsa_low2max = low2max;
        stats.vsa_high2min = high2min;
    }

    stats
}

fn valid_volumes(data: &[MinutePoint]) -> Vec<f64> {
    data.iter().filter_map(|point| point.vol).collect()
}

fn logvol_stats(data: &[MinutePoint]) -> (Option<f64>, Option<f64>, Option<f64>) {
    let pairs = data
        .iter()
        .filter_map(|point| {
            let vol = point.vol?;
            (vol > 0.0).then_some((vol.ln(), vol))
        })
        .collect::<Vec<_>>();
    if pairs.is_empty() {
        return (None, None, None);
    }

    let logs = pairs.iter().map(|(log, _)| *log).collect::<Vec<_>>();
    let skew = skewness(&logs);
    let total_vol = pairs.iter().map(|(_, vol)| *vol).sum::<f64>();
    if total_vol <= EPS {
        return (skew, None, None);
    }

    let q90 = quantile(&logs, 0.90);
    let q10 = quantile(&logs, 0.10);
    let tail90 = q90.map(|q| {
        pairs
            .iter()
            .filter(|(log, _)| *log >= q)
            .map(|(_, vol)| *vol)
            .sum::<f64>()
            / total_vol
    });
    let tail10 = q10.map(|q| {
        pairs
            .iter()
            .filter(|(log, _)| *log <= q)
            .map(|(_, vol)| *vol)
            .sum::<f64>()
            / total_vol
    });
    (
        skew,
        tail90.and_then(finite_value),
        tail10.and_then(finite_value),
    )
}

fn volroc_stats(data: &[MinutePoint]) -> (Option<f64>, Option<f64>) {
    let bucket_sums = data
        .chunks(FIFTEEN_MINUTE_BUCKET_SIZE)
        .take(FIFTEEN_MINUTE_BUCKETS)
        .filter(|chunk| chunk.len() == FIFTEEN_MINUTE_BUCKET_SIZE)
        .map(|chunk| chunk.iter().filter_map(|point| point.vol).sum::<f64>())
        .collect::<Vec<_>>();
    if bucket_sums.len() < 2 {
        return (None, None);
    }
    let changes = bucket_sums
        .windows(2)
        .filter_map(|pair| (pair[0].abs() > EPS).then_some(pair[1] / pair[0] - 1.0))
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    (skewness(&changes), kurtosis(&changes))
}

fn cumsumvol_stats(data: &[MinutePoint]) -> (Option<f64>, Option<f64>) {
    let volumes = valid_volumes(data);
    let total = volumes.iter().sum::<f64>();
    if volumes.is_empty() || total <= EPS {
        return (None, None);
    }
    let mut cumulative = 0.0;
    let shares = volumes
        .iter()
        .map(|vol| {
            cumulative += vol;
            cumulative / total
        })
        .collect::<Vec<_>>();
    (mean(&shares), std_dev(&shares))
}

fn volume_entropy(volumes: &[f64]) -> Option<f64> {
    if volumes.is_empty() {
        return None;
    }
    let min_value = volumes.iter().copied().reduce(f64::min)?;
    let max_value = volumes.iter().copied().reduce(f64::max)?;
    if !min_value.is_finite() || !max_value.is_finite() {
        return None;
    }
    if (max_value - min_value).abs() <= EPS {
        return Some(0.0);
    }

    let width = (max_value - min_value) / ENTROPY_BINS as f64;
    let mut counts = [0usize; ENTROPY_BINS];
    for vol in volumes {
        let mut bin = ((vol - min_value) / width).floor() as usize;
        if bin >= ENTROPY_BINS {
            bin = ENTROPY_BINS - 1;
        }
        counts[bin] += 1;
    }

    let total = volumes.len() as f64;
    let entropy = counts
        .iter()
        .filter(|count| **count > 0)
        .map(|count| {
            let p = *count as f64 / total;
            -p * p.ln()
        })
        .sum::<f64>();
    finite_value(entropy)
}

// We do not simulate 500 bootstrap samples here. For a bootstrap sample of size T
// drawn from the empirical distribution, the maximum has an exact empirical CDF:
// P(max <= u_j) = F_n(u_j)^T. Computing this distribution removes Monte Carlo
// noise and avoids generating 500 * T sampled volumes for every stock-day.
fn bootstrap_max_adjusted_stats(volumes: &[f64]) -> (Option<f64>, Option<f64>) {
    if volumes.is_empty() {
        return (None, None);
    }
    let Some(mean_vol) = mean(volumes) else {
        return (None, None);
    };

    let mut sorted = volumes.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let total = sorted.len() as f64;
    let exponent = sorted.len() as f64;
    let mut cumulative_count = 0usize;
    let mut previous_cdf = 0.0;
    let mut expected = 0.0;
    let mut expected_square = 0.0;
    let mut idx = 0usize;

    while idx < sorted.len() {
        let value = sorted[idx];
        let mut next = idx + 1;
        while next < sorted.len() && sorted[next] == value {
            next += 1;
        }
        cumulative_count += next - idx;
        let cdf = (cumulative_count as f64 / total).powf(exponent);
        let probability = (cdf - previous_cdf).max(0.0);
        expected += value * probability;
        expected_square += value * value * probability;
        previous_cdf = cdf;
        idx = next;
    }

    let variance = (expected_square - expected * expected).max(0.0);
    (
        finite_value(expected - mean_vol),
        finite_value(variance.sqrt()),
    )
}

fn vsa_stats(data: &[MinutePoint]) -> (Option<f64>, Option<f64>, Option<f64>) {
    let mut price_volumes = data
        .iter()
        .filter_map(|point| Some((point.close?, point.vol?)))
        .filter(|(close, vol)| close.is_finite() && *close > 0.0 && vol.is_finite() && *vol >= 0.0)
        .collect::<Vec<_>>();
    if price_volumes.is_empty() {
        return (None, None, None);
    }
    price_volumes.sort_by(|left, right| left.0.total_cmp(&right.0));

    let mut levels = Vec::<(f64, f64)>::new();
    for (price, vol) in price_volumes {
        if let Some((last_price, last_vol)) = levels.last_mut() {
            if *last_price == price {
                *last_vol += vol;
                continue;
            }
        }
        levels.push((price, vol));
    }

    let total_vol = levels.iter().map(|(_, vol)| *vol).sum::<f64>();
    if total_vol <= EPS {
        return (None, None, None);
    }
    let close_t = data.iter().rev().find_map(|point| point.close);
    let max_close = data.iter().filter_map(|point| point.close).reduce(f64::max);
    let min_close = data.iter().filter_map(|point| point.close).reduce(f64::min);
    let Some((vsp_idx, (vsp_price, _))) =
        levels.iter().enumerate().max_by(|(_, left), (_, right)| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| right.0.total_cmp(&left.0))
        })
    else {
        return (None, None, None);
    };

    let mut selected = vec![false; levels.len()];
    selected[vsp_idx] = true;
    let mut cumulative = levels[vsp_idx].1;
    let mut candidates = levels
        .iter()
        .enumerate()
        .filter(|(idx, _)| *idx != vsp_idx)
        .map(|(idx, (price, vol))| (idx, (price - vsp_price).abs(), *vol, *price))
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| right.2.total_cmp(&left.2))
            .then_with(|| left.3.total_cmp(&right.3))
    });
    for (idx, _, _, _) in candidates {
        if cumulative / total_vol > 0.5 {
            break;
        }
        selected[idx] = true;
        cumulative += levels[idx].1;
    }

    let selected_prices = levels
        .iter()
        .zip(selected)
        .filter_map(|((price, _), selected)| selected.then_some(*price))
        .collect::<Vec<_>>();
    let vsa_low = selected_prices.iter().copied().reduce(f64::min);
    let vsa_high = selected_prices.iter().copied().reduce(f64::max);

    (
        safe_div(vsa_low, close_t),
        safe_div(vsa_low, max_close),
        safe_div(vsa_high, min_close),
    )
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

fn central_moment(values: &[f64], order: i32) -> Option<f64> {
    let mean = mean(values)?;
    finite_value(
        values
            .iter()
            .map(|value| (value - mean).powi(order))
            .sum::<f64>()
            / values.len() as f64,
    )
}

fn skewness(values: &[f64]) -> Option<f64> {
    let std = std_dev(values)?;
    (std > EPS).then(|| central_moment(values, 3).map(|moment| moment / std.powi(3)))?
}

fn kurtosis(values: &[f64]) -> Option<f64> {
    let std = std_dev(values)?;
    (std > EPS).then(|| central_moment(values, 4).map(|moment| moment / std.powi(4)))?
}

fn quantile(values: &[f64], q: f64) -> Option<f64> {
    let mut values = values.to_vec();
    crate::factor::common::quantile_linear(&mut values, q)
}

fn safe_div(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (numerator, denominator) {
        (Some(numerator), Some(denominator)) if denominator.abs() > EPS => {
            finite_value(numerator / denominator)
        }
        _ => None,
    }
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

    fn point(close: f64, vol: f64) -> MinutePoint {
        MinutePoint {
            close: Some(close),
            vol: Some(vol),
        }
    }

    #[test]
    fn xyzq_volume_logvol_tails_use_volume_share_and_left_tail_direction() {
        let data = [
            point(1.0, 1.0),
            point(1.0, 2.0),
            point(1.0, 3.0),
            point(1.0, 4.0),
        ];
        let (_, tail90, tail10) = logvol_stats(&data);

        assert_close(tail90, Some(4.0 / 10.0));
        assert_close(tail10, Some(1.0 / 10.0));
    }

    #[test]
    fn xyzq_volume_volroc_uses_sixteen_15_minute_buckets() {
        let mut data = Vec::new();
        for bucket in 1..=FIFTEEN_MINUTE_BUCKETS {
            for _ in 0..FIFTEEN_MINUTE_BUCKET_SIZE {
                data.push(point(1.0, bucket as f64));
            }
        }
        let (skew, kurt) = volroc_stats(&data);

        assert!(skew.is_some());
        assert!(kurt.is_some());
    }

    #[test]
    fn xyzq_volume_cumsum_stats_match_small_sample() {
        let data = [point(1.0, 1.0), point(1.0, 1.0), point(1.0, 2.0)];
        let (mean, std) = cumsumvol_stats(&data);
        let shares = [0.25, 0.5, 1.0];

        assert_close(mean, super::mean(&shares));
        assert_close(std, std_dev(&shares));
    }

    #[test]
    fn xyzq_volume_entropy_handles_constant_volume() {
        let volumes = vec![3.0; 10];
        assert_close(volume_entropy(&volumes), Some(0.0));
    }

    #[test]
    fn xyzq_volume_bootstrap_max_distribution_handles_repeats_and_zeroes() {
        let values = [0.0, 0.0, 2.0, 2.0];
        let (mean, std) = bootstrap_max_adjusted_stats(&values);

        let p0 = (2.0_f64 / 4.0).powf(4.0);
        let p2 = 1.0 - p0;
        let expected_max = 2.0 * p2;
        let expected_square = 4.0 * p2;
        let expected_std = (expected_square - expected_max * expected_max).sqrt();
        assert_close(mean, Some(expected_max - 1.0));
        assert_close(std, Some(expected_std));

        let (mean, std) = bootstrap_max_adjusted_stats(&[5.0, 5.0, 5.0]);
        assert_close(mean, Some(0.0));
        assert_close(std, Some(0.0));
    }

    #[test]
    fn xyzq_volume_vsa_expands_by_distance_then_volume() {
        let data = [
            point(10.0, 10.0),
            point(11.0, 50.0),
            point(12.0, 20.0),
            point(9.0, 30.0),
        ];
        let (ratio, low2max, high2min) = vsa_stats(&data);

        assert_close(ratio, Some(11.0 / 9.0));
        assert_close(low2max, Some(11.0 / 12.0));
        assert_close(high2min, Some(12.0 / 9.0));
    }

    #[test]
    fn xyzq_volume_factor_spec_uses_shared_raw_lookback() {
        let def = XyzqVolumeFactorDef {
            id: "vol_entropy",
            alias: "vol_entropy",
            name: "vol_entropy",
            raw_id: VOL_ENTROPY_SHAPE_RAW_ID,
            window: ENTROPY_WINDOW,
            aggregation: XyzqVolumeAggregation::Std,
        };
        let spec = factor_spec(def);

        assert_eq!(spec.lookback.trading_days, SHARED_RAW_LOOKBACK);
        assert_eq!(
            spec.intraday_raw_dependencies[0].daily_lookback,
            SHARED_RAW_LOOKBACK
        );
    }
}
