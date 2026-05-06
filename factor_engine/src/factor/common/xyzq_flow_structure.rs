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
    CUTVOL_ENTROPY_RAW_ID, CUTVOL_RTN_MEAN_RAW_ID, CUTVOL_RTN_VAR_RAW_ID, CUTVOL_TIME_COR_RAW_ID,
    CUTVOL_TIME_MEAN_RAW_ID, CUTVOL_TIME_VAR_RAW_ID, RHL_COR_RAW_ID, RVC_COR_RAW_ID, TE_R2V_RAW_ID,
    TE_V2R_RAW_ID, VOH_COR_RAW_ID, VOL_COR_RAW_ID,
};
use crate::factor::common::{clean_intraday_value, intraday_time_in_range, stock_minute_raw_spec};
use crate::operators::{cs_pctrank, ts_mean};

pub const RAW_VERSION: &str = "0.1.0";
pub const VERSION: &str = "0.1.0";

const RAW_WINDOW_DAYS: usize = 1;
const DEFAULT_WINDOW: usize = 15;
const TE_WINDOW: usize = 20;
const SHARED_RAW_LOOKBACK: usize = TE_WINDOW - 1;
const MIN_PERIODS: usize = 1;
const START_TIME: &str = "09:41:00";
const END_TIME: &str = "14:50:00";
const TE_MAX_LAG: usize = 5;
const CUTVOL_BUCKETS: usize = 50;
const CUTVOL_ENTROPY_BINS: usize = 5;
const EPS: f64 = f64::EPSILON;

#[derive(Clone, Copy, Debug)]
pub struct XyzqFlowFactorDef {
    pub id: &'static str,
    pub alias: &'static str,
    pub name: &'static str,
    pub raw_id: &'static str,
    pub window: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum XyzqFlowRawFamily {
    Correlation,
    TransferEntropy,
    CutVol,
}

#[derive(Clone, Copy, Debug, Default)]
struct FlowStats {
    rvc_cor: Option<f64>,
    rhl_cor: Option<f64>,
    voh_cor: Option<f64>,
    vol_cor: Option<f64>,
    te_v2r: Option<f64>,
    te_r2v: Option<f64>,
    cutvol_rtn_mean: Option<f64>,
    cutvol_rtn_var: Option<f64>,
    cutvol_time_mean: Option<f64>,
    cutvol_time_var: Option<f64>,
    cutvol_time_cor: Option<f64>,
    cutvol_entropy: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
struct MinutePoint {
    in_window: bool,
    close: Option<f64>,
    high: Option<f64>,
    low: Option<f64>,
    vol: Option<f64>,
}

pub const fn default_window() -> usize {
    DEFAULT_WINDOW
}

pub const fn te_window() -> usize {
    TE_WINDOW
}

pub fn all_raw_ids() -> [&'static str; 12] {
    [
        RVC_COR_RAW_ID,
        RHL_COR_RAW_ID,
        VOH_COR_RAW_ID,
        VOL_COR_RAW_ID,
        TE_V2R_RAW_ID,
        TE_R2V_RAW_ID,
        CUTVOL_RTN_MEAN_RAW_ID,
        CUTVOL_RTN_VAR_RAW_ID,
        CUTVOL_TIME_MEAN_RAW_ID,
        CUTVOL_TIME_VAR_RAW_ID,
        CUTVOL_TIME_COR_RAW_ID,
        CUTVOL_ENTROPY_RAW_ID,
    ]
}

pub fn correlation_raw_ids() -> [&'static str; 4] {
    [
        RVC_COR_RAW_ID,
        RHL_COR_RAW_ID,
        VOH_COR_RAW_ID,
        VOL_COR_RAW_ID,
    ]
}

pub fn transfer_entropy_raw_ids() -> [&'static str; 2] {
    [TE_V2R_RAW_ID, TE_R2V_RAW_ID]
}

pub fn cutvol_raw_ids() -> [&'static str; 6] {
    [
        CUTVOL_RTN_MEAN_RAW_ID,
        CUTVOL_RTN_VAR_RAW_ID,
        CUTVOL_TIME_MEAN_RAW_ID,
        CUTVOL_TIME_VAR_RAW_ID,
        CUTVOL_TIME_COR_RAW_ID,
        CUTVOL_ENTROPY_RAW_ID,
    ]
}

fn raw_ids_for_family(family: XyzqFlowRawFamily) -> Vec<&'static str> {
    match family {
        XyzqFlowRawFamily::Correlation => correlation_raw_ids().to_vec(),
        XyzqFlowRawFamily::TransferEntropy => transfer_entropy_raw_ids().to_vec(),
        XyzqFlowRawFamily::CutVol => cutvol_raw_ids().to_vec(),
    }
}

pub fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
    stock_minute_raw_spec(
        raw_id,
        RAW_VERSION,
        &["close", "high", "low", "vol"],
        RAW_WINDOW_DAYS,
    )
}

pub fn raw_specs() -> Vec<IntradayDailyRawSpec> {
    all_raw_ids()
        .iter()
        .map(|raw_id| raw_spec(raw_id))
        .collect()
}

pub fn correlation_raw_specs() -> Vec<IntradayDailyRawSpec> {
    correlation_raw_ids()
        .iter()
        .map(|raw_id| raw_spec(raw_id))
        .collect()
}

pub fn transfer_entropy_raw_specs() -> Vec<IntradayDailyRawSpec> {
    transfer_entropy_raw_ids()
        .iter()
        .map(|raw_id| raw_spec(raw_id))
        .collect()
}

pub fn cutvol_raw_specs() -> Vec<IntradayDailyRawSpec> {
    cutvol_raw_ids()
        .iter()
        .map(|raw_id| raw_spec(raw_id))
        .collect()
}

pub fn factor_spec(def: XyzqFlowFactorDef) -> FactorSpec {
    FactorSpec {
        id: def.id.to_string(),
        aliases: vec![def.alias.to_string()],
        name: def.name.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: tags(),
        description: format!(
            "{} from intraday return-volume flow raw, rolling mean, cross-sectional percentile rank, and SIZE/SW-sector neutralization.",
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

pub fn compute_factor(def: XyzqFlowFactorDef, data: &DataPool) -> Result<FactorSeries> {
    let panel = data.intraday_daily_raw_panel(def.raw_id)?;
    let raw = panel.column(def.raw_id)?;
    let smoothed = raw.ts(|values| ts_mean(values, def.window, MIN_PERIODS))?;
    let ranked = smoothed.cs(|values| cs_pctrank(values, true))?;
    let factor = neutralize_size_sector(&ranked, &panel, data)?;
    Ok(factor.to_factor_series(factor_spec(def)))
}

#[macro_export]
macro_rules! define_xyzq_flow_structure_factor {
    ($struct_name:ident, $id:expr, $alias:expr, $name:expr, $raw_id:expr, $window:expr) => {
        const DEF: $crate::factor::common::xyzq_flow_structure::XyzqFlowFactorDef =
            $crate::factor::common::xyzq_flow_structure::XyzqFlowFactorDef {
                id: $id,
                alias: $alias,
                name: $name,
                raw_id: $raw_id,
                window: $window,
            };

        pub struct $struct_name;

        pub fn create() -> Box<dyn $crate::factor::Factor> {
            Box::new($struct_name)
        }

        impl $crate::factor::Factor for $struct_name {
            fn spec(&self) -> $crate::core::FactorSpec {
                $crate::factor::common::xyzq_flow_structure::factor_spec(DEF)
            }

            fn compute(
                &self,
                _context: &$crate::core::FactorContext,
                data: &$crate::data::DataPool,
            ) -> $crate::error::Result<$crate::core::FactorSeries> {
                $crate::factor::common::xyzq_flow_structure::compute_factor(DEF, data)
            }
        }
    };
}

pub fn minute_compute_many(
    raw_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
) -> Result<Vec<IntradayDailyRawSeries>> {
    minute_compute_many_for(raw_ids, context, data, XyzqFlowRawFamily::Correlation)
}

pub fn minute_compute_many_for(
    raw_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
    family: XyzqFlowRawFamily,
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
        let high = table.required_f64_cast("high")?;
        let low = table.required_f64_cast("low")?;
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

        for (ts_code, mut indices) in grouped {
            indices.sort_by(|left, right| trade_times[*left].cmp(&trade_times[*right]));
            let points =
                minute_points_from_indices(&indices, &trade_times, &close, &high, &low, &vol);
            let stats = flow_stats_for(&points, family);
            let key = FactorRowKey::Daily {
                trade_date: *trade_date,
                ts_code,
            };

            push_requested(&mut values, &requested, RVC_COR_RAW_ID, &key, stats.rvc_cor);
            push_requested(&mut values, &requested, RHL_COR_RAW_ID, &key, stats.rhl_cor);
            push_requested(&mut values, &requested, VOH_COR_RAW_ID, &key, stats.voh_cor);
            push_requested(&mut values, &requested, VOL_COR_RAW_ID, &key, stats.vol_cor);
            push_requested(&mut values, &requested, TE_V2R_RAW_ID, &key, stats.te_v2r);
            push_requested(&mut values, &requested, TE_R2V_RAW_ID, &key, stats.te_r2v);
            push_requested(
                &mut values,
                &requested,
                CUTVOL_RTN_MEAN_RAW_ID,
                &key,
                stats.cutvol_rtn_mean,
            );
            push_requested(
                &mut values,
                &requested,
                CUTVOL_RTN_VAR_RAW_ID,
                &key,
                stats.cutvol_rtn_var,
            );
            push_requested(
                &mut values,
                &requested,
                CUTVOL_TIME_MEAN_RAW_ID,
                &key,
                stats.cutvol_time_mean,
            );
            push_requested(
                &mut values,
                &requested,
                CUTVOL_TIME_VAR_RAW_ID,
                &key,
                stats.cutvol_time_var,
            );
            push_requested(
                &mut values,
                &requested,
                CUTVOL_TIME_COR_RAW_ID,
                &key,
                stats.cutvol_time_cor,
            );
            push_requested(
                &mut values,
                &requested,
                CUTVOL_ENTROPY_RAW_ID,
                &key,
                stats.cutvol_entropy,
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
        "volume",
        "intraday",
        "minute_agg",
        "transfer_entropy",
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
    high: &[Option<f64>],
    low: &[Option<f64>],
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
                high: clean_intraday_value(high[*idx]).filter(|value| *value > 0.0),
                low: clean_intraday_value(low[*idx]).filter(|value| *value > 0.0),
                vol: clean_intraday_value(vol[*idx]).filter(|value| *value >= 0.0),
            }
        })
        .collect()
}

fn flow_stats_for(points: &[MinutePoint], family: XyzqFlowRawFamily) -> FlowStats {
    match family {
        XyzqFlowRawFamily::Correlation => correlation_stats(points),
        XyzqFlowRawFamily::TransferEntropy => transfer_entropy_stats(points),
        XyzqFlowRawFamily::CutVol => cutvol_flow_stats(points),
    }
}

fn correlation_stats(points: &[MinutePoint]) -> FlowStats {
    let returns = simple_returns(points);
    let selected = points
        .iter()
        .enumerate()
        .filter_map(|(idx, point)| point.in_window.then_some(idx))
        .collect::<Vec<_>>();
    let selected_returns = selected.iter().map(|idx| returns[*idx]).collect::<Vec<_>>();
    let selected_vol = selected
        .iter()
        .map(|idx| points[*idx].vol)
        .collect::<Vec<_>>();
    let selected_hl = selected
        .iter()
        .map(|idx| high_low_change(points[*idx].high, points[*idx].low))
        .collect::<Vec<_>>();
    let vol_pct = volume_pct(&selected_vol);

    FlowStats {
        rvc_cor: pair_corr(&selected_returns, &vol_pct),
        rhl_cor: pair_corr(&selected_returns, &selected_hl),
        voh_cor: conditional_pair_corr(&selected_returns, &vol_pct, |ret| ret > 0.0),
        vol_cor: conditional_pair_corr(&selected_returns, &vol_pct, |ret| ret < 0.0),
        ..FlowStats::default()
    }
}

fn transfer_entropy_stats(points: &[MinutePoint]) -> FlowStats {
    let rtn5 = five_minute_returns(points);
    let vol5 = five_minute_volume_sum(points);
    let selected = points
        .iter()
        .enumerate()
        .filter_map(|(idx, point)| point.in_window.then_some(idx))
        .collect::<Vec<_>>();
    let selected_rtn5 = selected.iter().map(|idx| rtn5[*idx]).collect::<Vec<_>>();
    let selected_vol5 = selected.iter().map(|idx| vol5[*idx]).collect::<Vec<_>>();

    FlowStats {
        te_v2r: transfer_entropy_mean(&selected_vol5, &selected_rtn5),
        te_r2v: transfer_entropy_mean(&selected_rtn5, &selected_vol5),
        ..FlowStats::default()
    }
}

fn cutvol_flow_stats(points: &[MinutePoint]) -> FlowStats {
    let (
        cutvol_rtn_mean,
        cutvol_rtn_var,
        cutvol_time_mean,
        cutvol_time_var,
        cutvol_time_cor,
        cutvol_entropy,
    ) = cutvol_stats(points);

    FlowStats {
        cutvol_rtn_mean,
        cutvol_rtn_var,
        cutvol_time_mean,
        cutvol_time_var,
        cutvol_time_cor,
        cutvol_entropy,
        ..FlowStats::default()
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

fn five_minute_returns(points: &[MinutePoint]) -> Vec<Option<f64>> {
    let mut returns = vec![None; points.len()];
    for idx in 5..points.len() {
        returns[idx] = match (points[idx].close, points[idx - 5].close) {
            (Some(current), Some(previous)) if previous.abs() > EPS => {
                finite_value(current / previous - 1.0)
            }
            _ => None,
        };
    }
    returns
}

fn five_minute_volume_sum(points: &[MinutePoint]) -> Vec<Option<f64>> {
    let mut output = vec![None; points.len()];
    for idx in 4..points.len() {
        let mut sum = 0.0;
        let mut valid = true;
        for point in &points[idx - 4..=idx] {
            match point.vol {
                Some(vol) => sum += vol,
                None => {
                    valid = false;
                    break;
                }
            }
        }
        if valid {
            output[idx] = finite_value(sum);
        }
    }
    output
}

fn high_low_change(high: Option<f64>, low: Option<f64>) -> Option<f64> {
    match (high, low) {
        (Some(high), Some(low)) if low.abs() > EPS => finite_value(high / low - 1.0),
        _ => None,
    }
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

fn pair_corr(left: &[Option<f64>], right: &[Option<f64>]) -> Option<f64> {
    let pairs = left
        .iter()
        .zip(right)
        .filter_map(|(left, right)| Some(((*left).as_ref().copied()?, (*right).as_ref().copied()?)))
        .collect::<Vec<_>>();
    pearson_pairs(&pairs)
}

fn conditional_pair_corr<F>(
    left: &[Option<f64>],
    right: &[Option<f64>],
    predicate: F,
) -> Option<f64>
where
    F: Fn(f64) -> bool,
{
    let pairs = left
        .iter()
        .zip(right)
        .filter_map(|(left, right)| {
            let left = (*left).as_ref().copied()?;
            let right = (*right).as_ref().copied()?;
            predicate(left).then_some((left, right))
        })
        .collect::<Vec<_>>();
    pearson_pairs(&pairs)
}

fn transfer_entropy_mean(
    source_values: &[Option<f64>],
    target_values: &[Option<f64>],
) -> Option<f64> {
    let source_states = states_from_values(source_values)?;
    let target_states = states_from_values(target_values)?;
    let values = (1..=TE_MAX_LAG)
        .filter_map(|lag| transfer_entropy(&source_states, &target_states, lag))
        .collect::<Vec<_>>();
    mean(&values)
}

fn states_from_values(values: &[Option<f64>]) -> Option<Vec<Option<usize>>> {
    let valid = values.iter().filter_map(|value| *value).collect::<Vec<_>>();
    let q30 = quantile(&valid, 0.30)?;
    let q70 = quantile(&valid, 0.70)?;
    Some(
        values
            .iter()
            .map(|value| {
                let value = (*value).as_ref().copied()?;
                if value < q30 {
                    Some(0)
                } else if value > q70 {
                    Some(2)
                } else {
                    Some(1)
                }
            })
            .collect(),
    )
}

fn transfer_entropy(source: &[Option<usize>], target: &[Option<usize>], lag: usize) -> Option<f64> {
    if source.len() != target.len() || lag == 0 || source.len() <= lag {
        return None;
    }
    let mut joint = [[0usize; 3]; 3];
    let mut source_counts = [0usize; 3];
    let mut target_counts = [0usize; 3];
    let mut total = 0usize;
    for idx in lag..target.len() {
        let Some(x) = source[idx - lag] else {
            continue;
        };
        let Some(y) = target[idx] else {
            continue;
        };
        if x >= 3 || y >= 3 {
            continue;
        }
        joint[x][y] += 1;
        source_counts[x] += 1;
        target_counts[y] += 1;
        total += 1;
    }
    if total == 0 {
        return None;
    }
    let total_f = total as f64;
    let mut te = 0.0;
    for x in 0..3 {
        for y in 0..3 {
            let count = joint[x][y];
            if count == 0 || source_counts[x] == 0 || target_counts[y] == 0 {
                continue;
            }
            let p_xy = count as f64 / total_f;
            let p_y_given_x = count as f64 / source_counts[x] as f64;
            let p_y = target_counts[y] as f64 / total_f;
            te += p_xy * (p_y_given_x / p_y).ln();
        }
    }
    finite_value(te)
}

fn cutvol_stats(
    points: &[MinutePoint],
) -> (
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
) {
    let rows = points
        .iter()
        .filter(|point| point.in_window)
        .filter_map(|point| Some((point.close?, point.vol?)))
        .filter(|(close, vol)| close.is_finite() && *close > 0.0 && vol.is_finite() && *vol >= 0.0)
        .collect::<Vec<_>>();
    let total_vol = rows.iter().map(|(_, vol)| *vol).sum::<f64>();
    if total_vol <= EPS {
        return (None, None, None, None, None, None);
    }
    let threshold = total_vol / CUTVOL_BUCKETS as f64;
    if threshold <= EPS {
        return (None, None, None, None, None, None);
    }
    let buckets = cutvol_buckets(&rows, threshold);
    if buckets.len() != CUTVOL_BUCKETS || buckets.iter().any(|bucket| bucket.volume <= EPS) {
        return (None, None, None, None, None, None);
    }
    let close = buckets
        .iter()
        .map(|bucket| bucket.weighted_close / bucket.volume)
        .collect::<Vec<_>>();
    let intervals = buckets
        .iter()
        .map(|bucket| bucket.interval as f64)
        .collect::<Vec<_>>();
    let returns = close
        .windows(2)
        .filter_map(|pair| (pair[0].abs() > EPS).then_some(pair[1] / pair[0] - 1.0))
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let signed_intervals = intervals
        .iter()
        .skip(1)
        .zip(&returns)
        .map(|(interval, ret)| interval * sign(*ret))
        .collect::<Vec<_>>();

    (
        mean(&returns),
        variance(&returns),
        mean(&intervals),
        variance(&intervals),
        lag_corr_f64(&intervals),
        entropy(&signed_intervals, CUTVOL_ENTROPY_BINS),
    )
}

#[derive(Clone, Copy, Debug, Default)]
struct CutVolBucket {
    volume: f64,
    weighted_close: f64,
    interval: usize,
}

fn cutvol_buckets(rows: &[(f64, f64)], threshold: f64) -> Vec<CutVolBucket> {
    let mut buckets = Vec::with_capacity(CUTVOL_BUCKETS);
    let mut current = CutVolBucket::default();
    for (close, volume) in rows {
        let mut remaining = *volume;
        while remaining > EPS && buckets.len() < CUTVOL_BUCKETS {
            let need = threshold - current.volume;
            let used = remaining.min(need);
            current.volume += used;
            current.weighted_close += close * used;
            current.interval += 1;
            remaining -= used;
            if current.volume + EPS >= threshold {
                buckets.push(current);
                current = CutVolBucket::default();
            }
        }
        if buckets.len() >= CUTVOL_BUCKETS {
            break;
        }
        if remaining <= EPS && *volume <= EPS && buckets.len() < CUTVOL_BUCKETS {
            current.interval += 1;
        }
    }
    if buckets.len() < CUTVOL_BUCKETS && current.volume > EPS {
        buckets.push(current);
    }
    buckets
}

fn sign(value: f64) -> f64 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        0.0
    }
}

fn lag_corr_f64(values: &[f64]) -> Option<f64> {
    let pairs = values
        .windows(2)
        .map(|pair| (pair[0], pair[1]))
        .collect::<Vec<_>>();
    pearson_pairs(&pairs)
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

fn entropy(values: &[f64], bins: usize) -> Option<f64> {
    if values.is_empty() || bins == 0 {
        return None;
    }
    let min_value = values.iter().copied().reduce(f64::min)?;
    let max_value = values.iter().copied().reduce(f64::max)?;
    if !min_value.is_finite() || !max_value.is_finite() {
        return None;
    }
    if (max_value - min_value).abs() <= EPS {
        return Some(0.0);
    }
    let width = (max_value - min_value) / bins as f64;
    let mut counts = vec![0usize; bins];
    for value in values {
        let mut bin = ((value - min_value) / width).floor() as usize;
        if bin >= bins {
            bin = bins - 1;
        }
        counts[bin] += 1;
    }
    let total = values.len() as f64;
    finite_value(
        counts
            .iter()
            .filter(|count| **count > 0)
            .map(|count| {
                let p = *count as f64 / total;
                -p * p.ln()
            })
            .sum(),
    )
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    finite_value(values.iter().sum::<f64>() / values.len() as f64)
}

fn variance(values: &[f64]) -> Option<f64> {
    let mean = mean(values)?;
    finite_value(
        values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / values.len() as f64,
    )
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

    fn point(close: f64, high: f64, low: f64, vol: f64, in_window: bool) -> MinutePoint {
        MinutePoint {
            in_window,
            close: Some(close),
            high: Some(high),
            low: Some(low),
            vol: Some(vol),
        }
    }

    #[test]
    fn xyzq_flow_returns_are_computed_before_window_filter() {
        let points = [
            point(100.0, 101.0, 99.0, 1.0, false),
            point(110.0, 111.0, 109.0, 1.0, true),
            point(121.0, 122.0, 120.0, 1.0, true),
        ];
        let returns = simple_returns(&points);

        assert_close(returns[1], Some(0.1));
    }

    #[test]
    fn xyzq_flow_correlation_inputs_use_window_volume_pct() {
        let left = [Some(1.0), Some(2.0), Some(3.0)];
        let pct = volume_pct(&[Some(1.0), Some(2.0), Some(3.0)]);

        assert_close(pair_corr(&left, &pct), Some(1.0));
    }

    #[test]
    fn xyzq_flow_transfer_entropy_detects_source_information() {
        let source = vec![
            Some(0.0),
            Some(0.0),
            Some(1.0),
            Some(1.0),
            Some(2.0),
            Some(2.0),
        ];
        let target = vec![
            Some(0.0),
            Some(1.0),
            Some(1.0),
            Some(2.0),
            Some(2.0),
            Some(0.0),
        ];
        let value = transfer_entropy_mean(&source, &target);

        assert!(value.is_some());
    }

    #[test]
    fn xyzq_flow_cutvol_splits_crossing_minute_into_both_buckets() {
        let rows = vec![(10.0, 6.0), (20.0, 6.0), (30.0, 6.0), (40.0, 6.0)];
        let buckets = cutvol_buckets(&rows, 4.0);

        assert_eq!(buckets.len(), 6);
        assert_eq!(buckets[0].interval, 1);
        assert_eq!(buckets[1].interval, 2);
    }

    #[test]
    fn xyzq_flow_cutvol_stats_return_all_components() {
        let points = (0..60)
            .map(|idx| {
                point(
                    10.0 + idx as f64,
                    11.0 + idx as f64,
                    9.0 + idx as f64,
                    (idx % 7 + 1) as f64,
                    true,
                )
            })
            .collect::<Vec<_>>();
        let stats = cutvol_stats(&points);

        assert!(stats.0.is_some());
        assert!(stats.1.is_some());
        assert!(stats.2.is_some());
        assert!(stats.3.is_some());
        assert!(stats.4.is_some());
        assert!(stats.5.is_some());
    }

    #[test]
    fn xyzq_flow_factor_spec_uses_shared_raw_lookback() {
        let def = XyzqFlowFactorDef {
            id: "te_v2r",
            alias: "te_v2r",
            name: "te_v2r",
            raw_id: TE_V2R_RAW_ID,
            window: TE_WINDOW,
        };
        let spec = factor_spec(def);

        assert_eq!(spec.lookback.trading_days, SHARED_RAW_LOOKBACK);
        assert_eq!(
            spec.intraday_raw_dependencies[0].daily_lookback,
            SHARED_RAW_LOOKBACK
        );
    }
}
