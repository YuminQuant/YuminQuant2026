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
    DIFF_IDX_RAW_ID, DIFF_STD_RAW_ID, DIFF_VOL_RAW_ID, LH_RTN_DIFF_RAW_ID, LH_STD_DIFF_RAW_ID,
    LH_VOL_DIFF_RAW_ID,
};
use crate::factor::common::{clean_intraday_value, intraday_time_in_range, stock_minute_raw_spec};
use crate::operators::{cs_rank, ts_mean};

pub const RAW_VERSION: &str = "0.2.0";
pub const VERSION: &str = "0.2.0";

const RAW_WINDOW_DAYS: usize = 1;
const WINDOW: usize = 15;
const LOOKBACK: usize = WINDOW - 1;
const MIN_PERIODS: usize = 1;
const OPEN_START: &str = "09:30:00";
const OPEN_SAMPLE_START: &str = "09:31:00";
const OPEN_END: &str = "10:00:00";
const CLOSE_START: &str = "14:30:00";
const CLOSE_SAMPLE_START: &str = "14:31:00";
const CLOSE_END: &str = "15:00:00";
const POST_OPEN_START: &str = "09:31:00";
const ROLLING_WINDOW: usize = 15;
const EPS: f64 = f64::EPSILON;

#[derive(Clone, Copy, Debug)]
pub struct XyzqIntradayContrastFactorDef {
    pub id: &'static str,
    pub alias: &'static str,
    pub name: &'static str,
    pub raw_id: &'static str,
}

#[derive(Clone, Copy, Debug)]
pub enum XyzqIntradayContrastRawFamily {
    LhIntradayDiff,
    HighLowTiming,
}

#[derive(Clone, Copy, Debug, Default)]
struct LhStats {
    lh_rtn_diff: Option<f64>,
    lh_vol_diff: Option<f64>,
    lh_std_diff: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default)]
struct HighLowStats {
    diff_idx: Option<f64>,
    diff_std: Option<f64>,
    diff_vol: Option<f64>,
}

#[derive(Clone, Debug)]
struct MinutePoint {
    time: String,
    close: Option<f64>,
    high: Option<f64>,
    low: Option<f64>,
    vol: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
struct AnalysisPoint {
    ret: Option<f64>,
    high: Option<f64>,
    low: Option<f64>,
    vol: Option<f64>,
}

pub fn all_raw_ids() -> [&'static str; 6] {
    [
        LH_RTN_DIFF_RAW_ID,
        LH_VOL_DIFF_RAW_ID,
        LH_STD_DIFF_RAW_ID,
        DIFF_IDX_RAW_ID,
        DIFF_STD_RAW_ID,
        DIFF_VOL_RAW_ID,
    ]
}

pub fn lh_raw_ids() -> [&'static str; 3] {
    [LH_RTN_DIFF_RAW_ID, LH_VOL_DIFF_RAW_ID, LH_STD_DIFF_RAW_ID]
}

pub fn high_low_raw_ids() -> [&'static str; 3] {
    [DIFF_IDX_RAW_ID, DIFF_STD_RAW_ID, DIFF_VOL_RAW_ID]
}

fn raw_ids_for_family(family: XyzqIntradayContrastRawFamily) -> Vec<&'static str> {
    match family {
        XyzqIntradayContrastRawFamily::LhIntradayDiff => lh_raw_ids().to_vec(),
        XyzqIntradayContrastRawFamily::HighLowTiming => high_low_raw_ids().to_vec(),
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

pub fn lh_raw_specs() -> Vec<IntradayDailyRawSpec> {
    lh_raw_ids().iter().map(|raw_id| raw_spec(raw_id)).collect()
}

pub fn high_low_raw_specs() -> Vec<IntradayDailyRawSpec> {
    high_low_raw_ids()
        .iter()
        .map(|raw_id| raw_spec(raw_id))
        .collect()
}

pub fn factor_spec(def: XyzqIntradayContrastFactorDef) -> FactorSpec {
    FactorSpec {
        id: def.id.to_string(),
        aliases: vec![def.alias.to_string()],
        name: def.name.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: tags(),
        description: format!(
            "{} from intraday open-close or high-low timing raw, cs_rank, 15-day mean, cs_rank, and SIZE/SW-sector neutralization.",
            def.name
        ),
        dependencies: dependencies(),
        intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(def.raw_id, LOOKBACK)],
        lookback: Lookback {
            trading_days: LOOKBACK,
        },
    }
}

pub fn compute_factor(def: XyzqIntradayContrastFactorDef, data: &DataPool) -> Result<FactorSeries> {
    let panel = data.intraday_daily_raw_panel(def.raw_id)?;
    let raw = panel.column(def.raw_id)?;
    let ranked_daily = raw.cs(|values| cs_rank(values, true))?;
    let smoothed = ranked_daily.ts(|values| ts_mean(values, WINDOW, MIN_PERIODS))?;
    let reranked = smoothed.cs(|values| cs_rank(values, true))?;
    let factor = neutralize_size_sector(&reranked, &panel, data)?;
    Ok(factor.to_factor_series(factor_spec(def)))
}

#[macro_export]
macro_rules! define_xyzq_intraday_contrast_factor {
    ($struct_name:ident, $id:expr, $alias:expr, $name:expr, $raw_id:expr) => {
        const DEF: $crate::factor::common::xyzq_intraday_contrast::XyzqIntradayContrastFactorDef =
            $crate::factor::common::xyzq_intraday_contrast::XyzqIntradayContrastFactorDef {
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
                $crate::factor::common::xyzq_intraday_contrast::factor_spec(DEF)
            }

            fn compute(
                &self,
                _context: &$crate::core::FactorContext,
                data: &$crate::data::DataPool,
            ) -> $crate::error::Result<$crate::core::FactorSeries> {
                $crate::factor::common::xyzq_intraday_contrast::compute_factor(DEF, data)
            }
        }
    };
}

pub fn minute_compute_many_for(
    raw_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
    family: XyzqIntradayContrastRawFamily,
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
            let key = FactorRowKey::Daily {
                trade_date: *trade_date,
                ts_code,
            };

            match family {
                XyzqIntradayContrastRawFamily::LhIntradayDiff => {
                    let stats = lh_stats(&points);
                    push_requested(
                        &mut values,
                        &requested,
                        LH_RTN_DIFF_RAW_ID,
                        &key,
                        stats.lh_rtn_diff,
                    );
                    push_requested(
                        &mut values,
                        &requested,
                        LH_VOL_DIFF_RAW_ID,
                        &key,
                        stats.lh_vol_diff,
                    );
                    push_requested(
                        &mut values,
                        &requested,
                        LH_STD_DIFF_RAW_ID,
                        &key,
                        stats.lh_std_diff,
                    );
                }
                XyzqIntradayContrastRawFamily::HighLowTiming => {
                    let stats = high_low_stats(&points);
                    push_requested(
                        &mut values,
                        &requested,
                        DIFF_IDX_RAW_ID,
                        &key,
                        stats.diff_idx,
                    );
                    push_requested(
                        &mut values,
                        &requested,
                        DIFF_STD_RAW_ID,
                        &key,
                        stats.diff_std,
                    );
                    push_requested(
                        &mut values,
                        &requested,
                        DIFF_VOL_RAW_ID,
                        &key,
                        stats.diff_vol,
                    );
                }
            }
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
        "rank",
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
        .filter_map(|idx| {
            let time = trade_times[*idx].clone()?;
            Some(MinutePoint {
                time,
                close: clean_intraday_value(close[*idx]).filter(|value| *value > 0.0),
                high: clean_intraday_value(high[*idx]).filter(|value| *value > 0.0),
                low: clean_intraday_value(low[*idx]).filter(|value| *value > 0.0),
                vol: clean_intraday_value(vol[*idx]).filter(|value| *value >= 0.0),
            })
        })
        .collect()
}

fn lh_stats(points: &[MinutePoint]) -> LhStats {
    let returns = simple_returns(points);
    let open_return = match (close_at(points, OPEN_START), close_at(points, OPEN_END)) {
        (Some(start), Some(end)) if start > EPS => finite_value(end / start - 1.0),
        _ => None,
    };
    let close_return = match (close_at(points, CLOSE_START), close_at(points, CLOSE_END)) {
        (Some(start), Some(end)) if start > EPS => finite_value(end / start - 1.0),
        _ => None,
    };
    let open_volume = volume_sum_in_window(points, OPEN_SAMPLE_START, OPEN_END);
    let close_volume = volume_sum_in_window(points, CLOSE_SAMPLE_START, CLOSE_END);
    let open_std = std_in_time_window(points, &returns, OPEN_SAMPLE_START, OPEN_END);
    let close_std = std_in_time_window(points, &returns, CLOSE_SAMPLE_START, CLOSE_END);

    LhStats {
        lh_rtn_diff: ratio_with_zero_denominator(open_return, close_return),
        lh_vol_diff: ratio_with_zero_denominator(open_volume, close_volume),
        lh_std_diff: ratio_with_zero_denominator(open_std, close_std),
    }
}

fn high_low_stats(points: &[MinutePoint]) -> HighLowStats {
    let returns = simple_returns(points);
    let points = post_open_analysis_points(points, &returns);
    let high_ranks = intraday_pct_ranks(&points.iter().map(|point| point.high).collect::<Vec<_>>());
    let low_ranks = intraday_pct_ranks(&points.iter().map(|point| point.low).collect::<Vec<_>>());
    let high_rank_sum = rolling_full_sum(&high_ranks, ROLLING_WINDOW);
    let low_rank_sum = rolling_full_sum(&low_ranks, ROLLING_WINDOW);
    let Some(high_idx) = earliest_extreme_index(&high_rank_sum, true) else {
        return HighLowStats::default();
    };
    let Some(low_idx) = earliest_extreme_index(&low_rank_sum, false) else {
        return HighLowStats::default();
    };

    let returns = points.iter().map(|point| point.ret).collect::<Vec<_>>();
    let std_high = trailing_std_at(&returns, high_idx, ROLLING_WINDOW);
    let std_low = trailing_std_at(&returns, low_idx, ROLLING_WINDOW);
    let vol_pct = volume_pct_analysis(&points);
    let vol_high = trailing_sum_at(&vol_pct, high_idx, ROLLING_WINDOW);
    let vol_low = trailing_sum_at(&vol_pct, low_idx, ROLLING_WINDOW);

    HighLowStats {
        diff_idx: Some((high_idx as isize - low_idx as isize) as f64),
        diff_std: match (std_high, std_low) {
            (Some(high), Some(low)) => finite_value(high - low),
            _ => None,
        },
        diff_vol: match (vol_high, vol_low) {
            (Some(high), Some(low)) => finite_value(high - low),
            _ => None,
        },
    }
}

fn close_at(points: &[MinutePoint], target: &str) -> Option<f64> {
    points
        .iter()
        .find(|point| intraday_time_in_range(&point.time, target, target))
        .and_then(|point| point.close)
}

fn post_open_analysis_points(
    points: &[MinutePoint],
    returns: &[Option<f64>],
) -> Vec<AnalysisPoint> {
    points
        .iter()
        .enumerate()
        .filter(|(_, point)| intraday_time_in_range(&point.time, POST_OPEN_START, CLOSE_END))
        .map(|(idx, point)| AnalysisPoint {
            ret: returns[idx],
            high: point.high,
            low: point.low,
            vol: point.vol,
        })
        .collect()
}

fn volume_sum_in_window(points: &[MinutePoint], start: &str, end: &str) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for point in points
        .iter()
        .filter(|point| intraday_time_in_range(&point.time, start, end))
    {
        let vol = point.vol?;
        sum += vol;
        count += 1;
    }
    (count > 0).then_some(sum)
}

fn std_in_time_window(
    points: &[MinutePoint],
    returns: &[Option<f64>],
    start: &str,
    end: &str,
) -> Option<f64> {
    let values = points
        .iter()
        .enumerate()
        .filter(|(_, point)| intraday_time_in_range(&point.time, start, end))
        .filter_map(|(idx, _)| returns[idx])
        .collect::<Vec<_>>();
    std_dev(&values)
}

fn ratio_with_zero_denominator(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    let numerator = numerator.filter(|value| !value.is_nan())?;
    let denominator = denominator.filter(|value| !value.is_nan())?;
    if denominator.abs() <= EPS {
        if !numerator.is_finite() {
            return None;
        }
        if numerator > 0.0 {
            Some(f64::INFINITY)
        } else if numerator < 0.0 {
            Some(f64::NEG_INFINITY)
        } else {
            Some(0.0)
        }
    } else {
        let value = numerator / denominator;
        (!value.is_nan()).then_some(value)
    }
}

fn simple_returns(points: &[MinutePoint]) -> Vec<Option<f64>> {
    let mut returns = vec![None; points.len()];
    for idx in 1..points.len() {
        returns[idx] = match (points[idx].close, points[idx - 1].close) {
            (Some(current), Some(previous)) if previous > EPS => {
                finite_value(current / previous - 1.0)
            }
            _ => None,
        };
    }
    returns
}

fn intraday_pct_ranks(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut pairs = values
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| {
            (*value)
                .filter(|value| value.is_finite())
                .map(|value| (idx, value))
        })
        .collect::<Vec<_>>();
    if pairs.len() < 2 {
        return vec![None; values.len()];
    }
    pairs.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    let denominator = pairs.len() as f64 - 1.0;
    let mut output = vec![None; values.len()];
    for (rank_idx, (idx, _)) in pairs.into_iter().enumerate() {
        output[idx] = Some(rank_idx as f64 / denominator);
    }
    output
}

fn rolling_full_sum(values: &[Option<f64>], window: usize) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    if window == 0 {
        return output;
    }
    for idx in window - 1..values.len() {
        let mut sum = 0.0;
        let mut valid = true;
        for value in &values[idx + 1 - window..=idx] {
            match *value {
                Some(value) if value.is_finite() => sum += value,
                _ => {
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

fn earliest_extreme_index(values: &[Option<f64>], max_mode: bool) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (idx, value) in values.iter().enumerate() {
        let Some(value) = *value else {
            continue;
        };
        if !value.is_finite() {
            continue;
        }
        match best {
            None => best = Some((idx, value)),
            Some((_, best_value)) => {
                let better = if max_mode {
                    value > best_value
                } else {
                    value < best_value
                };
                if better {
                    best = Some((idx, value));
                }
            }
        }
    }
    best.map(|(idx, _)| idx)
}

fn volume_pct_analysis(points: &[AnalysisPoint]) -> Vec<Option<f64>> {
    let total = points.iter().filter_map(|point| point.vol).sum::<f64>();
    if total <= EPS {
        return vec![None; points.len()];
    }
    points
        .iter()
        .map(|point| point.vol.and_then(|vol| finite_value(vol / total)))
        .collect()
}

fn trailing_std_at(values: &[Option<f64>], idx: usize, window: usize) -> Option<f64> {
    if idx + 1 < window {
        return None;
    }
    let mut output = Vec::with_capacity(window);
    for value in &values[idx + 1 - window..=idx] {
        let value = (*value)?;
        if !value.is_finite() {
            return None;
        }
        output.push(value);
    }
    std_dev(&output)
}

fn trailing_sum_at(values: &[Option<f64>], idx: usize, window: usize) -> Option<f64> {
    if idx + 1 < window {
        return None;
    }
    let mut sum = 0.0;
    for value in &values[idx + 1 - window..=idx] {
        let value = (*value)?;
        if !value.is_finite() {
            return None;
        }
        sum += value;
    }
    finite_value(sum)
}

fn std_dev(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    finite_value(
        (values
            .iter()
            .map(|value| {
                let diff = value - mean;
                diff * diff
            })
            .sum::<f64>()
            / values.len() as f64)
            .sqrt(),
    )
}

fn finite_value(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(time: &str, close: f64, high: f64, low: f64, vol: f64) -> MinutePoint {
        MinutePoint {
            time: time.to_string(),
            close: Some(close),
            high: Some(high),
            low: Some(low),
            vol: Some(vol),
        }
    }

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("expected value");
        assert!(
            (actual - expected).abs() < 1e-10,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn ratio_with_zero_denominator_preserves_meaningful_infinity() {
        assert_eq!(
            ratio_with_zero_denominator(Some(1.0), Some(0.0)),
            Some(f64::INFINITY)
        );
        assert_eq!(
            ratio_with_zero_denominator(Some(-1.0), Some(0.0)),
            Some(f64::NEG_INFINITY)
        );
        assert_eq!(ratio_with_zero_denominator(Some(0.0), Some(0.0)), Some(0.0));
        assert_eq!(ratio_with_zero_denominator(None, Some(0.0)), None);
    }

    #[test]
    fn lh_stats_uses_exact_half_hour_boundaries() {
        let points = vec![
            point("09:30:00", 100.0, 101.0, 99.0, 999.0),
            point("09:31:00", 101.0, 102.0, 100.0, 10.0),
            point("10:00:00", 110.0, 111.0, 109.0, 10.0),
            point("14:30:00", 200.0, 201.0, 199.0, 999.0),
            point("14:31:00", 202.0, 203.0, 201.0, 5.0),
            point("15:00:00", 210.0, 211.0, 209.0, 5.0),
        ];
        let stats = lh_stats(&points);

        assert_close(stats.lh_rtn_diff, 0.1 / 0.05);
        assert_close(stats.lh_vol_diff, 20.0 / 10.0);
        assert!(stats.lh_std_diff.is_some());
    }

    #[test]
    fn post_open_analysis_keeps_0930_anchor_return_but_excludes_0930_sample() {
        let points = vec![
            point("09:30:00", 100.0, 100.0, 100.0, 999.0),
            point("09:31:00", 101.0, 101.0, 101.0, 1.0),
            point("09:32:00", 102.0, 102.0, 102.0, 1.0),
        ];
        let returns = simple_returns(&points);
        let analysis = post_open_analysis_points(&points, &returns);

        assert_eq!(analysis.len(), 2);
        assert_close(analysis[0].ret, 0.01);
        assert_eq!(analysis[0].vol, Some(1.0));
    }

    #[test]
    fn intraday_pct_rank_uses_zero_to_one_scale() {
        let ranks = intraday_pct_ranks(&[Some(3.0), Some(1.0), Some(2.0)]);

        assert_eq!(ranks, vec![Some(1.0), Some(0.0), Some(0.5)]);
    }

    #[test]
    fn high_low_timing_uses_earliest_extreme_and_one_based_diff() {
        let mut high_rank_sum = vec![None; 20];
        high_rank_sum[14] = Some(3.0);
        high_rank_sum[15] = Some(3.0);
        let mut low_rank_sum = vec![None; 20];
        low_rank_sum[16] = Some(1.0);
        low_rank_sum[17] = Some(1.0);

        assert_eq!(earliest_extreme_index(&high_rank_sum, true), Some(14));
        assert_eq!(earliest_extreme_index(&low_rank_sum, false), Some(16));
        let diff_idx = 14isize - 16isize;
        assert_eq!(diff_idx as f64, -2.0);
    }

    #[test]
    fn high_low_stats_compute_diff_components() {
        let mut points = Vec::new();
        for idx in 0..30 {
            let time = format!("09:{:02}:00", 30 + idx);
            points.push(point(
                &time,
                100.0 + idx as f64,
                if idx == 20 { 1000.0 } else { idx as f64 + 1.0 },
                if idx == 15 { 0.1 } else { idx as f64 + 1.0 },
                1.0 + idx as f64,
            ));
        }

        let stats = high_low_stats(&points);

        assert_eq!(stats.diff_idx, Some(14.0));
        assert!(stats.diff_std.is_some());
        assert!(stats.diff_vol.is_some());
    }

    #[test]
    fn cs_rank_turns_infinity_into_finite_rank_before_smoothing() {
        let ranked = cs_rank(
            &[Some(f64::NEG_INFINITY), Some(0.0), Some(f64::INFINITY)],
            true,
        );

        assert_eq!(ranked, vec![Some(1.0), Some(2.0), Some(3.0)]);
    }
}
