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
    MINV_RAW_ID, NEGVWGT_MAX_RAW_ID, NEGVWGT_MEAN_RAW_ID, NEGV_MAX_RAW_ID, NEGV_MEAN_RAW_ID,
};
use crate::factor::common::{
    clean_intraday_value, intraday_time_in_range, quantile_linear, stock_minute_raw_spec,
};
use crate::operators::{cs_pctrank, ts_mean};

pub const RAW_VERSION: &str = "0.1.0";
pub const VERSION: &str = "0.1.0";

const RAW_WINDOW_DAYS: usize = 1;
const DEFAULT_WINDOW: usize = 15;
const FLASH_WINDOW: usize = 20;
const FLASH_LAMBDA_WINDOW: usize = 21;
const SHARED_RAW_LOOKBACK: usize = FLASH_WINDOW - 1;
const MIN_PERIODS: usize = 1;
const START_TIME: &str = "09:41:00";
const END_TIME: &str = "14:50:00";
const VRANGE_SCALE: f64 = 1.0e4;
const EPS: f64 = f64::EPSILON;

#[derive(Clone, Copy, Debug)]
pub struct XyzqVshapeFactorDef {
    pub id: &'static str,
    pub alias: &'static str,
    pub name: &'static str,
    pub kind: XyzqVshapeFactorKind,
}

#[derive(Clone, Copy, Debug)]
pub enum XyzqVshapeFactorKind {
    RollingMean { raw_id: &'static str, window: usize },
    FlashCrashProbV,
}

#[derive(Clone, Copy, Debug, Default)]
struct VshapeStats {
    negv_mean: Option<f64>,
    negv_max: Option<f64>,
    negvwgt_mean: Option<f64>,
    negvwgt_max: Option<f64>,
    minv: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
struct MinutePoint {
    in_window: bool,
    close: Option<f64>,
    vol: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
struct Segment {
    sum: f64,
    duration: usize,
}

pub const fn default_window() -> usize {
    DEFAULT_WINDOW
}

pub const fn flash_window() -> usize {
    FLASH_WINDOW
}

pub fn all_raw_ids() -> [&'static str; 5] {
    [
        NEGV_MEAN_RAW_ID,
        NEGV_MAX_RAW_ID,
        NEGVWGT_MEAN_RAW_ID,
        NEGVWGT_MAX_RAW_ID,
        MINV_RAW_ID,
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

pub fn factor_spec(def: XyzqVshapeFactorDef) -> FactorSpec {
    let intraday_raw_dependencies = match def.kind {
        XyzqVshapeFactorKind::RollingMean { raw_id, .. } => {
            vec![IntradayDailyRawRequest::new(raw_id, SHARED_RAW_LOOKBACK)]
        }
        XyzqVshapeFactorKind::FlashCrashProbV => vec![
            IntradayDailyRawRequest::new(MINV_RAW_ID, SHARED_RAW_LOOKBACK),
            IntradayDailyRawRequest::new(NEGV_MEAN_RAW_ID, SHARED_RAW_LOOKBACK),
        ],
    };

    FactorSpec {
        id: def.id.to_string(),
        aliases: vec![def.alias.to_string()],
        name: def.name.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: tags(),
        description: format!(
            "{} from intraday V-shaped price-volume impact raw, cross-sectional percentile rank, and SIZE/SW-sector neutralization.",
            def.name
        ),
        dependencies: dependencies(),
        intraday_raw_dependencies,
        lookback: Lookback {
            trading_days: SHARED_RAW_LOOKBACK,
        },
    }
}

pub fn compute_factor(def: XyzqVshapeFactorDef, data: &DataPool) -> Result<FactorSeries> {
    let factor = match def.kind {
        XyzqVshapeFactorKind::RollingMean { raw_id, window } => {
            let panel = data.intraday_daily_raw_panel(raw_id)?;
            let raw = panel.column(raw_id)?;
            let smoothed = raw.ts(|values| ts_mean(values, window, MIN_PERIODS))?;
            let ranked = smoothed.cs(|values| cs_pctrank(values, true))?;
            neutralize_size_sector(&ranked, panel, data)?
        }
        XyzqVshapeFactorKind::FlashCrashProbV => compute_flash_crash_prob_v(data)?,
    };
    Ok(factor.to_factor_series(factor_spec(def)))
}

fn compute_flash_crash_prob_v(data: &DataPool) -> Result<crate::factor::common::PanelColumn> {
    let panel = data.intraday_daily_raw_panel(MINV_RAW_ID)?;
    let minv = panel.column(MINV_RAW_ID)?;
    let negv_mean = panel.column(NEGV_MEAN_RAW_ID)?;

    let mean_prior_minv = minv.ts(|values| ts_mean(values, FLASH_LAMBDA_WINDOW, MIN_PERIODS))?;
    let lambda = mean_prior_minv.map_values(|value| match clean(value) {
        Some(value) if value > EPS => finite_value(1.0 / value),
        _ => None,
    });
    let threshold = negv_mean.cs(|values| {
        let mut valid = values
            .iter()
            .filter_map(|value| clean(*value))
            .collect::<Vec<_>>();
        let q75 = quantile_linear(&mut valid, 0.75);
        vec![q75; values.len()]
    })?;
    let flash_raw = lambda.zip_binary(&threshold, |lambda, threshold| {
        match (clean(lambda), clean(threshold)) {
            (Some(lambda), Some(threshold)) if threshold >= 0.0 => {
                finite_value((-lambda * threshold).exp())
            }
            _ => None,
        }
    })?;
    let smoothed = flash_raw.ts(|values| ts_mean(values, FLASH_WINDOW, MIN_PERIODS))?;
    let ranked = smoothed.cs(|values| cs_pctrank(values, true))?;
    neutralize_size_sector(&ranked, panel, data)
}

#[macro_export]
macro_rules! define_xyzq_vshape_structure_factor {
    ($struct_name:ident, $id:expr, $alias:expr, $name:expr, $kind:expr) => {
        const DEF: $crate::factor::common::xyzq_vshape_structure::XyzqVshapeFactorDef =
            $crate::factor::common::xyzq_vshape_structure::XyzqVshapeFactorDef {
                id: $id,
                alias: $alias,
                name: $name,
                kind: $kind,
            };

        pub struct $struct_name;

        pub fn create() -> Box<dyn $crate::factor::Factor> {
            Box::new($struct_name)
        }

        impl $crate::factor::Factor for $struct_name {
            fn spec(&self) -> $crate::core::FactorSpec {
                $crate::factor::common::xyzq_vshape_structure::factor_spec(DEF)
            }

            fn compute(
                &self,
                _context: &$crate::core::FactorContext,
                data: &$crate::data::DataPool,
            ) -> $crate::error::Result<$crate::core::FactorSeries> {
                $crate::factor::common::xyzq_vshape_structure::compute_factor(DEF, data)
            }
        }
    };
}

pub fn minute_compute_many(
    raw_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
) -> Result<Vec<IntradayDailyRawSeries>> {
    let family_raw_ids = all_raw_ids();
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
            let points = minute_points_from_indices(&indices, &trade_times, &close, &vol);
            let stats = vshape_stats(&points);
            let key = FactorRowKey::Daily {
                trade_date: *trade_date,
                ts_code,
            };

            push_requested(
                &mut values,
                &requested,
                NEGV_MEAN_RAW_ID,
                &key,
                stats.negv_mean,
            );
            push_requested(
                &mut values,
                &requested,
                NEGV_MAX_RAW_ID,
                &key,
                stats.negv_max,
            );
            push_requested(
                &mut values,
                &requested,
                NEGVWGT_MEAN_RAW_ID,
                &key,
                stats.negvwgt_mean,
            );
            push_requested(
                &mut values,
                &requested,
                NEGVWGT_MAX_RAW_ID,
                &key,
                stats.negvwgt_max,
            );
            push_requested(&mut values, &requested, MINV_RAW_ID, &key, stats.minv);
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
        "vshape",
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

fn vshape_stats(points: &[MinutePoint]) -> VshapeStats {
    let returns = simple_returns(points);
    let total_vol = points
        .iter()
        .filter(|point| point.in_window)
        .filter_map(|point| point.vol)
        .sum::<f64>();
    if total_vol <= EPS {
        return VshapeStats::default();
    }

    let vranges = points
        .iter()
        .enumerate()
        .filter(|(_, point)| point.in_window)
        .map(|(idx, point)| match (returns[idx], point.vol) {
            (Some(ret), Some(vol)) => finite_value(ret * vol / total_vol * VRANGE_SCALE),
            _ => None,
        })
        .collect::<Vec<_>>();
    let segments = same_sign_segments(&vranges);
    let (values, weighted_values) = vshape_values(&segments);
    if values.is_empty() {
        return VshapeStats::default();
    }

    VshapeStats {
        negv_mean: mean(&values),
        negv_max: values
            .iter()
            .copied()
            .reduce(f64::max)
            .and_then(finite_value),
        negvwgt_mean: mean(&weighted_values),
        negvwgt_max: weighted_values
            .iter()
            .copied()
            .reduce(f64::max)
            .and_then(finite_value),
        minv: values
            .iter()
            .copied()
            .reduce(f64::max)
            .and_then(finite_value),
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

fn same_sign_segments(values: &[Option<f64>]) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut current_sum = 0.0;
    let mut current_sign = 0i8;
    let mut current_duration = 0usize;

    for value in values {
        let sign = match value.and_then(finite_value) {
            Some(value) if value > 0.0 => 1,
            Some(value) if value < 0.0 => -1,
            _ => {
                flush_segment(
                    &mut segments,
                    &mut current_sum,
                    &mut current_sign,
                    &mut current_duration,
                );
                continue;
            }
        };
        let value = value.unwrap();
        if current_sign == 0 || current_sign == sign {
            current_sum += value;
            current_sign = sign;
            current_duration += 1;
        } else {
            flush_segment(
                &mut segments,
                &mut current_sum,
                &mut current_sign,
                &mut current_duration,
            );
            current_sum = value;
            current_sign = sign;
            current_duration = 1;
        }
    }
    flush_segment(
        &mut segments,
        &mut current_sum,
        &mut current_sign,
        &mut current_duration,
    );
    segments
}

fn flush_segment(segments: &mut Vec<Segment>, sum: &mut f64, sign: &mut i8, duration: &mut usize) {
    if *sign != 0 && *duration > 0 {
        segments.push(Segment {
            sum: *sum,
            duration: *duration,
        });
    }
    *sum = 0.0;
    *sign = 0;
    *duration = 0;
}

fn vshape_values(segments: &[Segment]) -> (Vec<f64>, Vec<f64>) {
    let mut values = Vec::new();
    let mut weighted_values = Vec::new();
    for pair in segments.windows(2) {
        let previous = pair[0];
        let current = pair[1];
        if previous.sum < 0.0 && current.sum > 0.0 {
            let pair_sum = previous.sum + current.sum;
            if pair_sum < 0.0 {
                let value = pair_sum.abs();
                let duration = previous.duration + current.duration;
                values.push(value);
                weighted_values.push(value * duration as f64);
            }
        }
    }
    (values, weighted_values)
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    finite_value(values.iter().sum::<f64>() / values.len() as f64)
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite())
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

    #[test]
    fn same_sign_segments_breaks_on_zero_or_missing() {
        let segments = same_sign_segments(&[
            Some(-1.0),
            Some(-2.0),
            Some(0.0),
            Some(3.0),
            None,
            Some(-4.0),
        ]);

        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].sum, -3.0);
        assert_eq!(segments[0].duration, 2);
        assert_eq!(segments[1].sum, 3.0);
        assert_eq!(segments[1].duration, 1);
        assert_eq!(segments[2].sum, -4.0);
        assert_eq!(segments[2].duration, 1);
    }

    #[test]
    fn vshape_values_use_negative_positive_pairs_still_below_zero() {
        let segments = vec![
            Segment {
                sum: -5.0,
                duration: 3,
            },
            Segment {
                sum: 2.0,
                duration: 2,
            },
            Segment {
                sum: -6.0,
                duration: 1,
            },
            Segment {
                sum: 1.0,
                duration: 2,
            },
        ];

        let (values, weighted) = vshape_values(&segments);
        assert_eq!(values, vec![3.0, 5.0]);
        assert_eq!(weighted, vec![15.0, 15.0]);
    }

    #[test]
    fn vshape_stats_minv_uses_largest_abs_pair_sum() {
        let points = vec![
            MinutePoint {
                in_window: false,
                close: Some(100.0),
                vol: Some(1.0),
            },
            MinutePoint {
                in_window: true,
                close: Some(98.0),
                vol: Some(1.0),
            },
            MinutePoint {
                in_window: true,
                close: Some(98.98),
                vol: Some(1.0),
            },
            MinutePoint {
                in_window: true,
                close: Some(96.0106),
                vol: Some(1.0),
            },
            MinutePoint {
                in_window: true,
                close: Some(96.970706),
                vol: Some(1.0),
            },
        ];

        let stats = vshape_stats(&points);
        assert_close(stats.minv, Some(50.0));
    }

    #[test]
    fn vshape_stats_compute_raw_values_from_vrange() {
        let points = vec![
            MinutePoint {
                in_window: false,
                close: Some(100.0),
                vol: Some(1.0),
            },
            MinutePoint {
                in_window: true,
                close: Some(99.0),
                vol: Some(1.0),
            },
            MinutePoint {
                in_window: true,
                close: Some(98.0),
                vol: Some(1.0),
            },
            MinutePoint {
                in_window: true,
                close: Some(99.0),
                vol: Some(1.0),
            },
        ];

        let stats = vshape_stats(&points);
        let r1 = 99.0 / 100.0 - 1.0;
        let r2 = 98.0 / 99.0 - 1.0;
        let r3 = 99.0 / 98.0 - 1.0;
        let neg = (r1 + r2) / 3.0 * VRANGE_SCALE;
        let pos = r3 / 3.0 * VRANGE_SCALE;
        let expected = (neg + pos).abs();

        assert_close(stats.negv_mean, Some(expected));
        assert_close(stats.negv_max, Some(expected));
        assert_close(stats.negvwgt_mean, Some(expected * 3.0));
        assert_close(stats.negvwgt_max, Some(expected * 3.0));
        assert_close(stats.minv, Some(expected));
    }
}
