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
use crate::factor::common::{DailyPanel, PanelColumn};
use crate::operators::{cs_zscore, ts_std_dev};

pub const PROVIDER_KEY: &str = "mszq_momentum_pulse_provider";
pub const RAW_VERSION: &str = "0.1.0";
pub const VERSION: &str = "0.1.0";

pub const EDGE_RETURN_RAW_ID: &str = "daily_mszq_edge_return_raw";
pub const VOLUME_SURGE_VOLATILITY_RAW_ID: &str = "daily_mszq_volume_surge_volatility_raw";

const RAW_WINDOW_DAYS: usize = 1;
const ROLLING_WINDOW: usize = 20;
const MIN_PERIODS: usize = 1;
const SPLIT_ROUNDS: usize = 6;
const OPEN_SKIP_MINUTES: usize = 10;
const EPS: f64 = f64::EPSILON;

#[derive(Clone, Copy, Debug)]
pub struct MszqMomentumPulseFactorDef {
    pub id: &'static str,
    pub alias: &'static str,
    pub name: &'static str,
}

#[derive(Clone, Copy, Debug)]
struct MinuteReturn {
    minute_idx: usize,
    value: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
struct SurgePoint {
    close: f64,
    volume: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Segment {
    start: usize,
    end: usize,
}

pub fn all_raw_ids() -> [&'static str; 2] {
    [EDGE_RETURN_RAW_ID, VOLUME_SURGE_VOLATILITY_RAW_ID]
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

pub fn factor_spec(def: MszqMomentumPulseFactorDef) -> FactorSpec {
    FactorSpec {
        id: def.id.to_string(),
        aliases: vec![def.alias.to_string()],
        name: def.name.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: tags(),
        description: description(def),
        dependencies: dependencies(),
        intraday_raw_dependencies: all_raw_ids()
            .iter()
            .map(|raw_id| IntradayDailyRawRequest::new(raw_id, ROLLING_WINDOW - 1))
            .collect(),
        lookback: Lookback {
            trading_days: ROLLING_WINDOW - 1,
        },
    }
}

pub fn compute_factor(def: MszqMomentumPulseFactorDef, data: &DataPool) -> Result<FactorSeries> {
    let panel = data.intraday_daily_raw_panel(EDGE_RETURN_RAW_ID)?;
    let edge_raw = panel.column(EDGE_RETURN_RAW_ID)?;
    let surge_raw = panel.column(VOLUME_SURGE_VOLATILITY_RAW_ID)?;

    let edge_component = edge_raw
        .ts(|series| ts_std_dev(series, ROLLING_WINDOW, MIN_PERIODS))?
        .cs(cs_zscore)?;
    let surge_component = surge_raw
        .cs(cs_zscore)?
        .ts(|series| ts_std_dev(series, ROLLING_WINDOW, MIN_PERIODS))?
        .cs(cs_zscore)?;
    let composite = average_columns(&panel, &[&edge_component, &surge_component])?;
    let factor = neutralize_size_sector(&composite, &panel, data)?;
    Ok(factor.to_factor_series(factor_spec(def)))
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
        let volume = table.required_f64_cast("vol")?;

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

        let mut returns_by_stock = BTreeMap::<String, Vec<MinuteReturn>>::new();
        let mut returns_by_minute = BTreeMap::<usize, Vec<(String, Option<f64>)>>::new();
        let mut surge_by_stock = BTreeMap::<String, Option<f64>>::new();

        for (ts_code, mut indices) in grouped {
            indices.sort_by(|left, right| trade_times[*left].cmp(&trade_times[*right]));
            let returns = minute_returns_from_indices(&indices, &trade_times, &close);
            for item in &returns {
                returns_by_minute
                    .entry(item.minute_idx)
                    .or_default()
                    .push((ts_code.clone(), item.value));
            }
            let surge_points = surge_points_from_indices(&indices, &trade_times, &close, &volume);
            surge_by_stock.insert(
                ts_code.clone(),
                volume_surge_volatility_from_points(&surge_points),
            );
            returns_by_stock.insert(ts_code, returns);
        }

        let edge_by_stock = edge_return_raw_by_stock(&returns_by_stock, &returns_by_minute);
        let stocks = returns_by_stock
            .keys()
            .chain(surge_by_stock.keys())
            .cloned()
            .collect::<BTreeSet<_>>();
        for ts_code in stocks {
            let key = FactorRowKey::Daily {
                trade_date: *trade_date,
                ts_code: ts_code.clone(),
            };
            push_requested(
                &mut values,
                &requested,
                EDGE_RETURN_RAW_ID,
                &key,
                Some(*edge_by_stock.get(&ts_code).unwrap_or(&0.0)),
            );
            push_requested(
                &mut values,
                &requested,
                VOLUME_SURGE_VOLATILITY_RAW_ID,
                &key,
                surge_by_stock.get(&ts_code).copied().unwrap_or(None),
            );
        }
    }

    let mut output = Vec::new();
    for raw_id in all_raw_ids() {
        if !requested.contains(raw_id) {
            continue;
        }
        output.push(IntradayDailyRawSeries {
            spec: raw_spec(raw_id),
            values: values.remove(raw_id).unwrap_or_default(),
        });
    }
    Ok(output)
}

fn minute_returns_from_indices(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
) -> Vec<MinuteReturn> {
    let mut output = Vec::new();
    let mut previous_close = None;
    for idx in indices {
        let Some(trade_time) = trade_times[*idx].as_deref() else {
            continue;
        };
        let Some(minute_idx) = minute_index(trade_time) else {
            continue;
        };
        let current_close = clean_positive(close[*idx]);
        let value = minute_return(previous_close, current_close);
        if current_close.is_some() {
            previous_close = current_close;
        }
        output.push(MinuteReturn { minute_idx, value });
    }
    output
}

fn edge_return_raw_by_stock(
    returns_by_stock: &BTreeMap<String, Vec<MinuteReturn>>,
    returns_by_minute: &BTreeMap<usize, Vec<(String, Option<f64>)>>,
) -> BTreeMap<String, f64> {
    let mut output = returns_by_stock
        .keys()
        .map(|ts_code| (ts_code.clone(), 0.0))
        .collect::<BTreeMap<_, _>>();
    for rows in returns_by_minute.values() {
        let (positive_mean, negative_mean) = directional_means(rows);
        let deviations = rows
            .iter()
            .map(|(_, value)| edge_deviation(*value, positive_mean, negative_mean))
            .collect::<Vec<_>>();
        let Some(skewness) = skew(&deviations) else {
            continue;
        };
        if skewness <= 0.0 {
            continue;
        }
        for ((ts_code, _), deviation) in rows.iter().zip(deviations) {
            *output.entry(ts_code.clone()).or_default() += deviation;
        }
    }
    output
}

fn directional_means(rows: &[(String, Option<f64>)]) -> (Option<f64>, Option<f64>) {
    let mut positive_sum = 0.0;
    let mut positive_count = 0usize;
    let mut negative_sum = 0.0;
    let mut negative_count = 0usize;
    for (_, value) in rows {
        let Some(value) = finite_option(*value) else {
            continue;
        };
        if value > 0.0 {
            positive_sum += value;
            positive_count += 1;
        } else if value < 0.0 {
            negative_sum += value;
            negative_count += 1;
        }
    }
    (
        (positive_count > 0).then_some(positive_sum / positive_count as f64),
        (negative_count > 0).then_some(negative_sum / negative_count as f64),
    )
}

fn edge_deviation(
    value: Option<f64>,
    positive_mean: Option<f64>,
    negative_mean: Option<f64>,
) -> f64 {
    let Some(value) = finite_option(value) else {
        return 0.0;
    };
    if value > 0.0 {
        positive_mean
            .and_then(|mean| finite_value(value - mean))
            .unwrap_or(0.0)
    } else if value < 0.0 {
        negative_mean
            .and_then(|mean| finite_value(value - mean))
            .unwrap_or(0.0)
    } else {
        0.0
    }
}

fn surge_points_from_indices(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
    volume: &[Option<f64>],
) -> Vec<SurgePoint> {
    indices
        .iter()
        .filter_map(|idx| {
            let trade_time = trade_times[*idx].as_deref()?;
            let minute_idx = minute_index(trade_time)?;
            if minute_idx < OPEN_SKIP_MINUTES {
                return None;
            }
            Some(SurgePoint {
                close: clean_positive(close[*idx])?,
                volume: clean_nonnegative(volume[*idx])?,
            })
        })
        .collect()
}

fn volume_surge_volatility_from_points(points: &[SurgePoint]) -> Option<f64> {
    if points.is_empty() {
        return None;
    }
    let segments = split_volume_surge_segments(points, SPLIT_ROUNDS);
    let returns = segments
        .iter()
        .filter_map(|segment| segment_return(points, *segment))
        .collect::<Vec<_>>();
    std_dev(&returns)
}

fn split_volume_surge_segments(points: &[SurgePoint], rounds: usize) -> Vec<Segment> {
    if points.is_empty() {
        return Vec::new();
    }
    let mut segments = vec![Segment {
        start: 0,
        end: points.len() - 1,
    }];
    for _ in 0..rounds {
        let mut next = Vec::with_capacity(segments.len() * 2);
        for segment in segments {
            let len = segment.end - segment.start + 1;
            if len == 1 {
                next.push(segment);
            } else if len == 2 {
                next.push(Segment {
                    start: segment.start,
                    end: segment.start,
                });
                next.push(Segment {
                    start: segment.end,
                    end: segment.end,
                });
            } else {
                let split = max_volume_split_point(points, segment);
                next.push(Segment {
                    start: segment.start,
                    end: split,
                });
                next.push(Segment {
                    start: split + 1,
                    end: segment.end,
                });
            }
        }
        segments = next;
    }
    segments
}

fn max_volume_split_point(points: &[SurgePoint], segment: Segment) -> usize {
    let mut best = segment.start + 1;
    let mut best_volume = points[best].volume;
    for idx in (segment.start + 2)..segment.end {
        if points[idx].volume > best_volume {
            best = idx;
            best_volume = points[idx].volume;
        }
    }
    best
}

fn segment_return(points: &[SurgePoint], segment: Segment) -> Option<f64> {
    if segment.start == segment.end {
        return Some(0.0);
    }
    let first = points.get(segment.start)?.close;
    let last = points.get(segment.end)?.close;
    if first.abs() <= EPS {
        return None;
    }
    finite_value(last / first - 1.0)
}

fn minute_return(previous_close: Option<f64>, current_close: Option<f64>) -> Option<f64> {
    let (Some(previous), Some(current)) = (previous_close, current_close) else {
        return None;
    };
    if previous.abs() <= EPS {
        return None;
    }
    finite_value(current / previous - 1.0)
}

fn average_columns(panel: &DailyPanel, columns: &[&PanelColumn]) -> Result<PanelColumn> {
    if columns.is_empty() {
        return panel.column_from_values(vec![None; panel.shape_len()]);
    }
    let mut values = Vec::with_capacity(panel.shape_len());
    for offset in 0..panel.shape_len() {
        let mut sum = 0.0;
        let mut count = 0usize;
        for column in columns {
            if let Some(value) = finite_option(column.values()[offset]) {
                sum += value;
                count += 1;
            }
        }
        values.push((count > 0).then_some(sum / count as f64));
    }
    panel.column_from_values(values)
}

fn skew(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let mean = mean(values)?;
    let mut second = 0.0;
    let mut third = 0.0;
    for value in values {
        let diff = value - mean;
        second += diff * diff;
        third += diff * diff * diff;
    }
    let variance = second / values.len() as f64;
    if variance <= EPS {
        return None;
    }
    finite_value((third / values.len() as f64) / variance.powf(1.5))
}

fn std_dev(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mean = mean(values)?;
    let variance = values
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64;
    finite_value(variance.max(0.0).sqrt())
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    finite_value(values.iter().sum::<f64>() / values.len() as f64)
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

fn tags() -> Vec<String> {
    [
        "price_volume",
        "volume",
        "return",
        "volatility",
        "momentum",
        "intraday",
        "minute_agg",
        "neutralize",
        "barra",
        "size",
        "sector",
        "daily",
        "MSZQ",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect()
}

fn description(def: MszqMomentumPulseFactorDef) -> String {
    format!(
        "{} composites an intraday edge-return component and a volume-surge volatility component from 1-minute close/volume data, keeps the report's reverse raw direction, and neutralizes by Barra SIZE and SW sector.",
        def.name
    )
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

fn clean_positive(value: Option<f64>) -> Option<f64> {
    clean_intraday_value(value)
        .and_then(finite_value)
        .filter(|value| *value > 0.0)
}

fn clean_nonnegative(value: Option<f64>) -> Option<f64> {
    clean_intraday_value(value)
        .and_then(finite_value)
        .filter(|value| *value >= 0.0)
}

fn finite_option(value: Option<f64>) -> Option<f64> {
    value.and_then(finite_value)
}

fn finite_value(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("expected value");
        assert!(
            (actual - expected).abs() < 1e-10,
            "expected {expected}, got {actual}"
        );
    }

    fn point(close: f64, volume: f64) -> SurgePoint {
        SurgePoint { close, volume }
    }

    #[test]
    fn momentum_pulse_minute_return_uses_close_to_close() {
        assert_close(minute_return(Some(100.0), Some(101.0)), 0.01);
        assert_eq!(minute_return(None, Some(101.0)), None);
        assert_eq!(minute_return(Some(0.0), Some(101.0)), None);
    }

    #[test]
    fn momentum_pulse_directional_means_and_edge_deviation_use_same_direction_benchmark() {
        let rows = vec![
            ("a".to_string(), Some(0.03)),
            ("b".to_string(), Some(0.01)),
            ("c".to_string(), Some(-0.02)),
            ("d".to_string(), Some(0.0)),
        ];
        let (positive_mean, negative_mean) = directional_means(&rows);

        assert_close(positive_mean, 0.02);
        assert_close(negative_mean, -0.02);
        assert_close(
            Some(edge_deviation(Some(0.03), positive_mean, negative_mean)),
            0.01,
        );
        assert_close(
            Some(edge_deviation(Some(-0.02), positive_mean, negative_mean)),
            0.0,
        );
        assert_close(
            Some(edge_deviation(Some(0.0), positive_mean, negative_mean)),
            0.0,
        );
    }

    #[test]
    fn momentum_pulse_edge_raw_only_accumulates_positive_skew_moments() {
        let mut returns_by_stock = BTreeMap::new();
        returns_by_stock.insert(
            "a".to_string(),
            vec![
                MinuteReturn {
                    minute_idx: 0,
                    value: Some(0.04),
                },
                MinuteReturn {
                    minute_idx: 1,
                    value: Some(0.01),
                },
            ],
        );
        returns_by_stock.insert(
            "b".to_string(),
            vec![
                MinuteReturn {
                    minute_idx: 0,
                    value: Some(0.01),
                },
                MinuteReturn {
                    minute_idx: 1,
                    value: Some(0.04),
                },
            ],
        );
        returns_by_stock.insert(
            "c".to_string(),
            vec![
                MinuteReturn {
                    minute_idx: 0,
                    value: Some(0.01),
                },
                MinuteReturn {
                    minute_idx: 1,
                    value: Some(0.04),
                },
            ],
        );
        let mut returns_by_minute = BTreeMap::new();
        for (ts_code, returns) in &returns_by_stock {
            for item in returns {
                returns_by_minute
                    .entry(item.minute_idx)
                    .or_insert_with(Vec::new)
                    .push((ts_code.clone(), item.value));
            }
        }

        let output = edge_return_raw_by_stock(&returns_by_stock, &returns_by_minute);

        assert!(output["a"] > 0.0);
        assert!(output["b"] < 0.0);
        assert!(output["c"] < 0.0);
    }

    #[test]
    fn momentum_pulse_surge_points_skip_first_ten_regular_minutes() {
        let times = (0..12)
            .map(|idx| Some(format!("09:{:02}:00", 31 + idx)))
            .collect::<Vec<_>>();
        let indices = (0..times.len()).collect::<Vec<_>>();
        let close = vec![Some(10.0); times.len()];
        let volume = vec![Some(1.0); times.len()];

        let points = surge_points_from_indices(&indices, &times, &close, &volume);

        assert_eq!(points.len(), 2);
    }

    #[test]
    fn momentum_pulse_volume_surge_split_handles_length_boundaries_and_peak_assignment() {
        let one = vec![point(1.0, 1.0)];
        let two = vec![point(1.0, 1.0), point(2.0, 2.0)];
        let four = vec![
            point(1.0, 1.0),
            point(2.0, 5.0),
            point(3.0, 3.0),
            point(4.0, 1.0),
        ];

        assert_eq!(
            split_volume_surge_segments(&one, 1),
            vec![Segment { start: 0, end: 0 }]
        );
        assert_eq!(
            split_volume_surge_segments(&two, 1),
            vec![Segment { start: 0, end: 0 }, Segment { start: 1, end: 1 }]
        );
        assert_eq!(
            split_volume_surge_segments(&four, 1),
            vec![Segment { start: 0, end: 1 }, Segment { start: 2, end: 3 }]
        );
    }

    #[test]
    fn momentum_pulse_segment_return_uses_endpoint_close_ratio_and_singletons_zero() {
        let points = vec![point(10.0, 1.0), point(11.0, 2.0), point(12.0, 3.0)];

        assert_close(segment_return(&points, Segment { start: 0, end: 2 }), 0.2);
        assert_close(segment_return(&points, Segment { start: 1, end: 1 }), 0.0);
    }

    #[test]
    fn momentum_pulse_component_postprocess_matches_plan() {
        let panel = DailyPanel::from_index(
            vec![20260423, 20260424],
            vec!["a".to_string(), "b".to_string()],
            &[20260423, 20260424],
            vec![true, true, true, true],
        )
        .unwrap();
        let edge = panel
            .column_from_values(vec![Some(1.0), Some(2.0), Some(3.0), Some(5.0)])
            .unwrap();
        let surge = panel
            .column_from_values(vec![Some(10.0), Some(12.0), Some(11.0), Some(16.0)])
            .unwrap();

        let edge_component = edge
            .ts(|series| ts_std_dev(series, ROLLING_WINDOW, MIN_PERIODS))
            .unwrap()
            .cs(cs_zscore)
            .unwrap();
        let surge_component = surge
            .cs(cs_zscore)
            .unwrap()
            .ts(|series| ts_std_dev(series, ROLLING_WINDOW, MIN_PERIODS))
            .unwrap()
            .cs(cs_zscore)
            .unwrap();
        let composite = average_columns(&panel, &[&edge_component, &surge_component]).unwrap();

        assert_eq!(composite.values().len(), 4);
    }

    #[test]
    fn momentum_pulse_factor_spec_has_mszq_tag_and_single_output() {
        let spec = factor_spec(MszqMomentumPulseFactorDef {
            id: "momentum_pulse",
            alias: "momentum_pulse",
            name: "Momentum Pulse",
        });

        assert_eq!(spec.id, "momentum_pulse");
        assert!(spec.tags.iter().any(|tag| tag == "MSZQ"));
        assert!(spec.tags.iter().any(|tag| tag == "momentum"));
        assert_eq!(spec.intraday_raw_dependencies.len(), 2);
        assert!(spec.description.contains("1-minute close/volume"));
    }
}
