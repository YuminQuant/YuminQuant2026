use std::collections::{BTreeMap, BTreeSet};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::neutralize_size_sector;
use crate::factor::common::{
    clean_intraday_value, minute_vwap_from_amount_vol, stock_minute_raw_spec,
};
use crate::factor::common::{DailyPanel, PanelColumn};
use crate::operators::{cs_zscore, ts_mean, ts_std_dev};

pub const PROVIDER_KEY: &str = "mszq_price_volume_tension_provider";
pub const RAW_VERSION: &str = "0.1.0";
pub const VERSION: &str = "0.1.0";

pub const ELASTIC_TOTAL_RETURN_RAW_ID: &str = "daily_mszq_elastic_total_return_raw";
pub const ELASTIC_TOTAL_VOLUME_RAW_ID: &str = "daily_mszq_elastic_total_volume_raw";
pub const ELASTIC_COEFFICIENT_RAW_ID: &str = "daily_mszq_elastic_coefficient_raw";
pub const VOLUME_ENERGY_DIVERGENCE_RAW_ID: &str = "daily_mszq_volume_energy_divergence_raw";

const RAW_WINDOW_DAYS: usize = 1;
const ROLLING_WINDOW: usize = 20;
const MIN_PERIODS: usize = 1;
const THREE_MINUTE_BARS: usize = 80;
const THREE_MINUTE_BAR_SIZE: usize = 3;
const EXTREMA_RADIUS: usize = 13;
const EPS: f64 = f64::EPSILON;

#[derive(Clone, Copy, Debug)]
pub struct MszqPriceVolumeTensionFactorDef {
    pub id: &'static str,
    pub alias: &'static str,
    pub name: &'static str,
}

#[derive(Clone, Copy, Debug, Default)]
struct ThreeMinuteBarBuilder {
    valid_count: usize,
    open: Option<f64>,
    high: Option<f64>,
    low: Option<f64>,
    close: Option<f64>,
    volume: f64,
}

#[derive(Clone, Copy, Debug)]
struct ThreeMinuteBar {
    high: f64,
    low: f64,
    volume: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct ElasticDailyStats {
    total_return: Option<f64>,
    total_volume: Option<f64>,
    coefficient: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
struct Segment {
    start: usize,
    end: usize,
}

#[derive(Clone, Copy, Debug)]
struct DivergencePoint {
    price_deviation: f64,
    volume_share: f64,
}

pub fn all_raw_ids() -> [&'static str; 4] {
    [
        ELASTIC_TOTAL_RETURN_RAW_ID,
        ELASTIC_TOTAL_VOLUME_RAW_ID,
        ELASTIC_COEFFICIENT_RAW_ID,
        VOLUME_ENERGY_DIVERGENCE_RAW_ID,
    ]
}

pub fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
    stock_minute_raw_spec(
        raw_id,
        RAW_VERSION,
        &["open", "high", "low", "close", "vol", "amount"],
        RAW_WINDOW_DAYS,
    )
}

pub fn raw_specs() -> Vec<IntradayDailyRawSpec> {
    all_raw_ids()
        .iter()
        .map(|raw_id| raw_spec(raw_id))
        .collect()
}

pub fn factor_spec(def: MszqPriceVolumeTensionFactorDef) -> FactorSpec {
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

pub fn compute_factor(
    def: MszqPriceVolumeTensionFactorDef,
    data: &DataPool,
) -> Result<FactorSeries> {
    let panel = data.intraday_daily_raw_panel(ELASTIC_TOTAL_RETURN_RAW_ID)?;
    let total_return = panel.column(ELASTIC_TOTAL_RETURN_RAW_ID)?;
    let total_volume = panel.column(ELASTIC_TOTAL_VOLUME_RAW_ID)?;
    let coefficient = panel.column(ELASTIC_COEFFICIENT_RAW_ID)?;
    let divergence = panel.column(VOLUME_ENERGY_DIVERGENCE_RAW_ID)?;

    let smoothed_coefficient =
        coefficient.ts(|series| ts_mean(series, ROLLING_WINDOW, MIN_PERIODS))?;
    let expected_return = smoothed_coefficient.zip_binary(&total_volume, multiply_pair)?;
    let elastic_gap = total_return.zip_binary(&expected_return, subtract_pair)?;

    let elastic_component = std20_after_daily_zscore(&elastic_gap)?;
    let divergence_component = std20_after_daily_zscore(&divergence)?;
    let composite = average_columns(&panel, &[&elastic_component, &divergence_component])?;
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
        let open = table.required_f64_cast("open")?;
        let high = table.required_f64_cast("high")?;
        let low = table.required_f64_cast("low")?;
        let close = table.required_f64_cast("close")?;
        let volume = table.required_f64_cast("vol")?;
        let amount = table.required_f64_cast("amount")?;

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
            let bars = three_minute_bars_from_indices(
                &indices,
                &trade_times,
                &open,
                &high,
                &low,
                &close,
                &volume,
            );
            let elastic = elastic_daily_stats(&bars);
            let divergence = volume_energy_divergence_from_indices(
                &indices,
                &trade_times,
                &close,
                &amount,
                &volume,
            );
            let key = FactorRowKey::Daily {
                trade_date: *trade_date,
                ts_code,
            };
            push_requested(
                &mut values,
                &requested,
                ELASTIC_TOTAL_RETURN_RAW_ID,
                &key,
                elastic.total_return,
            );
            push_requested(
                &mut values,
                &requested,
                ELASTIC_TOTAL_VOLUME_RAW_ID,
                &key,
                elastic.total_volume,
            );
            push_requested(
                &mut values,
                &requested,
                ELASTIC_COEFFICIENT_RAW_ID,
                &key,
                elastic.coefficient,
            );
            push_requested(
                &mut values,
                &requested,
                VOLUME_ENERGY_DIVERGENCE_RAW_ID,
                &key,
                divergence,
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

fn std20_after_daily_zscore(values: &PanelColumn) -> Result<PanelColumn> {
    let standardized = values.cs(cs_zscore)?;
    standardized.ts(|series| ts_std_dev(series, ROLLING_WINDOW, MIN_PERIODS))
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

fn multiply_pair(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    let (Some(left), Some(right)) = (finite_option(left), finite_option(right)) else {
        return None;
    };
    finite_value(left * right)
}

fn subtract_pair(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    let (Some(left), Some(right)) = (finite_option(left), finite_option(right)) else {
        return None;
    };
    finite_value(left - right)
}

fn three_minute_bars_from_indices(
    indices: &[usize],
    trade_times: &[Option<String>],
    open: &[Option<f64>],
    high: &[Option<f64>],
    low: &[Option<f64>],
    close: &[Option<f64>],
    volume: &[Option<f64>],
) -> Vec<ThreeMinuteBar> {
    let mut builders = std::iter::repeat_with(ThreeMinuteBarBuilder::default)
        .take(THREE_MINUTE_BARS)
        .collect::<Vec<_>>();
    for idx in indices {
        let Some(trade_time) = trade_times[*idx].as_deref() else {
            continue;
        };
        let Some(minute_idx) = minute_index(trade_time) else {
            continue;
        };
        let slot = minute_idx / THREE_MINUTE_BAR_SIZE;
        if slot >= THREE_MINUTE_BARS {
            continue;
        }
        builders[slot].push(open[*idx], high[*idx], low[*idx], close[*idx], volume[*idx]);
    }
    builders
        .into_iter()
        .filter_map(ThreeMinuteBarBuilder::finish)
        .collect()
}

fn elastic_daily_stats(bars: &[ThreeMinuteBar]) -> ElasticDailyStats {
    let segments = uptrend_segments(bars);
    if segments.is_empty() {
        return ElasticDailyStats::default();
    }

    let mut total_return = 0.0;
    let mut total_volume = 0.0;
    let mut count = 0usize;
    for segment in segments {
        let start = bars[segment.start];
        let end = bars[segment.end];
        if start.low <= EPS {
            continue;
        }
        let segment_return = end.high / start.low - 1.0;
        let segment_volume = bars[segment.start..=segment.end]
            .iter()
            .map(|bar| bar.volume)
            .sum::<f64>();
        let (Some(segment_return), Some(segment_volume)) =
            (finite_value(segment_return), finite_value(segment_volume))
        else {
            continue;
        };
        total_return += segment_return;
        total_volume += segment_volume;
        count += 1;
    }
    if count == 0 || total_volume <= EPS {
        return ElasticDailyStats::default();
    }
    let coefficient = total_return / total_volume;
    ElasticDailyStats {
        total_return: finite_value(total_return),
        total_volume: finite_value(total_volume),
        coefficient: finite_value(coefficient),
    }
}

fn uptrend_segments(bars: &[ThreeMinuteBar]) -> Vec<Segment> {
    uptrend_segments_with_radius(bars, EXTREMA_RADIUS)
}

fn uptrend_segments_with_radius(bars: &[ThreeMinuteBar], radius: usize) -> Vec<Segment> {
    let starts = (0..bars.len())
        .map(|idx| is_local_low(bars, idx, radius))
        .collect::<Vec<_>>();
    let ends = (0..bars.len())
        .map(|idx| is_local_high(bars, idx, radius))
        .collect::<Vec<_>>();

    let mut output = Vec::new();
    let mut idx = 0usize;
    while idx < bars.len() {
        let Some(start) = (idx..bars.len()).find(|candidate| starts[*candidate]) else {
            break;
        };
        let Some(end) = ((start + 1)..bars.len()).find(|candidate| ends[*candidate]) else {
            break;
        };
        output.push(Segment { start, end });
        idx = end + 1;
    }
    output
}

fn is_local_low(bars: &[ThreeMinuteBar], idx: usize, radius: usize) -> bool {
    if idx >= bars.len() {
        return false;
    }
    let start = idx.saturating_sub(radius);
    let end = (idx + radius + 1).min(bars.len());
    let current = bars[idx].low;
    bars[start..end].iter().all(|bar| current <= bar.low + EPS)
}

fn is_local_high(bars: &[ThreeMinuteBar], idx: usize, radius: usize) -> bool {
    if idx >= bars.len() {
        return false;
    }
    let start = idx.saturating_sub(radius);
    let end = (idx + radius + 1).min(bars.len());
    let current = bars[idx].high;
    bars[start..end].iter().all(|bar| current + EPS >= bar.high)
}

fn volume_energy_divergence_from_indices(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
    amount: &[Option<f64>],
    volume: &[Option<f64>],
) -> Option<f64> {
    let mut day_close = None;
    let mut total_volume = 0.0;
    let mut raw_points = Vec::<(f64, f64)>::new();

    for idx in indices {
        let Some(trade_time) = trade_times[*idx].as_deref() else {
            continue;
        };
        if minute_index(trade_time).is_none() {
            continue;
        }
        if let Some(close) = clean_positive(close[*idx]) {
            day_close = Some(close);
        }
        let volume = clean_nonnegative(volume[*idx]).unwrap_or(0.0);
        total_volume += volume;
        if volume <= EPS {
            continue;
        }
        let Some(vwap) = minute_vwap_from_amount_vol(amount[*idx], Some(volume))
            .and_then(finite_value)
            .filter(|value| *value > 0.0)
        else {
            continue;
        };
        raw_points.push((vwap, volume));
    }

    let day_close = day_close.filter(|value| *value > EPS)?;
    if total_volume <= EPS || raw_points.len() < 2 {
        return None;
    }
    let mut points = raw_points
        .into_iter()
        .filter_map(|(vwap, volume)| {
            let price_deviation = vwap / day_close - 1.0;
            let volume_share = volume / total_volume;
            Some(DivergencePoint {
                price_deviation: finite_value(price_deviation)?,
                volume_share: finite_value(volume_share)?,
            })
        })
        .collect::<Vec<_>>();
    if points.len() < 2 {
        return None;
    }

    points.sort_by(|left, right| {
        left.price_deviation
            .partial_cmp(&right.price_deviation)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let positive_integral = cumulative_share_integral(&points);
    points.reverse();
    let negative_integral = cumulative_share_integral(&points);
    finite_value(positive_integral - negative_integral)
}

fn cumulative_share_integral(points: &[DivergencePoint]) -> f64 {
    let mut cumulative = 0.0;
    let mut total = 0.0;
    for point in points {
        cumulative += point.volume_share;
        total += cumulative;
    }
    total
}

impl ThreeMinuteBarBuilder {
    fn push(
        &mut self,
        open: Option<f64>,
        high: Option<f64>,
        low: Option<f64>,
        close: Option<f64>,
        volume: Option<f64>,
    ) {
        let (Some(open), Some(high), Some(low), Some(close), Some(volume)) = (
            clean_positive(open),
            clean_positive(high),
            clean_positive(low),
            clean_positive(close),
            clean_nonnegative(volume),
        ) else {
            return;
        };
        if self.open.is_none() {
            self.open = Some(open);
        }
        self.high = Some(self.high.map_or(high, |current| current.max(high)));
        self.low = Some(self.low.map_or(low, |current| current.min(low)));
        self.close = Some(close);
        self.volume += volume;
        self.valid_count += 1;
    }

    fn finish(self) -> Option<ThreeMinuteBar> {
        if self.valid_count != THREE_MINUTE_BAR_SIZE {
            return None;
        }
        let _open = self.open?;
        let _close = self.close?;
        Some(ThreeMinuteBar {
            high: self.high?,
            low: self.low?,
            volume: self.volume,
        })
    }
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
        "intraday",
        "minute_agg",
        "tension",
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

fn description(def: MszqPriceVolumeTensionFactorDef) -> String {
    format!(
        "{} composites an elastic potential gap component and a volume energy divergence component from 1-minute data, keeps the report's reverse raw direction, and neutralizes by Barra SIZE and SW sector; it does not depend on derived intraday bars.",
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
