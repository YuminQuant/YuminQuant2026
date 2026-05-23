use std::collections::{BTreeMap, BTreeSet};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::neutralize_size_sector;
use crate::factor::common::{clean_intraday_value, quantile_linear, stock_minute_raw_spec};
use crate::factor::common::{DailyPanel, PanelColumn};
use crate::operators::{cs_zscore, ts_mean};

pub const PROVIDER_KEY: &str = "dbzq_intraday_volume_distribution_provider";
pub const RAW_VERSION: &str = "0.1.0";
pub const VERSION: &str = "0.1.0";

pub const V_P_SKEWNESS_RAW_ID: &str = "daily_dbzq_v_p_skewness";
pub const V_P_REVERSAL_RAW_ID: &str = "daily_dbzq_v_p_reversal";
pub const SIG_UP_P_V_RATIO_RAW_ID: &str = "daily_dbzq_sig_up_p_v_ratio";
pub const SIG_UP_P_V_INTRADAY_STD_RAW_ID: &str = "daily_dbzq_sig_up_p_v_intraday_std";

const RAW_WINDOW_DAYS: usize = 1;
const ROLLING_WINDOW: usize = 20;
const MIN_PERIODS: usize = 1;
const FIVE_MINUTE_BARS: usize = 48;
const MIN_PRICE_BARS: usize = 10;
const PRICE_BIN_COUNT: usize = 10;
const EPS: f64 = f64::EPSILON;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DbzqIntradayVolumeDistributionKind {
    VolumePrice,
    SignificantUpVolumeReturn,
}

#[derive(Clone, Copy, Debug)]
pub struct DbzqIntradayVolumeDistributionFactorDef {
    pub id: &'static str,
    pub alias: &'static str,
    pub name: &'static str,
    pub kind: DbzqIntradayVolumeDistributionKind,
}

#[derive(Clone, Copy, Debug, Default)]
struct FiveMinuteBarBuilder {
    minute_count: usize,
    open: Option<f64>,
    close: Option<f64>,
    volume: f64,
    amount: f64,
    has_volume: bool,
    has_amount: bool,
}

#[derive(Clone, Copy, Debug)]
struct FiveMinuteBar {
    open: f64,
    close: f64,
    volume: f64,
    amount: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct DailyStats {
    v_p_skewness: Option<f64>,
    v_p_reversal: Option<f64>,
    sig_up_p_v_ratio: Option<f64>,
    sig_up_p_v_intraday_std: Option<f64>,
}

pub fn all_raw_ids() -> [&'static str; 4] {
    [
        V_P_SKEWNESS_RAW_ID,
        V_P_REVERSAL_RAW_ID,
        SIG_UP_P_V_RATIO_RAW_ID,
        SIG_UP_P_V_INTRADAY_STD_RAW_ID,
    ]
}

pub fn raw_ids_for_kind(kind: DbzqIntradayVolumeDistributionKind) -> &'static [&'static str] {
    match kind {
        DbzqIntradayVolumeDistributionKind::VolumePrice => {
            &[V_P_SKEWNESS_RAW_ID, V_P_REVERSAL_RAW_ID]
        }
        DbzqIntradayVolumeDistributionKind::SignificantUpVolumeReturn => {
            &[SIG_UP_P_V_RATIO_RAW_ID, SIG_UP_P_V_INTRADAY_STD_RAW_ID]
        }
    }
}

pub fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
    stock_minute_raw_spec(
        raw_id,
        RAW_VERSION,
        &["open", "close", "vol", "amount"],
        RAW_WINDOW_DAYS,
    )
}

pub fn raw_specs_for_kind(kind: DbzqIntradayVolumeDistributionKind) -> Vec<IntradayDailyRawSpec> {
    raw_ids_for_kind(kind)
        .iter()
        .map(|raw_id| raw_spec(raw_id))
        .collect()
}

pub fn factor_spec(def: DbzqIntradayVolumeDistributionFactorDef) -> FactorSpec {
    FactorSpec {
        id: def.id.to_string(),
        aliases: vec![def.alias.to_string()],
        name: def.name.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: tags(def.kind),
        description: description(def),
        dependencies: dependencies(),
        intraday_raw_dependencies: raw_ids_for_kind(def.kind)
            .iter()
            .map(|raw_id| IntradayDailyRawRequest::new(raw_id, ROLLING_WINDOW - 1))
            .collect(),
        lookback: Lookback {
            trading_days: ROLLING_WINDOW - 1,
        },
    }
}

pub fn compute_factor(
    def: DbzqIntradayVolumeDistributionFactorDef,
    data: &DataPool,
) -> Result<FactorSeries> {
    match def.kind {
        DbzqIntradayVolumeDistributionKind::VolumePrice => compute_volume_price_factor(def, data),
        DbzqIntradayVolumeDistributionKind::SignificantUpVolumeReturn => {
            compute_significant_up_volume_return_factor(def, data)
        }
    }
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
            let bars = five_minute_bars_from_indices(
                &indices,
                &trade_times,
                &open,
                &close,
                &volume,
                &amount,
            );
            let stats = daily_stats(&bars);
            let key = FactorRowKey::Daily {
                trade_date: *trade_date,
                ts_code,
            };
            push_requested(
                &mut values,
                &requested,
                V_P_SKEWNESS_RAW_ID,
                &key,
                stats.v_p_skewness,
            );
            push_requested(
                &mut values,
                &requested,
                V_P_REVERSAL_RAW_ID,
                &key,
                stats.v_p_reversal,
            );
            push_requested(
                &mut values,
                &requested,
                SIG_UP_P_V_RATIO_RAW_ID,
                &key,
                stats.sig_up_p_v_ratio,
            );
            push_requested(
                &mut values,
                &requested,
                SIG_UP_P_V_INTRADAY_STD_RAW_ID,
                &key,
                stats.sig_up_p_v_intraday_std,
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

fn compute_volume_price_factor(
    def: DbzqIntradayVolumeDistributionFactorDef,
    data: &DataPool,
) -> Result<FactorSeries> {
    let panel = data.intraday_daily_raw_panel(V_P_SKEWNESS_RAW_ID)?;
    let skew = panel.column(V_P_SKEWNESS_RAW_ID)?;
    let reversal = panel.column(V_P_REVERSAL_RAW_ID)?;

    let skew_component = rolling_mean_negative_zscore(&skew)?;
    let reversal_component = rolling_mean_negative_zscore(&reversal)?;
    let composite = average_columns(&panel, &[&skew_component, &reversal_component])?;
    let factor = neutralize_size_sector(&composite, &panel, data)?;
    Ok(factor.to_factor_series(factor_spec(def)))
}

fn compute_significant_up_volume_return_factor(
    def: DbzqIntradayVolumeDistributionFactorDef,
    data: &DataPool,
) -> Result<FactorSeries> {
    let panel = data.intraday_daily_raw_panel(SIG_UP_P_V_RATIO_RAW_ID)?;
    let ratio = panel.column(SIG_UP_P_V_RATIO_RAW_ID)?;
    let intraday_std = panel.column(SIG_UP_P_V_INTRADAY_STD_RAW_ID)?;

    let ratio_mean = rolling_mean_negative_zscore(&ratio)?;
    let ratio_stability = rolling_variance_negative_zscore(&ratio)?;
    let intraday_std_mean = rolling_mean_negative_zscore(&intraday_std)?;
    let intraday_std_stability = rolling_variance_negative_zscore(&intraday_std)?;
    let composite = average_columns(
        &panel,
        &[
            &ratio_mean,
            &ratio_stability,
            &intraday_std_mean,
            &intraday_std_stability,
        ],
    )?;
    let factor = neutralize_size_sector(&composite, &panel, data)?;
    Ok(factor.to_factor_series(factor_spec(def)))
}

fn rolling_mean_negative_zscore(values: &PanelColumn) -> Result<PanelColumn> {
    let smoothed = values.ts(|series| ts_mean(series, ROLLING_WINDOW, MIN_PERIODS))?;
    smoothed
        .map_values(|value| finite_option(value.map(|value| -value)))
        .cs(cs_zscore)
}

fn rolling_variance_negative_zscore(values: &PanelColumn) -> Result<PanelColumn> {
    let variance = values.ts(|series| rolling_variance(series, ROLLING_WINDOW, MIN_PERIODS))?;
    variance
        .map_values(|value| finite_option(value.map(|value| -value)))
        .cs(cs_zscore)
}

fn rolling_variance(values: &[Option<f64>], window: usize, min_periods: usize) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    let min_periods = min_periods.max(1).min(window);
    for idx in 0..values.len() {
        let start = (idx + 1).saturating_sub(window);
        let valid = values[start..=idx]
            .iter()
            .filter_map(|value| finite_option(*value))
            .collect::<Vec<_>>();
        if valid.len() < min_periods {
            continue;
        }
        output[idx] = variance(&valid);
    }
    output
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

fn tags(kind: DbzqIntradayVolumeDistributionKind) -> Vec<String> {
    let mut tags = [
        "price_volume",
        "volume",
        "intraday",
        "minute_agg",
        "distribution",
        "neutralize",
        "barra",
        "size",
        "sector",
        "daily",
        "DBZQ",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect::<Vec<_>>();
    match kind {
        DbzqIntradayVolumeDistributionKind::VolumePrice => tags.push("price".to_string()),
        DbzqIntradayVolumeDistributionKind::SignificantUpVolumeReturn => {
            tags.push("return".to_string())
        }
    }
    tags
}

fn description(def: DbzqIntradayVolumeDistributionFactorDef) -> String {
    match def.kind {
        DbzqIntradayVolumeDistributionKind::VolumePrice => format!(
            "{} composites intraday 5-minute VWAP volume-at-price skewness and POC reversal raws from 1-minute bars, then z-scores subfactors and neutralizes by Barra SIZE and SW sector; it does not depend on derived 5-minute parquet bars.",
            def.name
        ),
        DbzqIntradayVolumeDistributionKind::SignificantUpVolumeReturn => format!(
            "{} composites significant-up 5-minute return-volume raws from 1-minute bars, then z-scores subfactors and neutralizes by Barra SIZE and SW sector; it does not depend on derived 5-minute parquet bars.",
            def.name
        ),
    }
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

fn five_minute_bars_from_indices(
    indices: &[usize],
    trade_times: &[Option<String>],
    open: &[Option<f64>],
    close: &[Option<f64>],
    volume: &[Option<f64>],
    amount: &[Option<f64>],
) -> Vec<FiveMinuteBar> {
    let mut builders = std::iter::repeat_with(FiveMinuteBarBuilder::default)
        .take(FIVE_MINUTE_BARS)
        .collect::<Vec<_>>();
    for idx in indices {
        let Some(trade_time) = trade_times[*idx].as_deref() else {
            continue;
        };
        let Some(minute_idx) = minute_index(trade_time) else {
            continue;
        };
        let slot = minute_idx / 5;
        if slot >= FIVE_MINUTE_BARS {
            continue;
        }
        builders[slot].push(open[*idx], close[*idx], volume[*idx], amount[*idx]);
    }
    builders
        .into_iter()
        .filter_map(FiveMinuteBarBuilder::finish)
        .collect()
}

fn daily_stats(bars: &[FiveMinuteBar]) -> DailyStats {
    let (v_p_skewness, v_p_reversal) = price_distribution_stats(bars);
    let (sig_up_p_v_ratio, sig_up_p_v_intraday_std) = significant_up_stats(bars);
    DailyStats {
        v_p_skewness,
        v_p_reversal,
        sig_up_p_v_ratio,
        sig_up_p_v_intraday_std,
    }
}

fn price_distribution_stats(bars: &[FiveMinuteBar]) -> (Option<f64>, Option<f64>) {
    let price_volume = bars
        .iter()
        .filter_map(|bar| {
            let vwap = bar.vwap()?;
            let volume = finite_value(bar.volume)?;
            (volume > EPS).then_some((vwap, volume))
        })
        .collect::<Vec<_>>();
    if price_volume.len() < MIN_PRICE_BARS {
        return (None, None);
    }
    let total_volume = price_volume.iter().map(|(_, volume)| *volume).sum::<f64>();
    if total_volume <= EPS {
        return (None, None);
    }

    let prices = price_volume
        .iter()
        .map(|(price, _)| *price)
        .collect::<Vec<_>>();
    let Some(edges) = price_quantile_edges(&prices) else {
        return (None, None);
    };
    let mut bin_volumes = [0.0; PRICE_BIN_COUNT];
    for (price, volume) in &price_volume {
        let Some(bin) = price_bin(*price, &edges) else {
            continue;
        };
        bin_volumes[bin] += *volume;
    }

    let skewness = pearson_median_skewness(&bin_volumes, total_volume);
    let reversal = price_reversal(bars, &edges, &bin_volumes);
    (skewness, reversal)
}

fn significant_up_stats(bars: &[FiveMinuteBar]) -> (Option<f64>, Option<f64>) {
    let points = bars
        .iter()
        .filter_map(|bar| {
            let ret = bar_return(bar)?;
            let volume = finite_value(bar.volume)?;
            Some((ret, volume))
        })
        .collect::<Vec<_>>();
    if points.len() < 2 {
        return (None, None);
    }
    let returns = points.iter().map(|(ret, _)| *ret).collect::<Vec<_>>();
    let volumes = points.iter().map(|(_, volume)| *volume).collect::<Vec<_>>();
    let Some(return_mean) = mean(&returns) else {
        return (None, None);
    };
    let Some(return_std) = std_dev(&returns) else {
        return (None, None);
    };
    let threshold = return_mean + return_std;
    let significant = points
        .iter()
        .copied()
        .filter(|(ret, _)| *ret > 0.0 && *ret > threshold)
        .collect::<Vec<_>>();
    if significant.len() < 2 {
        return (None, None);
    }

    let sig_returns = significant.iter().map(|(ret, _)| *ret).collect::<Vec<_>>();
    let sig_volumes = significant
        .iter()
        .map(|(_, volume)| *volume)
        .collect::<Vec<_>>();
    let Some(sig_return_std) = std_dev(&sig_returns) else {
        return (None, None);
    };
    let Some(all_avg_volume) = mean(&volumes).filter(|value| value.abs() > EPS) else {
        return (None, None);
    };
    let Some(sig_avg_volume) = mean(&sig_volumes) else {
        return (None, None);
    };
    let volume_ratio = sig_avg_volume / all_avg_volume;
    let ratio = (volume_ratio.abs() > EPS).then_some(sig_return_std / volume_ratio);
    let intraday_std = std_dev(&sig_volumes)
        .and_then(|sig_volume_std| finite_value(sig_return_std * sig_volume_std / all_avg_volume));
    (ratio.and_then(finite_value), intraday_std)
}

fn price_quantile_edges(prices: &[f64]) -> Option<Vec<f64>> {
    if prices.is_empty() {
        return None;
    }
    let mut edges = Vec::with_capacity(PRICE_BIN_COUNT + 1);
    for idx in 0..=PRICE_BIN_COUNT {
        let mut values = prices.to_vec();
        edges.push(quantile_linear(
            &mut values,
            idx as f64 / PRICE_BIN_COUNT as f64,
        )?);
    }
    Some(edges)
}

fn price_bin(price: f64, edges: &[f64]) -> Option<usize> {
    if edges.len() != PRICE_BIN_COUNT + 1 || !price.is_finite() {
        return None;
    }
    for idx in 0..PRICE_BIN_COUNT {
        if idx + 1 == PRICE_BIN_COUNT {
            if price >= edges[idx] - EPS && price <= edges[idx + 1] + EPS {
                return Some(idx);
            }
        } else if price >= edges[idx] - EPS && price <= edges[idx + 1] + EPS {
            return Some(idx);
        }
    }
    None
}

fn pearson_median_skewness(bin_volumes: &[f64; PRICE_BIN_COUNT], total_volume: f64) -> Option<f64> {
    if total_volume <= EPS {
        return None;
    }
    let shares = bin_volumes
        .iter()
        .map(|volume| *volume / total_volume)
        .collect::<Vec<_>>();
    let mean_label = shares
        .iter()
        .enumerate()
        .map(|(idx, share)| bin_label(idx) * share)
        .sum::<f64>();
    let median_label = weighted_median_label(&shares)?;
    let variance = shares
        .iter()
        .enumerate()
        .map(|(idx, share)| {
            let diff = bin_label(idx) - mean_label;
            share * diff * diff
        })
        .sum::<f64>();
    if variance <= EPS {
        return None;
    }
    finite_value(3.0 * (mean_label - median_label) / variance.sqrt())
}

fn price_reversal(
    bars: &[FiveMinuteBar],
    edges: &[f64],
    bin_volumes: &[f64; PRICE_BIN_COUNT],
) -> Option<f64> {
    let poc_bin = poc_bin(bin_volumes)?;
    let close = bars.last()?.close;
    let close_bin = price_bin(close, edges)?;
    let first_open = bars.first()?.open;
    let last_close = bars.last()?.close;
    if first_open.abs() <= EPS {
        return None;
    }
    let daily_return = last_close / first_open - 1.0;
    let poc = bin_label(poc_bin);
    let close_label = bin_label(close_bin);
    let value = if daily_return < 0.0 {
        (poc - close_label + 1.0) * daily_return
    } else {
        (close_label - poc + 1.0) * daily_return
    };
    finite_value(value)
}

fn poc_bin(bin_volumes: &[f64; PRICE_BIN_COUNT]) -> Option<usize> {
    let mut best_idx = None;
    let mut best_value = f64::NEG_INFINITY;
    for idx in 0..PRICE_BIN_COUNT {
        let value = if idx == 0 {
            2.0 * bin_volumes[0] + bin_volumes[1]
        } else if idx + 1 == PRICE_BIN_COUNT {
            bin_volumes[PRICE_BIN_COUNT - 2] + 2.0 * bin_volumes[PRICE_BIN_COUNT - 1]
        } else {
            bin_volumes[idx - 1] + bin_volumes[idx] + bin_volumes[idx + 1]
        };
        if value > best_value {
            best_value = value;
            best_idx = Some(idx);
        }
    }
    best_idx.filter(|_| best_value.is_finite() && best_value > EPS)
}

fn weighted_median_label(shares: &[f64]) -> Option<f64> {
    let mut cumulative = 0.0;
    for (idx, share) in shares.iter().enumerate() {
        cumulative += *share;
        if cumulative >= 0.5 {
            return Some(bin_label(idx));
        }
    }
    shares.iter().rposition(|share| *share > 0.0).map(bin_label)
}

fn bin_label(idx: usize) -> f64 {
    idx as f64 / PRICE_BIN_COUNT as f64
}

fn bar_return(bar: &FiveMinuteBar) -> Option<f64> {
    if bar.open.abs() <= EPS {
        return None;
    }
    finite_value(bar.close / bar.open - 1.0)
}

impl FiveMinuteBarBuilder {
    fn push(
        &mut self,
        open: Option<f64>,
        close: Option<f64>,
        volume: Option<f64>,
        amount: Option<f64>,
    ) {
        self.minute_count += 1;
        if self.open.is_none() {
            self.open = clean_positive(open);
        }
        if let Some(close) = clean_positive(close) {
            self.close = Some(close);
        }
        if let Some(volume) = clean_nonnegative(volume) {
            self.volume += volume;
            self.has_volume = true;
        }
        if let Some(amount) = clean_nonnegative(amount) {
            self.amount += amount;
            self.has_amount = true;
        }
    }

    fn finish(self) -> Option<FiveMinuteBar> {
        if self.minute_count != 5 || !self.has_volume || !self.has_amount {
            return None;
        }
        Some(FiveMinuteBar {
            open: self.open?,
            close: self.close?,
            volume: self.volume,
            amount: self.amount,
        })
    }
}

impl FiveMinuteBar {
    fn vwap(&self) -> Option<f64> {
        if self.volume.abs() <= EPS {
            return None;
        }
        finite_value(self.amount / self.volume)
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
            .map(|value| {
                let diff = value - mean;
                diff * diff
            })
            .sum::<f64>()
            / values.len() as f64,
    )
}

fn std_dev(values: &[f64]) -> Option<f64> {
    variance(values).and_then(|value| finite_value(value.max(0.0).sqrt()))
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

    fn minute_rows_for_one_hour() -> (Vec<usize>, Vec<Option<String>>, Vec<Option<f64>>) {
        let indices = (0..60).collect::<Vec<_>>();
        let times = (0..60)
            .map(|idx| Some(format!("09:{:02}:00", 31 + idx)))
            .collect::<Vec<_>>();
        let values = (0..60)
            .map(|idx| Some((idx + 1) as f64))
            .collect::<Vec<_>>();
        (indices, times, values)
    }

    #[test]
    fn dbzq_volume_distribution_minute_index_maps_stock_sessions() {
        assert_eq!(minute_index("09:31:00"), Some(0));
        assert_eq!(minute_index("09:35:00"), Some(4));
        assert_eq!(minute_index("11:30:00"), Some(119));
        assert_eq!(minute_index("13:01:00"), Some(120));
        assert_eq!(minute_index("15:00:00"), Some(239));
        assert_eq!(minute_index("09:30:00"), None);
    }

    #[test]
    fn dbzq_volume_distribution_builds_complete_five_minute_bars() {
        let (indices, times, values) = minute_rows_for_one_hour();
        let volume = vec![Some(10.0); 60];
        let amount = vec![Some(1000.0); 60];

        let bars =
            five_minute_bars_from_indices(&indices, &times, &values, &values, &volume, &amount);

        assert_eq!(bars.len(), 12);
        assert_close(Some(bars[0].open), 1.0);
        assert_close(Some(bars[0].close), 5.0);
        assert_close(Some(bars[0].volume), 50.0);
        assert_close(bars[0].vwap(), 100.0);
    }

    #[test]
    fn dbzq_volume_distribution_significant_up_requires_positive_tail_returns() {
        let bars = [1.0, 1.0, 1.0, 2.0, 2.1]
            .into_iter()
            .enumerate()
            .map(|(idx, close)| FiveMinuteBar {
                open: 1.0,
                close,
                volume: 10.0 + idx as f64,
                amount: 1000.0 + idx as f64,
            })
            .collect::<Vec<_>>();

        let (ratio, intraday_std) = significant_up_stats(&bars);

        assert!(ratio.is_some());
        assert!(intraday_std.is_some());
    }

    #[test]
    fn dbzq_volume_distribution_price_bins_and_poc_use_vwap_volume() {
        let bars = (0..10)
            .map(|idx| FiveMinuteBar {
                open: 10.0 + idx as f64,
                close: 10.0 + idx as f64,
                volume: if idx == 7 { 100.0 } else { 10.0 },
                amount: (10.0 + idx as f64) * if idx == 7 { 100.0 } else { 10.0 },
            })
            .collect::<Vec<_>>();

        let (skewness, reversal) = price_distribution_stats(&bars);

        assert!(skewness.is_some());
        assert!(reversal.is_some());
        let prices = bars
            .iter()
            .filter_map(FiveMinuteBar::vwap)
            .collect::<Vec<_>>();
        let edges = price_quantile_edges(&prices).expect("edges");
        let close_bin = price_bin(bars.last().unwrap().close, &edges);
        assert_eq!(close_bin, Some(9));
    }

    #[test]
    fn dbzq_volume_distribution_rolling_variance_uses_population_variance() {
        let values = vec![Some(1.0), Some(3.0), Some(5.0)];

        let output = rolling_variance(&values, 3, 1);

        assert_close(output[2], 8.0 / 3.0);
    }

    #[test]
    fn dbzq_volume_distribution_factor_spec_has_dbzq_tag() {
        let spec = factor_spec(DbzqIntradayVolumeDistributionFactorDef {
            id: "volume_price_distribution",
            alias: "volume_price_distribution",
            name: "Volume Price Distribution",
            kind: DbzqIntradayVolumeDistributionKind::VolumePrice,
        });

        assert!(spec.tags.iter().any(|tag| tag == "DBZQ"));
        assert!(spec
            .description
            .contains("does not depend on derived 5-minute parquet bars"));
    }
}
