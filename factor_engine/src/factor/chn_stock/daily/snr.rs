use std::collections::BTreeMap;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_size_sector};
use crate::factor::common::{clean_intraday_value, stock_minute_raw_spec};
use crate::factor::Factor;
use crate::operators::{cs_minmax_scale, ts_ewm};

const VERSION: &str = "0.1.0";
const RAW_VERSION: &str = "0.2.0";
const PROVIDER_KEY: &str = "hazq_snr_provider";
const RAW_ID: &str = "daily_hazq_snr_raw";

const MINUTES_PER_DAY: usize = 240;
const EMA_SPAN: usize = 15;
const MIN_PERIODS: usize = 1;
const EPS: f64 = 1e-12;

pub struct StockDailySnr;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailySnr)
}

impl Factor for StockDailySnr {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "snr".to_string(),
            aliases: vec!["SNR".to_string()],
            name: "snr".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "HAZQ single-day 1-minute close-price EMD signal-to-noise ratio factor. Layer-2 and layer-3 EMD SNR are combined by cross-sectional minmax-scaled intraday close-price volatility, then 15-day EMA smoothed and neutralized by Barra SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(RAW_ID, EMA_SPAN - 1)],
            lookback: Lookback {
                trading_days: EMA_SPAN - 1,
            },
        }
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        vec![raw_spec()]
    }

    fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
        PROVIDER_KEY.to_string()
    }

    fn minute_compute_many(
        &self,
        raw_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<IntradayDailyRawSeries>> {
        let requested = raw_ids
            .iter()
            .map(String::as_str)
            .any(|raw_id| raw_id == RAW_ID);
        if !requested {
            return Ok(Vec::new());
        }

        let mut values = Vec::<FactorValue>::new();
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
                if is_bj_stock(&ts_code) || trade_times[idx].is_none() {
                    continue;
                }
                grouped.entry(ts_code).or_default().push(idx);
            }

            let mut rows = Vec::<(String, StockSnrValues)>::new();
            for (ts_code, mut indices) in grouped {
                indices.sort_by(|left, right| trade_times[*left].cmp(&trade_times[*right]));
                let close_day = close_day_from_indices(&indices, &trade_times, &close);
                rows.push((ts_code, stock_snr_values(&close_day)));
            }

            let volatilities = rows
                .iter()
                .map(|(_, values)| values.intraday_volatility)
                .collect::<Vec<_>>();
            let vol_minmax = cs_minmax_scale(&volatilities);

            for ((ts_code, stock_values), vol_scale) in rows.into_iter().zip(vol_minmax.into_iter())
            {
                let key = FactorRowKey::Daily {
                    trade_date: *trade_date,
                    ts_code,
                };
                let composite = match (
                    stock_values.layer2_snr,
                    stock_values.layer3_snr,
                    clean_intraday_value(vol_scale),
                ) {
                    (Some(layer2), Some(layer3), Some(weight)) => {
                        Some(weight * layer3 + (1.0 - weight) * layer2)
                    }
                    _ => None,
                };
                values.push(FactorValue {
                    key,
                    value: composite,
                });
            }
        }

        Ok(vec![IntradayDailyRawSeries {
            spec: raw_spec(),
            values,
        }])
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(RAW_ID)?;
        let raw = panel.column(RAW_ID)?;
        let smoothed = raw.ts(|series| ts_ewm(series, EMA_SPAN, MIN_PERIODS))?;
        let factor = neutralize_size_sector(&smoothed, &panel, data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

#[derive(Clone, Debug)]
struct CloseDay {
    close: [Option<f64>; MINUTES_PER_DAY],
}

impl Default for CloseDay {
    fn default() -> Self {
        Self {
            close: [None; MINUTES_PER_DAY],
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct StockSnrValues {
    layer2_snr: Option<f64>,
    layer3_snr: Option<f64>,
    intraday_volatility: Option<f64>,
}

fn raw_spec() -> IntradayDailyRawSpec {
    stock_minute_raw_spec(RAW_ID, RAW_VERSION, &["close"], 1)
}

fn tags() -> Vec<String> {
    [
        "HAZQ",
        "price",
        "emd",
        "snr",
        "signal_noise",
        "intraday",
        "minute_agg",
        "neutralize",
        "barra",
        "size",
        "sector",
        "daily",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect()
}

fn close_day_from_indices(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
) -> CloseDay {
    let mut day = CloseDay::default();
    for idx in indices {
        let value = clean_intraday_value(close[*idx]).filter(|value| *value > 0.0);
        let Some(trade_time) = trade_times[*idx].as_deref() else {
            continue;
        };
        if is_anchor_minute(trade_time) {
            continue;
        }
        if let Some(minute_idx) = minute_index(trade_time) {
            day.close[minute_idx] = value;
        }
    }
    day
}

fn stock_snr_values(day: &CloseDay) -> StockSnrValues {
    let mut prices = Vec::with_capacity(MINUTES_PER_DAY);
    for value in day.close {
        let Some(value) = value else {
            return StockSnrValues {
                layer2_snr: None,
                layer3_snr: None,
                intraday_volatility: None,
            };
        };
        prices.push(value);
    }
    let (layer2_snr, layer3_snr) = snr_layer_values(&prices, true, true);
    let intraday_volatility = sample_std(&prices);
    StockSnrValues {
        layer2_snr,
        layer3_snr,
        intraday_volatility,
    }
}

fn snr_layer_values(
    prices: &[f64],
    need_layer2: bool,
    need_layer3: bool,
) -> (Option<f64>, Option<f64>) {
    if !need_layer2 && !need_layer3 {
        return (None, None);
    }
    if prices.len() < 3 || is_constant(prices) {
        return (need_layer2.then_some(0.0), need_layer3.then_some(0.0));
    }

    let max_layer = if need_layer3 { 3 } else { 2 };
    let mut residual = prices.to_vec();
    let mut layer2_snr = None;
    let mut layer3_snr = None;
    for layer in 1..=max_layer {
        let Some(next) = emd_trend_once(&residual) else {
            if layer <= 2 && need_layer2 {
                layer2_snr = Some(0.0);
            }
            if need_layer3 {
                layer3_snr = Some(0.0);
            }
            return (layer2_snr, layer3_snr);
        };
        residual = next;
        if layer == 2 && need_layer2 {
            layer2_snr = Some(snr_from_signal(prices, &residual));
        }
        if layer == 3 && need_layer3 {
            layer3_snr = Some(snr_from_signal(prices, &residual));
        }
    }
    (layer2_snr, layer3_snr)
}

fn snr_from_signal(prices: &[f64], signal: &[f64]) -> f64 {
    let noise = prices
        .iter()
        .zip(signal.iter())
        .map(|(price, trend)| price - trend)
        .collect::<Vec<_>>();
    let (Some(signal_std), Some(noise_std)) = (sample_std(&signal), sample_std(&noise)) else {
        return 0.0;
    };
    if signal_std <= EPS || noise_std <= EPS {
        return 0.0;
    }
    let value = (signal_std / noise_std).ln();
    value.is_finite().then_some(value).unwrap_or(0.0)
}

fn emd_trend_once(values: &[f64]) -> Option<Vec<f64>> {
    let maxima = extrema_knots(values, ExtremaKind::Max);
    let minima = extrema_knots(values, ExtremaKind::Min);
    if maxima.len() < 3 || minima.len() < 3 {
        return None;
    }
    let upper = natural_cubic_spline_values(&maxima, values.len())?;
    let lower = natural_cubic_spline_values(&minima, values.len())?;
    Some(
        upper
            .iter()
            .zip(lower.iter())
            .map(|(upper, lower)| (upper + lower) / 2.0)
            .collect(),
    )
}

#[derive(Clone, Copy)]
enum ExtremaKind {
    Max,
    Min,
}

fn extrema_knots(values: &[f64], kind: ExtremaKind) -> Vec<(usize, f64)> {
    let mut knots = Vec::new();
    knots.push((0, values[0]));
    for idx in 1..values.len() - 1 {
        let is_extrema = match kind {
            ExtremaKind::Max => values[idx - 1] < values[idx] && values[idx] > values[idx + 1],
            ExtremaKind::Min => values[idx - 1] > values[idx] && values[idx] < values[idx + 1],
        };
        if is_extrema {
            knots.push((idx, values[idx]));
        }
    }
    let last_idx = values.len() - 1;
    knots.push((last_idx, values[last_idx]));
    knots
}

fn natural_cubic_spline_values(knots: &[(usize, f64)], shape: usize) -> Option<Vec<f64>> {
    if knots.len() < 3 || shape == 0 {
        return None;
    }
    let second = natural_cubic_second_derivatives(knots)?;
    let mut output = Vec::with_capacity(shape);
    let mut seg = 0usize;
    for x_idx in 0..shape {
        while seg + 1 < knots.len() - 1 && x_idx > knots[seg + 1].0 {
            seg += 1;
        }
        let x0 = knots[seg].0 as f64;
        let x1 = knots[seg + 1].0 as f64;
        let y0 = knots[seg].1;
        let y1 = knots[seg + 1].1;
        let h = x1 - x0;
        if h <= 0.0 {
            return None;
        }
        let x = x_idx as f64;
        let a = (x1 - x) / h;
        let b = (x - x0) / h;
        let value = a * y0
            + b * y1
            + ((a * a * a - a) * second[seg] + (b * b * b - b) * second[seg + 1]) * h * h / 6.0;
        output.push(value);
    }
    Some(output)
}

fn natural_cubic_second_derivatives(knots: &[(usize, f64)]) -> Option<Vec<f64>> {
    let k = knots.len();
    if k < 3 {
        return None;
    }
    let interior = k - 2;
    let mut second = vec![0.0; k];
    let mut diag = vec![0.0; interior];
    let mut upper = vec![0.0; interior.saturating_sub(1)];
    let mut lower = vec![0.0; interior.saturating_sub(1)];
    let mut rhs = vec![0.0; interior];

    for row in 0..interior {
        let i = row + 1;
        let h_prev = (knots[i].0 - knots[i - 1].0) as f64;
        let h_next = (knots[i + 1].0 - knots[i].0) as f64;
        if h_prev <= 0.0 || h_next <= 0.0 {
            return None;
        }
        diag[row] = 2.0 * (h_prev + h_next);
        rhs[row] =
            6.0 * ((knots[i + 1].1 - knots[i].1) / h_next - (knots[i].1 - knots[i - 1].1) / h_prev);
        if row > 0 {
            lower[row - 1] = h_prev;
        }
        if row + 1 < interior {
            upper[row] = h_next;
        }
    }

    let solution = solve_tridiagonal(&lower, &diag, &upper, &rhs)?;
    second[1..(interior + 1)].copy_from_slice(&solution[..interior]);
    Some(second)
}

fn solve_tridiagonal(lower: &[f64], diag: &[f64], upper: &[f64], rhs: &[f64]) -> Option<Vec<f64>> {
    let n = diag.len();
    if rhs.len() != n || lower.len() + 1 < n || upper.len() + 1 < n {
        return None;
    }
    if n == 0 {
        return Some(Vec::new());
    }

    let mut c_prime = vec![0.0; n.saturating_sub(1)];
    let mut d_prime = vec![0.0; n];
    if diag[0].abs() <= EPS {
        return None;
    }
    if n > 1 {
        c_prime[0] = upper[0] / diag[0];
    }
    d_prime[0] = rhs[0] / diag[0];

    for idx in 1..n {
        let denom = diag[idx] - lower[idx - 1] * c_prime[idx - 1];
        if denom.abs() <= EPS {
            return None;
        }
        if idx < n - 1 {
            c_prime[idx] = upper[idx] / denom;
        }
        d_prime[idx] = (rhs[idx] - lower[idx - 1] * d_prime[idx - 1]) / denom;
    }

    let mut output = vec![0.0; n];
    output[n - 1] = d_prime[n - 1];
    for idx in (0..n - 1).rev() {
        output[idx] = d_prime[idx] - c_prime[idx] * output[idx + 1];
    }
    Some(output)
}

fn sample_std(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    let std = variance.sqrt();
    std.is_finite().then_some(std)
}

fn is_constant(values: &[f64]) -> bool {
    let first = values[0];
    values.iter().all(|value| (*value - first).abs() <= EPS)
}

fn is_anchor_minute(trade_time: &str) -> bool {
    matches!(time_to_minutes(trade_time), Some(minutes) if minutes == 9 * 60 + 30)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-10,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn hazq_snr_natural_cubic_preserves_linear_path() {
        let knots = vec![(0, 0.0), (2, 2.0), (3, 3.0)];
        let values = natural_cubic_spline_values(&knots, 4).expect("spline");
        assert_close(values[0], 0.0);
        assert_close(values[1], 1.0);
        assert_close(values[2], 2.0);
        assert_close(values[3], 3.0);
    }

    #[test]
    fn hazq_snr_extrema_are_strict_and_include_endpoints() {
        let values = vec![1.0, 3.0, 2.0, 4.0, 1.0];
        let maxima = extrema_knots(&values, ExtremaKind::Max);
        let minima = extrema_knots(&values, ExtremaKind::Min);
        assert_eq!(maxima, vec![(0, 1.0), (1, 3.0), (3, 4.0), (4, 1.0)]);
        assert_eq!(minima, vec![(0, 1.0), (2, 2.0), (4, 1.0)]);
    }

    #[test]
    fn hazq_snr_flat_or_extrema_poor_series_returns_zero() {
        assert_eq!(
            snr_layer_values(&vec![10.0; MINUTES_PER_DAY], true, true),
            (Some(0.0), Some(0.0))
        );
        let monotonic = (0..MINUTES_PER_DAY)
            .map(|idx| idx as f64 + 1.0)
            .collect::<Vec<_>>();
        assert_eq!(
            snr_layer_values(&monotonic, true, true),
            (Some(0.0), Some(0.0))
        );
    }

    #[test]
    fn hazq_snr_requires_complete_regular_session_prices() {
        let day = CloseDay::default();
        let values = stock_snr_values(&day);
        assert_eq!(values.layer2_snr, None);
        assert_eq!(values.layer3_snr, None);
        assert_eq!(values.intraday_volatility, None);
    }

    #[test]
    fn hazq_snr_minute_index_uses_regular_session_and_anchor() {
        assert!(is_anchor_minute("09:30:00"));
        assert_eq!(minute_index("09:31:00"), Some(0));
        assert_eq!(minute_index("11:30:00"), Some(119));
        assert_eq!(minute_index("13:01:00"), Some(120));
        assert_eq!(minute_index("15:00:00"), Some(239));
        assert_eq!(minute_index("09:30:00"), None);
    }

    #[test]
    fn hazq_snr_factor_spec_has_hazq_tag() {
        let spec = StockDailySnr.spec();
        assert_eq!(spec.id, "snr");
        assert!(spec.tags.iter().any(|tag| tag == "HAZQ"));
        assert_eq!(spec.intraday_raw_dependencies[0].raw_id, RAW_ID);
    }
}
