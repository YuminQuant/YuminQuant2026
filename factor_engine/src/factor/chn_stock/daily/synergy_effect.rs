use std::collections::{BTreeSet, HashMap};

use rayon::prelude::*;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawAuxiliaryRequest, IntradayDailyRawRequest,
    IntradayDailyRawSeries, IntradayDailyRawSpec, Lookback,
};
use crate::data::{DataPool, Table};
use crate::error::Result;
use crate::factor::common::{
    clean_intraday_value, intraday_time_in_range, minute_vwap_from_amount_vol,
    stock_minute_raw_spec, DailyPanel, PanelColumn,
};
use crate::factor::Factor;
use crate::operators::{cs_zscore, ts_mean, ts_std_dev};

pub const VOLUME_SYNERGY_RAW_ID: &str = "daily_volume_synergy";
pub const SYNERGY_SPREAD_RAW_ID: &str = "daily_synergy_spread";

const RAW_VERSION: &str = "0.3.0";
const VERSION: &str = "0.4.0";
const WINDOW: usize = 20;
const OHLC_WINDOW: usize = 5;
const DIFF_WINDOW: usize = 5;
const TOP_PEER_COUNT: usize = 30;

pub struct StockDailySynergyEffect;

#[derive(Clone, Debug)]
struct MinuteSynergyMatrix {
    times: Vec<String>,
    codes: Vec<String>,
    open: Vec<Option<f64>>,
    high: Vec<Option<f64>>,
    low: Vec<Option<f64>>,
    close: Vec<Option<f64>>,
    volume: Vec<Option<f64>>,
    amount: Vec<Option<f64>>,
}

impl MinuteSynergyMatrix {
    fn time_count(&self) -> usize {
        self.times.len()
    }

    fn code_count(&self) -> usize {
        self.codes.len()
    }

    fn offset(&self, time_idx: usize, code_idx: usize) -> usize {
        time_idx * self.code_count() + code_idx
    }
}

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailySynergyEffect)
}

fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
    stock_minute_raw_spec(
        raw_id,
        RAW_VERSION,
        &["open", "high", "low", "close", "vol", "amount"],
        1,
    )
}

impl Factor for StockDailySynergyEffect {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "synergy_effect".to_string(),
            aliases: Vec::new(),
            name: "Synergy Effect".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "return",
                "volume",
                "intraday",
                "minute_agg",
                "correlation",
                "composite",
                "daily",
                "FZZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Composite intraday synergy factor from volume-state co-movement and top-peer excess return spread.".to_string(),
            dependencies: Vec::new(),
            intraday_raw_dependencies: vec![
                IntradayDailyRawRequest::new(VOLUME_SYNERGY_RAW_ID, WINDOW - 1),
                IntradayDailyRawRequest::new(SYNERGY_SPREAD_RAW_ID, WINDOW - 1),
            ],
            lookback: Lookback {
                trading_days: WINDOW - 1,
            },
        }
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        vec![
            raw_spec(VOLUME_SYNERGY_RAW_ID),
            raw_spec(SYNERGY_SPREAD_RAW_ID),
        ]
    }

    fn intraday_raw_auxiliary_requirements(
        &self,
        raw_ids: &[String],
    ) -> Vec<IntradayDailyRawAuxiliaryRequest> {
        if raw_ids.iter().any(|raw_id| raw_id == SYNERGY_SPREAD_RAW_ID) {
            vec![IntradayDailyRawAuxiliaryRequest::new(
                DataRequest::new(DatasetId::StockDailyPv, &["close", "pre_close"]),
                0,
            )]
        } else {
            Vec::new()
        }
    }

    fn minute_compute(
        &self,
        raw_id: &str,
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Option<IntradayDailyRawSeries>> {
        let raw_ids = vec![raw_id.to_string()];
        Ok(self
            .minute_compute_many(&raw_ids, context, data)?
            .into_iter()
            .next())
    }

    fn minute_compute_many(
        &self,
        raw_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<IntradayDailyRawSeries>> {
        let requested = raw_ids.iter().map(String::as_str).collect::<BTreeSet<_>>();
        let wants_volume = requested.contains(VOLUME_SYNERGY_RAW_ID);
        let wants_spread = requested.contains(SYNERGY_SPREAD_RAW_ID);
        if !wants_volume && !wants_spread {
            return Ok(Vec::new());
        }

        let (daily_return, daily_pre_close) = if wants_spread {
            let panel = data.daily_panel(DatasetId::StockDailyPv)?;
            (
                Some(daily_return_map(panel)?),
                Some(panel_column_map(panel, &panel.column("pre_close")?)),
            )
        } else {
            (None, None)
        };

        let mut volume_values = Vec::new();
        let mut spread_values = Vec::new();
        for trade_date in &context.target_dates {
            let Some(table) = data.minute(DatasetId::StockMinute1m, *trade_date) else {
                continue;
            };
            let matrix = MinuteSynergyMatrix::from_table(table)?;
            let volume_synergy = if wants_volume {
                Some(volume_synergy_from_matrix(&matrix))
            } else {
                None
            };
            let synergy_spread = if wants_spread {
                Some(synergy_spread_from_matrix(
                    &matrix,
                    &values_for_codes(
                        *trade_date,
                        &matrix.codes,
                        daily_return
                            .as_ref()
                            .expect("daily return map is loaded for spread raw"),
                    ),
                    &values_for_codes(
                        *trade_date,
                        &matrix.codes,
                        daily_pre_close
                            .as_ref()
                            .expect("pre-close map is loaded for spread raw"),
                    ),
                ))
            } else {
                None
            };

            for (code_idx, ts_code) in matrix.codes.iter().enumerate() {
                if let Some(values) = volume_synergy.as_ref() {
                    volume_values.push(FactorValue {
                        key: FactorRowKey::Daily {
                            trade_date: *trade_date,
                            ts_code: ts_code.clone(),
                        },
                        value: values[code_idx],
                    });
                }
                if let Some(values) = synergy_spread.as_ref() {
                    spread_values.push(FactorValue {
                        key: FactorRowKey::Daily {
                            trade_date: *trade_date,
                            ts_code: ts_code.clone(),
                        },
                        value: values[code_idx],
                    });
                }
            }
        }

        let mut output = Vec::new();
        if wants_volume {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(VOLUME_SYNERGY_RAW_ID),
                values: volume_values,
            });
        }
        if wants_spread {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(SYNERGY_SPREAD_RAW_ID),
                values: spread_values,
            });
        }
        Ok(output)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(VOLUME_SYNERGY_RAW_ID)?;
        let volume_synergy = panel.column(VOLUME_SYNERGY_RAW_ID)?;
        let synergy_spread = panel.column(SYNERGY_SPREAD_RAW_ID)?;
        let volume_component = rolling_component(&volume_synergy)?;
        let spread_component = rolling_component(&synergy_spread)?;
        let factor = average_pair(
            &volume_component.cs(cs_zscore)?,
            &spread_component.cs(cs_zscore)?,
        )?;

        Ok(factor.to_factor_series(self.spec()))
    }
}

impl MinuteSynergyMatrix {
    fn from_table(table: &Table) -> Result<Self> {
        let ts_codes = table.required_utf8("ts_code")?;
        let trade_times = table.required_utf8("trade_time")?;
        let open = table.required_f64_cast("open")?;
        let high = table.required_f64_cast("high")?;
        let low = table.required_f64_cast("low")?;
        let close = table.required_f64_cast("close")?;
        let volume = table.required_f64_cast("vol")?;
        let amount = table.required_f64_cast("amount")?;

        let mut time_set = BTreeSet::new();
        let mut code_set = BTreeSet::new();
        for idx in 0..table.len {
            let Some(time) = trade_times[idx].as_deref() else {
                continue;
            };
            if !intraday_time_in_range(time, "09:31:00", "15:00:00") {
                continue;
            }
            if let Some(code) = ts_codes[idx].clone() {
                time_set.insert(time.to_string());
                code_set.insert(code);
            }
        }

        let times = time_set.into_iter().collect::<Vec<_>>();
        let codes = code_set.into_iter().collect::<Vec<_>>();
        let time_lookup = times
            .iter()
            .enumerate()
            .map(|(idx, value)| (value.clone(), idx))
            .collect::<HashMap<_, _>>();
        let code_lookup = codes
            .iter()
            .enumerate()
            .map(|(idx, value)| (value.clone(), idx))
            .collect::<HashMap<_, _>>();

        let shape_len = times.len() * codes.len();
        let mut open_values = vec![None; shape_len];
        let mut high_values = vec![None; shape_len];
        let mut low_values = vec![None; shape_len];
        let mut close_values = vec![None; shape_len];
        let mut volume_values = vec![None; shape_len];
        let mut amount_values = vec![None; shape_len];
        for idx in 0..table.len {
            let (Some(time), Some(code)) = (trade_times[idx].clone(), ts_codes[idx].clone()) else {
                continue;
            };
            let (Some(time_idx), Some(code_idx)) = (
                time_lookup.get(&time).copied(),
                code_lookup.get(&code).copied(),
            ) else {
                continue;
            };
            let offset = time_idx * codes.len() + code_idx;
            open_values[offset] = clean_intraday_value(open[idx]);
            high_values[offset] = clean_intraday_value(high[idx]);
            low_values[offset] = clean_intraday_value(low[idx]);
            close_values[offset] = clean_intraday_value(close[idx]);
            volume_values[offset] = clean_intraday_value(volume[idx]);
            amount_values[offset] = clean_intraday_value(amount[idx]);
        }

        Ok(Self {
            times,
            codes,
            open: open_values,
            high: high_values,
            low: low_values,
            close: close_values,
            volume: volume_values,
            amount: amount_values,
        })
    }
}

fn rolling_component(values: &PanelColumn) -> Result<PanelColumn> {
    let mean20 = values.ts(|series| ts_mean(series, WINDOW, 1))?;
    let std20 = values.ts(|series| ts_std_dev(series, WINDOW, 1))?;
    average_pair(&mean20.cs(cs_zscore)?, &std20.cs(cs_zscore)?)
}

fn volume_synergy_from_matrix(matrix: &MinuteSynergyMatrix) -> Vec<Option<f64>> {
    let time_count = matrix.time_count();
    let code_count = matrix.code_count();
    if time_count < OHLC_WINDOW || code_count < 2 {
        return vec![None; code_count];
    }

    let states = price_states(matrix);
    let volume_pct = volume_percentages(matrix);
    let mut own_series = vec![Vec::<Option<f64>>::new(); code_count];
    let mut synergy_series = vec![Vec::<Option<f64>>::new(); code_count];

    for time_idx in OHLC_WINDOW - 1..time_count {
        let mut state_sums = [0.0; 3];
        for code_idx in 0..code_count {
            let offset = matrix.offset(time_idx, code_idx);
            let (Some(state), Some(pct)) = (states[offset], volume_pct[offset]) else {
                continue;
            };
            state_sums[state_index(state)] += pct;
        }
        for code_idx in 0..code_count {
            let offset = matrix.offset(time_idx, code_idx);
            let (Some(state), Some(pct)) = (states[offset], volume_pct[offset]) else {
                continue;
            };
            own_series[code_idx].push(Some(pct));
            synergy_series[code_idx].push(Some(state_sums[state_index(state)] - pct));
        }
    }

    own_series
        .iter()
        .zip(&synergy_series)
        .map(|(own, synergy)| pearson_corr(own, synergy))
        .collect()
}

fn price_states(matrix: &MinuteSynergyMatrix) -> Vec<Option<i8>> {
    let time_count = matrix.time_count();
    let code_count = matrix.code_count();
    let mut output = vec![None; time_count * code_count];
    for time_idx in OHLC_WINDOW - 1..time_count {
        for code_idx in 0..code_count {
            let mut prices = Vec::with_capacity(OHLC_WINDOW * 4);
            let mut complete = true;
            for window_idx in (time_idx + 1 - OHLC_WINDOW)..=time_idx {
                for value in [
                    matrix.high[matrix.offset(window_idx, code_idx)],
                    matrix.open[matrix.offset(window_idx, code_idx)],
                    matrix.low[matrix.offset(window_idx, code_idx)],
                    matrix.close[matrix.offset(window_idx, code_idx)],
                ] {
                    match clean(value) {
                        Some(value) => prices.push(value),
                        None => {
                            complete = false;
                            break;
                        }
                    }
                }
                if !complete {
                    break;
                }
            }
            if !complete {
                continue;
            }
            let Some((mean, std)) = mean_std(prices.iter().copied()) else {
                continue;
            };
            let Some(close) = clean(matrix.close[matrix.offset(time_idx, code_idx)]) else {
                continue;
            };
            output[matrix.offset(time_idx, code_idx)] = Some(if close > mean + std {
                1
            } else if close < mean - std {
                -1
            } else {
                0
            });
        }
    }
    output
}

fn volume_percentages(matrix: &MinuteSynergyMatrix) -> Vec<Option<f64>> {
    let time_count = matrix.time_count();
    let code_count = matrix.code_count();
    let mut totals = vec![0.0; code_count];
    for (code_idx, total) in totals.iter_mut().enumerate() {
        for time_idx in 0..time_count {
            if let Some(value) = clean(matrix.volume[matrix.offset(time_idx, code_idx)]) {
                *total += value;
            }
        }
    }

    let mut output = vec![None; time_count * code_count];
    for time_idx in 0..time_count {
        for code_idx in 0..code_count {
            let total = totals[code_idx];
            if total.abs() <= f64::EPSILON {
                continue;
            }
            let Some(volume) = clean(matrix.volume[matrix.offset(time_idx, code_idx)]) else {
                continue;
            };
            output[matrix.offset(time_idx, code_idx)] = Some(volume / total);
        }
    }
    output
}

fn synergy_spread_from_matrix(
    matrix: &MinuteSynergyMatrix,
    daily_returns: &[Option<f64>],
    daily_pre_close: &[Option<f64>],
) -> Vec<Option<f64>> {
    let time_count = matrix.time_count();
    let code_count = matrix.code_count();
    if time_count <= DIFF_WINDOW || code_count < 2 {
        return vec![None; code_count];
    }
    let signals = synergy_signals(matrix, daily_pre_close);
    let bits = signal_bitsets(&signals, time_count - DIFF_WINDOW, code_count);
    (0..code_count)
        .into_par_iter()
        .map(|code_idx| {
            let self_return = clean(daily_returns[code_idx])?;
            let peers = top_synergy_peers(&bits, code_idx, code_count);
            let peer_mean = mean(
                peers
                    .into_iter()
                    .filter_map(|peer_idx| clean(daily_returns[peer_idx])),
            )?;
            Some(self_return - peer_mean)
        })
        .collect()
}

fn synergy_signals(
    matrix: &MinuteSynergyMatrix,
    daily_pre_close: &[Option<f64>],
) -> Vec<[Option<i8>; 3]> {
    let time_count = matrix.time_count();
    let code_count = matrix.code_count();
    let mut output = vec![[None; 3]; (time_count.saturating_sub(DIFF_WINDOW)) * code_count];
    if time_count <= DIFF_WINDOW {
        return output;
    }

    let intraday_return = intraday_returns(matrix);
    for time_idx in DIFF_WINDOW..time_count {
        let signal_time_idx = time_idx - DIFF_WINDOW;
        for code_idx in 0..code_count {
            let mut signals = [
                sign(intraday_return[matrix.offset(time_idx, code_idx)]),
                sign(diff_option(
                    intraday_return[matrix.offset(time_idx, code_idx)],
                    previous_window_mean(&intraday_return, matrix, time_idx, code_idx),
                )),
                sign(diff_option(
                    matrix.volume[matrix.offset(time_idx, code_idx)],
                    previous_window_mean(&matrix.volume, matrix, time_idx, code_idx),
                )),
            ];
            if all_zero(signals) {
                let fallback = sign(diff_option(
                    matrix.close[matrix.offset(time_idx, code_idx)],
                    previous_window_vwap(matrix, time_idx, code_idx),
                ));
                signals = [fallback; 3];
            }
            if all_zero(signals) {
                let fallback = sign(diff_option(
                    matrix.close[matrix.offset(time_idx, code_idx)],
                    daily_pre_close[code_idx],
                ));
                signals = [fallback; 3];
            }
            output[signal_time_idx * code_count + code_idx] = signals;
        }
    }
    output
}

fn previous_window_mean(
    values: &[Option<f64>],
    matrix: &MinuteSynergyMatrix,
    time_idx: usize,
    code_idx: usize,
) -> Option<f64> {
    if time_idx < DIFF_WINDOW {
        return None;
    }
    let mut sum = 0.0;
    for previous_idx in time_idx - DIFF_WINDOW..time_idx {
        sum += clean(values[matrix.offset(previous_idx, code_idx)])?;
    }
    Some(sum / DIFF_WINDOW as f64)
}

fn intraday_returns(matrix: &MinuteSynergyMatrix) -> Vec<Option<f64>> {
    let time_count = matrix.time_count();
    let code_count = matrix.code_count();
    let mut output = vec![None; time_count * code_count];
    for time_idx in 0..time_count {
        for code_idx in 0..code_count {
            output[matrix.offset(time_idx, code_idx)] = ret(
                matrix.close[matrix.offset(time_idx, code_idx)],
                matrix.open[matrix.offset(time_idx, code_idx)],
            );
        }
    }
    output
}

fn previous_window_vwap(
    matrix: &MinuteSynergyMatrix,
    time_idx: usize,
    code_idx: usize,
) -> Option<f64> {
    if time_idx < DIFF_WINDOW {
        return None;
    }
    let mut amount_sum = 0.0;
    let mut volume_sum = 0.0;
    for previous_idx in time_idx - DIFF_WINDOW..time_idx {
        amount_sum += clean(matrix.amount[matrix.offset(previous_idx, code_idx)])?;
        volume_sum += clean(matrix.volume[matrix.offset(previous_idx, code_idx)])?;
    }
    minute_vwap_from_amount_vol(Some(amount_sum), Some(volume_sum))
}

#[derive(Clone, Debug)]
struct SignalBitsets {
    word_count: usize,
    bits: Vec<Vec<Vec<u64>>>,
}

fn signal_bitsets(
    signals: &[[Option<i8>; 3]],
    signal_time_count: usize,
    code_count: usize,
) -> SignalBitsets {
    let word_count = signal_time_count.div_ceil(64);
    let mut bits = vec![vec![vec![0u64; code_count * word_count]; 3]; 3];
    for time_idx in 0..signal_time_count {
        let word_idx = time_idx / 64;
        let bit = 1u64 << (time_idx % 64);
        for code_idx in 0..code_count {
            let signals = signals[time_idx * code_count + code_idx];
            for metric_idx in 0..3 {
                let Some(signal) = signals[metric_idx] else {
                    continue;
                };
                let state_idx = state_index(signal);
                bits[metric_idx][state_idx][code_idx * word_count + word_idx] |= bit;
            }
        }
    }
    SignalBitsets { word_count, bits }
}

fn top_synergy_peers(bits: &SignalBitsets, target_idx: usize, code_count: usize) -> Vec<usize> {
    let mut counts = Vec::with_capacity(code_count.saturating_sub(1));
    for peer_idx in 0..code_count {
        if peer_idx == target_idx {
            continue;
        }
        counts.push((peer_idx, pair_synergy_count(bits, target_idx, peer_idx)));
    }
    if counts.is_empty() {
        return Vec::new();
    }
    let keep = TOP_PEER_COUNT.min(counts.len());
    if counts.len() > keep {
        counts.select_nth_unstable_by(keep, |left, right| compare_peer_count(*left, *right));
        counts.truncate(keep);
    }
    counts.sort_by(|left, right| compare_peer_count(*left, *right));
    counts.into_iter().map(|(idx, _)| idx).collect()
}

fn pair_synergy_count(bits: &SignalBitsets, left_idx: usize, right_idx: usize) -> u32 {
    let mut output = 0u32;
    for metric_idx in 0..3 {
        for state_idx in 0..3 {
            let metric_state = &bits.bits[metric_idx][state_idx];
            let left_offset = left_idx * bits.word_count;
            let right_offset = right_idx * bits.word_count;
            for word_idx in 0..bits.word_count {
                output += (metric_state[left_offset + word_idx]
                    & metric_state[right_offset + word_idx])
                    .count_ones();
            }
        }
    }
    output
}

fn compare_peer_count(left: (usize, u32), right: (usize, u32)) -> std::cmp::Ordering {
    right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0))
}

fn daily_return_map(data: &DailyPanel) -> Result<HashMap<(i32, String), Option<f64>>> {
    let returns = data
        .column("close")?
        .zip_binary(&data.column("pre_close")?, ret)?;
    Ok(panel_column_map(data, &returns))
}

fn panel_column_map(
    panel: &DailyPanel,
    column: &PanelColumn,
) -> HashMap<(i32, String), Option<f64>> {
    let mut output = HashMap::new();
    let code_count = panel.instruments().len();
    for (date_idx, trade_date) in panel.dates().iter().enumerate() {
        for (code_idx, ts_code) in panel.instruments().iter().enumerate() {
            output.insert(
                (*trade_date, ts_code.clone()),
                column.values()[date_idx * code_count + code_idx],
            );
        }
    }
    output
}

fn values_for_codes(
    trade_date: i32,
    codes: &[String],
    values: &HashMap<(i32, String), Option<f64>>,
) -> Vec<Option<f64>> {
    codes
        .iter()
        .map(|code| values.get(&(trade_date, code.clone())).copied().flatten())
        .collect()
}

fn average_pair(left: &PanelColumn, right: &PanelColumn) -> Result<PanelColumn> {
    left.zip_binary(right, |left, right| match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some((left + right) / 2.0),
        _ => None,
    })
}

fn pearson_corr(left: &[Option<f64>], right: &[Option<f64>]) -> Option<f64> {
    let pairs = left
        .iter()
        .zip(right)
        .filter_map(|(left, right)| Some((clean(*left)?, clean(*right)?)))
        .collect::<Vec<_>>();
    if pairs.is_empty() {
        return None;
    }
    let mean_left = pairs.iter().map(|(left, _)| *left).sum::<f64>() / pairs.len() as f64;
    let mean_right = pairs.iter().map(|(_, right)| *right).sum::<f64>() / pairs.len() as f64;
    let cov = pairs
        .iter()
        .map(|(left, right)| (left - mean_left) * (right - mean_right))
        .sum::<f64>()
        / pairs.len() as f64;
    let std_left = (pairs
        .iter()
        .map(|(left, _)| (left - mean_left).powi(2))
        .sum::<f64>()
        / pairs.len() as f64)
        .sqrt();
    let std_right = (pairs
        .iter()
        .map(|(_, right)| (right - mean_right).powi(2))
        .sum::<f64>()
        / pairs.len() as f64)
        .sqrt();
    if std_left <= f64::EPSILON || std_right <= f64::EPSILON {
        return None;
    }
    Some(cov / (std_left * std_right))
}

fn ret(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (clean(numerator), clean(denominator)) {
        (Some(numerator), Some(denominator)) if denominator.abs() > f64::EPSILON => {
            Some(numerator / denominator - 1.0)
        }
        _ => None,
    }
}

fn diff_option(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    Some(clean(left)? - clean(right)?)
}

fn sign(value: Option<f64>) -> Option<i8> {
    let value = clean(value)?;
    Some(if value > 0.0 {
        1
    } else if value < 0.0 {
        -1
    } else {
        0
    })
}

fn all_zero(values: [Option<i8>; 3]) -> bool {
    values.iter().all(|value| matches!(value, Some(0)))
}

fn state_index(state: i8) -> usize {
    match state {
        -1 => 0,
        0 => 1,
        1 => 2,
        _ => 1,
    }
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| !value.is_nan())
}

fn mean(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let values = values
        .into_iter()
        .filter(|value| !value.is_nan())
        .collect::<Vec<_>>();
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn mean_std(values: impl IntoIterator<Item = f64>) -> Option<(f64, f64)> {
    let values = values
        .into_iter()
        .filter(|value| !value.is_nan())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    Some((mean, variance.sqrt()))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::core::{AssetClass, FactorContext};
    use crate::data::{ColumnData, Table};
    use crate::factor::common::DailyPanel;

    use super::*;

    fn time(value: &str) -> Option<String> {
        Some(value.to_string())
    }

    #[test]
    fn minute_matrix_filters_0931_before_rolling_inputs() {
        let table = Table::new(BTreeMap::from([
            (
                "trade_date".to_string(),
                ColumnData::I32(vec![Some(20260101); 12]),
            ),
            (
                "ts_code".to_string(),
                ColumnData::Utf8(vec![Some("a".to_string()); 12]),
            ),
            (
                "trade_time".to_string(),
                ColumnData::Utf8(vec![
                    time("09:30:00"),
                    time("09:31:00"),
                    time("09:32:00"),
                    time("09:33:00"),
                    time("09:34:00"),
                    time("09:35:00"),
                    time("15:00:00"),
                    time("15:01:00"),
                    time("09:30:00"),
                    time("09:31:00"),
                    time("15:00:00"),
                    time("15:01:00"),
                ]),
            ),
            ("open".to_string(), ColumnData::F64(vec![Some(1.0); 12])),
            ("high".to_string(), ColumnData::F64(vec![Some(1.0); 12])),
            ("low".to_string(), ColumnData::F64(vec![Some(1.0); 12])),
            ("close".to_string(), ColumnData::F64(vec![Some(1.0); 12])),
            ("vol".to_string(), ColumnData::F64(vec![Some(1.0); 12])),
            ("amount".to_string(), ColumnData::F64(vec![Some(1.0); 12])),
        ]))
        .expect("table");
        let matrix = MinuteSynergyMatrix::from_table(&table).expect("matrix");

        assert_eq!(matrix.times.first().map(String::as_str), Some("09:31:00"));
        assert_eq!(matrix.times.last().map(String::as_str), Some("15:00:00"));
        assert!(!matrix.times.iter().any(|time| time == "09:30:00"));
        assert!(!matrix.times.iter().any(|time| time == "15:01:00"));
    }

    #[test]
    fn price_state_requires_complete_twenty_prices() {
        let matrix = MinuteSynergyMatrix {
            times: (0..5).map(|idx| format!("09:3{idx}:00")).collect(),
            codes: vec!["a".to_string(), "b".to_string()],
            open: vec![Some(10.0); 10],
            high: vec![Some(10.0); 10],
            low: vec![Some(10.0); 10],
            close: vec![
                Some(10.0),
                Some(10.0),
                Some(10.0),
                Some(10.0),
                Some(10.0),
                Some(10.0),
                Some(10.0),
                Some(10.0),
                Some(12.0),
                Some(12.0),
            ],
            volume: vec![Some(1.0); 10],
            amount: vec![Some(1.0); 10],
        };
        let states = price_states(&matrix);
        assert_eq!(states[matrix.offset(4, 0)], Some(1));

        let mut missing = matrix.clone();
        let missing_offset = missing.offset(0, 0);
        missing.open[missing_offset] = None;
        let states = price_states(&missing);
        assert_eq!(states[missing.offset(4, 0)], None);
    }

    #[test]
    fn volume_synergy_subtracts_self_and_correlates() {
        let matrix = MinuteSynergyMatrix {
            times: (0..6).map(|idx| format!("09:3{idx}:00")).collect(),
            codes: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            open: vec![Some(10.0); 18],
            high: vec![Some(10.0); 18],
            low: vec![Some(10.0); 18],
            close: vec![
                Some(10.0),
                Some(10.0),
                Some(10.0),
                Some(10.0),
                Some(10.0),
                Some(10.0),
                Some(10.0),
                Some(10.0),
                Some(10.0),
                Some(10.0),
                Some(10.0),
                Some(10.0),
                Some(12.0),
                Some(12.0),
                Some(8.0),
                Some(11.0),
                Some(13.0),
                Some(7.0),
            ],
            volume: vec![
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(2.0),
                Some(3.0),
                Some(5.0),
                Some(4.0),
                Some(6.0),
                Some(7.0),
            ],
            amount: vec![Some(1.0); 18],
        };
        let output = volume_synergy_from_matrix(&matrix);
        assert!(output[0].is_some());
        assert!(output[1].is_some());
        assert!(output[2].is_none());
    }

    #[test]
    fn synergy_signals_compare_current_values_with_previous_five_minute_means() {
        let matrix = MinuteSynergyMatrix {
            times: (0..6).map(|idx| format!("09:3{}:00", idx + 1)).collect(),
            codes: vec!["a".to_string()],
            open: vec![Some(100.0); 6],
            high: vec![Some(100.0); 6],
            low: vec![Some(100.0); 6],
            close: vec![
                Some(150.0),
                Some(150.0),
                Some(150.0),
                Some(150.0),
                Some(50.0),
                Some(140.0),
            ],
            volume: vec![
                Some(100.0),
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(50.0),
            ],
            amount: vec![Some(1.0); 6],
        };
        let signals = synergy_signals(&matrix, &[Some(100.0)]);

        assert_eq!(signals[0][0], Some(1));
        assert_eq!(
            signals[0][1],
            Some(1),
            "0.4 is above the previous-five-minute mean return 0.3, but below t-5 return 0.5"
        );
        assert_eq!(
            signals[0][2],
            Some(1),
            "50 is above the previous-five-minute mean volume 20.8, but below t-5 volume 100"
        );
    }

    #[test]
    fn synergy_signals_apply_two_robust_fallbacks_with_previous_window_vwap() {
        let matrix = MinuteSynergyMatrix {
            times: (0..6).map(|idx| format!("09:3{}:00", idx + 1)).collect(),
            codes: vec!["a".to_string()],
            open: vec![Some(10.0); 6],
            high: vec![Some(10.0); 6],
            low: vec![Some(10.0); 6],
            close: vec![Some(10.0); 6],
            volume: vec![Some(10.0); 6],
            amount: vec![
                Some(50.0),
                Some(50.0),
                Some(50.0),
                Some(50.0),
                Some(50.0),
                Some(500.0),
            ],
        };
        let signals = synergy_signals(&matrix, &[Some(9.0)]);
        assert_eq!(signals[0], [Some(1), Some(1), Some(1)]);

        let mut fallback_preclose = matrix.clone();
        for idx in 0..5 {
            fallback_preclose.amount[idx] = Some(100.0);
        }
        fallback_preclose.close[5] = Some(10.0);
        let signals = synergy_signals(&fallback_preclose, &[Some(11.0)]);
        assert_eq!(signals[0], [Some(-1), Some(-1), Some(-1)]);
    }

    #[test]
    fn previous_window_vwap_requires_complete_previous_five_minutes() {
        let matrix = MinuteSynergyMatrix {
            times: (0..6).map(|idx| format!("09:3{}:00", idx + 1)).collect(),
            codes: vec!["a".to_string()],
            open: vec![Some(10.0); 6],
            high: vec![Some(10.0); 6],
            low: vec![Some(10.0); 6],
            close: vec![Some(10.0); 6],
            volume: vec![
                Some(10.0),
                Some(10.0),
                Some(10.0),
                None,
                Some(10.0),
                Some(10.0),
            ],
            amount: vec![Some(100.0); 6],
        };

        assert_eq!(previous_window_vwap(&matrix, 5, 0), None);
    }

    #[test]
    fn bitset_top_peers_match_direct_counts() {
        let signals = vec![
            [Some(1), Some(0), Some(-1)],
            [Some(1), Some(1), Some(-1)],
            [Some(-1), Some(0), Some(-1)],
            [Some(1), Some(0), Some(1)],
            [Some(-1), Some(1), Some(1)],
            [Some(-1), Some(0), Some(-1)],
        ];
        let bits = signal_bitsets(&signals, 2, 3);
        assert_eq!(pair_synergy_count(&bits, 0, 1), 3);
        assert_eq!(pair_synergy_count(&bits, 0, 2), 3);
        assert_eq!(top_synergy_peers(&bits, 0, 3), vec![1, 2]);
    }

    #[test]
    fn final_component_averages_two_rolling_branches() {
        let mut trade_dates = Vec::new();
        let mut ts_codes = Vec::new();
        let mut raw = Vec::new();
        for idx in 0..20 {
            trade_dates.push(Some(20260101 + idx));
            ts_codes.push(Some("a".to_string()));
            raw.push(Some(idx as f64));
            trade_dates.push(Some(20260101 + idx));
            ts_codes.push(Some("b".to_string()));
            raw.push(Some((idx * 2) as f64));
            trade_dates.push(Some(20260101 + idx));
            ts_codes.push(Some("c".to_string()));
            raw.push(Some((20 - idx) as f64));
        }
        let table = Table::new(BTreeMap::from([
            ("trade_date".to_string(), ColumnData::I32(trade_dates)),
            ("ts_code".to_string(), ColumnData::Utf8(ts_codes)),
            ("raw".to_string(), ColumnData::F64(raw)),
        ]))
        .expect("table");
        let context = FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: 20260120,
            end_date: 20260120,
            load_start_date: 20260101,
            load_dates: (0..20).map(|idx| 20260101 + idx).collect(),
            target_dates: vec![20260120],
        };
        let panel = DailyPanel::from_table(&table, &context).expect("panel");
        let component = rolling_component(&panel.column("raw").expect("raw")).expect("component");

        assert_eq!(component.values().len(), 60);
        assert!(component.values().iter().any(Option::is_some));
    }
}
