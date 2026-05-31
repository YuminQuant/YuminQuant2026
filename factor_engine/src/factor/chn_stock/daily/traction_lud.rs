use std::any::Any;
use std::collections::{BTreeMap, HashMap, VecDeque};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawAuxiliaryRequest, IntradayDailyRawRequest,
    IntradayDailyRawSeries, IntradayDailyRawSpec, Lookback,
};
use crate::data::{DataPool, Table};
use crate::error::{err, Result};
use crate::factor::common::stock_daily_ops::{
    adjusted_20d_return, is_bj_stock, neutralize_ret20_size_sector,
};
use crate::factor::common::vector::clean;
use crate::factor::common::{clean_intraday_value, stock_minute_raw_spec, DailyPanel};
use crate::factor::{Factor, IntradayRawMaterializeMode};

const VERSION: &str = "0.1.0";
const RAW_VERSION: &str = "0.1.0";
const RAW_ID: &str = "daily_kyzq_traction_lud_expave_raw";
const PROVIDER_KEY: &str = "kyzq_traction_lud_provider";

const WINDOW: usize = 20;
const RET20_WINDOW: usize = 20;
const MORNING_MINUTES: usize = 90;
const EDGE_PRUNE_FRACTION: f64 = 0.50;
const EPS: f64 = 1e-12;

pub struct StockDailyTractionLud;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyTractionLud)
}

impl Factor for StockDailyTractionLud {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "traction_lud".to_string(),
            aliases: vec!["Traction_LUD".to_string(), "ExpAve_LUD".to_string()],
            name: "traction_lud".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "KYZQ limit-up/down spillover directed cross-sectional network traction factor. It builds a 20-day first-90-minute directed limit spillover network, prunes the weakest 50% edges, computes incoming association-weighted peer Ret20 ExpAve, and neutralizes by Ret20, Barra SIZE, and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(RAW_ID, WINDOW - 1)],
            lookback: Lookback {
                trading_days: WINDOW - 1,
            },
        }
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        vec![raw_spec()]
    }

    fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
        PROVIDER_KEY.to_string()
    }

    fn intraday_raw_materialize_mode(&self, _raw_ids: &[String]) -> IntradayRawMaterializeMode {
        IntradayRawMaterializeMode::Stateful
    }

    fn initial_intraday_raw_state(&self, _raw_ids: &[String]) -> Box<dyn Any + Send> {
        Box::new(TractionLudState::default())
    }

    fn intraday_raw_auxiliary_requirements(
        &self,
        raw_ids: &[String],
    ) -> Vec<IntradayDailyRawAuxiliaryRequest> {
        if !raw_ids.iter().any(|raw_id| raw_id == RAW_ID) {
            return Vec::new();
        }
        vec![
            IntradayDailyRawAuxiliaryRequest::new(
                DataRequest::new(DatasetId::StockDailyLimit, &["up_limit", "down_limit"]),
                0,
            ),
            IntradayDailyRawAuxiliaryRequest::new(
                DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                RET20_WINDOW - 1,
            ),
            IntradayDailyRawAuxiliaryRequest::new(
                DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
                RET20_WINDOW - 1,
            ),
        ]
    }

    fn minute_compute_stateful_many(
        &self,
        raw_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
        state: &mut dyn Any,
    ) -> Result<Vec<IntradayDailyRawSeries>> {
        if !raw_ids.iter().any(|raw_id| raw_id == RAW_ID) {
            return Ok(Vec::new());
        }
        let state = state
            .downcast_mut::<TractionLudState>()
            .ok_or_else(|| err("traction_lud raw received incompatible state"))?;
        let trade_date = *context
            .target_dates
            .first()
            .ok_or_else(|| err("traction_lud raw requires one target date"))?;
        let limit_map = current_limit_map(data, trade_date)?;
        let ret20 = current_ret20_map(data, trade_date)?;
        let day = match data.minute(DatasetId::StockMinute1m, trade_date) {
            Some(table) => state.day_from_table(table, &limit_map)?,
            None => DailyLudDay::default(),
        };
        let current_instruments = day.current_instruments.clone();
        state.push_day(day);
        let expave = state.expave(&ret20);

        let values = current_instruments
            .into_iter()
            .map(|instrument_idx| FactorValue {
                key: FactorRowKey::Daily {
                    trade_date,
                    ts_code: state.instruments[instrument_idx].clone(),
                },
                value: expave
                    .get(instrument_idx)
                    .copied()
                    .flatten()
                    .filter(|value| value.is_finite()),
            })
            .collect();

        Ok(vec![IntradayDailyRawSeries {
            spec: raw_spec(),
            values,
        }])
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(RAW_ID)?;
        let expave = panel.column(RAW_ID)?;
        let factor = neutralize_ret20_size_sector(&expave, panel, data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn raw_spec() -> IntradayDailyRawSpec {
    stock_minute_raw_spec(RAW_ID, RAW_VERSION, &["close"], WINDOW)
}

fn tags() -> Vec<String> {
    [
        "KYZQ",
        "cs_network",
        "limit_up_down",
        "spillover",
        "network",
        "intraday",
        "minute_agg",
        "ret20",
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

#[derive(Default)]
struct TractionLudState {
    instruments: Vec<String>,
    index: BTreeMap<String, usize>,
    days: VecDeque<DailyLudDay>,
    total_counts: HashMap<u64, PairCounts>,
}

impl TractionLudState {
    fn instrument_idx(&mut self, ts_code: &str) -> usize {
        if let Some(idx) = self.index.get(ts_code) {
            return *idx;
        }
        let idx = self.instruments.len();
        self.instruments.push(ts_code.to_string());
        self.index.insert(ts_code.to_string(), idx);
        idx
    }

    fn day_from_table(
        &mut self,
        table: &Table,
        limit_map: &HashMap<String, LimitBand>,
    ) -> Result<DailyLudDay> {
        let ts_codes = table.required_utf8("ts_code")?;
        let trade_times = table.required_utf8("trade_time")?;
        let close = table.required_f64_cast("close")?;
        let mut stock_days = BTreeMap::<String, MorningStockDay>::new();

        for row_idx in 0..table.len {
            let Some(ts_code) = ts_codes[row_idx].as_deref() else {
                continue;
            };
            if is_bj_stock(ts_code) {
                continue;
            }
            let Some(slot) = trade_times[row_idx].as_deref().and_then(morning_slot) else {
                continue;
            };
            let Some(close_value) =
                clean_intraday_value(close[row_idx]).filter(|value| *value > 0.0)
            else {
                continue;
            };
            stock_days.entry(ts_code.to_string()).or_default().close[slot] = Some(close_value);
        }

        let mut signals = Vec::new();
        let mut current_instruments = Vec::new();
        for (ts_code, day) in stock_days {
            let instrument_idx = self.instrument_idx(&ts_code);
            current_instruments.push(instrument_idx);
            if let Some(limit) = limit_map.get(&ts_code) {
                signals.push(day.to_signals(instrument_idx, *limit));
            }
        }
        current_instruments.sort_unstable();
        let counts = daily_pair_counts(&signals);
        Ok(DailyLudDay {
            current_instruments,
            counts,
        })
    }

    fn push_day(&mut self, day: DailyLudDay) {
        add_counts(&mut self.total_counts, &day.counts);
        self.days.push_back(day);
        if self.days.len() > WINDOW {
            if let Some(old_day) = self.days.pop_front() {
                subtract_counts(&mut self.total_counts, &old_day.counts);
            }
        }
    }

    fn expave(&self, ret20_by_code: &HashMap<String, Option<f64>>) -> Vec<Option<f64>> {
        if self.days.len() < WINDOW {
            return vec![None; self.instruments.len()];
        }
        let mut ret20 = vec![None; self.instruments.len()];
        for (idx, ts_code) in self.instruments.iter().enumerate() {
            ret20[idx] = ret20_by_code.get(ts_code).copied().flatten();
        }
        directed_expave(&self.total_counts, &ret20, self.instruments.len())
    }
}

#[derive(Clone, Debug, Default)]
struct DailyLudDay {
    current_instruments: Vec<usize>,
    counts: HashMap<u64, PairCounts>,
}

#[derive(Clone, Debug)]
struct MorningStockDay {
    close: [Option<f64>; MORNING_MINUTES],
}

impl Default for MorningStockDay {
    fn default() -> Self {
        Self {
            close: [None; MORNING_MINUTES],
        }
    }
}

impl MorningStockDay {
    fn to_signals(&self, instrument_idx: usize, limit: LimitBand) -> StockSignals {
        let mut limit_direction = [0i8; MORNING_MINUTES];
        let mut direction = [0i8; MORNING_MINUTES];
        for slot in 0..MORNING_MINUTES {
            limit_direction[slot] = limit_status(self.close[slot], limit);
            if slot > 0 {
                direction[slot] = minute_direction(self.close[slot - 1], self.close[slot]);
            }
        }
        StockSignals {
            instrument_idx,
            limit_direction,
            direction,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StockSignals {
    instrument_idx: usize,
    limit_direction: [i8; MORNING_MINUTES],
    direction: [i8; MORNING_MINUTES],
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct PairCounts {
    same: u16,
    valid: u16,
}

#[derive(Clone, Copy, Debug)]
struct LimitBand {
    up: f64,
    down: f64,
}

fn daily_pair_counts(signals: &[StockSignals]) -> HashMap<u64, PairCounts> {
    let mut counts = HashMap::<u64, PairCounts>::new();
    for slot in 1..MORNING_MINUTES {
        let sources = signals
            .iter()
            .filter_map(|signal| {
                let direction = signal.limit_direction[slot - 1];
                (direction != 0).then_some((signal.instrument_idx, direction))
            })
            .collect::<Vec<_>>();
        if sources.is_empty() {
            continue;
        }
        let followers = signals
            .iter()
            .filter_map(|signal| {
                let direction = signal.direction[slot];
                (direction != 0).then_some((signal.instrument_idx, direction))
            })
            .collect::<Vec<_>>();
        if followers.is_empty() {
            continue;
        }
        for (source_idx, source_direction) in &sources {
            for (target_idx, target_direction) in &followers {
                if source_idx == target_idx {
                    continue;
                }
                let entry = counts
                    .entry(directed_key(*source_idx, *target_idx))
                    .or_default();
                entry.valid = entry.valid.saturating_add(1);
                if source_direction == target_direction {
                    entry.same = entry.same.saturating_add(1);
                }
            }
        }
    }
    counts
}

fn add_counts(total: &mut HashMap<u64, PairCounts>, day: &HashMap<u64, PairCounts>) {
    for (key, counts) in day {
        let entry = total.entry(*key).or_default();
        entry.same = entry.same.saturating_add(counts.same);
        entry.valid = entry.valid.saturating_add(counts.valid);
    }
}

fn subtract_counts(total: &mut HashMap<u64, PairCounts>, day: &HashMap<u64, PairCounts>) {
    let mut empty_keys = Vec::new();
    for (key, counts) in day {
        if let Some(entry) = total.get_mut(key) {
            entry.same = entry.same.saturating_sub(counts.same);
            entry.valid = entry.valid.saturating_sub(counts.valid);
            if entry.valid == 0 {
                empty_keys.push(*key);
            }
        }
    }
    for key in empty_keys {
        total.remove(&key);
    }
}

fn directed_expave(
    counts: &HashMap<u64, PairCounts>,
    ret20: &[Option<f64>],
    instrument_count: usize,
) -> Vec<Option<f64>> {
    let mut weights = counts.values().filter_map(edge_weight).collect::<Vec<_>>();
    let Some(threshold) = edge_prune_threshold(&mut weights) else {
        return vec![None; instrument_count];
    };

    let mut numerator = vec![0.0; instrument_count];
    let mut denominator = vec![0.0; instrument_count];
    for (key, pair_counts) in counts {
        let Some(weight) = edge_weight(pair_counts) else {
            continue;
        };
        if weight < threshold || weight <= EPS {
            continue;
        }
        let (source_idx, target_idx) = decode_directed_key(*key);
        if source_idx >= instrument_count || target_idx >= instrument_count {
            continue;
        }
        if let Some(source_return) = clean(ret20[source_idx]) {
            numerator[target_idx] += weight * source_return;
            denominator[target_idx] += weight;
        }
    }

    numerator
        .into_iter()
        .zip(denominator)
        .map(|(num, den)| {
            if den > EPS {
                let value = num / den;
                value.is_finite().then_some(value)
            } else {
                None
            }
        })
        .collect()
}

fn edge_weight(counts: &PairCounts) -> Option<f64> {
    if counts.valid == 0 {
        return None;
    }
    Some(counts.same as f64 / counts.valid as f64)
}

fn edge_prune_threshold(weights: &mut [f64]) -> Option<f64> {
    if weights.is_empty() {
        return None;
    }
    let prune_count = ((weights.len() as f64) * EDGE_PRUNE_FRACTION).floor() as usize;
    if prune_count == 0 {
        return Some(f64::NEG_INFINITY);
    }
    let threshold_idx = prune_count.min(weights.len() - 1);
    let (_, threshold, _) =
        weights.select_nth_unstable_by(threshold_idx, |left, right| left.total_cmp(right));
    Some(*threshold)
}

fn directed_key(source_idx: usize, target_idx: usize) -> u64 {
    ((source_idx as u64) << 32) | target_idx as u64
}

fn decode_directed_key(key: u64) -> (usize, usize) {
    ((key >> 32) as usize, (key & 0xffff_ffff) as usize)
}

fn limit_status(close: Option<f64>, limit: LimitBand) -> i8 {
    let Some(close) = clean_intraday_value(close).filter(|value| *value > 0.0) else {
        return 0;
    };
    let close = round_price(close);
    if close >= round_price(limit.up) {
        1
    } else if close <= round_price(limit.down) {
        -1
    } else {
        0
    }
}

fn minute_direction(previous: Option<f64>, current: Option<f64>) -> i8 {
    let (Some(previous), Some(current)) = (
        clean_intraday_value(previous).filter(|value| *value > 0.0),
        clean_intraday_value(current).filter(|value| *value > 0.0),
    ) else {
        return 0;
    };
    if current > previous {
        1
    } else if current < previous {
        -1
    } else {
        0
    }
}

fn round_price(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

fn morning_slot(value: &str) -> Option<usize> {
    let minute = time_to_minutes(value)?;
    let start = 9 * 60 + 31;
    let end = 11 * 60;
    ((start..=end).contains(&minute)).then_some((minute - start) as usize)
}

fn time_to_minutes(value: &str) -> Option<i32> {
    let value = value.trim();
    let value = value
        .rsplit_once(' ')
        .map(|(_, right)| right)
        .or_else(|| value.rsplit_once('T').map(|(_, right)| right))
        .unwrap_or(value)
        .trim();
    if value.len() < 5 {
        return None;
    }
    let hour = value.get(0..2)?.parse::<i32>().ok()?;
    let minute = value.get(3..5)?.parse::<i32>().ok()?;
    Some(hour * 60 + minute)
}

fn current_limit_map(data: &DataPool, trade_date: i32) -> Result<HashMap<String, LimitBand>> {
    let panel = data.daily_panel(DatasetId::StockDailyLimit)?;
    let up_limit = panel.column("up_limit")?;
    let down_limit = panel.column("down_limit")?;
    let Some(date_idx) = panel.dates().iter().position(|date| *date == trade_date) else {
        return Ok(HashMap::new());
    };
    let code_count = panel.instruments().len();
    let offset = date_idx * code_count;
    let mut output = HashMap::new();
    for (code_idx, ts_code) in panel.instruments().iter().enumerate() {
        if is_bj_stock(ts_code) {
            continue;
        }
        let (Some(up), Some(down)) = (
            clean(up_limit.values()[offset + code_idx]),
            clean(down_limit.values()[offset + code_idx]),
        ) else {
            continue;
        };
        if up > 0.0 && down > 0.0 {
            output.insert(ts_code.clone(), LimitBand { up, down });
        }
    }
    Ok(output)
}

fn current_ret20_map(data: &DataPool, trade_date: i32) -> Result<HashMap<String, Option<f64>>> {
    let panel = data.daily_panel(DatasetId::StockDailyPv)?;
    let ret20 = adjusted_20d_return(data, &panel)?;
    current_panel_column_map(&panel, &ret20, trade_date)
}

fn current_panel_column_map(
    panel: &DailyPanel,
    values: &crate::factor::common::PanelColumn,
    trade_date: i32,
) -> Result<HashMap<String, Option<f64>>> {
    let Some(date_idx) = panel.dates().iter().position(|date| *date == trade_date) else {
        return Ok(HashMap::new());
    };
    let code_count = panel.instruments().len();
    let offset = date_idx * code_count;
    Ok(panel
        .instruments()
        .iter()
        .enumerate()
        .map(|(code_idx, ts_code)| (ts_code.clone(), values.values()[offset + code_idx]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair_counts(same: u16, valid: u16) -> PairCounts {
        PairCounts { same, valid }
    }

    fn empty_signals(instrument_idx: usize) -> StockSignals {
        StockSignals {
            instrument_idx,
            limit_direction: [0; MORNING_MINUTES],
            direction: [0; MORNING_MINUTES],
        }
    }

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("value");
        assert!(
            (actual - expected).abs() < 1e-12,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn traction_lud_morning_slot_keeps_opening_ninety_minutes() {
        assert_eq!(morning_slot("09:31:00"), Some(0));
        assert_eq!(morning_slot("10:00:00"), Some(29));
        assert_eq!(morning_slot("11:00:00"), Some(89));
        assert_eq!(morning_slot("11:01:00"), None);
        assert_eq!(morning_slot("13:01:00"), None);
    }

    #[test]
    fn traction_lud_limit_status_rounds_minute_close_and_limits() {
        let limit = LimitBand {
            up: 10.0,
            down: 8.0,
        };
        assert_eq!(limit_status(Some(10.004), limit), 1);
        assert_eq!(limit_status(Some(7.996), limit), -1);
        assert_eq!(limit_status(Some(9.0), limit), 0);
    }

    #[test]
    fn traction_lud_minute_direction_uses_close_to_close() {
        assert_eq!(minute_direction(Some(10.0), Some(10.1)), 1);
        assert_eq!(minute_direction(Some(10.0), Some(9.9)), -1);
        assert_eq!(minute_direction(Some(10.0), Some(10.0)), 0);
        assert_eq!(minute_direction(None, Some(10.0)), 0);
    }

    #[test]
    fn traction_lud_daily_pair_counts_are_directed_and_lagged() {
        let mut source = empty_signals(0);
        let mut target = empty_signals(1);
        source.limit_direction[0] = 1;
        target.direction[1] = 1;

        let counts = daily_pair_counts(&[source, target]);

        assert_eq!(counts.get(&directed_key(0, 1)), Some(&pair_counts(1, 1)));
        assert_eq!(counts.get(&directed_key(1, 0)), None);
    }

    #[test]
    fn traction_lud_daily_pair_counts_match_limit_down_to_down_direction() {
        let mut source = empty_signals(0);
        let mut target = empty_signals(1);
        source.limit_direction[3] = -1;
        target.direction[4] = -1;

        let counts = daily_pair_counts(&[source, target]);

        assert_eq!(counts.get(&directed_key(0, 1)), Some(&pair_counts(1, 1)));
    }

    #[test]
    fn traction_lud_edge_prune_threshold_drops_lowest_half_rank() {
        let mut weights = vec![0.9, 0.1, 0.7, 0.2, 0.5, 0.4, 0.8, 0.3, 0.6, 1.0];
        assert_close(edge_prune_threshold(&mut weights), 0.6);
    }

    #[test]
    fn traction_lud_directed_expave_uses_incoming_edges_and_prunes_weak_edges() {
        let mut counts_map = HashMap::new();
        counts_map.insert(directed_key(0, 1), pair_counts(10, 10));
        counts_map.insert(directed_key(2, 1), pair_counts(8, 10));
        counts_map.insert(directed_key(3, 1), pair_counts(1, 10));
        counts_map.insert(directed_key(1, 0), pair_counts(4, 10));
        let ret20 = vec![Some(0.10), Some(0.20), Some(0.30), Some(0.40)];

        let expave = directed_expave(&counts_map, &ret20, 4);

        assert_close(expave[1], (1.0 * 0.10 + 0.8 * 0.30) / 1.8);
        assert_eq!(expave[0], None);
        assert_eq!(expave[2], None);
        assert_eq!(expave[3], None);
    }

    #[test]
    fn traction_lud_state_rolls_twenty_days() {
        let mut state = TractionLudState::default();
        let key = directed_key(0, 1);
        for _ in 0..20 {
            state.push_day(DailyLudDay {
                current_instruments: vec![0, 1],
                counts: HashMap::from([(key, pair_counts(1, 1))]),
            });
        }
        assert_eq!(state.total_counts.get(&key), Some(&pair_counts(20, 20)));

        state.push_day(DailyLudDay {
            current_instruments: vec![0, 1],
            counts: HashMap::new(),
        });
        assert_eq!(state.total_counts.get(&key), Some(&pair_counts(19, 19)));
    }

    #[test]
    fn traction_lud_spec_has_kyzq_and_network_tags() {
        let spec = StockDailyTractionLud.spec();
        assert_eq!(spec.id, "traction_lud");
        assert!(spec.tags.iter().any(|tag| tag == "KYZQ"));
        assert!(spec.tags.iter().any(|tag| tag == "cs_network"));
        assert_eq!(spec.intraday_raw_dependencies[0].raw_id, RAW_ID);
        assert_eq!(spec.lookback.trading_days, WINDOW - 1);
    }

    #[test]
    fn traction_lud_source_has_no_inner_parallelism_keywords() {
        let source = include_str!("traction_lud.rs");
        let needles = [
            ['r', 'a', 'y', 'o', 'n'].iter().collect::<String>(),
            ['p', 'a', 'r', '_', 'i', 't', 'e', 'r']
                .iter()
                .collect::<String>(),
            [
                'i', 'n', 't', 'o', '_', 'p', 'a', 'r', '_', 'i', 't', 'e', 'r',
            ]
            .iter()
            .collect::<String>(),
        ];
        for needle in needles {
            assert!(!source.contains(&needle));
        }
    }
}
