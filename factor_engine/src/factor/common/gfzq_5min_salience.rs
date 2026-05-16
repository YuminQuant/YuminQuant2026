use std::any::Any;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec,
    Lookback,
};
use crate::data::{DataPool, Table};
use crate::error::{err, Result};
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_size_sector};
use crate::factor::common::stock_daily_raw_ids::{
    STR_5MIN_MA_ACTIVE_RAW_ID, STR_5MIN_MA_RAW_ID, STR_5MIN_MA_SIGMA_RAW_ID,
};
use crate::factor::common::{clean_intraday_value, stock_minute_raw_spec};
use crate::factor::IntradayRawMaterializeMode;
use crate::operators::cs_zscore;

pub const VERSION: &str = "0.1.0";
pub const RAW_VERSION: &str = "0.1.0";
pub const PROVIDER_KEY: &str = "gfzq_5min_salience_provider";

const RAW_WINDOW_DAYS: usize = 5;
const FIVE_MINUTE_SLOTS: usize = 48;
const MORNING_START_MINUTE: i32 = 9 * 60 + 31;
const MORNING_END_MINUTE: i32 = 11 * 60 + 30;
const AFTERNOON_START_MINUTE: i32 = 13 * 60 + 1;
const AFTERNOON_END_MINUTE: i32 = 15 * 60;
const THETA: f64 = 0.9;
const DELTA: f64 = 0.9;
const MIN_PERIODS: usize = 1;
const EPS: f64 = f64::EPSILON;

#[derive(Clone, Copy, Debug)]
pub struct Gfzq5minSalienceFactorDef {
    pub id: &'static str,
    pub alias: &'static str,
    pub name: &'static str,
    pub raw_id: &'static str,
}

#[derive(Clone, Copy, Debug, Default)]
struct FiveMinuteObservation {
    ret: f64,
    salience: f64,
}

#[derive(Clone, Debug, Default)]
struct DayState {
    by_stock: BTreeMap<String, [Option<FiveMinuteObservation>; FIVE_MINUTE_SLOTS]>,
}

#[derive(Debug, Default)]
pub struct Gfzq5minSalienceState {
    days: VecDeque<DayState>,
}

#[derive(Clone, Copy, Debug, Default)]
struct RawValues {
    str_ma: Option<f64>,
    str_ma_active: Option<f64>,
    str_ma_sigma: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default)]
struct FiveMinuteBarBuilder {
    first_minute: Option<i32>,
    last_minute: Option<i32>,
    open: Option<f64>,
    close: Option<f64>,
}

pub fn all_raw_ids() -> [&'static str; 3] {
    [
        STR_5MIN_MA_RAW_ID,
        STR_5MIN_MA_ACTIVE_RAW_ID,
        STR_5MIN_MA_SIGMA_RAW_ID,
    ]
}

pub fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
    stock_minute_raw_spec(raw_id, RAW_VERSION, &["open", "close"], RAW_WINDOW_DAYS)
}

pub fn raw_specs() -> Vec<IntradayDailyRawSpec> {
    all_raw_ids()
        .iter()
        .map(|raw_id| raw_spec(raw_id))
        .collect()
}

pub fn factor_spec(def: Gfzq5minSalienceFactorDef) -> FactorSpec {
    FactorSpec {
        id: def.id.to_string(),
        aliases: vec![def.alias.to_string()],
        name: def.name.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: tags(),
        description: format!(
            "{} GFZQ cross-day 5-minute salience factor, z-scored and neutralized by Barra SIZE and SW sector.",
            def.name
        ),
        dependencies: vec![
            DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
            DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
        ],
        intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(def.raw_id, RAW_WINDOW_DAYS - 1)],
        lookback: Lookback {
            trading_days: RAW_WINDOW_DAYS - 1,
        },
    }
}

pub fn compute_factor(def: Gfzq5minSalienceFactorDef, data: &DataPool) -> Result<FactorSeries> {
    let panel = data.intraday_daily_raw_panel(def.raw_id)?;
    let raw = panel.column(def.raw_id)?;
    let standardized = raw.cs(cs_zscore)?;
    let factor = neutralize_size_sector(&standardized, &panel, data)?;
    Ok(factor.to_factor_series(factor_spec(def)))
}

#[macro_export]
macro_rules! define_gfzq_5min_salience_factor {
    ($struct_name:ident, $id:expr, $alias:expr, $name:expr, $raw_id:expr) => {
        const DEF: $crate::factor::common::gfzq_5min_salience::Gfzq5minSalienceFactorDef =
            $crate::factor::common::gfzq_5min_salience::Gfzq5minSalienceFactorDef {
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
                $crate::factor::common::gfzq_5min_salience::factor_spec(DEF)
            }

            fn intraday_raw_specs(&self) -> Vec<$crate::core::IntradayDailyRawSpec> {
                vec![$crate::factor::common::gfzq_5min_salience::raw_spec(
                    DEF.raw_id,
                )]
            }

            fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
                $crate::factor::common::gfzq_5min_salience::PROVIDER_KEY.to_string()
            }

            fn intraday_raw_materialize_mode(
                &self,
                _raw_ids: &[String],
            ) -> $crate::factor::IntradayRawMaterializeMode {
                $crate::factor::common::gfzq_5min_salience::intraday_raw_materialize_mode()
            }

            fn initial_intraday_raw_state(
                &self,
                _raw_ids: &[String],
            ) -> Box<dyn std::any::Any + Send> {
                $crate::factor::common::gfzq_5min_salience::initial_intraday_raw_state()
            }

            fn minute_compute_stateful_many(
                &self,
                raw_ids: &[String],
                context: &$crate::core::FactorContext,
                data: &$crate::data::DataPool,
                state: &mut dyn std::any::Any,
            ) -> $crate::error::Result<Vec<$crate::core::IntradayDailyRawSeries>> {
                $crate::factor::common::gfzq_5min_salience::minute_compute_stateful_many(
                    raw_ids, context, data, state,
                )
            }

            fn compute(
                &self,
                _context: &$crate::core::FactorContext,
                data: &$crate::data::DataPool,
            ) -> $crate::error::Result<$crate::core::FactorSeries> {
                $crate::factor::common::gfzq_5min_salience::compute_factor(DEF, data)
            }
        }
    };
}

pub fn intraday_raw_materialize_mode() -> IntradayRawMaterializeMode {
    IntradayRawMaterializeMode::Stateful
}

pub fn initial_intraday_raw_state() -> Box<dyn Any + Send> {
    Box::new(Gfzq5minSalienceState::default())
}

pub fn minute_compute_stateful_many(
    raw_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
    state: &mut dyn Any,
) -> Result<Vec<IntradayDailyRawSeries>> {
    let requested = raw_ids
        .iter()
        .map(String::as_str)
        .filter(|raw_id| all_raw_ids().contains(raw_id))
        .collect::<BTreeSet<_>>();
    if requested.is_empty() {
        return Ok(Vec::new());
    }

    let state = state
        .downcast_mut::<Gfzq5minSalienceState>()
        .ok_or_else(|| err("GFZQ 5min salience raw received incompatible state"))?;
    let trade_date = *context
        .target_dates
        .first()
        .ok_or_else(|| err("GFZQ 5min salience raw requires one target date"))?;

    let day_state = match data.minute(DatasetId::StockMinute1m, trade_date) {
        Some(table) => day_state_from_table(table)?,
        None => DayState::default(),
    };
    let current_stocks = day_state.by_stock.keys().cloned().collect::<Vec<_>>();
    state.push_day(day_state);

    let values = current_stocks
        .into_iter()
        .map(|ts_code| {
            let observations = state.observations_for(&ts_code);
            let values = raw_values_from_observations(&observations);
            (ts_code, values)
        })
        .collect::<BTreeMap<_, _>>();

    Ok(series_from_values(trade_date, requested, values))
}

fn tags() -> Vec<String> {
    [
        "GFZQ",
        "behavioral",
        "salience",
        "intraday",
        "5min",
        "stateful_raw",
        "daily",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect()
}

impl Gfzq5minSalienceState {
    fn push_day(&mut self, day: DayState) {
        self.days.push_back(day);
        while self.days.len() > RAW_WINDOW_DAYS {
            self.days.pop_front();
        }
    }

    fn observations_for(&self, ts_code: &str) -> Vec<FiveMinuteObservation> {
        let mut output = Vec::with_capacity(RAW_WINDOW_DAYS * FIVE_MINUTE_SLOTS);
        for day in &self.days {
            let Some(observations) = day.by_stock.get(ts_code) else {
                continue;
            };
            output.extend(observations.iter().filter_map(|value| *value));
        }
        output
    }
}

fn day_state_from_table(table: &Table) -> Result<DayState> {
    let returns_by_stock = five_minute_returns_by_stock(table)?;
    let mut market_returns = [None; FIVE_MINUTE_SLOTS];
    for slot in 0..FIVE_MINUTE_SLOTS {
        let mut sum = 0.0;
        let mut count = 0usize;
        for (ts_code, returns) in &returns_by_stock {
            if is_bj_stock(ts_code) {
                continue;
            }
            if let Some(value) = returns[slot] {
                sum += value;
                count += 1;
            }
        }
        if count > 0 {
            market_returns[slot] = Some(sum / count as f64);
        }
    }

    let by_stock = returns_by_stock
        .into_iter()
        .map(|(ts_code, returns)| {
            let observations = std::array::from_fn(|slot| {
                let (Some(ret), Some(market_ret)) = (returns[slot], market_returns[slot]) else {
                    return None;
                };
                salience_value(ret, market_ret)
                    .map(|salience| FiveMinuteObservation { ret, salience })
            });
            (ts_code, observations)
        })
        .collect();
    Ok(DayState { by_stock })
}

fn five_minute_returns_by_stock(
    table: &Table,
) -> Result<BTreeMap<String, [Option<f64>; FIVE_MINUTE_SLOTS]>> {
    let ts_codes = table.required_utf8("ts_code")?;
    let trade_times = table.required_utf8("trade_time")?;
    let open = table.required_f64_cast("open")?;
    let close = table.required_f64_cast("close")?;
    let mut builders = BTreeMap::<String, [FiveMinuteBarBuilder; FIVE_MINUTE_SLOTS]>::new();

    for idx in 0..table.len {
        let (Some(ts_code), Some(trade_time)) =
            (ts_codes[idx].clone(), trade_times[idx].as_deref())
        else {
            continue;
        };
        let Some((slot, minute)) = five_minute_slot(trade_time) else {
            continue;
        };
        builders
            .entry(ts_code)
            .or_insert_with(|| [FiveMinuteBarBuilder::default(); FIVE_MINUTE_SLOTS])[slot]
            .push(minute, open[idx], close[idx]);
    }

    Ok(builders
        .into_iter()
        .map(|(ts_code, builders)| {
            (
                ts_code,
                std::array::from_fn(|slot| builders[slot].return_value()),
            )
        })
        .collect())
}

fn five_minute_slot(trade_time: &str) -> Option<(usize, i32)> {
    let minute = minute_of_day(trade_time)?;
    if (MORNING_START_MINUTE..=MORNING_END_MINUTE).contains(&minute) {
        return Some((((minute - MORNING_START_MINUTE) / 5) as usize, minute));
    }
    if (AFTERNOON_START_MINUTE..=AFTERNOON_END_MINUTE).contains(&minute) {
        return Some((
            24 + ((minute - AFTERNOON_START_MINUTE) / 5) as usize,
            minute,
        ));
    }
    None
}

fn minute_of_day(value: &str) -> Option<i32> {
    let time = value.split_whitespace().last().unwrap_or(value).trim();
    let mut parts = time.split(':');
    let hour = parts.next()?.parse::<i32>().ok()?;
    let minute = parts.next()?.parse::<i32>().ok()?;
    Some(hour * 60 + minute)
}

impl FiveMinuteBarBuilder {
    fn push(&mut self, minute: i32, open: Option<f64>, close: Option<f64>) {
        if let Some(value) = clean_intraday_value(open) {
            if self.first_minute.is_none_or(|current| minute < current) {
                self.first_minute = Some(minute);
                self.open = Some(value);
            }
        }
        if let Some(value) = clean_intraday_value(close) {
            if self.last_minute.is_none_or(|current| minute > current) {
                self.last_minute = Some(minute);
                self.close = Some(value);
            }
        }
    }

    fn return_value(&self) -> Option<f64> {
        let (Some(open), Some(close)) = (self.open, self.close) else {
            return None;
        };
        if open.abs() <= EPS {
            return None;
        }
        finite_value(close / open - 1.0)
    }
}

fn salience_value(ret: f64, market_ret: f64) -> Option<f64> {
    let denominator = ret.abs() + market_ret.abs() + THETA;
    if denominator.abs() <= EPS {
        return None;
    }
    finite_value((ret - market_ret).abs() / denominator)
}

fn raw_values_from_observations(observations: &[FiveMinuteObservation]) -> RawValues {
    if observations.len() < MIN_PERIODS {
        return RawValues::default();
    }
    let equal_mean =
        observations.iter().map(|row| row.ret).sum::<f64>() / observations.len() as f64;
    let weighted = weighted_mean_by_salience(observations, SortDirection::Descending);
    let reverse_weighted = weighted_mean_by_salience(observations, SortDirection::Ascending);
    RawValues {
        str_ma: weighted.and_then(|value| finite_value(value - equal_mean)),
        str_ma_active: weighted,
        str_ma_sigma: reverse_weighted.and_then(|value| finite_value(value - equal_mean)),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SortDirection {
    Ascending,
    Descending,
}

fn weighted_mean_by_salience(
    observations: &[FiveMinuteObservation],
    direction: SortDirection,
) -> Option<f64> {
    let saliences = observations
        .iter()
        .map(|row| row.salience)
        .collect::<Vec<_>>();
    let ranks = one_based_ranks(&saliences, direction);
    let weights = ranks
        .iter()
        .map(|rank| DELTA.powf(*rank))
        .collect::<Vec<_>>();
    let denominator = weights.iter().sum::<f64>();
    if denominator.abs() <= EPS || !denominator.is_finite() {
        return None;
    }
    finite_value(
        observations
            .iter()
            .zip(weights.iter())
            .map(|(row, weight)| row.ret * weight / denominator)
            .sum::<f64>(),
    )
}

fn one_based_ranks(values: &[f64], direction: SortDirection) -> Vec<f64> {
    let mut pairs = values
        .iter()
        .enumerate()
        .map(|(idx, value)| (idx, *value))
        .collect::<Vec<_>>();
    pairs.sort_by(|left, right| {
        let ordering = match direction {
            SortDirection::Ascending => left.1.total_cmp(&right.1),
            SortDirection::Descending => right.1.total_cmp(&left.1),
        };
        ordering.then_with(|| left.0.cmp(&right.0))
    });

    let mut output = vec![0.0; values.len()];
    let mut start = 0usize;
    while start < pairs.len() {
        let mut end = start + 1;
        while end < pairs.len() && (pairs[end].1 - pairs[start].1).abs() <= EPS {
            end += 1;
        }
        let avg_rank = (start + 1 + end) as f64 / 2.0;
        for idx in start..end {
            output[pairs[idx].0] = avg_rank;
        }
        start = end;
    }
    output
}

fn series_from_values(
    trade_date: i32,
    requested: BTreeSet<&str>,
    values: BTreeMap<String, RawValues>,
) -> Vec<IntradayDailyRawSeries> {
    let mut by_raw_id = all_raw_ids()
        .iter()
        .map(|raw_id| (*raw_id, Vec::<FactorValue>::new()))
        .collect::<BTreeMap<_, _>>();
    for (ts_code, values) in values {
        let key = FactorRowKey::Daily {
            trade_date,
            ts_code,
        };
        push_value(
            &mut by_raw_id,
            &requested,
            STR_5MIN_MA_RAW_ID,
            &key,
            values.str_ma,
        );
        push_value(
            &mut by_raw_id,
            &requested,
            STR_5MIN_MA_ACTIVE_RAW_ID,
            &key,
            values.str_ma_active,
        );
        push_value(
            &mut by_raw_id,
            &requested,
            STR_5MIN_MA_SIGMA_RAW_ID,
            &key,
            values.str_ma_sigma,
        );
    }

    all_raw_ids()
        .iter()
        .filter(|raw_id| requested.contains(**raw_id))
        .map(|raw_id| IntradayDailyRawSeries {
            spec: raw_spec(raw_id),
            values: by_raw_id.remove(raw_id).unwrap_or_default(),
        })
        .collect()
}

fn push_value(
    by_raw_id: &mut BTreeMap<&'static str, Vec<FactorValue>>,
    requested: &BTreeSet<&str>,
    raw_id: &'static str,
    key: &FactorRowKey,
    value: Option<f64>,
) {
    if requested.contains(raw_id) {
        by_raw_id.entry(raw_id).or_default().push(FactorValue {
            key: key.clone(),
            value,
        });
    }
}

fn finite_value(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::data::ColumnData;

    use super::*;

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("expected value");
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn gfzq_5min_slot_uses_0931_to_1500_as_48_bars() {
        assert_eq!(five_minute_slot("09:30:00"), None);
        assert_eq!(five_minute_slot("09:31:00"), Some((0, 571)));
        assert_eq!(five_minute_slot("09:35:00"), Some((0, 575)));
        assert_eq!(five_minute_slot("09:36:00"), Some((1, 576)));
        assert_eq!(five_minute_slot("11:30:00"), Some((23, 690)));
        assert_eq!(five_minute_slot("13:01:00"), Some((24, 781)));
        assert_eq!(five_minute_slot("15:00:00"), Some((47, 900)));
    }

    #[test]
    fn gfzq_5min_state_keeps_latest_five_days() {
        let mut state = Gfzq5minSalienceState::default();
        for day in 0..6 {
            let mut day_state = DayState::default();
            let mut observations = [None; FIVE_MINUTE_SLOTS];
            observations[0] = Some(FiveMinuteObservation {
                ret: day as f64,
                salience: day as f64,
            });
            day_state
                .by_stock
                .insert("000001.SZ".to_string(), observations);
            state.push_day(day_state);
        }
        let observations = state.observations_for("000001.SZ");
        assert_eq!(observations.len(), 5);
        assert_eq!(observations[0].ret, 1.0);
        assert_eq!(observations[4].ret, 5.0);
    }

    #[test]
    fn gfzq_5min_market_return_excludes_bj_stocks() {
        let table = minute_table(vec![
            ("000001.SZ", "09:31:00", 100.0, 101.0),
            ("000001.SZ", "09:35:00", 101.0, 102.0),
            ("430001.BJ", "09:31:00", 100.0, 200.0),
            ("430001.BJ", "09:35:00", 200.0, 300.0),
        ]);
        let day = day_state_from_table(&table).expect("day state");
        let sz = day.by_stock.get("000001.SZ").unwrap()[0].unwrap();
        let bj = day.by_stock.get("430001.BJ").unwrap()[0].unwrap();
        assert_close(Some(sz.ret), 0.02);
        assert_close(Some(sz.salience), 0.0);
        assert!(bj.salience > 0.0);
    }

    #[test]
    fn gfzq_5min_raw_values_match_manual_weighting() {
        let observations = vec![
            FiveMinuteObservation {
                ret: 0.01,
                salience: 0.3,
            },
            FiveMinuteObservation {
                ret: 0.04,
                salience: 0.1,
            },
            FiveMinuteObservation {
                ret: -0.02,
                salience: 0.2,
            },
        ];
        let values = raw_values_from_observations(&observations);
        let mean = 0.01;
        let desc_weights = [DELTA.powf(1.0), DELTA.powf(3.0), DELTA.powf(2.0)];
        let desc_den = desc_weights.iter().sum::<f64>();
        let weighted = observations
            .iter()
            .zip(desc_weights.iter())
            .map(|(row, weight)| row.ret * weight / desc_den)
            .sum::<f64>();
        let asc_weights = [DELTA.powf(3.0), DELTA.powf(1.0), DELTA.powf(2.0)];
        let asc_den = asc_weights.iter().sum::<f64>();
        let reverse_weighted = observations
            .iter()
            .zip(asc_weights.iter())
            .map(|(row, weight)| row.ret * weight / asc_den)
            .sum::<f64>();
        assert_close(values.str_ma, weighted - mean);
        assert_close(values.str_ma_active, weighted);
        assert_close(values.str_ma_sigma, reverse_weighted - mean);
    }

    fn minute_table(rows: Vec<(&str, &str, f64, f64)>) -> Table {
        let len = rows.len();
        Table::new(BTreeMap::from([
            (
                "ts_code".to_string(),
                ColumnData::Utf8(
                    rows.iter()
                        .map(|row| Some(row.0.to_string()))
                        .collect::<Vec<_>>(),
                ),
            ),
            (
                "trade_time".to_string(),
                ColumnData::Utf8(
                    rows.iter()
                        .map(|row| Some(row.1.to_string()))
                        .collect::<Vec<_>>(),
                ),
            ),
            (
                "open".to_string(),
                ColumnData::F64(rows.iter().map(|row| Some(row.2)).collect::<Vec<_>>()),
            ),
            (
                "close".to_string(),
                ColumnData::F64(rows.iter().map(|row| Some(row.3)).collect::<Vec<_>>()),
            ),
        ]))
        .unwrap_or_else(|err| panic!("valid table with {len} rows: {err}"))
    }
}
