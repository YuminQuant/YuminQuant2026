use std::any::Any;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawAuxiliaryRequest, IntradayDailyRawRequest,
    IntradayDailyRawSeries, IntradayDailyRawSpec, Lookback,
};
use crate::data::{DataPool, Table};
use crate::error::{err, Result};
use crate::factor::common::{
    clean_intraday_value, intraday_time_in_range, stock_minute_raw_spec, ClassificationLevel,
    ClassificationMap, PanelColumn,
};
use crate::factor::{Factor, IntradayRawMaterializeMode};
use crate::operators::{cs_zscore, ts_mean};

pub const RELATIVE_VWAP_RAW_ID: &str = "daily_trend_capital_relative_vwap";
pub const NET_SUPPORT_VOLUME_RAW_ID: &str = "daily_trend_capital_net_support_volume";

const RAW_VERSION: &str = "0.2.0";
const VERSION: &str = "0.2.0";
const HISTORY_DAYS: usize = 5;
const RAW_WINDOW_DAYS: usize = HISTORY_DAYS + 1;
const DAILY_TOP_VOLUME_COUNT: usize = 120;
const WINDOW: usize = 20;
const MIN_PERIODS: usize = 1;
const FLOAT_SHARE_UNIT: f64 = 10_000.0;

pub struct StockDailyTrendCapitalTradingComposite;

#[derive(Debug, Default)]
struct TrendCapitalState {
    volume_history: BTreeMap<String, VecDeque<DailyVolumeState>>,
}

#[derive(Clone, Debug, Default)]
struct DailyVolumeState {
    top_values: Vec<f64>,
    valid_count: usize,
}

#[derive(Clone, Copy, Debug)]
struct MinutePoint {
    volume: f64,
    amount: Option<f64>,
    close: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default)]
struct TrendCapitalRawValues {
    relative_vwap: Option<f64>,
    net_support_volume: Option<f64>,
}

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyTrendCapitalTradingComposite)
}

fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
    stock_minute_raw_spec(
        raw_id,
        RAW_VERSION,
        &["amount", "close", "vol"],
        RAW_WINDOW_DAYS,
    )
}

impl Factor for StockDailyTrendCapitalTradingComposite {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "trend_capital_trading_composite".to_string(),
            aliases: vec![
                "TrendCapitalTradingComposite".to_string(),
                "Trend Capital Trading Composite".to_string(),
            ],
            name: "Trend Capital Trading Composite".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "volume",
                "amount",
                "vwap",
                "support",
                "trend_capital",
                "intraday",
                "minute_agg",
                "composite",
                "neutralize",
                "barra",
                "size",
                "sector",
                "daily",
                "GSZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Trend capital trading composite factor from previous-five-day volume threshold relative VWAP and net support volume, neutralized by Barra SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: vec![
                IntradayDailyRawRequest::new(RELATIVE_VWAP_RAW_ID, WINDOW - 1),
                IntradayDailyRawRequest::new(NET_SUPPORT_VOLUME_RAW_ID, WINDOW - 1),
            ],
            lookback: Lookback {
                trading_days: WINDOW - 1,
            },
        }
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        vec![
            raw_spec(RELATIVE_VWAP_RAW_ID),
            raw_spec(NET_SUPPORT_VOLUME_RAW_ID),
        ]
    }

    fn intraday_raw_auxiliary_requirements(
        &self,
        raw_ids: &[String],
    ) -> Vec<IntradayDailyRawAuxiliaryRequest> {
        let requested = raw_ids.iter().map(String::as_str).collect::<BTreeSet<_>>();
        if requested.contains(NET_SUPPORT_VOLUME_RAW_ID) {
            vec![IntradayDailyRawAuxiliaryRequest::new(
                DataRequest::new(DatasetId::StockDailyBasic, &["float_share"]),
                0,
            )]
        } else {
            Vec::new()
        }
    }

    fn intraday_raw_materialize_mode(&self, raw_ids: &[String]) -> IntradayRawMaterializeMode {
        if raw_ids
            .iter()
            .any(|raw_id| raw_id == RELATIVE_VWAP_RAW_ID || raw_id == NET_SUPPORT_VOLUME_RAW_ID)
        {
            IntradayRawMaterializeMode::Stateful
        } else {
            IntradayRawMaterializeMode::Stateless
        }
    }

    fn initial_intraday_raw_state(&self, _raw_ids: &[String]) -> Box<dyn Any + Send> {
        Box::new(TrendCapitalState::default())
    }

    fn minute_compute_stateful_many(
        &self,
        raw_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
        state: &mut dyn Any,
    ) -> Result<Vec<IntradayDailyRawSeries>> {
        let requested = raw_ids.iter().map(String::as_str).collect::<BTreeSet<_>>();
        let wants_relative = requested.contains(RELATIVE_VWAP_RAW_ID);
        let wants_support = requested.contains(NET_SUPPORT_VOLUME_RAW_ID);
        if !wants_relative && !wants_support {
            return Ok(Vec::new());
        }

        let state = state
            .downcast_mut::<TrendCapitalState>()
            .ok_or_else(|| err("trend capital stateful raw received incompatible state"))?;
        let trade_date = *context
            .target_dates
            .first()
            .ok_or_else(|| err("trend capital stateful raw requires one target date"))?;
        let float_share = if wants_support {
            float_share_by_code(data.daily(DatasetId::StockDailyBasic)?, trade_date)?
        } else {
            BTreeMap::new()
        };

        let mut current_rows = BTreeMap::<String, Vec<MinutePoint>>::new();
        let mut current_volume_history = BTreeMap::<String, DailyVolumeState>::new();
        if let Some(table) = data.minute(DatasetId::StockMinute1m, trade_date) {
            let (rows, volumes) = selected_minute_points(table)?;
            current_rows = rows;
            current_volume_history = volumes;
        }

        let mut relative_values = Vec::new();
        let mut support_values = Vec::new();
        for (ts_code, rows) in &current_rows {
            let threshold = state
                .volume_history
                .get(ts_code)
                .and_then(previous_five_day_volume_cutoff);
            let raw_values = threshold
                .map(|threshold| {
                    trend_capital_raw_values(
                        rows,
                        threshold,
                        float_share.get(ts_code).copied().flatten(),
                    )
                })
                .unwrap_or_default();

            if wants_relative {
                relative_values.push(FactorValue {
                    key: FactorRowKey::Daily {
                        trade_date,
                        ts_code: ts_code.clone(),
                    },
                    value: raw_values.relative_vwap,
                });
            }
            if wants_support {
                support_values.push(FactorValue {
                    key: FactorRowKey::Daily {
                        trade_date,
                        ts_code: ts_code.clone(),
                    },
                    value: raw_values.net_support_volume,
                });
            }
        }

        state.advance(current_volume_history);

        let mut output = Vec::new();
        if wants_relative {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(RELATIVE_VWAP_RAW_ID),
                values: relative_values,
            });
        }
        if wants_support {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(NET_SUPPORT_VOLUME_RAW_ID),
                values: support_values,
            });
        }
        Ok(output)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let sector_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockSwClassification)?,
            ClassificationLevel::Sector,
        )?;
        let panel = data.intraday_daily_raw_panel(RELATIVE_VWAP_RAW_ID)?;
        let relative_raw = panel.column(RELATIVE_VWAP_RAW_ID)?;
        let support_raw = panel.column(NET_SUPPORT_VOLUME_RAW_ID)?;
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;

        let relative20 = relative_raw.ts(|values| ts_mean(values, WINDOW, MIN_PERIODS))?;
        let support20 = support_raw.ts(|values| ts_mean(values, WINDOW, MIN_PERIODS))?;
        let relative_neutralized = relative20.cs_neutralize_regression_by_group(
            &[&size],
            None,
            |trade_date, ts_codes| sector_map.groups_for(trade_date, ts_codes),
        )?;
        let support_neutralized = support20.cs_neutralize_regression_by_group(
            &[&size],
            None,
            |trade_date, ts_codes| sector_map.groups_for(trade_date, ts_codes),
        )?;
        let factor = combine_components(
            &relative_neutralized.cs(cs_zscore)?,
            &support_neutralized.cs(cs_zscore)?,
        )?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

impl TrendCapitalState {
    fn advance(&mut self, mut current_volume_history: BTreeMap<String, DailyVolumeState>) {
        let all_codes = self
            .volume_history
            .keys()
            .cloned()
            .chain(current_volume_history.keys().cloned())
            .collect::<BTreeSet<_>>();
        for ts_code in all_codes {
            let entry = self.volume_history.entry(ts_code.clone()).or_default();
            entry.push_back(current_volume_history.remove(&ts_code).unwrap_or_default());
            while entry.len() > HISTORY_DAYS {
                entry.pop_front();
            }
        }
    }
}

fn selected_minute_points(
    table: &Table,
) -> Result<(
    BTreeMap<String, Vec<MinutePoint>>,
    BTreeMap<String, DailyVolumeState>,
)> {
    let ts_codes = table.required_utf8("ts_code")?;
    let trade_times = table.required_utf8("trade_time")?;
    let amount = table.required_f64_cast("amount")?;
    let close = table.required_f64_cast("close")?;
    let volume = table.required_f64_cast("vol")?;

    let mut grouped = BTreeMap::<String, Vec<usize>>::new();
    for idx in 0..table.len {
        let Some(ts_code) = ts_codes[idx].clone() else {
            continue;
        };
        let Some(trade_time) = trade_times[idx].as_deref() else {
            continue;
        };
        if intraday_time_in_range(trade_time, "09:31:00", "15:00:00") {
            grouped.entry(ts_code).or_default().push(idx);
        }
    }

    let mut rows_by_code = BTreeMap::<String, Vec<MinutePoint>>::new();
    let mut volume_history_by_code = BTreeMap::<String, DailyVolumeState>::new();
    for (ts_code, mut indices) in grouped {
        indices.sort_by(|left, right| trade_times[*left].cmp(&trade_times[*right]));
        let mut rows = Vec::new();
        let mut history_volumes = Vec::new();
        for idx in indices {
            let Some(volume) = clean_nonnegative(volume[idx]) else {
                continue;
            };
            rows.push(MinutePoint {
                volume,
                amount: clean_intraday_value(amount[idx]),
                close: clean_intraday_value(close[idx]),
            });
            history_volumes.push(volume);
        }
        if !rows.is_empty() {
            rows_by_code.insert(ts_code.clone(), rows);
            volume_history_by_code.insert(ts_code, DailyVolumeState::from_volumes(history_volumes));
        }
    }
    Ok((rows_by_code, volume_history_by_code))
}

fn float_share_by_code(table: &Table, trade_date: i32) -> Result<BTreeMap<String, Option<f64>>> {
    let trade_dates = table.required_i32_date_cast("trade_date")?;
    let ts_codes = table.required_utf8("ts_code")?;
    let float_share = table.required_f64_cast("float_share")?;
    let mut output = BTreeMap::new();
    for idx in 0..table.len {
        if trade_dates[idx] != Some(trade_date) {
            continue;
        }
        let Some(ts_code) = ts_codes[idx].clone() else {
            continue;
        };
        output.insert(ts_code, clean_positive(float_share[idx]));
    }
    Ok(output)
}

impl DailyVolumeState {
    fn from_volumes(mut volumes: Vec<f64>) -> Self {
        let valid_count = volumes.len();
        Self {
            top_values: top_k_unordered(&mut volumes, DAILY_TOP_VOLUME_COUNT),
            valid_count,
        }
    }
}

fn previous_five_day_volume_cutoff(history: &VecDeque<DailyVolumeState>) -> Option<f64> {
    if history.len() != HISTORY_DAYS || history.iter().any(|day| day.valid_count == 0) {
        return None;
    }
    let total_valid_count = history.iter().map(|day| day.valid_count).sum::<usize>();
    let rank_from_high = ((total_valid_count as f64) * 0.1).ceil() as usize;
    if rank_from_high == 0 {
        return None;
    }
    let mut candidates = history
        .iter()
        .flat_map(|day| day.top_values.iter().copied())
        .collect::<Vec<_>>();
    kth_largest(&mut candidates, rank_from_high)
}

fn top_k_unordered(values: &mut [f64], k: usize) -> Vec<f64> {
    if values.is_empty() || k == 0 {
        return Vec::new();
    }
    if values.len() <= k {
        return values.to_vec();
    }
    let split = k - 1;
    values.select_nth_unstable_by(split, descending_f64_cmp);
    values[..k].to_vec()
}

fn kth_largest(values: &mut [f64], k: usize) -> Option<f64> {
    if values.is_empty() || k == 0 || values.len() < k {
        return None;
    }
    let split = k - 1;
    Some(*values.select_nth_unstable_by(split, descending_f64_cmp).1)
}

fn descending_f64_cmp(left: &f64, right: &f64) -> Ordering {
    right.partial_cmp(left).unwrap_or(Ordering::Equal)
}

fn trend_capital_raw_values(
    rows: &[MinutePoint],
    threshold: f64,
    float_share: Option<f64>,
) -> TrendCapitalRawValues {
    let trend_rows = rows
        .iter()
        .copied()
        .filter(|row| row.volume >= threshold)
        .collect::<Vec<_>>();
    TrendCapitalRawValues {
        relative_vwap: relative_vwap(rows, &trend_rows),
        net_support_volume: net_support_volume(&trend_rows, float_share),
    }
}

fn relative_vwap(rows: &[MinutePoint], trend_rows: &[MinutePoint]) -> Option<f64> {
    let all_vwap = minute_vwap(rows)?;
    let trend_vwap = minute_vwap(trend_rows)?;
    (all_vwap.abs() > f64::EPSILON).then_some(trend_vwap / all_vwap - 1.0)
}

fn minute_vwap(rows: &[MinutePoint]) -> Option<f64> {
    let mut amount_sum = 0.0;
    let mut volume_sum = 0.0;
    for row in rows {
        let Some(amount) = clean_intraday_value(row.amount) else {
            continue;
        };
        if row.volume <= f64::EPSILON {
            continue;
        }
        amount_sum += amount;
        volume_sum += row.volume;
    }
    (volume_sum > f64::EPSILON).then_some(amount_sum / volume_sum)
}

fn net_support_volume(trend_rows: &[MinutePoint], float_share: Option<f64>) -> Option<f64> {
    let float_share = clean_positive(float_share)?;
    let close_mean = mean_clean(trend_rows.iter().filter_map(|row| row.close))?;
    let mut support = 0.0;
    let mut resistance = 0.0;
    for row in trend_rows {
        let Some(close) = clean_intraday_value(row.close) else {
            continue;
        };
        if close < close_mean {
            support += row.volume;
        } else if close > close_mean {
            resistance += row.volume;
        }
    }
    Some((support - resistance) / (float_share * FLOAT_SHARE_UNIT))
}

fn combine_components(relative: &PanelColumn, support: &PanelColumn) -> Result<PanelColumn> {
    relative.zip_binary(support, |relative, support| {
        match (
            clean_intraday_value(relative),
            clean_intraday_value(support),
        ) {
            (Some(relative), Some(support)) => Some(-relative + support),
            _ => None,
        }
    })
}

fn mean_clean(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        if value.is_nan() {
            continue;
        }
        sum += value;
        count += 1;
    }
    (count > 0).then_some(sum / count as f64)
}

fn clean_nonnegative(value: Option<f64>) -> Option<f64> {
    clean_intraday_value(value).filter(|value| *value >= 0.0)
}

fn clean_positive(value: Option<f64>) -> Option<f64> {
    clean_intraday_value(value).filter(|value| *value > 0.0)
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::*;

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("expected value");
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn previous_threshold_requires_complete_five_day_window() {
        let mut history = VecDeque::from(vec![
            DailyVolumeState::from_volumes(vec![1.0]),
            DailyVolumeState::from_volumes(vec![2.0]),
            DailyVolumeState::from_volumes(vec![3.0]),
            DailyVolumeState::from_volumes(vec![4.0]),
        ]);
        assert_eq!(previous_five_day_volume_cutoff(&history), None);
        history.push_back(DailyVolumeState::from_volumes(Vec::new()));
        assert_eq!(previous_five_day_volume_cutoff(&history), None);
    }

    #[test]
    fn previous_threshold_uses_top_ten_percent_cutoff() {
        let history = VecDeque::from(vec![
            DailyVolumeState::from_volumes(vec![1.0, 2.0, 3.0, 4.0]),
            DailyVolumeState::from_volumes(vec![5.0, 6.0, 7.0, 8.0]),
            DailyVolumeState::from_volumes(vec![9.0, 10.0, 11.0, 12.0]),
            DailyVolumeState::from_volumes(vec![13.0, 14.0, 15.0, 16.0]),
            DailyVolumeState::from_volumes(vec![17.0, 18.0, 19.0, 20.0]),
        ]);
        assert_close(previous_five_day_volume_cutoff(&history), 19.0);
    }

    #[test]
    fn daily_volume_state_keeps_top_one_hundred_twenty_without_sorting_requirement() {
        let state = DailyVolumeState::from_volumes((1..=240).map(|value| value as f64).collect());
        assert_eq!(state.valid_count, 240);
        assert_eq!(state.top_values.len(), DAILY_TOP_VOLUME_COUNT);
        let min_top = state
            .top_values
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let max_top = state
            .top_values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(min_top, 121.0);
        assert_eq!(max_top, 240.0);
    }

    #[test]
    fn raw_values_include_threshold_boundary_and_use_minute_vwap_units() {
        let rows = vec![
            MinutePoint {
                volume: 100.0,
                amount: Some(1_000.0),
                close: Some(10.0),
            },
            MinutePoint {
                volume: 200.0,
                amount: Some(2_200.0),
                close: Some(11.0),
            },
            MinutePoint {
                volume: 300.0,
                amount: Some(3_900.0),
                close: Some(12.0),
            },
        ];
        let raw = trend_capital_raw_values(&rows, 200.0, Some(1.0));
        let all_vwap = (1_000.0 + 2_200.0 + 3_900.0) / (100.0 + 200.0 + 300.0);
        let trend_vwap = (2_200.0 + 3_900.0) / (200.0 + 300.0);
        assert_close(raw.relative_vwap, trend_vwap / all_vwap - 1.0);
    }

    #[test]
    fn net_support_uses_float_share_unit_and_ignores_equal_close() {
        let rows = vec![
            MinutePoint {
                volume: 100.0,
                amount: Some(1_000.0),
                close: Some(9.0),
            },
            MinutePoint {
                volume: 50.0,
                amount: Some(550.0),
                close: Some(10.0),
            },
            MinutePoint {
                volume: 20.0,
                amount: Some(240.0),
                close: Some(11.0),
            },
        ];
        assert_close(net_support_volume(&rows, Some(2.0)), 80.0 / 20_000.0);
    }
}
