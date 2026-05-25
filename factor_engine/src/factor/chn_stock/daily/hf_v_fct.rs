use std::any::Any;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec,
    Lookback,
};
use crate::data::{DataPool, Table};
use crate::error::{err, Result};
use crate::factor::common::stock_daily_ops::neutralize_size_sector;
use crate::factor::common::{clean_intraday_value, stock_minute_raw_spec, DailyPanel, PanelColumn};
use crate::factor::{Factor, IntradayRawMaterializeMode};
use crate::operators::{cs_zscore, ts_mean, ts_std_dev};

const VERSION: &str = "0.1.0";
const RAW_VERSION: &str = "0.1.0";
const PROVIDER_KEY: &str = "kyzq_hf_v_fct_provider";

const MINUTE_IDEAL_RAW_ID: &str = "daily_kyzq_hf_v_fct_minute_ideal_raw";
const INTRADAY_CUT_RAW_ID: &str = "daily_kyzq_hf_v_fct_intraday_cut_raw";

const WINDOW: usize = 10;
const MIN_VALID_DAYS: usize = 5;
const MINUTE_IDEAL_CUT_RATIO: f64 = 0.25;
const INTRADAY_CUT_RATIO: f64 = 0.20;
const MIN_PERIODS: usize = 1;

pub struct StockDailyHfVFct;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyHfVFct)
}

#[derive(Debug, Default)]
struct HfVFctState {
    days: VecDeque<MinuteDay>,
}

#[derive(Clone, Debug, Default)]
struct MinuteDay {
    by_stock: BTreeMap<String, Vec<MinuteObservation>>,
}

#[derive(Clone, Copy, Debug)]
struct MinuteObservation {
    close: Option<f64>,
    amplitude: Option<f64>,
    minute_return: Option<f64>,
}

impl Factor for StockDailyHfVFct {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "hf_v_fct".to_string(),
            aliases: vec![
                "HF V Fct".to_string(),
                "High Frequency Amplitude Composite".to_string(),
            ],
            name: "hf_v_fct".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "KYZQ high-frequency amplitude composite from 10-day merged 1-minute ideal amplitude and 10-day intraday amplitude-cut mean/std components, neutralized by Barra SIZE and SW sector. Negative report direction is preserved.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: all_raw_ids()
                .iter()
                .map(|raw_id| IntradayDailyRawRequest::new(raw_id, WINDOW - 1))
                .collect(),
            lookback: Lookback {
                trading_days: WINDOW - 1,
            },
        }
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        all_raw_ids()
            .iter()
            .map(|raw_id| raw_spec(raw_id))
            .collect()
    }

    fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
        PROVIDER_KEY.to_string()
    }

    fn intraday_raw_materialize_mode(&self, _raw_ids: &[String]) -> IntradayRawMaterializeMode {
        IntradayRawMaterializeMode::Stateful
    }

    fn initial_intraday_raw_state(&self, _raw_ids: &[String]) -> Box<dyn Any + Send> {
        Box::new(HfVFctState::default())
    }

    fn minute_compute_stateful_many(
        &self,
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
            .downcast_mut::<HfVFctState>()
            .ok_or_else(|| err("KYZQ hf_v_fct raw received incompatible state"))?;
        let trade_date = *context
            .target_dates
            .first()
            .ok_or_else(|| err("KYZQ hf_v_fct raw requires one target date"))?;

        let (minute_day, current_stocks) = match data.minute(DatasetId::StockMinute1m, trade_date) {
            Some(table) => minute_day_from_table(table)?,
            None => (MinuteDay::default(), BTreeSet::new()),
        };
        let intraday_cut_values = if requested.contains(INTRADAY_CUT_RAW_ID) {
            daily_cut_values(&minute_day, &current_stocks)
        } else {
            BTreeMap::new()
        };

        state.push_day(minute_day);
        let minute_ideal_values = if requested.contains(MINUTE_IDEAL_RAW_ID) {
            state.minute_ideal_values(&current_stocks)
        } else {
            BTreeMap::new()
        };

        let mut output = Vec::new();
        if requested.contains(MINUTE_IDEAL_RAW_ID) {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(MINUTE_IDEAL_RAW_ID),
                values: values_for_raw(trade_date, minute_ideal_values),
            });
        }
        if requested.contains(INTRADAY_CUT_RAW_ID) {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(INTRADAY_CUT_RAW_ID),
                values: values_for_raw(trade_date, intraday_cut_values),
            });
        }
        Ok(output)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(MINUTE_IDEAL_RAW_ID)?;
        let minute_ideal = panel.column(MINUTE_IDEAL_RAW_ID)?;
        let intraday_cut = panel.column(INTRADAY_CUT_RAW_ID)?;

        let minute_ideal_component = minute_ideal.cs(cs_zscore)?;
        let intraday_cut_component =
            intraday_cut_component(&intraday_cut, &panel)?.cs(cs_zscore)?;
        let composite =
            average_columns(&panel, &[&minute_ideal_component, &intraday_cut_component])?;
        let factor = neutralize_size_sector(&composite, &panel, data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn all_raw_ids() -> [&'static str; 2] {
    [MINUTE_IDEAL_RAW_ID, INTRADAY_CUT_RAW_ID]
}

fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
    let window_days = if raw_id == MINUTE_IDEAL_RAW_ID {
        WINDOW
    } else {
        1
    };
    stock_minute_raw_spec(raw_id, RAW_VERSION, &["high", "low", "close"], window_days)
}

fn tags() -> Vec<String> {
    [
        "KYZQ",
        "price_volume",
        "amplitude",
        "intraday",
        "minute_agg",
        "hidden_structure",
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

fn minute_day_from_table(table: &Table) -> Result<(MinuteDay, BTreeSet<String>)> {
    let ts_codes = table.required_utf8("ts_code")?;
    let trade_times = table.required_utf8("trade_time")?;
    let high = table.required_f64_cast("high")?;
    let low = table.required_f64_cast("low")?;
    let close = table.required_f64_cast("close")?;

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

    let mut by_stock = BTreeMap::new();
    let mut current_stocks = BTreeSet::new();
    for (ts_code, mut indices) in grouped {
        indices.sort_by(|left, right| trade_times[*left].cmp(&trade_times[*right]));
        current_stocks.insert(ts_code.clone());
        let observations = observations_from_indices(&indices, &trade_times, &high, &low, &close);
        if !observations.is_empty() {
            by_stock.insert(ts_code, observations);
        }
    }

    Ok((MinuteDay { by_stock }, current_stocks))
}

fn observations_from_indices(
    indices: &[usize],
    trade_times: &[Option<String>],
    high: &[Option<f64>],
    low: &[Option<f64>],
    close: &[Option<f64>],
) -> Vec<MinuteObservation> {
    let mut output = Vec::new();
    let mut previous_close = None;
    for idx in indices {
        let current_close = clean_positive(close[*idx]);
        if let Some(trade_time) = trade_times[*idx].as_deref() {
            if minute_index(trade_time).is_some() {
                let observation = MinuteObservation {
                    close: current_close,
                    amplitude: amplitude_value(high[*idx], low[*idx]),
                    minute_return: simple_return(previous_close, current_close),
                };
                if observation.close.is_some()
                    || observation.amplitude.is_some()
                    || observation.minute_return.is_some()
                {
                    output.push(observation);
                }
            }
        }
        if current_close.is_some() {
            previous_close = current_close;
        }
    }
    output
}

impl HfVFctState {
    fn push_day(&mut self, day: MinuteDay) {
        self.days.push_back(day);
        while self.days.len() > WINDOW {
            self.days.pop_front();
        }
    }

    fn minute_ideal_values(
        &self,
        current_stocks: &BTreeSet<String>,
    ) -> BTreeMap<String, Option<f64>> {
        current_stocks
            .iter()
            .map(|ts_code| {
                let observations = self.minute_ideal_observations_for(ts_code);
                (
                    ts_code.clone(),
                    minute_ideal_value_from_observations(&observations),
                )
            })
            .collect()
    }

    fn minute_ideal_observations_for(&self, ts_code: &str) -> Vec<MinuteObservation> {
        let valid_days = self
            .days
            .iter()
            .filter(|day| {
                day.by_stock
                    .get(ts_code)
                    .is_some_and(|items| items.iter().any(has_ideal_observation))
            })
            .count();
        if valid_days < MIN_VALID_DAYS {
            return Vec::new();
        }
        let mut output = Vec::new();
        for day in &self.days {
            if let Some(values) = day.by_stock.get(ts_code) {
                output.extend(values.iter().copied().filter(has_ideal_observation));
            }
        }
        output
    }
}

fn daily_cut_values(
    minute_day: &MinuteDay,
    current_stocks: &BTreeSet<String>,
) -> BTreeMap<String, Option<f64>> {
    current_stocks
        .iter()
        .map(|ts_code| {
            let value = minute_day
                .by_stock
                .get(ts_code)
                .and_then(|values| intraday_cut_value_from_observations(values));
            (ts_code.clone(), value)
        })
        .collect()
}

fn minute_ideal_value_from_observations(observations: &[MinuteObservation]) -> Option<f64> {
    spread_from_pairs(
        observations
            .iter()
            .filter_map(|item| Some((finite_option(item.close)?, finite_option(item.amplitude)?)))
            .collect(),
        MINUTE_IDEAL_CUT_RATIO,
    )
}

fn intraday_cut_value_from_observations(observations: &[MinuteObservation]) -> Option<f64> {
    spread_from_pairs(
        observations
            .iter()
            .filter_map(|item| {
                Some((
                    finite_option(item.minute_return)?,
                    finite_option(item.amplitude)?,
                ))
            })
            .collect(),
        INTRADAY_CUT_RATIO,
    )
}

fn spread_from_pairs(mut pairs: Vec<(f64, f64)>, cut_ratio: f64) -> Option<f64> {
    if pairs.is_empty() {
        return None;
    }
    pairs.sort_by(|left, right| left.0.total_cmp(&right.0));
    let take_count = cut_count(pairs.len(), cut_ratio);
    let low_mean = mean_metric(&pairs[..take_count]);
    let high_mean = mean_metric(&pairs[pairs.len() - take_count..]);
    finite_option(Some(high_mean - low_mean))
}

fn cut_count(valid_count: usize, cut_ratio: f64) -> usize {
    if valid_count == 0 {
        return 0;
    }
    ((valid_count as f64) * cut_ratio)
        .ceil()
        .max(1.0)
        .min(valid_count as f64) as usize
}

fn mean_metric(pairs: &[(f64, f64)]) -> f64 {
    pairs.iter().map(|(_, metric)| *metric).sum::<f64>() / pairs.len() as f64
}

fn intraday_cut_component(values: &PanelColumn, panel: &DailyPanel) -> Result<PanelColumn> {
    let mean = values.ts(|series| ts_mean(series, WINDOW, MIN_PERIODS))?;
    let std = values.ts(|series| ts_std_dev(series, WINDOW, MIN_PERIODS))?;
    let mean_z = mean.cs(cs_zscore)?;
    let std_z = std.cs(cs_zscore)?;
    average_columns(panel, &[&mean_z, &std_z])
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

fn values_for_raw(trade_date: i32, values: BTreeMap<String, Option<f64>>) -> Vec<FactorValue> {
    values
        .into_iter()
        .map(|(ts_code, value)| FactorValue {
            key: FactorRowKey::Daily {
                trade_date,
                ts_code,
            },
            value,
        })
        .collect()
}

fn has_ideal_observation(item: &MinuteObservation) -> bool {
    item.close.is_some() && item.amplitude.is_some()
}

fn amplitude_value(high: Option<f64>, low: Option<f64>) -> Option<f64> {
    let (Some(high), Some(low)) = (clean_positive(high), clean_positive(low)) else {
        return None;
    };
    if low.abs() <= f64::EPSILON {
        return None;
    }
    finite_option(Some(high / low - 1.0))
}

fn simple_return(start: Option<f64>, end: Option<f64>) -> Option<f64> {
    let (Some(start), Some(end)) = (clean_positive(start), clean_positive(end)) else {
        return None;
    };
    if start.abs() <= f64::EPSILON {
        return None;
    }
    finite_option(Some(end / start - 1.0))
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
    clean_intraday_value(value).filter(|value| *value > 0.0)
}

fn finite_option(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("value");
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn kyzq_hf_v_fct_minute_ideal_uses_ceiled_quarter_split_by_close() {
        let observations = (1..=8)
            .map(|value| MinuteObservation {
                close: Some(value as f64),
                amplitude: Some(value as f64 * 0.01),
                minute_return: None,
            })
            .collect::<Vec<_>>();

        assert_close(
            minute_ideal_value_from_observations(&observations),
            (0.07 + 0.08) / 2.0 - (0.01 + 0.02) / 2.0,
        );
    }

    #[test]
    fn kyzq_hf_v_fct_intraday_cut_uses_ceiled_twenty_percent_split_by_return() {
        let observations = vec![
            MinuteObservation {
                close: Some(1.0),
                amplitude: Some(0.10),
                minute_return: Some(-0.03),
            },
            MinuteObservation {
                close: Some(1.0),
                amplitude: Some(0.20),
                minute_return: Some(-0.02),
            },
            MinuteObservation {
                close: Some(1.0),
                amplitude: Some(0.30),
                minute_return: Some(0.01),
            },
            MinuteObservation {
                close: Some(1.0),
                amplitude: Some(0.40),
                minute_return: Some(0.02),
            },
            MinuteObservation {
                close: Some(1.0),
                amplitude: Some(0.50),
                minute_return: Some(0.03),
            },
        ];

        assert_close(
            intraday_cut_value_from_observations(&observations),
            0.50 - 0.10,
        );
    }

    #[test]
    fn kyzq_hf_v_fct_state_requires_five_valid_days() {
        let mut state = HfVFctState::default();
        for _ in 0..4 {
            state.push_day(MinuteDay {
                by_stock: BTreeMap::from([(
                    "000001.SZ".to_string(),
                    vec![MinuteObservation {
                        close: Some(10.0),
                        amplitude: Some(0.01),
                        minute_return: Some(0.01),
                    }],
                )]),
            });
        }
        assert!(state.minute_ideal_observations_for("000001.SZ").is_empty());
        state.push_day(MinuteDay {
            by_stock: BTreeMap::from([(
                "000001.SZ".to_string(),
                vec![MinuteObservation {
                    close: Some(11.0),
                    amplitude: Some(0.02),
                    minute_return: Some(0.01),
                }],
            )]),
        });
        assert_eq!(state.minute_ideal_observations_for("000001.SZ").len(), 5);
    }

    #[test]
    fn kyzq_hf_v_fct_minute_index_uses_regular_session_numbering() {
        assert_eq!(minute_index("09:31:00"), Some(0));
        assert_eq!(minute_index("11:30:00"), Some(119));
        assert_eq!(minute_index("13:01:00"), Some(120));
        assert_eq!(minute_index("15:00:00"), Some(239));
        assert_eq!(minute_index("09:30:00"), None);
    }

    #[test]
    fn kyzq_hf_v_fct_factor_spec_has_kyzq_tag() {
        let spec = StockDailyHfVFct.spec();
        assert_eq!(spec.id, "hf_v_fct");
        assert!(spec.tags.iter().any(|tag| tag == "KYZQ"));
        assert!(spec.tags.iter().any(|tag| tag == "amplitude"));
    }
}
