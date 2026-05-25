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
use crate::factor::common::{clean_intraday_value, stock_minute_raw_spec};
use crate::factor::{Factor, IntradayRawMaterializeMode};

const PROVIDER_KEY: &str = "kyzq_smart_money_provider";
const RAW_VERSION: &str = "0.2.0";
const VERSION: &str = "0.1.0";

const Q_FCT_RAW_ID: &str = "daily_kyzq_q_fct_raw";

const Q_WINDOW_DAYS: usize = 10;
const Q_MIN_VALID_DAYS: usize = 5;
const Q_SMART_VOLUME_SHARE: f64 = 0.20;
const EPS: f64 = f64::EPSILON;

pub struct StockDailyQFct;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyQFct)
}

#[derive(Debug, Default)]
struct QFctState {
    smart_days: VecDeque<SmartDay>,
}

#[derive(Clone, Debug, Default)]
struct SmartDay {
    by_stock: BTreeMap<String, Vec<SmartObservation>>,
}

#[derive(Clone, Copy, Debug)]
struct SmartObservation {
    score: Option<f64>,
    volume: f64,
    amount: f64,
}

impl Factor for StockDailyQFct {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "q_fct".to_string(),
            aliases: vec!["Q_fct".to_string()],
            name: "Q_fct".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "KYZQ",
                "price_volume",
                "return",
                "intraday",
                "minute_agg",
                "neutralize",
                "barra",
                "size",
                "sector",
                "daily",
                "smart_money",
                "vwap",
                "volume",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Q_fct KYZQ smart-money Q factor from 10-day minute smart VWAP divided by all-minute VWAP, neutralized by Barra SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(
                Q_FCT_RAW_ID,
                Q_WINDOW_DAYS - 1,
            )],
            lookback: Lookback {
                trading_days: Q_WINDOW_DAYS - 1,
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
        Box::new(QFctState::default())
    }

    fn minute_compute_stateful_many(
        &self,
        raw_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
        state: &mut dyn Any,
    ) -> Result<Vec<IntradayDailyRawSeries>> {
        if !raw_ids.iter().any(|raw_id| raw_id == Q_FCT_RAW_ID) {
            return Ok(Vec::new());
        }

        let state = state
            .downcast_mut::<QFctState>()
            .ok_or_else(|| err("KYZQ q_fct raw received incompatible state"))?;
        let trade_date = *context
            .target_dates
            .first()
            .ok_or_else(|| err("KYZQ q_fct raw requires one target date"))?;

        let (smart_day, current_stocks) = match data.minute(DatasetId::StockMinute1m, trade_date) {
            Some(table) => smart_day_from_table(table)?,
            None => (SmartDay::default(), BTreeSet::new()),
        };
        state.push_smart_day(smart_day);
        let q_values = state.q_values_for_current_stocks(&current_stocks);

        let values = q_values
            .into_iter()
            .map(|(ts_code, value)| FactorValue {
                key: FactorRowKey::Daily {
                    trade_date,
                    ts_code,
                },
                value,
            })
            .collect();
        Ok(vec![IntradayDailyRawSeries {
            spec: raw_spec(),
            values,
        }])
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(Q_FCT_RAW_ID)?;
        let raw = panel.column(Q_FCT_RAW_ID)?;
        let factor = neutralize_size_sector(&raw, panel, data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn raw_spec() -> IntradayDailyRawSpec {
    stock_minute_raw_spec(
        Q_FCT_RAW_ID,
        RAW_VERSION,
        &["close", "vol", "amount"],
        Q_WINDOW_DAYS,
    )
}

fn smart_day_from_table(table: &Table) -> Result<(SmartDay, BTreeSet<String>)> {
    let ts_codes = table.required_utf8("ts_code")?;
    let trade_times = table.required_utf8("trade_time")?;
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

    let mut smart_by_stock = BTreeMap::new();
    let mut current_stocks = BTreeSet::new();
    for (ts_code, mut indices) in grouped {
        indices.sort_by(|left, right| trade_times[*left].cmp(&trade_times[*right]));
        current_stocks.insert(ts_code.clone());
        let observations = smart_observations_from_indices(&indices, &close, &volume, &amount);
        if !observations.is_empty() {
            smart_by_stock.insert(ts_code, observations);
        }
    }

    Ok((
        SmartDay {
            by_stock: smart_by_stock,
        },
        current_stocks,
    ))
}

fn smart_observations_from_indices(
    indices: &[usize],
    close: &[Option<f64>],
    volume: &[Option<f64>],
    amount: &[Option<f64>],
) -> Vec<SmartObservation> {
    let mut output = Vec::new();
    let mut previous_close = None;
    for idx in indices {
        let current_close = clean_positive(close[*idx]);
        let volume = clean_positive(volume[*idx]);
        let amount = finite_option(amount[*idx]);
        if let (Some(volume), Some(amount)) = (volume, amount) {
            let score = smart_score(previous_close, current_close, volume);
            output.push(SmartObservation {
                score,
                volume,
                amount,
            });
        }
        if current_close.is_some() {
            previous_close = current_close;
        }
    }
    output
}

fn smart_score(
    previous_close: Option<f64>,
    current_close: Option<f64>,
    volume: f64,
) -> Option<f64> {
    let (Some(previous_close), Some(current_close)) = (previous_close, current_close) else {
        return None;
    };
    if previous_close <= EPS {
        return None;
    }
    let minute_return = current_close / previous_close - 1.0;
    if !minute_return.is_finite() {
        return None;
    }
    let denominator = volume.max(1.0).ln();
    if denominator <= EPS || !denominator.is_finite() {
        return Some(0.0);
    }
    finite_option(Some(minute_return.abs() / denominator))
}

impl QFctState {
    fn push_smart_day(&mut self, day: SmartDay) {
        self.smart_days.push_back(day);
        while self.smart_days.len() > Q_WINDOW_DAYS {
            self.smart_days.pop_front();
        }
    }

    fn q_values_for_current_stocks(
        &self,
        current_stocks: &BTreeSet<String>,
    ) -> BTreeMap<String, Option<f64>> {
        current_stocks
            .iter()
            .map(|ts_code| {
                let observations = self.smart_observations_for(ts_code);
                (ts_code.clone(), q_value_from_observations(&observations))
            })
            .collect()
    }

    fn smart_observations_for(&self, ts_code: &str) -> Vec<SmartObservation> {
        let valid_day_count = self
            .smart_days
            .iter()
            .filter(|day| {
                day.by_stock
                    .get(ts_code)
                    .is_some_and(|items| !items.is_empty())
            })
            .count();
        if valid_day_count < Q_MIN_VALID_DAYS {
            return Vec::new();
        }
        let mut output = Vec::new();
        for day in &self.smart_days {
            if let Some(values) = day.by_stock.get(ts_code) {
                output.extend(values.iter().copied());
            }
        }
        output
    }
}

fn q_value_from_observations(observations: &[SmartObservation]) -> Option<f64> {
    if observations.is_empty() {
        return None;
    }
    let total_volume = observations.iter().map(|item| item.volume).sum::<f64>();
    let total_amount = observations.iter().map(|item| item.amount).sum::<f64>();
    if total_volume <= EPS || total_amount <= 0.0 {
        return None;
    }

    let mut scored = observations
        .iter()
        .filter_map(|item| item.score.map(|score| (score, item.volume, item.amount)))
        .collect::<Vec<_>>();
    if scored.is_empty() {
        return None;
    }
    scored.sort_by(|left, right| right.0.total_cmp(&left.0));

    let threshold = total_volume * Q_SMART_VOLUME_SHARE;
    let mut cumulative_volume = 0.0;
    let mut smart_volume = 0.0;
    let mut smart_amount = 0.0;
    for (_, volume, amount) in scored {
        cumulative_volume += volume;
        smart_volume += volume;
        smart_amount += amount;
        if cumulative_volume + EPS >= threshold {
            break;
        }
    }
    if smart_volume <= EPS || smart_amount <= 0.0 {
        return None;
    }
    let smart_vwap = smart_amount / smart_volume;
    let all_vwap = total_amount / total_volume;
    if all_vwap <= EPS {
        return None;
    }
    finite_option(Some(smart_vwap / all_vwap))
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
            (actual - expected).abs() < 1e-10,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn kyzq_q_score_zero_when_log_denominator_is_zero() {
        assert_close(smart_score(Some(10.0), Some(11.0), 1.0), 0.0);
    }

    #[test]
    fn kyzq_q_value_includes_threshold_crossing_minute() {
        let observations = vec![
            SmartObservation {
                score: Some(3.0),
                volume: 10.0,
                amount: 20.0,
            },
            SmartObservation {
                score: Some(2.0),
                volume: 15.0,
                amount: 45.0,
            },
            SmartObservation {
                score: Some(1.0),
                volume: 75.0,
                amount: 150.0,
            },
        ];
        let smart_vwap = (20.0 + 45.0) / (10.0 + 15.0);
        let all_vwap = (20.0 + 45.0 + 150.0) / 100.0;
        assert_close(
            q_value_from_observations(&observations),
            smart_vwap / all_vwap,
        );
    }

    #[test]
    fn kyzq_state_requires_five_valid_q_days() {
        let mut state = QFctState::default();
        for _ in 0..4 {
            state.push_smart_day(SmartDay {
                by_stock: BTreeMap::from([(
                    "000001.SZ".to_string(),
                    vec![SmartObservation {
                        score: Some(1.0),
                        volume: 10.0,
                        amount: 20.0,
                    }],
                )]),
            });
        }
        assert!(state.smart_observations_for("000001.SZ").is_empty());
        state.push_smart_day(SmartDay {
            by_stock: BTreeMap::from([(
                "000001.SZ".to_string(),
                vec![SmartObservation {
                    score: Some(1.0),
                    volume: 10.0,
                    amount: 20.0,
                }],
            )]),
        });
        assert_eq!(state.smart_observations_for("000001.SZ").len(), 5);
    }

    #[test]
    fn kyzq_q_factor_spec_has_kyzq_tag() {
        let spec = StockDailyQFct.spec();
        assert_eq!(spec.id, "q_fct");
        assert!(spec.tags.iter().any(|tag| tag == "KYZQ"));
        assert!(spec.tags.iter().any(|tag| tag == "smart_money"));
    }
}
