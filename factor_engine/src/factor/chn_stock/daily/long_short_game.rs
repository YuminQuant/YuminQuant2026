use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use crate::core::{
    AssetClass, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec, FactorValue,
    Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec, Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::{clean_intraday_value, intraday_time_in_range, stock_minute_raw_spec};
use crate::factor::Factor;
use crate::operators::{cs_demean_abs, cs_zscore, ts_mean, ts_std_dev};

pub const VOLUME_GAME_RETURN_RAW_ID: &str = "daily_volume_game_return";
pub const VOLUME_GAME_RELATIVE_RAW_ID: &str = "daily_volume_game_relative_position";
pub const AMPLITUDE_GAME_RAW_ID: &str = "daily_amplitude_game";

const RAW_VERSION: &str = "0.1.0";
const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;
const RETURN_LAG: usize = 4;

pub struct StockDailyLongShortGame;

#[derive(Clone, Copy, Debug)]
struct GameValues {
    volume_return: Option<f64>,
    volume_relative: Option<f64>,
    amplitude: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
struct RankedPayload {
    time_idx: usize,
    metric: f64,
    payload: f64,
}

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyLongShortGame)
}

fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
    stock_minute_raw_spec(raw_id, RAW_VERSION, &["close", "high", "low", "vol"], 1)
}

impl Factor for StockDailyLongShortGame {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "long_short_game".to_string(),
            aliases: Vec::new(),
            name: "Long-Short Game".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "volume",
                "amplitude",
                "return",
                "intraday",
                "minute_agg",
                "composite",
                "daily",
                "FZZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Composite long-short game factor from intraday volume and amplitude sorted by 5-minute returns and relative position.".to_string(),
            dependencies: Vec::new(),
            intraday_raw_dependencies: vec![
                IntradayDailyRawRequest::new(VOLUME_GAME_RETURN_RAW_ID, WINDOW - 1),
                IntradayDailyRawRequest::new(VOLUME_GAME_RELATIVE_RAW_ID, WINDOW - 1),
                IntradayDailyRawRequest::new(AMPLITUDE_GAME_RAW_ID, WINDOW - 1),
            ],
            lookback: Lookback {
                trading_days: WINDOW - 1,
            },
        }
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        vec![
            raw_spec(VOLUME_GAME_RETURN_RAW_ID),
            raw_spec(VOLUME_GAME_RELATIVE_RAW_ID),
            raw_spec(AMPLITUDE_GAME_RAW_ID),
        ]
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
        let wants_volume_return = requested.contains(VOLUME_GAME_RETURN_RAW_ID);
        let wants_volume_relative = requested.contains(VOLUME_GAME_RELATIVE_RAW_ID);
        let wants_amplitude = requested.contains(AMPLITUDE_GAME_RAW_ID);
        if !wants_volume_return && !wants_volume_relative && !wants_amplitude {
            return Ok(Vec::new());
        }

        let mut volume_return_values = Vec::new();
        let mut volume_relative_values = Vec::new();
        let mut amplitude_values = Vec::new();
        for trade_date in &context.target_dates {
            let Some(table) = data.minute(DatasetId::StockMinute1m, *trade_date) else {
                continue;
            };
            let ts_codes = table.required_utf8("ts_code")?;
            let trade_times = table.required_utf8("trade_time")?;
            let close = table.required_f64_cast("close")?;
            let high = table.required_f64_cast("high")?;
            let low = table.required_f64_cast("low")?;
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

            for (ts_code, mut indices) in grouped {
                indices.sort_by(|left, right| trade_times[*left].cmp(&trade_times[*right]));
                let game_values =
                    game_values_from_rows(&indices, trade_times, &close, &high, &low, &volume);
                if wants_volume_return {
                    volume_return_values.push(FactorValue {
                        key: FactorRowKey::Daily {
                            trade_date: *trade_date,
                            ts_code: ts_code.clone(),
                        },
                        value: game_values.volume_return,
                    });
                }
                if wants_volume_relative {
                    volume_relative_values.push(FactorValue {
                        key: FactorRowKey::Daily {
                            trade_date: *trade_date,
                            ts_code: ts_code.clone(),
                        },
                        value: game_values.volume_relative,
                    });
                }
                if wants_amplitude {
                    amplitude_values.push(FactorValue {
                        key: FactorRowKey::Daily {
                            trade_date: *trade_date,
                            ts_code,
                        },
                        value: game_values.amplitude,
                    });
                }
            }
        }

        let mut output = Vec::new();
        if wants_volume_return {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(VOLUME_GAME_RETURN_RAW_ID),
                values: volume_return_values,
            });
        }
        if wants_volume_relative {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(VOLUME_GAME_RELATIVE_RAW_ID),
                values: volume_relative_values,
            });
        }
        if wants_amplitude {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(AMPLITUDE_GAME_RAW_ID),
                values: amplitude_values,
            });
        }
        Ok(output)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(VOLUME_GAME_RETURN_RAW_ID)?;
        let volume_return = panel.column(VOLUME_GAME_RETURN_RAW_ID)?.cs(cs_demean_abs)?;
        let volume_relative = panel
            .column(VOLUME_GAME_RELATIVE_RAW_ID)?
            .cs(cs_demean_abs)?;
        let amplitude = panel.column(AMPLITUDE_GAME_RAW_ID)?.cs(cs_demean_abs)?;

        let volume_return_component = rolling_game_component(&volume_return)?;
        let volume_relative_component = rolling_game_component(&volume_relative)?;
        let amplitude_component = rolling_game_component(&amplitude)?;

        let volume_game = average_pair(
            &volume_return_component.cs(cs_zscore)?,
            &volume_relative_component.cs(cs_zscore)?,
        )?;
        let factor = average_pair(
            &volume_game.cs(cs_zscore)?,
            &amplitude_component.cs(cs_zscore)?,
        )?;

        Ok(factor.to_factor_series(self.spec()))
    }
}

fn rolling_game_component(
    values: &crate::factor::common::PanelColumn,
) -> Result<crate::factor::common::PanelColumn> {
    let mean20 = values.ts(|series| ts_mean(series, WINDOW, 1))?;
    let std20 = values.ts(|series| ts_std_dev(series, WINDOW, 1))?;
    average_pair(&mean20.cs(cs_zscore)?, &std20.cs(cs_zscore)?)
}

fn game_values_from_rows(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
    high: &[Option<f64>],
    low: &[Option<f64>],
    volume: &[Option<f64>],
) -> GameValues {
    let times = indices
        .iter()
        .map(|idx| trade_times[*idx].clone())
        .collect::<Vec<_>>();
    let close_series = indices
        .iter()
        .map(|idx| clean_intraday_value(close[*idx]))
        .collect::<Vec<_>>();
    let high_series = indices
        .iter()
        .map(|idx| clean_intraday_value(high[*idx]))
        .collect::<Vec<_>>();
    let low_series = indices
        .iter()
        .map(|idx| clean_intraday_value(low[*idx]))
        .collect::<Vec<_>>();
    let volume_series = indices
        .iter()
        .map(|idx| clean_intraday_value(volume[*idx]))
        .collect::<Vec<_>>();
    game_values_from_series(
        &times,
        &close_series,
        &high_series,
        &low_series,
        &volume_series,
    )
}

fn game_values_from_series(
    trade_times: &[Option<String>],
    close: &[Option<f64>],
    high: &[Option<f64>],
    low: &[Option<f64>],
    volume: &[Option<f64>],
) -> GameValues {
    let return5 = five_minute_returns(close);
    let relative_position = relative_positions(trade_times, close);

    let mut volume_return_pairs = Vec::new();
    let mut volume_relative_pairs = Vec::new();
    let mut amplitude_pairs = Vec::new();

    for idx in 0..trade_times.len() {
        let Some(time) = trade_times[idx].as_deref() else {
            continue;
        };
        if !intraday_time_in_range(time, "09:36:00", "14:57:00") {
            continue;
        }

        if let (Some(metric), Some(payload)) = (clean(return5[idx]), clean(volume[idx])) {
            volume_return_pairs.push(RankedPayload {
                time_idx: idx,
                metric,
                payload,
            });
        }

        if let (Some(metric), Some(payload)) = (clean(relative_position[idx]), clean(volume[idx])) {
            volume_relative_pairs.push(RankedPayload {
                time_idx: idx,
                metric,
                payload,
            });
        }

        if let (Some(metric), Some(payload)) =
            (clean(return5[idx]), amplitude_at(idx, high, low, close))
        {
            amplitude_pairs.push(RankedPayload {
                time_idx: idx,
                metric,
                payload,
            });
        }
    }

    GameValues {
        volume_return: cumulative_sorted_game(&volume_return_pairs),
        volume_relative: cumulative_sorted_game(&volume_relative_pairs),
        amplitude: cumulative_sorted_game(&amplitude_pairs),
    }
}

fn five_minute_returns(close: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut output = vec![None; close.len()];
    for idx in RETURN_LAG..close.len() {
        let (Some(current), Some(previous)) = (clean(close[idx]), clean(close[idx - RETURN_LAG]))
        else {
            continue;
        };
        if previous.abs() <= f64::EPSILON {
            continue;
        }
        output[idx] = Some(current / previous - 1.0);
    }
    output
}

fn relative_positions(trade_times: &[Option<String>], close: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut output = vec![None; close.len()];
    let mut running_max: Option<f64> = None;
    let mut running_min: Option<f64> = None;
    for idx in 0..close.len() {
        let Some(time) = trade_times[idx].as_deref() else {
            continue;
        };
        if !intraday_time_in_range(time, "09:36:00", "14:57:00") {
            continue;
        }
        let Some(value) = clean(close[idx]) else {
            continue;
        };
        if value.abs() <= f64::EPSILON {
            continue;
        }
        running_max = Some(running_max.map_or(value, |current| current.max(value)));
        running_min = Some(running_min.map_or(value, |current| current.min(value)));
        let (Some(high_water), Some(low_water)) = (running_max, running_min) else {
            continue;
        };
        if high_water.abs() <= f64::EPSILON || low_water.abs() <= f64::EPSILON {
            continue;
        }
        output[idx] = Some(((value / high_water - 1.0) + (value / low_water - 1.0)) / 2.0);
    }
    output
}

fn amplitude_at(
    idx: usize,
    high: &[Option<f64>],
    low: &[Option<f64>],
    close: &[Option<f64>],
) -> Option<f64> {
    let (Some(high), Some(low), Some(close)) =
        (clean(high[idx]), clean(low[idx]), clean(close[idx]))
    else {
        return None;
    };
    if close.abs() <= f64::EPSILON {
        return None;
    }
    Some((high - low) / close)
}

fn cumulative_sorted_game(values: &[RankedPayload]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let mut ascending = values.to_vec();
    ascending.sort_by(compare_metric_ascending);
    let mut descending = values.to_vec();
    descending.sort_by(compare_metric_descending);

    let mut ascending_cumulative = 0.0;
    let mut descending_cumulative = 0.0;
    let mut output = 0.0;
    for idx in 0..values.len() {
        ascending_cumulative += ascending[idx].payload;
        descending_cumulative += descending[idx].payload;
        output += ascending_cumulative - descending_cumulative;
    }
    Some(output)
}

fn compare_metric_ascending(left: &RankedPayload, right: &RankedPayload) -> Ordering {
    left.metric
        .partial_cmp(&right.metric)
        .unwrap_or(Ordering::Equal)
        .then_with(|| left.time_idx.cmp(&right.time_idx))
}

fn compare_metric_descending(left: &RankedPayload, right: &RankedPayload) -> Ordering {
    right
        .metric
        .partial_cmp(&left.metric)
        .unwrap_or(Ordering::Equal)
        .then_with(|| left.time_idx.cmp(&right.time_idx))
}

fn average_pair(
    left: &crate::factor::common::PanelColumn,
    right: &crate::factor::common::PanelColumn,
) -> Result<crate::factor::common::PanelColumn> {
    left.zip_binary(right, |left, right| match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some((left + right) / 2.0),
        _ => None,
    })
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| !value.is_nan())
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

    fn assert_close(actual: Option<f64>, expected: Option<f64>) {
        match (actual, expected) {
            (Some(actual), Some(expected)) => assert!((actual - expected).abs() < 1e-10),
            (None, None) => {}
            _ => panic!("expected {:?}, got {:?}", expected, actual),
        }
    }

    #[test]
    fn five_minute_return_is_computed_before_time_filter() {
        let times = vec![
            time("09:31:00"),
            time("09:32:00"),
            time("09:33:00"),
            time("09:34:00"),
            time("09:35:00"),
            time("09:36:00"),
        ];
        let close = vec![
            Some(10.0),
            Some(20.0),
            Some(20.0),
            Some(20.0),
            Some(20.0),
            Some(22.0),
        ];
        let high = close.clone();
        let low = close.clone();
        let volume = vec![Some(1.0); 6];

        let output = game_values_from_series(&times, &close, &high, &low, &volume);

        assert!(output.volume_return.is_none());
        assert_close(five_minute_returns(&close)[5], Some(0.1));
    }

    #[test]
    fn cumulative_game_uses_true_metric_sort_and_time_ties() {
        let values = vec![
            RankedPayload {
                time_idx: 0,
                metric: 2.0,
                payload: 10.0,
            },
            RankedPayload {
                time_idx: 1,
                metric: 1.0,
                payload: 20.0,
            },
            RankedPayload {
                time_idx: 2,
                metric: 2.0,
                payload: 30.0,
            },
        ];

        assert_close(cumulative_sorted_game(&values), Some(0.0));
    }

    #[test]
    fn game_values_match_small_sample() {
        let times = vec![
            time("09:31:00"),
            time("09:32:00"),
            time("09:33:00"),
            time("09:34:00"),
            time("09:35:00"),
            time("09:36:00"),
            time("09:37:00"),
            time("09:38:00"),
        ];
        let close = vec![
            Some(10.0),
            Some(10.0),
            Some(10.0),
            Some(10.0),
            Some(10.0),
            Some(11.0),
            Some(9.0),
            Some(12.0),
        ];
        let high = vec![
            Some(10.0),
            Some(10.0),
            Some(10.0),
            Some(10.0),
            Some(10.0),
            Some(12.0),
            Some(10.0),
            Some(14.0),
        ];
        let low = vec![
            Some(10.0),
            Some(10.0),
            Some(10.0),
            Some(10.0),
            Some(10.0),
            Some(10.0),
            Some(8.0),
            Some(10.0),
        ];
        let volume = vec![
            Some(0.0),
            Some(0.0),
            Some(0.0),
            Some(0.0),
            Some(0.0),
            Some(10.0),
            Some(20.0),
            Some(30.0),
        ];

        let output = game_values_from_series(&times, &close, &high, &low, &volume);

        assert_close(output.volume_return, Some(-20.0));
        assert_close(output.volume_relative, Some(-20.0));
        assert_close(output.amplitude, Some(-22.0 / 99.0));
    }

    #[test]
    fn rolling_component_uses_distance_then_twenty_day_mean_and_std() {
        let mut trade_dates = Vec::new();
        let mut ts_codes = Vec::new();
        let mut raw = Vec::new();
        for idx in 0..20 {
            trade_dates.push(Some(20260101 + idx));
            ts_codes.push(Some("a".to_string()));
            raw.push(Some(idx as f64));
            trade_dates.push(Some(20260101 + idx));
            ts_codes.push(Some("b".to_string()));
            raw.push(Some((idx + 2) as f64));
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
        let distance = panel
            .column("raw")
            .expect("raw")
            .cs(cs_demean_abs)
            .expect("distance");
        let component = rolling_game_component(&distance).expect("component");

        assert!(component.values().iter().all(Option::is_none));
    }
}
