use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawAuxiliaryRequest, IntradayDailyRawRequest,
    IntradayDailyRawSeries, IntradayDailyRawSpec, Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::{
    clean_intraday_value, intraday_time_in_range, stock_minute_raw_spec, vector::clean, DailyPanel,
    PanelColumn,
};
use crate::factor::Factor;
use crate::operators::{cs_zscore, ts_delay};

const AM_RETURN_RAW_ID: &str = "mom1430_daily_am_return";
const PM_RETURN_RAW_ID: &str = "mom1430_daily_pm_return";
const AM_SMART_TURNOVER_RAW_ID: &str = "mom1430_daily_am_smart_turnover";
const PM_SMART_TURNOVER_RAW_ID: &str = "mom1430_daily_pm_smart_turnover";
const LAST30_TURNOVER_RAW_ID: &str = "mom1430_daily_last30m_turnover";
const RAW_VERSION: &str = "0.1.0";
const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;
const GROUP_COUNT: usize = 5;
const GROUP_SIZE: usize = WINDOW / GROUP_COUNT;
const FLOAT_SHARE_UNIT: f64 = 10_000.0;
const SMART_MINUTES: usize = 24;

pub struct StockDailyMom1430Smart;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyMom1430Smart)
}

fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
    stock_minute_raw_spec(raw_id, RAW_VERSION, &["open", "close", "vol"], 1)
}

impl Factor for StockDailyMom1430Smart {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "mom_1430_smart".to_string(),
            aliases: vec![
                "MOM_1430_SMART".to_string(),
                "MOM 1430 Smart".to_string(),
            ],
            name: "MOM 1430 Smart".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "return",
                "momentum",
                "turnover",
                "smart",
                "intraday",
                "overnight_return",
                "minute_agg",
                "composite",
                "daily",
                "DWZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Smart momentum factor combining AM/PM smart-turnover sorted intraday returns and previous-day last-30-minute-turnover sorted overnight returns.".to_string(),
            dependencies: vec![DataRequest::new(
                DatasetId::StockDailyPv,
                &["open", "pre_close"],
            )],
            intraday_raw_dependencies: vec![
                IntradayDailyRawRequest::new(AM_RETURN_RAW_ID, WINDOW),
                IntradayDailyRawRequest::new(PM_RETURN_RAW_ID, WINDOW),
                IntradayDailyRawRequest::new(AM_SMART_TURNOVER_RAW_ID, WINDOW),
                IntradayDailyRawRequest::new(PM_SMART_TURNOVER_RAW_ID, WINDOW),
                IntradayDailyRawRequest::new(LAST30_TURNOVER_RAW_ID, WINDOW),
            ],
            lookback: Lookback {
                trading_days: WINDOW,
            },
        }
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        vec![
            raw_spec(AM_RETURN_RAW_ID),
            raw_spec(PM_RETURN_RAW_ID),
            raw_spec(AM_SMART_TURNOVER_RAW_ID),
            raw_spec(PM_SMART_TURNOVER_RAW_ID),
            raw_spec(LAST30_TURNOVER_RAW_ID),
        ]
    }

    fn intraday_raw_auxiliary_requirements(
        &self,
        raw_ids: &[String],
    ) -> Vec<IntradayDailyRawAuxiliaryRequest> {
        if raw_ids.iter().any(|raw_id| {
            raw_id == AM_SMART_TURNOVER_RAW_ID
                || raw_id == PM_SMART_TURNOVER_RAW_ID
                || raw_id == LAST30_TURNOVER_RAW_ID
        }) {
            vec![IntradayDailyRawAuxiliaryRequest::new(
                DataRequest::new(DatasetId::StockDailyBasic, &["float_share"]),
                0,
            )]
        } else {
            Vec::new()
        }
    }

    fn minute_compute_many(
        &self,
        raw_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<IntradayDailyRawSeries>> {
        let requested = raw_ids.iter().map(String::as_str).collect::<BTreeSet<_>>();
        let wants_am_return = requested.contains(AM_RETURN_RAW_ID);
        let wants_pm_return = requested.contains(PM_RETURN_RAW_ID);
        let wants_am_smart_turnover = requested.contains(AM_SMART_TURNOVER_RAW_ID);
        let wants_pm_smart_turnover = requested.contains(PM_SMART_TURNOVER_RAW_ID);
        let wants_last30_turnover = requested.contains(LAST30_TURNOVER_RAW_ID);
        if !wants_am_return
            && !wants_pm_return
            && !wants_am_smart_turnover
            && !wants_pm_smart_turnover
            && !wants_last30_turnover
        {
            return Ok(Vec::new());
        }

        let float_share =
            if wants_am_smart_turnover || wants_pm_smart_turnover || wants_last30_turnover {
                let basic_panel = data.daily_panel(DatasetId::StockDailyBasic)?;
                Some(panel_column_map(
                    basic_panel,
                    &basic_panel.column("float_share")?,
                ))
            } else {
                None
            };

        let mut am_return_values = Vec::new();
        let mut pm_return_values = Vec::new();
        let mut am_smart_turnover_values = Vec::new();
        let mut pm_smart_turnover_values = Vec::new();
        let mut last30_turnover_values = Vec::new();
        for trade_date in &context.target_dates {
            let Some(table) = data.minute(DatasetId::StockMinute1m, *trade_date) else {
                continue;
            };
            let ts_codes = table.required_utf8("ts_code")?;
            let trade_times = table.required_utf8("trade_time")?;
            let open = table.required_f64_cast("open")?;
            let close = table.required_f64_cast("close")?;
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
                let share = float_share
                    .as_ref()
                    .and_then(|map| map.get(&(*trade_date, ts_code.clone())))
                    .copied()
                    .flatten();
                let key = FactorRowKey::Daily {
                    trade_date: *trade_date,
                    ts_code: ts_code.clone(),
                };
                if wants_am_return {
                    am_return_values.push(FactorValue {
                        key: key.clone(),
                        value: session_close_open_return(
                            &indices,
                            trade_times,
                            &open,
                            &close,
                            "09:31:00",
                            "11:30:00",
                        ),
                    });
                }
                if wants_pm_return {
                    pm_return_values.push(FactorValue {
                        key: key.clone(),
                        value: session_close_open_return(
                            &indices,
                            trade_times,
                            &open,
                            &close,
                            "13:01:00",
                            "15:00:00",
                        ),
                    });
                }
                if wants_am_smart_turnover {
                    am_smart_turnover_values.push(FactorValue {
                        key: key.clone(),
                        value: smart_turnover(
                            &indices,
                            trade_times,
                            &open,
                            &close,
                            &volume,
                            share,
                            "09:31:00",
                            "11:30:00",
                        ),
                    });
                }
                if wants_pm_smart_turnover {
                    pm_smart_turnover_values.push(FactorValue {
                        key: key.clone(),
                        value: smart_turnover(
                            &indices,
                            trade_times,
                            &open,
                            &close,
                            &volume,
                            share,
                            "13:01:00",
                            "15:00:00",
                        ),
                    });
                }
                if wants_last30_turnover {
                    last30_turnover_values.push(FactorValue {
                        key,
                        value: window_turnover(
                            &indices,
                            trade_times,
                            &volume,
                            share,
                            "14:30:00",
                            "15:00:00",
                        ),
                    });
                }
            }
        }

        let mut output = Vec::new();
        if wants_am_return {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(AM_RETURN_RAW_ID),
                values: am_return_values,
            });
        }
        if wants_pm_return {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(PM_RETURN_RAW_ID),
                values: pm_return_values,
            });
        }
        if wants_am_smart_turnover {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(AM_SMART_TURNOVER_RAW_ID),
                values: am_smart_turnover_values,
            });
        }
        if wants_pm_smart_turnover {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(PM_SMART_TURNOVER_RAW_ID),
                values: pm_smart_turnover_values,
            });
        }
        if wants_last30_turnover {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(LAST30_TURNOVER_RAW_ID),
                values: last30_turnover_values,
            });
        }
        Ok(output)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(AM_RETURN_RAW_ID)?;
        let pv_table = data.daily(DatasetId::StockDailyPv)?;
        let open = panel.column_from_table(pv_table, "open")?;
        let pre_close = panel.column_from_table(pv_table, "pre_close")?;
        let overnight_return = open.zip_binary(&pre_close, ret)?;
        let last30_turnover = panel
            .column(LAST30_TURNOVER_RAW_ID)?
            .ts(|values| ts_delay(values, 1))?;
        let night_part1 = rolling_group_mean(&overnight_return, &last30_turnover, 0)?;
        let night_part5 = rolling_group_mean(&overnight_return, &last30_turnover, 4)?;
        let night_1430 = weighted_pair(
            &night_part1.cs(cs_zscore)?,
            &night_part5.cs(cs_zscore)?,
            1.0,
            -1.0,
        )?;

        let am_return = panel.column(AM_RETURN_RAW_ID)?;
        let am_smart_turnover = panel.column(AM_SMART_TURNOVER_RAW_ID)?;
        let am_part1 = rolling_group_mean(&am_return, &am_smart_turnover, 0)?;
        let am_part5 = rolling_group_mean(&am_return, &am_smart_turnover, 4)?;
        let am_smart = weighted_pair(
            &am_part1.cs(cs_zscore)?,
            &am_part5.cs(cs_zscore)?,
            -1.0,
            1.0,
        )?;

        let pm_return = panel.column(PM_RETURN_RAW_ID)?;
        let pm_smart_turnover = panel.column(PM_SMART_TURNOVER_RAW_ID)?;
        let pm_part1 = rolling_group_mean(&pm_return, &pm_smart_turnover, 0)?;
        let pm_part5 = rolling_group_mean(&pm_return, &pm_smart_turnover, 4)?;
        let pm_smart = weighted_pair(
            &pm_part1.cs(cs_zscore)?,
            &pm_part5.cs(cs_zscore)?,
            -1.0,
            1.0,
        )?;

        let day_smart = add_pair(&am_smart.cs(cs_zscore)?, &pm_smart.cs(cs_zscore)?)?;
        let factor = add_pair(&day_smart.cs(cs_zscore)?, &night_1430.cs(cs_zscore)?)?;
        Ok(factor.to_factor_series(self.spec()))
    }
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

fn session_close_open_return(
    indices: &[usize],
    trade_times: &[Option<String>],
    open: &[Option<f64>],
    close: &[Option<f64>],
    start_time: &str,
    end_time: &str,
) -> Option<f64> {
    let mut start_open = None;
    let mut end_close = None;
    for idx in indices {
        let Some(trade_time) = trade_times[*idx].as_deref() else {
            continue;
        };
        if intraday_time_in_range(trade_time, start_time, start_time) {
            start_open = clean_intraday_value(open[*idx]);
        }
        if intraday_time_in_range(trade_time, end_time, end_time) {
            end_close = clean_intraday_value(close[*idx]);
        }
    }
    match (end_close, start_open) {
        (Some(close), Some(open)) if open.abs() > f64::EPSILON => Some(close / open - 1.0),
        _ => None,
    }
}

fn smart_turnover(
    indices: &[usize],
    trade_times: &[Option<String>],
    open: &[Option<f64>],
    close: &[Option<f64>],
    volume: &[Option<f64>],
    float_share: Option<f64>,
    start_time: &str,
    end_time: &str,
) -> Option<f64> {
    let denominator = float_share_denominator(float_share)?;
    let mut candidates = Vec::<(usize, f64, f64)>::new();
    for pos in 0..indices.len() {
        let idx = indices[pos];
        let Some(trade_time) = trade_times[idx].as_deref() else {
            continue;
        };
        if !intraday_time_in_range(trade_time, start_time, end_time) {
            continue;
        }
        let Some(smart) = smart_score(open[idx], close[idx], volume[idx]) else {
            continue;
        };
        let Some(volume) = clean_intraday_value(volume[idx]) else {
            continue;
        };
        candidates.push((pos, smart, volume));
    }
    if candidates.len() < SMART_MINUTES {
        return None;
    }
    candidates.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
    let selected_volume = candidates
        .iter()
        .take(SMART_MINUTES)
        .map(|(_, _, volume)| *volume)
        .sum::<f64>();
    Some(selected_volume / denominator)
}

fn smart_score(open: Option<f64>, close: Option<f64>, volume: Option<f64>) -> Option<f64> {
    match (
        clean_intraday_value(open),
        clean_intraday_value(close),
        clean_intraday_value(volume),
    ) {
        (Some(open), Some(close), Some(volume)) if open.abs() > f64::EPSILON && volume > 0.0 => {
            Some((close / open - 1.0).abs() / volume.sqrt())
        }
        _ => None,
    }
}

fn window_turnover(
    indices: &[usize],
    trade_times: &[Option<String>],
    volume: &[Option<f64>],
    float_share: Option<f64>,
    start_time: &str,
    end_time: &str,
) -> Option<f64> {
    let denominator = float_share_denominator(float_share)?;
    let mut total_volume = 0.0;
    let mut count = 0usize;
    for idx in indices {
        let Some(trade_time) = trade_times[*idx].as_deref() else {
            continue;
        };
        if !intraday_time_in_range(trade_time, start_time, end_time) {
            continue;
        }
        let Some(volume) = clean_intraday_value(volume[*idx]) else {
            continue;
        };
        total_volume += volume;
        count += 1;
    }
    (count > 0).then_some(total_volume / denominator)
}

fn float_share_denominator(float_share: Option<f64>) -> Option<f64> {
    let float_share = clean(float_share)?;
    if float_share <= 0.0 {
        return None;
    }
    let denominator = float_share * FLOAT_SHARE_UNIT;
    (denominator > f64::EPSILON).then_some(denominator)
}

fn rolling_group_mean(
    returns: &PanelColumn,
    sort_values: &PanelColumn,
    group_idx: usize,
) -> Result<PanelColumn> {
    returns.ts_binary(sort_values, |returns, sort_values| {
        grouped_part_series(returns, sort_values, group_idx)
    })
}

fn grouped_part_series(
    returns: &[Option<f64>],
    sort_values: &[Option<f64>],
    group_idx: usize,
) -> Vec<Option<f64>> {
    let mut output = vec![None; returns.len()];
    if group_idx >= GROUP_COUNT {
        return output;
    }

    for idx in 0..returns.len() {
        if idx + 1 < WINDOW {
            continue;
        }
        let start = idx + 1 - WINDOW;
        let mut pairs = Vec::<(f64, usize, f64)>::with_capacity(WINDOW);
        for window_idx in start..=idx {
            let (Some(return_value), Some(sort_value)) =
                (clean(returns[window_idx]), clean(sort_values[window_idx]))
            else {
                continue;
            };
            pairs.push((sort_value, window_idx, return_value));
        }
        if pairs.len() != WINDOW {
            continue;
        }
        pairs.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        let group_start = group_idx * GROUP_SIZE;
        let group_end = group_start + GROUP_SIZE;
        let sum = pairs[group_start..group_end]
            .iter()
            .map(|(_, _, return_value)| *return_value)
            .sum::<f64>();
        output[idx] = Some(sum / GROUP_SIZE as f64);
    }
    output
}

fn ret(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (clean(numerator), clean(denominator)) {
        (Some(numerator), Some(denominator)) if denominator.abs() > f64::EPSILON => {
            Some(numerator / denominator - 1.0)
        }
        _ => None,
    }
}

fn weighted_pair(
    left: &PanelColumn,
    right: &PanelColumn,
    left_weight: f64,
    right_weight: f64,
) -> Result<PanelColumn> {
    left.zip_binary(right, |left, right| match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some(left * left_weight + right * right_weight),
        _ => None,
    })
}

fn add_pair(left: &PanelColumn, right: &PanelColumn) -> Result<PanelColumn> {
    weighted_pair(left, right, 1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("expected value");
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn grouped_part_series_sorts_twenty_days_into_equal_quintiles() {
        let returns = (0..20).map(|idx| Some(idx as f64)).collect::<Vec<_>>();
        let sort_values = (0..20)
            .rev()
            .map(|idx| Some(idx as f64))
            .collect::<Vec<_>>();

        let low = grouped_part_series(&returns, &sort_values, 0);
        let high = grouped_part_series(&returns, &sort_values, 4);

        assert_close(low[19], 17.5);
        assert_close(high[19], 1.5);
    }

    #[test]
    fn grouped_part_series_requires_twenty_valid_pairs() {
        let returns = vec![Some(1.0); 20];
        let mut sort_values = vec![Some(1.0); 20];
        sort_values[3] = None;

        let output = grouped_part_series(&returns, &sort_values, 0);

        assert_eq!(output[19], None);
    }

    #[test]
    fn session_return_uses_exact_start_open_and_end_close() {
        let indices = vec![0, 1, 2, 3];
        let times = vec![
            Some("09:31:00".to_string()),
            Some("10:00:00".to_string()),
            Some("11:30:00".to_string()),
            Some("13:01:00".to_string()),
        ];
        let open = vec![Some(10.0), Some(99.0), Some(99.0), Some(20.0)];
        let close = vec![Some(99.0), Some(99.0), Some(11.0), Some(99.0)];

        assert_close(
            session_close_open_return(&indices, &times, &open, &close, "09:31:00", "11:30:00"),
            0.1,
        );
    }

    #[test]
    fn smart_score_uses_bar_close_over_open_return() {
        assert_close(smart_score(Some(10.0), Some(11.0), Some(100.0)), 0.01);
        assert_eq!(smart_score(Some(0.0), Some(11.0), Some(100.0)), None);
        assert_eq!(smart_score(Some(10.0), Some(11.0), Some(0.0)), None);
    }

    #[test]
    fn window_turnover_uses_float_share_in_ten_thousand_shares() {
        let indices = vec![0, 1, 2];
        let times = vec![
            Some("14:29:00".to_string()),
            Some("14:30:00".to_string()),
            Some("15:00:00".to_string()),
        ];
        let volume = vec![Some(999.0), Some(1_000.0), Some(2_000.0)];

        assert_eq!(
            window_turnover(&indices, &times, &volume, Some(1.0), "14:30:00", "15:00:00"),
            Some(0.3),
        );
    }

    #[test]
    fn smart_turnover_requires_twenty_four_valid_minutes() {
        let mut indices = Vec::new();
        let mut times = Vec::new();
        let mut open = Vec::new();
        let mut close = Vec::new();
        let mut volume = Vec::new();
        for minute in 0..25 {
            indices.push(minute);
            times.push(Some(format!("09:{:02}:00", minute + 31)));
            open.push(Some(10.0));
            close.push(Some(10.0 + minute as f64 / 100.0));
            volume.push(Some(100.0 + minute as f64));
        }

        let value = smart_turnover(
            &indices,
            &times,
            &open,
            &close,
            &volume,
            Some(1.0),
            "09:31:00",
            "11:30:00",
        );

        assert!(value.is_some());
    }

    #[test]
    fn return_rejects_zero_denominator() {
        assert_close(ret(Some(11.0), Some(10.0)), 0.1);
        assert_eq!(ret(Some(11.0), Some(0.0)), None);
    }
}
