use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawAuxiliaryRequest, IntradayDailyRawRequest,
    IntradayDailyRawSeries, IntradayDailyRawSpec, Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::common::{
    clean_intraday_value, intraday_time_in_range, stock_minute_raw_spec, DailyPanel, PanelColumn,
};
use crate::factor::Factor;
use crate::operators::ts_corr;

pub const PM_CO_RAW_ID: &str = "daily_pm_co";
pub const PM_SMART_TURNOVER_RAW_ID: &str = "daily_pm_smart_turnover";
pub const LAST30_TURNOVER_RAW_ID: &str = "daily_last30m_turnover";

const RAW_VERSION: &str = "0.1.0";
const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;
const FLOAT_SHARE_UNIT: f64 = 10_000.0;
const SMART_MINUTES: usize = 24;

pub struct StockDailySccoiv;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailySccoiv)
}

fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
    stock_minute_raw_spec(raw_id, RAW_VERSION, &["open", "close", "vol"], 1)
}

impl Factor for StockDailySccoiv {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "sccoiv".to_string(),
            aliases: vec!["SCCOIV".to_string()],
            name: "SCCOIV".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "turnover",
                "price",
                "correlation",
                "smart",
                "intraday",
                "minute_agg",
                "daily",
                "DWZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Smart intraday price-volume correlation between afternoon price change and afternoon smart-minute turnover.".to_string(),
            dependencies: Vec::new(),
            intraday_raw_dependencies: vec![
                IntradayDailyRawRequest::new(PM_CO_RAW_ID, WINDOW),
                IntradayDailyRawRequest::new(PM_SMART_TURNOVER_RAW_ID, WINDOW),
            ],
            lookback: Lookback {
                trading_days: WINDOW,
            },
        }
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        vec![
            raw_spec(PM_CO_RAW_ID),
            raw_spec(PM_SMART_TURNOVER_RAW_ID),
            raw_spec(LAST30_TURNOVER_RAW_ID),
        ]
    }

    fn intraday_raw_auxiliary_requirements(
        &self,
        raw_ids: &[String],
    ) -> Vec<IntradayDailyRawAuxiliaryRequest> {
        if raw_ids
            .iter()
            .any(|raw_id| raw_id == PM_SMART_TURNOVER_RAW_ID || raw_id == LAST30_TURNOVER_RAW_ID)
        {
            vec![IntradayDailyRawAuxiliaryRequest::new(
                DataRequest::new(DatasetId::StockDailyBasic, &["float_share"]),
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
        let wants_pm_co = requested.contains(PM_CO_RAW_ID);
        let wants_pm_smart_turnover = requested.contains(PM_SMART_TURNOVER_RAW_ID);
        let wants_last30_turnover = requested.contains(LAST30_TURNOVER_RAW_ID);
        if !wants_pm_co && !wants_pm_smart_turnover && !wants_last30_turnover {
            return Ok(Vec::new());
        }

        let float_share = if wants_pm_smart_turnover || wants_last30_turnover {
            let basic_panel = data.daily_panel(DatasetId::StockDailyBasic)?;
            Some(panel_column_map(
                basic_panel,
                &basic_panel.column("float_share")?,
            ))
        } else {
            None
        };

        let mut pm_co_values = Vec::new();
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
                if wants_pm_co {
                    pm_co_values.push(FactorValue {
                        key: key.clone(),
                        value: afternoon_close_open_change(&indices, trade_times, &open, &close),
                    });
                }
                if wants_pm_smart_turnover {
                    pm_smart_turnover_values.push(FactorValue {
                        key: key.clone(),
                        value: afternoon_smart_turnover(
                            &indices,
                            trade_times,
                            &close,
                            &volume,
                            share,
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
        if wants_pm_co {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(PM_CO_RAW_ID),
                values: pm_co_values,
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
        let panel = data.intraday_daily_raw_panel(PM_CO_RAW_ID)?;
        let pm_co = panel.column(PM_CO_RAW_ID)?;
        let pm_smart_turnover = panel.column(PM_SMART_TURNOVER_RAW_ID)?;
        let factor =
            pm_co.ts_binary(&pm_smart_turnover, |co, sv| ts_corr(co, sv, WINDOW, WINDOW))?;
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

fn afternoon_close_open_change(
    indices: &[usize],
    trade_times: &[Option<String>],
    open: &[Option<f64>],
    close: &[Option<f64>],
) -> Option<f64> {
    let mut open_1301 = None;
    let mut close_1500 = None;
    for idx in indices {
        let Some(trade_time) = trade_times[*idx].as_deref() else {
            continue;
        };
        if intraday_time_in_range(trade_time, "13:01:00", "13:01:00") {
            open_1301 = clean_intraday_value(open[*idx]);
        }
        if intraday_time_in_range(trade_time, "15:00:00", "15:00:00") {
            close_1500 = clean_intraday_value(close[*idx]);
        }
    }
    match (close_1500, open_1301) {
        (Some(close), Some(open)) => Some(close - open),
        _ => None,
    }
}

fn afternoon_smart_turnover(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
    volume: &[Option<f64>],
    float_share: Option<f64>,
) -> Option<f64> {
    let denominator = float_share_denominator(float_share)?;
    let mut candidates = Vec::<(usize, f64, f64)>::new();
    for pos in 0..indices.len() {
        let idx = indices[pos];
        let Some(trade_time) = trade_times[idx].as_deref() else {
            continue;
        };
        if !intraday_time_in_range(trade_time, "13:01:00", "15:00:00") {
            continue;
        }
        let Some(current_close) = clean_intraday_value(close[idx]) else {
            continue;
        };
        if pos == 0 {
            continue;
        }
        let Some(previous_close) = clean_intraday_value(close[indices[pos - 1]]) else {
            continue;
        };
        if previous_close.abs() <= f64::EPSILON {
            continue;
        }
        let Some(volume) = clean_intraday_value(volume[idx]) else {
            continue;
        };
        if volume <= 0.0 {
            continue;
        }
        let minute_return = current_close / previous_close - 1.0;
        let smart = minute_return.abs() / volume.sqrt();
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

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Option<f64>, expected: Option<f64>) {
        match (actual, expected) {
            (Some(actual), Some(expected)) => assert!(
                (actual - expected).abs() < 1e-10,
                "expected {expected}, got {actual}"
            ),
            (None, None) => {}
            _ => panic!("expected {:?}, got {:?}", expected, actual),
        }
    }

    #[test]
    fn afternoon_co_uses_1301_open_and_1500_close() {
        let indices = vec![0, 1, 2];
        let times = vec![
            Some("13:01:00".to_string()),
            Some("14:00:00".to_string()),
            Some("15:00:00".to_string()),
        ];
        let open = vec![Some(10.0), Some(11.0), Some(12.0)];
        let close = vec![Some(10.5), Some(11.5), Some(12.25)];

        assert_close(
            afternoon_close_open_change(&indices, &times, &open, &close),
            Some(2.25),
        );
    }

    #[test]
    fn last30_turnover_uses_float_share_in_ten_thousand_shares() {
        let indices = vec![0, 1, 2];
        let times = vec![
            Some("14:29:00".to_string()),
            Some("14:30:00".to_string()),
            Some("15:00:00".to_string()),
        ];
        let volume = vec![Some(999.0), Some(1_000.0), Some(2_000.0)];

        assert_close(
            window_turnover(&indices, &times, &volume, Some(1.0), "14:30:00", "15:00:00"),
            Some(0.3),
        );
    }

    #[test]
    fn afternoon_smart_turnover_requires_top_twenty_four_valid_minutes() {
        let mut indices = Vec::new();
        let mut times = Vec::new();
        let mut close = Vec::new();
        let mut volume = Vec::new();
        for minute in 0..25 {
            indices.push(minute);
            times.push(Some(format!("13:{:02}:00", minute + 1)));
            close.push(Some(10.0 + minute as f64));
            volume.push(Some(100.0 + minute as f64));
        }

        let value = afternoon_smart_turnover(&indices, &times, &close, &volume, Some(1.0));

        assert!(value.is_some());
    }
}
