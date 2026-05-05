use std::collections::BTreeMap;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::common::{intraday_time_in_range, stock_minute_raw_spec, PanelColumn};
use crate::factor::Factor;
use crate::operators::{cs_regression_residual, cs_zscore, ts_mean, ts_std_dev};

pub const CP_INTRADAY_RAW_ID: &str = "daily_cp_intraday";

const RAW_VERSION: &str = "0.2.0";
const VERSION: &str = "0.2.0";
const WINDOW: usize = 20;
const MIN_PERIODS: usize = 1;

pub struct StockDailyRcp;

#[derive(Clone, Copy, Debug)]
struct MinuteReturn {
    sequence: usize,
    value: f64,
}

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyRcp)
}

fn raw_spec() -> IntradayDailyRawSpec {
    stock_minute_raw_spec(CP_INTRADAY_RAW_ID, RAW_VERSION, &["close"], 1)
}

impl Factor for StockDailyRcp {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "rcp".to_string(),
            aliases: vec!["RCP".to_string(), "RCP_new".to_string()],
            name: "RCP".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "return",
                "intraday",
                "minute_agg",
                "regression",
                "neutralize",
                "barra",
                "size",
                "daily",
                "DWZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "RCP confidence recovery factor from intraday fast-up and fast-down close-to-close minute timing, de-intraday-returned and SIZE-neutralized.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["open", "close"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
            ],
            intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(
                CP_INTRADAY_RAW_ID,
                WINDOW - 1,
            )],
            lookback: Lookback {
                trading_days: WINDOW - 1,
            },
        }
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        vec![raw_spec()]
    }

    fn minute_compute(
        &self,
        raw_id: &str,
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Option<IntradayDailyRawSeries>> {
        if raw_id != CP_INTRADAY_RAW_ID {
            return Ok(None);
        }

        let mut values = Vec::new();
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
                if trade_times[idx].is_none() {
                    continue;
                }
                grouped.entry(ts_code).or_default().push(idx);
            }

            for (ts_code, mut indices) in grouped {
                indices.sort_by(|left, right| trade_times[*left].cmp(&trade_times[*right]));
                values.push(FactorValue {
                    key: FactorRowKey::Daily {
                        trade_date: *trade_date,
                        ts_code,
                    },
                    value: cp_intraday_from_rows(&indices, trade_times, &close),
                });
            }
        }

        Ok(Some(IntradayDailyRawSeries {
            spec: raw_spec(),
            values,
        }))
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(CP_INTRADAY_RAW_ID)?;
        let cp_intraday = panel.column(CP_INTRADAY_RAW_ID)?;
        let pv_table = data.daily(DatasetId::StockDailyPv)?;
        let open = panel.column_from_table(pv_table, "open")?;
        let close = panel.column_from_table(pv_table, "close")?;
        let ret_intraday = close.zip_binary(&open, ret)?;
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;

        let rcp_intraday = cp_intraday.cs_binary(&ret_intraday, cs_regression_residual)?;
        let rcp_mean = rcp_intraday.ts(|values| ts_mean(values, WINDOW, MIN_PERIODS))?;
        let rcp_std = rcp_intraday.ts(|values| ts_std_dev(values, WINDOW, MIN_PERIODS))?;
        let rcp_mean_desize = rcp_mean.cs_neutralize_regression(&[&size], None)?;
        let rcp_std_desize = rcp_std.cs_neutralize_regression(&[&size], None)?;
        let factor = subtract_pair(
            &rcp_mean_desize.cs(cs_zscore)?,
            &rcp_std_desize.cs(cs_zscore)?,
        )?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn cp_intraday_from_rows(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
) -> Option<f64> {
    let minute_returns = minute_close_returns_from_rows(indices, trade_times, close);
    cp_intraday_from_returns(&minute_returns)
}

fn minute_close_returns_from_rows(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
) -> Vec<MinuteReturn> {
    let mut minute_returns = Vec::new();
    let mut sequence = 0usize;
    let mut previous_close = None;
    for idx in indices {
        let Some(trade_time) = trade_times[*idx].as_deref() else {
            continue;
        };
        if !is_rcp_session_time(trade_time) {
            continue;
        }
        sequence += 1;
        let current_close = close[*idx];
        if let Some(value) = ret(current_close, previous_close) {
            minute_returns.push(MinuteReturn { sequence, value });
        }
        if clean(current_close).is_some() {
            previous_close = current_close;
        }
    }
    minute_returns
}

fn cp_intraday_from_returns(minute_returns: &[MinuteReturn]) -> Option<f64> {
    if minute_returns.len() < 2 {
        return None;
    }
    let mean =
        minute_returns.iter().map(|item| item.value).sum::<f64>() / minute_returns.len() as f64;
    let variance = minute_returns
        .iter()
        .map(|item| {
            let diff = item.value - mean;
            diff * diff
        })
        .sum::<f64>()
        / minute_returns.len() as f64;
    let std = variance.sqrt();
    let upper = mean + std;
    let lower = mean - std;
    let up_sequences = minute_returns
        .iter()
        .filter_map(|item| (item.value > upper).then_some(item.sequence as f64))
        .collect::<Vec<_>>();
    let down_sequences = minute_returns
        .iter()
        .filter_map(|item| (item.value < lower).then_some(item.sequence as f64))
        .collect::<Vec<_>>();
    let down_median = median_sorted(&down_sequences)?;
    let up_median = median_sorted(&up_sequences)?;
    Some(down_median - up_median)
}

fn is_rcp_session_time(trade_time: &str) -> bool {
    intraday_time_in_range(trade_time, "09:30:00", "11:30:00")
        || intraday_time_in_range(trade_time, "13:01:00", "15:00:00")
}

fn median_sorted(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        Some(values[mid])
    } else {
        Some((values[mid - 1] + values[mid]) / 2.0)
    }
}

fn ret(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (clean(numerator), clean(denominator)) {
        (Some(numerator), Some(denominator)) if denominator.abs() > f64::EPSILON => {
            Some(numerator / denominator - 1.0)
        }
        _ => None,
    }
}

fn subtract_pair(left: &PanelColumn, right: &PanelColumn) -> Result<PanelColumn> {
    left.zip_binary(right, |left, right| match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some(left - right),
        _ => None,
    })
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
    fn session_time_keeps_morning_and_afternoon_continuous_auction() {
        assert!(is_rcp_session_time("09:30:00"));
        assert!(is_rcp_session_time("09:31:00"));
        assert!(is_rcp_session_time("11:30:00"));
        assert!(!is_rcp_session_time("11:31:00"));
        assert!(!is_rcp_session_time("13:00:00"));
        assert!(is_rcp_session_time("13:01:00"));
        assert!(is_rcp_session_time("15:00:00"));
    }

    #[test]
    fn minute_return_uses_close_over_previous_close() {
        assert_close(ret(Some(102.0), Some(100.0)), Some(0.02));
        assert_eq!(ret(Some(102.0), Some(0.0)), None);
    }

    #[test]
    fn minute_close_returns_keep_0930_as_sequence_one_anchor() {
        let indices = vec![0, 1, 2, 3];
        let trade_times = vec![
            Some("09:30:00".to_string()),
            Some("09:31:00".to_string()),
            Some("09:32:00".to_string()),
            Some("11:31:00".to_string()),
        ];
        let close = vec![Some(100.0), Some(110.0), Some(99.0), Some(88.0)];

        let returns = minute_close_returns_from_rows(&indices, &trade_times, &close);

        assert_eq!(returns.len(), 2);
        assert_eq!(returns[0].sequence, 2);
        assert_close(Some(returns[0].value), Some(0.1));
        assert_eq!(returns[1].sequence, 3);
        assert_close(Some(returns[1].value), Some(-0.1));
    }

    #[test]
    fn cp_intraday_uses_down_median_minus_up_median() {
        let returns = vec![
            MinuteReturn {
                sequence: 1,
                value: -2.0,
            },
            MinuteReturn {
                sequence: 2,
                value: 0.0,
            },
            MinuteReturn {
                sequence: 3,
                value: 2.0,
            },
        ];

        assert_close(cp_intraday_from_returns(&returns), Some(-2.0));
    }

    #[test]
    fn cp_intraday_uses_even_median_for_event_sequences() {
        let returns = vec![
            MinuteReturn {
                sequence: 1,
                value: -10.0,
            },
            MinuteReturn {
                sequence: 2,
                value: -10.0,
            },
            MinuteReturn {
                sequence: 3,
                value: 0.0,
            },
            MinuteReturn {
                sequence: 4,
                value: 10.0,
            },
            MinuteReturn {
                sequence: 5,
                value: 10.0,
            },
        ];

        assert_close(cp_intraday_from_returns(&returns), Some(-3.0));
    }

    #[test]
    fn cp_intraday_returns_none_without_both_fast_sides() {
        let returns = vec![
            MinuteReturn {
                sequence: 1,
                value: 0.0,
            },
            MinuteReturn {
                sequence: 2,
                value: 0.0,
            },
        ];

        assert_eq!(cp_intraday_from_returns(&returns), None);
    }
}
