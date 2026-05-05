use std::collections::BTreeMap;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::common::{clean_intraday_value, intraday_time_in_range, stock_minute_raw_spec};
use crate::factor::Factor;

const RAW_ID: &str = "daily_intraday_rsi";
const RAW_VERSION: &str = "0.1.0";
const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;
const MIN_PERIODS: usize = 20;

pub struct StockDailyVolumeWeightedIntradayRsi;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyVolumeWeightedIntradayRsi)
}

fn raw_spec() -> IntradayDailyRawSpec {
    stock_minute_raw_spec(RAW_ID, RAW_VERSION, &["open", "close"], 1)
}

impl Factor for StockDailyVolumeWeightedIntradayRsi {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "volume_weighted_intraday_rsi".to_string(),
            aliases: vec![
                "VolumeWeightedIntradayRSI".to_string(),
                "Volume-Weighted Intraday RSI".to_string(),
            ],
            name: "Volume-Weighted Intraday RSI".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "return",
                "intraday",
                "minute_agg",
                "rsi",
                "turnover",
                "neutralize",
                "barra",
                "size",
                "daily",
                "GSZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Volume-Weighted Intraday RSI factor, computed as the 20-day turnover-weighted average of daily high-frequency RSI after SIZE neutralization.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyBasic, &["turnover_rate_f"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
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

    fn minute_compute(
        &self,
        raw_id: &str,
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Option<IntradayDailyRawSeries>> {
        if raw_id != RAW_ID {
            return Ok(None);
        }

        let mut values = Vec::new();
        for trade_date in &context.target_dates {
            let Some(table) = data.minute(DatasetId::StockMinute1m, *trade_date) else {
                continue;
            };
            let ts_codes = table.required_utf8("ts_code")?;
            let trade_times = table.required_utf8("trade_time")?;
            let open = table.required_f64_cast("open")?;
            let close = table.required_f64_cast("close")?;

            let mut grouped = BTreeMap::<String, Vec<usize>>::new();
            for idx in 0..table.len {
                let Some(ts_code) = ts_codes[idx].clone() else {
                    continue;
                };
                let Some(trade_time) = trade_times[idx].as_deref() else {
                    continue;
                };
                if !intraday_time_in_range(trade_time, "09:30:00", "15:00:00") {
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
                    value: daily_intraday_rsi(&indices, &open, &close),
                });
            }
        }

        Ok(Some(IntradayDailyRawSeries {
            spec: raw_spec(),
            values,
        }))
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(RAW_ID)?;
        let daily_rsi = panel.column(RAW_ID)?;
        let turnover = panel
            .column_from_table(data.daily(DatasetId::StockDailyBasic)?, "turnover_rate_f")?
            .map_values(turnover_percent_to_decimal);
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;

        let raw_factor = daily_rsi.ts_binary(&turnover, turnover_weighted_average)?;
        let factor = raw_factor.cs_neutralize_regression(&[&size], None)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn daily_intraday_rsi(
    indices: &[usize],
    open: &[Option<f64>],
    close: &[Option<f64>],
) -> Option<f64> {
    let mut up_sum = 0.0;
    let mut up_count = 0usize;
    let mut down_abs_sum = 0.0;
    let mut down_count = 0usize;

    for idx in indices {
        let Some(return_value) = minute_bar_return(open[*idx], close[*idx]) else {
            continue;
        };
        if return_value > 0.0 {
            up_sum += return_value;
            up_count += 1;
        } else if return_value < 0.0 {
            down_abs_sum += -return_value;
            down_count += 1;
        }
    }

    match (up_count, down_count) {
        (0, 0) => None,
        (_, 0) => Some(100.0),
        (0, _) => Some(0.0),
        _ => {
            let up_avg = up_sum / up_count as f64;
            let down_abs_avg = down_abs_sum / down_count as f64;
            let denominator = up_avg + down_abs_avg;
            (denominator.abs() > f64::EPSILON).then_some(up_avg / denominator * 100.0)
        }
    }
}

fn turnover_weighted_average(
    rsi_values: &[Option<f64>],
    turnover_values: &[Option<f64>],
) -> Vec<Option<f64>> {
    let mut output = vec![None; rsi_values.len()];
    for idx in 0..rsi_values.len() {
        if idx + 1 < WINDOW {
            continue;
        }
        let start = idx + 1 - WINDOW;
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        let mut count = 0usize;
        for window_idx in start..=idx {
            let (Some(rsi), Some(weight)) = (
                clean(rsi_values[window_idx]),
                clean(turnover_values[window_idx]),
            ) else {
                continue;
            };
            if weight <= 0.0 {
                continue;
            }
            weighted_sum += rsi * weight;
            weight_sum += weight;
            count += 1;
        }
        if count >= MIN_PERIODS && weight_sum > f64::EPSILON {
            output[idx] = Some(weighted_sum / weight_sum);
        }
    }
    output
}

fn turnover_percent_to_decimal(value: Option<f64>) -> Option<f64> {
    clean(value).map(|value| value / 100.0)
}

fn minute_bar_return(open: Option<f64>, close: Option<f64>) -> Option<f64> {
    let (Some(open), Some(close)) = (clean_intraday_value(open), clean_intraday_value(close))
    else {
        return None;
    };
    if open.abs() <= f64::EPSILON {
        return None;
    }
    Some(close / open - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Option<f64>, expected: Option<f64>) {
        match (actual, expected) {
            (Some(actual), Some(expected)) => assert!(
                (actual - expected).abs() < 1e-12,
                "expected {expected}, got {actual}"
            ),
            (None, None) => {}
            _ => panic!("expected {:?}, got {:?}", expected, actual),
        }
    }

    #[test]
    fn daily_intraday_rsi_uses_up_and_down_average_returns() {
        let indices = vec![0, 1, 2];
        let open = vec![Some(100.0), Some(100.0), Some(100.0)];
        let close = vec![Some(101.0), Some(98.0), Some(103.0)];

        assert_close(daily_intraday_rsi(&indices, &open, &close), Some(50.0));
    }

    #[test]
    fn daily_intraday_rsi_handles_one_sided_days() {
        let indices = vec![0, 1];
        let open = vec![Some(100.0), Some(100.0)];

        let all_up_close = vec![Some(101.0), Some(102.0)];
        assert_eq!(
            daily_intraday_rsi(&indices, &open, &all_up_close),
            Some(100.0)
        );

        let all_down_close = vec![Some(99.0), Some(98.0)];
        assert_eq!(
            daily_intraday_rsi(&indices, &open, &all_down_close),
            Some(0.0)
        );

        let flat_close = vec![Some(100.0), Some(100.0)];
        assert_eq!(daily_intraday_rsi(&indices, &open, &flat_close), None);
    }

    #[test]
    fn turnover_weighted_average_requires_full_valid_window() {
        let rsi = vec![Some(10.0); WINDOW];
        let mut turnover = vec![Some(1.0); WINDOW];
        turnover[0] = None;

        let output = turnover_weighted_average(&rsi, &turnover);
        assert_eq!(output[WINDOW - 1], None);
    }

    #[test]
    fn turnover_weighted_average_uses_turnover_weights() {
        let mut rsi = vec![Some(10.0); WINDOW];
        let mut turnover = vec![Some(1.0); WINDOW];
        rsi[WINDOW - 1] = Some(30.0);
        turnover[WINDOW - 1] = Some(3.0);

        let output = turnover_weighted_average(&rsi, &turnover);
        let expected = (10.0 * 19.0 + 30.0 * 3.0) / 22.0;
        assert_close(output[WINDOW - 1], Some(expected));
    }
}
