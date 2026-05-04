use std::collections::BTreeMap;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::{clean_intraday_value, intraday_time_in_range, stock_minute_raw_spec};
use crate::factor::Factor;
use crate::operators::ts_std_dev;

pub const PRICE_VOLUME_CORR_1430_RAW_ID: &str = "daily_price_volume_corr_1430";

const RAW_VERSION: &str = "0.1.0";
const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;

pub struct StockDailyPvCorrStd1430;

#[derive(Clone, Copy, Debug, Default)]
struct CorrAccumulator {
    count: usize,
    sum_x: f64,
    sum_y: f64,
    sum_xx: f64,
    sum_yy: f64,
    sum_xy: f64,
}

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyPvCorrStd1430)
}

fn raw_spec() -> IntradayDailyRawSpec {
    stock_minute_raw_spec(
        PRICE_VOLUME_CORR_1430_RAW_ID,
        RAW_VERSION,
        &["close", "vol"],
        1,
    )
}

impl Factor for StockDailyPvCorrStd1430 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "pv_corr_std_1430".to_string(),
            aliases: vec!["PV_corr_std_1430".to_string(), "PV_CORR_STD_1430".to_string()],
            name: "PV_corr_std_1430".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "volume",
                "correlation",
                "intraday",
                "minute_agg",
                "neutralize",
                "barra",
                "size",
                "daily",
                "DWZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Last-30-minute price-volume correlation volatility over 20 trading days, neutralized by SIZE.".to_string(),
            dependencies: vec![DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"])],
            intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(
                PRICE_VOLUME_CORR_1430_RAW_ID,
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
        if raw_id != PRICE_VOLUME_CORR_1430_RAW_ID {
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
                values.push(FactorValue {
                    key: FactorRowKey::Daily {
                        trade_date: *trade_date,
                        ts_code,
                    },
                    value: minute_price_volume_corr_1430(&indices, trade_times, &close, &volume),
                });
            }
        }

        Ok(Some(IntradayDailyRawSeries {
            spec: raw_spec(),
            values,
        }))
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(PRICE_VOLUME_CORR_1430_RAW_ID)?;
        let corr_1430 = panel.column(PRICE_VOLUME_CORR_1430_RAW_ID)?;
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;

        let corr_std = corr_1430.ts(|values| ts_std_dev(values, WINDOW, WINDOW))?;
        let factor = corr_std.cs_neutralize_regression(&[&size], None)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn minute_price_volume_corr_1430(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
    volume: &[Option<f64>],
) -> Option<f64> {
    let mut accumulator = CorrAccumulator::default();
    for idx in indices {
        let Some(trade_time) = trade_times[*idx].as_deref() else {
            continue;
        };
        if !intraday_time_in_range(trade_time, "14:31:00", "15:00:00") {
            continue;
        }
        let (Some(close), Some(volume)) = (
            clean_intraday_value(close[*idx]),
            clean_intraday_value(volume[*idx]),
        ) else {
            continue;
        };
        accumulator.push(close, volume);
    }
    accumulator.corr()
}

impl CorrAccumulator {
    fn push(&mut self, x: f64, y: f64) {
        self.count += 1;
        self.sum_x += x;
        self.sum_y += y;
        self.sum_xx += x * x;
        self.sum_yy += y * y;
        self.sum_xy += x * y;
    }

    fn corr(self) -> Option<f64> {
        if self.count < 2 {
            return None;
        }
        let n = self.count as f64;
        let cov = self.sum_xy - self.sum_x * self.sum_y / n;
        let var_x = self.sum_xx - self.sum_x * self.sum_x / n;
        let var_y = self.sum_yy - self.sum_y * self.sum_y / n;
        if var_x <= f64::EPSILON || var_y <= f64::EPSILON {
            return None;
        }
        Some(cov / (var_x.sqrt() * var_y.sqrt()))
    }
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
    fn minute_price_volume_corr_uses_1431_to_1500_window() {
        let indices = vec![0, 1, 2, 3, 4];
        let times = vec![
            Some("14:30:00".to_string()),
            Some("14:31:00".to_string()),
            Some("14:32:00".to_string()),
            Some("15:00:00".to_string()),
            Some("15:01:00".to_string()),
        ];
        let close = vec![Some(1000.0), Some(2.0), Some(4.0), Some(6.0), Some(-1000.0)];
        let volume = vec![Some(-1000.0), Some(1.0), Some(2.0), Some(3.0), Some(1000.0)];

        assert_close(
            minute_price_volume_corr_1430(&indices, &times, &close, &volume),
            Some(1.0),
        );
    }

    #[test]
    fn minute_price_volume_corr_rejects_zero_variance() {
        let indices = vec![0, 1, 2];
        let times = vec![
            Some("14:31:00".to_string()),
            Some("14:32:00".to_string()),
            Some("14:33:00".to_string()),
        ];
        let close = vec![Some(10.0), Some(10.0), Some(10.0)];
        let volume = vec![Some(1.0), Some(2.0), Some(3.0)];

        assert_eq!(
            minute_price_volume_corr_1430(&indices, &times, &close, &volume),
            None
        );
    }
}
