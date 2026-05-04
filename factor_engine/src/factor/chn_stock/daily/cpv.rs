use std::collections::BTreeMap;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::common::{
    clean_intraday_value, intraday_time_in_range, stock_minute_raw_spec, DailyPanel, PanelColumn,
};
use crate::factor::Factor;
use crate::operators::{cs_zscore, ts_mean, ts_regression, ts_std_dev, ts_sum};

pub const PRICE_VOLUME_CORR_RAW_ID: &str = "daily_price_volume_corr";

const RAW_VERSION: &str = "0.1.0";
const VERSION: &str = "0.2.0";
const WINDOW: usize = 20;
const MIN_PERIODS: usize = 1;

pub struct StockDailyCpv;

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
    Box::new(StockDailyCpv)
}

fn raw_spec() -> IntradayDailyRawSpec {
    stock_minute_raw_spec(PRICE_VOLUME_CORR_RAW_ID, RAW_VERSION, &["close", "vol"], 1)
}

impl Factor for StockDailyCpv {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "cpv".to_string(),
            aliases: Vec::new(),
            name: "CPV".to_string(),
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
            description: "CPV factor from intraday close-volume correlation level, volatility, and trend after SIZE, reversal, turnover, and volatility neutralization.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close", "pre_close"]),
                DataRequest::new(DatasetId::StockDailyBasic, &["turnover_rate_f"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
            ],
            intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(
                PRICE_VOLUME_CORR_RAW_ID,
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
        if raw_id != PRICE_VOLUME_CORR_RAW_ID {
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
                    value: minute_price_volume_corr(&indices, trade_times, &close, &volume),
                });
            }
        }

        Ok(Some(IntradayDailyRawSeries {
            spec: raw_spec(),
            values,
        }))
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(PRICE_VOLUME_CORR_RAW_ID)?;
        let pv_table = data.daily(DatasetId::StockDailyPv)?;
        let basic_table = data.daily(DatasetId::StockDailyBasic)?;

        let daily_price_volume_corr = panel.column(PRICE_VOLUME_CORR_RAW_ID)?;
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;
        let close = panel.column_from_table(pv_table, "close")?;
        let pre_close = panel.column_from_table(pv_table, "pre_close")?;
        let turnover = panel
            .column_from_table(basic_table, "turnover_rate_f")?
            .map_values(turnover_percent_to_decimal);

        let stock_return = close.zip_binary(&pre_close, ret)?;
        let ret20 = stock_return.ts(|values| ts_sum(values, WINDOW, MIN_PERIODS))?;
        let turnover20 = turnover.ts(|values| ts_mean(values, WINDOW, MIN_PERIODS))?;
        let vol20 = stock_return.ts(|values| ts_std_dev(values, WINDOW, MIN_PERIODS))?;

        let pv_corr_avg_raw =
            daily_price_volume_corr.ts(|values| ts_mean(values, WINDOW, MIN_PERIODS))?;
        let pv_corr_std_raw =
            daily_price_volume_corr.ts(|values| ts_std_dev(values, WINDOW, MIN_PERIODS))?;

        let pv_corr_avg = pv_corr_avg_raw.cs_neutralize_regression(&[&size], None)?;
        let pv_corr_std = pv_corr_std_raw.cs_neutralize_regression(&[&size], None)?;
        let pv_corr_avg_deret20 = pv_corr_avg.cs_neutralize_regression(&[&ret20], None)?;
        let pv_corr_std_deret20 = pv_corr_std.cs_neutralize_regression(&[&ret20], None)?;
        let pv_corr_de_ret20 = average_pair(
            &pv_corr_avg_deret20.cs(cs_zscore)?,
            &pv_corr_std_deret20.cs(cs_zscore)?,
        )?;

        let time_index = time_index_column(panel)?;
        let pv_corr_trend_raw = daily_price_volume_corr
            .ts_binary(&time_index, |y, x| ts_regression(y, x, WINDOW, MIN_PERIODS))?;
        let pv_corr_trend = pv_corr_trend_raw
            .cs_neutralize_regression(&[&size, &ret20, &turnover20, &vol20], None)?;

        let cpv = average_pair(
            &pv_corr_de_ret20.cs(cs_zscore)?,
            &pv_corr_trend.cs(cs_zscore)?,
        )?;
        Ok(cpv.to_factor_series(self.spec()))
    }
}

fn minute_price_volume_corr(
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
        if !intraday_time_in_range(trade_time, "09:31:00", "15:00:00") {
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

fn time_index_column(panel: &DailyPanel) -> Result<PanelColumn> {
    let mut values = Vec::with_capacity(panel.shape_len());
    for date_idx in 0..panel.dates().len() {
        let value = Some((date_idx + 1) as f64);
        for _ in panel.instruments() {
            values.push(value);
        }
    }
    panel.column_from_values(values)
}

fn average_pair(left: &PanelColumn, right: &PanelColumn) -> Result<PanelColumn> {
    left.zip_binary(right, |left, right| match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some((left + right) / 2.0),
        _ => None,
    })
}

fn ret(close: Option<f64>, pre_close: Option<f64>) -> Option<f64> {
    match (clean(close), clean(pre_close)) {
        (Some(close), Some(pre_close)) if pre_close.abs() > f64::EPSILON => {
            Some(close / pre_close - 1.0)
        }
        _ => None,
    }
}

fn turnover_percent_to_decimal(value: Option<f64>) -> Option<f64> {
    clean(value).map(|value| value / 100.0)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::core::{AssetClass, Frequency};
    use crate::data::{ColumnData, Table};

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
    fn minute_price_volume_corr_uses_0931_to_1500_window() {
        let indices = vec![0, 1, 2, 3, 4];
        let times = vec![
            Some("09:30:00".to_string()),
            Some("09:31:00".to_string()),
            Some("09:32:00".to_string()),
            Some("15:00:00".to_string()),
            Some("15:01:00".to_string()),
        ];
        let close = vec![Some(1000.0), Some(2.0), Some(4.0), Some(6.0), Some(-1000.0)];
        let volume = vec![Some(-1000.0), Some(1.0), Some(2.0), Some(3.0), Some(1000.0)];

        assert_close(
            minute_price_volume_corr(&indices, &times, &close, &volume),
            Some(1.0),
        );
    }

    #[test]
    fn minute_price_volume_corr_rejects_zero_variance() {
        let indices = vec![0, 1, 2];
        let times = vec![
            Some("09:31:00".to_string()),
            Some("09:32:00".to_string()),
            Some("09:33:00".to_string()),
        ];
        let close = vec![Some(10.0), Some(10.0), Some(10.0)];
        let volume = vec![Some(1.0), Some(2.0), Some(3.0)];

        assert_eq!(
            minute_price_volume_corr(&indices, &times, &close, &volume),
            None
        );
    }

    #[test]
    fn time_index_regression_matches_one_to_twenty_slope() {
        let dates = (0..20).map(|idx| 20260101 + idx).collect::<Vec<_>>();
        let mut trade_dates = Vec::new();
        let mut ts_codes = Vec::new();
        let mut y = Vec::new();
        for (idx, date) in dates.iter().enumerate() {
            trade_dates.push(Some(*date));
            ts_codes.push(Some("000001.SZ".to_string()));
            y.push(Some(1.0 + 2.0 * (idx + 1) as f64));
        }
        let table = Table::new(BTreeMap::from([
            ("trade_date".to_string(), ColumnData::I32(trade_dates)),
            ("ts_code".to_string(), ColumnData::Utf8(ts_codes)),
            ("y".to_string(), ColumnData::F64(y)),
        ]))
        .expect("table");
        let context = FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: dates[19],
            end_date: dates[19],
            load_start_date: dates[0],
            load_dates: dates.clone(),
            target_dates: vec![dates[19]],
        };
        let panel = DailyPanel::from_table(&table, &context).expect("panel");
        let y = panel.column("y").expect("y");
        let time_index = time_index_column(&panel).expect("time");
        let slope = y
            .ts_binary(&time_index, |y, x| ts_regression(y, x, WINDOW, MIN_PERIODS))
            .expect("slope");

        assert_close(slope.values()[19], Some(2.0));
    }
}
