use std::collections::BTreeMap;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::neutralize_size_sector;
use crate::factor::common::{clean_intraday_value, intraday_time_in_range, stock_minute_raw_spec};
use crate::factor::Factor;
use crate::operators::{ts_max, ts_mean, ts_min, ts_sum};

pub const ARPP_PRICE_SUM_RAW_ID: &str = "daily_arpp_ohlc_price_sum";
pub const ARPP_PRICE_COUNT_RAW_ID: &str = "daily_arpp_ohlc_price_count";
pub const ARPP_HIGH_RAW_ID: &str = "daily_arpp_high";
pub const ARPP_LOW_RAW_ID: &str = "daily_arpp_low";

const RAW_VERSION: &str = "0.1.0";
const VERSION: &str = "0.1.0";
const PRICE_WINDOW: usize = 5;
const SMOOTH_WINDOW: usize = 20;
const MIN_PERIODS: usize = 1;

pub struct StockDailyArpp5d20d;

#[derive(Clone, Copy, Debug, Default)]
struct DailyStats {
    price_sum: f64,
    count: usize,
    high: Option<f64>,
    low: Option<f64>,
}

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyArpp5d20d)
}

fn raw_specs() -> Vec<IntradayDailyRawSpec> {
    [
        ARPP_PRICE_SUM_RAW_ID,
        ARPP_PRICE_COUNT_RAW_ID,
        ARPP_HIGH_RAW_ID,
        ARPP_LOW_RAW_ID,
    ]
    .iter()
    .map(|raw_id| stock_minute_raw_spec(raw_id, RAW_VERSION, &["open", "high", "low", "close"], 1))
    .collect()
}

impl Factor for StockDailyArpp5d20d {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "arpp_5d_20d".to_string(),
            aliases: vec!["ARPP_5d_20d".to_string(), "ARPP5D20D".to_string()],
            name: "ARPP 5d 20d".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "price",
                "intraday",
                "minute_agg",
                "neutralize",
                "barra",
                "size",
                "sector",
                "daily",
                "DFZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "5-day intraday average relative price position averaged over 20 days and neutralized by SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: vec![
                IntradayDailyRawRequest::new(ARPP_PRICE_SUM_RAW_ID, lookback_days()),
                IntradayDailyRawRequest::new(ARPP_PRICE_COUNT_RAW_ID, lookback_days()),
                IntradayDailyRawRequest::new(ARPP_HIGH_RAW_ID, lookback_days()),
                IntradayDailyRawRequest::new(ARPP_LOW_RAW_ID, lookback_days()),
            ],
            lookback: Lookback {
                trading_days: lookback_days(),
            },
        }
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        raw_specs()
    }

    fn minute_compute_many(
        &self,
        raw_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<IntradayDailyRawSeries>> {
        let requested = raw_ids
            .iter()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        let specs = raw_specs();
        let mut price_sum_values = Vec::new();
        let mut count_values = Vec::new();
        let mut high_values = Vec::new();
        let mut low_values = Vec::new();

        for trade_date in &context.target_dates {
            let Some(table) = data.minute(DatasetId::StockMinute1m, *trade_date) else {
                continue;
            };
            let ts_codes = table.required_utf8("ts_code")?;
            let trade_times = table.required_utf8("trade_time")?;
            let open = table.required_f64_cast("open")?;
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

            for (ts_code, mut indices) in grouped {
                indices.sort_by(|left, right| trade_times[*left].cmp(&trade_times[*right]));
                let stats = daily_stats(&indices, trade_times, &open, &high, &low, &close);
                let key = FactorRowKey::Daily {
                    trade_date: *trade_date,
                    ts_code,
                };
                price_sum_values.push(FactorValue {
                    key: key.clone(),
                    value: stats.price_sum(),
                });
                count_values.push(FactorValue {
                    key: key.clone(),
                    value: stats.count(),
                });
                high_values.push(FactorValue {
                    key: key.clone(),
                    value: stats.high,
                });
                low_values.push(FactorValue {
                    key,
                    value: stats.low,
                });
            }
        }

        let mut output = Vec::new();
        for spec in specs {
            if !requested.contains(spec.raw_id.as_str()) {
                continue;
            }
            let values = match spec.raw_id.as_str() {
                ARPP_PRICE_SUM_RAW_ID => price_sum_values.clone(),
                ARPP_PRICE_COUNT_RAW_ID => count_values.clone(),
                ARPP_HIGH_RAW_ID => high_values.clone(),
                ARPP_LOW_RAW_ID => low_values.clone(),
                _ => Vec::new(),
            };
            output.push(IntradayDailyRawSeries { spec, values });
        }
        Ok(output)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(ARPP_PRICE_SUM_RAW_ID)?;
        let price_sum = panel.column(ARPP_PRICE_SUM_RAW_ID)?;
        let price_count = panel.column(ARPP_PRICE_COUNT_RAW_ID)?;
        let high = panel.column(ARPP_HIGH_RAW_ID)?;
        let low = panel.column(ARPP_LOW_RAW_ID)?;

        let sum5 = price_sum.ts(|values| ts_sum(values, PRICE_WINDOW, MIN_PERIODS))?;
        let count5 = price_count.ts(|values| ts_sum(values, PRICE_WINDOW, MIN_PERIODS))?;
        let twap5 = sum5.zip_binary(&count5, safe_div)?;
        let high5 = high.ts(|values| ts_max(values, PRICE_WINDOW, MIN_PERIODS))?;
        let low5 = low.ts(|values| ts_min(values, PRICE_WINDOW, MIN_PERIODS))?;
        let raw = twap5
            .zip_binary(&low5, subtract)?
            .zip_binary(&high5.zip_binary(&low5, subtract)?, safe_div)?;
        let smoothed = raw.ts(|values| ts_mean(values, SMOOTH_WINDOW, MIN_PERIODS))?;
        let factor = neutralize_size_sector(&smoothed, panel, data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn lookback_days() -> usize {
    (PRICE_WINDOW - 1) + (SMOOTH_WINDOW - 1)
}

fn daily_stats(
    indices: &[usize],
    trade_times: &[Option<String>],
    open: &[Option<f64>],
    high: &[Option<f64>],
    low: &[Option<f64>],
    close: &[Option<f64>],
) -> DailyStats {
    let mut stats = DailyStats::default();
    for idx in indices {
        let Some(trade_time) = trade_times[*idx].as_deref() else {
            continue;
        };
        if !intraday_time_in_range(trade_time, "09:31:00", "15:00:00") {
            continue;
        }
        let (Some(open), Some(high), Some(low), Some(close)) = (
            clean_intraday_value(open[*idx]),
            clean_intraday_value(high[*idx]),
            clean_intraday_value(low[*idx]),
            clean_intraday_value(close[*idx]),
        ) else {
            continue;
        };
        stats.price_sum += (open + high + low + close) / 4.0;
        stats.count += 1;
        stats.high = Some(stats.high.map_or(high, |current| current.max(high)));
        stats.low = Some(stats.low.map_or(low, |current| current.min(low)));
    }
    stats
}

impl DailyStats {
    fn price_sum(self) -> Option<f64> {
        if self.count > 0 {
            Some(self.price_sum)
        } else {
            None
        }
    }

    fn count(self) -> Option<f64> {
        if self.count > 0 {
            Some(self.count as f64)
        } else {
            None
        }
    }
}

fn safe_div(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (numerator, denominator) {
        (Some(numerator), Some(denominator))
            if numerator.is_finite()
                && denominator.is_finite()
                && denominator.abs() > f64::EPSILON =>
        {
            Some(numerator / denominator)
        }
        _ => None,
    }
}

fn subtract(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (left, right) {
        (Some(left), Some(right)) if left.is_finite() && right.is_finite() => Some(left - right),
        _ => None,
    }
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
    fn daily_stats_use_0931_to_1500_and_ohlc_average() {
        let indices = vec![0, 1, 2];
        let times = vec![
            Some("09:30:00".to_string()),
            Some("09:31:00".to_string()),
            Some("15:00:00".to_string()),
        ];
        let open = vec![Some(1000.0), Some(1.0), Some(5.0)];
        let high = vec![Some(1000.0), Some(3.0), Some(9.0)];
        let low = vec![Some(1000.0), Some(1.0), Some(3.0)];
        let close = vec![Some(1000.0), Some(3.0), Some(7.0)];

        let stats = daily_stats(&indices, &times, &open, &high, &low, &close);

        assert_eq!(stats.count(), Some(2.0));
        assert_close(stats.price_sum(), 2.0 + 6.0);
        assert_close(stats.high, 9.0);
        assert_close(stats.low, 1.0);
    }

    #[test]
    fn safe_div_rejects_zero_denominator() {
        assert_eq!(safe_div(Some(1.0), Some(0.0)), None);
        assert_close(safe_div(Some(6.0), Some(3.0)), 2.0);
    }
}
