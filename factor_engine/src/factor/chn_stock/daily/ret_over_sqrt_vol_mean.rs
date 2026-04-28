use std::collections::BTreeMap;

use crate::core::{
    AssetClass, FactorContext, FactorRowKey, FactorSeries, FactorSpec, FactorValue, Frequency,
    IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec, Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::{
    clean_intraday_value, intraday_time_in_range, stock_minute_raw_spec, DailyPanel,
};
use crate::factor::Factor;

pub const RAW_ID: &str = "ret_over_sqrt_vol_mean";
const RAW_VERSION: &str = "0.1.0";

pub struct StockDailyRetOverSqrtVolMean;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyRetOverSqrtVolMean)
}

pub fn raw_spec() -> IntradayDailyRawSpec {
    stock_minute_raw_spec(RAW_ID, RAW_VERSION, &["close", "vol"], 1)
}

impl Factor for StockDailyRetOverSqrtVolMean {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "ret_over_sqrt_vol_mean".to_string(),
            aliases: Vec::new(),
            name: "Stock intraday mean return over sqrt volume".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: [
                "price_volume",
                "return",
                "volume",
                "intraday",
                "minute_agg",
                "daily",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description:
                "Mean intraday minute return divided by sqrt(volume), using 09:31-15:00 bars."
                    .to_string(),
            dependencies: Vec::new(),
            intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(RAW_ID, 0)],
            lookback: Lookback { trading_days: 0 },
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
            let Some(table) = data.minute(raw_spec().source_dataset, *trade_date) else {
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
                    value: ret_over_sqrt_vol_mean_from_rows(&indices, trade_times, &close, &volume),
                });
            }
        }
        Ok(Some(IntradayDailyRawSeries {
            spec: raw_spec(),
            values,
        }))
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = DailyPanel::from_table(data.intraday_daily_raw(RAW_ID)?, context)?;
        let factor = panel.column(RAW_ID)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn ret_over_sqrt_vol_mean_from_rows(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
    volume: &[Option<f64>],
) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for pos in 1..indices.len() {
        let idx = indices[pos];
        let Some(trade_time) = trade_times[idx].as_deref() else {
            continue;
        };
        if !intraday_time_in_range(trade_time, "09:31:00", "15:00:00") {
            continue;
        }
        let prev_idx = indices[pos - 1];
        let (Some(current), Some(previous), Some(vol)) = (
            clean_intraday_value(close[idx]),
            clean_intraday_value(close[prev_idx]),
            clean_intraday_value(volume[idx]),
        ) else {
            continue;
        };
        if previous.abs() <= f64::EPSILON {
            continue;
        }
        let ret = current / previous - 1.0;
        let value = if vol == 0.0 {
            0.0
        } else if vol > 0.0 {
            ret / vol.sqrt()
        } else {
            continue;
        };
        sum += value;
        count += 1;
    }
    (count > 0).then_some(sum / count as f64)
}

#[cfg(test)]
mod tests {
    use super::ret_over_sqrt_vol_mean_from_rows;

    #[test]
    fn ret_over_sqrt_vol_mean_skips_missing_and_uses_zero_for_zero_volume() {
        let indices = vec![0, 1, 2, 3, 4];
        let trade_times = vec![
            Some("09:30:00".to_string()),
            Some("09:31:00".to_string()),
            Some("09:32:00".to_string()),
            Some("09:33:00".to_string()),
            Some("15:01:00".to_string()),
        ];
        let close = vec![Some(10.0), Some(11.0), Some(9.9), None, Some(20.0)];
        let volume = vec![Some(1.0), Some(100.0), Some(0.0), Some(9.0), Some(4.0)];

        let actual = ret_over_sqrt_vol_mean_from_rows(&indices, &trade_times, &close, &volume)
            .expect("factor should be valid");
        assert!((actual - 0.005).abs() < 1e-12);
    }
}
