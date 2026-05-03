use std::collections::BTreeMap;

use crate::core::{
    AssetClass, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec, FactorValue,
    Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec, Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::{intraday_time_in_range, stock_minute_raw_spec};
use crate::factor::Factor;
use crate::operators::{ts_mean, ts_sum};

pub const RAW_ID: &str = "daily_rapid_current_spread";

const RAW_VERSION: &str = "0.3.0";
const VERSION: &str = "0.3.0";
const WINDOW: usize = 20;
const INTRADAY_WINDOW: usize = 5;
const TREND_LAG: usize = 5;
const OHLC_COUNT: usize = INTRADAY_WINDOW * 4;

pub struct StockDailyRapidCurrentAdvance;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyRapidCurrentAdvance)
}

pub fn raw_spec() -> IntradayDailyRawSpec {
    stock_minute_raw_spec(
        RAW_ID,
        RAW_VERSION,
        &["open", "high", "low", "close", "vol", "amount"],
        1,
    )
}

impl Factor for StockDailyRapidCurrentAdvance {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "rapid_current_advance".to_string(),
            aliases: Vec::new(),
            name: "Rapid Current Advance".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "volume",
                "amount",
                "intraday",
                "minute_agg",
                "daily",
                "FZZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Intraday volume-expansion downtrend amount and volume ratios summed, then averaged over 20 trading days.".to_string(),
            dependencies: Vec::new(),
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
            let high = table.required_f64_cast("high")?;
            let low = table.required_f64_cast("low")?;
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

            for (ts_code, mut indices) in grouped {
                indices.sort_by(|left, right| trade_times[*left].cmp(&trade_times[*right]));
                let value = rapid_current_spread_from_rows(
                    &indices,
                    trade_times,
                    &open,
                    &high,
                    &low,
                    &close,
                    &volume,
                    &amount,
                );
                values.push(FactorValue {
                    key: FactorRowKey::Daily {
                        trade_date: *trade_date,
                        ts_code,
                    },
                    value,
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
        let raw = panel.column(RAW_ID)?;
        let factor = raw.ts(|values| ts_mean(values, WINDOW, 1))?;

        Ok(factor.to_factor_series(self.spec()))
    }
}

pub fn rapid_current_spread_from_rows(
    indices: &[usize],
    trade_times: &[Option<String>],
    open: &[Option<f64>],
    high: &[Option<f64>],
    low: &[Option<f64>],
    close: &[Option<f64>],
    volume: &[Option<f64>],
    amount: &[Option<f64>],
) -> Option<f64> {
    let selected = indices
        .iter()
        .copied()
        .filter(|idx| {
            trade_times[*idx]
                .as_deref()
                .is_some_and(|time| intraday_time_in_range(time, "09:31:00", "14:57:00"))
        })
        .collect::<Vec<_>>();
    if selected.len() <= INTRADAY_WINDOW {
        return None;
    }

    let Some(amount_all_mean) = mean_clean(selected.iter().map(|idx| amount[*idx])) else {
        return None;
    };
    let Some(volume_all_mean) = mean_clean(selected.iter().map(|idx| volume[*idx])) else {
        return None;
    };
    if amount_all_mean.abs() <= f64::EPSILON || volume_all_mean.abs() <= f64::EPSILON {
        return None;
    }

    let volume_series = selected.iter().map(|idx| volume[*idx]).collect::<Vec<_>>();
    let rolling_volume = ts_sum(&volume_series, INTRADAY_WINDOW, INTRADAY_WINDOW);
    let rolling_ohlc_mean = rolling_ohlc_mean(&selected, open, high, low, close);
    let mut event_amount = Vec::new();
    let mut event_volume = Vec::new();
    for pos in INTRADAY_WINDOW - 1 + TREND_LAG..selected.len() {
        let (Some(current_volume), Some(previous_volume)) =
            (rolling_volume[pos], rolling_volume[pos - 1])
        else {
            continue;
        };
        let (Some(current_ohlc), Some(lagged_ohlc)) =
            (rolling_ohlc_mean[pos], rolling_ohlc_mean[pos - TREND_LAG])
        else {
            continue;
        };
        let volume_expanding = current_volume > previous_volume;
        let trend_down = current_ohlc < lagged_ohlc;
        if !volume_expanding || !trend_down {
            continue;
        }

        let idx = selected[pos];
        if let Some(value) = clean(amount[idx]) {
            event_amount.push(value);
        }
        if let Some(value) = clean(volume[idx]) {
            event_volume.push(value);
        }
    }

    if event_amount.is_empty() || event_volume.is_empty() {
        return None;
    }
    let amount_ratio =
        event_amount.iter().sum::<f64>() / event_amount.len() as f64 / amount_all_mean;
    let volume_ratio =
        event_volume.iter().sum::<f64>() / event_volume.len() as f64 / volume_all_mean;
    Some(amount_ratio + volume_ratio)
}

fn rolling_ohlc_mean(
    indices: &[usize],
    open: &[Option<f64>],
    high: &[Option<f64>],
    low: &[Option<f64>],
    close: &[Option<f64>],
) -> Vec<Option<f64>> {
    let mut output = vec![None; indices.len()];
    for pos in INTRADAY_WINDOW - 1..indices.len() {
        let mut sum = 0.0;
        let mut count = 0usize;
        let mut complete = true;
        for offset in 0..INTRADAY_WINDOW {
            let idx = indices[pos - offset];
            for value in [open[idx], high[idx], low[idx], close[idx]] {
                let Some(value) = clean(value) else {
                    complete = false;
                    break;
                };
                sum += value;
                count += 1;
            }
            if !complete {
                break;
            }
        }
        if complete && count == OHLC_COUNT {
            output[pos] = Some(sum / OHLC_COUNT as f64);
        }
    }
    output
}

fn mean_clean(values: impl IntoIterator<Item = Option<f64>>) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        let Some(value) = clean(value) else {
            continue;
        };
        sum += value;
        count += 1;
    }
    (count > 0).then(|| sum / count as f64)
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| !value.is_nan())
}

#[cfg(test)]
mod tests {
    use super::{rapid_current_spread_from_rows, rolling_ohlc_mean};

    fn times(values: &[&str]) -> Vec<Option<String>> {
        values
            .iter()
            .map(|value| Some((*value).to_string()))
            .collect()
    }

    #[test]
    fn rolling_ohlc_mean_requires_twenty_complete_prices() {
        let indices = (0..6).collect::<Vec<_>>();
        let open = vec![Some(1.0); 6];
        let high = vec![Some(2.0); 6];
        let low = vec![Some(0.5); 6];
        let mut close = vec![Some(1.5); 6];
        close[4] = None;

        let means = rolling_ohlc_mean(&indices, &open, &high, &low, &close);

        assert_eq!(means[4], None);
        assert_eq!(means[5], None);
    }

    #[test]
    fn raw_uses_filtered_minutes_and_sums_amount_volume_ratios_for_events() {
        let indices = (0..11).collect::<Vec<_>>();
        let trade_times = times(&[
            "09:30:00", "09:31:00", "09:32:00", "09:33:00", "09:34:00", "09:35:00", "09:36:00",
            "09:37:00", "09:38:00", "09:39:00", "09:40:00",
        ]);
        let open = vec![
            Some(10.0),
            Some(10.0),
            Some(10.0),
            Some(10.0),
            Some(10.0),
            Some(10.0),
            Some(5.0),
            Some(5.0),
            Some(5.0),
            Some(5.0),
            Some(5.0),
        ];
        let high = open.clone();
        let low = open.clone();
        let close = open.clone();
        let volume = vec![
            Some(99.0),
            Some(1.0),
            Some(1.0),
            Some(1.0),
            Some(1.0),
            Some(1.0),
            Some(1.0),
            Some(1.0),
            Some(1.0),
            Some(1.0),
            Some(6.0),
        ];
        let amount = vec![
            Some(99.0),
            Some(10.0),
            Some(10.0),
            Some(10.0),
            Some(10.0),
            Some(10.0),
            Some(10.0),
            Some(10.0),
            Some(10.0),
            Some(10.0),
            Some(30.0),
        ];

        let value = rapid_current_spread_from_rows(
            &indices,
            &trade_times,
            &open,
            &high,
            &low,
            &close,
            &volume,
            &amount,
        );

        let amount_mean_all = (10.0 * 9.0 + 30.0) / 10.0;
        let volume_mean_all = (1.0 * 9.0 + 6.0) / 10.0;
        assert_eq!(value, Some(30.0 / amount_mean_all + 6.0 / volume_mean_all));
    }

    #[test]
    fn raw_returns_none_without_volume_expanding_downtrend_events() {
        let indices = (0..6).collect::<Vec<_>>();
        let trade_times = times(&[
            "09:31:00", "09:32:00", "09:33:00", "09:34:00", "09:35:00", "09:36:00",
        ]);
        let open = vec![Some(1.0); 6];
        let high = open.clone();
        let low = open.clone();
        let close = open.clone();
        let volume = vec![
            Some(6.0),
            Some(5.0),
            Some(4.0),
            Some(3.0),
            Some(2.0),
            Some(1.0),
        ];
        let amount = vec![Some(1.0); 6];

        assert_eq!(
            rapid_current_spread_from_rows(
                &indices,
                &trade_times,
                &open,
                &high,
                &low,
                &close,
                &volume,
                &amount,
            ),
            None
        );
    }
}
