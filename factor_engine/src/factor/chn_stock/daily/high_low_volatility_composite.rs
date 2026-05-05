use std::collections::BTreeMap;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::{
    clean_intraday_value, intraday_time_in_range, stock_minute_raw_spec, vector::clean, PanelColumn,
};
use crate::factor::Factor;
use crate::operators::{cs_zscore, ts_pctchg};

const RAW_ID: &str = "daily_high_low_intraday_return_volatility";
const RAW_VERSION: &str = "0.1.0";
const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;
const GROUP_COUNT: usize = 5;
const GROUP_SIZE: usize = WINDOW / GROUP_COUNT;

pub struct StockDailyHighLowVolatilityComposite;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyHighLowVolatilityComposite)
}

fn raw_spec() -> IntradayDailyRawSpec {
    stock_minute_raw_spec(RAW_ID, RAW_VERSION, &["close"], 1)
}

impl Factor for StockDailyHighLowVolatilityComposite {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "high_low_volatility_composite".to_string(),
            aliases: vec![
                "HighLowVolatilityComposite".to_string(),
                "High-Low Volatility Composite".to_string(),
            ],
            name: "High-Low Volatility Composite".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "return",
                "volatility",
                "intraday",
                "minute_agg",
                "composite",
                "neutralize",
                "barra",
                "size",
                "daily",
                "GSZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "High-Low Volatility Composite factor combining the high-price volatility ratio and high-volatility price ratio after SIZE neutralization.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
            ],
            intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(
                RAW_ID,
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
                    value: daily_intraday_return_volatility(&indices, trade_times, &close),
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
        let close = panel.column_from_table(data.daily(DatasetId::StockDailyPv)?, "close")?;
        let volatility = panel.column(RAW_ID)?;
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;

        let high_price_volatility_ratio = rolling_top_group_ratio(&volatility, &close)?;
        let high_volatility_price_ratio = rolling_top_group_ratio(&close, &volatility)?;
        let high_price_volatility_desize =
            high_price_volatility_ratio.cs_neutralize_regression(&[&size], None)?;
        let high_volatility_price_desize =
            high_volatility_price_ratio.cs_neutralize_regression(&[&size], None)?;
        let factor = add_pair(
            &high_price_volatility_desize.cs(cs_zscore)?,
            &high_volatility_price_desize.cs(cs_zscore)?,
        )?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn rolling_top_group_ratio(values: &PanelColumn, sort_values: &PanelColumn) -> Result<PanelColumn> {
    values.ts_binary(sort_values, top_group_ratio_series)
}

fn daily_intraday_return_volatility(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
) -> Option<f64> {
    let close_series = indices
        .iter()
        .filter_map(|idx| {
            let trade_time = trade_times[*idx].as_deref()?;
            intraday_time_in_range(trade_time, "09:31:00", "15:00:00")
                .then(|| clean_intraday_value(close[*idx]))
        })
        .collect::<Vec<_>>();
    let returns = ts_pctchg(&close_series, 1);
    let mut moments = MomentAccumulator::default();
    for value in returns.into_iter().filter_map(clean) {
        moments.push(value);
    }
    moments.std_dev()
}

fn top_group_ratio_series(values: &[Option<f64>], sort_values: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    for idx in 0..values.len() {
        if idx + 1 < WINDOW {
            continue;
        }
        let start = idx + 1 - WINDOW;
        let mut pairs = Vec::<(f64, usize, f64)>::with_capacity(WINDOW);
        for window_idx in start..=idx {
            let (Some(value), Some(sort_value)) =
                (clean(values[window_idx]), clean(sort_values[window_idx]))
            else {
                continue;
            };
            pairs.push((sort_value, window_idx, value));
        }
        if pairs.len() != WINDOW {
            continue;
        }
        pairs.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        });

        let denominator = pairs.iter().map(|(_, _, value)| *value).sum::<f64>() / WINDOW as f64;
        if denominator.abs() <= f64::EPSILON {
            continue;
        }
        let group_start = (GROUP_COUNT - 1) * GROUP_SIZE;
        let numerator = pairs[group_start..]
            .iter()
            .map(|(_, _, value)| *value)
            .sum::<f64>()
            / GROUP_SIZE as f64;
        output[idx] = Some(numerator / denominator);
    }
    output
}

fn add_pair(left: &PanelColumn, right: &PanelColumn) -> Result<PanelColumn> {
    left.zip_binary(right, |left, right| match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some(left + right),
        _ => None,
    })
}

#[derive(Clone, Copy, Debug, Default)]
struct MomentAccumulator {
    count: usize,
    sum: f64,
    sum_sq: f64,
}

impl MomentAccumulator {
    fn push(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.sum_sq += value * value;
    }

    fn std_dev(self) -> Option<f64> {
        if self.count < 2 {
            return None;
        }
        let n = self.count as f64;
        let variance = (self.sum_sq - self.sum * self.sum / n) / n;
        Some(variance.max(0.0).sqrt())
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
    fn top_group_ratio_takes_highest_sorted_quintile() {
        let values = (1..=20).map(|value| Some(value as f64)).collect::<Vec<_>>();
        let sort_values = (1..=20).map(|value| Some(value as f64)).collect::<Vec<_>>();

        let output = top_group_ratio_series(&values, &sort_values);

        assert_close(output[19], 18.5 / 10.5);
    }

    #[test]
    fn intraday_return_volatility_uses_own_raw_formula() {
        let indices = vec![0, 1, 2, 3, 4];
        let trade_times = vec![
            Some("09:30:00".to_string()),
            Some("09:31:00".to_string()),
            Some("09:32:00".to_string()),
            Some("15:00:00".to_string()),
            Some("15:01:00".to_string()),
        ];
        let close = vec![
            Some(10.0),
            Some(100.0),
            Some(200.0),
            Some(200.0),
            Some(10_000.0),
        ];

        let actual = daily_intraday_return_volatility(&indices, &trade_times, &close);

        assert_close(actual, 0.5);
    }

    #[test]
    fn top_group_ratio_uses_sort_values_not_input_values() {
        let values = (1..=20).map(|value| Some(value as f64)).collect::<Vec<_>>();
        let sort_values = (1..=20)
            .rev()
            .map(|value| Some(value as f64))
            .collect::<Vec<_>>();

        let output = top_group_ratio_series(&values, &sort_values);

        assert_close(output[19], 2.5 / 10.5);
    }

    #[test]
    fn top_group_ratio_requires_complete_twenty_day_pairs() {
        let values = vec![Some(1.0); WINDOW];
        let mut sort_values = vec![Some(1.0); WINDOW];
        sort_values[3] = None;

        let output = top_group_ratio_series(&values, &sort_values);

        assert_eq!(output[WINDOW - 1], None);
    }

    #[test]
    fn top_group_ratio_rejects_zero_denominator() {
        let values = vec![Some(0.0); WINDOW];
        let sort_values = (1..=20).map(|value| Some(value as f64)).collect::<Vec<_>>();

        let output = top_group_ratio_series(&values, &sort_values);

        assert_eq!(output[WINDOW - 1], None);
    }
}
