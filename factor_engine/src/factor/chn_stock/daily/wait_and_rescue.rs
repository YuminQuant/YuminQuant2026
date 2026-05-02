use std::cmp::Ordering;
use std::collections::BTreeMap;

use crate::core::{
    AssetClass, FactorContext, FactorRowKey, FactorSeries, FactorSpec, FactorValue, Frequency,
    IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec, Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::{clean_intraday_value, intraday_time_in_range, stock_minute_raw_spec};
use crate::factor::Factor;
use crate::operators::{cs_zscore, ts_mean, ts_std_dev};

pub const RAW_ID: &str = "daily_wait_rescue_coefficient";

const RAW_VERSION: &str = "0.1.0";
const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;
const MASSIVE_MOMENT_COUNT: usize = 10;
const ADVANTAGE_GAP: usize = 5;
const FOLLOW_WINDOW: usize = 5;

pub struct StockDailyWaitAndRescue;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyWaitAndRescue)
}

pub fn raw_spec() -> IntradayDailyRawSpec {
    stock_minute_raw_spec(RAW_ID, RAW_VERSION, &["vol"], 1)
}

impl Factor for StockDailyWaitAndRescue {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "wait_and_rescue".to_string(),
            aliases: Vec::new(),
            name: "Wait and Rescue".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "volume",
                "intraday",
                "minute_agg",
                "composite",
                "daily",
                "FZZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Intraday follow-volume coefficient after independent massive-volume moments, blended from 20-day mean and standard deviation.".to_string(),
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
            let Some(table) = data.minute(raw_spec().source_dataset, *trade_date) else {
                continue;
            };
            let ts_codes = table.required_utf8("ts_code")?;
            let trade_times = table.required_utf8("trade_time")?;
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
                    value: wait_rescue_coefficient_from_rows(&indices, trade_times, &volume),
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
        let monthly_mean = raw.ts(|values| ts_mean(values, WINDOW, 1))?;
        let monthly_std = raw.ts(|values| ts_std_dev(values, WINDOW, 1))?;
        let factor = average_pair(&monthly_mean.cs(cs_zscore)?, &monthly_std.cs(cs_zscore)?)?;

        Ok(factor.to_factor_series(self.spec()))
    }
}

fn wait_rescue_coefficient_from_rows(
    indices: &[usize],
    trade_times: &[Option<String>],
    volume: &[Option<f64>],
) -> Option<f64> {
    let selected_volume = indices
        .iter()
        .filter_map(|idx| {
            let trade_time = trade_times[*idx].as_deref()?;
            intraday_time_in_range(trade_time, "09:46:00", "15:00:00")
                .then(|| clean_intraday_value(volume[*idx]))
        })
        .collect::<Vec<_>>();
    if selected_volume.len() <= FOLLOW_WINDOW {
        return None;
    }

    let advantage_positions = advantage_positions(&selected_volume);
    let coefficients = follow_coefficients(&selected_volume, &advantage_positions);
    mean(coefficients)
}

fn advantage_positions(volume: &[Option<f64>]) -> Vec<usize> {
    filter_independent_positions(top_massive_positions(volume, MASSIVE_MOMENT_COUNT))
}

fn top_massive_positions(volume: &[Option<f64>], count: usize) -> Vec<usize> {
    let mut candidates = volume
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| clean(*value).map(|value| (idx, value)))
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    let mut positions = candidates
        .into_iter()
        .take(count)
        .map(|(idx, _)| idx)
        .collect::<Vec<_>>();
    positions.sort_unstable();
    positions
}

fn filter_independent_positions(positions: Vec<usize>) -> Vec<usize> {
    let mut output = Vec::new();
    for position in positions {
        let keep = match output.last().copied() {
            Some(previous) => position.saturating_sub(previous) > ADVANTAGE_GAP,
            None => true,
        };
        if keep {
            output.push(position);
        }
    }
    output
}

fn follow_coefficients(volume: &[Option<f64>], advantage_positions: &[usize]) -> Vec<f64> {
    let mut output = Vec::new();
    for position in advantage_positions {
        if position + FOLLOW_WINDOW >= volume.len() {
            continue;
        }
        let Some(denominator) = clean(volume[*position]) else {
            continue;
        };
        if denominator.abs() <= f64::EPSILON {
            continue;
        }
        let mut sum = 0.0;
        let mut complete = true;
        for offset in 1..=FOLLOW_WINDOW {
            let Some(value) = clean(volume[position + offset]) else {
                complete = false;
                break;
            };
            sum += value;
        }
        if complete {
            output.push(sum / denominator);
        }
    }
    output
}

fn average_pair(
    left: &crate::factor::common::PanelColumn,
    right: &crate::factor::common::PanelColumn,
) -> Result<crate::factor::common::PanelColumn> {
    left.zip_binary(right, |left, right| match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some((left + right) / 2.0),
        _ => None,
    })
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| !value.is_nan())
}

fn mean(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let values = values
        .into_iter()
        .filter(|value| !value.is_nan())
        .collect::<Vec<_>>();
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::{
        advantage_positions, filter_independent_positions, follow_coefficients,
        top_massive_positions, wait_rescue_coefficient_from_rows,
    };

    #[test]
    fn top_massive_positions_keep_earliest_ties() {
        let volume = vec![Some(10.0); 12];

        assert_eq!(
            top_massive_positions(&volume, 10),
            (0..10).collect::<Vec<_>>()
        );
    }

    #[test]
    fn advantage_filter_requires_strictly_more_than_five_minutes() {
        assert_eq!(
            filter_independent_positions(vec![0, 5, 6, 12]),
            vec![0, 6, 12]
        );
    }

    #[test]
    fn follow_coefficients_use_next_five_minutes_only() {
        let volume = vec![
            Some(10.0),
            Some(1.0),
            Some(2.0),
            Some(3.0),
            Some(4.0),
            Some(5.0),
            Some(100.0),
        ];

        assert_eq!(follow_coefficients(&volume, &[0]), vec![1.5]);
    }

    #[test]
    fn follow_coefficients_skip_tail_and_zero_denominator() {
        let volume = vec![
            Some(0.0),
            Some(1.0),
            Some(1.0),
            Some(1.0),
            Some(1.0),
            Some(1.0),
            Some(10.0),
        ];

        assert!(follow_coefficients(&volume, &[0, 6]).is_empty());
    }

    #[test]
    fn wait_rescue_raw_returns_none_without_valid_advantage() {
        let indices = (0..8).collect::<Vec<_>>();
        let trade_times = vec![
            Some("09:46:00".to_string()),
            Some("09:47:00".to_string()),
            Some("09:48:00".to_string()),
            Some("09:49:00".to_string()),
            Some("09:50:00".to_string()),
            Some("09:51:00".to_string()),
            Some("09:52:00".to_string()),
            Some("09:53:00".to_string()),
        ];
        let volume = vec![None; 8];

        assert_eq!(
            wait_rescue_coefficient_from_rows(&indices, &trade_times, &volume),
            None
        );
    }

    #[test]
    fn wait_rescue_raw_uses_filtered_intraday_window() {
        let indices = (0..8).collect::<Vec<_>>();
        let trade_times = vec![
            Some("09:45:00".to_string()),
            Some("09:46:00".to_string()),
            Some("09:47:00".to_string()),
            Some("09:48:00".to_string()),
            Some("09:49:00".to_string()),
            Some("09:50:00".to_string()),
            Some("09:51:00".to_string()),
            Some("09:52:00".to_string()),
        ];
        let volume = vec![
            Some(10_000.0),
            Some(10.0),
            Some(1.0),
            Some(2.0),
            Some(3.0),
            Some(4.0),
            Some(5.0),
            Some(100.0),
        ];

        assert_eq!(
            wait_rescue_coefficient_from_rows(&indices, &trade_times, &volume),
            Some(1.5)
        );
    }

    #[test]
    fn advantage_positions_use_top_ten_then_time_filter() {
        let volume = vec![
            Some(30.0),
            Some(29.0),
            Some(28.0),
            Some(27.0),
            Some(26.0),
            Some(25.0),
            Some(24.0),
            Some(23.0),
            Some(22.0),
            Some(21.0),
            Some(1.0),
            Some(1.0),
            Some(100.0),
        ];

        assert_eq!(advantage_positions(&volume), vec![0, 6, 12]);
    }
}
