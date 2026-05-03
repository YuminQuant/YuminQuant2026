use std::collections::{BTreeMap, BTreeSet};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::{
    clean_intraday_value, intraday_time_in_range, stock_minute_raw_spec, PanelColumn,
};
use crate::factor::Factor;
use crate::operators::{cs_zscore, ts_mean, ts_pctchg};

pub const FAIR_VOLATILITY_RAW_ID: &str = "daily_fair_volatility";
pub const FAIR_RETURN_RAW_ID: &str = "daily_fair_return";

const RAW_VERSION: &str = "0.3.0";
const VERSION: &str = "0.5.0";
const WINDOW: usize = 20;
const EVENT_WINDOW: usize = 5;

pub struct StockDailyTreatSame;

#[derive(Clone, Copy, Debug)]
struct FairValues {
    volatility: Option<f64>,
    return_mean: Option<f64>,
}

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyTreatSame)
}

fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
    stock_minute_raw_spec(raw_id, RAW_VERSION, &["close", "vol"], 1)
}

impl Factor for StockDailyTreatSame {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "treat_same".to_string(),
            aliases: Vec::new(),
            name: "Treat Same".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "return",
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
            description:
                "Composite fair volatility and fair return response to normalized intraday volume spikes and drops."
                    .to_string(),
            dependencies: vec![DataRequest::new(DatasetId::StockDailyPv, &["open", "close"])],
            intraday_raw_dependencies: vec![
                IntradayDailyRawRequest::new(FAIR_VOLATILITY_RAW_ID, WINDOW - 1),
                IntradayDailyRawRequest::new(FAIR_RETURN_RAW_ID, WINDOW - 1),
            ],
            lookback: Lookback {
                trading_days: WINDOW - 1,
            },
        }
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        vec![
            raw_spec(FAIR_VOLATILITY_RAW_ID),
            raw_spec(FAIR_RETURN_RAW_ID),
        ]
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
        let wants_volatility = requested.contains(FAIR_VOLATILITY_RAW_ID);
        let wants_return = requested.contains(FAIR_RETURN_RAW_ID);
        if !wants_volatility && !wants_return {
            return Ok(Vec::new());
        }

        let mut volatility_values = Vec::new();
        let mut return_values = Vec::new();
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
                let values = fair_values_from_rows(&indices, trade_times, &close, &volume);
                if wants_volatility {
                    volatility_values.push(FactorValue {
                        key: FactorRowKey::Daily {
                            trade_date: *trade_date,
                            ts_code: ts_code.clone(),
                        },
                        value: values.volatility,
                    });
                }
                if wants_return {
                    return_values.push(FactorValue {
                        key: FactorRowKey::Daily {
                            trade_date: *trade_date,
                            ts_code,
                        },
                        value: values.return_mean,
                    });
                }
            }
        }

        let mut output = Vec::new();
        if wants_volatility {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(FAIR_VOLATILITY_RAW_ID),
                values: volatility_values,
            });
        }
        if wants_return {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(FAIR_RETURN_RAW_ID),
                values: return_values,
            });
        }
        Ok(output)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(FAIR_VOLATILITY_RAW_ID)?;
        let open = panel.column_from_table(data.daily(DatasetId::StockDailyPv)?, "open")?;
        let close = panel.column_from_table(data.daily(DatasetId::StockDailyPv)?, "close")?;
        let intraday_return = close.zip_binary(&open, ret)?;

        let fair_volatility = panel
            .column(FAIR_VOLATILITY_RAW_ID)?
            .zip_binary(&intraday_return, multiply)?;
        let fair_return = panel
            .column(FAIR_RETURN_RAW_ID)?
            .zip_binary(&intraday_return, multiply)?;

        let vol_fair = fair_volatility.ts(|values| ts_mean(values, WINDOW, 1))?;
        let ret_fair = fair_return.ts(|values| ts_mean(values, WINDOW, 1))?;
        let raw_factor = average_pair(&vol_fair.cs(cs_zscore)?, &ret_fair.cs(cs_zscore)?)?;

        Ok(raw_factor.to_factor_series(self.spec()))
    }
}

fn average_pair(left: &PanelColumn, right: &PanelColumn) -> Result<PanelColumn> {
    left.zip_binary(right, |left, right| match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some((left + right) / 2.0),
        _ => None,
    })
}

fn fair_values_from_rows(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
    volume: &[Option<f64>],
) -> FairValues {
    let selected = indices
        .iter()
        .enumerate()
        .filter_map(|(pos, idx)| {
            trade_times[*idx]
                .as_deref()
                .is_some_and(|time| intraday_time_in_range(time, "09:31:00", "14:57:00"))
                .then_some(pos)
        })
        .collect::<Vec<_>>();
    if selected.len() < EVENT_WINDOW {
        return FairValues {
            volatility: None,
            return_mean: None,
        };
    }

    let close_series = selected
        .iter()
        .map(|pos| clean_intraday_value(close[indices[*pos]]))
        .collect::<Vec<_>>();
    let returns = ts_pctchg(&close_series, 1);
    let log_volume = selected
        .iter()
        .map(|pos| positive_log(clean_intraday_value(volume[indices[*pos]])))
        .collect::<Vec<_>>();
    let volume_diff = diff_from_previous_valid(&log_volume);
    let Some((diff_mean, diff_std)) = mean_std(volume_diff.iter().filter_map(|value| *value))
    else {
        return FairValues {
            volatility: None,
            return_mean: None,
        };
    };

    let spike_threshold = diff_mean + diff_std;
    let drop_threshold = diff_mean - diff_std;
    let mut spike_std = Vec::new();
    let mut drop_std = Vec::new();
    let mut spike_mean = Vec::new();
    let mut drop_mean = Vec::new();
    for pos in 0..selected.len() {
        let Some(diff) = volume_diff[pos] else {
            continue;
        };
        let event_return = returns[pos];
        let return_std = if pos + EVENT_WINDOW <= returns.len() {
            let window = &returns[pos..pos + EVENT_WINDOW];
            window
                .iter()
                .all(Option::is_some)
                .then(|| mean_std(window.iter().filter_map(|value| *value)).map(|(_, std)| std))
                .flatten()
        } else {
            None
        };
        if diff > spike_threshold {
            if let Some(return_std) = return_std {
                spike_std.push(return_std);
            }
            if let Some(event_return) = event_return {
                spike_mean.push(event_return);
            }
        } else if diff < drop_threshold {
            if let Some(return_std) = return_std {
                drop_std.push(return_std);
            }
            if let Some(event_return) = event_return {
                drop_mean.push(event_return);
            }
        }
    }

    FairValues {
        volatility: abs_diff_mean(&spike_std, &drop_std),
        return_mean: abs_diff_mean(&spike_mean, &drop_mean),
    }
}

fn diff_from_previous_valid(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    let mut previous_valid: Option<f64> = None;
    for (idx, value) in values.iter().enumerate() {
        let Some(current) = clean(*value) else {
            continue;
        };
        if let Some(previous) = previous_valid {
            output[idx] = Some(current - previous);
        }
        previous_valid = Some(current);
    }
    output
}

fn abs_diff_mean(left: &[f64], right: &[f64]) -> Option<f64> {
    Some((mean(left.iter().copied())? - mean(right.iter().copied())?).abs())
}

fn ret(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (clean(numerator), clean(denominator)) {
        (Some(numerator), Some(denominator)) if denominator.abs() > f64::EPSILON => {
            Some(numerator / denominator - 1.0)
        }
        _ => None,
    }
}

fn multiply(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    Some(clean(left)? * clean(right)?)
}

fn positive_log(value: Option<f64>) -> Option<f64> {
    clean(value).filter(|value| *value > 0.0).map(f64::ln)
}

fn mean(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let values = values
        .into_iter()
        .filter(|value| !value.is_nan())
        .collect::<Vec<_>>();
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn mean_std(values: impl IntoIterator<Item = f64>) -> Option<(f64, f64)> {
    let values = values
        .into_iter()
        .filter(|value| !value.is_nan())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    Some((mean, variance.sqrt()))
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| !value.is_nan())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_volume_diff_uses_previous_valid_and_keeps_true_zero_diff() {
        let values = vec![
            Some(10.0_f64.ln()),
            None,
            None,
            Some(20.0_f64.ln()),
            Some(20.0_f64.ln()),
            Some(20.0_f64.ln()),
            Some(20.0_f64.ln()),
            Some(5.0_f64.ln()),
        ];
        let diff = diff_from_previous_valid(&values);

        assert_eq!(diff[0], None);
        assert_eq!(diff[1], None);
        assert_eq!(diff[2], None);
        assert!(diff[3].is_some_and(|value| value > 0.0));
        assert_eq!(diff[4], Some(0.0));
        assert_eq!(diff[5], Some(0.0));
        assert!(diff[7].is_some_and(|value| value < 0.0));
    }

    #[test]
    fn log_volume_diff_does_not_add_zero_for_missing_values() {
        let values = vec![Some(10.0_f64.ln()), None, None, Some(20.0_f64.ln())];
        let diff = diff_from_previous_valid(&values);

        assert_eq!(diff[0], None);
        assert_eq!(diff[1], None);
        assert_eq!(diff[2], None);
        assert!((diff[3].unwrap() - (20.0_f64 / 10.0).ln()).abs() < 1e-12);
    }

    #[test]
    fn fair_values_use_spike_and_drop_complete_five_minute_windows() {
        let indices = (0..8).collect::<Vec<_>>();
        let times = vec![
            Some("09:31:00".to_string()),
            Some("09:32:00".to_string()),
            Some("09:33:00".to_string()),
            Some("09:34:00".to_string()),
            Some("09:35:00".to_string()),
            Some("09:36:00".to_string()),
            Some("09:37:00".to_string()),
            Some("14:58:00".to_string()),
        ];
        let close = vec![
            Some(100.0),
            Some(101.0),
            Some(99.0),
            Some(102.0),
            Some(103.0),
            Some(101.0),
            Some(104.0),
            Some(105.0),
        ];
        let volume = vec![
            Some(1.0),
            Some(10.0_f64.exp()),
            Some(1.0),
            Some(1.0_f64.exp()),
            Some(2.0_f64.exp()),
            Some(3.0_f64.exp()),
            Some(4.0_f64.exp()),
            Some(1000.0),
        ];

        let values = fair_values_from_rows(&indices, &times, &close, &volume);

        assert!(values.volatility.is_some());
        assert!(values.return_mean.is_some());
    }

    #[test]
    fn fair_return_uses_event_minute_return_not_five_minute_mean() {
        let indices = (0..8).collect::<Vec<_>>();
        let times = (31..=38)
            .map(|minute| Some(format!("09:{minute:02}:00")))
            .collect::<Vec<_>>();
        let close = vec![
            Some(100.0),
            Some(110.0),
            Some(88.0),
            Some(90.0),
            Some(91.0),
            Some(92.0),
            Some(93.0),
            Some(94.0),
        ];
        let volume = [0.0_f64, 10.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
            .into_iter()
            .map(|value| Some(value.exp()))
            .collect::<Vec<_>>();

        let values = fair_values_from_rows(&indices, &times, &close, &volume);

        assert!((values.return_mean.unwrap() - 0.3).abs() < 1e-12);
        assert!(values.volatility.is_some());
    }

    #[test]
    fn fair_values_require_both_spike_and_drop_sides() {
        let indices = (0..7).collect::<Vec<_>>();
        let times = (31..=37)
            .map(|minute| Some(format!("09:{minute:02}:00")))
            .collect::<Vec<_>>();
        let close = vec![
            Some(100.0),
            Some(101.0),
            Some(102.0),
            Some(103.0),
            Some(104.0),
            Some(105.0),
            Some(106.0),
        ];
        let volume = vec![Some(100.0); 7];

        let values = fair_values_from_rows(&indices, &times, &close, &volume);

        assert_eq!(values.volatility, None);
        assert_eq!(values.return_mean, None);
    }
}
