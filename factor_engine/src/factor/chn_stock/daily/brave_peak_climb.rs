use std::collections::BTreeMap;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::{
    clean_intraday_value, intraday_time_in_range, stock_minute_raw_spec, ClassificationLevel,
    ClassificationMap, PanelColumn,
};
use crate::factor::Factor;
use crate::operators::{cs_zscore, ts_mean, ts_pctchg, ts_std_dev};

pub const RAW_ID: &str = "daily_peak_climb_covariance";

const RAW_VERSION: &str = "0.2.0";
const WINDOW: usize = 20;
const PRICE_BARS: usize = 5;
const PRICE_COUNT: usize = PRICE_BARS * 4;

pub struct StockDailyBravePeakClimb;

#[derive(Clone, Copy, Debug)]
struct ClimbPoint {
    better_volatility: f64,
    return_volatility_ratio: f64,
}

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyBravePeakClimb)
}

pub fn raw_spec() -> IntradayDailyRawSpec {
    stock_minute_raw_spec(RAW_ID, RAW_VERSION, &["open", "high", "low", "close"], 1)
}

impl Factor for StockDailyBravePeakClimb {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "brave_peak_climb".to_string(),
            aliases: Vec::new(),
            name: "Brave Peak Climb".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.2.0".to_string(),
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
                "sector",
                "daily",
                "FZZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Intraday high-volatility return/volatility covariance composite, using a 20-day mean/std blend neutralized by Barra SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
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
                values.push(FactorValue {
                    key: FactorRowKey::Daily {
                        trade_date: *trade_date,
                        ts_code,
                    },
                    value: peak_climb_covariance_from_rows(
                        &indices,
                        trade_times,
                        &open,
                        &high,
                        &low,
                        &close,
                    ),
                });
            }
        }

        Ok(Some(IntradayDailyRawSeries {
            spec: raw_spec(),
            values,
        }))
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let sector_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockSwClassification)?,
            ClassificationLevel::Sector,
        )?;
        let panel = data.intraday_daily_raw_panel(RAW_ID)?;
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;
        let raw = panel.column(RAW_ID)?;

        let monthly_mean_climb = raw.ts(|values| ts_mean(values, WINDOW, 1))?;
        let monthly_stable_climb = raw.ts(|values| ts_std_dev(values, WINDOW, 1))?;
        let raw_factor = half_difference(
            &monthly_mean_climb.cs(cs_zscore)?,
            &monthly_stable_climb.cs(cs_zscore)?,
        )?;
        let neutralized = raw_factor.cs_neutralize_regression_by_group(
            &[&size],
            None,
            |trade_date, ts_codes| sector_map.groups_for(trade_date, ts_codes),
        )?;

        Ok(neutralized.to_factor_series(self.spec()))
    }
}

fn half_difference(left: &PanelColumn, right: &PanelColumn) -> Result<PanelColumn> {
    left.zip_binary(right, |left, right| match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some((left - right) / 2.0),
        _ => None,
    })
}

fn peak_climb_covariance_from_rows(
    indices: &[usize],
    trade_times: &[Option<String>],
    open: &[Option<f64>],
    high: &[Option<f64>],
    low: &[Option<f64>],
    close: &[Option<f64>],
) -> Option<f64> {
    let points = peak_climb_points_from_rows(indices, trade_times, open, high, low, close);
    let (mean, std) = mean_std(points.iter().map(|point| point.better_volatility))?;
    let threshold = mean + std;
    let selected = points
        .iter()
        .filter(|point| point.better_volatility > threshold)
        .map(|point| (point.return_volatility_ratio, point.better_volatility))
        .collect::<Vec<_>>();
    covariance(&selected)
}

fn peak_climb_points_from_rows(
    indices: &[usize],
    trade_times: &[Option<String>],
    open: &[Option<f64>],
    high: &[Option<f64>],
    low: &[Option<f64>],
    close: &[Option<f64>],
) -> Vec<ClimbPoint> {
    let close_series = indices
        .iter()
        .map(|idx| clean_intraday_value(close[*idx]))
        .collect::<Vec<_>>();
    let returns = ts_pctchg(&close_series, 1);
    let selected_positions = indices
        .iter()
        .enumerate()
        .filter_map(|(pos, idx)| {
            trade_times[*idx]
                .as_deref()
                .is_some_and(|time| intraday_time_in_range(time, "09:31:00", "14:57:00"))
                .then_some(pos)
        })
        .collect::<Vec<_>>();
    if selected_positions.len() < PRICE_BARS {
        return Vec::new();
    }

    let mut prefix_sum = vec![0.0; selected_positions.len() + 1];
    let mut prefix_sumsq = vec![0.0; selected_positions.len() + 1];
    let mut prefix_count = vec![0usize; selected_positions.len() + 1];
    for (selected_idx, pos) in selected_positions.iter().enumerate() {
        let idx = indices[*pos];
        let bar_prices = [
            clean_intraday_value(open[idx]),
            clean_intraday_value(high[idx]),
            clean_intraday_value(low[idx]),
            clean_intraday_value(close[idx]),
        ];
        let (bar_sum, bar_sumsq, bar_count) = if bar_prices.iter().all(Option::is_some) {
            let values = bar_prices
                .iter()
                .map(|value| value.expect("checked all prices are present"))
                .collect::<Vec<_>>();
            (
                values.iter().sum::<f64>(),
                values.iter().map(|value| value * value).sum::<f64>(),
                values.len(),
            )
        } else {
            (0.0, 0.0, 0)
        };
        prefix_sum[selected_idx + 1] = prefix_sum[selected_idx] + bar_sum;
        prefix_sumsq[selected_idx + 1] = prefix_sumsq[selected_idx] + bar_sumsq;
        prefix_count[selected_idx + 1] = prefix_count[selected_idx] + bar_count;
    }

    let mut points = Vec::new();
    for selected_idx in (PRICE_BARS - 1)..selected_positions.len() {
        let start = selected_idx + 1 - PRICE_BARS;
        let end = selected_idx + 1;
        let count = prefix_count[end] - prefix_count[start];
        if count != PRICE_COUNT {
            continue;
        }
        let sum = prefix_sum[end] - prefix_sum[start];
        let sumsq = prefix_sumsq[end] - prefix_sumsq[start];
        let mean = sum / PRICE_COUNT as f64;
        if mean.abs() <= f64::EPSILON {
            continue;
        }
        let variance = (sumsq / PRICE_COUNT as f64 - mean * mean).max(0.0);
        let better_volatility = variance / (mean * mean);
        if better_volatility <= f64::EPSILON || better_volatility.is_nan() {
            continue;
        }

        let pos = selected_positions[selected_idx];
        let Some(minute_return) = returns[pos] else {
            continue;
        };

        points.push(ClimbPoint {
            better_volatility,
            return_volatility_ratio: minute_return / better_volatility,
        });
    }
    points
}

fn covariance(points: &[(f64, f64)]) -> Option<f64> {
    match points.len() {
        0 => return None,
        1 => return Some(0.0),
        _ => {}
    }
    let mean_x = points.iter().map(|(x, _)| *x).sum::<f64>() / points.len() as f64;
    let mean_y = points.iter().map(|(_, y)| *y).sum::<f64>() / points.len() as f64;
    Some(
        points
            .iter()
            .map(|(x, y)| (x - mean_x) * (y - mean_y))
            .sum::<f64>()
            / points.len() as f64,
    )
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
