use std::collections::{BTreeMap, BTreeSet};

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
use crate::operators::{cs_zscore, ts_mean, ts_std_dev};

pub const STRONG_HALF_TIDE_RAW_ID: &str = "daily_strong_half_tide_rate";
pub const WEAK_HALF_TIDE_RAW_ID: &str = "daily_weak_half_tide_rate";

const RAW_VERSION: &str = "0.2.0";
const WINDOW: usize = 20;
const NEIGHBOR_RADIUS: usize = 4;
const NEIGHBOR_WIDTH: usize = NEIGHBOR_RADIUS * 2 + 1;

pub struct StockDailyCompleteTide;

#[derive(Clone, Copy, Debug)]
struct TideRates {
    strong: Option<f64>,
    weak: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
struct TidePoint {
    neighbor_volume: f64,
    close: f64,
}

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyCompleteTide)
}

fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
    stock_minute_raw_spec(raw_id, RAW_VERSION, &["close", "vol"], 1)
}

impl Factor for StockDailyCompleteTide {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "complete_tide".to_string(),
            aliases: Vec::new(),
            name: "Complete Tide".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.2.0".to_string(),
            tags: [
                "price_volume",
                "return",
                "volume",
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
            description: "Composite intraday tide factor from 20-day mean strong half-tide rate and 20-day stability of weak half-tide rate, neutralized by Barra SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: vec![
                IntradayDailyRawRequest::new(STRONG_HALF_TIDE_RAW_ID, WINDOW - 1),
                IntradayDailyRawRequest::new(WEAK_HALF_TIDE_RAW_ID, WINDOW - 1),
            ],
            lookback: Lookback {
                trading_days: WINDOW - 1,
            },
        }
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        vec![
            raw_spec(STRONG_HALF_TIDE_RAW_ID),
            raw_spec(WEAK_HALF_TIDE_RAW_ID),
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
        let wants_strong = requested.contains(STRONG_HALF_TIDE_RAW_ID);
        let wants_weak = requested.contains(WEAK_HALF_TIDE_RAW_ID);
        if !wants_strong && !wants_weak {
            return Ok(Vec::new());
        }

        let mut strong_values = Vec::new();
        let mut weak_values = Vec::new();
        for trade_date in &context.target_dates {
            let Some(table) = data.minute(
                raw_spec(STRONG_HALF_TIDE_RAW_ID).source_dataset,
                *trade_date,
            ) else {
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
                let tide = tide_rates_from_rows(&indices, trade_times, &close, &volume);
                if wants_strong {
                    strong_values.push(FactorValue {
                        key: FactorRowKey::Daily {
                            trade_date: *trade_date,
                            ts_code: ts_code.clone(),
                        },
                        value: tide.strong,
                    });
                }
                if wants_weak {
                    weak_values.push(FactorValue {
                        key: FactorRowKey::Daily {
                            trade_date: *trade_date,
                            ts_code,
                        },
                        value: tide.weak,
                    });
                }
            }
        }

        let mut output = Vec::new();
        if wants_strong {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(STRONG_HALF_TIDE_RAW_ID),
                values: strong_values,
            });
        }
        if wants_weak {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(WEAK_HALF_TIDE_RAW_ID),
                values: weak_values,
            });
        }
        Ok(output)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let sector_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockSwClassification)?,
            ClassificationLevel::Sector,
        )?;
        let panel = data.intraday_daily_raw_panel(STRONG_HALF_TIDE_RAW_ID)?;
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;
        let strong_raw = panel.column(STRONG_HALF_TIDE_RAW_ID)?;
        let weak_raw = panel.column(WEAK_HALF_TIDE_RAW_ID)?;

        let strong_mean20 = strong_raw.ts(|values| ts_mean(values, WINDOW, WINDOW))?;
        let weak_stable20 = weak_raw.ts(|values| ts_std_dev(values, WINDOW, WINDOW))?;
        let factor = average_pair(&strong_mean20.cs(cs_zscore)?, &weak_stable20.cs(cs_zscore)?)?;
        let neutralized =
            factor.cs_neutralize_regression_by_group(&[&size], None, |trade_date, ts_codes| {
                sector_map.groups_for(trade_date, ts_codes)
            })?;

        Ok(neutralized.to_factor_series(self.spec()))
    }
}

fn average_pair(left: &PanelColumn, right: &PanelColumn) -> Result<PanelColumn> {
    left.zip_binary(right, |left, right| match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some((left + right) / 2.0),
        _ => None,
    })
}

fn tide_rates_from_rows(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
    volume: &[Option<f64>],
) -> TideRates {
    let points = tide_points_from_rows(indices, trade_times, close, volume);
    if points.len() < 2 {
        return TideRates {
            strong: None,
            weak: None,
        };
    }

    let peak_idx = max_point_position(&points);
    let rise_idx = if peak_idx > 0 {
        min_point_position(&points, 0, peak_idx)
    } else {
        peak_idx
    };
    let ebb_idx = if peak_idx + 1 < points.len() {
        min_point_position(&points, peak_idx + 1, points.len())
    } else {
        peak_idx
    };

    if rise_idx == peak_idx && ebb_idx == peak_idx {
        TideRates {
            strong: None,
            weak: None,
        }
    } else if rise_idx == peak_idx {
        let rate = half_tide_rate(&points, peak_idx, ebb_idx);
        TideRates {
            strong: rate,
            weak: rate,
        }
    } else if ebb_idx == peak_idx {
        let rate = half_tide_rate(&points, rise_idx, peak_idx);
        TideRates {
            strong: rate,
            weak: rate,
        }
    } else {
        let rise_volume = points[rise_idx].neighbor_volume;
        let ebb_volume = points[ebb_idx].neighbor_volume;
        if (rise_volume - ebb_volume).abs() <= f64::EPSILON {
            TideRates {
                strong: None,
                weak: None,
            }
        } else {
            let rise_rate = half_tide_rate(&points, rise_idx, peak_idx);
            let ebb_rate = half_tide_rate(&points, peak_idx, ebb_idx);
            if rise_volume < ebb_volume {
                TideRates {
                    strong: rise_rate,
                    weak: ebb_rate,
                }
            } else {
                TideRates {
                    strong: ebb_rate,
                    weak: rise_rate,
                }
            }
        }
    }
}

fn tide_points_from_rows(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
    volume: &[Option<f64>],
) -> Vec<TidePoint> {
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
    if selected_positions.len() < NEIGHBOR_WIDTH {
        return Vec::new();
    }

    let mut prefix_sum = vec![0.0; selected_positions.len() + 1];
    let mut prefix_count = vec![0usize; selected_positions.len() + 1];
    for (selected_idx, pos) in selected_positions.iter().enumerate() {
        let idx = indices[*pos];
        match clean_intraday_value(volume[idx]) {
            Some(vol) => {
                prefix_sum[selected_idx + 1] = prefix_sum[selected_idx] + vol;
                prefix_count[selected_idx + 1] = prefix_count[selected_idx] + 1;
            }
            None => {
                prefix_sum[selected_idx + 1] = prefix_sum[selected_idx];
                prefix_count[selected_idx + 1] = prefix_count[selected_idx];
            }
        }
    }

    let mut points = Vec::new();
    for selected_idx in NEIGHBOR_RADIUS..(selected_positions.len() - NEIGHBOR_RADIUS) {
        let start = selected_idx - NEIGHBOR_RADIUS;
        let end = selected_idx + NEIGHBOR_RADIUS + 1;
        if prefix_count[end] - prefix_count[start] != NEIGHBOR_WIDTH {
            continue;
        }
        let idx = indices[selected_positions[selected_idx]];
        let Some(center_close) = clean_intraday_value(close[idx]) else {
            continue;
        };
        points.push(TidePoint {
            neighbor_volume: prefix_sum[end] - prefix_sum[start],
            close: center_close,
        });
    }
    points
}

fn half_tide_rate(points: &[TidePoint], start_idx: usize, end_idx: usize) -> Option<f64> {
    if end_idx <= start_idx {
        return None;
    }
    let start_close = points[start_idx].close;
    let end_close = points[end_idx].close;
    if start_close.abs() <= f64::EPSILON {
        return None;
    }
    Some((end_close / start_close - 1.0) / (end_idx - start_idx) as f64)
}

fn max_point_position(points: &[TidePoint]) -> usize {
    points
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            left.neighbor_volume
                .partial_cmp(&right.neighbor_volume)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
        .expect("points is not empty")
}

fn min_point_position(points: &[TidePoint], start: usize, end: usize) -> usize {
    points
        .iter()
        .enumerate()
        .take(end)
        .skip(start)
        .min_by(|(_, left), (_, right)| {
            left.neighbor_volume
                .partial_cmp(&right.neighbor_volume)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
        .expect("point range is not empty")
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| !value.is_nan())
}
