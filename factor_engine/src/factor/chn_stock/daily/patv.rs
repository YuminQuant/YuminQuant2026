use std::collections::BTreeMap;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawAuxiliaryRequest, IntradayDailyRawRequest,
    IntradayDailyRawSeries, IntradayDailyRawSpec, Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::is_bj_stock;
use crate::factor::common::{clean_intraday_value, stock_derived_bar_raw_spec};
use crate::factor::Factor;
use crate::operators::{cs_pctrank, ts_ewm};

const VERSION: &str = "0.1.0";
const RAW_VERSION: &str = "0.2.0";
const RAW_ID: &str = "daily_patv_raw";
const PROVIDER_KEY: &str = "patv_provider";

const MINUTES_PER_DAY: usize = 240;
const FIVE_MINUTE_BARS: usize = 48;
const FIVE_MINUTE_BAR_SIZE: usize = 5;
const EMA_SPAN: usize = 20;
const MIN_PERIODS: usize = 1;
const EPS: f64 = 1e-12;

pub struct StockDailyPatv;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyPatv)
}

impl Factor for StockDailyPatv {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "patv".to_string(),
            aliases: vec!["PATV".to_string()],
            name: "patv".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "PATV factor from 1-minute volume aggregated into 5-minute bars: each bar's volume share is ranked cross-sectionally by time slot; per-stock daily raw is mean(rank) / std(rank) plus kurtosis(rank), followed by a 20-day EMA.".to_string(),
            dependencies: Vec::new(),
            intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(RAW_ID, EMA_SPAN - 1)],
            lookback: Lookback {
                trading_days: EMA_SPAN - 1,
            },
        }
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        vec![raw_spec()]
    }

    fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
        PROVIDER_KEY.to_string()
    }

    fn intraday_raw_auxiliary_requirements(
        &self,
        raw_ids: &[String],
    ) -> Vec<IntradayDailyRawAuxiliaryRequest> {
        if raw_ids.iter().any(|raw_id| raw_id == RAW_ID) {
            vec![IntradayDailyRawAuxiliaryRequest::new(
                DataRequest::new(DatasetId::StockMinute1m, &["vol"]),
                0,
            )]
        } else {
            Vec::new()
        }
    }

    fn minute_compute_many(
        &self,
        raw_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<IntradayDailyRawSeries>> {
        if !raw_ids.iter().any(|raw_id| raw_id == RAW_ID) {
            return Ok(Vec::new());
        }

        let mut values = Vec::<FactorValue>::new();
        for trade_date in &context.target_dates {
            let daily_values = match data.derived_bar(FIVE_MINUTE_BAR_SIZE, *trade_date) {
                Some(table) => match daily_raw_values_from_derived_bar(table) {
                    Ok(Some(values)) => values,
                    Ok(None) | Err(_) => {
                        eprintln!(
                            "warning: patv falling back to 1m minute data for {trade_date}; derived 5m bar is incomplete or incompatible"
                        );
                        daily_raw_values_from_minute(data, *trade_date)?
                    }
                },
                None => {
                    eprintln!(
                        "warning: patv falling back to 1m minute data for {trade_date}; derived 5m bar is missing"
                    );
                    daily_raw_values_from_minute(data, *trade_date)?
                }
            };

            for (ts_code, value) in daily_values {
                values.push(FactorValue {
                    key: FactorRowKey::Daily {
                        trade_date: *trade_date,
                        ts_code,
                    },
                    value,
                });
            }
        }

        Ok(vec![IntradayDailyRawSeries {
            spec: raw_spec(),
            values,
        }])
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(RAW_ID)?;
        let raw = panel.column(RAW_ID)?;
        let factor = raw.ts(|series| ts_ewm(series, EMA_SPAN, MIN_PERIODS))?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

#[derive(Clone, Debug)]
struct MinuteStockDay {
    volume: [Option<f64>; MINUTES_PER_DAY],
}

impl Default for MinuteStockDay {
    fn default() -> Self {
        Self {
            volume: [None; MINUTES_PER_DAY],
        }
    }
}

#[derive(Clone, Debug)]
struct ComputedStockDay {
    ts_code: String,
    volume_share: [Option<f64>; FIVE_MINUTE_BARS],
}

fn raw_spec() -> IntradayDailyRawSpec {
    stock_derived_bar_raw_spec(RAW_ID, RAW_VERSION, FIVE_MINUTE_BAR_SIZE, &["volume"], 1)
}

fn tags() -> Vec<String> {
    [
        "price_volume",
        "ZSZQ",
        "PATV",
        "volume",
        "intraday",
        "minute_agg",
        "rank",
        "kurtosis",
        "ema",
        "daily",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect()
}

fn daily_raw_values(stock_days: &BTreeMap<String, MinuteStockDay>) -> Vec<(String, Option<f64>)> {
    let computed = computed_stock_days(stock_days);
    let ranks = slot_volume_share_ranks(&computed);
    computed
        .into_iter()
        .zip(ranks)
        .map(|(day, ranks)| (day.ts_code, patv_raw(&ranks)))
        .collect()
}

fn daily_raw_values_from_minute(
    data: &DataPool,
    trade_date: i32,
) -> Result<Vec<(String, Option<f64>)>> {
    let Some(table) = data.minute(DatasetId::StockMinute1m, trade_date) else {
        return Ok(Vec::new());
    };
    let ts_codes = table.required_utf8("ts_code")?;
    let trade_times = table.required_utf8("trade_time")?;
    let volume = table.required_f64_cast("vol")?;

    let mut stock_days = BTreeMap::<String, MinuteStockDay>::new();
    for idx in 0..table.len {
        let Some(ts_code) = ts_codes[idx].clone() else {
            continue;
        };
        if is_bj_stock(&ts_code) {
            continue;
        }
        let Some(trade_time) = trade_times[idx].as_deref() else {
            continue;
        };
        let Some(minute_idx) = minute_index(trade_time) else {
            continue;
        };
        stock_days.entry(ts_code).or_default().volume[minute_idx] =
            clean_intraday_value(volume[idx]).filter(|value| *value >= 0.0);
    }
    Ok(daily_raw_values(&stock_days))
}

fn daily_raw_values_from_derived_bar(
    table: &crate::data::Table,
) -> Result<Option<Vec<(String, Option<f64>)>>> {
    if !["ts_code", "bar_index", "minute_count", "volume"]
        .iter()
        .all(|column| table.columns.contains_key(*column))
    {
        return Ok(None);
    }
    let ts_codes = table.required_utf8("ts_code")?;
    let bar_indices = table.required_i32("bar_index")?;
    let minute_counts = table.required_i32("minute_count")?;
    let volume = table.required_f64_cast("volume")?;
    let mut by_stock = BTreeMap::<String, [Option<f64>; FIVE_MINUTE_BARS]>::new();
    for idx in 0..table.len {
        let Some(ts_code) = ts_codes[idx].clone() else {
            continue;
        };
        if is_bj_stock(&ts_code) {
            continue;
        }
        let Some(bar_index) = bar_indices[idx].and_then(|value| usize::try_from(value).ok()) else {
            continue;
        };
        if bar_index >= FIVE_MINUTE_BARS || minute_counts[idx] != Some(FIVE_MINUTE_BAR_SIZE as i32)
        {
            continue;
        }
        by_stock.entry(ts_code).or_insert([None; FIVE_MINUTE_BARS])[bar_index] =
            clean_intraday_value(volume[idx]).filter(|value| *value >= 0.0);
    }
    let computed = computed_stock_days_from_volumes(&by_stock);
    let ranks = slot_volume_share_ranks(&computed);
    Ok(Some(
        computed
            .into_iter()
            .zip(ranks)
            .map(|(day, ranks)| (day.ts_code, patv_raw(&ranks)))
            .collect(),
    ))
}

fn computed_stock_days(stock_days: &BTreeMap<String, MinuteStockDay>) -> Vec<ComputedStockDay> {
    stock_days
        .iter()
        .map(|(ts_code, day)| {
            let volumes = five_minute_volumes(day);
            let total_volume = volumes
                .iter()
                .filter_map(|value| clean_intraday_value(*value))
                .sum::<f64>();
            let mut volume_share = [None; FIVE_MINUTE_BARS];
            if volumes.iter().all(Option::is_some) && total_volume > EPS {
                for idx in 0..FIVE_MINUTE_BARS {
                    volume_share[idx] = clean_intraday_value(volumes[idx]).and_then(|value| {
                        let share = value / total_volume;
                        share.is_finite().then_some(share)
                    });
                }
            }
            ComputedStockDay {
                ts_code: ts_code.clone(),
                volume_share,
            }
        })
        .collect()
}

fn computed_stock_days_from_volumes(
    stock_days: &BTreeMap<String, [Option<f64>; FIVE_MINUTE_BARS]>,
) -> Vec<ComputedStockDay> {
    stock_days
        .iter()
        .map(|(ts_code, volumes)| {
            let total_volume = volumes
                .iter()
                .filter_map(|value| clean_intraday_value(*value))
                .sum::<f64>();
            let mut volume_share = [None; FIVE_MINUTE_BARS];
            if volumes.iter().all(Option::is_some) && total_volume > EPS {
                for idx in 0..FIVE_MINUTE_BARS {
                    volume_share[idx] = clean_intraday_value(volumes[idx]).and_then(|value| {
                        let share = value / total_volume;
                        share.is_finite().then_some(share)
                    });
                }
            }
            ComputedStockDay {
                ts_code: ts_code.clone(),
                volume_share,
            }
        })
        .collect()
}

fn five_minute_volumes(day: &MinuteStockDay) -> [Option<f64>; FIVE_MINUTE_BARS] {
    let mut output = [None; FIVE_MINUTE_BARS];
    for slot in 0..FIVE_MINUTE_BARS {
        let start = slot * FIVE_MINUTE_BAR_SIZE;
        let mut sum = 0.0;
        let mut count = 0usize;
        for idx in start..start + FIVE_MINUTE_BAR_SIZE {
            if let Some(value) = clean_intraday_value(day.volume[idx]).filter(|value| *value >= 0.0)
            {
                sum += value;
                count += 1;
            }
        }
        if count == FIVE_MINUTE_BAR_SIZE {
            output[slot] = Some(sum);
        }
    }
    output
}

fn slot_volume_share_ranks(rows: &[ComputedStockDay]) -> Vec<[Option<f64>; FIVE_MINUTE_BARS]> {
    let mut output = vec![[None; FIVE_MINUTE_BARS]; rows.len()];
    for slot in 0..FIVE_MINUTE_BARS {
        let values = rows
            .iter()
            .map(|row| row.volume_share[slot])
            .collect::<Vec<_>>();
        let ranks = cs_pctrank(&values, true);
        for (row_idx, rank) in ranks.into_iter().enumerate() {
            output[row_idx][slot] = rank;
        }
    }
    output
}

fn patv_raw(ranks: &[Option<f64>; FIVE_MINUTE_BARS]) -> Option<f64> {
    if !ranks.iter().all(Option::is_some) {
        return None;
    }
    let values = ranks
        .iter()
        .filter_map(|value| clean_intraday_value(*value))
        .collect::<Vec<_>>();
    if values.len() != FIVE_MINUTE_BARS {
        return None;
    }
    let mean = mean(&values)?;
    let std = std_dev(&values)?;
    if std <= EPS {
        return None;
    }
    let kurt = kurtosis(&values)?;
    let value = mean / std + kurt;
    value.is_finite().then_some(value)
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let value = values.iter().sum::<f64>() / values.len() as f64;
    value.is_finite().then_some(value)
}

fn std_dev(values: &[f64]) -> Option<f64> {
    let mean = mean(values)?;
    let variance = values
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64;
    variance
        .max(0.0)
        .sqrt()
        .is_finite()
        .then_some(variance.max(0.0).sqrt())
}

fn kurtosis(values: &[f64]) -> Option<f64> {
    let mean = mean(values)?;
    let mut m2 = 0.0;
    let mut m4 = 0.0;
    for value in values {
        let diff = value - mean;
        let diff2 = diff * diff;
        m2 += diff2;
        m4 += diff2 * diff2;
    }
    m2 /= values.len() as f64;
    m4 /= values.len() as f64;
    if m2 <= EPS {
        return None;
    }
    let value = m4 / (m2 * m2);
    value.is_finite().then_some(value)
}

fn minute_index(trade_time: &str) -> Option<usize> {
    let total = time_to_minutes(trade_time)?;
    let morning_start = 9 * 60 + 31;
    let morning_end = 11 * 60 + 30;
    let afternoon_start = 13 * 60 + 1;
    let afternoon_end = 15 * 60;
    if (morning_start..=morning_end).contains(&total) {
        Some((total - morning_start) as usize)
    } else if (afternoon_start..=afternoon_end).contains(&total) {
        Some(120 + (total - afternoon_start) as usize)
    } else {
        None
    }
}

fn time_to_minutes(value: &str) -> Option<i32> {
    let value = value.trim();
    let value = value
        .rsplit_once(' ')
        .map(|(_, right)| right)
        .or_else(|| value.rsplit_once('T').map(|(_, right)| right))
        .unwrap_or(value)
        .trim();
    if value.len() < 5 {
        return None;
    }
    let hour = value.get(0..2)?.parse::<i32>().ok()?;
    let minute = value.get(3..5)?.parse::<i32>().ok()?;
    Some(hour * 60 + minute)
}
