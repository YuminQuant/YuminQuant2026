use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::Path;

use rayon::prelude::*;

use crate::artifact_io::StoragePathExt;
use crate::core::{
    DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec, FactorValue,
};
use crate::data::{parquet_io::read_parquet, DataPool, Table};
use crate::error::{err, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FutureMainSource {
    OpenInterest,
    FutMapping,
    AllContracts,
}

#[derive(Clone, Copy, Debug)]
pub struct FutureMinuteOptions {
    pub main_source: FutureMainSource,
    pub adjust_daily: bool,
    pub adjust_night: bool,
    pub adjust_morning: bool,
    pub adjust_noon: bool,
}

impl Default for FutureMinuteOptions {
    fn default() -> Self {
        Self {
            main_source: FutureMainSource::FutMapping,
            adjust_daily: true,
            adjust_night: false,
            adjust_morning: false,
            adjust_noon: false,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct FutureMinuteInput {
    options: FutureMinuteOptions,
    main: BTreeMap<i32, BTreeMap<String, String>>,
}

impl FutureMinuteInput {
    pub(crate) fn load(options: FutureMinuteOptions, root: &Path, dates: &[i32]) -> Result<Self> {
        let mut main = BTreeMap::new();
        match options.main_source {
            FutureMainSource::AllContracts => {}
            FutureMainSource::FutMapping | FutureMainSource::OpenInterest => {
                let directory = if options.main_source == FutureMainSource::FutMapping {
                    "future_data/basic/mapping"
                } else {
                    crate::derive::future_main_oi::DIRECTORY
                };
                let columns = ["trade_date", "ts_code", "mapping_ts_code"].map(str::to_string);
                if options.main_source == FutureMainSource::FutMapping {
                    let wanted = dates.iter().copied().collect::<BTreeSet<_>>();
                    let years = dates
                        .iter()
                        .map(|date| date / 10_000)
                        .collect::<BTreeSet<_>>();
                    for year in years {
                        let path = root.join(directory).join(format!("{year}.parquet"));
                        if path.storage_exists() {
                            main.extend(read_mapping(
                                &read_parquet(&path, Some(&columns))?,
                                &wanted,
                            )?);
                        }
                    }
                }
                for &date in dates {
                    if main.contains_key(&date) {
                        continue;
                    }
                    let path = root
                        .join(directory)
                        .join((date / 10_000).to_string())
                        .join(format!("{date}.parquet"));
                    if path.storage_exists() {
                        let mut day = read_mapping(
                            &read_parquet(&path, Some(&columns))?,
                            &BTreeSet::from([date]),
                        )?;
                        main.insert(date, day.remove(&date).unwrap_or_default());
                    }
                }
            }
        }
        Ok(Self { options, main })
    }
}

// Exclude continuous, index, TAS and special delivery variants from OI ranking.
pub(crate) fn product(code: &str) -> Option<String> {
    let (symbol, exchange) = code.split_once('.')?;
    let start = symbol.find(|ch: char| !ch.is_ascii_alphabetic())?;
    let month = &symbol[start..];
    if start == 0
        || !matches!(month.len(), 3 | 4)
        || !month.bytes().all(|b| b.is_ascii_digit())
        || !(1..=12).contains(&month[month.len() - 2..].parse::<u8>().ok()?)
    {
        return None;
    }
    Some(format!("{}.{exchange}", &symbol[..start]))
}

fn read_mapping(
    table: &Table,
    wanted: &BTreeSet<i32>,
) -> Result<BTreeMap<i32, BTreeMap<String, String>>> {
    let dates = table.required_i32_date_cast("trade_date")?;
    let keys = table.required_utf8("ts_code")?;
    let codes = table.required_utf8("mapping_ts_code")?;
    let mut result = BTreeMap::<i32, BTreeMap<String, String>>::new();
    for idx in 0..table.len {
        let Some(date) = dates[idx].filter(|date| wanted.contains(date)) else {
            continue;
        };
        let day = result.entry(date).or_default();
        let (Some(key), Some(code)) = (&keys[idx], &codes[idx]) else {
            continue;
        };
        if product(code).as_ref() != Some(key) {
            continue;
        }
        if day
            .insert(key.clone(), code.clone())
            .is_some_and(|old| old != *code)
        {
            return Err(err(format!(
                "conflicting fut_mapping rows for {key} on {date}"
            )));
        }
    }
    Ok(result)
}

pub fn future_minute_request(column: &str) -> DataRequest {
    DataRequest::new(DatasetId::FutureMinute1m, &["open", "close", column])
}

struct DayTail {
    first_open: Option<f64>,
    first_time: String,
    last_close: Option<f64>,
    last_time: String,
    end_scale: f64,
    values: Vec<Option<f64>>,
}

pub fn compute_future_minute_rolling(
    spec: FactorSpec,
    context: &FactorContext,
    data: &DataPool,
    column: &str,
    history_bars: usize,
    expr: impl Fn(&[Option<f64>]) -> Vec<Option<f64>> + Sync,
) -> Result<FactorSeries> {
    let input = data
        .future_minute_input()
        .ok_or_else(|| err("missing futures minute input configuration"))?;
    let is_price = matches!(column, "open" | "high" | "low" | "close");
    let targets = context
        .target_dates
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut history = VecDeque::<BTreeMap<String, DayTail>>::new();
    let mut output = Vec::new();
    for &date in &context.load_dates {
        let mut next = BTreeMap::new();
        if let Some(table) = data
            .minute(DatasetId::FutureMinute1m, date)
            .filter(|t| t.len > 0)
        {
            let dates = table.required_i32("trade_date")?;
            let codes = table.required_utf8("ts_code")?;
            let times = table.required_utf8("trade_time")?;
            let opens = table.required_f64_cast("open")?;
            let closes = table.required_f64_cast("close")?;
            let values = table.required_f64_cast(column)?;
            let mut groups = BTreeMap::<&str, Vec<usize>>::new();
            for idx in 0..table.len {
                if dates[idx] == Some(date) && times[idx].is_some() {
                    if let Some(code) = codes[idx].as_deref() {
                        groups.entry(code).or_default().push(idx);
                    }
                }
            }
            let selected = select_contracts(input, date, groups)?;
            let emit = targets.contains(&date);
            let compute = |(key, code, mut indices): (String, &str, Vec<usize>)| {
                let time = |idx: usize| times[idx].as_deref().expect("grouped minute timestamp");
                indices.sort_unstable_by(|&left, &right| time(left).cmp(time(right)));
                if let Some(pair) = indices
                    .windows(2)
                    .find(|pair| time(pair[0]) == time(pair[1]))
                {
                    return Err(err(format!(
                        "duplicate minute timestamp {} for {code} on {date}",
                        time(pair[0])
                    )));
                }
                let first = indices[0];
                let last = indices[indices.len() - 1];
                let exchange = code
                    .split_once('.')
                    .map(|(_, exchange)| exchange)
                    .unwrap_or_default();
                let mut intraday_scale = 1.0;
                let mut day_values = Vec::with_capacity(indices.len());
                let mut prev = first;
                for &idx in &indices {
                    if is_price && intraday_boundary(input.options, exchange, time(prev), time(idx))
                    {
                        intraday_scale *= boundary_ratio(closes[prev], opens[idx]).unwrap_or(1.0);
                    }
                    day_values.push(scale_value(values[idx], intraday_scale));
                    prev = idx;
                }
                let tail = DayTail {
                    first_open: opens[first],
                    first_time: time(first).to_string(),
                    last_close: closes[last],
                    last_time: time(last).to_string(),
                    end_scale: intraday_scale,
                    values: day_values[day_values.len().saturating_sub(history_bars)..].to_vec(),
                };
                let rows = if emit {
                    let mut seed = VecDeque::with_capacity(history_bars);
                    let mut previous = None;
                    let mut scale = 1.0;
                    for day in history.iter().filter_map(|day| day.get(&key)) {
                        scale = next_day_scale(
                            input.options,
                            is_price,
                            exchange,
                            previous,
                            day,
                            scale,
                            &key,
                        )?;
                        for &value in &day.values {
                            if seed.len() == history_bars {
                                seed.pop_front();
                            }
                            seed.push_back(scale_value(value, scale));
                        }
                        previous = Some(day);
                    }
                    scale = next_day_scale(
                        input.options,
                        is_price,
                        exchange,
                        previous,
                        &tail,
                        scale,
                        &key,
                    )?;
                    let skip = seed.len();
                    let series = seed
                        .into_iter()
                        .chain(
                            day_values
                                .into_iter()
                                .map(|value| scale_value(value, scale)),
                        )
                        .collect::<Vec<_>>();
                    let computed = expr(&series);
                    if computed.len() != series.len() {
                        return Err(err(format!(
                            "factor {} returned {} values for {code} on {date}, expected {}",
                            spec.id,
                            computed.len(),
                            series.len()
                        )));
                    }
                    indices
                        .into_iter()
                        .zip(computed.into_iter().skip(skip))
                        .map(|(idx, value)| FactorValue {
                            key: FactorRowKey::Minute {
                                trade_date: date,
                                trade_time: time(idx).to_string(),
                                ts_code: code.to_string(),
                            },
                            value,
                        })
                        .collect::<Vec<_>>()
                } else {
                    Vec::new()
                };
                Ok((key, tail, rows))
            };
            let results = if crate::resources::parallel_work(table.len) && selected.len() > 1 {
                selected
                    .into_par_iter()
                    .map(compute)
                    .collect::<Result<Vec<_>>>()?
            } else {
                selected
                    .into_iter()
                    .map(compute)
                    .collect::<Result<Vec<_>>>()?
            };
            for (key, tail, rows) in results {
                next.insert(key, tail);
                output.extend(rows);
            }
        }
        history.push_back(next);
        if history.len() > spec.lookback.trading_days {
            history.pop_front();
        }
    }
    Ok(FactorSeries {
        spec,
        values: output,
    })
}

fn select_contracts<'a>(
    input: &FutureMinuteInput,
    date: i32,
    mut groups: BTreeMap<&'a str, Vec<usize>>,
) -> Result<Vec<(String, &'a str, Vec<usize>)>> {
    if input.options.main_source == FutureMainSource::AllContracts {
        return Ok(groups
            .into_iter()
            .map(|(code, indices)| (code.to_string(), code, indices))
            .collect());
    }
    let Some(main) = input.main.get(&date) else {
        let task = if input.options.main_source == FutureMainSource::FutMapping {
            "sync future_mapping"
        } else {
            "run derive-future-main-oi"
        };
        return Err(err(format!(
            "missing futures main mapping on {date}; {task} including the warmup dates"
        )));
    };
    Ok(main
        .iter()
        .filter_map(|(key, code)| {
            groups
                .remove_entry(code.as_str())
                .map(|(code, indices)| (key.clone(), code, indices))
        })
        .collect())
}

fn next_day_scale(
    options: FutureMinuteOptions,
    is_price: bool,
    exchange: &str,
    previous: Option<&DayTail>,
    day: &DayTail,
    scale: f64,
    key: &str,
) -> Result<f64> {
    let Some(previous) = previous else {
        return Ok(scale);
    };
    if previous.last_time >= day.first_time {
        return Err(err(format!(
            "warmup overlaps target minutes for {key}: {} >= {}",
            previous.last_time, day.first_time
        )));
    }
    let mut scale = scale * previous.end_scale;
    if is_price
        && options.adjust_daily
        && daily_boundary(exchange, &previous.last_time, &day.first_time)
    {
        scale *= boundary_ratio(previous.last_close, day.first_open).unwrap_or(1.0);
    }
    Ok(scale)
}

fn scale_value(value: Option<f64>, scale: f64) -> Option<f64> {
    value
        .map(|value| value * scale)
        .filter(|value| value.is_finite())
}

fn boundary_ratio(close: Option<f64>, open: Option<f64>) -> Option<f64> {
    let (close, open) = (close?, open?);
    if close > 0.0 && open > 0.0 {
        let ratio = close / open;
        return (ratio.is_finite() && ratio > 0.0).then_some(ratio);
    }
    None
}

fn minute(time: &str) -> Option<u16> {
    Some(time.get(11..13)?.parse::<u16>().ok()? * 60 + time.get(14..16)?.parse::<u16>().ok()?)
}

fn near(time: u16, hour: u16, minute: u16) -> bool {
    time.abs_diff(hour * 60 + minute) <= 1
}

fn day_open(exchange: &str, time: u16) -> bool {
    if exchange == "CFX" {
        near(time, 9, 15) || near(time, 9, 30)
    } else {
        near(time, 9, 0)
    }
}

fn daily_boundary(exchange: &str, previous: &str, current: &str) -> bool {
    let (Some(previous), Some(current)) = (minute(previous), minute(current)) else {
        return false;
    };
    let close = near(previous, 15, 0) || (exchange == "CFX" && near(previous, 15, 15));
    close && (day_open(exchange, current) || near(current, 21, 0))
}

fn intraday_boundary(
    options: FutureMinuteOptions,
    exchange: &str,
    previous: &str,
    current: &str,
) -> bool {
    let (Some(previous), Some(current)) = (minute(previous), minute(current)) else {
        return false;
    };
    (options.adjust_night
        && (previous >= 21 * 60 || previous <= 3 * 60)
        && day_open(exchange, current))
        || (options.adjust_morning
            && exchange != "CFX"
            && near(previous, 10, 15)
            && near(current, 10, 30))
        || (options.adjust_noon
            && near(previous, 11, 30)
            && near(current, 13, if exchange == "CFX" { 0 } else { 30 }))
}
