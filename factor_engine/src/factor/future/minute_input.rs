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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FutureWarmupSource {
    MainSequence,
    SameContract,
}

#[derive(Clone, Copy, Debug)]
pub struct FutureMinuteOptions {
    pub main_source: FutureMainSource,
    pub main_only: bool,
    pub warmup_source: FutureWarmupSource,
    pub adjust_daily: bool,
    pub adjust_night: bool,
    pub adjust_morning: bool,
    pub adjust_noon: bool,
}

impl Default for FutureMinuteOptions {
    fn default() -> Self {
        Self {
            main_source: FutureMainSource::FutMapping,
            main_only: true,
            warmup_source: FutureWarmupSource::MainSequence,
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
    multipliers: BTreeMap<String, f64>,
}

impl FutureMinuteInput {
    pub(crate) fn options(&self) -> FutureMinuteOptions {
        self.options
    }

    pub(crate) fn load(options: FutureMinuteOptions, root: &Path, dates: &[i32]) -> Result<Self> {
        let mut main = BTreeMap::new();
        match if options.main_only {
            options.main_source
        } else {
            FutureMainSource::AllContracts
        } {
            FutureMainSource::AllContracts => {}
            FutureMainSource::FutMapping | FutureMainSource::OpenInterest => {
                let directory = if options.main_source == FutureMainSource::FutMapping {
                    "future_data/basic/mapping"
                } else {
                    crate::derive::future_main_oi::DIRECTORY
                };
                let columns = ["trade_date", "ts_code", "mapping_ts_code"].map(str::to_string);
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
        Ok(Self { options, main, multipliers: BTreeMap::new() })
    }

    pub(crate) fn load_multipliers(&mut self, root: &Path) -> Result<()> {
        let path = root.join("future_data/basic/fut_basic.parquet");
        if path.storage_exists() {
            let columns = ["ts_code", "multiplier"].map(str::to_string);
            let table = read_parquet(&path, Some(&columns))?;
            if table.columns.contains_key("multiplier") {
                let codes = table.required_utf8("ts_code")?;
                let multipliers = table.required_f64_cast("multiplier")?;
                self.multipliers = codes.iter().zip(multipliers).filter_map(|(code, value)| {
                    Some((code.clone()?, value.filter(|v| v.is_finite() && *v > 0.0)?))
                }).collect();
            }
        }
        Ok(())
    }

    pub(crate) fn selection(&self, date: i32) -> Option<&BTreeMap<String, String>> {
        self.main.get(&date).filter(|mapping| !mapping.is_empty())
    }

    pub(crate) fn history_contract<'a>(
        &'a self, date: i32, code: &'a str, product: Option<&str>, same_contract: bool,
    ) -> Option<&'a str> {
        match (same_contract, product, self.selection(date)) {
            (false, Some(product), Some(mapping)) => mapping.get(product).map(String::as_str),
            _ => Some(code),
        }
    }

    pub(crate) fn full_contracts(
        &self,
        date: i32,
        is_target: bool,
        later_targets: impl Iterator<Item = i32>,
        force_same_contract: bool,
    ) -> Option<BTreeSet<String>> {
        let same_contract = force_same_contract
            || self.options.warmup_source == FutureWarmupSource::SameContract;
        let mut codes = BTreeSet::new();
        if is_target || !same_contract {
            codes.extend(self.selection(date)?.values().cloned());
        }
        for target in later_targets {
            // A target with no map can need history for any contract.
            let mapping = self.selection(target)?;
            if same_contract {
                codes.extend(mapping.values().cloned());
            }
        }
        Some(codes)
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

/// Select a raw bar field or its one-bar difference within the actual contract.
/// The first bar uses the preceding trading day's final raw observation.
#[derive(Clone, Copy)]
pub enum FutureMinuteField {
    Raw(&'static str),
    ContractDiff(&'static str),
    /// A level whose history must stay on the actual contract even in main-sequence mode.
    ContractLevel(&'static str),
    /// Amount divided by contract multiplier, in price times lots.
    VwapNotional,
}

impl FutureMinuteField {
    const DIFF_PREFIX: &str = "contract_diff:";
    const CONTRACT_PREFIX: &str = "contract_level:";
    pub(crate) const VWAP_NOTIONAL: &str = "vwap_notional";

    fn raw(self) -> &'static str {
        match self {
            Self::Raw(name) | Self::ContractDiff(name) | Self::ContractLevel(name) => name,
            Self::VwapNotional => "amount",
        }
    }

    pub(crate) fn diff_source(requested: &str) -> Option<&str> {
        requested.strip_prefix(Self::DIFF_PREFIX)
    }

    pub(crate) fn contract_source(requested: &str) -> Option<&str> {
        requested.strip_prefix(Self::CONTRACT_PREFIX)
    }
}

pub fn future_minute_request(fields: &[FutureMinuteField]) -> DataRequest {
    let mut request = DataRequest::new(DatasetId::FutureMinute1m, &["open", "close"]);
    for &field in fields {
        let name = match field {
            FutureMinuteField::Raw(name) => name.to_string(),
            FutureMinuteField::ContractDiff(name) => format!("{}{name}", FutureMinuteField::DIFF_PREFIX),
            FutureMinuteField::ContractLevel(name) => format!("{}{name}", FutureMinuteField::CONTRACT_PREFIX),
            FutureMinuteField::VwapNotional => FutureMinuteField::VWAP_NOTIONAL.to_string(),
        };
        if !request.columns.contains(&name) {
            request.columns.push(name);
        }
    }
    request
}

pub(crate) type ContractTails = BTreeMap<String, BTreeMap<String, Option<f64>>>;

#[derive(Clone, Debug)]
pub(crate) struct FutureMinuteDayIndex {
    pub(crate) groups: BTreeMap<String, Vec<usize>>,
    pub(crate) tails: ContractTails,
}

impl FutureMinuteDayIndex {
    pub(crate) fn build(
        date: i32,
        table: &Table,
        diff_columns: &[String],
        mut tails: ContractTails,
    ) -> Result<Self> {
        let mut groups = BTreeMap::<&str, Vec<usize>>::new();
        if table.len > 0 {
            let dates = table.required_i32("trade_date")?;
            let codes = table.required_utf8("ts_code")?;
            let times = table.required_utf8("trade_time")?;
            for idx in 0..table.len {
                if dates[idx] == Some(date) && times[idx].is_some() {
                    if let Some(code) = codes[idx].as_deref() {
                        groups.entry(code).or_default().push(idx);
                    }
                }
            }
            for indices in groups.values_mut() {
                indices.sort_unstable_by_key(|&idx| (times[idx].as_deref(), idx));
            }
            for name in diff_columns {
                let values = table.required_f64_cast(name)?;
                let tails = tails.entry(name.clone()).or_default();
                for (&code, indices) in &groups {
                    tails.insert(code.to_string(), values[indices[indices.len() - 1]]);
                }
            }
        }
        let groups = groups.into_iter().map(|(code, rows)| (code.to_string(), rows)).collect();
        Ok(Self { groups, tails })
    }
}

struct DayTail {
    first_open: Option<f64>,
    first_time: String,
    last_close: Option<f64>,
    last_time: String,
    end_scale: f64,
    values: Vec<Vec<Option<f64>>>,
}

pub fn compute_future_minute_rolling(
    spec: FactorSpec,
    context: &FactorContext,
    data: &DataPool,
    field: FutureMinuteField,
    history_bars: usize,
    expr: impl Fn(&[Option<f64>]) -> Vec<Option<f64>> + Sync,
) -> Result<FactorSeries> {
    compute_future_minute(spec, context, data, &[field], history_bars, |inputs| expr(&inputs[0]))
}

pub fn compute_future_minute(
    spec: FactorSpec,
    context: &FactorContext,
    data: &DataPool,
    fields: &[FutureMinuteField],
    history_bars: usize,
    expr: impl Fn(&[Vec<Option<f64>>]) -> Vec<Option<f64>> + Sync,
) -> Result<FactorSeries> {
    let mut result = compute_future_minute_many(&[spec], context, data, fields, history_bars, |inputs| {
        vec![expr(inputs)]
    })?;
    Ok(result.remove(0))
}

pub fn compute_future_minute_many(
    specs: &[FactorSpec],
    context: &FactorContext,
    data: &DataPool,
    fields: &[FutureMinuteField],
    history_bars: usize,
    expr: impl Fn(&[Vec<Option<f64>>]) -> Vec<Vec<Option<f64>>> + Sync,
) -> Result<Vec<FactorSeries>> {
    if fields.is_empty() {
        return Err(err("futures minute factors require at least one input field"));
    }
    let lookback = specs
        .iter()
        .map(|spec| spec.lookback.trading_days)
        .max()
        .unwrap_or(0);
    let input = data
        .future_minute_input()
        .ok_or_else(|| err("missing futures minute input configuration"))?;
    let price_columns = fields
        .iter()
        .map(|field| matches!(field, FutureMinuteField::Raw("open" | "high" | "low" | "close") | FutureMinuteField::VwapNotional))
        .collect::<Vec<_>>();
    let force_same_contract = fields.iter().any(|field| matches!(field, FutureMinuteField::ContractLevel(_)));
    let same_contract = force_same_contract || input.options.warmup_source == FutureWarmupSource::SameContract;
    let is_price = price_columns.iter().any(|&price| price);
    let mut previous_day = None;
    let targets = context
        .target_dates
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut history = VecDeque::<(i32, BTreeMap<String, DayTail>)>::new();
    let mut output = specs
        .iter()
        .cloned()
        .map(|spec| FactorSeries {
            spec,
            values: Vec::new(),
        })
        .collect::<Vec<_>>();
    for (day_idx, &date) in context.load_dates.iter().enumerate() {
        let mut next = BTreeMap::new();
        let day_index = data.future_minute_day_index(date);
        if let Some(table) = data
            .minute(DatasetId::FutureMinute1m, date)
            .filter(|t| t.len > 0)
        {
            let day_index = day_index.ok_or_else(|| err("missing futures minute day index"))?;
            let times = table.required_utf8("trade_time")?;
            let opens = table.required_f64_cast("open")?;
            let closes = table.required_f64_cast("close")?;
            let values = fields
                .iter()
                .map(|field| {
                    if matches!(field, FutureMinuteField::VwapNotional) && !table.columns.contains_key("amount") {
                        Ok(vec![None; table.len])
                    } else {
                        table.required_f64_cast(field.raw())
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            let current_main = input.selection(date).map(|mapping| {
                mapping
                    .iter()
                    .map(|(product, code)| (code.as_str(), product.as_str()))
                    .collect::<BTreeMap<_, _>>()
            });
            let target = targets.contains(&date);
            let keep_codes = input.full_contracts(
                date,
                target,
                context.load_dates.iter()
                    .skip(day_idx + 1)
                    .take(lookback)
                    .copied()
                    .filter(|future| targets.contains(future)),
                force_same_contract,
            );
            let selected = day_index.groups
                .iter()
                .filter_map(|(code, indices)| {
                    let code = code.as_str();
                    let product = current_main
                        .as_ref()
                        .and_then(|mapping| mapping.get(code))
                        .copied();
                    let in_universe = current_main.is_none() || product.is_some();
                    keep_codes
                        .as_ref()
                        .is_none_or(|codes| codes.contains(code))
                        .then_some((code, product, target && in_universe, indices.as_slice()))
                })
                .collect::<Vec<_>>();
            let compute = |(code, product, emit, indices): (
                &str,
                Option<&str>,
                bool,
                &[usize],
            )| {
                let time = |idx: usize| times[idx].as_deref().expect("grouped minute timestamp");
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
                let mut day_values = fields
                    .iter()
                    .map(|_| Vec::with_capacity(indices.len()))
                    .collect::<Vec<_>>();
                let mut prev = first;
                let multiplier = input.multipliers.get(code).copied();
                for (position, &idx) in indices.iter().enumerate() {
                    if is_price && intraday_boundary(input.options, exchange, time(prev), time(idx))
                    {
                        intraday_scale *= boundary_ratio(closes[prev], opens[idx]).unwrap_or(1.0);
                    }
                    for (column_idx, &field) in fields.iter().enumerate() {
                        let value = values[column_idx][idx];
                        let value = if let FutureMinuteField::ContractDiff(name) = field {
                            let prior = if position == 0 {
                                previous_day
                                    .and_then(|day: &FutureMinuteDayIndex| day.tails.get(name))
                                    .and_then(|tails| tails.get(code))
                                    .copied()
                                    .flatten()
                            } else {
                                values[column_idx][prev]
                            };
                            value.zip(prior).map(|(current, prior)| current - prior)
                        } else if matches!(field, FutureMinuteField::VwapNotional) {
                            value.filter(|amount| *amount > 0.0).zip(multiplier)
                                .map(|(amount, multiplier)| amount / multiplier)
                        } else {
                            value
                        };
                        day_values[column_idx].push(scale_value(
                            value,
                            if price_columns[column_idx] {
                                intraday_scale
                            } else {
                                1.0
                            },
                        ));
                    }
                    prev = idx;
                }
                let tail = DayTail {
                    first_open: opens[first],
                    first_time: time(first).to_string(),
                    last_close: closes[last],
                    last_time: time(last).to_string(),
                    end_scale: intraday_scale,
                    values: day_values
                        .iter()
                        .map(|values| values[values.len().saturating_sub(history_bars)..].to_vec())
                        .collect(),
                };
                let rows = if emit {
                    let mut seeds = fields
                        .iter()
                        .map(|_| VecDeque::with_capacity(history_bars))
                        .collect::<Vec<_>>();
                    let mut previous = None;
                    let mut scale = 1.0;
                    for day in history.iter().filter_map(|(history_date, tails)| {
                        let history_code = input.history_contract(*history_date, code, product, same_contract)?;
                        tails.get(history_code)
                    }) {
                        scale = next_day_scale(
                            input.options,
                            is_price,
                            exchange,
                            previous,
                            day,
                            scale,
                            code,
                        )?;
                        for ((seed, values), &price) in
                            seeds.iter_mut().zip(&day.values).zip(&price_columns)
                        {
                            for &value in values {
                                if seed.len() == history_bars {
                                    seed.pop_front();
                                }
                                seed.push_back(scale_value(value, if price { scale } else { 1.0 }));
                            }
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
                        code,
                    )?;
                    let skip = seeds[0].len();
                    let series = seeds
                        .into_iter()
                        .zip(day_values)
                        .zip(&price_columns)
                        .map(|((seed, day), &price)| {
                            seed.into_iter()
                                .chain(day.into_iter().map(|value| {
                                    scale_value(value, if price { scale } else { 1.0 })
                                }))
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    let computed = expr(&series);
                    if computed.len() != specs.len()
                        || computed
                            .iter()
                            .any(|values| values.len() != skip + indices.len())
                    {
                        return Err(err(format!("futures minute provider returned invalid dimensions for {code} on {date}")));
                    }
                    computed
                        .into_iter()
                        .map(|values| {
                            indices
                                .iter()
                                .copied()
                                .zip(values.into_iter().skip(skip))
                                .map(|(idx, value)| FactorValue {
                                    key: FactorRowKey::Minute {
                                        trade_date: date,
                                        trade_time: time(idx).to_string(),
                                        ts_code: code.to_string(),
                                    },
                                    value,
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                } else {
                    Vec::new()
                };
                Ok((code.to_string(), tail, rows))
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
                for (series, values) in output.iter_mut().zip(rows) {
                    series.values.extend(values);
                }
            }
        }
        history.push_back((date, next));
        previous_day = day_index;
        if history.len() > lookback {
            history.pop_front();
        }
    }
    Ok(output)
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

pub(crate) fn boundary_ratio(close: Option<f64>, open: Option<f64>) -> Option<f64> {
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

pub(crate) fn intraday_boundary(
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
