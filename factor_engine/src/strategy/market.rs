use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::data::parquet_io::{read_parquet, read_parquet_symbols};
use crate::data::{ColumnData, Table};
use crate::error::Result;
use crate::strategy::config::{future_product_from_symbol, BarFrequency};

#[derive(Clone, Debug)]
pub struct Bar {
    pub symbol: String,
    pub trade_date: i32,
    pub trade_time: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub settle: Option<f64>,
    pub multiplier: f64,
    pub volume: Option<f64>,
    pub amount: Option<f64>,
    pub is_limit_up: bool,
    pub is_limit_down: bool,
}

#[derive(Clone, Debug)]
pub struct BarEvent {
    pub trade_date: i32,
    pub trade_time: String,
    pub bar_frequency: BarFrequency,
    pub is_session_first: bool,
    pub is_session_end: bool,
}

#[derive(Clone, Debug)]
pub struct SessionOpenEvent {
    pub trade_date: i32,
    pub trade_time: String,
    pub bar_frequency: BarFrequency,
}

#[derive(Clone, Debug)]
pub struct MarketSnapshot {
    pub event: BarEvent,
    bars: BTreeMap<String, Bar>,
}

impl MarketSnapshot {
    pub fn new(event: BarEvent, bars: Vec<Bar>) -> Self {
        Self {
            event,
            bars: bars
                .into_iter()
                .map(|bar| (bar.symbol.clone(), bar))
                .collect(),
        }
    }

    pub fn bar(&self, symbol: &str) -> Option<&Bar> {
        self.bars.get(symbol)
    }

    pub fn symbols(&self) -> impl Iterator<Item = &String> {
        self.bars.keys()
    }

    pub fn open_price(&self, symbol: &str) -> Option<f64> {
        clean_positive(self.bars.get(symbol)?.open)
    }

    pub fn close_price(&self, symbol: &str) -> Option<f64> {
        clean_positive(self.bars.get(symbol)?.close)
    }
}

#[derive(Clone, Debug)]
pub struct FutureContractMeta {
    pub ts_code: String,
    pub fut_code: Option<String>,
    pub multiplier: f64,
    pub list_date: Option<i32>,
    pub delist_date: Option<i32>,
}

#[derive(Clone, Debug)]
pub struct MarketFrame {
    pub event: BarEvent,
    pub bars: Vec<Bar>,
}

pub fn load_stock_daily_frames(data_root: &Path, dates: &[i32]) -> Result<Vec<MarketFrame>> {
    let mut frames = Vec::new();
    for trade_date in dates {
        let path = daily_pv_path(data_root, *trade_date);
        if !path.exists() {
            continue;
        }
        let columns = vec![
            "trade_date".to_string(),
            "ts_code".to_string(),
            "open".to_string(),
            "high".to_string(),
            "low".to_string(),
            "close".to_string(),
            "vol".to_string(),
            "amount".to_string(),
        ];
        let table = read_parquet(&path, Some(&columns))?;
        let trade_dates = table.required_i32_date_cast("trade_date")?;
        let codes = table.required_utf8("ts_code")?;
        let open = numeric_column(&table, "open")?;
        let high = numeric_column(&table, "high")?;
        let low = numeric_column(&table, "low")?;
        let close = numeric_column(&table, "close")?;
        let volume = numeric_column(&table, "vol")?;
        let amount = numeric_column(&table, "amount")?;
        let limits = load_trade_filter_or_warn(data_root, *trade_date);
        let mut bars = Vec::new();
        for idx in 0..table.len {
            if trade_dates[idx] != Some(*trade_date) {
                continue;
            }
            let Some(symbol) = codes[idx].clone() else {
                continue;
            };
            let Some(open) = clean_positive_opt(open[idx]) else {
                continue;
            };
            let Some(high) = clean_positive_opt(high[idx]) else {
                continue;
            };
            let Some(low) = clean_positive_opt(low[idx]) else {
                continue;
            };
            let Some(close) = clean_positive_opt(close[idx]) else {
                continue;
            };
            let (is_limit_up, is_limit_down) = limits.get(&symbol).copied().unwrap_or_default();
            bars.push(Bar {
                symbol,
                trade_date: *trade_date,
                trade_time: "daily".to_string(),
                open,
                high,
                low,
                close,
                settle: None,
                multiplier: 1.0,
                volume: finite(volume[idx]),
                amount: finite(amount[idx]),
                is_limit_up,
                is_limit_down,
            });
        }
        frames.push(MarketFrame {
            event: BarEvent {
                trade_date: *trade_date,
                trade_time: "daily".to_string(),
                bar_frequency: BarFrequency::Daily,
                is_session_first: true,
                is_session_end: true,
            },
            bars,
        });
    }
    Ok(frames)
}

pub fn load_stock_minute_frames(data_root: &Path, dates: &[i32]) -> Result<Vec<MarketFrame>> {
    load_stock_minute_frames_for_run(data_root, dates, None, false)
}

pub fn load_stock_minute_frames_for_run(
    data_root: &Path,
    dates: &[i32],
    symbols: Option<&BTreeSet<String>>,
    minute_limits: bool,
) -> Result<Vec<MarketFrame>> {
    let mut frames = Vec::new();
    for trade_date in dates {
        let path = minute_path(data_root, *trade_date);
        if !path.exists() {
            if minute_limits {
                return Err(crate::error::err(format!(
                    "missing minute file {}",
                    path.display()
                )));
            }
            continue;
        }
        let columns = vec![
            "ts_code".to_string(),
            "trade_time".to_string(),
            "open".to_string(),
            "high".to_string(),
            "low".to_string(),
            "close".to_string(),
            "vol".to_string(),
            "amount".to_string(),
        ];
        let table = if let Some(symbols) = symbols {
            let selected = symbols.clone();
            read_parquet_symbols(&path, &columns, move |s| selected.contains(s))?
        } else {
            read_parquet(&path, Some(&columns))?
        };
        let codes = table.required_utf8("ts_code")?;
        let times = table.required_utf8("trade_time")?;
        let open = numeric_column(&table, "open")?;
        let high = numeric_column(&table, "high")?;
        let low = numeric_column(&table, "low")?;
        let close = numeric_column(&table, "close")?;
        let volume = numeric_column(&table, "vol")?;
        let amount = numeric_column(&table, "amount")?;
        let limits = if minute_limits {
            BTreeMap::new()
        } else {
            load_trade_filter_or_warn(data_root, *trade_date)
        };
        let price_bands = if minute_limits {
            load_price_bands(data_root, *trade_date)?
        } else {
            BTreeMap::new()
        };
        let mut grouped = BTreeMap::<String, Vec<Bar>>::new();
        if minute_limits {
            // Keep the exchange clock even if all selected/held stocks are suspended.
            for minute in (570..=690).chain(781..=900) {
                let date = *trade_date;
                grouped.insert(
                    format!(
                        "{:04}-{:02}-{:02} {:02}:{:02}:00",
                        date / 10000,
                        date / 100 % 100,
                        date % 100,
                        minute / 60,
                        minute % 60
                    ),
                    Vec::new(),
                );
            }
        }
        for idx in 0..table.len {
            let Some(symbol) = codes[idx].clone() else {
                continue;
            };
            let Some(trade_time) = times[idx].clone() else {
                continue;
            };
            let Some(open) = clean_positive_opt(open[idx]) else {
                continue;
            };
            let Some(high) = clean_positive_opt(high[idx]) else {
                continue;
            };
            let Some(low) = clean_positive_opt(low[idx]) else {
                continue;
            };
            let Some(close) = clean_positive_opt(close[idx]) else {
                continue;
            };
            let (mut is_limit_up, mut is_limit_down) = if minute_limits {
                match price_bands.get(&symbol) {
                    Some((up, down)) => (
                        up.is_some_and(|p| open >= p - 0.0001),
                        down.is_some_and(|p| open <= p + 0.0001),
                    ),
                    // Missing price bands are unknown, not permission to trade.
                    None => (true, true),
                }
            } else {
                limits.get(&symbol).copied().unwrap_or_default()
            };
            if minute_limits && !volume[idx].is_some_and(|v| v.is_finite() && v > 0.0) {
                is_limit_up = true;
                is_limit_down = true;
            }
            grouped.entry(trade_time.clone()).or_default().push(Bar {
                symbol,
                trade_date: *trade_date,
                trade_time,
                open,
                high,
                low,
                close,
                settle: None,
                multiplier: 1.0,
                volume: finite(volume[idx]),
                amount: finite(amount[idx]),
                is_limit_up,
                is_limit_down,
            });
        }
        let total = grouped.len();
        for (idx, (trade_time, bars)) in grouped.into_iter().enumerate() {
            frames.push(MarketFrame {
                event: BarEvent {
                    trade_date: *trade_date,
                    trade_time,
                    bar_frequency: BarFrequency::Minute,
                    is_session_first: idx == 0,
                    is_session_end: idx + 1 == total,
                },
                bars,
            });
        }
    }
    Ok(frames)
}

/// At most one annual file is retained per run, indexed once by date.
#[derive(Default)]
pub struct FutureDailyCache {
    annual: Option<(std::path::PathBuf, Table, BTreeMap<i32, Vec<usize>>)>,
}

impl FutureDailyCache {
    fn day(&mut self, path: &Path, date: i32) -> Result<Table> {
        if self.annual.as_ref().map(|entry| entry.0.as_path()) != Some(path) {
            // Release the previous year before loading the next one.
            self.annual = None;
            let table = read_future_daily_table(path)?;
            let mut indices = BTreeMap::<i32, Vec<usize>>::new();
            for (index, date) in table
                .required_i32_date_cast("trade_date")?
                .into_iter()
                .enumerate()
            {
                if let Some(date) = date {
                    indices.entry(date).or_default().push(index);
                }
            }
            self.annual = Some((path.to_path_buf(), table, indices));
        }
        let (_, table, indices) = self.annual.as_ref().unwrap();
        table.take(indices.get(&date).map(Vec::as_slice).unwrap_or(&[]))
    }

    pub fn load(
        &mut self,
        data_root: &Path,
        dates: &[i32],
        meta: &BTreeMap<String, FutureContractMeta>,
    ) -> Result<Vec<MarketFrame>> {
        let mut frames = Vec::new();
        for trade_date in dates {
            let path = future_daily_date_path(data_root, *trade_date);
            let table = if path.exists() {
                read_future_daily_table(&path)?
            } else {
                let annual = future_daily_annual_path(data_root, *trade_date);
                if !annual.exists() {
                    continue;
                }
                self.day(&annual, *trade_date)?
            };
            let trade_dates = table.required_i32_date_cast("trade_date")?;
            let codes = table.required_utf8("ts_code")?;
            let open = numeric_column(&table, "open")?;
            let high = numeric_column(&table, "high")?;
            let low = numeric_column(&table, "low")?;
            let close = numeric_column(&table, "close")?;
            let settle = numeric_column(&table, "settle")?;
            let volume = numeric_column(&table, "vol")?;
            let amount = numeric_column(&table, "amount")?;
            let mut bars = Vec::new();
            let mut missing_meta = BTreeSet::new();
            for idx in 0..table.len {
                if trade_dates[idx] != Some(*trade_date) {
                    continue;
                }
                let Some(symbol) = codes[idx].clone() else {
                    continue;
                };
                let Some(open) = clean_positive_opt(open[idx]) else {
                    continue;
                };
                let Some(high) = clean_positive_opt(high[idx]) else {
                    continue;
                };
                let Some(low) = clean_positive_opt(low[idx]) else {
                    continue;
                };
                let Some(close) = clean_positive_opt(close[idx]) else {
                    continue;
                };
                let multiplier = contract_multiplier(meta, &symbol, &mut missing_meta);
                bars.push(Bar {
                    symbol,
                    trade_date: *trade_date,
                    trade_time: "daily".to_string(),
                    open,
                    high,
                    low,
                    close,
                    settle: clean_positive_opt(settle[idx]),
                    multiplier,
                    volume: finite(volume[idx]),
                    amount: finite(amount[idx]),
                    is_limit_up: false,
                    is_limit_down: false,
                });
            }
            warn_missing_multipliers(&missing_meta, *trade_date);
            frames.push(MarketFrame {
                event: BarEvent {
                    trade_date: *trade_date,
                    trade_time: "daily".to_string(),
                    bar_frequency: BarFrequency::Daily,
                    is_session_first: true,
                    is_session_end: true,
                },
                bars,
            });
        }
        Ok(frames)
    }
}

pub fn load_future_daily_frames(
    data_root: &Path,
    dates: &[i32],
    meta: &BTreeMap<String, FutureContractMeta>,
) -> Result<Vec<MarketFrame>> {
    FutureDailyCache::default().load(data_root, dates, meta)
}

pub fn load_future_minute_frames(
    data_root: &Path,
    dates: &[i32],
    meta: &BTreeMap<String, FutureContractMeta>,
    products: &[String],
    symbols: &[String],
) -> Result<Vec<MarketFrame>> {
    let mut frames = Vec::new();
    for trade_date in dates {
        let path = future_minute_path(data_root, *trade_date);
        if !path.exists() {
            continue;
        }
        let columns = vec![
            "trade_date".to_string(),
            "ts_code".to_string(),
            "trade_time".to_string(),
            "open".to_string(),
            "high".to_string(),
            "low".to_string(),
            "close".to_string(),
            "vol".to_string(),
            "amount".to_string(),
        ];
        let table = if products.is_empty() && symbols.is_empty() {
            read_parquet(&path, Some(&columns))?
        } else {
            let products = products.to_vec();
            let symbols = symbols.to_vec();
            read_parquet_symbols(&path, &columns, move |symbol| {
                future_symbol_allowed(symbol, &products, &symbols)
            })?
        };
        let trade_dates = table.required_i32_date_cast("trade_date")?;
        let codes = table.required_utf8("ts_code")?;
        let times = table.required_utf8("trade_time")?;
        let open = numeric_column(&table, "open")?;
        let high = numeric_column(&table, "high")?;
        let low = numeric_column(&table, "low")?;
        let close = numeric_column(&table, "close")?;
        let volume = numeric_column(&table, "vol")?;
        let amount = numeric_column(&table, "amount")?;
        let mut grouped = BTreeMap::<String, Vec<Bar>>::new();
        let mut missing_meta = BTreeSet::new();
        for idx in 0..table.len {
            if trade_dates[idx] != Some(*trade_date) {
                continue;
            }
            let Some(symbol) = codes[idx].clone() else {
                continue;
            };
            let Some(trade_time) = times[idx].clone() else {
                continue;
            };
            let Some(open) = clean_positive_opt(open[idx]) else {
                continue;
            };
            let Some(high) = clean_positive_opt(high[idx]) else {
                continue;
            };
            let Some(low) = clean_positive_opt(low[idx]) else {
                continue;
            };
            let Some(close) = clean_positive_opt(close[idx]) else {
                continue;
            };
            let multiplier = contract_multiplier(meta, &symbol, &mut missing_meta);
            grouped.entry(trade_time.clone()).or_default().push(Bar {
                symbol,
                trade_date: *trade_date,
                trade_time,
                open,
                high,
                low,
                close,
                settle: None,
                multiplier,
                volume: finite(volume[idx]),
                amount: finite(amount[idx]),
                is_limit_up: false,
                is_limit_down: false,
            });
        }
        warn_missing_multipliers(&missing_meta, *trade_date);
        let total = grouped.len();
        for (idx, (trade_time, bars)) in grouped.into_iter().enumerate() {
            frames.push(MarketFrame {
                event: BarEvent {
                    trade_date: *trade_date,
                    trade_time,
                    bar_frequency: BarFrequency::Minute,
                    is_session_first: idx == 0,
                    is_session_end: idx + 1 == total,
                },
                bars,
            });
        }
    }
    Ok(frames)
}

pub fn load_future_metadata(data_root: &Path) -> Result<BTreeMap<String, FutureContractMeta>> {
    let path = data_root
        .join("future_data")
        .join("basic")
        .join("fut_basic.parquet");
    if !path.exists() {
        eprintln!(
            "[strategy warning] missing {}; using existing defaults",
            path.display()
        );
        return Ok(BTreeMap::new());
    }
    let columns = vec![
        "ts_code".to_string(),
        "fut_code".to_string(),
        "multiplier".to_string(),
        "list_date".to_string(),
        "delist_date".to_string(),
    ];
    let table = read_parquet(&path, Some(&columns))?;
    let codes = table.required_utf8("ts_code")?;
    let fut_codes = table.required_utf8("fut_code")?;
    let multiplier = numeric_column(&table, "multiplier")?;
    let list_date = table.required_i32_date_cast("list_date")?;
    let delist_date = table.required_i32_date_cast("delist_date")?;
    let mut out = BTreeMap::new();
    for idx in 0..table.len {
        let Some(ts_code) = codes[idx].clone() else {
            continue;
        };
        if !multiplier[idx].is_some_and(|value| value.is_finite() && value > 0.0) {
            eprintln!("[strategy warning] invalid multiplier for {ts_code}; using 1");
        }
        let multiplier = multiplier[idx]
            .filter(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(1.0);
        out.insert(
            ts_code.clone(),
            FutureContractMeta {
                ts_code,
                fut_code: fut_codes[idx].clone(),
                multiplier,
                list_date: list_date[idx],
                delist_date: delist_date[idx],
            },
        );
    }
    Ok(out)
}

fn read_future_daily_table(path: &Path) -> Result<crate::data::Table> {
    let columns = vec![
        "trade_date".to_string(),
        "ts_code".to_string(),
        "open".to_string(),
        "high".to_string(),
        "low".to_string(),
        "close".to_string(),
        "settle".to_string(),
        "vol".to_string(),
        "amount".to_string(),
    ];
    read_parquet(path, Some(&columns))
}

fn daily_pv_path(data_root: &Path, trade_date: i32) -> std::path::PathBuf {
    data_root
        .join("chn_stock_data")
        .join("daily")
        .join("pv")
        .join((trade_date / 10_000).to_string())
        .join(format!("{trade_date}.parquet"))
}

fn future_symbol_allowed(symbol: &str, products: &[String], symbols: &[String]) -> bool {
    if products.is_empty() && symbols.is_empty() {
        return true;
    }
    let symbol_upper = symbol.to_ascii_uppercase();
    symbols.iter().any(|item| item == &symbol_upper)
        || products
            .iter()
            .any(|product| product == &future_product_from_symbol(symbol))
}

fn minute_path(data_root: &Path, trade_date: i32) -> std::path::PathBuf {
    data_root
        .join("chn_stock_data")
        .join("minute")
        .join((trade_date / 10_000).to_string())
        .join(format!("{trade_date}.parquet"))
}

fn future_daily_date_path(data_root: &Path, trade_date: i32) -> std::path::PathBuf {
    data_root
        .join("future_data")
        .join("daily")
        .join((trade_date / 10_000).to_string())
        .join(format!("{trade_date}.parquet"))
}

fn future_daily_annual_path(data_root: &Path, trade_date: i32) -> std::path::PathBuf {
    data_root
        .join("future_data")
        .join("daily")
        .join(format!("{}.parquet", trade_date / 10_000))
}

fn future_minute_path(data_root: &Path, trade_date: i32) -> std::path::PathBuf {
    data_root
        .join("future_data")
        .join("minute")
        .join((trade_date / 10_000).to_string())
        .join(format!("{trade_date}.parquet"))
}

fn load_trade_filter(data_root: &Path, trade_date: i32) -> Result<BTreeMap<String, (bool, bool)>> {
    let path = data_root
        .join("chn_stock_data")
        .join("daily")
        .join("trade_filter")
        .join((trade_date / 10_000).to_string())
        .join(format!("{trade_date}.parquet"));
    if !path.exists() {
        eprintln!(
            "[strategy warning] missing {}; using existing defaults",
            path.display()
        );
        return Ok(BTreeMap::new());
    }
    let columns = vec![
        "trade_date".to_string(),
        "ts_code".to_string(),
        "is_limit_up".to_string(),
        "is_limit_down".to_string(),
    ];
    let table = read_parquet(&path, Some(&columns))?;
    let dates = table.required_i32_date_cast("trade_date")?;
    let codes = table.required_utf8("ts_code")?;
    let up = match table.columns.get("is_limit_up") {
        Some(crate::data::ColumnData::Bool(values)) => values.clone(),
        _ => vec![None; table.len],
    };
    let down = match table.columns.get("is_limit_down") {
        Some(crate::data::ColumnData::Bool(values)) => values.clone(),
        _ => vec![None; table.len],
    };
    let mut out = BTreeMap::new();
    for idx in 0..table.len {
        if dates[idx] == Some(trade_date) {
            if let Some(code) = codes[idx].clone() {
                out.insert(code, (up[idx].unwrap_or(false), down[idx].unwrap_or(false)));
            }
        }
    }
    Ok(out)
}

fn clean_positive(value: f64) -> Option<f64> {
    (value.is_finite() && value > 0.0).then_some(value)
}

fn clean_positive_opt(value: Option<f64>) -> Option<f64> {
    clean_positive(value?)
}

fn finite(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite())
}

fn numeric_column<'a>(table: &'a Table, name: &str) -> Result<Cow<'a, [Option<f64>]>> {
    match table.columns.get(name) {
        Some(ColumnData::F64(values)) => Ok(Cow::Borrowed(values)),
        _ => Ok(Cow::Owned(table.required_f64_cast(name)?)),
    }
}

fn load_trade_filter_or_warn(data_root: &Path, trade_date: i32) -> BTreeMap<String, (bool, bool)> {
    match load_trade_filter(data_root, trade_date) {
        Ok(limits) => limits,
        Err(error) => {
            eprintln!(
                "[strategy warning] trade filter {trade_date}: {error}; using no limit flags"
            );
            BTreeMap::new()
        }
    }
}

fn contract_multiplier(
    meta: &BTreeMap<String, FutureContractMeta>,
    symbol: &str,
    missing: &mut BTreeSet<String>,
) -> f64 {
    meta.get(symbol)
        .map(|item| item.multiplier)
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or_else(|| {
            if !missing.contains(symbol) {
                missing.insert(symbol.to_string());
            }
            1.0
        })
}

fn warn_missing_multipliers(missing: &BTreeSet<String>, date: i32) {
    if !missing.is_empty() {
        eprintln!("[strategy warning] date={date}: {} contracts with missing/invalid multiplier; using 1 (examples: {})", missing.len(), missing.iter().take(5).map(String::as_str).collect::<Vec<_>>().join(","));
    }
}

fn load_price_bands(
    root: &Path,
    date: i32,
) -> Result<BTreeMap<String, (Option<f64>, Option<f64>)>> {
    let path = root
        .join("chn_stock_data/daily/limit")
        .join((date / 10000).to_string())
        .join(format!("{date}.parquet"));
    let cols = ["ts_code", "up_limit", "down_limit"].map(str::to_string);
    let table = read_parquet(&path, Some(&cols))?;
    let codes = table.required_utf8("ts_code")?;
    let up = table.required_f64_cast("up_limit")?;
    let down = table.required_f64_cast("down_limit")?;
    Ok((0..table.len)
        .filter_map(|i| {
            codes[i]
                .clone()
                .map(|s| (s, (clean_positive_opt(up[i]), clean_positive_opt(down[i]))))
        })
        .collect())
}
