use crate::artifact_io::StoragePathExt;
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
    /// True when a minute open reaches a known upper limit, independently of volume.
    pub open_at_upper_limit: bool,
    /// -1 or +1 when this open reaches the acceleration threshold on a 10%/20% day.
    pub near_price_limit: i8,
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
    pub future_day: Option<std::sync::Arc<super::future_market::FutureMarketDay>>,
}

impl MarketSnapshot {
    pub fn new(event: BarEvent, bars: Vec<Bar>) -> Self {
        Self {
            event,
            future_day: None,
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
    pub tick_size: Option<f64>,
    pub exchange: String,
    pub daily_close_minute: Option<u16>,
    pub list_date: Option<i32>,
    pub delist_date: Option<i32>,
}

pub(crate) fn clock_minute(time: &str) -> Option<u16> {
    let time = time.rsplit_once([' ', 'T']).map_or(time, |(_, time)| time);
    let mut parts = time.split(':');
    let hour = parts.next()?.parse::<u16>().ok()?;
    let minute = parts.next()?.parse::<u16>().ok()?;
    (hour < 24 && minute < 60).then_some(hour * 60 + minute)
}

fn daily_close_minute(description: &str) -> Option<u16> {
    let mut hours = description;
    let mut values = Vec::new();
    while let Some(start) = hours.find("下午") {
        hours = &hours[start + "下午".len()..];
        let range = hours.split(|ch: char| !ch.is_ascii()).next().unwrap_or("");
        values.extend(range.split(|ch: char| !(ch.is_ascii_digit() || ch == ':'))
            .filter_map(clock_minute)
            .filter(|minute| (60..=5 * 60).contains(minute))
            .map(|minute| minute + 12 * 60));
    }
    values.into_iter().chain(
        description.split(|ch: char| !(ch.is_ascii_digit() || ch == ':')).filter_map(clock_minute)
    )
        .filter(|minute| (13 * 60..=17 * 60).contains(minute))
        .max()
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
        if !path.storage_exists() {
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
        let (table, limits) = super::session_io::join(
            || read_parquet(&path, Some(&columns)),
            || load_trade_filter_or_warn(data_root, *trade_date),
        )?;
        let table = table?;
        let trade_dates = table.required_i32_date_cast("trade_date")?;
        let codes = table.required_utf8("ts_code")?;
        let open = numeric_column(&table, "open")?;
        let high = numeric_column(&table, "high")?;
        let low = numeric_column(&table, "low")?;
        let close = numeric_column(&table, "close")?;
        let volume = numeric_column(&table, "vol")?;
        let amount = numeric_column(&table, "amount")?;
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
                open_at_upper_limit: false,
                near_price_limit: 0,
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
        if !path.storage_exists() {
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
        let (table, (price_bands, limits)) = super::session_io::join(|| if let Some(symbols) = symbols {
            let selected = symbols.clone();
            read_parquet_symbols(&path, &columns, move |s| selected.contains(s))
        } else {
            read_parquet(&path, Some(&columns))
        }, || if minute_limits {
            (load_price_bands(data_root, *trade_date), BTreeMap::new())
        } else {
            (Ok(BTreeMap::new()), load_trade_filter_or_warn(data_root, *trade_date))
        })?;
        frames.extend(stock_minute_frames(
            *trade_date,
            table?,
            minute_limits,
            price_bands?,
            limits,
        )?);
    }
    Ok(frames)
}

fn stock_minute_frames(
    date: i32,
    table: Table,
    minute_limits: bool,
    price_bands: BTreeMap<String, PriceBand>,
    limits: BTreeMap<String, (bool, bool)>,
) -> Result<Vec<MarketFrame>> {
    let mut frames = Vec::new();
    let trade_date = &date;
    let codes = table.required_utf8("ts_code")?;
    let times = table.required_utf8("trade_time")?;
    let open = numeric_column(&table, "open")?;
    let high = numeric_column(&table, "high")?;
    let low = numeric_column(&table, "low")?;
    let close = numeric_column(&table, "close")?;
    let volume = numeric_column(&table, "vol")?;
    let amount = numeric_column(&table, "amount")?;

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
        let band = price_bands.get(&symbol);
        let open_at_upper_limit = band.and_then(|band| band.up).is_some_and(|p| open >= p - 0.0001);
        let (mut is_limit_up, mut is_limit_down) = if minute_limits {
            match band {
                Some(band) => (
                    open_at_upper_limit,
                    band.down.is_some_and(|p| open <= p + 0.0001),
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
            open_at_upper_limit,
            near_price_limit: band.map_or(0, |band| band.direction(open)),
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
    Ok(frames)
}

/// Prefetch an immutable universe superset, consume sessions in date order.
/// Only input tables run concurrently; no account or strategy state enters workers.
pub(crate) fn visit_stock_minute_sessions(
    data_root: &Path,
    dates: &[i32],
    symbols: Option<&BTreeSet<String>>,
    minute_limits: bool,
    mut consume: impl FnMut(i32, Vec<MarketFrame>) -> Result<()>,
) -> Result<()> {
    let mut paths = Vec::new();
    for &date in dates {
        paths.push(minute_path(data_root, date));
        paths.push(
            data_root
                .join("chn_stock_data/daily/limit")
                .join((date / 10000).to_string())
                .join(format!("{date}.parquet")),
        );
        paths.push(daily_pv_path(data_root, date));
    }
    let minute_columns = [
        "ts_code",
        "trade_time",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
    ]
    .map(str::to_string);
    let band_columns = ["ts_code", "up_limit", "down_limit"].map(str::to_string);
    let previous_close_columns = ["ts_code", "pre_close"].map(str::to_string);
    let mut pending_minute = None;
    let mut pending_bands = None;
    crate::data::read_ahead::visit_tables(
        &paths,
        |i, reader| {
            let columns = match i % 3 {
                0 => &minute_columns[..],
                1 => &band_columns[..],
                _ => &previous_close_columns[..],
            };
            let prepared = crate::data::parquet_io::PreparedProjection::new(
                reader,
                &paths[i],
                columns,
                &BTreeSet::new(),
            )?;
            Ok(if let Some(symbols) = symbols {
                prepared.with_symbols(symbols.clone())
            } else {
                prepared
            })
        },
        |_, table| Ok(table),
        |i, table| {
            match i % 3 {
                0 => {
                    pending_minute = Some(table);
                    return Ok(());
                }
                1 => {
                    pending_bands = Some(table);
                    return Ok(());
                }
                _ => {}
            }
            let bands = price_bands_from_tables(&pending_bands.take().expect("preceding band table"), &table)?;
            let date = dates[i / 3];
            let frames = stock_minute_frames(
                date,
                pending_minute.take().expect("preceding minute table"),
                minute_limits,
                bands,
                BTreeMap::new(),
            )?;
            consume(date, frames)
        },
    )
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
            let table = if path.storage_exists() {
                read_future_daily_table(&path)?
            } else {
                let annual = future_daily_annual_path(data_root, *trade_date);
                if !annual.storage_exists() {
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
                    open_at_upper_limit: false,
                    near_price_limit: 0,
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
    load_selected_future_minute_frames(data_root, dates, meta, products, symbols, None)
}

pub(crate) fn load_selected_future_minute_frames(
    data_root: &Path,
    dates: &[i32],
    meta: &BTreeMap<String, FutureContractMeta>,
    products: &[String],
    symbols: &[String],
    universe: Option<&BTreeSet<String>>,
) -> Result<Vec<MarketFrame>> {
    let mut frames = Vec::new();
    for trade_date in dates {
        let path = future_minute_path(data_root, *trade_date);
        if !path.storage_exists() {
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
        let table = if products.is_empty() && symbols.is_empty() && universe.is_none() {
            read_parquet(&path, Some(&columns))?
        } else {
            let products = products.to_vec();
            let symbols = symbols.to_vec();
            let universe = universe.cloned();
            read_parquet_symbols(&path, &columns, move |symbol| {
                universe.as_ref().map_or_else(
                    || future_symbol_allowed(symbol, &products, &symbols),
                    |selected| selected.contains(symbol),
                )
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
            let bars = grouped.entry(trade_time.clone()).or_default();
            if bars.iter().any(|bar| bar.symbol == symbol) {
                return Err(crate::error::err(format!("duplicate futures minute {symbol} {trade_time}")));
            }
            bars.push(Bar {
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
                open_at_upper_limit: false,
                near_price_limit: 0,
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
    if !path.storage_exists() {
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
        "per_unit".to_string(),
        "exchange".to_string(),
        "trade_time_desc".to_string(),
        "trade_unit".to_string(),
        "quote_unit".to_string(),
        "quote_unit_desc".to_string(),
        "list_date".to_string(),
        "delist_date".to_string(),
    ];
    let table = read_parquet(&path, Some(&columns))?;
    let codes = table.required_utf8("ts_code")?;
    let fut_codes = table.required_utf8("fut_code")?;
    let multiplier = numeric_column(&table, "multiplier")?;
    let per_unit = table.columns.contains_key("per_unit")
        .then(|| numeric_column(&table, "per_unit")).transpose()?;
    let exchanges = table.columns.contains_key("exchange")
        .then(|| table.required_utf8("exchange")).transpose()?;
    let hours = table.columns.contains_key("trade_time_desc")
        .then(|| table.required_utf8("trade_time_desc")).transpose()?;
    let trade_units = table.columns.contains_key("trade_unit")
        .then(|| table.required_utf8("trade_unit")).transpose()?;
    let quote_units = table.columns.contains_key("quote_unit")
        .then(|| table.required_utf8("quote_unit")).transpose()?;
    let tick_descriptions = table.columns.contains_key("quote_unit_desc")
        .then(|| table.required_utf8("quote_unit_desc")).transpose()?;
    let list_date = table.required_i32_date_cast("list_date")?;
    let delist_date = table.required_i32_date_cast("delist_date")?;
    let mut out = BTreeMap::new();
    for idx in 0..table.len {
        let Some(ts_code) = codes[idx].clone() else {
            continue;
        };
        let multiplier = clean_positive_opt(multiplier[idx])
            .or_else(|| {
                let unit = trade_units?.get(idx)?.as_deref()?;
                let quote = quote_units?.get(idx)?.as_deref()?;
                let quoted_unit = quote.strip_prefix("元/").or_else(|| quote.strip_prefix("人民币元/"))?;
                let per_lot = clean_positive_opt(per_unit.as_ref().and_then(|values| values[idx]))?;
                match (unit, quoted_unit) {
                    (unit, quoted) if unit == quoted => Some(per_lot),
                    ("吨", "500千克") => Some(per_lot * 2.0),
                    _ => None,
                }
            })
            .unwrap_or(0.0);
        out.insert(
            ts_code.clone(),
            FutureContractMeta {
                ts_code,
                fut_code: fut_codes[idx].clone(),
                multiplier,
                tick_size: tick_descriptions
                    .and_then(|values| values[idx].as_deref())
                    .and_then(parse_tick_size),
                exchange: exchanges.and_then(|values| values[idx].clone()).unwrap_or_default(),
                daily_close_minute: hours.and_then(|values| values[idx].as_deref()).and_then(daily_close_minute),
                list_date: list_date[idx],
                delist_date: delist_date[idx],
            },
        );
    }
    Ok(out)
}

fn parse_tick_size(description: &str) -> Option<f64> {
    let number = description.trim_start().split(|ch: char| !ch.is_ascii_digit() && ch != '.').next()?;
    clean_positive(number.parse().ok()?)
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
    if !path.storage_exists() {
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
) -> Result<BTreeMap<String, PriceBand>> {
    let path = root
        .join("chn_stock_data/daily/limit")
        .join((date / 10000).to_string())
        .join(format!("{date}.parquet"));
    let cols = ["ts_code", "up_limit", "down_limit"].map(str::to_string);
    let (table, previous) = super::session_io::join(
        || read_parquet(&path, Some(&cols)),
        || read_parquet(&daily_pv_path(root, date), Some(&["ts_code".into(), "pre_close".into()])),
    )?;
    price_bands_from_tables(&table?, &previous?)
}

#[derive(Default)]
struct PriceBand {
    up: Option<f64>,
    down: Option<f64>,
    acceleration: Option<(f64, f64)>,
}

impl PriceBand {
    fn direction(&self, price: f64) -> i8 {
        match self.acceleration {
            Some((lower, _)) if price <= lower => -1,
            Some((_, upper)) if price >= upper => 1,
            _ => 0,
        }
    }
}

fn acceleration_prices(previous: f64, up: f64, down: f64) -> Option<(f64, f64)> {
    // Compare the actual dated bands to cent-rounded exchange limits.
    let matches = |rate: f64| {
        (up - previous * (1.0 + rate)).abs() <= 0.0051
            && (down - previous * (1.0 - rate)).abs() <= 0.0051
    };
    let trigger = match (matches(0.10), matches(0.20)) {
        (true, false) => 0.09,
        (false, true) => 0.18,
        _ => return None,
    };
    let tolerance = previous * 1e-7;
    Some((previous * (1.0 - trigger) + tolerance, previous * (1.0 + trigger) - tolerance))
}

fn price_bands_from_tables(table: &Table, previous: &Table) -> Result<BTreeMap<String, PriceBand>> {
    let previous_codes = previous.required_utf8("ts_code")?;
    let previous_prices = previous.required_f64_cast("pre_close")?;
    let previous: BTreeMap<_, _> = previous_codes.iter().zip(previous_prices)
        .filter_map(|(code, price)| Some((code.as_deref()?, clean_positive_opt(price)?)))
        .collect();
    let codes = table.required_utf8("ts_code")?;
    let up = table.required_f64_cast("up_limit")?;
    let down = table.required_f64_cast("down_limit")?;
    Ok((0..table.len)
        .filter_map(|i| {
            let symbol = codes[i].clone()?;
            let up = clean_positive_opt(up[i]);
            let down = clean_positive_opt(down[i]);
            let acceleration = previous.get(symbol.as_str()).zip(up.zip(down))
                .and_then(|(&price, (up, down))| acceleration_prices(price, up, down));
            Some((symbol, PriceBand { up, down, acceleration }))
        })
        .collect())
}
