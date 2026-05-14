use std::collections::BTreeMap;
use std::path::Path;

use crate::data::parquet_io::read_parquet;
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
        let open = table.required_f64_cast("open")?;
        let high = table.required_f64_cast("high")?;
        let low = table.required_f64_cast("low")?;
        let close = table.required_f64_cast("close")?;
        let volume = table.required_f64_cast("vol")?;
        let amount = table.required_f64_cast("amount")?;
        let limits = load_trade_filter(data_root, *trade_date).unwrap_or_default();
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
    let mut frames = Vec::new();
    for trade_date in dates {
        let path = minute_path(data_root, *trade_date);
        if !path.exists() {
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
        let table = read_parquet(&path, Some(&columns))?;
        let codes = table.required_utf8("ts_code")?;
        let times = table.required_utf8("trade_time")?;
        let open = table.required_f64_cast("open")?;
        let high = table.required_f64_cast("high")?;
        let low = table.required_f64_cast("low")?;
        let close = table.required_f64_cast("close")?;
        let volume = table.required_f64_cast("vol")?;
        let amount = table.required_f64_cast("amount")?;
        let limits = load_trade_filter(data_root, *trade_date).unwrap_or_default();
        let mut grouped = BTreeMap::<String, Vec<Bar>>::new();
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
            let (is_limit_up, is_limit_down) = limits.get(&symbol).copied().unwrap_or_default();
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

pub fn load_future_daily_frames(
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
            read_future_daily_table(&annual)?.filter_i32_range(
                "trade_date",
                *trade_date,
                *trade_date,
            )?
        };
        let trade_dates = table.required_i32_date_cast("trade_date")?;
        let codes = table.required_utf8("ts_code")?;
        let open = table.required_f64_cast("open")?;
        let high = table.required_f64_cast("high")?;
        let low = table.required_f64_cast("low")?;
        let close = table.required_f64_cast("close")?;
        let settle = table.required_f64_cast("settle")?;
        let volume = table.required_f64_cast("vol")?;
        let amount = table.required_f64_cast("amount")?;
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
            let multiplier = meta
                .get(&symbol)
                .map(|item| item.multiplier)
                .filter(|value| value.is_finite() && *value > 0.0)
                .unwrap_or(1.0);
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
        let table = read_parquet(&path, Some(&columns))?;
        let trade_dates = table.required_i32_date_cast("trade_date")?;
        let codes = table.required_utf8("ts_code")?;
        let times = table.required_utf8("trade_time")?;
        let open = table.required_f64_cast("open")?;
        let high = table.required_f64_cast("high")?;
        let low = table.required_f64_cast("low")?;
        let close = table.required_f64_cast("close")?;
        let volume = table.required_f64_cast("vol")?;
        let amount = table.required_f64_cast("amount")?;
        let mut grouped = BTreeMap::<String, Vec<Bar>>::new();
        for idx in 0..table.len {
            if trade_dates[idx] != Some(*trade_date) {
                continue;
            }
            let Some(symbol) = codes[idx].clone() else {
                continue;
            };
            if !future_symbol_allowed(&symbol, products, symbols) {
                continue;
            }
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
            let multiplier = meta
                .get(&symbol)
                .map(|item| item.multiplier)
                .filter(|value| value.is_finite() && *value > 0.0)
                .unwrap_or(1.0);
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
    let multiplier = table.required_f64_cast("multiplier")?;
    let list_date = table.required_i32_date_cast("list_date")?;
    let delist_date = table.required_i32_date_cast("delist_date")?;
    let mut out = BTreeMap::new();
    for idx in 0..table.len {
        let Some(ts_code) = codes[idx].clone() else {
            continue;
        };
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
        .join("stock_data")
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
        .join("stock_data")
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
        .join("stock_data")
        .join("daily")
        .join("trade_filter")
        .join((trade_date / 10_000).to_string())
        .join(format!("{trade_date}.parquet"));
    if !path.exists() {
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

#[cfg(test)]
mod tests {
    use super::future_symbol_allowed;

    #[test]
    fn future_symbol_filter_uses_product_or_explicit_symbol() {
        assert!(future_symbol_allowed("AG2606.SHF", &[], &[]));
        assert!(future_symbol_allowed(
            "AG2606.SHF",
            &["AG".to_string()],
            &[]
        ));
        assert!(!future_symbol_allowed(
            "IF2606.CFX",
            &["AG".to_string()],
            &[]
        ));
        assert!(future_symbol_allowed(
            "IF2606.CFX",
            &["AG".to_string()],
            &["IF2606.CFX".to_string()]
        ));
    }
}
