use std::collections::BTreeMap;

use crate::data::Table;
use crate::error::{err, Result};

pub const ALLOWED_STOCK_MINUTE_BAR_SIZES: &[usize] = &[
    2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 30, 40, 48, 60, 80, 120,
];

pub const BAR_VALUE_COLUMNS: &[&str] = &[
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "vwap",
    "minute_count",
];
pub const BAR_KEY_COLUMNS: &[&str] = &["trade_date", "trade_time", "bar_index", "ts_code"];

pub fn selected_bar_columns(columns: Option<&[String]>) -> Result<Vec<String>> {
    let columns = columns
        .map(|v| v.to_vec())
        .unwrap_or_else(|| BAR_VALUE_COLUMNS.iter().map(|s| s.to_string()).collect());
    if columns.is_empty() {
        return Err(err("derive-bar --columns must not be empty"));
    }
    let mut selected = Vec::new();
    for column in columns {
        if !BAR_VALUE_COLUMNS.contains(&column.as_str()) {
            return Err(err(format!("unknown derive-bar column: {column}")));
        }
        if !selected.contains(&column) {
            selected.push(column);
        }
    }
    Ok(selected)
}

pub fn bar_source_columns(columns: &[String]) -> Vec<String> {
    let mut source = Vec::new();
    for column in columns {
        let dependencies: Vec<&str> = match column.as_str() {
            "volume" => vec!["vol"],
            "vwap" => vec!["vol", "amount"],
            "minute_count" => vec![],
            name => vec![name],
        };
        for name in dependencies {
            if !source.iter().any(|s| s == name) {
                source.push(name.to_string());
            }
        }
    }
    // Explicit keys also make a minute_count-only projection nonempty.
    source.extend(["ts_code".to_string(), "trade_time".to_string()]);
    source
}

#[derive(Clone, Debug, PartialEq)]
pub struct DerivedBarRow {
    pub trade_date: i32,
    pub trade_time: String,
    pub bar_index: i32,
    pub ts_code: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub amount: f64,
    pub vwap: Option<f64>,
    pub minute_count: i32,
}

#[derive(Clone, Debug)]
struct MinuteRow {
    ts_code: String,
    minute_index: usize,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    amount: f64,
}

#[derive(Clone, Debug)]
struct BarBuilder {
    trade_date: i32,
    ts_code: String,
    bar_index: usize,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    amount: f64,
    minute_count: i32,
}

impl BarBuilder {
    fn new(trade_date: i32, ts_code: String, bar_index: usize, row: &MinuteRow) -> Self {
        Self {
            trade_date,
            ts_code,
            bar_index,
            open: row.open,
            high: row.high,
            low: row.low,
            close: row.close,
            volume: row.volume,
            amount: row.amount,
            minute_count: 1,
        }
    }

    fn push(&mut self, row: &MinuteRow) {
        self.high = self.high.max(row.high);
        self.low = self.low.min(row.low);
        self.close = row.close;
        self.volume += row.volume;
        self.amount += row.amount;
        self.minute_count += 1;
    }

    fn finish(self, bar_size: usize) -> DerivedBarRow {
        let vwap = if self.volume > 0.0 {
            Some(self.amount / self.volume)
        } else if self.volume == 0.0 {
            Some(0.0)
        } else {
            None
        };
        DerivedBarRow {
            trade_date: self.trade_date,
            trade_time: bar_end_label(self.bar_index, bar_size),
            bar_index: self.bar_index as i32,
            ts_code: self.ts_code,
            open: self.open,
            high: self.high,
            low: self.low,
            close: self.close,
            volume: self.volume,
            amount: self.amount,
            vwap,
            minute_count: self.minute_count,
        }
    }
}

pub fn validate_stock_minute_bar_size(bar_size: usize) -> Result<()> {
    if ALLOWED_STOCK_MINUTE_BAR_SIZES.contains(&bar_size) {
        Ok(())
    } else {
        Err(err(format!(
            "stock minute bar_size must be a divisor of 240 and satisfy 1 < bar_size <= 120; allowed values: {}",
            ALLOWED_STOCK_MINUTE_BAR_SIZES
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
                .join(",")
        )))
    }
}

pub fn bars_per_stock_session(bar_size: usize) -> Result<usize> {
    validate_stock_minute_bar_size(bar_size)?;
    Ok(240 / bar_size)
}

pub fn derive_stock_minute_bars(
    table: &Table,
    trade_date: i32,
    bar_size: usize,
) -> Result<Vec<DerivedBarRow>> {
    derive_stock_minute_bars_selected(table, trade_date, bar_size, &selected_bar_columns(None)?)
}

pub fn derive_stock_minute_bars_selected(
    table: &Table,
    trade_date: i32,
    bar_size: usize,
    columns: &[String],
) -> Result<Vec<DerivedBarRow>> {
    validate_stock_minute_bar_size(bar_size)?;
    selected_bar_columns(Some(columns))?;
    let ts_codes = table.required_utf8("ts_code")?;
    let trade_times = table.required_utf8("trade_time")?;
    let source = bar_source_columns(columns);
    let read = |name: &str| -> Result<Vec<Option<f64>>> {
        if source.iter().any(|c| c == name) {
            table.required_f64_cast(name)
        } else {
            Ok(Vec::new())
        }
    };
    let value = |values: &[Option<f64>], idx| {
        if values.is_empty() {
            Some(0.0)
        } else {
            finite(values[idx])
        }
    };
    let opens = read("open")?;
    let highs = read("high")?;
    let lows = read("low")?;
    let closes = read("close")?;
    let volumes = read("vol")?;
    let amounts = read("amount")?;

    let mut rows = Vec::with_capacity(table.len);
    for idx in 0..table.len {
        let Some(ts_code) = ts_codes[idx].as_ref() else {
            continue;
        };
        let Some(trade_time) = trade_times[idx].as_deref() else {
            continue;
        };
        let Some(minute_index) = minute_index(trade_time) else {
            continue;
        };
        let Some(open) = value(&opens, idx) else {
            continue;
        };
        let Some(high) = value(&highs, idx) else {
            continue;
        };
        let Some(low) = value(&lows, idx) else {
            continue;
        };
        let Some(close) = value(&closes, idx) else {
            continue;
        };
        let Some(volume) = value(&volumes, idx) else {
            continue;
        };
        let Some(amount) = value(&amounts, idx) else {
            continue;
        };
        rows.push(MinuteRow {
            ts_code: ts_code.clone(),
            minute_index,
            open,
            high,
            low,
            close,
            volume,
            amount,
        });
    }

    rows.sort_by(|left, right| {
        left.ts_code
            .cmp(&right.ts_code)
            .then_with(|| left.minute_index.cmp(&right.minute_index))
    });

    let mut builders: BTreeMap<(String, usize), BarBuilder> = BTreeMap::new();
    for row in rows {
        let bar_index = row.minute_index / bar_size;
        let key = (row.ts_code.clone(), bar_index);
        if let Some(builder) = builders.get_mut(&key) {
            builder.push(&row);
        } else {
            builders.insert(
                key,
                BarBuilder::new(trade_date, row.ts_code.clone(), bar_index, &row),
            );
        }
    }
    let mut output = builders
        .into_values()
        .map(|builder| builder.finish(bar_size))
        .collect::<Vec<_>>();
    output.sort_by(|left, right| {
        left.bar_index
            .cmp(&right.bar_index)
            .then_with(|| left.ts_code.cmp(&right.ts_code))
    });
    Ok(output)
}

fn finite(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite())
}

pub(super) fn minute_index(trade_time: &str) -> Option<usize> {
    let (hour, minute) = parse_hour_minute(trade_time)?;
    let minutes = hour * 60 + minute;
    let morning_start = 9 * 60 + 31;
    let morning_end = 11 * 60 + 30;
    let afternoon_start = 13 * 60 + 1;
    let afternoon_end = 15 * 60;
    if (morning_start..=morning_end).contains(&minutes) {
        return Some((minutes - morning_start) as usize);
    }
    if (afternoon_start..=afternoon_end).contains(&minutes) {
        return Some(120 + (minutes - afternoon_start) as usize);
    }
    None
}

fn parse_hour_minute(value: &str) -> Option<(i32, i32)> {
    let time = value
        .trim()
        .rsplit_once(' ')
        .map(|(_, time)| time)
        .unwrap_or_else(|| value.trim());
    let mut parts = time.split(':');
    let hour = parts.next()?.parse::<i32>().ok()?;
    let minute = parts.next()?.parse::<i32>().ok()?;
    if !(0..=23).contains(&hour) || !(0..=59).contains(&minute) {
        return None;
    }
    Some((hour, minute))
}

fn bar_end_label(bar_index: usize, bar_size: usize) -> String {
    let minute_index = ((bar_index + 1) * bar_size).saturating_sub(1).min(239);
    let total_minutes = if minute_index < 120 {
        9 * 60 + 31 + minute_index as i32
    } else {
        13 * 60 + 1 + (minute_index as i32 - 120)
    };
    format!("{:02}:{:02}:00", total_minutes / 60, total_minutes % 60)
}
