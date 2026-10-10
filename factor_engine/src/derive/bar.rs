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
        let vwap = (self.volume > 0.0).then_some(self.amount / self.volume);
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

#[cfg(test)]
mod tests {
    use super::{bar_source_columns, derive_stock_minute_bars_selected, selected_bar_columns};
    use std::collections::BTreeMap;

    use crate::data::{ColumnData, Table};
    use crate::derive::bar::{
        bar_end_label, bars_per_stock_session, derive_stock_minute_bars,
        validate_stock_minute_bar_size,
    };

    #[test]
    fn selected_volume_needs_no_prices_or_amount() {
        let selected = selected_bar_columns(Some(&["volume".into()])).unwrap();
        assert_eq!(
            bar_source_columns(&selected),
            vec!["vol", "ts_code", "trade_time"]
        );
        let table = Table::new(BTreeMap::from([
            (
                "ts_code".into(),
                ColumnData::Utf8(vec![Some("000001.SZ".into()); 2]),
            ),
            (
                "trade_time".into(),
                ColumnData::Utf8(vec![Some("09:31:00".into()), Some("09:32:00".into())]),
            ),
            ("vol".into(), ColumnData::F64(vec![Some(2.0), Some(3.0)])),
        ]))
        .unwrap();
        let rows = derive_stock_minute_bars_selected(&table, 20260105, 5, &selected).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].volume, 5.0);
        assert_eq!(rows[0].minute_count, 2);
        assert!(derive_stock_minute_bars(&table, 20260105, 5).is_err());
        assert!(selected_bar_columns(Some(&["bad".into()])).is_err());
        assert!(selected_bar_columns(Some(&[])).is_err());
        assert_eq!(
            bar_source_columns(&["vwap".into()]),
            vec!["vol", "amount", "ts_code", "trade_time"]
        );
    }

    #[test]
    fn validates_stock_minute_bar_size() {
        assert_eq!(bars_per_stock_session(15).unwrap(), 16);
        assert_eq!(bars_per_stock_session(80).unwrap(), 3);
        assert_eq!(bars_per_stock_session(120).unwrap(), 2);
        for invalid in [1, 7, 121] {
            assert!(validate_stock_minute_bar_size(invalid).is_err());
        }
    }

    #[test]
    fn labels_right_end_of_continuous_a_share_session() {
        assert_eq!(bar_end_label(0, 15), "09:45:00");
        assert_eq!(bar_end_label(7, 15), "11:30:00");
        assert_eq!(bar_end_label(15, 15), "15:00:00");
        assert_eq!(bar_end_label(2, 80), "15:00:00");
        assert_eq!(bar_end_label(0, 120), "11:30:00");
        assert_eq!(bar_end_label(1, 120), "15:00:00");
    }

    #[test]
    fn aggregates_ohlcv_and_skips_0930() {
        let table = Table::new(BTreeMap::from([
            (
                "ts_code".to_string(),
                ColumnData::Utf8(vec![
                    Some("000001.SZ".to_string()),
                    Some("000001.SZ".to_string()),
                    Some("000001.SZ".to_string()),
                    Some("000001.SZ".to_string()),
                ]),
            ),
            (
                "trade_time".to_string(),
                ColumnData::Utf8(vec![
                    Some("09:30:00".to_string()),
                    Some("09:31:00".to_string()),
                    Some("09:32:00".to_string()),
                    Some("09:33:00".to_string()),
                ]),
            ),
            (
                "open".to_string(),
                ColumnData::F64(vec![Some(99.0), Some(10.0), Some(11.0), Some(12.0)]),
            ),
            (
                "high".to_string(),
                ColumnData::F64(vec![Some(99.0), Some(10.5), Some(12.5), Some(12.2)]),
            ),
            (
                "low".to_string(),
                ColumnData::F64(vec![Some(99.0), Some(9.5), Some(10.8), Some(11.7)]),
            ),
            (
                "close".to_string(),
                ColumnData::F64(vec![Some(99.0), Some(10.2), Some(12.0), Some(12.1)]),
            ),
            (
                "vol".to_string(),
                ColumnData::F64(vec![Some(100.0), Some(1.0), Some(2.0), Some(3.0)]),
            ),
            (
                "amount".to_string(),
                ColumnData::F64(vec![Some(1000.0), Some(10.0), Some(40.0), Some(90.0)]),
            ),
        ]))
        .unwrap();

        let rows = derive_stock_minute_bars(&table, 20260424, 3).unwrap();
        let selected = vec!["volume".into(), "vwap".into()];
        let mut projected = table.clone();
        let source = bar_source_columns(&selected);
        projected.columns.retain(|name, _| source.contains(name));
        let sparse = derive_stock_minute_bars_selected(&projected, 20260424, 3, &selected).unwrap();
        assert_eq!(sparse[0].volume, rows[0].volume);
        assert_eq!(sparse[0].vwap, rows[0].vwap);
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.trade_date, 20260424);
        assert_eq!(row.trade_time, "09:33:00");
        assert_eq!(row.bar_index, 0);
        assert_eq!(row.open, 10.0);
        assert_eq!(row.high, 12.5);
        assert_eq!(row.low, 9.5);
        assert_eq!(row.close, 12.1);
        assert_eq!(row.volume, 6.0);
        assert_eq!(row.amount, 140.0);
        assert_eq!(row.vwap, Some(140.0 / 6.0));
        assert_eq!(row.minute_count, 3);
    }
}
