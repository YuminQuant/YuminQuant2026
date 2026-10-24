use crate::artifact_io::StoragePathExt;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use parquet::basic::Compression;

use crate::data::parquet_io::{read_parquet, write_parquet_with_compression};
use crate::data::{ColumnData, Table};
use crate::derive::bar::{selected_bar_columns, DerivedBarRow, BAR_KEY_COLUMNS};
use crate::error::{err, Result};

pub fn derived_stock_bar_path(data_root: &Path, bar_size: usize, trade_date: i32) -> PathBuf {
    data_root
        .join("derived")
        .join("stock")
        .join("bar")
        .join(format!("{bar_size}m"))
        .join((trade_date / 10_000).to_string())
        .join(format!("{trade_date}.parquet"))
}

pub fn write_bar_rows(path: &Path, rows: &[DerivedBarRow]) -> Result<()> {
    write_bar_rows_selected(path, rows, &selected_bar_columns(None)?)
}

pub fn write_bar_rows_selected(
    path: &Path,
    rows: &[DerivedBarRow],
    columns: &[String],
) -> Result<()> {
    selected_bar_columns(Some(columns))?;
    let table = bar_rows_table(rows, columns)?;
    write_derived_parquet(path, &table)
}

/// Shared production default for derived datasets, including bars and consensus.
pub fn write_derived_parquet(path: &Path, table: &Table) -> Result<()> {
    write_parquet_with_compression(path, table, Compression::SNAPPY)
}

/// Replace requested columns for one date, retaining other columns and stock rows.
pub(super) fn write_consensus_columns(path: &Path, updates: Table, date: i32) -> Result<()> {
    let _guard = crate::artifact_io::lock(path)?;
    let table = if path.storage_exists() {
        merge_consensus_columns(read_parquet(path, None)?, updates, date)?
    } else {
        consensus_row_keys(&updates, date)?;
        updates
    };
    write_derived_parquet(path, &table)
}

fn consensus_row_keys(table: &Table, date: i32) -> Result<Vec<String>> {
    let dates = table.required_i32_date_cast("trade_date")?;
    let codes = table.required_utf8("ts_code")?;
    let mut seen = std::collections::HashSet::new();
    let mut keys = Vec::with_capacity(table.len);
    for (day, code) in dates.iter().zip(codes) {
        let code = code
            .as_deref()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| err("null/empty consensus ts_code"))?;
        if *day != Some(date) || !seen.insert(code) {
            return Err(err("consensus column update requires unique (trade_date, ts_code) keys for the target date"));
        }
        keys.push(code.to_owned());
    }
    Ok(keys)
}

fn merge_consensus_columns(mut old: Table, updates: Table, date: i32) -> Result<Table> {
    if old.len > 0 && updates.len == 0 {
        return Err(err(
            "refusing to replace consensus columns with an empty daily input universe",
        ));
    }
    let mut keys = consensus_row_keys(&old, date)?;
    let new_keys = consensus_row_keys(&updates, date)?;
    let old_count = keys.len();
    let mut known: HashMap<String, usize> = keys
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, code)| (code, i))
        .collect();
    for key in &new_keys {
        if !known.contains_key(key) {
            known.insert(key.clone(), keys.len());
            keys.push(key.clone());
        }
    }
    // Extend each old column only when the update introduces new stocks.
    if keys.len() != old_count {
        let indices = (0..keys.len())
            .map(|i| (i < old_count).then_some(i))
            .collect::<Vec<_>>();
        for column in old.columns.values_mut() {
            *column = align_column(column, &indices);
        }
    }
    let mut indices = vec![None; keys.len()];
    for (i, code) in new_keys.iter().enumerate() {
        indices[known[code]] = Some(i);
    }
    for (name, column) in updates.columns {
        if name != "trade_date" && name != "ts_code" {
            // An absent update row or explicit null clears the selected value.
            old.columns.insert(name, align_column(&column, &indices));
        }
    }
    old.columns.insert(
        "trade_date".into(),
        ColumnData::I32(vec![Some(date); keys.len()]),
    );
    old.columns.insert(
        "ts_code".into(),
        ColumnData::Utf8(keys.into_iter().map(Some).collect()),
    );
    Table::new(old.columns)
}

fn align_column(column: &ColumnData, indices: &[Option<usize>]) -> ColumnData {
    macro_rules! align {
        ($variant:ident, $values:expr) => {
            ColumnData::$variant(
                indices
                    .iter()
                    .map(|i| i.and_then(|i| $values[i].clone()))
                    .collect(),
            )
        };
    }
    match column {
        ColumnData::Utf8(v) => align!(Utf8, v),
        ColumnData::I32(v) => align!(I32, v),
        ColumnData::I64(v) => align!(I64, v),
        ColumnData::F32(v) => align!(F32, v),
        ColumnData::F64(v) => align!(F64, v),
        ColumnData::Bool(v) => align!(Bool, v),
    }
}

fn bar_rows_table(rows: &[DerivedBarRow], columns: &[String]) -> Result<Table> {
    let mut output = BTreeMap::new();
    for name in BAR_KEY_COLUMNS
        .iter()
        .copied()
        .chain(columns.iter().map(String::as_str))
    {
        let values = match name {
            "trade_date" => ColumnData::I32(rows.iter().map(|r| Some(r.trade_date)).collect()),
            "trade_time" => {
                ColumnData::Utf8(rows.iter().map(|r| Some(r.trade_time.clone())).collect())
            }
            "bar_index" => ColumnData::I32(rows.iter().map(|r| Some(r.bar_index)).collect()),
            "ts_code" => ColumnData::Utf8(rows.iter().map(|r| Some(r.ts_code.clone())).collect()),
            "open" => ColumnData::F32(rows.iter().map(|r| Some(r.open as f32)).collect()),
            "high" => ColumnData::F32(rows.iter().map(|r| Some(r.high as f32)).collect()),
            "low" => ColumnData::F32(rows.iter().map(|r| Some(r.low as f32)).collect()),
            "close" => ColumnData::F32(rows.iter().map(|r| Some(r.close as f32)).collect()),
            "volume" => ColumnData::F64(rows.iter().map(|r| Some(r.volume)).collect()),
            "amount" => ColumnData::F64(rows.iter().map(|r| Some(r.amount)).collect()),
            "vwap" => ColumnData::F64(rows.iter().map(|r| r.vwap).collect()),
            "minute_count" => ColumnData::I32(rows.iter().map(|r| Some(r.minute_count)).collect()),
            _ => return Err(err(format!("unknown derive-bar column: {name}"))),
        };
        output.insert(name.to_string(), values);
    }
    Table::new(output)
}
