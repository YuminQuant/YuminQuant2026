use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use parquet::basic::Compression;

use crate::data::parquet_io::{read_parquet, write_parquet_with_compression};
use crate::data::{ColumnData, Table};
use crate::derive::bar::DerivedBarRow;
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
    write_derived_parquet(path, &bar_rows_table(rows)?)
}

/// Shared production default for derived datasets, including bars and consensus.
pub fn write_derived_parquet(path: &Path, table: &Table) -> Result<()> {
    write_parquet_with_compression(path, table, Compression::SNAPPY)
}

/// Replace requested columns for one date, retaining other columns and stock rows.
pub(super) fn write_consensus_columns(path: &Path, updates: Table, date: i32) -> Result<()> {
    let table = if path.exists() {
        merge_consensus_columns(read_parquet(path, None)?, updates, date)?
    } else {
        consensus_row_keys(&updates, date)?;
        updates
    };
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let temp = path.with_extension(format!("{}.{}.tmp", std::process::id(), nonce));
    let result = (|| {
        write_derived_parquet(&temp, &table)?;
        std::fs::OpenOptions::new()
            .write(true)
            .open(&temp)?
            .sync_all()?;
        std::fs::rename(&temp, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result
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

fn bar_rows_table(rows: &[DerivedBarRow]) -> Result<Table> {
    Table::new(BTreeMap::from([
        (
            "trade_date".to_string(),
            ColumnData::I32(rows.iter().map(|row| Some(row.trade_date)).collect()),
        ),
        (
            "trade_time".to_string(),
            ColumnData::Utf8(
                rows.iter()
                    .map(|row| Some(row.trade_time.clone()))
                    .collect(),
            ),
        ),
        (
            "bar_index".to_string(),
            ColumnData::I32(rows.iter().map(|row| Some(row.bar_index)).collect()),
        ),
        (
            "ts_code".to_string(),
            ColumnData::Utf8(rows.iter().map(|row| Some(row.ts_code.clone())).collect()),
        ),
        (
            "open".to_string(),
            ColumnData::F32(rows.iter().map(|row| Some(row.open as f32)).collect()),
        ),
        (
            "high".to_string(),
            ColumnData::F32(rows.iter().map(|row| Some(row.high as f32)).collect()),
        ),
        (
            "low".to_string(),
            ColumnData::F32(rows.iter().map(|row| Some(row.low as f32)).collect()),
        ),
        (
            "close".to_string(),
            ColumnData::F32(rows.iter().map(|row| Some(row.close as f32)).collect()),
        ),
        (
            "volume".to_string(),
            ColumnData::F64(rows.iter().map(|row| Some(row.volume)).collect()),
        ),
        (
            "amount".to_string(),
            ColumnData::F64(rows.iter().map(|row| Some(row.amount)).collect()),
        ),
        (
            "vwap".to_string(),
            ColumnData::F64(rows.iter().map(|row| row.vwap).collect()),
        ),
        (
            "minute_count".to_string(),
            ColumnData::I32(rows.iter().map(|row| Some(row.minute_count)).collect()),
        ),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::parquet_io::read_parquet;
    use parquet::file::reader::{FileReader, SerializedFileReader};

    fn consensus_table(codes: &[&str], name: &str, values: Vec<Option<f64>>) -> Table {
        Table::new(BTreeMap::from([
            (
                "trade_date".into(),
                ColumnData::I32(vec![Some(20260105); codes.len()]),
            ),
            (
                "ts_code".into(),
                ColumnData::Utf8(codes.iter().map(|code| Some(code.to_string())).collect()),
            ),
            (name.into(), ColumnData::F64(values)),
        ]))
        .unwrap()
    }

    #[test]
    fn consensus_column_merge_aligns_stocks_and_replaces_nulls() {
        let mut old = consensus_table(&["B", "A", "D"], "growth", vec![Some(9.0); 3]);
        old.columns.insert(
            "untouched".into(),
            ColumnData::F64(vec![Some(2.0), Some(1.0), Some(4.0)]),
        );
        old.columns.insert(
            "text".into(),
            ColumnData::Utf8(vec![Some("keep".into()), None, Some("also".into())]),
        );
        let updates = consensus_table(
            &["A", "C", "B"],
            "growth",
            vec![None, Some(30.0), Some(20.0)],
        );
        let merged = merge_consensus_columns(old, updates, 20260105).unwrap();
        assert_eq!(
            merged.required_utf8("ts_code").unwrap(),
            &[
                Some("B".into()),
                Some("A".into()),
                Some("D".into()),
                Some("C".into())
            ]
        );
        assert_eq!(
            merged.required_f64_cast("growth").unwrap(),
            vec![Some(20.0), None, None, Some(30.0)]
        );
        assert_eq!(
            merged.required_f64_cast("untouched").unwrap(),
            vec![Some(2.0), Some(1.0), Some(4.0), None]
        );
        assert_eq!(
            merged.required_utf8("text").unwrap(),
            &[Some("keep".into()), None, Some("also".into()), None]
        );
    }

    #[test]
    fn consensus_column_merge_rejects_duplicate_and_wrong_date_keys() {
        let old = consensus_table(&["A"], "growth", vec![Some(1.0)]);
        let duplicate = consensus_table(&["A", "A"], "growth", vec![None; 2]);
        assert!(merge_consensus_columns(old.clone(), duplicate, 20260105).is_err());
        let updates = consensus_table(&["A"], "growth", vec![None]);
        assert!(merge_consensus_columns(old.clone(), updates, 20260106).is_err());
        assert!(
            merge_consensus_columns(old, consensus_table(&[], "growth", vec![]), 20260105).is_err()
        );
    }

    #[test]
    fn consensus_partial_write_is_snappy_and_preserves_existing_file_on_error() {
        let dir = std::env::temp_dir().join(format!(
            "yq-consensus-partial-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = dir.join("20260105.parquet");
        let old = consensus_table(&["A"], "untouched", vec![Some(7.0)]);
        write_consensus_columns(&path, old, 20260105).unwrap();
        let update = consensus_table(&["A"], "growth", vec![Some(20.0)]);
        write_consensus_columns(&path, update, 20260105).unwrap();
        let read = read_parquet(&path, None).unwrap();
        assert_eq!(
            read.required_f64_cast("untouched").unwrap(),
            vec![Some(7.0)]
        );
        assert_eq!(read.required_f64_cast("growth").unwrap(), vec![Some(20.0)]);
        let before = std::fs::read(&path).unwrap();
        let bad = consensus_table(&["A", "A"], "growth", vec![None; 2]);
        assert!(write_consensus_columns(&path, bad, 20260105).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), before);
        let reader = SerializedFileReader::new(std::fs::File::open(&path).unwrap()).unwrap();
        for group in reader.metadata().row_groups() {
            for column in group.columns() {
                assert_eq!(column.compression(), Compression::SNAPPY);
            }
        }
        drop(reader);
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn derived_storage_snappy_roundtrip_preserves_values_and_nulls() {
        let path = std::env::temp_dir().join(format!(
            "yq-derived-snappy-{}-{}.parquet",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let table = Table::new(BTreeMap::from([
            (
                "trade_date".into(),
                ColumnData::I32(vec![Some(20260105), Some(20260106)]),
            ),
            (
                "value".into(),
                ColumnData::F64(vec![Some(1.234567890123), None]),
            ),
        ]))
        .unwrap();
        write_derived_parquet(&path, &table).unwrap();
        let reader = SerializedFileReader::new(std::fs::File::open(&path).unwrap()).unwrap();
        for group in reader.metadata().row_groups() {
            for column in group.columns() {
                assert_eq!(column.compression(), Compression::SNAPPY);
            }
        }
        drop(reader);
        let restored = read_parquet(&path, None).unwrap();
        assert_eq!(
            restored.required_f64_cast("value").unwrap(),
            vec![Some(1.234567890123), None]
        );
        assert_eq!(
            restored.required_i32("trade_date").unwrap(),
            &vec![Some(20260105), Some(20260106)]
        );
        std::fs::remove_file(path).unwrap();
    }
}
