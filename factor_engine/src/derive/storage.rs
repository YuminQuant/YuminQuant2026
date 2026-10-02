use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use parquet::basic::Compression;

use crate::data::parquet_io::write_parquet_with_compression;
use crate::data::{ColumnData, Table};
use crate::derive::bar::DerivedBarRow;
use crate::error::Result;

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
