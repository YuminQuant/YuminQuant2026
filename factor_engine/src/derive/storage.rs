use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::data::parquet_io::write_parquet;
use crate::data::{ColumnData, Table};
use crate::derive::bar::DerivedBarRow;
use crate::derive::logsig::LogsigSignatureRow;
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
    write_parquet(path, &bar_rows_table(rows)?)
}

pub fn derived_logsig_volume_signature_path(data_root: &Path, trade_date: i32) -> PathBuf {
    data_root
        .join("derived")
        .join("stock")
        .join("logsig_alpha_v")
        .join("signature_o10_20d_5m")
        .join((trade_date / 10_000).to_string())
        .join(format!("{trade_date}.parquet"))
}

pub fn write_logsig_signature_rows(path: &Path, rows: &[LogsigSignatureRow]) -> Result<()> {
    write_parquet(path, &logsig_signature_rows_table(rows)?)
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

fn logsig_signature_rows_table(rows: &[LogsigSignatureRow]) -> Result<Table> {
    let mut columns = BTreeMap::from([
        (
            "trade_date".to_string(),
            ColumnData::I32(rows.iter().map(|row| Some(row.trade_date)).collect()),
        ),
        (
            "ts_code".to_string(),
            ColumnData::Utf8(rows.iter().map(|row| Some(row.ts_code.clone())).collect()),
        ),
    ]);
    let width = rows.first().map(|row| row.values.len()).unwrap_or(0);
    for idx in 0..width {
        columns.insert(
            format!("sig_{:04}", idx + 1),
            ColumnData::F32(rows.iter().map(|row| Some(row.values[idx])).collect()),
        );
    }
    Table::new(columns)
}
