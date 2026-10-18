use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use arrow_array::{BooleanArray, RecordBatch, StringArray};
use arrow_schema::Schema;
use parquet::arrow::arrow_reader::{ArrowPredicateFn, ParquetRecordBatchReaderBuilder, RowFilter};
use parquet::arrow::{ArrowWriter, ProjectionMask};
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use crate::data::table::{ColumnData, Table};
use crate::error::{err, Result};

pub fn read_parquet(path: &Path, columns: Option<&[String]>) -> Result<Table> {
    read_parquet_impl(path, columns, None)
}

/// Filter symbols before materializing the remaining projected columns into Table.
pub(crate) fn read_parquet_symbols(
    path: &Path,
    columns: &[String],
    predicate: impl Fn(&str) -> bool + Send + 'static,
) -> Result<Table> {
    read_parquet_impl(path, Some(columns), Some(Box::new(predicate)))
}

type SymbolPredicate = Box<dyn Fn(&str) -> bool + Send>;

fn read_parquet_impl(
    path: &Path,
    columns: Option<&[String]>,
    predicate: Option<SymbolPredicate>,
) -> Result<Table> {
    let context = |error| {
        err(format!(
            "failed to read parquet {}: {error}",
            path.display()
        ))
    };
    let file = File::open(path).map_err(|error| context(error.to_string()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|error| context(error.to_string()))?;
    let mut builder = builder.with_batch_size(65_536);

    if let Some(requested_columns) = columns {
        let schema = builder.schema().clone();
        let missing_required = requested_columns
            .iter()
            .filter(|requested| {
                matches!(requested.as_str(), "trade_date" | "trade_time" | "ts_code")
                    && !schema
                        .fields()
                        .iter()
                        .any(|field| field.name() == *requested)
            })
            .cloned()
            .collect::<Vec<_>>();
        if !missing_required.is_empty() {
            return Err(err(format!(
                "missing required parquet column(s) {} in {}; requested columns {:?}",
                missing_required.join(","),
                path.display(),
                requested_columns
            )));
        }
        let mut projection_indices = Vec::new();
        for requested in requested_columns {
            if let Some((idx, _)) = schema
                .fields()
                .iter()
                .enumerate()
                .find(|(_, field)| field.name() == requested)
            {
                projection_indices.push(idx);
            }
        }
        if projection_indices.is_empty() && !requested_columns.is_empty() {
            return Err(err(format!(
                "none of requested columns {:?} exist in {}",
                requested_columns,
                path.display()
            )));
        }
        let mask = ProjectionMask::roots(builder.parquet_schema(), projection_indices);
        builder = builder.with_projection(mask);
    }

    if let Some(predicate) = predicate {
        let index = builder.schema().index_of("ts_code")?;
        let projection = ProjectionMask::roots(builder.parquet_schema(), [index]);
        let filter = ArrowPredicateFn::new(projection, move |batch: RecordBatch| {
            let codes = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    arrow_schema::ArrowError::ComputeError("ts_code is not utf8".into())
                })?;
            Ok(BooleanArray::from(
                codes
                    .iter()
                    .map(|code| code.map(&predicate).unwrap_or(false))
                    .collect::<Vec<_>>(),
            ))
        });
        builder = builder.with_row_filter(RowFilter::new(vec![Box::new(filter)]));
    }
    // Keep the schema even when a predicate selects no rows.
    let empty_batch = RecordBatch::new_empty(
        builder
            .schema()
            .clone()
            .project(
                &builder
                    .schema()
                    .fields()
                    .iter()
                    .enumerate()
                    .filter(|(_, field)| columns.map_or(true, |names| names.contains(field.name())))
                    .map(|(index, _)| index)
                    .collect::<Vec<_>>(),
            )?
            .into(),
    );
    let reader = builder
        .build()
        .map_err(|error| context(error.to_string()))?;
    let mut table = table_from_record_batch(&empty_batch)?;
    for batch_result in reader {
        let batch = batch_result.map_err(|error| context(error.to_string()))?;
        let batch_table = table_from_record_batch(&batch)?;
        table.append(&batch_table)?;
    }
    Ok(table)
}

pub fn parquet_column_names(path: &Path) -> Result<BTreeSet<String>> {
    let file = File::open(path).map_err(|error| {
        err(format!(
            "failed to read parquet {}: {error}",
            path.display()
        ))
    })?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|error| {
        err(format!(
            "failed to read parquet {}: {error}",
            path.display()
        ))
    })?;
    Ok(builder
        .schema()
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect())
}

pub fn write_parquet(path: &Path, table: &Table) -> Result<()> {
    write_parquet_with_compression(path, table, Compression::UNCOMPRESSED)
}

pub fn write_parquet_with_compression(
    path: &Path,
    table: &Table,
    compression: Compression,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let ordered_names = ordered_column_names(table);
    let fields = ordered_names
        .iter()
        .map(|name| table.columns[name].field(name))
        .collect::<Vec<_>>();
    let arrays = ordered_names
        .iter()
        .map(|name| table.columns[name].to_arrow())
        .collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), arrays)?;
    let props = WriterProperties::builder()
        .set_compression(compression)
        .build();
    static WRITE_SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let sequence = WRITE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temp = path.with_extension(format!("{}.{}.{}.tmp", std::process::id(), nonce, sequence));
    let file = File::create_new(&temp).map_err(|error| {
        err(format!(
            "failed to write parquet {}: {error}",
            path.display()
        ))
    })?;
    // Never truncate the previous output before a complete footer is durable.
    let result: Result<()> = (|| {
        let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
        writer.write(&batch)?;
        writer.close()?;
        File::options().write(true).open(&temp)?.sync_all()?;
        std::fs::rename(&temp, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result.map_err(|error| {
        err(format!(
            "failed to write parquet {}: {error}",
            path.display()
        ))
    })
}

/// Incremental Parquet output. Only publish after the footer is complete and synced.
pub(crate) struct ParquetStreamWriter {
    path: std::path::PathBuf,
    temp: std::path::PathBuf,
    writer: Option<ArrowWriter<File>>,
}

impl ParquetStreamWriter {
    pub(crate) fn new(path: &Path, empty: &Table) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        static SEQUENCE: AtomicU64 = AtomicU64::new(0);
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let temp = path.with_extension(format!(
            "{}.{}.{}.tmp",
            std::process::id(),
            nonce,
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let batch = table_record_batch(empty)?;
        let file = File::create_new(&temp)?;
        let mut out = Self {
            path: path.to_path_buf(),
            temp,
            writer: None,
        };
        let props = WriterProperties::builder()
            .set_compression(Compression::UNCOMPRESSED)
            .build();
        out.writer = Some(ArrowWriter::try_new(file, batch.schema(), Some(props))?);
        Ok(out)
    }

    pub(crate) fn write(&mut self, table: &Table) -> Result<()> {
        let batch = table_record_batch(table)?;
        let writer = self.writer.as_mut().expect("writer already finished");
        writer.write(&batch)?;
        // Do not let ArrowWriter retain row groups across our bounded batches.
        writer.flush()?;
        Ok(())
    }

    pub(crate) fn finish(mut self) -> Result<std::path::PathBuf> {
        self.writer
            .take()
            .expect("writer already finished")
            .close()?;
        File::options().write(true).open(&self.temp)?.sync_all()?;
        std::fs::rename(&self.temp, &self.path)?;
        Ok(self.path.clone())
    }
}

impl Drop for ParquetStreamWriter {
    fn drop(&mut self) {
        drop(self.writer.take());
        let _ = std::fs::remove_file(&self.temp);
    }
}

fn table_record_batch(table: &Table) -> Result<RecordBatch> {
    let names = ordered_column_names(table);
    let schema = Arc::new(Schema::new(
        names
            .iter()
            .map(|name| table.columns[name].field(name))
            .collect::<Vec<_>>(),
    ));
    Ok(RecordBatch::try_new(
        schema,
        names
            .iter()
            .map(|name| table.columns[name].to_arrow())
            .collect(),
    )?)
}

fn ordered_column_names(table: &Table) -> Vec<String> {
    let priority = [
        "trade_date",
        "trade_time",
        "bar_index",
        "ts_code",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "vwap",
        "minute_count",
        "raw_id",
        "factor_id",
        "version",
        "source_dataset",
        "source_bar_size",
        "output_column",
        "name",
        "asset_class",
        "frequency",
        "columns_json",
        "window_days",
        "tags_json",
        "dependencies_json",
        "description",
        "updated_at",
    ];
    let mut names = Vec::new();
    for column in priority {
        if table.columns.contains_key(column) {
            names.push(column.to_string());
        }
    }
    for column in table.columns.keys() {
        if !names.iter().any(|name| name == column) {
            names.push(column.clone());
        }
    }
    names
}

fn table_from_record_batch(batch: &RecordBatch) -> Result<Table> {
    let mut columns = BTreeMap::new();
    for (field, array) in batch.schema().fields().iter().zip(batch.columns()) {
        columns.insert(field.name().clone(), ColumnData::from_arrow(array)?);
    }
    Table::new(columns)
}
