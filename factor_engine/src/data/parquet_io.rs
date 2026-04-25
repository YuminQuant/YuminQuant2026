use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::{ArrowWriter, ProjectionMask};
use parquet::file::properties::WriterProperties;

use crate::data::table::{ColumnData, Table};
use crate::error::{err, Result};

pub fn read_parquet(path: &Path, columns: Option<&[String]>) -> Result<Table> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let mut builder = builder.with_batch_size(65_536);

    if let Some(requested_columns) = columns {
        let schema = builder.schema().clone();
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

    let reader = builder.build()?;
    let mut table = Table::empty();
    for batch_result in reader {
        let batch = batch_result?;
        let batch_table = table_from_record_batch(&batch)?;
        table.append(&batch_table)?;
    }
    Ok(table)
}

pub fn write_parquet(path: &Path, table: &Table) -> Result<()> {
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
    let file = File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn ordered_column_names(table: &Table) -> Vec<String> {
    let priority = [
        "trade_date",
        "trade_time",
        "ts_code",
        "factor_id",
        "version",
        "output_column",
        "name",
        "asset_class",
        "frequency",
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
