//! One logical daily factor table over independently committed manual/ML files.
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::artifact_io::{self, StoragePathExt};
use crate::data::parquet_io::{parquet_column_names, read_parquet, read_projected_reader, read_projected_reader_excluding, write_parquet};
use crate::data::{ColumnData, Table};
use crate::error::{err, Result};

pub fn factor_file_date(path: &Path) -> Option<i32> {
    if path.extension()?.to_str()? != "parquet" { return None; }
    let stem = path.file_stem()?.to_str()?;
    let date = stem.strip_suffix("_ml").unwrap_or(stem);
    (date.len() == 8 && date.bytes().all(|c| c.is_ascii_digit())).then(|| date.parse().ok()).flatten()
}

pub fn ml_path(path: &Path) -> PathBuf {
    path.with_file_name(format!("{}_ml.parquet", path.file_stem().unwrap_or_default().to_string_lossy()))
}

#[derive(Debug, Default)]
pub struct FactorFiles {
    owners: BTreeMap<String, String>,
}

struct ReadPlan { path: PathBuf, manual: bool, ml: bool, requested_manual: bool, primary_ml: bool }

impl FactorFiles {
    pub fn new(root: &Path) -> Result<Self> {
        let mut result = Self::default();
        for parent in root.ancestors().take(3) {
            let metadata = parent.join("factor_metadata.parquet");
            if !metadata.storage_exists() { continue; }
            let table = read_parquet(&metadata, Some(&["output_column".into(), "tags_json".into(), "storage_partition".into(),
                "asset_class".into(), "frequency".into()]))?;
            let names = table.required_utf8("output_column")?;
            let tags = table.required_utf8("tags_json")?;
            let partitions = table.columns.get("storage_partition").map(|_| table.required_utf8("storage_partition")).transpose()?;
            let assets = table.columns.get("asset_class").map(|_| table.required_utf8("asset_class")).transpose()?;
            let frequencies = table.columns.get("frequency").map(|_| table.required_utf8("frequency")).transpose()?;
            let asset = root.parent().and_then(|path| path.file_name()).and_then(|name| name.to_str());
            let frequency = root.file_name().and_then(|name| name.to_str());
            let scoped = asset.is_some() && frequency.is_some()
                && assets.is_some_and(|values| values.iter().any(|value| value.as_deref() == asset))
                && frequencies.is_some_and(|values| values.iter().any(|value| value.as_deref() == frequency));
            let mut identities = BTreeSet::new();
            for i in 0..table.len {
                if scoped && (assets.unwrap()[i].as_deref() != asset || frequencies.unwrap()[i].as_deref() != frequency) { continue; }
                let column = names[i].as_deref().ok_or_else(|| err("null factor output_column"))?;
                let identity = (assets.and_then(|values| values[i].as_deref()).unwrap_or(""),
                    frequencies.and_then(|values| values[i].as_deref()).unwrap_or(""), column);
                if !identities.insert(identity) { return Err(err(format!("duplicate factor metadata output_column: {column}"))); }
                let tags: Vec<String> = serde_json::from_str(tags[i].as_deref().filter(|s| !s.is_empty()).unwrap_or("[]"))?;
                let inferred = if tags.iter().any(|t| t == "model_generated") { "ml" } else { "manual" };
                let owner = partitions.and_then(|values| values[i].as_deref()).filter(|s| !s.is_empty()).unwrap_or(inferred);
                if owner != inferred { return Err(err(format!("invalid storage_partition for {column}: {owner}"))); }
                if result.owners.get(column).is_some_and(|previous| previous != owner) {
                    return Err(err(format!("ambiguous factor metadata ownership for {column}; use an asset/frequency factor root")));
                }
                result.owners.insert(column.into(), owner.into());
            }
            break;
        }
        Ok(result)
    }

    pub fn exists(&self, path: &Path) -> bool {
        path.storage_exists() || ml_path(path).storage_exists()
    }

    pub fn read(&self, path: &Path, columns: Option<&[String]>) -> Result<Table> {
        let plan = self.read_plan(path, columns);
        if !plan.manual && !plan.ml { return Ok(Table::empty()); }
        let projection = source_columns(columns);
        let primary_path = plan.primary();
        let primary = self.read_primary(&plan, artifact_io::open_reader(&primary_path)?, projection.as_deref())?;
        self.finish_read(plan, columns, primary)
    }

    /// Share the file open with schema discovery, and preserve caller date order.
    pub(crate) fn read_many(
        &self, paths: &[PathBuf], columns: &[String],
        mut consume: impl FnMut(&Path, Table) -> Result<()>,
    ) -> Result<()> {
        let plans = paths.iter().map(|path| self.read_plan(path, Some(columns)))
            .filter(|plan| plan.manual || plan.ml).collect::<Vec<_>>();
        let primary = plans.iter().map(ReadPlan::primary).collect::<Vec<_>>();
        let mut plans = plans.into_iter();
        let projection = source_columns(Some(columns)).expect("explicit projection");
        crate::data::read_ahead::visit_files(&primary, |_, reader| {
            let plan = plans.next().expect("one plan per primary reader");
            let output_path = plan.path.clone();
            let table = self.read_primary(&plan, reader, Some(&projection))?;
            consume(&output_path, self.finish_read(plan, Some(columns), table)?)
        })
    }

    fn read_primary(&self, plan: &ReadPlan, reader: artifact_io::StorageReader, columns: Option<&[String]>) -> Result<Table> {
        let path = plan.primary();
        if plan.primary_ml {
            if plan.requested_manual {
                return read_projected_reader(reader, &path, Some(&["trade_date".into(), "ts_code".into()]));
            }
            let excluded = self.owners.iter().filter(|(_, owner)| owner.as_str() == "manual")
                .map(|(name, _)| name.as_str()).collect();
            read_projected_reader_excluding(reader, &path, columns, &excluded)
        } else {
            read_projected_reader(reader, &path, columns)
        }
    }

    fn read_plan(&self, path: &Path, columns: Option<&[String]>) -> ReadPlan {
        let manual = path.storage_exists();
        let requested_manual = columns.is_some_and(|names| names.iter().any(|n| !is_key(n)) &&
            names.iter().filter(|n| !is_key(n)).all(|n| self.owners.get(n).is_some_and(|p| p == "manual")));
        let ml = ml_path(path).storage_exists();
        ReadPlan { path: path.to_owned(), manual, ml, requested_manual, primary_ml: ml && (!requested_manual || !manual) }
    }

    fn finish_read(&self, plan: ReadPlan, columns: Option<&[String]>, primary: Table) -> Result<Table> {
        let ml_schema = if plan.primary_ml && !plan.requested_manual { primary.column_names() } else { Vec::new() };
        let manual_needed = columns.is_none_or(|names| names.iter().all(|n| is_key(n)) || names.iter().filter(|n| !is_key(n))
            .any(|n| self.owners.get(n).is_some_and(|p| p == "manual") || !ml_schema.contains(n)));
        let (mut manual, mut ml) = if plan.primary_ml { (None, Some(primary)) } else { (Some(primary), None) };
        if plan.manual && manual_needed && manual.is_none() {
            let projection = source_columns(columns).map(|names| names.into_iter().filter(|n| is_key(n)
                || self.owners.get(n).is_some_and(|p| p == "manual") || !ml_schema.contains(n))
                .collect::<Vec<_>>());
            manual = Some(read_parquet(&plan.path, projection.as_deref())?);
        }
        let manual_schema = manual.as_ref().map(Table::column_names).unwrap_or_default();
        let names = match columns {
            Some(names) => names.iter().filter(|n| !is_key(n)).cloned().collect::<BTreeSet<_>>(),
            None => manual_schema.iter().chain(&ml_schema).filter(|n| !is_key(n)).cloned().collect(),
        };
        let mut left = Vec::new();
        let mut right = Vec::new();
        for name in names {
            if self.owners.get(&name).is_none_or(|p| p != "manual") && ml_schema.contains(&name) {
                right.push(name);
            } else if manual_schema.contains(&name) {
                left.push(name);
            }
        }
        let mut tables = Vec::new();
        for (source, selected) in [(&mut manual, left), (&mut ml, right)] {
            if selected.is_empty() { continue; }
            let mut table = source.take().ok_or_else(|| err("missing factor source reader"))?;
            table.columns.retain(|name, _| is_key(name) || selected.contains(name));
            tables.push(table);
        }
        if tables.is_empty() {
            let mut table = manual.or(ml).ok_or_else(|| err("missing factor key source"))?;
            table.columns.retain(|name, _| is_key(name));
            tables.push(table);
        }
        merge_tables(tables, factor_file_date(&plan.path))
    }

    /// Explicit, restartable migration. Publish ML first, then remove old ML columns.
    /// Writers normally take only their own file's lock; migration takes manual then ML.
    pub fn migrate(&self, path: &Path, dry_run: bool) -> Result<Vec<String>> {
        if !path.storage_exists() { return Ok(Vec::new()); }
        let _manual = artifact_io::lock(path)?;
        let columns = parquet_column_names(path)?;
        let moving = columns.iter().filter(|c| self.owners.get(*c).is_some_and(|p| p == "ml")).cloned().collect::<Vec<_>>();
        if moving.is_empty() || dry_run { return Ok(moving); }
        let destination = ml_path(path);
        let _ml = artifact_io::lock(&destination)?;
        if destination.storage_exists() && parquet_column_names(&destination)?.iter()
            .any(|c| self.owners.get(c).is_some_and(|owner| owner == "manual")) {
            return Err(err("ML file contains a manual-owned column; resolve metadata ownership before migration"));
        }
        // Existing ML columns own their entire cross-section, including nulls.
        let all = self.read(path, None)?;
        let mut wanted = moving.iter().cloned().collect::<BTreeSet<_>>();
        if destination.storage_exists() { wanted.extend(parquet_column_names(&destination)?.into_iter().filter(|c| !is_key(c))); }
        let mut target = all.clone();
        target.columns.retain(|name, _| is_key(name) || wanted.contains(name));
        write_parquet(&destination, &target)?;
        let mut original = read_parquet(path, None)?;
        original.columns.retain(|name, _| !moving.contains(name));
        write_parquet(path, &original)?;
        Ok(moving)
    }
}

impl ReadPlan {
    fn primary(&self) -> PathBuf { if self.primary_ml { ml_path(&self.path) } else { self.path.clone() } }
}

fn is_key(name: &str) -> bool { matches!(name, "trade_date" | "ts_code") }

fn source_columns(columns: Option<&[String]>) -> Option<Vec<String>> {
    columns.map(|names| {
        let mut columns = vec!["trade_date".into(), "ts_code".into()];
        columns.extend(names.iter().filter(|name| !is_key(name)).cloned());
        columns
    })
}

fn merge_tables(tables: Vec<Table>, date: Option<i32>) -> Result<Table> {
    let mut indexed = Vec::new();
    let mut keys = BTreeSet::new();
    for table in &tables {
        let dates = table.required_i32_date_cast("trade_date")?;
        let symbols = table.required_utf8("ts_code")?;
        let mut indices = BTreeMap::new();
        for i in 0..table.len {
            let d = dates[i].ok_or_else(|| err("null factor trade_date"))?;
            let symbol = symbols[i].as_ref().ok_or_else(|| err("null factor ts_code"))?;
            if date.is_some_and(|expected| d != expected) { return Err(err("factor trade_date differs from filename")); }
            let key = (d, symbol.clone());
            if indices.insert(key.clone(), i).is_some() { return Err(err("duplicate daily factor row keys")); }
            keys.insert(key);
        }
        indexed.push(indices);
    }
    if tables.len() == 1 {
        return Ok(tables.into_iter().next().expect("one projected table"));
    }
    let keys = keys.into_iter().collect::<Vec<_>>();
    let mut result = Table::empty();
    result.insert("trade_date", ColumnData::I32(keys.iter().map(|k| Some(k.0)).collect()))?;
    result.insert("ts_code", ColumnData::Utf8(keys.iter().map(|k| Some(k.1.clone())).collect()))?;
    for (table, indices) in tables.iter().zip(indexed) {
        let rows = keys.iter().map(|key| indices.get(key).copied()).collect::<Vec<_>>();
        for (name, values) in &table.columns {
            if is_key(name) { continue; }
            if result.columns.contains_key(name) { return Err(err(format!("factor column has two owners: {name}"))); }
            result.insert(name, reindex(values, &rows))?;
        }
    }
    Ok(result)
}

fn reindex(column: &ColumnData, rows: &[Option<usize>]) -> ColumnData {
    macro_rules! take { ($values:expr, $variant:ident) => {
        ColumnData::$variant(rows.iter().map(|row| row.and_then(|i| $values[i].clone())).collect())
    }; }
    match column {
        ColumnData::Utf8(v) => take!(v, Utf8), ColumnData::I32(v) => take!(v, I32),
        ColumnData::I64(v) => take!(v, I64), ColumnData::F32(v) => take!(v, F32),
        ColumnData::F64(v) => take!(v, F64), ColumnData::Bool(v) => take!(v, Bool),
    }
}
