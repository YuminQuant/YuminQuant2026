use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::core::{
    AssetClass, FactorRowKey, FactorSeries, FactorSpec, Frequency, IntradayDailyRawSeries,
    IntradayDailyRawSpec,
};
use crate::data::parquet_io::{read_parquet, write_parquet};
use crate::data::table::{ColumnData, Table};
use crate::error::{err, Result};

#[derive(Clone, Debug)]
pub struct FactorStorage {
    factor_root: PathBuf,
}

#[derive(Clone, Debug)]
pub struct IntradayDailyRawStorage {
    factor_root: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
struct OutputKey {
    ts_code: String,
    trade_time: Option<String>,
}

#[derive(Clone, Debug, Default)]
struct PendingFrame {
    values: BTreeMap<OutputKey, BTreeMap<String, Option<f64>>>,
}

#[derive(Clone, Debug)]
struct MetadataRow {
    factor_id: String,
    aliases_json: String,
    version: String,
    output_column: String,
    name: String,
    asset_class: String,
    frequency: String,
    tags_json: String,
    dependencies_json: String,
    description: String,
    updated_at: String,
}

#[derive(Clone, Debug)]
struct RawMetadataRow {
    raw_id: String,
    version: String,
    asset_class: String,
    source_dataset: String,
    columns_json: String,
    window_days: i32,
    updated_at: String,
}

#[derive(Clone, Debug)]
pub struct FactorMetadata {
    pub factor_id: String,
    pub aliases: Vec<String>,
    pub aliases_json: String,
    pub version: String,
    pub output_column: String,
    pub name: String,
    pub asset_class: String,
    pub frequency: String,
    pub tags: Vec<String>,
    pub tags_json: String,
    pub dependencies_json: String,
    pub description: String,
    pub updated_at: String,
}

impl FactorStorage {
    pub fn new(factor_root: PathBuf) -> Self {
        Self { factor_root }
    }

    pub fn write_results(&self, results: &[FactorSeries]) -> Result<Vec<PathBuf>> {
        let mut grouped: BTreeMap<(AssetClass, Frequency, i32), PendingFrame> = BTreeMap::new();
        for series in results {
            let column_name = series.spec.output_column();
            for item in &series.values {
                let trade_date = item.key.trade_date();
                let output_key = match &item.key {
                    FactorRowKey::Daily { ts_code, .. } => OutputKey {
                        ts_code: ts_code.clone(),
                        trade_time: None,
                    },
                    FactorRowKey::Minute {
                        ts_code,
                        trade_time,
                        ..
                    } => OutputKey {
                        ts_code: ts_code.clone(),
                        trade_time: Some(trade_time.clone()),
                    },
                };
                grouped
                    .entry((series.spec.asset_class, series.spec.frequency, trade_date))
                    .or_default()
                    .values
                    .entry(output_key)
                    .or_default()
                    .insert(column_name.clone(), item.value);
            }
        }

        let mut written = Vec::new();
        for ((asset_class, frequency, trade_date), mut frame) in grouped {
            let path = self.output_path(asset_class, frequency, trade_date);
            merge_existing_output(&path, frequency, trade_date, &mut frame)?;
            let table = pending_frame_to_table(frequency, trade_date, &frame)?;
            write_parquet(&path, &table)?;
            written.push(path);
        }
        Ok(written)
    }

    pub fn write_metadata(&self, specs: &[FactorSpec]) -> Result<()> {
        std::fs::create_dir_all(&self.factor_root)?;
        let path = self.factor_root.join("factor_metadata.parquet");

        let updated_at = unix_timestamp_string();
        let mut rows = Vec::new();
        for spec in specs {
            rows.push(MetadataRow {
                factor_id: spec.id.clone(),
                aliases_json: string_list_json(&spec.aliases),
                version: spec.version.clone(),
                output_column: spec.output_column(),
                name: spec.name.clone(),
                asset_class: spec.asset_class.as_str().to_string(),
                frequency: spec.frequency.as_str().to_string(),
                tags_json: string_list_json(&spec.tags),
                dependencies_json: dependencies_json(spec),
                description: spec.description.clone(),
                updated_at: updated_at.clone(),
            });
        }

        let table = metadata_rows_to_table(rows)?;
        write_parquet(&path, &table)
    }

    pub fn read_metadata(&self) -> Result<Vec<FactorMetadata>> {
        let path = self.factor_root.join("factor_metadata.parquet");
        if !path.exists() {
            return Err(err(format!(
                "factor metadata not found: {}. Run `metadata` first.",
                path.display()
            )));
        }
        read_metadata_records(&path)
    }

    fn output_path(
        &self,
        asset_class: AssetClass,
        frequency: Frequency,
        trade_date: i32,
    ) -> PathBuf {
        let year = trade_date / 10_000;
        self.factor_root
            .join(asset_class.as_str())
            .join(frequency.as_str())
            .join(year.to_string())
            .join(format!("{}.parquet", trade_date))
    }
}

impl IntradayDailyRawStorage {
    pub fn new(factor_root: PathBuf) -> Self {
        Self { factor_root }
    }

    pub fn missing_dates(
        &self,
        spec: &IntradayDailyRawSpec,
        dates: &[i32],
        refresh: bool,
    ) -> Result<Vec<i32>> {
        if refresh || !self.metadata_matches(spec)? {
            return Ok(dates.to_vec());
        }
        let mut missing = Vec::new();
        for date in dates {
            let path = self.output_path(spec.asset_class, *date);
            if !path.exists() {
                missing.push(*date);
                continue;
            }
            let table = read_parquet(&path, None)?;
            if !table.columns.contains_key(&spec.raw_id) {
                missing.push(*date);
            }
        }
        Ok(missing)
    }

    pub fn write_results(&self, results: &[IntradayDailyRawSeries]) -> Result<Vec<PathBuf>> {
        let mut grouped: BTreeMap<(AssetClass, i32), PendingFrame> = BTreeMap::new();
        for series in results {
            for item in &series.values {
                let trade_date = item.key.trade_date();
                let FactorRowKey::Daily { ts_code, .. } = &item.key else {
                    return Err(err("intraday daily raw cache only supports daily row keys"));
                };
                grouped
                    .entry((series.spec.asset_class, trade_date))
                    .or_default()
                    .values
                    .entry(OutputKey {
                        ts_code: ts_code.clone(),
                        trade_time: None,
                    })
                    .or_default()
                    .insert(series.spec.raw_id.clone(), item.value);
            }
        }

        let mut written = Vec::new();
        for ((asset_class, trade_date), mut frame) in grouped {
            let path = self.output_path(asset_class, trade_date);
            merge_existing_output(&path, Frequency::Daily, trade_date, &mut frame)?;
            let table = pending_frame_to_table(Frequency::Daily, trade_date, &frame)?;
            write_parquet(&path, &table)?;
            written.push(path);
        }
        Ok(written)
    }

    pub fn load_raw_by_dates(
        &self,
        asset_class: AssetClass,
        raw_ids: &[String],
        dates: &[i32],
    ) -> Result<Table> {
        let mut requested = vec!["trade_date".to_string(), "ts_code".to_string()];
        requested.extend(raw_ids.iter().cloned());
        let mut output = Table::empty();
        for date in dates {
            let path = self.output_path(asset_class, *date);
            if !path.exists() {
                continue;
            }
            let mut table = read_parquet(&path, Some(&requested))?;
            ensure_raw_columns(&mut table, raw_ids)?;
            output.append(&table)?;
        }
        Ok(output)
    }

    pub fn write_metadata(&self, specs: &[IntradayDailyRawSpec]) -> Result<()> {
        let path = self.metadata_path();
        let mut rows = if path.exists() {
            read_raw_metadata_records(&path)?
        } else {
            Vec::new()
        };
        let updated_at = unix_timestamp_string();
        for spec in specs {
            rows.retain(|row| {
                !(row.raw_id == spec.raw_id && row.asset_class == spec.asset_class.as_str())
            });
            rows.push(RawMetadataRow {
                raw_id: spec.raw_id.clone(),
                version: spec.version.clone(),
                asset_class: spec.asset_class.as_str().to_string(),
                source_dataset: spec.source_dataset.as_str().to_string(),
                columns_json: string_list_json(&spec.columns),
                window_days: spec.window_days as i32,
                updated_at: updated_at.clone(),
            });
        }
        rows.sort_by(|left, right| {
            left.asset_class
                .cmp(&right.asset_class)
                .then_with(|| left.raw_id.cmp(&right.raw_id))
        });
        let table = raw_metadata_rows_to_table(rows)?;
        write_parquet(&path, &table)
    }

    fn metadata_matches(&self, spec: &IntradayDailyRawSpec) -> Result<bool> {
        let path = self.metadata_path();
        if !path.exists() {
            return Ok(false);
        }
        let rows = read_raw_metadata_records(&path)?;
        Ok(rows.iter().any(|row| {
            row.raw_id == spec.raw_id
                && row.asset_class == spec.asset_class.as_str()
                && row.version == spec.version
                && row.source_dataset == spec.source_dataset.as_str()
                && row.window_days == spec.window_days as i32
                && parse_string_list_json(&row.columns_json) == spec.columns
        }))
    }

    fn output_path(&self, asset_class: AssetClass, trade_date: i32) -> PathBuf {
        let year = trade_date / 10_000;
        self.factor_root
            .join("_cache")
            .join("intraday_daily")
            .join(intraday_daily_raw_namespace(asset_class))
            .join(year.to_string())
            .join(format!("{}.parquet", trade_date))
    }

    fn metadata_path(&self) -> PathBuf {
        self.factor_root
            .join("_cache")
            .join("intraday_daily")
            .join("raw_metadata.parquet")
    }
}

fn intraday_daily_raw_namespace(asset_class: AssetClass) -> &'static str {
    match asset_class {
        AssetClass::Stock => "chn_stock",
        AssetClass::Future => "future",
    }
}

fn merge_existing_output(
    path: &Path,
    frequency: Frequency,
    trade_date: i32,
    frame: &mut PendingFrame,
) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    let table = read_parquet(path, None)?;
    let ts_codes = table.required_utf8("ts_code")?;
    let trade_times = if frequency == Frequency::Minute1 {
        Some(table.required_utf8("trade_time")?)
    } else {
        None
    };
    let factor_columns = table
        .column_names()
        .into_iter()
        .filter(|name| name != "trade_date" && name != "ts_code" && name != "trade_time")
        .collect::<Vec<_>>();
    let mut factor_values = BTreeMap::new();
    for column in &factor_columns {
        factor_values.insert(column.clone(), table.required_f64_cast(column)?);
    }

    for idx in 0..table.len {
        let Some(ts_code) = ts_codes[idx].clone() else {
            continue;
        };
        let key = OutputKey {
            ts_code,
            trade_time: trade_times.and_then(|values| values[idx].clone()),
        };
        let row = frame.values.entry(key).or_default();
        for column in &factor_columns {
            if row.contains_key(column) {
                continue;
            }
            if let Some(values) = factor_values.get(column) {
                row.insert(column.clone(), values[idx]);
            }
        }
    }

    if table.len > 0 {
        let _ = trade_date;
    }
    Ok(())
}

fn pending_frame_to_table(
    frequency: Frequency,
    trade_date: i32,
    frame: &PendingFrame,
) -> Result<Table> {
    let mut factor_columns = BTreeSet::new();
    for row in frame.values.values() {
        factor_columns.extend(row.keys().cloned());
    }
    let factor_columns = factor_columns.into_iter().collect::<Vec<_>>();

    let len = frame.values.len();
    let mut trade_dates = Vec::with_capacity(len);
    let mut ts_codes = Vec::with_capacity(len);
    let mut trade_times = Vec::with_capacity(len);
    let mut factors: BTreeMap<String, Vec<Option<f64>>> = factor_columns
        .iter()
        .map(|name| (name.clone(), Vec::with_capacity(len)))
        .collect();

    for (key, row) in &frame.values {
        trade_dates.push(Some(trade_date));
        ts_codes.push(Some(key.ts_code.clone()));
        if frequency == Frequency::Minute1 {
            trade_times.push(key.trade_time.clone());
        }
        for column in &factor_columns {
            factors
                .get_mut(column)
                .expect("factor column initialized")
                .push(row.get(column).copied().unwrap_or(None));
        }
    }

    let mut table = Table::empty();
    table.insert("trade_date", ColumnData::I32(trade_dates))?;
    if frequency == Frequency::Minute1 {
        table.insert("trade_time", ColumnData::Utf8(trade_times))?;
    }
    table.insert("ts_code", ColumnData::Utf8(ts_codes))?;
    for (column, values) in factors {
        table.insert(column, ColumnData::F64(values))?;
    }
    Ok(table)
}

fn read_metadata_records(path: &Path) -> Result<Vec<FactorMetadata>> {
    let table = read_parquet(path, None)?;
    let factor_id = table.required_utf8("factor_id")?;
    let aliases = if table.columns.contains_key("aliases_json") {
        Some(table.required_utf8("aliases_json")?)
    } else {
        None
    };
    let version = table.required_utf8("version")?;
    let output_column = table.required_utf8("output_column")?;
    let name = table.required_utf8("name")?;
    let asset_class = table.required_utf8("asset_class")?;
    let frequency = table.required_utf8("frequency")?;
    let tags_json = table.required_utf8("tags_json")?;
    let dependencies_json = table.required_utf8("dependencies_json")?;
    let description = table.required_utf8("description")?;
    let updated_at = table.required_utf8("updated_at")?;

    let mut rows = Vec::new();
    for idx in 0..table.len {
        let aliases_json_value = aliases
            .as_ref()
            .and_then(|values| values[idx].clone())
            .unwrap_or_default();
        let tags_json_value = tags_json[idx].clone().unwrap_or_default();
        let row = FactorMetadata {
            factor_id: factor_id[idx].clone().unwrap_or_default(),
            aliases: parse_string_list_json(&aliases_json_value),
            aliases_json: aliases_json_value,
            version: version[idx].clone().unwrap_or_default(),
            output_column: output_column[idx].clone().unwrap_or_default(),
            name: name[idx].clone().unwrap_or_default(),
            asset_class: asset_class[idx].clone().unwrap_or_default(),
            frequency: frequency[idx].clone().unwrap_or_default(),
            tags: parse_string_list_json(&tags_json_value),
            tags_json: tags_json_value,
            dependencies_json: dependencies_json[idx].clone().unwrap_or_default(),
            description: description[idx].clone().unwrap_or_default(),
            updated_at: updated_at[idx].clone().unwrap_or_default(),
        };
        rows.push(row);
    }
    Ok(rows)
}

fn parse_string_list_json(value: &str) -> Vec<String> {
    let mut items = Vec::new();
    let mut current = String::new();
    let mut in_string = false;
    let mut escaped = false;

    for ch in value.chars() {
        if escaped {
            current.push(ch);
            escaped = false;
            continue;
        }
        if ch == '\\' && in_string {
            escaped = true;
            continue;
        }
        if ch == '"' {
            if in_string {
                items.push(current.clone());
                current.clear();
            }
            in_string = !in_string;
            continue;
        }
        if in_string {
            current.push(ch);
        }
    }

    items
}

fn metadata_rows_to_table(rows: Vec<MetadataRow>) -> Result<Table> {
    let mut table = Table::empty();
    table.insert(
        "factor_id",
        ColumnData::Utf8(rows.iter().map(|row| Some(row.factor_id.clone())).collect()),
    )?;
    table.insert(
        "aliases_json",
        ColumnData::Utf8(
            rows.iter()
                .map(|row| Some(row.aliases_json.clone()))
                .collect(),
        ),
    )?;
    table.insert(
        "version",
        ColumnData::Utf8(rows.iter().map(|row| Some(row.version.clone())).collect()),
    )?;
    table.insert(
        "output_column",
        ColumnData::Utf8(
            rows.iter()
                .map(|row| Some(row.output_column.clone()))
                .collect(),
        ),
    )?;
    table.insert(
        "name",
        ColumnData::Utf8(rows.iter().map(|row| Some(row.name.clone())).collect()),
    )?;
    table.insert(
        "asset_class",
        ColumnData::Utf8(
            rows.iter()
                .map(|row| Some(row.asset_class.clone()))
                .collect(),
        ),
    )?;
    table.insert(
        "frequency",
        ColumnData::Utf8(rows.iter().map(|row| Some(row.frequency.clone())).collect()),
    )?;
    table.insert(
        "tags_json",
        ColumnData::Utf8(rows.iter().map(|row| Some(row.tags_json.clone())).collect()),
    )?;
    table.insert(
        "dependencies_json",
        ColumnData::Utf8(
            rows.iter()
                .map(|row| Some(row.dependencies_json.clone()))
                .collect(),
        ),
    )?;
    table.insert(
        "description",
        ColumnData::Utf8(
            rows.iter()
                .map(|row| Some(row.description.clone()))
                .collect(),
        ),
    )?;
    table.insert(
        "updated_at",
        ColumnData::Utf8(
            rows.iter()
                .map(|row| Some(row.updated_at.clone()))
                .collect(),
        ),
    )?;
    Ok(table)
}

fn read_raw_metadata_records(path: &Path) -> Result<Vec<RawMetadataRow>> {
    let table = read_parquet(path, None)?;
    let raw_id = table.required_utf8("raw_id")?;
    let version = table.required_utf8("version")?;
    let asset_class = table.required_utf8("asset_class")?;
    let source_dataset = table.required_utf8("source_dataset")?;
    let columns_json = table.required_utf8("columns_json")?;
    let window_days = table.required_i32("window_days")?;
    let updated_at = table.required_utf8("updated_at")?;

    let mut rows = Vec::new();
    for idx in 0..table.len {
        rows.push(RawMetadataRow {
            raw_id: raw_id[idx].clone().unwrap_or_default(),
            version: version[idx].clone().unwrap_or_default(),
            asset_class: asset_class[idx].clone().unwrap_or_default(),
            source_dataset: source_dataset[idx].clone().unwrap_or_default(),
            columns_json: columns_json[idx].clone().unwrap_or_default(),
            window_days: window_days[idx].unwrap_or_default(),
            updated_at: updated_at[idx].clone().unwrap_or_default(),
        });
    }
    Ok(rows)
}

fn raw_metadata_rows_to_table(rows: Vec<RawMetadataRow>) -> Result<Table> {
    let mut table = Table::empty();
    table.insert(
        "raw_id",
        ColumnData::Utf8(rows.iter().map(|row| Some(row.raw_id.clone())).collect()),
    )?;
    table.insert(
        "version",
        ColumnData::Utf8(rows.iter().map(|row| Some(row.version.clone())).collect()),
    )?;
    table.insert(
        "asset_class",
        ColumnData::Utf8(
            rows.iter()
                .map(|row| Some(row.asset_class.clone()))
                .collect(),
        ),
    )?;
    table.insert(
        "source_dataset",
        ColumnData::Utf8(
            rows.iter()
                .map(|row| Some(row.source_dataset.clone()))
                .collect(),
        ),
    )?;
    table.insert(
        "columns_json",
        ColumnData::Utf8(
            rows.iter()
                .map(|row| Some(row.columns_json.clone()))
                .collect(),
        ),
    )?;
    table.insert(
        "window_days",
        ColumnData::I32(rows.iter().map(|row| Some(row.window_days)).collect()),
    )?;
    table.insert(
        "updated_at",
        ColumnData::Utf8(
            rows.iter()
                .map(|row| Some(row.updated_at.clone()))
                .collect(),
        ),
    )?;
    Ok(table)
}

fn ensure_raw_columns(table: &mut Table, raw_ids: &[String]) -> Result<()> {
    for raw_id in raw_ids {
        if !table.columns.contains_key(raw_id) {
            table.insert(raw_id.clone(), ColumnData::F64(vec![None; table.len]))?;
        }
    }
    Ok(())
}

fn string_list_json(values: &[String]) -> String {
    let items = values
        .iter()
        .map(|value| format!("\"{}\"", escape_json(value)))
        .collect::<Vec<_>>();
    format!("[{}]", items.join(","))
}

fn dependencies_json(spec: &FactorSpec) -> String {
    let mut items = spec
        .dependencies
        .iter()
        .map(|dependency| {
            format!(
                "{{\"dataset\":\"{}\",\"columns\":{}}}",
                dependency.dataset.as_str(),
                string_list_json(&dependency.columns)
            )
        })
        .collect::<Vec<_>>();
    items.extend(spec.intraday_raw_dependencies.iter().map(|dependency| {
        format!(
            "{{\"intraday_raw\":\"{}\",\"daily_lookback\":{}}}",
            escape_json(&dependency.raw_id),
            dependency.daily_lookback
        )
    }));
    format!("[{}]", items.join(","))
}

fn escape_json(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn unix_timestamp_string() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs().to_string())
        .unwrap_or_else(|_| "0".to_string())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::core::{
        AssetClass, DatasetId, FactorRowKey, FactorValue, IntradayDailyRawSeries,
        IntradayDailyRawSpec,
    };

    use super::IntradayDailyRawStorage;

    fn temp_factor_root() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "yq_factor_engine_raw_cache_test_{}_{}",
            std::process::id(),
            nanos
        ))
    }

    fn raw_spec(version: &str) -> IntradayDailyRawSpec {
        IntradayDailyRawSpec {
            raw_id: "raw_a".to_string(),
            version: version.to_string(),
            asset_class: AssetClass::Stock,
            source_dataset: DatasetId::StockMinute1m,
            columns: vec!["close".to_string(), "vol".to_string()],
            window_days: 1,
        }
    }

    #[test]
    fn raw_cache_missing_dates_respects_metadata_version_and_existing_columns() {
        let root = temp_factor_root();
        let storage = IntradayDailyRawStorage::new(root.clone());
        let spec = raw_spec("0.1.0");

        assert_eq!(
            storage
                .missing_dates(&spec, &[20260105, 20260106], false)
                .expect("missing"),
            vec![20260105, 20260106]
        );

        storage
            .write_results(&[IntradayDailyRawSeries {
                spec: spec.clone(),
                values: vec![FactorValue {
                    key: FactorRowKey::Daily {
                        trade_date: 20260105,
                        ts_code: "000001.SZ".to_string(),
                    },
                    value: Some(1.0),
                }],
            }])
            .expect("write raw");
        assert!(root
            .join("_cache")
            .join("intraday_daily")
            .join("chn_stock")
            .join("2026")
            .join("20260105.parquet")
            .exists());
        storage.write_metadata(&[spec.clone()]).expect("metadata");

        assert_eq!(
            storage
                .missing_dates(&spec, &[20260105, 20260106], false)
                .expect("missing"),
            vec![20260106]
        );
        assert_eq!(
            storage
                .missing_dates(&raw_spec("0.2.0"), &[20260105, 20260106], false)
                .expect("missing"),
            vec![20260105, 20260106]
        );

        let _ = std::fs::remove_dir_all(root);
    }
}
