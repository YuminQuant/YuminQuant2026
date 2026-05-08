use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use crate::backtest::request::{BacktestRunRequest, DEFAULT_DECAY_HORIZON};
use crate::barra::engine::DEFAULT_BARRA_MODEL;
use crate::calendar::TradingCalendar;
use crate::config::EngineConfig;
use crate::core::{AssetClass, Frequency};
use crate::data::parquet_io::read_parquet;
use crate::data::{ColumnData, Table};
use crate::error::{err, Result};
use crate::factor::common::{ClassificationLevel, ClassificationMap};
use crate::storage::{FactorMetadata, FactorStorage, LabelStorage};

#[derive(Clone, Debug)]
pub struct BacktestInput {
    pub factor_metadata: Vec<FactorMetadata>,
    pub label_metadata: LabelMetadataInfo,
    pub target_dates: Vec<i32>,
    pub all_dates: Vec<i32>,
    pub panel: BacktestPanel,
    pub sectors: Option<HashMap<i32, Vec<Option<String>>>>,
}

#[derive(Clone, Debug)]
pub struct LabelMetadataInfo {
    pub label_id: String,
    pub output_column: String,
    pub lookahead: usize,
}

#[derive(Clone, Debug)]
pub struct BacktestPanel {
    dates: Vec<i32>,
    instruments: Vec<String>,
    date_lookup: BTreeMap<i32, usize>,
    columns: BTreeMap<String, Vec<Option<f64>>>,
}

impl BacktestPanel {
    pub fn dates(&self) -> &[i32] {
        &self.dates
    }

    pub fn instruments(&self) -> &[String] {
        &self.instruments
    }

    pub fn date_index(&self, date: i32) -> Option<usize> {
        self.date_lookup.get(&date).copied()
    }

    pub fn column(&self, name: &str) -> Result<&[Option<f64>]> {
        self.columns
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| err(format!("backtest panel missing column {name}")))
    }

    pub fn cross_section(&self, name: &str, date_idx: usize) -> Result<Vec<Option<f64>>> {
        let column = self.column(name)?;
        let start = date_idx * self.instruments.len();
        let end = start + self.instruments.len();
        Ok(column[start..end].to_vec())
    }

    fn ensure_columns(&mut self, names: &[String]) {
        let shape_len = self.dates.len() * self.instruments.len();
        for name in names {
            self.columns
                .entry(name.clone())
                .or_insert_with(|| vec![None; shape_len]);
        }
    }
}

pub fn load_backtest_input(
    config: &EngineConfig,
    request: &BacktestRunRequest,
) -> Result<BacktestInput> {
    if request.asset_class != AssetClass::Stock || request.frequency != Frequency::Daily {
        return Err(err(
            "backtest v1 only supports --asset stock --frequency daily",
        ));
    }

    let calendar = TradingCalendar::load(&config.data_root, &config.stock_calendar_exchange)?;
    let target_dates = calendar.open_dates_between(request.start_date, request.end_date);
    if target_dates.is_empty() {
        return Err(err("no trading dates in requested backtest range"));
    }
    let label_end = target_dates
        .last()
        .and_then(|date| calendar.open_date_after(*date, DEFAULT_DECAY_HORIZON))
        .unwrap_or(*target_dates.last().expect("non-empty target dates"));
    let all_dates = calendar.open_dates_between(request.start_date, label_end);

    let factor_metadata = select_factors(config, request)?;
    if factor_metadata.is_empty() {
        return Err(err("no factors selected for backtest"));
    }
    let label_metadata = select_label(config, &request.label_id)?;

    let factor_columns = factor_metadata
        .iter()
        .map(|row| row.output_column.clone())
        .collect::<Vec<_>>();
    let label_columns = vec![label_metadata.output_column.clone()];
    let factor_table = load_output_table(
        &config.factor_root,
        request.asset_class,
        request.frequency,
        DEFAULT_BARRA_MODEL,
        false,
        &target_dates,
        &factor_columns,
    )?;
    let label_table = load_output_table(
        &config.label_root,
        request.asset_class,
        request.frequency,
        DEFAULT_BARRA_MODEL,
        false,
        &all_dates,
        &label_columns,
    )?;

    let mut tables = vec![factor_table, label_table];
    let barra_columns = request.neutralize.barra_columns();
    if !barra_columns.is_empty() {
        tables.push(load_output_table(
            &config.barra_root,
            request.asset_class,
            request.frequency,
            DEFAULT_BARRA_MODEL,
            true,
            &target_dates,
            &barra_columns,
        )?);
    }
    let mut panel = BacktestPanel::from_tables(all_dates.clone(), tables)?;
    panel.ensure_columns(&factor_columns);
    panel.ensure_columns(&label_columns);
    panel.ensure_columns(&barra_columns);

    let sectors = if request.neutralize.uses_industry() {
        let table = read_parquet(
            &config.stock_sw_classification_path,
            Some(&[
                "ts_code".to_string(),
                "in_date".to_string(),
                "out_date".to_string(),
                "l1_code".to_string(),
            ]),
        )?;
        let sector_map = ClassificationMap::from_table(&table, ClassificationLevel::Sector)?;
        let mut by_date = HashMap::new();
        for date in &target_dates {
            by_date.insert(*date, sector_map.groups_for(*date, panel.instruments()));
        }
        Some(by_date)
    } else {
        None
    };

    Ok(BacktestInput {
        factor_metadata,
        label_metadata,
        target_dates,
        all_dates,
        panel,
        sectors,
    })
}

impl BacktestPanel {
    fn from_tables(dates: Vec<i32>, tables: Vec<Table>) -> Result<Self> {
        let mut instrument_set = BTreeSet::new();
        for table in &tables {
            if table.columns.is_empty() {
                continue;
            }
            let ts_codes = table.required_utf8("ts_code")?;
            for ts_code in ts_codes.iter().flatten() {
                instrument_set.insert(ts_code.clone());
            }
        }
        let instruments = instrument_set.into_iter().collect::<Vec<_>>();
        let date_lookup = dates
            .iter()
            .enumerate()
            .map(|(idx, date)| (*date, idx))
            .collect::<BTreeMap<_, _>>();
        let instrument_lookup = instruments
            .iter()
            .enumerate()
            .map(|(idx, code)| (code.clone(), idx))
            .collect::<BTreeMap<_, _>>();
        let shape_len = dates.len() * instruments.len();
        let mut columns = BTreeMap::<String, Vec<Option<f64>>>::new();

        for table in tables {
            if table.columns.is_empty() {
                continue;
            }
            let ts_codes = table.required_utf8("ts_code")?;
            let trade_dates = table.required_i32("trade_date")?;
            let numeric_columns = table
                .columns
                .keys()
                .filter(|name| name.as_str() != "trade_date" && name.as_str() != "ts_code")
                .cloned()
                .collect::<Vec<_>>();
            let numeric_values = numeric_columns
                .iter()
                .map(|name| Ok((name.clone(), table.required_f64_cast(name)?)))
                .collect::<Result<BTreeMap<_, _>>>()?;
            for name in &numeric_columns {
                columns
                    .entry(name.clone())
                    .or_insert_with(|| vec![None; shape_len]);
            }
            for row_idx in 0..table.len {
                let (Some(trade_date), Some(ts_code)) =
                    (trade_dates[row_idx], ts_codes[row_idx].clone())
                else {
                    continue;
                };
                let (Some(date_idx), Some(instrument_idx)) = (
                    date_lookup.get(&trade_date),
                    instrument_lookup.get(&ts_code),
                ) else {
                    continue;
                };
                let offset = date_idx * instruments.len() + instrument_idx;
                for name in &numeric_columns {
                    let values = &numeric_values[name];
                    if let Some(target) = columns.get_mut(name) {
                        target[offset] = values.get(row_idx).copied().unwrap_or(None);
                    }
                }
            }
        }

        Ok(Self {
            dates,
            instruments,
            date_lookup,
            columns,
        })
    }
}

fn select_factors(
    config: &EngineConfig,
    request: &BacktestRunRequest,
) -> Result<Vec<FactorMetadata>> {
    let storage = FactorStorage::new(config.factor_root.clone());
    let metadata = storage.read_metadata()?;
    let selected = match (&request.factor_ids, &request.tags, request.all_factors) {
        (Some(ids), None, false) => {
            let mut rows = Vec::new();
            for id in ids {
                let Some(row) = metadata.iter().find(|row| &row.factor_id == id) else {
                    return Err(err(format!("factor not found in metadata: {id}")));
                };
                if row.tags.iter().any(|tag| tag == "deprecated") {
                    return Err(err(format!(
                        "deprecated factor cannot be backtested explicitly: {id}"
                    )));
                }
                rows.push(row.clone());
            }
            rows
        }
        (None, Some(tags), false) => metadata
            .into_iter()
            .filter(|row| !row.tags.iter().any(|tag| tag == "deprecated"))
            .filter(|row| {
                tags.iter()
                    .all(|tag| row.tags.iter().any(|row_tag| row_tag == tag))
            })
            .collect(),
        (None, None, true) => metadata
            .into_iter()
            .filter(|row| !row.tags.iter().any(|tag| tag == "deprecated"))
            .collect(),
        (None, None, false) => {
            return Err(err("backtest requires --factors, --tags or --all-factors"));
        }
        _ => {
            return Err(err(
                "--factors, --tags and --all-factors cannot be used together",
            ));
        }
    };
    Ok(dedup_factor_metadata(selected))
}

fn dedup_factor_metadata(rows: Vec<FactorMetadata>) -> Vec<FactorMetadata> {
    let mut seen = BTreeSet::new();
    let mut output = Vec::new();
    for row in rows {
        if seen.insert(row.factor_id.clone()) {
            output.push(row);
        }
    }
    output
}

fn select_label(config: &EngineConfig, label_id: &str) -> Result<LabelMetadataInfo> {
    let storage = LabelStorage::new(config.label_root.clone());
    let metadata = storage.read_metadata()?;
    let row = metadata
        .iter()
        .find(|row| row.label_id == label_id)
        .ok_or_else(|| err(format!("label not found in metadata: {label_id}")))?;
    Ok(LabelMetadataInfo {
        label_id: row.label_id.clone(),
        output_column: row.output_column.clone(),
        lookahead: parse_lookahead(&row.dependencies_json).unwrap_or(2),
    })
}

fn parse_lookahead(dependencies_json: &str) -> Option<usize> {
    let marker = "\"lookahead_trading_days\":";
    let start = dependencies_json.find(marker)? + marker.len();
    let tail = &dependencies_json[start..];
    let digits = tail
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    digits.parse::<usize>().ok()
}

fn load_output_table(
    root: &Path,
    asset_class: AssetClass,
    frequency: Frequency,
    model: &str,
    barra_layout: bool,
    dates: &[i32],
    requested_columns: &[String],
) -> Result<Table> {
    let mut columns = vec!["trade_date".to_string(), "ts_code".to_string()];
    for column in requested_columns {
        if !columns.iter().any(|existing| existing == column) {
            columns.push(column.clone());
        }
    }
    let mut table = Table::empty();
    for date in dates {
        let path = output_path(root, asset_class, frequency, model, barra_layout, *date);
        if !path.exists() {
            continue;
        }
        let mut daily = read_parquet(&path, Some(&columns))?;
        ensure_table_columns(&mut daily, requested_columns)?;
        if table.columns.is_empty() {
            table = daily;
        } else {
            table.append(&daily)?;
        }
    }
    if table.columns.is_empty() {
        empty_output_table(&columns)
    } else {
        Ok(table)
    }
}

fn ensure_table_columns(table: &mut Table, requested_columns: &[String]) -> Result<()> {
    for column in requested_columns {
        if !table.columns.contains_key(column) {
            table.insert(column.clone(), ColumnData::F64(vec![None; table.len]))?;
        }
    }
    Ok(())
}

fn output_path(
    root: &Path,
    asset_class: AssetClass,
    frequency: Frequency,
    model: &str,
    barra_layout: bool,
    trade_date: i32,
) -> PathBuf {
    let year = trade_date / 10_000;
    if barra_layout {
        root.join(asset_class.as_str())
            .join(frequency.as_str())
            .join(model)
            .join(year.to_string())
            .join(format!("{trade_date}.parquet"))
    } else {
        root.join(asset_class.as_str())
            .join(frequency.as_str())
            .join(year.to_string())
            .join(format!("{trade_date}.parquet"))
    }
}

fn empty_output_table(columns: &[String]) -> Result<Table> {
    let mut data = BTreeMap::new();
    for column in columns {
        let values = if column == "ts_code" {
            ColumnData::Utf8(Vec::new())
        } else if column == "trade_date" {
            ColumnData::I32(Vec::new())
        } else {
            ColumnData::F64(Vec::new())
        };
        data.insert(column.clone(), values);
    }
    Table::new(data)
}

#[cfg(test)]
mod tests {
    use super::parse_lookahead;

    #[test]
    fn parses_label_lookahead_from_metadata_json() {
        assert_eq!(
            parse_lookahead(r#"[{"dataset":"x"},{"lookahead_trading_days":2}]"#),
            Some(2)
        );
    }
}
