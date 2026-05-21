use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::backtest::preprocess::maybe_neutralize;
use crate::backtest::request::NeutralizeSpec;
use crate::barra::engine::DEFAULT_BARRA_MODEL;
use crate::config::EngineConfig;
use crate::core::{AssetClass, Frequency};
use crate::data::parquet_io::read_parquet;
use crate::data::table::{ColumnData, Table};
use crate::error::{err, Result};
use crate::factor::common::{ClassificationLevel, ClassificationMap};

pub const CNE6_PRIMARY_BARRA_COLUMNS: [&str; 9] = [
    "DIVIDEND_YIELD",
    "GROWTH",
    "LIQUIDITY",
    "MOMENTUM",
    "QUALITY",
    "SENTIMENT",
    "SIZE",
    "VALUE",
    "VOLATILITY",
];

#[derive(Clone, Debug)]
pub struct NeutralizeDailyValuesRequest {
    pub trade_dates: Vec<i32>,
    pub ts_codes: Vec<String>,
    pub scores: Vec<Option<f64>>,
    pub neutralize: NeutralizeSpec,
    pub asset_class: AssetClass,
    pub frequency: Frequency,
    pub start_date: i32,
    pub end_date: i32,
    pub project_config_path: Option<PathBuf>,
    pub sector: Option<Vec<Option<String>>>,
    pub barra: BTreeMap<String, Vec<Option<f64>>>,
}

pub fn neutralize_barra_columns(spec: &NeutralizeSpec) -> Vec<String> {
    let mut columns = spec.barra_columns();
    if spec.uses_all_barra() {
        columns.extend(
            CNE6_PRIMARY_BARRA_COLUMNS
                .iter()
                .map(|value| value.to_string()),
        );
    }
    columns.sort();
    columns.dedup();
    columns
}

pub fn neutralize_daily_values(request: &NeutralizeDailyValuesRequest) -> Result<Vec<Option<f64>>> {
    validate_request(request)?;
    if matches!(request.neutralize, NeutralizeSpec::None) {
        return Ok(request.scores.clone());
    }

    let barra_columns = neutralize_barra_columns(&request.neutralize);
    let needs_sector_from_config = request.neutralize.uses_sector() && request.sector.is_none();
    let missing_barra_columns = barra_columns
        .iter()
        .filter(|column| !request.barra.contains_key(*column))
        .cloned()
        .collect::<Vec<_>>();
    let config = if needs_sector_from_config || !missing_barra_columns.is_empty() {
        Some(EngineConfig::discover(request.project_config_path.clone())?)
    } else {
        None
    };
    let sector_map = if needs_sector_from_config {
        Some(load_sector_map(config.as_ref().expect("config loaded"))?)
    } else {
        None
    };

    let by_date = group_indices_by_date(&request.trade_dates);
    let mut output = vec![None; request.scores.len()];
    for (trade_date, indices) in by_date {
        if trade_date < request.start_date || trade_date > request.end_date {
            return Err(err(format!(
                "trade_date {trade_date} is outside neutralization range {}..{}",
                request.start_date, request.end_date
            )));
        }
        let scores = take_option_f64(&request.scores, &indices);
        let ts_codes = take_strings(&request.ts_codes, &indices);
        let groups = if request.neutralize.uses_sector() {
            Some(match (&request.sector, &sector_map) {
                (Some(sector), _) => take_option_string(sector, &indices),
                (None, Some(map)) => map.groups_for(trade_date, &ts_codes),
                (None, None) => unreachable!("sector source validated"),
            })
        } else {
            None
        };
        let barra = if barra_columns.is_empty() {
            Vec::new()
        } else {
            daily_barra_columns(
                request,
                config.as_ref(),
                trade_date,
                &ts_codes,
                &indices,
                &barra_columns,
            )?
        };
        let neutralized = maybe_neutralize(&scores, &request.neutralize, &barra, groups.as_deref());
        for (local_idx, source_idx) in indices.iter().copied().enumerate() {
            output[source_idx] = neutralized[local_idx];
        }
    }
    Ok(output)
}

fn validate_request(request: &NeutralizeDailyValuesRequest) -> Result<()> {
    if request.asset_class != AssetClass::Stock {
        return Err(err("neutralize_daily currently supports asset=stock only"));
    }
    if request.frequency != Frequency::Daily {
        return Err(err(
            "neutralize_daily currently supports frequency=daily only",
        ));
    }
    if request.start_date > request.end_date {
        return Err(err("neutralize start_date must be <= end_date"));
    }
    let len = request.trade_dates.len();
    if request.ts_codes.len() != len || request.scores.len() != len {
        return Err(err(
            "trade_date, ts_code, and score must have the same length",
        ));
    }
    if let Some(sector) = &request.sector {
        if sector.len() != len {
            return Err(err("sector vector must have the same length as score"));
        }
    }
    for (column, values) in &request.barra {
        if values.len() != len {
            return Err(err(format!(
                "barra column {column} vector must have the same length as score"
            )));
        }
    }
    Ok(())
}

fn group_indices_by_date(dates: &[i32]) -> BTreeMap<i32, Vec<usize>> {
    let mut grouped = BTreeMap::<i32, Vec<usize>>::new();
    for (idx, date) in dates.iter().copied().enumerate() {
        grouped.entry(date).or_default().push(idx);
    }
    grouped
}

fn take_strings(values: &[String], indices: &[usize]) -> Vec<String> {
    indices.iter().map(|idx| values[*idx].clone()).collect()
}

fn take_option_string(values: &[Option<String>], indices: &[usize]) -> Vec<Option<String>> {
    indices.iter().map(|idx| values[*idx].clone()).collect()
}

fn take_option_f64(values: &[Option<f64>], indices: &[usize]) -> Vec<Option<f64>> {
    indices.iter().map(|idx| values[*idx]).collect()
}

fn daily_barra_columns(
    request: &NeutralizeDailyValuesRequest,
    config: Option<&EngineConfig>,
    trade_date: i32,
    ts_codes: &[String],
    indices: &[usize],
    columns: &[String],
) -> Result<Vec<Vec<Option<f64>>>> {
    let missing = columns
        .iter()
        .filter(|column| !request.barra.contains_key(*column))
        .cloned()
        .collect::<Vec<_>>();
    let loaded = if missing.is_empty() {
        BTreeMap::new()
    } else {
        let config = config.ok_or_else(|| err("missing EngineConfig for Barra exposure load"))?;
        load_daily_barra_from_storage(
            config,
            request.asset_class,
            request.frequency,
            trade_date,
            ts_codes,
            &missing,
        )?
    };
    columns
        .iter()
        .map(|column| {
            if let Some(values) = request.barra.get(column) {
                Ok(take_option_f64(values, indices))
            } else {
                loaded
                    .get(column)
                    .cloned()
                    .ok_or_else(|| err(format!("missing Barra exposure column {column}")))
            }
        })
        .collect()
}

fn load_sector_map(config: &EngineConfig) -> Result<ClassificationMap> {
    let table = read_parquet(
        &config.stock_sw_classification_path,
        Some(&[
            "ts_code".to_string(),
            "in_date".to_string(),
            "out_date".to_string(),
            "l1_code".to_string(),
        ]),
    )?;
    ClassificationMap::from_table(&table, ClassificationLevel::Sector)
}

fn load_daily_barra_from_storage(
    config: &EngineConfig,
    asset_class: AssetClass,
    frequency: Frequency,
    trade_date: i32,
    ts_codes: &[String],
    columns: &[String],
) -> Result<BTreeMap<String, Vec<Option<f64>>>> {
    let path = barra_daily_path(&config.barra_root, asset_class, frequency, trade_date);
    if !path.exists() {
        return Err(err(format!(
            "Barra exposure file not found for neutralization date {trade_date}: {}",
            path.display()
        )));
    }
    let mut requested = vec!["ts_code".to_string()];
    requested.extend(columns.iter().cloned());
    let table = read_parquet(&path, Some(&requested))?;
    table_barra_by_ts_code(&table, ts_codes, columns)
}

fn barra_daily_path(
    barra_root: &Path,
    asset_class: AssetClass,
    frequency: Frequency,
    trade_date: i32,
) -> PathBuf {
    let year = trade_date / 10_000;
    barra_root
        .join(asset_class.as_str())
        .join(frequency.as_str())
        .join(DEFAULT_BARRA_MODEL)
        .join(year.to_string())
        .join(format!("{trade_date}.parquet"))
}

fn table_barra_by_ts_code(
    table: &Table,
    ts_codes: &[String],
    columns: &[String],
) -> Result<BTreeMap<String, Vec<Option<f64>>>> {
    let table_ts_codes = table.required_utf8("ts_code")?;
    let row_by_code = table_ts_codes
        .iter()
        .enumerate()
        .filter_map(|(idx, code)| code.as_ref().map(|code| (code.clone(), idx)))
        .collect::<BTreeMap<_, _>>();
    let mut output = BTreeMap::new();
    for column in columns {
        let table_values = optional_f64_column(table, column)?;
        let values = ts_codes
            .iter()
            .map(|ts_code| row_by_code.get(ts_code).and_then(|idx| table_values[*idx]))
            .collect::<Vec<_>>();
        output.insert(column.clone(), values);
    }
    Ok(output)
}

fn optional_f64_column(table: &Table, name: &str) -> Result<Vec<Option<f64>>> {
    match table.columns.get(name) {
        Some(ColumnData::F64(values)) => Ok(values.clone()),
        Some(ColumnData::F32(values)) => {
            Ok(values.iter().map(|value| value.map(f64::from)).collect())
        }
        Some(ColumnData::I64(values)) => {
            Ok(values.iter().map(|value| value.map(|v| v as f64)).collect())
        }
        Some(ColumnData::I32(values)) => {
            Ok(values.iter().map(|value| value.map(f64::from)).collect())
        }
        Some(_) => Err(err(format!(
            "Barra exposure column {name} cannot be cast to float64"
        ))),
        None => Ok(vec![None; table.len]),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::backtest::request::NeutralizeSpec;
    use crate::core::{AssetClass, Frequency};

    use super::{neutralize_barra_columns, neutralize_daily_values, NeutralizeDailyValuesRequest};

    #[test]
    fn sector_neutralization_demeans_within_sector() {
        let request = NeutralizeDailyValuesRequest {
            trade_dates: vec![20260105, 20260105, 20260105, 20260105],
            ts_codes: vec!["a".into(), "b".into(), "c".into(), "d".into()],
            scores: vec![Some(1.0), Some(3.0), Some(10.0), Some(14.0)],
            neutralize: NeutralizeSpec::Sector,
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: 20260101,
            end_date: 20260131,
            project_config_path: None,
            sector: Some(vec![
                Some("s1".into()),
                Some("s1".into()),
                Some("s2".into()),
                Some("s2".into()),
            ]),
            barra: BTreeMap::new(),
        };

        let output = neutralize_daily_values(&request).expect("neutralized");
        assert_eq!(output, vec![Some(-1.0), Some(1.0), Some(-2.0), Some(2.0)]);
    }

    #[test]
    fn barra_size_sector_matches_grouped_regression_behavior() {
        let request = NeutralizeDailyValuesRequest {
            trade_dates: vec![20260105, 20260105, 20260105, 20260105],
            ts_codes: vec!["a".into(), "b".into(), "c".into(), "d".into()],
            scores: vec![Some(1.0), Some(2.0), Some(10.0), Some(11.0)],
            neutralize: NeutralizeSpec::parse("barra:SIZE+sector").expect("spec"),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: 20260101,
            end_date: 20260131,
            project_config_path: None,
            sector: Some(vec![
                Some("s1".into()),
                Some("s1".into()),
                Some("s2".into()),
                Some("s2".into()),
            ]),
            barra: BTreeMap::from([(
                "SIZE".to_string(),
                vec![Some(1.0), Some(2.0), Some(1.0), Some(2.0)],
            )]),
        };

        let output = neutralize_daily_values(&request).expect("neutralized");
        for value in output {
            assert!(value.expect("value").abs() < 1e-10);
        }
    }

    #[test]
    fn barra_all_expands_primary_cne6_columns() {
        let columns =
            neutralize_barra_columns(&NeutralizeSpec::parse("barra:all+sector").expect("spec"));
        assert_eq!(columns.len(), 9);
        assert!(columns.contains(&"DIVIDEND_YIELD".to_string()));
        assert!(columns.contains(&"SIZE".to_string()));
        assert!(columns.contains(&"VOLATILITY".to_string()));
    }

    #[test]
    fn output_keeps_input_order_across_dates() {
        let request = NeutralizeDailyValuesRequest {
            trade_dates: vec![20260106, 20260105, 20260106, 20260105],
            ts_codes: vec!["a".into(), "b".into(), "c".into(), "d".into()],
            scores: vec![Some(4.0), Some(1.0), Some(6.0), Some(3.0)],
            neutralize: NeutralizeSpec::Sector,
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: 20260101,
            end_date: 20260131,
            project_config_path: None,
            sector: Some(vec![
                Some("s".into()),
                Some("s".into()),
                Some("s".into()),
                Some("s".into()),
            ]),
            barra: BTreeMap::new(),
        };

        let output = neutralize_daily_values(&request).expect("neutralized");
        assert_eq!(output, vec![Some(-1.0), Some(-1.0), Some(1.0), Some(1.0)]);
    }
}
