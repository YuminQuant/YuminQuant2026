use std::collections::{BTreeSet, HashMap};

use crate::core::{AssetClass, DatasetId};
use crate::data::catalog::DataCatalog;
use crate::data::parquet_io::read_parquet;
use crate::data::table::Table;
use crate::error::{err, Result};

#[derive(Clone, Debug)]
pub struct MarketDataLoader {
    catalog: DataCatalog,
}

impl MarketDataLoader {
    pub fn new(catalog: DataCatalog) -> Self {
        Self { catalog }
    }

    pub fn load_daily(
        &self,
        dataset: DatasetId,
        requested_columns: &[String],
        start_date: i32,
        end_date: i32,
    ) -> Result<Table> {
        let columns = with_required_columns(requested_columns, &["trade_date", "ts_code"]);
        let files = self.catalog.daily_year_files(dataset, start_date, end_date);
        let mut table = Table::empty();
        for file in files {
            let yearly = read_parquet(&file, Some(&columns))?;
            let filtered = yearly.filter_i32_range("trade_date", start_date, end_date)?;
            table.append(&filtered)?;
        }
        Ok(table)
    }

    pub fn load_index_daily(
        &self,
        ts_code: &str,
        requested_columns: &[String],
        start_date: i32,
        end_date: i32,
    ) -> Result<Table> {
        let columns = with_required_columns(requested_columns, &["trade_date", "ts_code"]);
        let files = self
            .catalog
            .index_daily_year_files(ts_code, start_date, end_date);
        let mut table = Table::empty();
        for file in files {
            let yearly = read_parquet(&file, Some(&columns))?;
            let filtered = yearly.filter_i32_range("trade_date", start_date, end_date)?;
            table.append(&filtered)?;
        }
        Ok(table)
    }

    pub fn load_stock_sw_classification(
        &self,
        requested_columns: &[String],
        start_date: i32,
        end_date: i32,
    ) -> Result<Table> {
        let path = self.catalog.stock_sw_classification_file().ok_or_else(|| {
            err("stock SW classification path is not configured; set stock_sw_classification_path in config.toml")
        })?;
        self.load_classification(path, requested_columns, start_date, end_date)
    }

    pub fn load_stock_ci_classification(
        &self,
        requested_columns: &[String],
        start_date: i32,
        end_date: i32,
    ) -> Result<Table> {
        let path = self.catalog.stock_ci_classification_file().ok_or_else(|| {
            err("stock CI classification path is not configured; set stock_ci_classification_path in config.toml")
        })?;
        self.load_classification(path, requested_columns, start_date, end_date)
    }

    fn load_classification(
        &self,
        path: &std::path::Path,
        requested_columns: &[String],
        start_date: i32,
        end_date: i32,
    ) -> Result<Table> {
        let columns = with_required_columns(requested_columns, &["ts_code", "in_date", "out_date"]);
        let table = read_parquet(path, Some(&columns))?;
        filter_classification_range(&table, start_date, end_date)
    }

    pub fn load_financial(
        &self,
        dataset: DatasetId,
        requested_columns: &[String],
        start_date: i32,
        end_date: i32,
        quarters: usize,
    ) -> Result<Table> {
        let columns = with_required_columns(
            requested_columns,
            &[
                "ts_code",
                "ann_date",
                "f_ann_date",
                "end_date",
                "report_type",
                "update_flag",
            ],
        );
        let start_year = start_date / 10_000;
        let end_year = end_date / 10_000;
        let lookback_years = financial_lookback_years(quarters);
        let file_start_year = start_year.saturating_sub(lookback_years);
        let files = self.catalog.daily_year_files(
            dataset,
            file_start_year * 10_000 + 101,
            end_year * 10_000 + 12_31,
        );
        let mut table = Table::empty();
        for file in files {
            let yearly = read_parquet(&file, Some(&columns))?;
            let filtered = filter_financial_disclosure_range(&yearly, end_date)?;
            table.append(&filtered)?;
        }
        Ok(table)
    }

    pub fn load_minute_by_date(
        &self,
        dataset: DatasetId,
        requested_columns: &[String],
        target_dates: &[i32],
    ) -> Result<HashMap<i32, Table>> {
        let columns = match dataset {
            DatasetId::StockMinute1m => {
                with_required_columns(requested_columns, &["trade_time", "ts_code"])
            }
            DatasetId::FutureMinute1m => {
                with_required_columns(requested_columns, &["trade_date", "trade_time", "ts_code"])
            }
            _ => with_required_columns(requested_columns, &["trade_time", "ts_code"]),
        };
        let mut tables = HashMap::new();
        for trade_date in target_dates {
            if let Some(file) = self.catalog.minute_file(dataset, *trade_date) {
                let mut table = read_parquet(&file, Some(&columns))?;
                if !table.columns.contains_key("trade_date") {
                    table.insert(
                        "trade_date",
                        crate::data::table::ColumnData::I32(vec![Some(*trade_date); table.len]),
                    )?;
                }
                tables.insert(*trade_date, table);
            }
        }
        Ok(tables)
    }

    pub fn load_barra_daily(
        &self,
        asset_class: AssetClass,
        model: &str,
        requested_columns: &[String],
        target_dates: &[i32],
    ) -> Result<Table> {
        let columns = with_required_columns(requested_columns, &["trade_date", "ts_code"]);
        let mut table = Table::empty();
        for trade_date in target_dates {
            if let Some(file) = self
                .catalog
                .barra_daily_file(asset_class, model, *trade_date)
            {
                let daily = read_parquet(&file, Some(&columns))?;
                table.append(&daily)?;
            }
        }
        Ok(table)
    }
}

fn filter_classification_range(table: &Table, start_date: i32, end_date: i32) -> Result<Table> {
    let in_dates = table.required_i32_date_cast("in_date")?;
    let out_dates = table.required_i32_date_cast("out_date")?;
    let indices = (0..table.len)
        .filter(|idx| {
            let Some(in_date) = in_dates[*idx] else {
                return false;
            };
            let out_date = out_dates[*idx].unwrap_or(99_991_231);
            in_date <= end_date && out_date >= start_date
        })
        .collect::<Vec<_>>();
    table.take(&indices)
}

fn financial_lookback_years(quarters: usize) -> i32 {
    if quarters == 0 {
        return 0;
    }
    (quarters.div_ceil(4) as i32) + 3
}

fn filter_financial_disclosure_range(table: &Table, end_date: i32) -> Result<Table> {
    let ann_dates = table.required_i32_date_cast("ann_date")?;
    let f_ann_dates = table.required_i32_date_cast("f_ann_date")?;
    let indices = (0..table.len)
        .filter(|idx| {
            let disclosure_date = f_ann_dates[*idx].or(ann_dates[*idx]);
            disclosure_date.is_some_and(|date| date <= end_date)
        })
        .collect::<Vec<_>>();
    table.take(&indices)
}

fn with_required_columns(requested: &[String], required: &[&str]) -> Vec<String> {
    let mut columns = BTreeSet::new();
    for column in required {
        columns.insert((*column).to_string());
    }
    for column in requested {
        columns.insert(column.clone());
    }
    columns.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::data::parquet_io::write_parquet;
    use crate::data::table::ColumnData;

    use super::*;

    #[test]
    fn index_daily_loader_reads_code_directory_and_filters_dates() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("yq_index_loader_test_{unique}"));
        let path = root
            .join("index_data")
            .join("daily")
            .join("000300_SH")
            .join("2026.parquet");
        let table = Table::new(BTreeMap::from([
            (
                "trade_date".to_string(),
                ColumnData::I32(vec![Some(20260102), Some(20260103)]),
            ),
            (
                "ts_code".to_string(),
                ColumnData::Utf8(vec![
                    Some("000300.SH".to_string()),
                    Some("000300.SH".to_string()),
                ]),
            ),
            (
                "close".to_string(),
                ColumnData::F32(vec![Some(10.0), Some(11.0)]),
            ),
            (
                "pre_close".to_string(),
                ColumnData::F32(vec![Some(9.5), Some(10.0)]),
            ),
        ]))
        .expect("table");
        write_parquet(&path, &table).expect("write parquet");

        let loader = MarketDataLoader::new(DataCatalog::new(root.clone()));
        let loaded = loader
            .load_index_daily("000300.SH", &["close".to_string()], 20260103, 20260103)
            .expect("load index daily");

        assert_eq!(loaded.len, 1);
        assert_eq!(
            loaded.required_i32("trade_date").expect("trade_date"),
            &vec![Some(20260103)]
        );
        assert_eq!(
            loaded.required_utf8("ts_code").expect("ts_code"),
            &vec![Some("000300.SH".to_string())]
        );
        assert_eq!(
            loaded.required_f64_cast("close").expect("close"),
            vec![Some(11.0)]
        );
        assert!(!loaded.columns.contains_key("pre_close"));

        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn ci_classification_loader_reads_configured_member_file_and_filters_active_rows() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("yq_ci_loader_test_{unique}"));
        let path = root
            .join("index_data")
            .join("member_ci")
            .join("ci_members.parquet");
        let table = Table::new(BTreeMap::from([
            (
                "ts_code".to_string(),
                ColumnData::Utf8(vec![
                    Some("000001.SZ".to_string()),
                    Some("000002.SZ".to_string()),
                    Some("000003.SZ".to_string()),
                ]),
            ),
            (
                "in_date".to_string(),
                ColumnData::Utf8(vec![
                    Some("20200101".to_string()),
                    Some("20270101".to_string()),
                    Some("20200101".to_string()),
                ]),
            ),
            (
                "out_date".to_string(),
                ColumnData::Utf8(vec![
                    Some("nan".to_string()),
                    Some("99991231".to_string()),
                    Some("20250101".to_string()),
                ]),
            ),
            (
                "l1_code".to_string(),
                ColumnData::Utf8(vec![
                    Some("CI001".to_string()),
                    Some("CI002".to_string()),
                    Some("CI003".to_string()),
                ]),
            ),
        ]))
        .expect("table");
        write_parquet(&path, &table).expect("write parquet");

        let catalog =
            DataCatalog::new(root.clone()).with_stock_ci_classification_path(path.clone());
        let loader = MarketDataLoader::new(catalog);
        let loaded = loader
            .load_stock_ci_classification(&["l1_code".to_string()], 20260424, 20260424)
            .expect("load ci classification");

        assert_eq!(loaded.len, 1);
        assert_eq!(
            loaded.required_utf8("ts_code").expect("ts_code"),
            &vec![Some("000001.SZ".to_string())]
        );
        assert_eq!(
            loaded.required_utf8("l1_code").expect("l1_code"),
            &vec![Some("CI001".to_string())]
        );

        fs::remove_dir_all(root).expect("cleanup");
    }
}
