use std::collections::{BTreeSet, HashMap};

use crate::core::DatasetId;
use crate::data::catalog::DataCatalog;
use crate::data::parquet_io::read_parquet;
use crate::data::table::Table;
use crate::error::Result;

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
