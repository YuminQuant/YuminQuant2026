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

    pub fn load_stock_sw_classification(
        &self,
        requested_columns: &[String],
        start_date: i32,
        end_date: i32,
    ) -> Result<Table> {
        let path = self.catalog.stock_sw_classification_file().ok_or_else(|| {
            err("stock SW classification path is not configured; set stock_sw_classification_path in config.toml")
        })?;
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
