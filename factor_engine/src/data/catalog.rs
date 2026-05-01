use std::path::{Path, PathBuf};

use crate::core::{AssetClass, DatasetId};

#[derive(Clone, Debug)]
pub struct DataCatalog {
    data_root: PathBuf,
    stock_sw_classification_path: Option<PathBuf>,
    stock_ci_classification_path: Option<PathBuf>,
}

impl DataCatalog {
    pub fn new(data_root: PathBuf) -> Self {
        Self {
            data_root,
            stock_sw_classification_path: None,
            stock_ci_classification_path: None,
        }
    }

    pub fn with_stock_sw_classification_path(mut self, path: PathBuf) -> Self {
        self.stock_sw_classification_path = Some(path);
        self
    }

    pub fn with_stock_ci_classification_path(mut self, path: PathBuf) -> Self {
        self.stock_ci_classification_path = Some(path);
        self
    }

    pub fn data_root(&self) -> &Path {
        &self.data_root
    }

    pub fn daily_year_files(
        &self,
        dataset: DatasetId,
        start_date: i32,
        end_date: i32,
    ) -> Vec<PathBuf> {
        let start_year = start_date / 10_000;
        let end_year = end_date / 10_000;
        let mut paths = Vec::new();
        for year in start_year..=end_year {
            let path = match dataset {
                DatasetId::StockDailyPv => self
                    .data_root
                    .join("stock_data")
                    .join("daily")
                    .join("pv")
                    .join(format!("{}.parquet", year)),
                DatasetId::StockDailyBasic => self
                    .data_root
                    .join("stock_data")
                    .join("daily")
                    .join("basic")
                    .join(format!("{}.parquet", year)),
                DatasetId::StockIncome => self
                    .data_root
                    .join("stock_data")
                    .join("income")
                    .join(format!("{}.parquet", year)),
                DatasetId::StockBalanceSheet => self
                    .data_root
                    .join("stock_data")
                    .join("balancesheet")
                    .join(format!("{}.parquet", year)),
                DatasetId::StockCashFlow => self
                    .data_root
                    .join("stock_data")
                    .join("cashflow")
                    .join(format!("{}.parquet", year)),
                DatasetId::StockDividend => self
                    .data_root
                    .join("stock_data")
                    .join("dividend")
                    .join(format!("{}.parquet", year)),
                DatasetId::StockAnalystReport => self
                    .data_root
                    .join("stock_data")
                    .join("analyst_report")
                    .join(format!("{}.parquet", year)),
                DatasetId::FutureDaily => self
                    .data_root
                    .join("future_data")
                    .join("daily")
                    .join(format!("{}.parquet", year)),
                _ => continue,
            };
            if path.exists() {
                paths.push(path);
            }
        }
        paths
    }

    pub fn daily_date_file(&self, dataset: DatasetId, trade_date: i32) -> Option<PathBuf> {
        let year = trade_date / 10_000;
        let path = match dataset {
            DatasetId::StockDailyPv => self
                .data_root
                .join("stock_data")
                .join("daily")
                .join("pv")
                .join(year.to_string())
                .join(format!("{}.parquet", trade_date)),
            DatasetId::StockDailyBasic => self
                .data_root
                .join("stock_data")
                .join("daily")
                .join("basic")
                .join(year.to_string())
                .join(format!("{}.parquet", trade_date)),
            DatasetId::FutureDaily => self
                .data_root
                .join("future_data")
                .join("daily")
                .join(year.to_string())
                .join(format!("{}.parquet", trade_date)),
            _ => return None,
        };
        path.exists().then_some(path)
    }

    pub fn index_daily_year_files(
        &self,
        ts_code: &str,
        start_date: i32,
        end_date: i32,
    ) -> Vec<PathBuf> {
        let start_year = start_date / 10_000;
        let end_year = end_date / 10_000;
        let code_dir = ts_code.replace('.', "_");
        let mut paths = Vec::new();
        for year in start_year..=end_year {
            let path = self
                .data_root
                .join("index_data")
                .join("daily")
                .join(&code_dir)
                .join(format!("{}.parquet", year));
            if path.exists() {
                paths.push(path);
            }
        }
        paths
    }

    pub fn index_daily_date_file(&self, ts_code: &str, trade_date: i32) -> Option<PathBuf> {
        let year = trade_date / 10_000;
        let code_dir = ts_code.replace('.', "_");
        let path = self
            .data_root
            .join("index_data")
            .join("daily")
            .join(&code_dir)
            .join(year.to_string())
            .join(format!("{}.parquet", trade_date));
        path.exists().then_some(path)
    }

    pub fn minute_file(&self, dataset: DatasetId, trade_date: i32) -> Option<PathBuf> {
        let year = trade_date / 10_000;
        let path = match dataset {
            DatasetId::StockMinute1m => self
                .data_root
                .join("stock_data")
                .join("minute")
                .join(year.to_string())
                .join(format!("{}.parquet", trade_date)),
            DatasetId::FutureMinute1m => self
                .data_root
                .join("future_data")
                .join("minute")
                .join(year.to_string())
                .join(format!("{}.parquet", trade_date)),
            _ => return None,
        };
        path.exists().then_some(path)
    }

    pub fn stock_sw_classification_file(&self) -> Option<&Path> {
        self.stock_sw_classification_path.as_deref()
    }

    pub fn stock_ci_classification_file(&self) -> Option<&Path> {
        self.stock_ci_classification_path.as_deref()
    }

    pub fn barra_daily_file(
        &self,
        asset_class: AssetClass,
        model: &str,
        trade_date: i32,
    ) -> Option<PathBuf> {
        let year = trade_date / 10_000;
        let path = self
            .data_root
            .join("barra")
            .join(asset_class.as_str())
            .join("daily")
            .join(model)
            .join(year.to_string())
            .join(format!("{}.parquet", trade_date));
        path.exists().then_some(path)
    }
}
