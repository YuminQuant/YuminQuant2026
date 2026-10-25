use crate::artifact_io::StoragePathExt;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::Arc;

use crate::core::{AssetClass, DatasetId};
use crate::data::catalog::DataCatalog;
use crate::data::parquet_io::read_parquet;
use crate::data::table::{ColumnData, Table};
use crate::error::{err, Result};
use crate::factor::common::FinancialPitIndex;
use crate::data::FundPortfolioIndex;

#[derive(Clone, Debug)]
pub struct MarketDataLoader {
    catalog: DataCatalog,
}

#[derive(Clone, Debug, Default)]
pub struct DisclosureTableCache {
    yearly_tables: HashMap<DisclosureCacheKey, DisclosureCacheEntry>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DisclosureCacheProfile {
    pub dataset: DatasetId,
    pub year: Option<i32>,
    pub loaded_columns: usize,
    pub row_count: usize,
}

impl DisclosureTableCache {
    pub fn len(&self) -> usize {
        self.yearly_tables.len()
    }

    pub fn profiles(&self) -> Vec<DisclosureCacheProfile> {
        let mut profiles = self
            .yearly_tables
            .iter()
            .map(|(key, entry)| DisclosureCacheProfile {
                dataset: key.dataset,
                year: file_stem_year(&key.file),
                loaded_columns: entry.loaded_columns.len(),
                row_count: entry.table.len,
            })
            .collect::<Vec<_>>();
        profiles.sort_by_key(|profile| (profile.dataset.as_str().to_string(), profile.year));
        profiles
    }

    pub fn retain_disclosure_windows(&mut self, windows: &HashMap<DatasetId, BTreeSet<i32>>) {
        self.yearly_tables.retain(|key, _| {
            if !is_cached_disclosure_dataset(key.dataset) {
                return true;
            }
            let Some(year) = file_stem_year(&key.file) else {
                return true;
            };
            windows
                .get(&key.dataset)
                .is_some_and(|years| years.contains(&year))
        });
    }

    pub fn retain_disclosure_windows_and_columns(
        &mut self,
        windows: &HashMap<DatasetId, BTreeSet<i32>>,
        columns: &HashMap<DatasetId, BTreeSet<String>>,
    ) -> Result<()> {
        self.yearly_tables.retain(|key, _| {
            if !is_cached_disclosure_dataset(key.dataset) {
                return true;
            }
            let Some(year) = file_stem_year(&key.file) else {
                return true;
            };
            windows
                .get(&key.dataset)
                .is_some_and(|years| years.contains(&year))
                && columns.contains_key(&key.dataset)
        });

        for (key, entry) in &mut self.yearly_tables {
            if !is_cached_disclosure_dataset(key.dataset) {
                continue;
            }
            let Some(needed_columns) = columns.get(&key.dataset) else {
                continue;
            };
            if entry.loaded_columns.is_subset(needed_columns) {
                continue;
            }
            let kept_columns = entry
                .loaded_columns
                .intersection(needed_columns)
                .cloned()
                .collect::<BTreeSet<_>>();
            let ordered_columns = kept_columns.iter().cloned().collect::<Vec<_>>();
            entry.table = Arc::new(project_table_columns(&entry.table, &ordered_columns)?);
            entry.loaded_columns = kept_columns;
            entry.pit_index = None;
            entry.fund_index = None;
        }
        Ok(())
    }

    fn ensure_year_loaded(
        &mut self,
        dataset: DatasetId,
        file: PathBuf,
        columns: &[String],
    ) -> Result<DisclosureCacheKey> {
        let key = DisclosureCacheKey { dataset, file };
        let requested = columns.iter().cloned().collect::<BTreeSet<_>>();
        let metadata = crate::artifact_io::metadata(&key.file)?;
        let file_stamp = (metadata.len(), metadata.modified()?);
        let should_reload = self.yearly_tables.get(&key).is_none_or(|entry| {
            !requested.is_subset(&entry.loaded_columns) || entry.file_stamp != file_stamp
        });
        if should_reload {
            let merged_columns = self
                .yearly_tables
                .get(&key)
                .map(|entry| {
                    entry
                        .loaded_columns
                        .union(&requested)
                        .cloned()
                        .collect::<Vec<_>>()
                })
                .unwrap_or_else(|| columns.to_vec());
            let table = read_parquet(&key.file, Some(&merged_columns))?;
            self.yearly_tables.insert(
                key.clone(),
                DisclosureCacheEntry {
                    table: Arc::new(table),
                    loaded_columns: merged_columns.into_iter().collect(),
                    file_stamp,
                    pit_index: None,
                    fund_index: None,
                },
            );
        }
        Ok(key)
    }

    fn load_year(
        &mut self,
        dataset: DatasetId,
        file: PathBuf,
        columns: &[String],
    ) -> Result<Table> {
        let key = self.ensure_year_loaded(dataset, file, columns)?;
        let entry = self
            .yearly_tables
            .get(&key)
            .ok_or_else(|| err("disclosure cache entry disappeared unexpectedly"))?;
        project_table_columns(&entry.table, columns)
    }

    fn load_year_source(
        &mut self,
        dataset: DatasetId,
        file: PathBuf,
        columns: &[String],
    ) -> Result<Arc<Table>> {
        let key = self.ensure_year_loaded(dataset, file, columns)?;
        self.yearly_tables
            .get(&key)
            .map(|entry| Arc::clone(&entry.table))
            .ok_or_else(|| err("disclosure cache entry disappeared unexpectedly"))
    }

    pub fn financial_index_for_source(
        &mut self,
        source: &Arc<Table>,
    ) -> Result<Arc<FinancialPitIndex>> {
        let entry = self
            .yearly_tables
            .values_mut()
            .find(|entry| Arc::ptr_eq(&entry.table, source))
            .ok_or_else(|| err("financial source is not owned by disclosure cache"))?;
        if entry.pit_index.is_none() {
            entry.pit_index = Some(Arc::new(FinancialPitIndex::from_table(Arc::clone(source))?));
        }
        Ok(Arc::clone(entry.pit_index.as_ref().unwrap()))
    }

    pub fn fund_index_for_source(&mut self, source: &Arc<Table>) -> Result<Arc<FundPortfolioIndex>> {
        let entry = self.yearly_tables.values_mut().find(|entry| Arc::ptr_eq(&entry.table, source))
            .ok_or_else(|| err("fund portfolio source is not owned by disclosure cache"))?;
        if entry.fund_index.is_none() {
            entry.fund_index = Some(Arc::new(FundPortfolioIndex::from_table(Arc::clone(source))?));
        }
        Ok(Arc::clone(entry.fund_index.as_ref().unwrap()))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct DisclosureCacheKey {
    dataset: DatasetId,
    file: PathBuf,
}

#[derive(Clone, Debug)]
struct DisclosureCacheEntry {
    table: Arc<Table>,
    loaded_columns: BTreeSet<String>,
    file_stamp: (u64, std::time::SystemTime),
    pit_index: Option<Arc<FinancialPitIndex>>,
    fund_index: Option<Arc<FundPortfolioIndex>>,
}

impl MarketDataLoader {
    pub fn new(catalog: DataCatalog) -> Self {
        Self { catalog }
    }

    pub fn load_fund_portfolio_sources_cached(
        &self, requested_columns: &[String], start_date: i32, end_date: i32,
        cache: &mut DisclosureTableCache,
    ) -> Result<Vec<Arc<Table>>> {
        let columns = with_required_columns(requested_columns, crate::data::FUND_PORTFOLIO_COLUMNS);
        self.catalog.daily_year_files(DatasetId::FundPortfolio, start_date, end_date)
            .into_iter().map(|file| cache.load_year_source(DatasetId::FundPortfolio, file, &columns)).collect()
    }

    pub fn fund_portfolio_history_files_before(&self, year: i32) -> Result<Vec<PathBuf>> {
        let directory = self.catalog.data_root().join("fund_data").join("portfolio");
        if !directory.storage_exists() { return Err(err(format!("fund portfolio directory not found: {}", directory.display()))); }
        let mut files = Vec::new();
        for entry in crate::artifact_io::read_dir(directory)? {
            let path = entry?.path();
            if path.extension().is_some_and(|extension| extension == "parquet")
                && file_stem_year(&path).is_some_and(|source_year| source_year < year)
            { files.push(path); }
        }
        files.sort();
        Ok(files)
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

    pub fn load_daily_by_dates(
        &self,
        dataset: DatasetId,
        requested_columns: &[String],
        trade_dates: &[i32],
    ) -> Result<Table> {
        let columns = with_required_columns(requested_columns, &["trade_date", "ts_code"]);
        let mut table = Table::empty();
        let mut yearly_cache = HashMap::<i32, Table>::new();

        for trade_date in trade_dates {
            if let Some(file) = self.catalog.daily_date_file(dataset, *trade_date) {
                append_file_filtered_by_dates(&mut table, file, &columns, &[*trade_date])?;
            } else {
                let year = trade_date / 10_000;
                if !yearly_cache.contains_key(&year) {
                    let start_date = year * 10_000 + 101;
                    let end_date = year * 10_000 + 12_31;
                    let mut yearly_table = Table::empty();
                    for file in self.catalog.daily_year_files(dataset, start_date, end_date) {
                        let yearly = read_parquet(&file, Some(&columns))?;
                        yearly_table.append(&yearly)?;
                    }
                    yearly_cache.insert(year, yearly_table);
                }
                if let Some(yearly_table) = yearly_cache.get(&year) {
                    append_table_filtered_by_dates(&mut table, yearly_table, &[*trade_date])?;
                }
            }
        }

        if table.columns.is_empty() {
            return empty_daily_keyed_table(&columns);
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

    pub fn load_index_daily_by_dates(
        &self,
        ts_code: &str,
        requested_columns: &[String],
        trade_dates: &[i32],
    ) -> Result<Table> {
        let columns = with_required_columns(requested_columns, &["trade_date", "ts_code"]);
        let mut table = Table::empty();
        let mut yearly_cache = HashMap::<i32, Table>::new();

        for trade_date in trade_dates {
            if let Some(file) = self.catalog.index_daily_date_file(ts_code, *trade_date) {
                append_file_filtered_by_dates(&mut table, file, &columns, &[*trade_date])?;
            } else {
                let year = trade_date / 10_000;
                if !yearly_cache.contains_key(&year) {
                    let start_date = year * 10_000 + 101;
                    let end_date = year * 10_000 + 12_31;
                    let mut yearly_table = Table::empty();
                    for file in self
                        .catalog
                        .index_daily_year_files(ts_code, start_date, end_date)
                    {
                        let yearly = read_parquet(&file, Some(&columns))?;
                        yearly_table.append(&yearly)?;
                    }
                    yearly_cache.insert(year, yearly_table);
                }
                if let Some(yearly_table) = yearly_cache.get(&year) {
                    append_table_filtered_by_dates(&mut table, yearly_table, &[*trade_date])?;
                }
            }
        }

        if table.columns.is_empty() {
            return empty_daily_keyed_table(&columns);
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

    pub fn load_stock_basic(&self, requested_columns: &[String]) -> Result<Table> {
        let columns = with_required_columns(requested_columns, &["ts_code"]);
        let Some(file) = self.catalog.stock_basic_file() else {
            return empty_static_keyed_table(&columns);
        };
        read_parquet(&file, Some(&columns))
    }

    pub fn load_fund_basic(&self, requested_columns: &[String]) -> Result<Table> {
        let columns = with_required_columns(requested_columns, &["ts_code"]);
        read_parquet(&self.catalog.fund_basic_file(), Some(&columns))
    }

    pub fn load_financial(
        &self,
        dataset: DatasetId,
        requested_columns: &[String],
        start_date: i32,
        end_date: i32,
        quarters: usize,
    ) -> Result<Table> {
        let mut cache = DisclosureTableCache::default();
        self.load_financial_cached(
            dataset,
            requested_columns,
            start_date,
            end_date,
            quarters,
            &mut cache,
        )
    }

    pub fn load_financial_cached(
        &self,
        dataset: DatasetId,
        requested_columns: &[String],
        start_date: i32,
        end_date: i32,
        quarters: usize,
        cache: &mut DisclosureTableCache,
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
        let years = financial_disclosure_years_for_range(start_date, end_date, quarters);
        let file_start_year = years.iter().next().copied().unwrap_or(end_date / 10_000);
        let end_year = years
            .iter()
            .next_back()
            .copied()
            .unwrap_or(end_date / 10_000);
        let files = self.catalog.daily_year_files(
            dataset,
            file_start_year * 10_000 + 101,
            end_year * 10_000 + 12_31,
        );
        let mut table = Table::empty();
        for file in files {
            let yearly = cache.load_year(dataset, file, &columns)?;
            let filtered = filter_financial_disclosure_range(&yearly, end_date)?;
            table.append(&filtered)?;
        }
        if table.columns.is_empty() {
            return empty_disclosure_table(&columns);
        }
        Ok(table)
    }

    pub fn load_financial_sources_cached(
        &self,
        dataset: DatasetId,
        requested_columns: &[String],
        start_date: i32,
        end_date: i32,
        quarters: usize,
        cache: &mut DisclosureTableCache,
    ) -> Result<Vec<Arc<Table>>> {
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
        let years = financial_disclosure_years_for_range(start_date, end_date, quarters);
        let file_start_year = years.iter().next().copied().unwrap_or(end_date / 10_000);
        let end_year = years
            .iter()
            .next_back()
            .copied()
            .unwrap_or(end_date / 10_000);
        let files = self.catalog.daily_year_files(
            dataset,
            file_start_year * 10_000 + 101,
            end_year * 10_000 + 12_31,
        );
        files
            .into_iter()
            .map(|file| cache.load_year_source(dataset, file, &columns))
            .collect()
    }

    pub fn load_stock_dividend(
        &self,
        requested_columns: &[String],
        start_date: i32,
        end_date: i32,
    ) -> Result<Table> {
        let columns = with_required_columns(
            requested_columns,
            &[
                "ts_code",
                "end_date",
                "ann_date",
                "div_proc",
                "cash_div_tax",
                "ex_date",
                "base_date",
                "base_share",
            ],
        );
        let files = self.catalog.daily_year_files(
            DatasetId::StockDividend,
            prior_year_start(start_date),
            end_date,
        );
        let mut table = Table::empty();
        for file in files {
            let yearly = read_parquet(&file, Some(&columns))?;
            let filtered = filter_dividend_range(&yearly, end_date)?;
            table.append(&filtered)?;
        }
        if table.columns.is_empty() {
            return empty_disclosure_table(&columns);
        }
        Ok(table)
    }

    pub fn load_stock_main_business_sources_cached(
        &self,
        requested_columns: &[String],
        start_date: i32,
        end_date: i32,
        quarters: usize,
        cache: &mut DisclosureTableCache,
    ) -> Result<Vec<Arc<Table>>> {
        let columns = with_required_columns(
            requested_columns,
            &[
                "ts_code",
                "end_date",
                "bz_type",
                "bz_item",
                "bz_sales",
                "update_flag",
            ],
        );
        let years = financial_disclosure_years_for_range(start_date, end_date, quarters);
        let file_start_year = years.iter().next().copied().unwrap_or(end_date / 10_000);
        let end_year = years
            .iter()
            .next_back()
            .copied()
            .unwrap_or(end_date / 10_000);
        let files = self.catalog.daily_year_files(
            DatasetId::StockMainBusiness,
            file_start_year * 10_000 + 101,
            end_year * 10_000 + 12_31,
        );
        files
            .into_iter()
            .map(|file| cache.load_year_source(DatasetId::StockMainBusiness, file, &columns))
            .collect()
    }

    pub fn load_stock_analyst_sources_cached(
        &self,
        requested_columns: &[String],
        start_date: i32,
        end_date: i32,
        cache: &mut DisclosureTableCache,
    ) -> Result<Vec<Arc<Table>>> {
        let columns = with_required_columns(
            requested_columns,
            &["ts_code", "report_date", "quarter", "rd"],
        );
        let files = self.catalog.daily_year_files(
            DatasetId::StockAnalystReport,
            prior_year_start(start_date),
            end_date,
        );
        files
            .into_iter()
            .map(|file| cache.load_year_source(DatasetId::StockAnalystReport, file, &columns))
            .collect()
    }

    pub fn load_stock_main_business_cached(
        &self,
        requested_columns: &[String],
        start_date: i32,
        end_date: i32,
        quarters: usize,
        cache: &mut DisclosureTableCache,
    ) -> Result<Table> {
        let columns = with_required_columns(
            requested_columns,
            &[
                "ts_code",
                "end_date",
                "bz_type",
                "bz_item",
                "bz_sales",
                "update_flag",
            ],
        );
        let years = financial_disclosure_years_for_range(start_date, end_date, quarters);
        let file_start_year = years.iter().next().copied().unwrap_or(end_date / 10_000);
        let end_year = years
            .iter()
            .next_back()
            .copied()
            .unwrap_or(end_date / 10_000);
        let files = self.catalog.daily_year_files(
            DatasetId::StockMainBusiness,
            file_start_year * 10_000 + 101,
            end_year * 10_000 + 12_31,
        );
        let mut table = Table::empty();
        for file in files {
            let yearly = cache.load_year_source(DatasetId::StockMainBusiness, file, &columns)?;
            append_disclosure_range(
                &mut table,
                &yearly,
                &columns,
                "end_date",
                i32::MIN,
                end_date,
            )?;
        }
        if table.columns.is_empty() {
            return empty_disclosure_table(&columns);
        }
        Ok(table)
    }

    pub fn load_stock_analyst_report(
        &self,
        requested_columns: &[String],
        start_date: i32,
        end_date: i32,
    ) -> Result<Table> {
        let mut cache = DisclosureTableCache::default();
        self.load_stock_analyst_report_cached(requested_columns, start_date, end_date, &mut cache)
    }

    pub fn load_stock_analyst_report_cached(
        &self,
        requested_columns: &[String],
        start_date: i32,
        end_date: i32,
        cache: &mut DisclosureTableCache,
    ) -> Result<Table> {
        let columns = with_required_columns(
            requested_columns,
            &["ts_code", "report_date", "quarter", "rd"],
        );
        let files = self.catalog.daily_year_files(
            DatasetId::StockAnalystReport,
            prior_year_start(start_date),
            end_date,
        );
        let mut table = Table::empty();
        for file in files {
            let yearly = cache.load_year_source(DatasetId::StockAnalystReport, file, &columns)?;
            append_disclosure_range(
                &mut table,
                &yearly,
                &columns,
                "report_date",
                i32::MIN,
                end_date,
            )?;
        }
        if table.columns.is_empty() {
            return empty_disclosure_table(&columns);
        }
        Ok(table)
    }

    pub fn load_stock_analyst_report_between_cached(
        &self,
        requested_columns: &[String],
        start_date: i32,
        end_date: i32,
        cache: &mut DisclosureTableCache,
    ) -> Result<Table> {
        let columns = with_required_columns(
            requested_columns,
            &["ts_code", "report_date", "quarter", "rd"],
        );
        let files =
            self.catalog
                .daily_year_files(DatasetId::StockAnalystReport, start_date, end_date);
        let mut table = Table::empty();
        for file in files {
            let yearly = cache.load_year_source(DatasetId::StockAnalystReport, file, &columns)?;
            append_disclosure_range(
                &mut table,
                &yearly,
                &columns,
                "report_date",
                start_date,
                end_date,
            )?;
        }
        if table.columns.is_empty() {
            return empty_disclosure_table(&columns);
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

    pub fn load_stock_derived_bar_by_date(
        &self,
        bar_size: usize,
        requested_columns: &[String],
        target_dates: &[i32],
    ) -> Result<HashMap<i32, Table>> {
        let columns =
            with_required_columns(requested_columns, &["bar_index", "ts_code", "minute_count"]);
        let mut tables = HashMap::new();
        for trade_date in target_dates {
            if let Some(file) = self.catalog.stock_derived_bar_file(bar_size, *trade_date) {
                let mut table = match read_parquet(&file, Some(&columns)) {
                    Ok(table) => table,
                    Err(error) => {
                        eprintln!(
                            "warning: failed to read stock derived {bar_size}m bar for {trade_date}: {error}; fallback data may be used"
                        );
                        continue;
                    }
                };
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
        if table.columns.is_empty() {
            return empty_daily_keyed_table(&columns);
        }
        Ok(table)
    }
}

fn empty_daily_keyed_table(columns: &[String]) -> Result<Table> {
    let mut values = BTreeMap::new();
    for column in columns {
        let data = match column.as_str() {
            "trade_date" => ColumnData::I32(Vec::new()),
            "ts_code" => ColumnData::Utf8(Vec::new()),
            _ => ColumnData::F64(Vec::new()),
        };
        values.insert(column.clone(), data);
    }
    Table::new(values)
}

fn empty_static_keyed_table(columns: &[String]) -> Result<Table> {
    let mut values = BTreeMap::new();
    for column in columns {
        let data = match column.as_str() {
            "ts_code" | "symbol" | "name" | "area" | "industry" | "fullname" | "enname"
            | "cnspell" | "market" | "exchange" | "curr_type" | "list_status" | "list_date"
            | "delist_date" | "is_hs" => ColumnData::Utf8(Vec::new()),
            _ => ColumnData::F64(Vec::new()),
        };
        values.insert(column.clone(), data);
    }
    Table::new(values)
}

fn empty_disclosure_table(columns: &[String]) -> Result<Table> {
    let mut values = BTreeMap::new();
    for column in columns {
        values.insert(column.clone(), empty_disclosure_column(column, 0));
    }
    Table::new(values)
}

fn project_table_columns(table: &Table, columns: &[String]) -> Result<Table> {
    let mut projected = BTreeMap::new();
    for column in columns {
        let data = table
            .columns
            .get(column)
            .cloned()
            .ok_or_else(|| err(format!("missing cached disclosure column {column}")))?;
        projected.insert(column.clone(), data);
    }
    Table::new(projected)
}

fn empty_disclosure_column(column: &str, len: usize) -> ColumnData {
    match column {
        "ts_code" | "quarter" | "org_name" | "author_name" | "create_time" | "rating"
        | "div_proc" | "bz_type" | "bz_item" | "bz_code" | "curr_type" => {
            ColumnData::Utf8(vec![None; len])
        }
        "ann_date" | "f_ann_date" | "end_date" | "report_date" | "ex_date" | "base_date" => {
            ColumnData::I32(vec![None; len])
        }
        "report_type" | "update_flag" => ColumnData::I64(vec![None; len]),
        _ => ColumnData::F64(vec![None; len]),
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

fn append_file_filtered_by_dates(
    target: &mut Table,
    file: PathBuf,
    columns: &[String],
    trade_dates: &[i32],
) -> Result<()> {
    let table = read_parquet(&file, Some(columns))?;
    append_table_filtered_by_dates(target, &table, trade_dates)
}

fn append_table_filtered_by_dates(
    target: &mut Table,
    table: &Table,
    trade_dates: &[i32],
) -> Result<()> {
    if table.len == 0 && !table.columns.contains_key("trade_date") {
        return Ok(());
    }
    let wanted = trade_dates.iter().copied().collect::<BTreeSet<_>>();
    let dates = table.required_i32("trade_date")?;
    let indices = dates
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| value.and_then(|date| wanted.contains(&date).then_some(idx)))
        .collect::<Vec<_>>();
    let filtered = table.take(&indices)?;
    target.append(&filtered)
}

pub fn financial_disclosure_years_for_range(
    start_date: i32,
    end_date: i32,
    quarters: usize,
) -> BTreeSet<i32> {
    let start_year = financial_file_start_year_for_range(start_date, end_date, quarters);
    let end_year = end_date / 10_000;
    (start_year..=end_year).collect()
}

fn financial_file_start_year_for_range(start_date: i32, end_date: i32, quarters: usize) -> i32 {
    let start = financial_file_start_year_for_date(start_date, quarters);
    let end = financial_file_start_year_for_date(end_date, quarters);
    start.min(end)
}

fn financial_file_start_year_for_date(date: i32, quarters: usize) -> i32 {
    let year = date / 10_000;
    if quarters == 0 {
        return year;
    }
    if quarters <= 8 {
        let previous_disclosure = previous_mandatory_disclosure_mmdd(date);
        let lookback_years = if previous_disclosure == 10_31 { 3 } else { 2 };
        return year.saturating_sub(lookback_years);
    }
    year.saturating_sub(quarters.div_ceil(4) as i32)
}

fn previous_mandatory_disclosure_mmdd(date: i32) -> i32 {
    let mmdd = date % 10_000;
    if mmdd < 4_30 {
        10_31
    } else if mmdd < 8_31 {
        4_30
    } else if mmdd < 10_31 {
        8_31
    } else {
        10_31
    }
}

fn is_financial_statement_dataset(dataset: DatasetId) -> bool {
    matches!(
        dataset,
        DatasetId::StockIncome
            | DatasetId::StockBalanceSheet
            | DatasetId::StockCashFlow
            | DatasetId::StockMainBusiness
    )
}

fn is_cached_disclosure_dataset(dataset: DatasetId) -> bool {
    is_financial_statement_dataset(dataset) || matches!(dataset, DatasetId::StockAnalystReport | DatasetId::FundPortfolio)
}

fn file_stem_year(path: &std::path::Path) -> Option<i32> {
    path.file_stem()?.to_str()?.parse::<i32>().ok()
}

fn prior_year_start(date: i32) -> i32 {
    (date / 10_000 - 1).max(0) * 10_000 + 101
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

fn filter_dividend_range(table: &Table, end_date: i32) -> Result<Table> {
    let ann_dates = table.required_i32_date_cast("ann_date")?;
    let ex_dates = table.required_i32_date_cast("ex_date")?;
    let indices = (0..table.len)
        .filter(|idx| {
            ann_dates[*idx].is_some_and(|date| date <= end_date)
                || ex_dates[*idx].is_some_and(|date| date <= end_date)
        })
        .collect::<Vec<_>>();
    table.take(&indices)
}

fn append_disclosure_range(
    target: &mut Table,
    source: &Table,
    columns: &[String],
    date_column: &str,
    start_date: i32,
    end_date: i32,
) -> Result<()> {
    for name in columns {
        if !source.columns.contains_key(name) {
            return Err(err(format!("missing cached disclosure column {name}")));
        }
    }
    let dates = source.required_i32_date_cast(date_column)?;
    let indices = (0..source.len)
        .filter(|idx| dates[*idx].is_some_and(|date| date >= start_date && date <= end_date))
        .collect::<Vec<_>>();
    // Keep at most one temporary column, not a projected yearly and filtered table.
    for name in columns {
        let selected = source.columns[name].take(&indices);
        if let Some(column) = target.columns.get_mut(name) {
            column.append(&selected)?;
        } else {
            target.columns.insert(name.clone(), selected);
        }
    }
    target.len += indices.len();
    Ok(())
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
