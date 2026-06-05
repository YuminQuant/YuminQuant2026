use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;

use crate::core::{AssetClass, DatasetId};
use crate::data::catalog::DataCatalog;
use crate::data::parquet_io::read_parquet;
use crate::data::table::{ColumnData, Table};
use crate::error::{err, Result};

#[derive(Clone, Debug)]
pub struct MarketDataLoader {
    catalog: DataCatalog,
}

#[derive(Clone, Debug, Default)]
pub struct DisclosureTableCache {
    yearly_tables: HashMap<DisclosureCacheKey, Table>,
}

impl DisclosureTableCache {
    pub fn len(&self) -> usize {
        self.yearly_tables.len()
    }

    fn load_year(
        &mut self,
        dataset: DatasetId,
        file: PathBuf,
        columns: &[String],
    ) -> Result<&Table> {
        let key = DisclosureCacheKey {
            dataset,
            file,
            columns: columns.to_vec(),
        };
        if !self.yearly_tables.contains_key(&key) {
            let table = read_parquet(&key.file, Some(columns))?;
            self.yearly_tables.insert(key.clone(), table);
        }
        self.yearly_tables
            .get(&key)
            .ok_or_else(|| err("disclosure cache entry disappeared unexpectedly"))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct DisclosureCacheKey {
    dataset: DatasetId,
    file: PathBuf,
    columns: Vec<String>,
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
            let yearly = cache.load_year(dataset, file, &columns)?;
            let filtered = filter_financial_disclosure_range(yearly, end_date)?;
            table.append(&filtered)?;
        }
        if table.columns.is_empty() {
            return empty_disclosure_table(&columns);
        }
        Ok(table)
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
            let yearly = cache.load_year(DatasetId::StockAnalystReport, file, &columns)?;
            let filtered = filter_analyst_report_range(yearly, end_date)?;
            table.append(&filtered)?;
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

fn empty_disclosure_table(columns: &[String]) -> Result<Table> {
    let mut values = BTreeMap::new();
    for column in columns {
        let data = match column.as_str() {
            "ts_code" | "quarter" | "div_proc" => ColumnData::Utf8(Vec::new()),
            "ann_date" | "f_ann_date" | "end_date" | "report_date" | "ex_date" | "base_date" => {
                ColumnData::I32(Vec::new())
            }
            "report_type" | "update_flag" => ColumnData::I64(Vec::new()),
            _ => ColumnData::F64(Vec::new()),
        };
        values.insert(column.clone(), data);
    }
    Table::new(values)
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

fn financial_lookback_years(quarters: usize) -> i32 {
    if quarters == 0 {
        return 0;
    }
    (quarters.div_ceil(4) as i32) + 3
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

fn filter_analyst_report_range(table: &Table, end_date: i32) -> Result<Table> {
    let report_dates = table.required_i32_date_cast("report_date")?;
    let indices = (0..table.len)
        .filter(|idx| report_dates[*idx].is_some_and(|date| date <= end_date))
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
    fn empty_disclosure_table_preserves_requested_schema() {
        let table = empty_disclosure_table(&[
            "ts_code".to_string(),
            "report_date".to_string(),
            "quarter".to_string(),
            "report_type".to_string(),
            "eps".to_string(),
        ])
        .expect("empty disclosure table");

        assert_eq!(table.len, 0);
        assert!(matches!(
            table.columns.get("ts_code"),
            Some(ColumnData::Utf8(_))
        ));
        assert!(matches!(
            table.columns.get("report_date"),
            Some(ColumnData::I32(_))
        ));
        assert!(matches!(
            table.columns.get("quarter"),
            Some(ColumnData::Utf8(_))
        ));
        assert!(matches!(
            table.columns.get("report_type"),
            Some(ColumnData::I64(_))
        ));
        assert!(matches!(table.columns.get("eps"), Some(ColumnData::F64(_))));
        assert!(table.required_utf8("ts_code").is_ok());
        assert!(table.required_i32_date_cast("report_date").is_ok());
        assert!(table.required_utf8("quarter").is_ok());
        assert!(table.required_i64_cast("report_type").is_ok());
        assert!(table.required_f64_cast("eps").is_ok());
    }

    #[test]
    fn disclosure_cache_reuses_raw_financial_year_table_across_end_dates() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("yq_financial_cache_test_{unique}"));
        let path = root.join("stock_data").join("income").join("2020.parquet");
        let table = Table::new(BTreeMap::from([
            (
                "ts_code".to_string(),
                ColumnData::Utf8(vec![
                    Some("000001.SZ".to_string()),
                    Some("000001.SZ".to_string()),
                ]),
            ),
            (
                "ann_date".to_string(),
                ColumnData::I32(vec![Some(20200115), Some(20200301)]),
            ),
            (
                "f_ann_date".to_string(),
                ColumnData::I32(vec![Some(20200115), Some(20200301)]),
            ),
            (
                "end_date".to_string(),
                ColumnData::I32(vec![Some(20191231), Some(20200331)]),
            ),
            (
                "report_type".to_string(),
                ColumnData::I64(vec![Some(1), Some(1)]),
            ),
            (
                "update_flag".to_string(),
                ColumnData::I64(vec![Some(0), Some(0)]),
            ),
            (
                "ebit".to_string(),
                ColumnData::F64(vec![Some(10.0), Some(20.0)]),
            ),
        ]))
        .expect("financial table");
        write_parquet(&path, &table).expect("write financial table");

        let loader = MarketDataLoader::new(DataCatalog::new(root.clone()));
        let mut cache = DisclosureTableCache::default();
        let early = loader
            .load_financial_cached(
                DatasetId::StockIncome,
                &["ebit".to_string()],
                20200101,
                20200131,
                0,
                &mut cache,
            )
            .expect("load early financial");
        let later = loader
            .load_financial_cached(
                DatasetId::StockIncome,
                &["ebit".to_string()],
                20200101,
                20200331,
                0,
                &mut cache,
            )
            .expect("load later financial");

        assert_eq!(cache.len(), 1);
        assert_eq!(early.len, 1);
        assert_eq!(later.len, 2);
        assert_eq!(
            later.required_f64_cast("ebit").expect("ebit"),
            vec![Some(10.0), Some(20.0)]
        );

        fs::remove_dir_all(root).expect("cleanup");
    }

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

        if root.exists() {
            fs::remove_dir_all(root).expect("cleanup");
        }
    }

    #[test]
    fn index_daily_loader_treats_missing_dates_as_empty_keyed_table() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("yq_index_missing_loader_test_{unique}"));
        let loader = MarketDataLoader::new(DataCatalog::new(root.clone()));
        let loaded = loader
            .load_index_daily_by_dates("000985.CSI", &["close".to_string()], &[20090105])
            .expect("missing index files should be tolerated");

        assert_eq!(loaded.len, 0);
        assert!(loaded.columns.contains_key("trade_date"));
        assert!(loaded.columns.contains_key("ts_code"));
        assert!(loaded.columns.contains_key("close"));

        if root.exists() {
            fs::remove_dir_all(root).expect("cleanup");
        }
    }

    #[test]
    fn daily_loader_reads_date_files_before_yearly_fallback() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("yq_daily_date_loader_test_{unique}"));
        let yearly_path = root
            .join("stock_data")
            .join("daily")
            .join("pv")
            .join("2026.parquet");
        let daily_path = root
            .join("stock_data")
            .join("daily")
            .join("pv")
            .join("2026")
            .join("20260103.parquet");
        let yearly = Table::new(BTreeMap::from([
            (
                "trade_date".to_string(),
                ColumnData::I32(vec![Some(20260102), Some(20260103)]),
            ),
            (
                "ts_code".to_string(),
                ColumnData::Utf8(vec![
                    Some("000001.SZ".to_string()),
                    Some("000001.SZ".to_string()),
                ]),
            ),
            (
                "close".to_string(),
                ColumnData::F32(vec![Some(10.0), Some(11.0)]),
            ),
        ]))
        .expect("yearly table");
        let daily = Table::new(BTreeMap::from([
            (
                "trade_date".to_string(),
                ColumnData::I32(vec![Some(20260103)]),
            ),
            (
                "ts_code".to_string(),
                ColumnData::Utf8(vec![Some("000001.SZ".to_string())]),
            ),
            ("close".to_string(), ColumnData::F32(vec![Some(99.0)])),
        ]))
        .expect("daily table");
        write_parquet(&yearly_path, &yearly).expect("write yearly");
        write_parquet(&daily_path, &daily).expect("write daily");

        let loader = MarketDataLoader::new(DataCatalog::new(root.clone()));
        let loaded = loader
            .load_daily_by_dates(
                DatasetId::StockDailyPv,
                &["close".to_string()],
                &[20260102, 20260103],
            )
            .expect("load daily by dates");

        assert_eq!(loaded.len, 2);
        assert_eq!(
            loaded.required_i32("trade_date").expect("trade_date"),
            &vec![Some(20260102), Some(20260103)]
        );
        assert_eq!(
            loaded.required_f64_cast("close").expect("close"),
            vec![Some(10.0), Some(99.0)]
        );

        if root.exists() {
            fs::remove_dir_all(root).expect("cleanup");
        }
    }

    #[test]
    fn daily_loader_tolerates_missing_date_and_yearly_files() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("yq_daily_missing_loader_test_{unique}"));
        let loader = MarketDataLoader::new(DataCatalog::new(root.clone()));
        let loaded = loader
            .load_daily_by_dates(DatasetId::StockDailyPv, &["open".to_string()], &[20260102])
            .expect("missing daily files should be tolerated");

        assert_eq!(loaded.len, 0);
        assert!(loaded.columns.contains_key("trade_date"));
        assert!(loaded.columns.contains_key("ts_code"));
        assert!(loaded.columns.contains_key("open"));

        if root.exists() {
            fs::remove_dir_all(root).expect("cleanup");
        }
    }

    #[test]
    fn stock_adj_factor_loader_reads_daily_date_files() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("yq_adj_factor_loader_test_{unique}"));
        let daily_path = root
            .join("stock_data")
            .join("daily")
            .join("adj_factor")
            .join("2026")
            .join("20260102.parquet");
        let daily = Table::new(BTreeMap::from([
            (
                "trade_date".to_string(),
                ColumnData::I32(vec![Some(20260102)]),
            ),
            (
                "ts_code".to_string(),
                ColumnData::Utf8(vec![Some("000001.SZ".to_string())]),
            ),
            ("adj_factor".to_string(), ColumnData::F32(vec![Some(1.25)])),
        ]))
        .expect("daily table");
        write_parquet(&daily_path, &daily).expect("write daily");

        let loader = MarketDataLoader::new(DataCatalog::new(root.clone()));
        let loaded = loader
            .load_daily_by_dates(
                DatasetId::StockAdjFactor,
                &["adj_factor".to_string()],
                &[20260102],
            )
            .expect("load adj factor by date");

        assert_eq!(loaded.len, 1);
        assert_eq!(
            loaded.required_f64_cast("adj_factor").expect("adj_factor"),
            vec![Some(1.25)]
        );

        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn stock_moneyflow_loader_reads_daily_date_files() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("yq_moneyflow_loader_test_{unique}"));
        let daily_path = root
            .join("stock_data")
            .join("daily")
            .join("moneyflow")
            .join("2026")
            .join("20260102.parquet");
        let daily = Table::new(BTreeMap::from([
            (
                "trade_date".to_string(),
                ColumnData::I32(vec![Some(20260102)]),
            ),
            (
                "ts_code".to_string(),
                ColumnData::Utf8(vec![Some("000001.SZ".to_string())]),
            ),
            (
                "buy_sm_amount".to_string(),
                ColumnData::F64(vec![Some(10.0)]),
            ),
            (
                "sell_sm_amount".to_string(),
                ColumnData::F64(vec![Some(12.0)]),
            ),
        ]))
        .expect("daily table");
        write_parquet(&daily_path, &daily).expect("write daily");

        let loader = MarketDataLoader::new(DataCatalog::new(root.clone()));
        let loaded = loader
            .load_daily_by_dates(
                DatasetId::StockMoneyflow,
                &["buy_sm_amount".to_string(), "sell_sm_amount".to_string()],
                &[20260102],
            )
            .expect("load moneyflow by date");

        assert_eq!(loaded.len, 1);
        assert_eq!(
            loaded.required_f64_cast("buy_sm_amount").expect("buy"),
            vec![Some(10.0)]
        );
        assert_eq!(
            loaded.required_f64_cast("sell_sm_amount").expect("sell"),
            vec![Some(12.0)]
        );

        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn barra_daily_loader_treats_missing_dates_as_empty_table() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("yq_barra_missing_loader_test_{unique}"));
        let loader = MarketDataLoader::new(DataCatalog::new(root.clone()));
        let loaded = loader
            .load_barra_daily(
                AssetClass::Stock,
                "CNE6",
                &["SIZE".to_string()],
                &[20260424],
            )
            .expect("missing barra files should be tolerated");

        assert_eq!(loaded.len, 0);
        assert!(loaded.columns.contains_key("trade_date"));
        assert!(loaded.columns.contains_key("ts_code"));
        assert!(loaded.columns.contains_key("SIZE"));

        if root.exists() {
            fs::remove_dir_all(root).expect("cleanup");
        }
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
