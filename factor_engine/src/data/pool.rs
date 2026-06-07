use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use crate::core::{DataRequest, DatasetId, FactorContext, Frequency};
use crate::data::loader::{DisclosureTableCache, MarketDataLoader};
use crate::data::table::Table;
use crate::error::{err, Result};
use crate::factor::common::{
    DailyPanel, FinancialPitIndex, FinancialPitReader, ReportTypePreference,
};

#[derive(Clone, Debug, Default)]
pub struct DataPool {
    daily: HashMap<DatasetId, Arc<Table>>,
    daily_panels: HashMap<DatasetId, DailyPanel>,
    financial_pit_indexes: HashMap<DatasetId, Arc<FinancialPitIndex>>,
    index_daily: HashMap<String, Arc<Table>>,
    index_daily_panels: HashMap<String, DailyPanel>,
    minute: HashMap<(DatasetId, Option<usize>, i32), Arc<Table>>,
    intraday_daily_raw: Option<Arc<Table>>,
    intraday_daily_raw_panel: Option<DailyPanel>,
}

impl DataPool {
    pub fn load(
        loader: &MarketDataLoader,
        requests: &[DataRequest],
        context: &FactorContext,
    ) -> Result<Self> {
        let mut disclosure_cache = DisclosureTableCache::default();
        Self::load_with_disclosure_cache(loader, requests, context, &mut disclosure_cache)
    }

    pub fn load_with_disclosure_cache(
        loader: &MarketDataLoader,
        requests: &[DataRequest],
        context: &FactorContext,
        disclosure_cache: &mut DisclosureTableCache,
    ) -> Result<Self> {
        let mut grouped: HashMap<
            (DatasetId, Option<String>, Option<usize>),
            (BTreeSet<String>, Option<usize>),
        > = HashMap::new();
        for request in requests {
            let entry = grouped
                .entry((request.dataset, request.entity_id.clone(), request.bar_size))
                .or_default();
            entry.0.extend(request.columns.iter().cloned());
            entry.1 = match (entry.1, request.financial_quarters) {
                (Some(left), Some(right)) => Some(left.max(right)),
                (None, Some(right)) => Some(right),
                (left, None) => left,
            };
        }

        let mut pool = Self::default();
        for ((dataset, entity_id, bar_size), (columns, financial_quarters)) in grouped {
            let columns = columns.into_iter().collect::<Vec<_>>();
            if dataset == DatasetId::IndexDaily {
                let ts_code = entity_id.ok_or_else(|| {
                    err("index.daily request requires entity_id; use DataRequest::index_daily")
                })?;
                let table =
                    loader.load_index_daily_by_dates(&ts_code, &columns, &context.load_dates)?;
                let panel = DailyPanel::from_table(&table, context)?;
                pool.index_daily_panels.insert(ts_code.clone(), panel);
                pool.index_daily.insert(ts_code, Arc::new(table));
                continue;
            }
            if dataset == DatasetId::StockSwClassification {
                let table = loader.load_stock_sw_classification(
                    &columns,
                    context.load_start_date,
                    context.end_date,
                )?;
                pool.daily.insert(dataset, Arc::new(table));
                continue;
            }
            if dataset == DatasetId::StockCiClassification {
                let table = loader.load_stock_ci_classification(
                    &columns,
                    context.load_start_date,
                    context.end_date,
                )?;
                pool.daily.insert(dataset, Arc::new(table));
                continue;
            }
            if dataset == DatasetId::StockBasic {
                let table = loader.load_stock_basic(&columns)?;
                pool.daily.insert(dataset, Arc::new(table));
                continue;
            }
            if dataset == DatasetId::StockBarraDaily {
                let table = loader.load_barra_daily(
                    context.asset_class,
                    "CNE6",
                    &columns,
                    &context.load_dates,
                )?;
                let panel = DailyPanel::from_table(&table, context)?;
                pool.daily_panels.insert(dataset, panel);
                pool.daily.insert(dataset, Arc::new(table));
                continue;
            }
            if matches!(
                dataset,
                DatasetId::StockIncome | DatasetId::StockBalanceSheet | DatasetId::StockCashFlow
            ) {
                let table = loader.load_financial_cached(
                    dataset,
                    &columns,
                    context.start_date,
                    context.end_date,
                    financial_quarters.unwrap_or(0),
                    disclosure_cache,
                )?;
                let table = Arc::new(table);
                let index = FinancialPitIndex::from_table(Arc::clone(&table))?;
                pool.financial_pit_indexes.insert(dataset, Arc::new(index));
                pool.daily.insert(dataset, table);
                continue;
            }
            if dataset == DatasetId::StockDividend {
                let table = loader.load_stock_dividend(
                    &columns,
                    context.load_start_date,
                    context.end_date,
                )?;
                pool.daily.insert(dataset, Arc::new(table));
                continue;
            }
            if dataset == DatasetId::StockAnalystReport {
                let table = loader.load_stock_analyst_report_cached(
                    &columns,
                    context.load_start_date,
                    context.end_date,
                    disclosure_cache,
                )?;
                pool.daily.insert(dataset, Arc::new(table));
                continue;
            }
            match dataset.frequency() {
                Frequency::Daily => {
                    let table =
                        loader.load_daily_by_dates(dataset, &columns, &context.load_dates)?;
                    if should_build_daily_panel(dataset) {
                        let panel = DailyPanel::from_table(&table, context)?;
                        pool.daily_panels.insert(dataset, panel);
                    }
                    pool.daily.insert(dataset, Arc::new(table));
                }
                Frequency::Minute1 => {
                    let target_dates = if context.frequency == Frequency::Daily {
                        &context.load_dates
                    } else {
                        &context.target_dates
                    };
                    let tables = if dataset == DatasetId::StockDerivedBar {
                        let bar_size = bar_size
                            .ok_or_else(|| err("stock.derived.bar request requires bar_size"))?;
                        loader.load_stock_derived_bar_by_date(bar_size, &columns, target_dates)?
                    } else {
                        loader.load_minute_by_date(dataset, &columns, target_dates)?
                    };
                    for (date, table) in tables {
                        pool.minute
                            .insert((dataset, bar_size, date), Arc::new(table));
                    }
                }
            }
        }
        Ok(pool)
    }

    pub fn with_target_dates(&self, target_dates: &[i32]) -> Self {
        Self {
            daily: self.daily.clone(),
            daily_panels: self
                .daily_panels
                .iter()
                .map(|(dataset, panel)| (*dataset, panel.with_target_dates(target_dates)))
                .collect(),
            financial_pit_indexes: self.financial_pit_indexes.clone(),
            index_daily: self.index_daily.clone(),
            index_daily_panels: self
                .index_daily_panels
                .iter()
                .map(|(ts_code, panel)| (ts_code.clone(), panel.with_target_dates(target_dates)))
                .collect(),
            minute: self.minute.clone(),
            intraday_daily_raw: self.intraday_daily_raw.clone(),
            intraday_daily_raw_panel: self
                .intraday_daily_raw_panel
                .as_ref()
                .map(|panel| panel.with_target_dates(target_dates)),
        }
    }

    pub fn slice_dates(&self, selected_dates: &[i32]) -> Self {
        Self {
            daily: self.daily.clone(),
            daily_panels: self
                .daily_panels
                .iter()
                .map(|(dataset, panel)| (*dataset, panel.slice_dates(selected_dates)))
                .collect(),
            financial_pit_indexes: self.financial_pit_indexes.clone(),
            index_daily: self.index_daily.clone(),
            index_daily_panels: self
                .index_daily_panels
                .iter()
                .map(|(ts_code, panel)| (ts_code.clone(), panel.slice_dates(selected_dates)))
                .collect(),
            minute: self.minute.clone(),
            intraday_daily_raw: self.intraday_daily_raw.clone(),
            intraday_daily_raw_panel: self
                .intraday_daily_raw_panel
                .as_ref()
                .map(|panel| panel.slice_dates(selected_dates)),
        }
    }

    pub fn daily(&self, dataset: DatasetId) -> Result<&Table> {
        self.daily
            .get(&dataset)
            .map(Arc::as_ref)
            .ok_or_else(|| err(format!("daily dataset not loaded: {}", dataset.as_str())))
    }

    pub fn daily_panel(&self, dataset: DatasetId) -> Result<&DailyPanel> {
        self.daily_panels
            .get(&dataset)
            .ok_or_else(|| err(format!("daily panel not loaded: {}", dataset.as_str())))
    }

    pub fn financial_reader(
        &self,
        dataset: DatasetId,
        preference: ReportTypePreference,
    ) -> Result<FinancialPitReader<'_>> {
        self.financial_pit_indexes
            .get(&dataset)
            .map(|index| index.reader(preference))
            .ok_or_else(|| {
                err(format!(
                    "financial PIT index not loaded: {}",
                    dataset.as_str()
                ))
            })
    }

    pub fn index_daily_panel(&self, ts_code: &str) -> Result<&DailyPanel> {
        self.index_daily_panels
            .get(ts_code)
            .ok_or_else(|| err(format!("index daily panel not loaded: {ts_code}")))
    }

    pub fn minute(&self, dataset: DatasetId, trade_date: i32) -> Option<&Table> {
        self.minute
            .get(&(dataset, None, trade_date))
            .map(Arc::as_ref)
    }

    pub fn derived_bar(&self, bar_size: usize, trade_date: i32) -> Option<&Table> {
        self.minute
            .get(&(DatasetId::StockDerivedBar, Some(bar_size), trade_date))
            .map(Arc::as_ref)
    }

    #[cfg(test)]
    pub(crate) fn insert_minute_table_for_test(
        &mut self,
        dataset: DatasetId,
        bar_size: Option<usize>,
        trade_date: i32,
        table: Table,
    ) {
        self.minute
            .insert((dataset, bar_size, trade_date), Arc::new(table));
    }

    pub fn extend(&mut self, other: Self) {
        self.daily.extend(other.daily);
        self.daily_panels.extend(other.daily_panels);
        self.financial_pit_indexes
            .extend(other.financial_pit_indexes);
        self.index_daily.extend(other.index_daily);
        self.index_daily_panels.extend(other.index_daily_panels);
        self.minute.extend(other.minute);
        if other.intraday_daily_raw.is_some() {
            self.intraday_daily_raw = other.intraday_daily_raw;
        }
        if other.intraday_daily_raw_panel.is_some() {
            self.intraday_daily_raw_panel = other.intraday_daily_raw_panel;
        }
    }

    pub fn set_intraday_daily_raw(&mut self, table: Table, context: &FactorContext) -> Result<()> {
        let panel = DailyPanel::from_table(&table, context)?;
        self.intraday_daily_raw = Some(Arc::new(table));
        self.intraday_daily_raw_panel = Some(panel);
        Ok(())
    }

    pub fn intraday_daily_raw(&self, raw_id: &str) -> Result<&Table> {
        let table = self
            .intraday_daily_raw
            .as_ref()
            .ok_or_else(|| err("intraday daily raw cache is not loaded"))?;
        if !table.columns.contains_key(raw_id) {
            return Err(err(format!(
                "intraday daily raw cache column not loaded: {raw_id}"
            )));
        }
        Ok(table.as_ref())
    }

    pub fn intraday_daily_raw_panel(&self, raw_id: &str) -> Result<&DailyPanel> {
        let panel = self
            .intraday_daily_raw_panel
            .as_ref()
            .ok_or_else(|| err("intraday daily raw panel is not loaded"))?;
        if !panel.has_column(raw_id) {
            return Err(err(format!(
                "intraday daily raw panel column not loaded: {raw_id}"
            )));
        }
        Ok(panel)
    }

    #[cfg(test)]
    pub fn from_minute_tables(minute: HashMap<(DatasetId, i32), Table>) -> Self {
        Self {
            daily: HashMap::new(),
            daily_panels: HashMap::new(),
            financial_pit_indexes: HashMap::new(),
            index_daily: HashMap::new(),
            index_daily_panels: HashMap::new(),
            minute: minute
                .into_iter()
                .map(|((dataset, date), table)| ((dataset, None, date), Arc::new(table)))
                .collect(),
            intraday_daily_raw: None,
            intraday_daily_raw_panel: None,
        }
    }

    #[cfg(test)]
    pub fn from_daily_tables_for_test(
        daily: HashMap<DatasetId, Table>,
        context: &FactorContext,
    ) -> Result<Self> {
        let mut daily_panels = HashMap::new();
        for (dataset, table) in &daily {
            if should_build_daily_panel(*dataset) {
                daily_panels.insert(*dataset, DailyPanel::from_table(table, context)?);
            }
        }
        let mut daily_arc = HashMap::new();
        let mut financial_pit_indexes = HashMap::new();
        for (dataset, table) in daily {
            let table = Arc::new(table);
            if matches!(
                dataset,
                DatasetId::StockIncome | DatasetId::StockBalanceSheet | DatasetId::StockCashFlow
            ) {
                financial_pit_indexes.insert(
                    dataset,
                    Arc::new(FinancialPitIndex::from_table(Arc::clone(&table))?),
                );
            }
            daily_arc.insert(dataset, table);
        }
        Ok(Self {
            daily: daily_arc,
            daily_panels,
            financial_pit_indexes,
            index_daily: HashMap::new(),
            index_daily_panels: HashMap::new(),
            minute: HashMap::new(),
            intraday_daily_raw: None,
            intraday_daily_raw_panel: None,
        })
    }
}

fn should_build_daily_panel(dataset: DatasetId) -> bool {
    matches!(
        dataset,
        DatasetId::StockDailyPv
            | DatasetId::StockDailyBasic
            | DatasetId::StockDailyLimit
            | DatasetId::StockAdjFactor
            | DatasetId::StockMoneyflow
            | DatasetId::StockBarraDaily
            | DatasetId::IndexDaily
            | DatasetId::FutureDaily
    )
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};

    use crate::core::{AssetClass, FactorContext};
    use crate::data::ColumnData;

    use super::*;

    fn context() -> FactorContext {
        FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: 20260102,
            end_date: 20260102,
            load_start_date: 20260101,
            load_dates: vec![20260101, 20260102],
            target_dates: vec![20260102],
        }
    }

    fn sample_daily_table() -> Table {
        Table::new(BTreeMap::from([
            (
                "trade_date".to_string(),
                ColumnData::I32(vec![Some(20260101), Some(20260102)]),
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
                ColumnData::F64(vec![Some(10.0), Some(11.0)]),
            ),
        ]))
        .expect("valid daily table")
    }

    #[test]
    fn daily_panel_cache_exposes_prebuilt_panel() {
        let context = context();
        let table = sample_daily_table();
        let expected = DailyPanel::from_table(&table, &context).expect("panel");
        let pool = DataPool {
            daily: HashMap::from([(DatasetId::StockDailyPv, Arc::new(table))]),
            daily_panels: HashMap::from([(DatasetId::StockDailyPv, expected.clone())]),
            financial_pit_indexes: HashMap::new(),
            index_daily: HashMap::new(),
            index_daily_panels: HashMap::new(),
            minute: HashMap::new(),
            intraday_daily_raw: None,
            intraday_daily_raw_panel: None,
        };

        let cached = pool.daily_panel(DatasetId::StockDailyPv).expect("cached");

        assert_eq!(
            cached.column("close").expect("cached close").values(),
            expected.column("close").expect("expected close").values()
        );
        assert!(pool.daily_panel(DatasetId::StockSwClassification).is_err());
        assert!(pool.daily_panel(DatasetId::StockCiClassification).is_err());
    }

    #[test]
    fn intraday_daily_raw_panel_is_built_when_raw_table_is_set() {
        let context = context();
        let mut pool = DataPool::default();

        pool.set_intraday_daily_raw(sample_daily_table(), &context)
            .expect("raw panel");

        let panel = pool
            .intraday_daily_raw_panel("close")
            .expect("raw panel with close");
        assert_eq!(
            panel.column("close").expect("close").values(),
            &[Some(10.0), Some(11.0)]
        );
        assert!(pool.intraday_daily_raw_panel("missing").is_err());
    }
}
