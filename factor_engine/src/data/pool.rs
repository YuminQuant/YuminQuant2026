use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use crate::core::{DataRequest, DatasetId, FactorContext, Frequency};
use crate::data::loader::{DisclosureTableCache, MarketDataLoader};
use crate::data::table::Table;
use crate::error::{err, Result};
use crate::factor::common::{
    DailyPanel, DividendIndex, DividendReader, FinancialPitIndex, FinancialPitReader,
    MainBusinessIndex, MainBusinessReader, ReportTypePreference,
};

#[derive(Clone, Debug, Default)]
pub struct FinancialBatchProfile {
    pub source_rows: usize,
    pub source_columns: usize,
    pub pit_records: usize,
    pub main_business_records: usize,
    pub dividend_records: usize,
    pub analyst_records: usize,
    pub builds: usize,
}

#[derive(Clone, Debug, Default)]
pub struct FinancialBatchContext {
    financial_pit_indexes: HashMap<DatasetId, Arc<FinancialPitIndex>>,
    dividend_index: Option<Arc<DividendIndex>>,
    main_business_index: Option<Arc<MainBusinessIndex>>,
    analyst_report: Option<Arc<Table>>,
    profile: FinancialBatchProfile,
}

impl FinancialBatchContext {
    pub fn build(
        loader: &MarketDataLoader,
        requests: &[DataRequest],
        context: &FactorContext,
        disclosure_cache: &mut DisclosureTableCache,
    ) -> Result<Option<Arc<Self>>> {
        let mut grouped: HashMap<DatasetId, (BTreeSet<String>, Option<usize>)> = HashMap::new();
        for request in requests {
            if !is_financial_context_dataset(request.dataset) {
                continue;
            }
            let entry = grouped.entry(request.dataset).or_default();
            entry.0.extend(request.columns.iter().cloned());
            entry.1 = match (entry.1, request.financial_quarters) {
                (Some(left), Some(right)) => Some(left.max(right)),
                (None, Some(right)) => Some(right),
                (left, None) => left,
            };
        }
        if grouped.is_empty() {
            return Ok(None);
        }

        let mut financial_pit_indexes = HashMap::new();
        let mut dividend_index = None;
        let mut main_business_index = None;
        let mut analyst_report = None;
        let mut profile = FinancialBatchProfile {
            builds: 1,
            ..Default::default()
        };

        for (dataset, (columns, financial_quarters)) in grouped {
            let columns = columns.into_iter().collect::<Vec<_>>();
            match dataset {
                DatasetId::StockIncome
                | DatasetId::StockBalanceSheet
                | DatasetId::StockCashFlow => {
                    let sources = loader.load_financial_sources_cached(
                        dataset,
                        &columns,
                        context.start_date,
                        context.end_date,
                        financial_quarters.unwrap_or(0),
                        disclosure_cache,
                    )?;
                    profile.source_rows += sources.iter().map(|table| table.len).sum::<usize>();
                    profile.source_columns += sources
                        .iter()
                        .map(|table| table.columns.len())
                        .sum::<usize>();
                    let index = Arc::new(FinancialPitIndex::from_source_tables(
                        sources,
                        Some(context.end_date),
                    )?);
                    profile.pit_records += index.len();
                    financial_pit_indexes.insert(dataset, index);
                }
                DatasetId::StockDividend => {
                    let table = loader.load_stock_dividend(
                        &columns,
                        context.load_start_date,
                        context.end_date,
                    )?;
                    profile.source_rows += table.len;
                    profile.source_columns += table.columns.len();
                    let table = Arc::new(table);
                    let index = Arc::new(DividendIndex::from_table(Arc::clone(&table))?);
                    profile.dividend_records = index.len();
                    dividend_index = Some(index);
                }
                DatasetId::StockMainBusiness => {
                    let table = loader.load_stock_main_business_cached(
                        &columns,
                        context.start_date,
                        context.end_date,
                        financial_quarters.unwrap_or(8),
                        disclosure_cache,
                    )?;
                    profile.source_rows += table.len;
                    profile.source_columns += table.columns.len();
                    let table = Arc::new(table);
                    let index = Arc::new(MainBusinessIndex::from_table(Arc::clone(&table))?);
                    profile.main_business_records = index.len();
                    main_business_index = Some(index);
                }
                DatasetId::StockAnalystReport => {
                    let table = loader.load_stock_analyst_report_cached(
                        &columns,
                        context.load_start_date,
                        context.end_date,
                        disclosure_cache,
                    )?;
                    profile.analyst_records = table.len;
                    profile.source_rows += table.len;
                    profile.source_columns += table.columns.len();
                    analyst_report = Some(Arc::new(table));
                }
                _ => {}
            }
        }

        Ok(Some(Arc::new(Self {
            financial_pit_indexes,
            dividend_index,
            main_business_index,
            analyst_report,
            profile,
        })))
    }

    pub fn financial_reader(
        &self,
        dataset: DatasetId,
        preference: ReportTypePreference,
    ) -> Option<FinancialPitReader<'_>> {
        self.financial_pit_indexes
            .get(&dataset)
            .map(|index| index.reader(preference))
    }

    pub fn dividend_reader(&self) -> Option<DividendReader<'_>> {
        self.dividend_index.as_ref().map(|index| index.reader())
    }

    pub fn main_business_reader(&self) -> Option<MainBusinessReader<'_>> {
        self.main_business_index
            .as_ref()
            .map(|index| index.reader())
    }

    pub fn analyst_report(&self) -> Option<&Table> {
        self.analyst_report.as_deref()
    }

    pub fn profile(&self) -> FinancialBatchProfile {
        self.profile.clone()
    }

    pub fn indexed_row_count(&self) -> usize {
        self.profile.pit_records
            + self.profile.main_business_records
            + self.profile.dividend_records
    }
}

#[derive(Clone, Debug, Default)]
pub struct DataPool {
    daily: HashMap<DatasetId, Arc<Table>>,
    daily_panels: HashMap<DatasetId, DailyPanel>,
    stock_universe_panel: Option<DailyPanel>,
    financial_pit_indexes: HashMap<DatasetId, Arc<FinancialPitIndex>>,
    dividend_index: Option<Arc<DividendIndex>>,
    main_business_index: Option<Arc<MainBusinessIndex>>,
    financial_context: Option<Arc<FinancialBatchContext>>,
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
        Self::load_with_financial_context(loader, requests, context, disclosure_cache, None)
    }

    pub fn load_with_financial_context(
        loader: &MarketDataLoader,
        requests: &[DataRequest],
        context: &FactorContext,
        disclosure_cache: &mut DisclosureTableCache,
        financial_context: Option<Arc<FinancialBatchContext>>,
    ) -> Result<Self> {
        let mut grouped: HashMap<
            (DatasetId, Option<String>, Option<usize>),
            (BTreeSet<String>, Option<usize>, BTreeSet<i32>),
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
            entry.2.extend(request.resolved_dates(context));
        }

        let mut pool = Self::default();
        pool.financial_context = financial_context;
        if grouped
            .keys()
            .any(|(dataset, _, _)| is_financial_context_dataset(*dataset))
        {
            let table = loader.load_stock_basic(&stock_universe_columns())?;
            pool.stock_universe_panel = Some(DailyPanel::from_stock_basic(&table, context)?);
            pool.daily
                .entry(DatasetId::StockBasic)
                .or_insert_with(|| Arc::new(table));
        }
        for ((dataset, entity_id, bar_size), (columns, financial_quarters, load_dates)) in grouped {
            let columns = columns.into_iter().collect::<Vec<_>>();
            let load_dates = load_dates.into_iter().collect::<Vec<_>>();
            if pool.financial_context.is_some() && is_financial_context_dataset(dataset) {
                continue;
            }
            if dataset == DatasetId::IndexDaily {
                let ts_code = entity_id.ok_or_else(|| {
                    err("index.daily request requires entity_id; use DataRequest::index_daily")
                })?;
                let table = loader.load_index_daily_by_dates(&ts_code, &columns, &load_dates)?;
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
                let table =
                    loader.load_barra_daily(context.asset_class, "CNE6", &columns, &load_dates)?;
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
                let table = Arc::new(table);
                let index = DividendIndex::from_table(Arc::clone(&table))?;
                pool.dividend_index = Some(Arc::new(index));
                pool.daily.insert(dataset, table);
                continue;
            }
            if dataset == DatasetId::StockMainBusiness {
                let table = loader.load_stock_main_business_cached(
                    &columns,
                    context.start_date,
                    context.end_date,
                    financial_quarters.unwrap_or(8),
                    disclosure_cache,
                )?;
                let table = Arc::new(table);
                let index = MainBusinessIndex::from_table(Arc::clone(&table))?;
                pool.main_business_index = Some(Arc::new(index));
                pool.daily.insert(dataset, table);
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
                    let table = loader.load_daily_by_dates(dataset, &columns, &load_dates)?;
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
            stock_universe_panel: self
                .stock_universe_panel
                .as_ref()
                .map(|panel| panel.with_target_dates(target_dates)),
            financial_pit_indexes: self.financial_pit_indexes.clone(),
            dividend_index: self.dividend_index.clone(),
            main_business_index: self.main_business_index.clone(),
            financial_context: self.financial_context.clone(),
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
        let selected = selected_dates.iter().copied().collect::<BTreeSet<_>>();
        Self {
            daily: self
                .daily
                .iter()
                .map(|(dataset, table)| {
                    let table = if should_build_daily_panel(*dataset) {
                        table_slice_dates_or_clone(table, &selected)
                    } else {
                        Arc::clone(table)
                    };
                    (*dataset, table)
                })
                .collect(),
            daily_panels: self
                .daily_panels
                .iter()
                .map(|(dataset, panel)| (*dataset, panel.slice_dates(selected_dates)))
                .collect(),
            stock_universe_panel: self
                .stock_universe_panel
                .as_ref()
                .map(|panel| panel.slice_dates(selected_dates)),
            financial_pit_indexes: self.financial_pit_indexes.clone(),
            dividend_index: self.dividend_index.clone(),
            main_business_index: self.main_business_index.clone(),
            financial_context: self.financial_context.clone(),
            index_daily: self
                .index_daily
                .iter()
                .map(|(ts_code, table)| {
                    let table = table_slice_dates_or_clone(table, &selected);
                    (ts_code.clone(), table)
                })
                .collect(),
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

    pub fn view_for_requests(&self, requests: &[DataRequest], context: &FactorContext) -> Self {
        let mut daily_panel_dates = HashMap::<DatasetId, BTreeSet<i32>>::new();
        let mut index_panel_dates = HashMap::<String, BTreeSet<i32>>::new();
        let needs_stock_universe_panel = requests
            .iter()
            .any(|request| is_financial_context_dataset(request.dataset));
        for request in requests {
            let dates = request.resolved_dates(context);
            if dates.is_empty() {
                continue;
            }
            if request.dataset == DatasetId::IndexDaily {
                if let Some(entity_id) = &request.entity_id {
                    index_panel_dates
                        .entry(entity_id.clone())
                        .or_default()
                        .extend(dates);
                }
                continue;
            }
            if should_build_daily_panel(request.dataset) {
                daily_panel_dates
                    .entry(request.dataset)
                    .or_default()
                    .extend(dates);
            }
        }

        Self {
            daily: self
                .daily
                .iter()
                .map(|(dataset, table)| {
                    let table = daily_panel_dates
                        .get(dataset)
                        .map(|dates| table_slice_dates_or_clone(table, dates))
                        .unwrap_or_else(|| Arc::clone(table));
                    (*dataset, table)
                })
                .collect(),
            daily_panels: self
                .daily_panels
                .iter()
                .map(|(dataset, panel)| {
                    let panel = daily_panel_dates
                        .get(dataset)
                        .map(|dates| {
                            let dates = dates.iter().copied().collect::<Vec<_>>();
                            panel
                                .slice_dates(&dates)
                                .with_target_dates(&context.target_dates)
                        })
                        .unwrap_or_else(|| panel.clone());
                    (*dataset, panel)
                })
                .collect(),
            stock_universe_panel: self.stock_universe_panel.as_ref().map(|panel| {
                if needs_stock_universe_panel {
                    let dates = if context.load_dates.is_empty() {
                        context.target_dates.clone()
                    } else {
                        context.load_dates.clone()
                    };
                    panel
                        .slice_dates(&dates)
                        .with_target_dates(&context.target_dates)
                } else {
                    panel.clone()
                }
            }),
            financial_pit_indexes: self.financial_pit_indexes.clone(),
            dividend_index: self.dividend_index.clone(),
            main_business_index: self.main_business_index.clone(),
            financial_context: self.financial_context.clone(),
            index_daily: self
                .index_daily
                .iter()
                .map(|(ts_code, table)| {
                    let table = index_panel_dates
                        .get(ts_code)
                        .map(|dates| table_slice_dates_or_clone(table, dates))
                        .unwrap_or_else(|| Arc::clone(table));
                    (ts_code.clone(), table)
                })
                .collect(),
            index_daily_panels: self
                .index_daily_panels
                .iter()
                .map(|(ts_code, panel)| {
                    let panel = index_panel_dates
                        .get(ts_code)
                        .map(|dates| {
                            let dates = dates.iter().copied().collect::<Vec<_>>();
                            panel
                                .slice_dates(&dates)
                                .with_target_dates(&context.target_dates)
                        })
                        .unwrap_or_else(|| panel.clone());
                    (ts_code.clone(), panel)
                })
                .collect(),
            minute: self.minute.clone(),
            intraday_daily_raw: self.intraday_daily_raw.clone(),
            intraday_daily_raw_panel: self.intraday_daily_raw_panel.clone(),
        }
    }

    pub fn daily(&self, dataset: DatasetId) -> Result<&Table> {
        if dataset == DatasetId::StockAnalystReport {
            if let Some(table) = self
                .financial_context
                .as_ref()
                .and_then(|context| context.analyst_report())
            {
                return Ok(table);
            }
        }
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

    pub fn stock_universe_panel(&self) -> Result<&DailyPanel> {
        self.stock_universe_panel
            .as_ref()
            .ok_or_else(|| err("stock universe panel not loaded"))
    }

    pub fn financial_reader(
        &self,
        dataset: DatasetId,
        preference: ReportTypePreference,
    ) -> Result<FinancialPitReader<'_>> {
        if let Some(reader) = self
            .financial_context
            .as_ref()
            .and_then(|context| context.financial_reader(dataset, preference.clone()))
        {
            return Ok(reader);
        }
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

    pub fn dividend_reader(&self) -> Result<DividendReader<'_>> {
        if let Some(reader) = self
            .financial_context
            .as_ref()
            .and_then(|context| context.dividend_reader())
        {
            return Ok(reader);
        }
        self.dividend_index
            .as_ref()
            .map(|index| index.reader())
            .ok_or_else(|| err("dividend index not loaded"))
    }

    pub fn main_business_reader(&self) -> Result<MainBusinessReader<'_>> {
        if let Some(reader) = self
            .financial_context
            .as_ref()
            .and_then(|context| context.main_business_reader())
        {
            return Ok(reader);
        }
        self.main_business_index
            .as_ref()
            .map(|index| index.reader())
            .ok_or_else(|| err("main business index not loaded"))
    }

    pub fn loaded_table_row_count(&self) -> usize {
        let daily_rows = self.daily.values().map(|table| table.len).sum::<usize>();
        let index_rows = self
            .index_daily
            .values()
            .map(|table| table.len)
            .sum::<usize>();
        let minute_rows = self.minute.values().map(|table| table.len).sum::<usize>();
        let raw_rows = self
            .intraday_daily_raw
            .as_ref()
            .map_or(0, |table| table.len);
        daily_rows + index_rows + minute_rows + raw_rows
    }

    pub fn indexed_row_count(&self) -> usize {
        let financial_rows = self
            .financial_pit_indexes
            .values()
            .map(|index| index.len())
            .sum::<usize>();
        let dividend_rows = self.dividend_index.as_ref().map_or(0, |index| index.len());
        let main_business_rows = self
            .main_business_index
            .as_ref()
            .map_or(0, |index| index.len());
        let context_rows = self
            .financial_context
            .as_ref()
            .map_or(0, |context| context.indexed_row_count());
        financial_rows + dividend_rows + main_business_rows + context_rows
    }

    pub fn financial_context_profile(&self) -> Option<FinancialBatchProfile> {
        self.financial_context
            .as_ref()
            .map(|context| context.profile())
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
        if other.stock_universe_panel.is_some() {
            self.stock_universe_panel = other.stock_universe_panel;
        }
        self.financial_pit_indexes
            .extend(other.financial_pit_indexes);
        if other.dividend_index.is_some() {
            self.dividend_index = other.dividend_index;
        }
        if other.main_business_index.is_some() {
            self.main_business_index = other.main_business_index;
        }
        if other.financial_context.is_some() {
            self.financial_context = other.financial_context;
        }
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
            stock_universe_panel: None,
            financial_pit_indexes: HashMap::new(),
            dividend_index: None,
            main_business_index: None,
            financial_context: None,
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
        let stock_universe_panel = daily
            .get(&DatasetId::StockBasic)
            .map(|table| DailyPanel::from_stock_basic(table, context))
            .transpose()?;
        for (dataset, table) in &daily {
            if should_build_daily_panel(*dataset) {
                daily_panels.insert(*dataset, DailyPanel::from_table(table, context)?);
            }
        }
        let mut daily_arc = HashMap::new();
        let mut financial_pit_indexes = HashMap::new();
        let mut dividend_index = None;
        let mut main_business_index = None;
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
            if dataset == DatasetId::StockDividend {
                dividend_index = Some(Arc::new(DividendIndex::from_table(Arc::clone(&table))?));
            }
            if dataset == DatasetId::StockMainBusiness {
                main_business_index =
                    Some(Arc::new(MainBusinessIndex::from_table(Arc::clone(&table))?));
            }
            daily_arc.insert(dataset, table);
        }
        Ok(Self {
            daily: daily_arc,
            daily_panels,
            stock_universe_panel,
            financial_pit_indexes,
            dividend_index,
            main_business_index,
            financial_context: None,
            index_daily: HashMap::new(),
            index_daily_panels: HashMap::new(),
            minute: HashMap::new(),
            intraday_daily_raw: None,
            intraday_daily_raw_panel: None,
        })
    }
}

fn table_slice_dates_or_clone(table: &Arc<Table>, selected_dates: &BTreeSet<i32>) -> Arc<Table> {
    let Ok(trade_dates) = table.required_i32("trade_date") else {
        return Arc::clone(table);
    };
    let indices = trade_dates
        .iter()
        .enumerate()
        .filter_map(|(idx, date)| {
            date.and_then(|date| selected_dates.contains(&date).then_some(idx))
        })
        .collect::<Vec<_>>();
    Arc::new(
        table
            .take(&indices)
            .expect("date-filtered table keeps all columns at equal length"),
    )
}

fn stock_universe_columns() -> Vec<String> {
    [
        "list_status",
        "list_date",
        "delist_date",
        "exchange",
        "market",
    ]
    .iter()
    .map(|column| column.to_string())
    .collect()
}

fn should_build_daily_panel(dataset: DatasetId) -> bool {
    matches!(
        dataset,
        DatasetId::StockDailyPv
            | DatasetId::StockDailyBasic
            | DatasetId::StockDailyLimit
            | DatasetId::StockAdjFactor
            | DatasetId::StockMoneyflow
            | DatasetId::StockConsensus
            | DatasetId::StockBarraDaily
            | DatasetId::IndexDaily
            | DatasetId::FutureDaily
    )
}

fn is_financial_context_dataset(dataset: DatasetId) -> bool {
    matches!(
        dataset,
        DatasetId::StockIncome
            | DatasetId::StockBalanceSheet
            | DatasetId::StockCashFlow
            | DatasetId::StockDividend
            | DatasetId::StockMainBusiness
            | DatasetId::StockAnalystReport
    )
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};

    use crate::core::{AssetClass, FactorContext, FactorSpec, Lookback};
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

    fn sample_income_table() -> Table {
        Table::new(BTreeMap::from([
            (
                "ts_code".to_string(),
                ColumnData::Utf8(vec![Some("000001.SZ".to_string())]),
            ),
            (
                "ann_date".to_string(),
                ColumnData::I32(vec![Some(20260430)]),
            ),
            ("f_ann_date".to_string(), ColumnData::I32(vec![None])),
            (
                "end_date".to_string(),
                ColumnData::I32(vec![Some(20260331)]),
            ),
            ("report_type".to_string(), ColumnData::I64(vec![Some(2)])),
            ("update_flag".to_string(), ColumnData::I64(vec![Some(0)])),
            ("n_income".to_string(), ColumnData::F64(vec![Some(12.5)])),
        ]))
        .expect("valid income table")
    }

    fn sample_stock_basic_table() -> Table {
        Table::new(BTreeMap::from([
            (
                "ts_code".to_string(),
                ColumnData::Utf8(vec![
                    Some("000001.SZ".to_string()),
                    Some("000002.SZ".to_string()),
                    Some("AAPL.US".to_string()),
                ]),
            ),
            (
                "list_date".to_string(),
                ColumnData::I32(vec![Some(20260102), Some(20260101), Some(20260101)]),
            ),
            (
                "delist_date".to_string(),
                ColumnData::I32(vec![None, Some(20260102), None]),
            ),
        ]))
        .expect("valid stock basic table")
    }

    fn factor_spec(id: &str) -> FactorSpec {
        FactorSpec {
            id: id.to_string(),
            aliases: Vec::new(),
            name: id.to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "test".to_string(),
            tags: Vec::new(),
            description: id.to_string(),
            dependencies: Vec::new(),
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 0 },
        }
    }

    #[test]
    fn daily_panel_cache_exposes_prebuilt_panel() {
        let context = context();
        let table = sample_daily_table();
        let expected = DailyPanel::from_table(&table, &context).expect("panel");
        let pool = DataPool {
            daily: HashMap::from([(DatasetId::StockDailyPv, Arc::new(table))]),
            daily_panels: HashMap::from([(DatasetId::StockDailyPv, expected.clone())]),
            stock_universe_panel: None,
            financial_pit_indexes: HashMap::new(),
            dividend_index: None,
            main_business_index: None,
            financial_context: None,
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
    fn financial_context_supplies_reader_without_daily_raw_table() {
        let income_table = Arc::new(sample_income_table());
        let index = Arc::new(FinancialPitIndex::from_table(income_table).expect("pit index"));
        let context = Arc::new(FinancialBatchContext {
            financial_pit_indexes: HashMap::from([(DatasetId::StockIncome, index)]),
            dividend_index: None,
            main_business_index: None,
            analyst_report: None,
            profile: FinancialBatchProfile {
                pit_records: 1,
                builds: 1,
                ..Default::default()
            },
        });
        let pool = DataPool {
            daily: HashMap::new(),
            daily_panels: HashMap::new(),
            stock_universe_panel: None,
            financial_pit_indexes: HashMap::new(),
            dividend_index: None,
            main_business_index: None,
            financial_context: Some(context),
            index_daily: HashMap::new(),
            index_daily_panels: HashMap::new(),
            minute: HashMap::new(),
            intraday_daily_raw: None,
            intraday_daily_raw_panel: None,
        };

        assert!(pool.daily(DatasetId::StockIncome).is_err());
        let reader = pool
            .financial_reader(
                DatasetId::StockIncome,
                ReportTypePreference::income_single_quarter(),
            )
            .expect("reader from shared context");
        let record = reader
            .record_for_end_date("000001.SZ", 20260501, 20260331)
            .expect("visible PIT record");
        assert_eq!(record.column("n_income"), Some(12.5));
        assert_eq!(pool.indexed_row_count(), 1);
    }

    #[test]
    fn stock_universe_panel_is_built_from_stock_basic_in_tests() {
        let mut context = context();
        context.load_dates = vec![20260101, 20260102, 20260103];
        let pool = DataPool::from_daily_tables_for_test(
            HashMap::from([(DatasetId::StockBasic, sample_stock_basic_table())]),
            &context,
        )
        .expect("pool");

        let panel = pool.stock_universe_panel().expect("stock universe");

        assert_eq!(panel.dates(), &[20260101, 20260102, 20260103]);
        assert_eq!(
            panel.instruments(),
            &["000001.SZ".to_string(), "000002.SZ".to_string()]
        );
        assert_eq!(
            (0..panel.shape_len())
                .map(|offset| panel.is_present_offset(offset))
                .collect::<Vec<_>>(),
            vec![false, true, true, true, true, false]
        );
    }

    #[test]
    fn financial_view_slices_stock_universe_panel_to_requested_context_dates() {
        let mut base_context = context();
        base_context.load_dates = vec![20260101, 20260102, 20260103];
        let pool = DataPool::from_daily_tables_for_test(
            HashMap::from([(DatasetId::StockBasic, sample_stock_basic_table())]),
            &base_context,
        )
        .expect("pool");
        let request_context = FactorContext {
            load_dates: vec![20260102],
            target_dates: vec![20260102],
            ..base_context
        };

        let view = pool.view_for_requests(
            &[DataRequest::financial_quarters(
                DatasetId::StockIncome,
                &["n_income"],
                4,
            )],
            &request_context,
        );
        let panel = view.stock_universe_panel().expect("stock universe");

        assert_eq!(panel.dates(), &[20260102]);
        assert_eq!(
            panel.instruments(),
            &["000001.SZ".to_string(), "000002.SZ".to_string()]
        );
        assert!(panel.is_present_offset(0));
        assert!(panel.is_present_offset(1));
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

    #[test]
    fn view_for_requests_slices_only_requested_daily_panels() {
        let context = context();
        let pv_table = sample_daily_table();
        let basic_table = Table::new(BTreeMap::from([
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
                "total_mv".to_string(),
                ColumnData::F64(vec![Some(100.0), Some(101.0)]),
            ),
        ]))
        .expect("valid basic table");
        let pool = DataPool::from_daily_tables_for_test(
            HashMap::from([
                (DatasetId::StockDailyPv, pv_table),
                (DatasetId::StockDailyBasic, basic_table),
            ]),
            &context,
        )
        .expect("pool");

        let view = pool.view_for_requests(
            &[DataRequest::explicit_dates(
                DatasetId::StockDailyPv,
                &["close"],
                vec![20260102],
            )],
            &context,
        );

        assert_eq!(
            view.daily_panel(DatasetId::StockDailyPv)
                .expect("pv")
                .dates(),
            &[20260102]
        );
        assert_eq!(
            view.daily(DatasetId::StockDailyPv)
                .expect("pv table")
                .required_i32("trade_date")
                .expect("trade_date"),
            &vec![Some(20260102)]
        );
        assert_eq!(
            view.daily_panel(DatasetId::StockDailyBasic)
                .expect("basic")
                .dates(),
            &[20260101, 20260102]
        );
    }

    #[test]
    fn view_for_requests_keeps_daily_panel_output_on_context_target_dates() {
        let context = context();
        let pool = DataPool::from_daily_tables_for_test(
            HashMap::from([(DatasetId::StockDailyPv, sample_daily_table())]),
            &context,
        )
        .expect("pool");

        let view = pool.view_for_requests(
            &[DataRequest::new(DatasetId::StockDailyPv, &["close"])],
            &context,
        );
        let panel = view.daily_panel(DatasetId::StockDailyPv).expect("pv");
        assert_eq!(panel.dates(), &[20260101, 20260102]);

        let series = panel
            .column("close")
            .expect("close")
            .to_factor_series(factor_spec("test_close_factor"));
        let dates = series
            .values
            .iter()
            .map(|value| value.key.trade_date())
            .collect::<Vec<_>>();

        assert_eq!(dates, vec![20260102]);
        assert_eq!(series.values[0].value, Some(11.0));
    }
}
