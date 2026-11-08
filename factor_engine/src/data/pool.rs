use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use crate::core::{DataRequest, DatasetId, FactorContext, Frequency};
use crate::data::loader::{DisclosureTableCache, MarketDataLoader};
use crate::data::table::Table;
use crate::data::{SourceRows, SourceRowsReader, FundPortfolioBatchIndex, FundPortfolioReader};
use crate::error::{err, Result};
use crate::factor::common::{
    DailyPanel, DividendIndex, DividendReader, FinancialPitIndex, FinancialPitReader,
    MainBusinessIndex, MainBusinessReader, ReportTypePreference,
};
use crate::factor::future::minute_input::{FutureMinuteDayIndex, FutureMinuteField, FutureMinuteInput};

#[derive(Clone, Debug, Default)]
pub struct FinancialBatchProfile {
    pub source_rows: usize,
    pub source_columns: usize,
    pub pit_records: usize,
    pub main_business_records: usize,
    pub dividend_records: usize,
    pub analyst_records: usize,
    pub fund_portfolio_records: usize,
    pub builds: usize,
}

#[derive(Clone, Debug, Default)]
pub struct FinancialBatchContext {
    financial_pit_indexes: HashMap<DatasetId, Arc<FinancialPitIndex>>,
    dividend_index: Option<Arc<DividendIndex>>,
    main_business_index: Option<Arc<MainBusinessIndex>>,
    analyst_report: Option<Arc<SourceRows>>,
    fund_portfolio: Option<Arc<FundPortfolioBatchIndex>>,
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
        let mut fund_portfolio = None;
        let mut profile = FinancialBatchProfile {
            builds: 1,
            ..Default::default()
        };

        for (dataset, (columns, financial_quarters)) in grouped {
            let columns = columns.into_iter().collect::<Vec<_>>();
            match dataset {
                DatasetId::FundPortfolio => {
                    let sources = loader.load_fund_portfolio_sources_cached(&columns, context.load_start_date, context.end_date, disclosure_cache)?;
                    profile.source_rows += sources.iter().map(|table| table.len).sum::<usize>();
                    profile.source_columns += sources.iter().map(|table| table.columns.len()).sum::<usize>();
                    let partitions = sources.iter().map(|source| disclosure_cache.fund_index_for_source(source)).collect::<Result<Vec<_>>>()?;
                    let history = loader.fund_portfolio_history_files_before(context.load_start_date / 10000)?;
                    let index = Arc::new(FundPortfolioBatchIndex::new(partitions, history, context.end_date));
                    profile.fund_portfolio_records = index.len();
                    fund_portfolio = Some(index);
                }
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
                    let partitions = sources
                        .iter()
                        .map(|source| disclosure_cache.financial_index_for_source(source))
                        .collect::<Result<Vec<_>>>()?;
                    let index = Arc::new(FinancialPitIndex::from_partitions(
                        partitions,
                        context.end_date,
                    ));
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
                    let sources = loader.load_stock_main_business_sources_cached(
                        &columns,
                        context.start_date,
                        context.end_date,
                        financial_quarters.unwrap_or(8),
                        disclosure_cache,
                    )?;
                    profile.source_rows += sources.iter().map(|table| table.len).sum::<usize>();
                    profile.source_columns += sources
                        .iter()
                        .map(|table| table.columns.len())
                        .sum::<usize>();
                    let index = Arc::new(MainBusinessIndex::from_source_tables(
                        sources,
                        context.end_date,
                    )?);
                    profile.main_business_records = index.len();
                    main_business_index = Some(index);
                }
                DatasetId::StockAnalystReport => {
                    let sources = loader.load_stock_analyst_sources_cached(
                        &columns,
                        context.load_start_date,
                        context.end_date,
                        disclosure_cache,
                    )?;
                    profile.source_rows += sources.iter().map(|table| table.len).sum::<usize>();
                    profile.source_columns += sources
                        .iter()
                        .map(|table| table.columns.len())
                        .sum::<usize>();
                    let rows = SourceRows::new(sources, "report_date", i32::MIN, context.end_date)?;
                    profile.analyst_records = rows.len();
                    analyst_report = Some(Arc::new(rows));
                }
                _ => {}
            }
        }

        Ok(Some(Arc::new(Self {
            financial_pit_indexes,
            dividend_index,
            main_business_index,
            analyst_report,
            fund_portfolio,
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

    pub fn analyst_report(&self) -> Option<&SourceRows> {
        self.analyst_report.as_deref()
    }

    pub fn profile(&self) -> FinancialBatchProfile {
        self.profile.clone()
    }

    pub fn fund_portfolio_reader(&self) -> Option<FundPortfolioReader<'_>> {
        self.fund_portfolio.as_ref().map(|index| index.reader())
    }

    pub fn indexed_row_count(&self) -> usize {
        self.profile.pit_records
            + self.profile.main_business_records
            + self.profile.dividend_records
            + self.profile.fund_portfolio_records
    }
}

#[derive(Clone, Debug, Default)]
pub struct DataPool {
    trading_calendar: Option<Arc<crate::calendar::TradingCalendar>>,
    future_minute_input: Option<Arc<FutureMinuteInput>>,
    future_minute_coverage: Option<Arc<(BTreeSet<i32>, BTreeSet<String>)>>,
    future_minute_days: Option<Arc<HashMap<i32, FutureMinuteDayIndex>>>,
    daily: HashMap<DatasetId, Arc<Table>>,
    daily_panels: HashMap<DatasetId, DailyPanel>,
    stock_universe_panel: Option<DailyPanel>,
    financial_pit_indexes: HashMap<DatasetId, Arc<FinancialPitIndex>>,
    dividend_index: Option<Arc<DividendIndex>>,
    main_business_index: Option<Arc<MainBusinessIndex>>,
    financial_context: Option<Arc<FinancialBatchContext>>,
    index_daily: HashMap<String, Arc<Table>>,
    index_membership: HashMap<String, Arc<crate::data::loader::IndexMembership>>,
    index_daily_panels: HashMap<String, DailyPanel>,
    minute: HashMap<(DatasetId, Option<usize>, i32), Arc<Table>>,
    intraday_daily_raw: Option<Arc<Table>>,
    intraday_daily_raw_panel: Option<DailyPanel>,
}

impl DataPool {
    pub(crate) fn load_future_minute_batch(
        &mut self,
        loader: &MarketDataLoader,
        requests: &[DataRequest],
        context: &FactorContext,
        input: Arc<FutureMinuteInput>,
        lookback: usize,
    ) -> Result<()> {
        let mut dates = BTreeSet::new();
        let mut columns = BTreeSet::new();
        let mut same_contract_lookback = 0;
        for request in requests.iter().filter(|request| request.dataset == DatasetId::FutureMinute1m) {
            let resolved = request.resolved_dates(context);
            if request.columns.iter().any(|name| FutureMinuteField::contract_source(name).is_some()) {
                let first_target = context.target_dates.first().copied().unwrap_or(context.start_date);
                same_contract_lookback = same_contract_lookback.max(
                    resolved.iter().filter(|&&date| date < first_target).count(),
                );
            }
            dates.extend(resolved);
            columns.extend(request.columns.iter().cloned());
        }
        let diff_columns = columns
            .iter()
            .filter_map(|name| FutureMinuteField::diff_source(name))
            .map(str::to_string)
            .collect::<Vec<_>>();
        let projection = columns
            .iter()
            .map(|name| {
                if name == FutureMinuteField::VWAP_NOTIONAL {
                    "amount"
                } else {
                    FutureMinuteField::diff_source(name)
                        .or_else(|| FutureMinuteField::contract_source(name)).unwrap_or(name)
                }.to_string()
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let targets = context.target_dates.iter().copied().collect::<BTreeSet<_>>();
        let plans = context.load_dates.iter().enumerate()
            .map(|(index, &date)| {
                let later_targets = |days| context.load_dates.iter()
                    .skip(index + 1)
                    .take(days)
                    .copied()
                    .filter(|date| targets.contains(date));
                let target = targets.contains(&date);
                let mut codes = input.full_contracts(date, target, later_targets(lookback), false)?;
                if same_contract_lookback > 0 {
                    codes.extend(input.full_contracts(date, target, later_targets(same_contract_lookback), true)?);
                }
                Some(codes)
            })
            .collect::<Vec<_>>();
        let mut day_indexes = HashMap::new();
        for (index, &date) in context.load_dates.iter().enumerate() {
            if !dates.contains(&date) {
                continue;
            }
            let diff_columns = if context.load_dates.get(index + 1).is_some_and(|next| dates.contains(next)) {
                diff_columns.as_slice()
            } else {
                &[]
            };
            if let Some((table, day_tails)) = loader.load_future_minute_day(
                date,
                &projection,
                plans[index].as_ref(),
                plans.get(index + 1).and_then(Option::as_ref),
                diff_columns,
            )? {
                let day_index = FutureMinuteDayIndex::build(date, &table, diff_columns, day_tails)?;
                day_indexes.insert(date, day_index);
                self.minute.insert((DatasetId::FutureMinute1m, None, date), Arc::new(table));
            }
        }
        self.future_minute_input = Some(input);
        self.future_minute_coverage = Some(Arc::new((dates, columns)));
        self.future_minute_days = Some(Arc::new(day_indexes));
        Ok(())
    }

    pub(crate) fn load_reused_batch_inputs(
        loader: &MarketDataLoader, requests: &[DataRequest], request_batches: &[Vec<DataRequest>],
        context: &FactorContext, disclosure_cache: &mut DisclosureTableCache,
        financial_context: Option<Arc<FinancialBatchContext>>,
    ) -> Result<Self> {
        let mut counts = HashMap::<DatasetId, usize>::new();
        for batch in request_batches {
            for dataset in batch.iter().map(|request| request.dataset).collect::<BTreeSet<_>>() {
                *counts.entry(dataset).or_default() += 1;
            }
        }
        let requests = requests.iter().filter(|request| is_financial_context_dataset(request.dataset)
            || counts.get(&request.dataset).copied().unwrap_or(0) > 1).cloned().collect::<Vec<_>>();
        Self::load_batch_inputs(loader, &requests, context, disclosure_cache, financial_context)
    }

    /// Physical date-batch inputs. Financial sources remain in their shared PIT context;
    /// stock minute tables retain the existing bounded streaming path.
    pub fn load_batch_inputs(
        loader: &MarketDataLoader,
        requests: &[DataRequest],
        context: &FactorContext,
        disclosure_cache: &mut DisclosureTableCache,
        financial_context: Option<Arc<FinancialBatchContext>>,
    ) -> Result<Self> {
        let shared = requests.iter().filter(|request| request.dataset.frequency() == Frequency::Daily
            || (context.frequency == Frequency::Minute1 && request.dataset == DatasetId::FutureMinute1m))
            .cloned().collect::<Vec<_>>();
        Self::load_inputs(loader, &shared, context, disclosure_cache, financial_context, true)
    }

    pub fn load_from_batch_inputs(
        &self,
        loader: &MarketDataLoader,
        requests: &[DataRequest],
        context: &FactorContext,
        disclosure_cache: &mut DisclosureTableCache,
    ) -> Result<Self> {
        let mut pool = self.view_for_requests(requests, context);
        for request in requests {
            let dataset = request.dataset;
            if should_build_daily_panel(dataset) && dataset != DatasetId::IndexDaily
                && !pool.daily_panels.contains_key(&dataset) {
                if let Some(table) = pool.daily.get(&dataset) {
                    let columns = requests.iter().filter(|r| r.dataset == dataset)
                        .flat_map(|r| r.columns.iter().cloned()).collect();
                    pool.daily_panels.insert(dataset, DailyPanel::from_table_columns(table, context, &columns)?);
                }
            }
        }
        // A physical union must not add instruments which only exist on other providers' dates.
        for panel in pool.daily_panels.values_mut() { *panel = panel.compact_instruments(); }
        for panel in pool.index_daily_panels.values_mut() { *panel = panel.compact_instruments(); }
        let uncached = requests.iter().filter(|request| !is_financial_context_dataset(request.dataset)
            && if request.dataset == DatasetId::FutureMinute1m && context.frequency == Frequency::Minute1 {
                !self.future_minute_coverage.as_ref().is_some_and(|coverage| {
                    let (dates, columns) = coverage.as_ref();
                    request.columns.iter().all(|column| columns.contains(column))
                        && request.resolved_dates(context).iter().all(|date| dates.contains(date))
                })
            } else if request.dataset == DatasetId::IndexDaily {
                request.entity_id.as_ref().is_none_or(|id| !self.index_daily_panels.contains_key(id))
            } else if request.dataset == DatasetId::IndexMonthlyWeight {
                request.entity_id.as_ref().is_none_or(|id| !pool.index_membership.contains_key(id))
            } else {
                !self.daily.get(&request.dataset).is_some_and(|table| request.columns.iter().all(|name| table.columns.contains_key(name)))
                    && !self.daily_panels.get(&request.dataset).is_some_and(|panel| request.columns.iter().all(|name| panel.has_column(name)))
            })
            .cloned().collect::<Vec<_>>();
        if !uncached.is_empty() {
            let datasets = uncached.iter().map(|request| request.dataset).collect::<BTreeSet<_>>();
            let complete = requests.iter().filter(|request| datasets.contains(&request.dataset)).cloned().collect::<Vec<_>>();
            pool.extend(Self::load_with_financial_context(loader, &complete, context,
                disclosure_cache, self.financial_context.clone())?);
        }
        Ok(pool)
    }

    pub fn set_trading_calendar(&mut self, calendar: Arc<crate::calendar::TradingCalendar>) {
        self.trading_calendar = Some(calendar);
    }

    pub(crate) fn share_intraday_daily_raw(&mut self, source: &Self, context: &FactorContext) {
        let dates = context.load_dates.iter().copied().collect();
        self.intraday_daily_raw = source.intraday_daily_raw.as_ref()
            .map(|table| table_slice_dates_or_clone(table, &dates));
        self.intraday_daily_raw_panel = source.intraday_daily_raw_panel.as_ref().map(|panel|
            panel.slice_dates(&context.load_dates).with_target_dates(&context.target_dates).compact_instruments());
    }

    pub fn trading_calendar(&self) -> Result<&crate::calendar::TradingCalendar> {
        self.trading_calendar
            .as_deref()
            .ok_or_else(|| err("trading calendar not attached to data pool"))
    }
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
        Self::load_inputs(loader, requests, context, disclosure_cache, financial_context, false)
    }

    fn load_inputs(
        loader: &MarketDataLoader,
        requests: &[DataRequest],
        context: &FactorContext,
        disclosure_cache: &mut DisclosureTableCache,
        financial_context: Option<Arc<FinancialBatchContext>>,
        defer_raw_panels: bool,
    ) -> Result<Self> {
        let financial_context = match financial_context {
            Some(context) => Some(context),
            None => FinancialBatchContext::build(loader, requests, context, disclosure_cache)?,
        };
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
        let panel_only = panel_only_datasets(requests);
        pool.financial_context = financial_context;
        if grouped
            .keys()
            .any(|(dataset, _, _)| needs_stock_universe(*dataset))
        {
            let table = loader.load_stock_basic(&stock_universe_columns())?;
            pool.stock_universe_panel = Some(DailyPanel::from_stock_basic(&table, context)?);
            pool.daily
                .entry(DatasetId::StockBasic)
                .or_insert_with(|| Arc::new(table));
        }
        for ((dataset, entity_id, bar_size), (columns, financial_quarters, load_dates)) in grouped
        {
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
            if dataset == DatasetId::IndexMonthlyWeight {
                let code = entity_id.ok_or_else(|| err("index.monthly_weight requires an index code"))?;
                let members = loader.load_index_monthly_members(
                    &code, context.load_start_date, context.end_date, financial_quarters.unwrap_or(0),
                )?;
                pool.index_membership.insert(code, Arc::new(members));
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
            if dataset == DatasetId::FundBasic {
                let table = loader.load_fund_basic(&columns)?;
                pool.daily.insert(dataset, Arc::new(table));
                continue;
            }
            if dataset == DatasetId::StockBarraDaily {
                let table =
                    loader.load_barra_daily(context.asset_class, "CNE6", &columns, &load_dates)?;
                if !defer_raw_panels || panel_only.contains(&dataset) {
                    let panel = DailyPanel::from_table(&table, context)?;
                    pool.daily_panels.insert(dataset, panel);
                }
                if !panel_only.contains(&dataset) {
                    pool.daily.insert(dataset, Arc::new(table));
                }
                continue;
            }
            match dataset.frequency() {
                Frequency::Daily => {
                    let table = loader.load_daily_by_dates(dataset, &columns, &load_dates)?;
                    let row_requests = requests.iter().filter(|request| request.dataset == dataset && !request.panel_only).collect::<Vec<_>>();
                    let row_dates = row_requests.iter().flat_map(|request| request.resolved_dates(context)).collect::<BTreeSet<_>>();
                    let row_columns = row_requests.iter().flat_map(|request| request.columns.iter().cloned()).collect::<BTreeSet<_>>();
                    let narrow_rows = row_dates.len() < load_dates.len() || row_columns.len() < columns.len();
                    if should_build_daily_panel(dataset) && (!defer_raw_panels || panel_only.contains(&dataset) || narrow_rows) {
                        let panel = DailyPanel::from_table(&table, context)?;
                        pool.daily_panels.insert(dataset, panel);
                    }
                    if !panel_only.contains(&dataset) {
                        let table = if defer_raw_panels && should_build_daily_panel(dataset) && narrow_rows {
                            project_row_inputs(table, &row_dates, &row_columns)?
                        } else { table };
                        pool.daily.insert(dataset, Arc::new(table));
                    }
                }
                Frequency::Minute1 => {
                    let minute_dates = if context.frequency == Frequency::Daily {
                        &context.load_dates
                    } else {
                        &load_dates
                    };
                    let tables = if dataset == DatasetId::StockDerivedBar {
                        let bar_size = bar_size
                            .ok_or_else(|| err("stock.derived.bar request requires bar_size"))?;
                        loader.load_stock_derived_bar_by_date(bar_size, &columns, minute_dates)?
                    } else {
                        loader.load_minute_by_date(dataset, &columns, minute_dates)?
                    };
                    for (date, table) in tables {
                        pool.minute
                            .insert((dataset, bar_size, date), Arc::new(table));
                    }
                    if dataset == DatasetId::FutureMinute1m && context.frequency == Frequency::Minute1 {
                        pool.future_minute_coverage = Some(Arc::new((
                            load_dates.into_iter().collect(),
                            columns.into_iter().collect(),
                        )));
                    }
                }
            }
        }
        Ok(pool)
    }

    pub fn with_target_dates(&self, target_dates: &[i32]) -> Self {
        Self {
            trading_calendar: self.trading_calendar.clone(),
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
            index_membership: self.index_membership.clone(),
            index_daily_panels: self
                .index_daily_panels
                .iter()
                .map(|(ts_code, panel)| (ts_code.clone(), panel.with_target_dates(target_dates)))
                .collect(),
            minute: self.minute.clone(),
            future_minute_input: self.future_minute_input.clone(),
            future_minute_coverage: self.future_minute_coverage.clone(),
            future_minute_days: self.future_minute_days.clone(),
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
            trading_calendar: self.trading_calendar.clone(),
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
            index_membership: self.index_membership.clone(),
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
            future_minute_input: self.future_minute_input.clone(),
            future_minute_coverage: self.future_minute_coverage.clone(),
            future_minute_days: self.future_minute_days.clone(),
            intraday_daily_raw: self.intraday_daily_raw.clone(),
            intraday_daily_raw_panel: self
                .intraday_daily_raw_panel
                .as_ref()
                .map(|panel| panel.slice_dates(selected_dates)),
        }
    }

    pub fn view_for_requests(&self, requests: &[DataRequest], context: &FactorContext) -> Self {
        let panel_only = panel_only_datasets(requests);
        let mut panel_columns: HashMap<DatasetId, BTreeSet<String>> = HashMap::new();
        for request in requests {
            panel_columns
                .entry(request.dataset)
                .or_default()
                .extend(request.columns.iter().cloned());
        }
        let mut daily_panel_dates = HashMap::<DatasetId, BTreeSet<i32>>::new();
        let mut index_panel_dates = HashMap::<String, BTreeSet<i32>>::new();
        let needs_stock_universe_panel = requests
            .iter()
            .any(|request| needs_stock_universe(request.dataset));
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
            trading_calendar: self.trading_calendar.clone(),
            daily: self
                .daily
                .iter()
                .filter(|(dataset, _)| panel_columns.contains_key(dataset)
                    || (**dataset == DatasetId::StockBasic && needs_stock_universe_panel))
                .filter(|(dataset, _)| !panel_only.contains(dataset))
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
                .filter(|(dataset, _)| panel_columns.contains_key(dataset))
                .map(|(dataset, panel)| {
                    let panel = panel.select_columns(&panel_columns[dataset]);
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
            index_membership: self.index_membership.iter().filter(|(code, _)| {
                requests.iter().any(|r| r.dataset == DatasetId::IndexMonthlyWeight
                    && r.entity_id.as_deref() == Some(code.as_str()))
            }).map(|(code, members)| (code.clone(), Arc::clone(members))).collect(),
            index_daily: self
                .index_daily
                .iter()
                .filter(|(ts_code, _)| index_panel_dates.contains_key(*ts_code))
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
                .filter(|(ts_code, _)| index_panel_dates.contains_key(*ts_code))
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
            future_minute_input: self.future_minute_input.clone(),
            future_minute_coverage: self.future_minute_coverage.clone(),
            future_minute_days: self.future_minute_days.clone(),
            intraday_daily_raw: self.intraday_daily_raw.clone(),
            intraday_daily_raw_panel: self.intraday_daily_raw_panel.clone(),
        }
    }

    pub fn analyst_report_reader(&self) -> Result<SourceRowsReader<'_>> {
        if let Some(rows) = self
            .financial_context
            .as_ref()
            .and_then(|context| context.analyst_report())
        {
            return Ok(rows.reader());
        }
        Ok(self.daily(DatasetId::StockAnalystReport)?.into())
    }

    pub fn fund_portfolio_reader(&self) -> Result<FundPortfolioReader<'_>> {
        self.financial_context.as_ref().and_then(|context| context.fund_portfolio_reader())
            .ok_or_else(|| err("fund portfolio index not loaded"))
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
        let panel_rows = self
            .daily_panels
            .iter()
            .filter(|(dataset, _)| !self.daily.contains_key(dataset))
            .map(|(_, panel)| panel.present_row_count())
            .sum::<usize>();
        let index_rows = self
            .index_daily
            .values()
            .map(|table| table.len)
            .sum::<usize>();
        let minute_rows = self.minute.values().map(|table| table.len).sum::<usize>();
        let raw_rows = self
            .intraday_daily_raw
            .as_ref()
            .map(|table| table.len)
            .unwrap_or_else(|| {
                self.intraday_daily_raw_panel
                    .as_ref()
                    .map_or(0, DailyPanel::present_row_count)
            });
        daily_rows + panel_rows + index_rows + minute_rows + raw_rows
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

    pub(crate) fn index_monthly_members(&self, code: &str) -> Result<&crate::data::loader::IndexMembership> {
        self.index_membership.get(code).map(Arc::as_ref)
            .ok_or_else(|| err(format!("index membership not loaded: {code}")))
    }

    pub(crate) fn future_minute_input(&self) -> Option<&FutureMinuteInput> {
        self.future_minute_input.as_deref()
    }

    pub(crate) fn future_minute_day_index(&self, date: i32) -> Option<&FutureMinuteDayIndex> {
        self.future_minute_days.as_ref()?.get(&date)
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


    pub fn extend(&mut self, other: Self) {
        if other.trading_calendar.is_some() {
            self.trading_calendar = other.trading_calendar;
        }
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
        self.index_membership.extend(other.index_membership);
        self.index_daily_panels.extend(other.index_daily_panels);
        self.minute.extend(other.minute);
        if other.future_minute_coverage.is_some() {
            self.future_minute_coverage = other.future_minute_coverage;
        }
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

    /// Consume the intermediate row table when all consumers use the daily panel.
    pub fn set_intraday_daily_raw_panel(&mut self, table: Table, context: &FactorContext) -> Result<()> {
        self.intraday_daily_raw_panel = Some(DailyPanel::from_table(&table, context)?);
        self.intraday_daily_raw = None;
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


}

fn project_row_inputs(mut table: Table, dates: &BTreeSet<i32>, columns: &BTreeSet<String>) -> Result<Table> {
    let selected = table.required_i32("trade_date")?.iter().enumerate()
        .filter_map(|(index, date)| date.is_some_and(|date| dates.contains(&date)).then_some(index))
        .collect::<Vec<_>>();
    table.columns.retain(|name, _| columns.contains(name) || matches!(name.as_str(), "trade_date" | "ts_code"));
    if selected.len() == table.len { return Ok(table); }
    table.take(&selected)
}

fn panel_only_datasets(requests: &[DataRequest]) -> BTreeSet<DatasetId> {
    let mut panel_only: BTreeSet<_> = requests
        .iter()
        .filter(|r| r.panel_only && should_build_daily_panel(r.dataset))
        .map(|r| r.dataset)
        .collect();
    for request in requests.iter().filter(|r| !r.panel_only) {
        panel_only.remove(&request.dataset);
    }
    panel_only
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
    if indices.len() == table.len {
        return Arc::clone(table);
    }
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
            | DatasetId::FundPortfolio
    )
}

pub(crate) fn needs_stock_universe(dataset: DatasetId) -> bool {
    is_financial_context_dataset(dataset)
        || matches!(dataset, DatasetId::StockConsensus | DatasetId::StockBasic)
}
