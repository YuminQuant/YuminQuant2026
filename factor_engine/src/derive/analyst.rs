use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::calendar::TradingCalendar;
use crate::config::EngineConfig;
use crate::core::DatasetId;
use crate::data::loader::{DisclosureTableCache, MarketDataLoader};
use crate::data::parquet_io::write_parquet;
use crate::data::{ColumnData, DataCatalog, Table};
use crate::error::{err, Result};
use crate::factor::common::{FinancialPitIndex, FinancialPitReader, ReportTypePreference};
use crate::progress::ProgressBar;

pub const DEFAULT_CONSENSUS_DATE_BATCH_SIZE: usize = 20;
const ANALYST_WARMUP_DAYS: i32 = 600;
const FORECAST_STRICT_DAYS: i32 = 90;
const FORECAST_LOOSE_DAYS: i32 = 120;
const FORECAST_CARRY_DAYS: i32 = 183;
const STRICT_FORECAST_INSTITUTIONS: usize = 6;
const STRICT_RATING_TARGET_INSTITUTIONS: usize = 4;
const EPS: f64 = 1e-12;

#[derive(Clone, Debug)]
pub struct AnalystConsensusRequest {
    pub start_date: i32,
    pub end_date: i32,
    pub overwrite: bool,
    pub date_batch_size: usize,
    pub project_config_path: Option<PathBuf>,
}

#[derive(Clone, Debug, Default)]
pub struct AnalystConsensusReport {
    pub output_files: Vec<PathBuf>,
    pub processed_dates: usize,
    pub skipped_existing_dates: Vec<i32>,
    pub total_rows: usize,
}

#[derive(Clone, Debug)]
pub struct ConsensusOutput {
    output_path: PathBuf,
    rows: usize,
}

pub fn consensus_output_path(data_root: &Path, trade_date: i32) -> PathBuf {
    data_root
        .join("derived")
        .join("stock")
        .join("consensus")
        .join(format!("{trade_date}.parquet"))
}

pub fn derive_analyst_consensus(
    config: &EngineConfig,
    request: &AnalystConsensusRequest,
) -> Result<AnalystConsensusReport> {
    ensure_supported_request(request)?;
    let calendar = TradingCalendar::load(&config.data_root, &config.stock_calendar_exchange)?;
    let target_dates = calendar.open_dates_between(request.start_date, request.end_date);
    let catalog = DataCatalog::new(config.data_root.clone())
        .with_stock_sw_classification_path(config.stock_sw_classification_path.clone())
        .with_stock_ci_classification_path(config.stock_ci_classification_path.clone());
    let loader = MarketDataLoader::new(catalog);
    let progress = ProgressBar::new("derive-consensus", target_dates.len(), true);
    let mut report = AnalystConsensusReport::default();
    let mut disclosure_cache = DisclosureTableCache::default();

    for date_batch in target_dates.chunks(request.date_batch_size) {
        let Some(&batch_start) = date_batch.first() else {
            continue;
        };
        let Some(&batch_end) = date_batch.last() else {
            continue;
        };
        let warmup_start = add_days(batch_start, -ANALYST_WARMUP_DAYS);
        let calc_dates = calendar.open_dates_between(warmup_start, batch_end);
        let batch_target_set = date_batch.iter().copied().collect::<BTreeSet<_>>();
        let np_history_dates = np_history_dates_for_batch(&calendar, date_batch);
        let market = DailyMarketData::load(&loader, date_batch)?;
        let analyst_rows =
            AnalystRows::load(&loader, warmup_start, batch_end, &mut disclosure_cache)?;
        let financial =
            ConsensusFinancialData::load(&loader, warmup_start, batch_end, &mut disclosure_cache)?;
        let mut state = AnalystConsensusState::default();
        for trade_date in calc_dates {
            state.ingest_until(trade_date, &analyst_rows);
            let output = if batch_target_set.contains(&trade_date) {
                let output_path = consensus_output_path(&config.data_root, trade_date);
                if output_path.exists() && !request.overwrite {
                    report.skipped_existing_dates.push(trade_date);
                    progress.tick(format!("date={trade_date} skipped existing"));
                    None
                } else {
                    let table =
                        build_consensus_table_for_date(trade_date, &market, &financial, &state)?;
                    let rows = table.len;
                    write_parquet(&output_path, &table)?;
                    progress.tick(format!("date={trade_date} rows={rows}"));
                    Some(ConsensusOutput { output_path, rows })
                }
            } else {
                None
            };

            if np_history_dates.contains(&trade_date) {
                let np_fy0 = state
                    .active_stocks()
                    .filter_map(|ts_code| {
                        let years = fiscal_years(trade_date);
                        let value = compute_annual_base(
                            ts_code,
                            trade_date,
                            years[0],
                            BaseMetric::NetProfit,
                            &state,
                            &financial,
                        )
                        .value?;
                        Some((ts_code.to_string(), value))
                    })
                    .collect::<HashMap<_, _>>();
                state.remember_np_fy0(trade_date, np_fy0);
            }

            if let Some(output) = output {
                report.total_rows += output.rows;
                report.processed_dates += 1;
                report.output_files.push(output.output_path);
            }
        }
    }
    progress.finish();
    Ok(report)
}

fn ensure_supported_request(request: &AnalystConsensusRequest) -> Result<()> {
    if request.start_date > request.end_date {
        return Err(err("--start-date must be <= --end-date"));
    }
    if request.date_batch_size == 0 {
        return Err(err("--date-batch-size must be greater than 0"));
    }
    Ok(())
}

fn np_history_dates_for_batch(calendar: &TradingCalendar, target_dates: &[i32]) -> BTreeSet<i32> {
    let mut dates = BTreeSet::new();
    for &trade_date in target_dates {
        dates.insert(trade_date);
        for days in [7, 28, 91, 182, 364] {
            if let Some(previous) = calendar.last_open_on_or_before(add_days(trade_date, -days)) {
                dates.insert(previous);
            }
        }
    }
    dates
}

#[derive(Clone, Debug, Default)]
struct AnalystConsensusState {
    next_row_idx: usize,
    forecasts: HashMap<ForecastBucketKey, HashMap<String, ForecastObservation>>,
    ratings: HashMap<String, HashMap<String, RatingObservation>>,
    targets: HashMap<String, HashMap<String, TargetObservation>>,
    np_fy0_history: BTreeMap<i32, HashMap<String, f64>>,
}

impl AnalystConsensusState {
    fn ingest_until(&mut self, trade_date: i32, rows: &AnalystRows) {
        while self.next_row_idx < rows.rows.len()
            && rows.rows[self.next_row_idx].report_date <= trade_date
        {
            let row = &rows.rows[self.next_row_idx];
            self.ingest_row(row);
            self.next_row_idx += 1;
        }
        let min_keep_date = add_days(trade_date, -ANALYST_WARMUP_DAYS);
        self.np_fy0_history.retain(|date, _| *date >= min_keep_date);
    }

    fn ingest_row(&mut self, row: &AnalystRow) {
        for (metric, value) in [
            (BaseMetric::OperatingRevenue, row.op_rt),
            (BaseMetric::NetProfit, row.np),
            (BaseMetric::Eps, row.eps),
        ] {
            if let Some(value) = clean(value) {
                let key = ForecastBucketKey {
                    ts_code: row.ts_code.clone(),
                    year: row.forecast_year,
                    metric,
                };
                let observation = ForecastObservation {
                    report_date: row.report_date,
                    create_time: row.create_time.clone(),
                    value,
                };
                let bucket = self.forecasts.entry(key).or_default();
                if should_replace_forecast(bucket.get(&row.org_name), &observation) {
                    bucket.insert(row.org_name.clone(), observation);
                }
            }
        }

        if let Some(strength) = row.rating_strength {
            let observation = RatingObservation {
                report_date: row.report_date,
                create_time: row.create_time.clone(),
                strength,
            };
            let bucket = self.ratings.entry(row.ts_code.clone()).or_default();
            if should_replace_rating(bucket.get(&row.org_name), &observation) {
                bucket.insert(row.org_name.clone(), observation);
            }
        }

        if let Some(target_price) = clean(row.min_price).filter(|value| *value > 0.0) {
            let observation = TargetObservation {
                report_date: row.report_date,
                create_time: row.create_time.clone(),
                target_price,
            };
            let bucket = self.targets.entry(row.ts_code.clone()).or_default();
            if should_replace_target(bucket.get(&row.org_name), &observation) {
                bucket.insert(row.org_name.clone(), observation);
            }
        }
    }

    fn active_stocks(&self) -> impl Iterator<Item = &str> {
        let mut stocks = BTreeSet::new();
        for key in self.forecasts.keys() {
            stocks.insert(key.ts_code.as_str());
        }
        for key in self.ratings.keys() {
            stocks.insert(key.as_str());
        }
        for key in self.targets.keys() {
            stocks.insert(key.as_str());
        }
        stocks.into_iter()
    }

    fn remember_np_fy0(&mut self, trade_date: i32, values: HashMap<String, f64>) {
        self.np_fy0_history.insert(trade_date, values);
    }

    fn previous_np_fy0(&self, target_date: i32, ts_code: &str) -> Option<f64> {
        self.np_fy0_history
            .range(..=target_date)
            .next_back()
            .and_then(|(_, values)| values.get(ts_code))
            .copied()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct ForecastBucketKey {
    ts_code: String,
    year: i32,
    metric: BaseMetric,
}

#[derive(Clone, Debug)]
struct ForecastObservation {
    report_date: i32,
    create_time: Option<String>,
    value: f64,
}

#[derive(Clone, Debug)]
struct RatingObservation {
    report_date: i32,
    create_time: Option<String>,
    strength: f64,
}

#[derive(Clone, Debug)]
struct TargetObservation {
    report_date: i32,
    create_time: Option<String>,
    target_price: f64,
}

#[derive(Clone, Debug)]
struct AnalystRows {
    rows: Vec<AnalystRow>,
}

#[derive(Clone, Debug)]
struct AnalystRow {
    ts_code: String,
    report_date: i32,
    org_name: String,
    create_time: Option<String>,
    forecast_year: i32,
    op_rt: Option<f64>,
    np: Option<f64>,
    eps: Option<f64>,
    rating_strength: Option<f64>,
    min_price: Option<f64>,
}

impl AnalystRows {
    fn load(
        loader: &MarketDataLoader,
        start_date: i32,
        end_date: i32,
        disclosure_cache: &mut DisclosureTableCache,
    ) -> Result<Self> {
        let columns = vec![
            "org_name".to_string(),
            "author_name".to_string(),
            "create_time".to_string(),
            "op_rt".to_string(),
            "np".to_string(),
            "eps".to_string(),
            "rating".to_string(),
            "min_price".to_string(),
            "max_price".to_string(),
        ];
        let table = loader.load_stock_analyst_report_cached(
            &columns,
            start_date,
            end_date,
            disclosure_cache,
        )?;
        let ts_codes = table.required_utf8("ts_code")?;
        let report_dates = table.required_i32_date_cast("report_date")?;
        let quarters = table.required_utf8("quarter")?;
        let org_names = table.required_utf8("org_name")?;
        let create_times = table.required_utf8("create_time")?;
        let op_rt = table.required_f64_cast("op_rt")?;
        let np = table.required_f64_cast("np")?;
        let eps = table.required_f64_cast("eps")?;
        let ratings = table.required_utf8("rating")?;
        let min_prices = table.required_f64_cast("min_price")?;

        let mut rows = Vec::new();
        for idx in 0..table.len {
            let (Some(ts_code), Some(report_date), Some(forecast_year), Some(org_name)) = (
                ts_codes[idx].as_deref(),
                report_dates[idx],
                quarters[idx]
                    .as_deref()
                    .and_then(parse_annual_forecast_year),
                org_names[idx].as_deref().and_then(non_empty_string),
            ) else {
                continue;
            };
            if report_date < start_date || report_date > end_date {
                continue;
            }
            rows.push(AnalystRow {
                ts_code: ts_code.to_string(),
                report_date,
                org_name: org_name.to_string(),
                create_time: create_times[idx].clone(),
                forecast_year,
                op_rt: clean(op_rt[idx]),
                np: clean(np[idx]),
                eps: clean(eps[idx]),
                rating_strength: ratings[idx].as_deref().and_then(rating_strength),
                min_price: clean(min_prices[idx]),
            });
        }
        rows.sort_by(|left, right| {
            left.report_date
                .cmp(&right.report_date)
                .then_with(|| left.create_time.cmp(&right.create_time))
        });
        Ok(Self { rows })
    }
}

#[derive(Clone, Debug, Default)]
struct DailyMarketData {
    by_date: HashMap<i32, DailyMarketSnapshot>,
}

#[derive(Clone, Debug, Default)]
struct DailyMarketSnapshot {
    rows: Vec<MarketRow>,
}

#[derive(Clone, Debug)]
struct MarketRow {
    ts_code: String,
    close: Option<f64>,
    pre_close: Option<f64>,
}

impl DailyMarketData {
    fn load(loader: &MarketDataLoader, target_dates: &[i32]) -> Result<Self> {
        let pv = loader.load_daily_by_dates(
            DatasetId::StockDailyPv,
            &["close".to_string(), "pre_close".to_string()],
            target_dates,
        )?;

        let mut by_date = HashMap::<i32, DailyMarketSnapshot>::new();
        if !pv.columns.is_empty() {
            let dates = pv.required_i32("trade_date")?;
            let ts_codes = pv.required_utf8("ts_code")?;
            let closes = pv.required_f64_cast("close")?;
            let pre_closes = pv.required_f64_cast("pre_close")?;
            for idx in 0..pv.len {
                let (Some(date), Some(ts_code)) = (dates[idx], ts_codes[idx].as_deref()) else {
                    continue;
                };
                by_date.entry(date).or_default().rows.push(MarketRow {
                    ts_code: ts_code.to_string(),
                    close: clean(closes[idx]),
                    pre_close: clean(pre_closes[idx]),
                });
            }
        }
        for snapshot in by_date.values_mut() {
            snapshot
                .rows
                .sort_by(|left, right| left.ts_code.cmp(&right.ts_code));
        }
        Ok(Self { by_date })
    }

    fn snapshot(&self, trade_date: i32) -> DailyMarketSnapshot {
        self.by_date.get(&trade_date).cloned().unwrap_or_default()
    }
}

#[derive(Clone)]
struct ConsensusFinancialData {
    income_index: Arc<FinancialPitIndex>,
    balance_index: Arc<FinancialPitIndex>,
}

impl ConsensusFinancialData {
    fn load(
        loader: &MarketDataLoader,
        start_date: i32,
        end_date: i32,
        disclosure_cache: &mut DisclosureTableCache,
    ) -> Result<Self> {
        let income_sources = loader.load_financial_sources_cached(
            DatasetId::StockIncome,
            &[
                "revenue".to_string(),
                "n_income_attr_p".to_string(),
                "basic_eps".to_string(),
            ],
            start_date,
            end_date,
            24,
            disclosure_cache,
        )?;
        let balance_sources = loader.load_financial_sources_cached(
            DatasetId::StockBalanceSheet,
            &["total_hldr_eqy_exc_min_int".to_string()],
            start_date,
            end_date,
            24,
            disclosure_cache,
        )?;
        Ok(Self {
            income_index: Arc::new(FinancialPitIndex::from_source_tables(
                income_sources,
                Some(end_date),
            )?),
            balance_index: Arc::new(FinancialPitIndex::from_source_tables(
                balance_sources,
                Some(end_date),
            )?),
        })
    }

    fn income(&self) -> FinancialPitReader<'_> {
        self.income_index
            .reader(ReportTypePreference::consolidated())
    }

    fn balance(&self) -> FinancialPitReader<'_> {
        self.balance_index
            .reader(ReportTypePreference::balance_sheet_consolidated())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum BaseMetric {
    OperatingRevenue,
    NetProfit,
    Eps,
}

impl BaseMetric {
    fn actual_column(self) -> &'static str {
        match self {
            Self::OperatingRevenue => "revenue",
            Self::NetProfit => "n_income_attr_p",
            Self::Eps => "basic_eps",
        }
    }

    fn prefix(self) -> &'static str {
        match self {
            Self::OperatingRevenue => "con_or",
            Self::NetProfit => "con_np",
            Self::Eps => "con_eps",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct BaseConsensus {
    value: Option<f64>,
    consensus_type: Option<i32>,
    institution_count: Option<i32>,
    hisdate: Option<i32>,
}

impl BaseConsensus {
    fn missing(consensus_type: i32) -> Self {
        Self {
            value: None,
            consensus_type: Some(consensus_type),
            institution_count: Some(0),
            hisdate: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct AnnualConsensus {
    operating_revenue: BaseConsensus,
    net_profit: BaseConsensus,
    eps: BaseConsensus,
    net_assets: Option<f64>,
    pb: Option<f64>,
    ps: Option<f64>,
    pe: Option<f64>,
    peg: Option<f64>,
    roe: Option<f64>,
    or_yoy: Option<f64>,
    np_yoy: Option<f64>,
    npcgrate_2y: Option<f64>,
}

impl Default for BaseConsensus {
    fn default() -> Self {
        Self::missing(4)
    }
}

fn build_consensus_table_for_date(
    trade_date: i32,
    market: &DailyMarketData,
    financial: &ConsensusFinancialData,
    state: &AnalystConsensusState,
) -> Result<Table> {
    let snapshot = market.snapshot(trade_date);
    let years = fiscal_years(trade_date);
    let mut rows = Vec::with_capacity(snapshot.rows.len());
    for market_row in snapshot.rows {
        let price = effective_price(market_row.close, market_row.pre_close);
        let mut by_year = HashMap::<i32, AnnualConsensus>::new();
        for year in (years[0] - 2)..=(years[3] + 1) {
            let annual = compute_annual_consensus(
                &market_row.ts_code,
                trade_date,
                year,
                price,
                state,
                financial,
            );
            by_year.insert(year, annual);
        }
        let rating = compute_rating(&market_row.ts_code, trade_date, state);
        let target = compute_target_price(&market_row.ts_code, trade_date, state);
        let roll = compute_roll_consensus(trade_date, price, &by_year);
        let np_grates = compute_np_grates(
            &market_row.ts_code,
            trade_date,
            state,
            by_year[&years[0]].net_profit.value,
        );
        rows.push(ConsensusRow {
            ts_code: market_row.ts_code,
            trade_date,
            annuals: years.map(|year| by_year.get(&year).copied().unwrap_or_default()),
            roll,
            np_grates,
            rating,
            target,
        });
    }
    consensus_rows_table(&rows)
}

fn compute_annual_consensus(
    ts_code: &str,
    trade_date: i32,
    year: i32,
    price: Option<f64>,
    state: &AnalystConsensusState,
    financial: &ConsensusFinancialData,
) -> AnnualConsensus {
    let operating_revenue = compute_annual_base(
        ts_code,
        trade_date,
        year,
        BaseMetric::OperatingRevenue,
        state,
        financial,
    );
    let net_profit = compute_annual_base(
        ts_code,
        trade_date,
        year,
        BaseMetric::NetProfit,
        state,
        financial,
    );
    let eps = compute_annual_base(ts_code, trade_date, year, BaseMetric::Eps, state, financial);
    let net_assets = compute_con_na(ts_code, trade_date, net_profit.value, financial);
    let shares = safe_div(net_profit.value, eps.value);
    let pb = safe_div(
        price.zip(shares).map(|(price, shares)| price * shares),
        net_assets,
    );
    let ps = safe_div(
        price.zip(shares).map(|(price, shares)| price * shares),
        operating_revenue.value,
    );
    let pe = safe_div(price, eps.value);
    let roe = safe_div(net_profit.value.map(|value| value * 100.0), net_assets);
    let or_prev = compute_annual_base(
        ts_code,
        trade_date,
        year - 1,
        BaseMetric::OperatingRevenue,
        state,
        financial,
    )
    .value;
    let np_prev = compute_annual_base(
        ts_code,
        trade_date,
        year - 1,
        BaseMetric::NetProfit,
        state,
        financial,
    )
    .value;
    let or_yoy = yoy(operating_revenue.value, or_prev);
    let np_yoy = yoy(net_profit.value, np_prev);
    let historical_np = actual_annual_value(
        ts_code,
        trade_date,
        year - 2,
        BaseMetric::NetProfit,
        financial,
    );
    let npcgrate_2y = cagr_2y(net_profit.value, historical_np);
    let peg = match (pe, npcgrate_2y) {
        (Some(pe), Some(growth)) if growth > EPS && pe >= 0.0 => Some(pe / growth),
        _ => None,
    };
    AnnualConsensus {
        operating_revenue,
        net_profit,
        eps,
        net_assets,
        pb,
        ps,
        pe,
        peg,
        roe,
        or_yoy,
        np_yoy,
        npcgrate_2y,
    }
}

fn compute_annual_base(
    ts_code: &str,
    trade_date: i32,
    year: i32,
    metric: BaseMetric,
    state: &AnalystConsensusState,
    financial: &ConsensusFinancialData,
) -> BaseConsensus {
    if let Some(value) = actual_annual_value(ts_code, trade_date, year, metric, financial) {
        return BaseConsensus {
            value: Some(value),
            consensus_type: Some(0),
            institution_count: Some(0),
            hisdate: None,
        };
    }
    forecast_consensus(ts_code, trade_date, year, metric, state)
}

fn actual_annual_value(
    ts_code: &str,
    trade_date: i32,
    year: i32,
    metric: BaseMetric,
    financial: &ConsensusFinancialData,
) -> Option<f64> {
    let end_date = year * 10_000 + 12_31;
    financial
        .income()
        .record_for_end_date(ts_code, trade_date, end_date)?
        .column(metric.actual_column())
        .and_then(clean_value)
}

fn compute_con_na(
    ts_code: &str,
    trade_date: i32,
    con_np: Option<f64>,
    financial: &ConsensusFinancialData,
) -> Option<f64> {
    let balance = financial.balance();
    let end_date = balance.latest_quarter_end_date(ts_code, trade_date)?;
    let equity = balance
        .record_for_end_date(ts_code, trade_date, end_date)?
        .column("total_hldr_eqy_exc_min_int")
        .and_then(clean_value)?;
    Some(equity + con_np?)
}

fn forecast_consensus(
    ts_code: &str,
    trade_date: i32,
    year: i32,
    metric: BaseMetric,
    state: &AnalystConsensusState,
) -> BaseConsensus {
    let key = ForecastBucketKey {
        ts_code: ts_code.to_string(),
        year,
        metric,
    };
    let observations = state
        .forecasts
        .get(&key)
        .map(|bucket| {
            bucket
                .values()
                .filter(|observation| observation.report_date <= trade_date)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let strict_start = add_days(trade_date, -FORECAST_STRICT_DAYS);
    let strict = observations
        .iter()
        .copied()
        .filter(|obs| obs.report_date >= strict_start)
        .collect::<Vec<_>>();
    if strict.len() >= STRICT_FORECAST_INSTITUTIONS {
        return BaseConsensus {
            value: equal_weight(&strict),
            consensus_type: Some(1),
            institution_count: Some(strict.len() as i32),
            hisdate: None,
        };
    }
    let loose_start = add_days(trade_date, -FORECAST_LOOSE_DAYS);
    let loose = observations
        .iter()
        .copied()
        .filter(|obs| obs.report_date >= loose_start)
        .collect::<Vec<_>>();
    if !loose.is_empty() {
        return BaseConsensus {
            value: equal_weight(&loose),
            consensus_type: Some(2),
            institution_count: Some(loose.len() as i32),
            hisdate: None,
        };
    }
    let carry_start = add_days(trade_date, -FORECAST_CARRY_DAYS);
    let carry = observations
        .iter()
        .copied()
        .filter(|obs| obs.report_date >= carry_start)
        .collect::<Vec<_>>();
    if !carry.is_empty() {
        let hisdate = carry.iter().map(|obs| obs.report_date).max();
        return BaseConsensus {
            value: equal_weight(&carry),
            consensus_type: Some(3),
            institution_count: Some(carry.len() as i32),
            hisdate,
        };
    }
    BaseConsensus::missing(4)
}

fn compute_rating(
    ts_code: &str,
    trade_date: i32,
    state: &AnalystConsensusState,
) -> RatingConsensus {
    let observations = state
        .ratings
        .get(ts_code)
        .map(|bucket| {
            bucket
                .values()
                .filter(|observation| observation.report_date <= trade_date)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let strict_start = add_days(trade_date, -FORECAST_STRICT_DAYS);
    let strict = observations
        .iter()
        .copied()
        .filter(|obs| obs.report_date >= strict_start)
        .collect::<Vec<_>>();
    if strict.len() >= STRICT_RATING_TARGET_INSTITUTIONS {
        return RatingConsensus {
            value: mean(strengths(&strict)),
            consensus_type: Some(1),
        };
    }
    if !strict.is_empty() {
        return RatingConsensus {
            value: mean(strengths(&strict)),
            consensus_type: Some(2),
        };
    }
    RatingConsensus {
        value: None,
        consensus_type: Some(3),
    }
}

fn compute_target_price(
    ts_code: &str,
    trade_date: i32,
    state: &AnalystConsensusState,
) -> TargetConsensus {
    let observations = state
        .targets
        .get(ts_code)
        .map(|bucket| {
            bucket
                .values()
                .filter(|observation| observation.report_date <= trade_date)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let strict_start = add_days(trade_date, -FORECAST_STRICT_DAYS);
    let strict = observations
        .iter()
        .copied()
        .filter(|obs| obs.report_date >= strict_start)
        .collect::<Vec<_>>();
    if strict.len() >= STRICT_RATING_TARGET_INSTITUTIONS {
        return TargetConsensus {
            value: mean(strict.iter().map(|obs| obs.target_price)),
            consensus_type: Some(1),
        };
    }
    if !strict.is_empty() {
        return TargetConsensus {
            value: mean(strict.iter().map(|obs| obs.target_price)),
            consensus_type: Some(2),
        };
    }
    TargetConsensus {
        value: None,
        consensus_type: Some(3),
    }
}

fn strengths<'a>(observations: &'a [&'a RatingObservation]) -> impl Iterator<Item = f64> + 'a {
    observations.iter().map(|obs| obs.strength)
}

fn equal_weight(observations: &[&ForecastObservation]) -> Option<f64> {
    mean(observations.iter().map(|obs| obs.value))
}

fn mean<I>(values: I) -> Option<f64>
where
    I: Iterator<Item = f64>,
{
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        let Some(value) = clean(Some(value)) else {
            continue;
        };
        sum += value;
        count += 1;
    }
    (count > 0).then_some(sum / count as f64)
}

#[derive(Clone, Copy, Debug, Default)]
struct RatingConsensus {
    value: Option<f64>,
    consensus_type: Option<i32>,
}

#[derive(Clone, Copy, Debug, Default)]
struct TargetConsensus {
    value: Option<f64>,
    consensus_type: Option<i32>,
}

#[derive(Clone, Copy, Debug, Default)]
struct RollConsensus {
    con_or_roll: Option<f64>,
    con_np_roll: Option<f64>,
    con_eps_roll: Option<f64>,
    con_na_roll: Option<f64>,
    con_pb_roll: Option<f64>,
    con_ps_roll: Option<f64>,
    con_pe_roll: Option<f64>,
    con_peg_roll: Option<f64>,
    con_roe_roll: Option<f64>,
    con_or_yoy_roll: Option<f64>,
    con_np_yoy_roll: Option<f64>,
    con_npcgrate_2y_roll: Option<f64>,
}

fn compute_roll_consensus(
    trade_date: i32,
    price: Option<f64>,
    by_year: &HashMap<i32, AnnualConsensus>,
) -> RollConsensus {
    let year = trade_date / 10_000;
    let w = days_until_year_end(trade_date) as f64 / 365.0;
    let current = |field: fn(AnnualConsensus) -> Option<f64>| {
        weighted(
            by_year.get(&year).copied().and_then(field),
            by_year.get(&(year + 1)).copied().and_then(field),
            w,
        )
    };
    let historical = |field: fn(AnnualConsensus) -> Option<f64>| {
        weighted(
            by_year.get(&(year - 1)).copied().and_then(field),
            by_year.get(&year).copied().and_then(field),
            w,
        )
    };
    let con_or_roll = current(|annual| annual.operating_revenue.value);
    let con_np_roll = current(|annual| annual.net_profit.value);
    let con_eps_roll = current(|annual| annual.eps.value);
    let con_na_roll = current(|annual| annual.net_assets);
    let con_or_yoy_roll = yoy(
        con_or_roll,
        historical(|annual| annual.operating_revenue.value),
    );
    let con_np_yoy_roll = yoy(con_np_roll, historical(|annual| annual.net_profit.value));
    let con_roe_roll = safe_div(con_np_roll.map(|value| value * 100.0), con_na_roll);
    let con_pe_roll = safe_div(price, con_eps_roll);
    let shares_current = current(|annual| safe_div(annual.net_profit.value, annual.eps.value));
    let con_pb_roll = safe_div(
        price
            .zip(shares_current)
            .map(|(price, shares)| price * shares),
        con_na_roll,
    );
    let con_ps_roll = safe_div(
        price
            .zip(shares_current)
            .map(|(price, shares)| price * shares),
        con_or_roll,
    );
    let historical_roll_np = weighted(
        by_year
            .get(&(year - 2))
            .and_then(|annual| annual.net_profit.value),
        by_year
            .get(&(year - 1))
            .and_then(|annual| annual.net_profit.value),
        w,
    );
    let con_npcgrate_2y_roll = cagr_2y(con_np_roll, historical_roll_np);
    let con_peg_roll = match (con_pe_roll, con_npcgrate_2y_roll) {
        (Some(pe), Some(growth)) if growth > EPS && pe >= 0.0 => Some(pe / growth),
        _ => None,
    };
    RollConsensus {
        con_or_roll,
        con_np_roll,
        con_eps_roll,
        con_na_roll,
        con_pb_roll,
        con_ps_roll,
        con_pe_roll,
        con_peg_roll,
        con_roe_roll,
        con_or_yoy_roll,
        con_np_yoy_roll,
        con_npcgrate_2y_roll,
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct NpGrates {
    con_npgrate_1w: Option<f64>,
    con_npgrate_4w: Option<f64>,
    con_npgrate_13w: Option<f64>,
    con_npgrate_26w: Option<f64>,
    con_npgrate_52w: Option<f64>,
}

fn compute_np_grates(
    ts_code: &str,
    trade_date: i32,
    state: &AnalystConsensusState,
    current_np: Option<f64>,
) -> NpGrates {
    let change = |days: i32| {
        let target_date = add_days(trade_date, -days);
        let previous = state.previous_np_fy0(target_date, ts_code);
        yoy(current_np, previous)
    };
    NpGrates {
        con_npgrate_1w: change(7),
        con_npgrate_4w: change(28),
        con_npgrate_13w: change(91),
        con_npgrate_26w: change(182),
        con_npgrate_52w: change(364),
    }
}

#[derive(Clone, Debug)]
struct ConsensusRow {
    trade_date: i32,
    ts_code: String,
    annuals: [AnnualConsensus; 4],
    roll: RollConsensus,
    np_grates: NpGrates,
    rating: RatingConsensus,
    target: TargetConsensus,
}

fn consensus_rows_table(rows: &[ConsensusRow]) -> Result<Table> {
    let mut columns = BTreeMap::<String, ColumnData>::new();
    columns.insert(
        "trade_date".to_string(),
        ColumnData::I32(rows.iter().map(|row| Some(row.trade_date)).collect()),
    );
    columns.insert(
        "ts_code".to_string(),
        ColumnData::Utf8(rows.iter().map(|row| Some(row.ts_code.clone())).collect()),
    );
    for (idx, suffix) in ["fy0", "fy1", "fy2", "fy3"].iter().enumerate() {
        let annual = rows.iter().map(|row| row.annuals[idx]).collect::<Vec<_>>();
        push_base_columns(
            &mut columns,
            rows,
            &annual,
            BaseMetric::OperatingRevenue,
            suffix,
        );
        push_base_columns(&mut columns, rows, &annual, BaseMetric::NetProfit, suffix);
        push_base_columns(&mut columns, rows, &annual, BaseMetric::Eps, suffix);
        push_f64_column(
            &mut columns,
            &format!("con_na_{suffix}"),
            annual.iter().map(|v| v.net_assets),
        );
        push_f64_column(
            &mut columns,
            &format!("con_pb_{suffix}"),
            annual.iter().map(|v| v.pb),
        );
        push_f64_column(
            &mut columns,
            &format!("con_ps_{suffix}"),
            annual.iter().map(|v| v.ps),
        );
        push_f64_column(
            &mut columns,
            &format!("con_pe_{suffix}"),
            annual.iter().map(|v| v.pe),
        );
        push_f64_column(
            &mut columns,
            &format!("con_peg_{suffix}"),
            annual.iter().map(|v| v.peg),
        );
        push_f64_column(
            &mut columns,
            &format!("con_roe_{suffix}"),
            annual.iter().map(|v| v.roe),
        );
        push_f64_column(
            &mut columns,
            &format!("con_or_yoy_{suffix}"),
            annual.iter().map(|v| v.or_yoy),
        );
        push_f64_column(
            &mut columns,
            &format!("con_np_yoy_{suffix}"),
            annual.iter().map(|v| v.np_yoy),
        );
        push_f64_column(
            &mut columns,
            &format!("con_npcgrate_2y_{suffix}"),
            annual.iter().map(|v| v.npcgrate_2y),
        );
    }
    push_f64_column(
        &mut columns,
        "con_npgrate_1w",
        rows.iter().map(|row| row.np_grates.con_npgrate_1w),
    );
    push_f64_column(
        &mut columns,
        "con_npgrate_4w",
        rows.iter().map(|row| row.np_grates.con_npgrate_4w),
    );
    push_f64_column(
        &mut columns,
        "con_npgrate_13w",
        rows.iter().map(|row| row.np_grates.con_npgrate_13w),
    );
    push_f64_column(
        &mut columns,
        "con_npgrate_26w",
        rows.iter().map(|row| row.np_grates.con_npgrate_26w),
    );
    push_f64_column(
        &mut columns,
        "con_npgrate_52w",
        rows.iter().map(|row| row.np_grates.con_npgrate_52w),
    );
    push_roll_columns(&mut columns, rows);
    push_f64_column(
        &mut columns,
        "con_rating_strength",
        rows.iter().map(|row| row.rating.value),
    );
    push_i32_column(
        &mut columns,
        "con_rating_type",
        rows.iter().map(|row| row.rating.consensus_type),
    );
    push_f64_column(
        &mut columns,
        "con_target_price",
        rows.iter().map(|row| row.target.value),
    );
    push_i32_column(
        &mut columns,
        "con_target_price_type",
        rows.iter().map(|row| row.target.consensus_type),
    );
    Table::new(columns)
}

fn push_base_columns(
    columns: &mut BTreeMap<String, ColumnData>,
    _rows: &[ConsensusRow],
    annual: &[AnnualConsensus],
    metric: BaseMetric,
    suffix: &str,
) {
    let base = metric.prefix();
    let values = annual.iter().map(|annual| match metric {
        BaseMetric::OperatingRevenue => annual.operating_revenue,
        BaseMetric::NetProfit => annual.net_profit,
        BaseMetric::Eps => annual.eps,
    });
    let values = values.collect::<Vec<_>>();
    push_f64_column(
        columns,
        &format!("{base}_{suffix}"),
        values.iter().map(|v| v.value),
    );
    push_i32_column(
        columns,
        &format!("{base}_type_{suffix}"),
        values.iter().map(|v| v.consensus_type),
    );
    push_i32_column(
        columns,
        &format!("{base}_inst_count_{suffix}"),
        values.iter().map(|v| v.institution_count),
    );
    push_i32_column(
        columns,
        &format!("{base}_hisdate_{suffix}"),
        values.iter().map(|v| v.hisdate),
    );
}

fn push_roll_columns(columns: &mut BTreeMap<String, ColumnData>, rows: &[ConsensusRow]) {
    push_f64_column(
        columns,
        "con_or_roll",
        rows.iter().map(|row| row.roll.con_or_roll),
    );
    push_f64_column(
        columns,
        "con_np_roll",
        rows.iter().map(|row| row.roll.con_np_roll),
    );
    push_f64_column(
        columns,
        "con_eps_roll",
        rows.iter().map(|row| row.roll.con_eps_roll),
    );
    push_f64_column(
        columns,
        "con_na_roll",
        rows.iter().map(|row| row.roll.con_na_roll),
    );
    push_f64_column(
        columns,
        "con_pb_roll",
        rows.iter().map(|row| row.roll.con_pb_roll),
    );
    push_f64_column(
        columns,
        "con_ps_roll",
        rows.iter().map(|row| row.roll.con_ps_roll),
    );
    push_f64_column(
        columns,
        "con_pe_roll",
        rows.iter().map(|row| row.roll.con_pe_roll),
    );
    push_f64_column(
        columns,
        "con_peg_roll",
        rows.iter().map(|row| row.roll.con_peg_roll),
    );
    push_f64_column(
        columns,
        "con_roe_roll",
        rows.iter().map(|row| row.roll.con_roe_roll),
    );
    push_f64_column(
        columns,
        "con_or_yoy_roll",
        rows.iter().map(|row| row.roll.con_or_yoy_roll),
    );
    push_f64_column(
        columns,
        "con_np_yoy_roll",
        rows.iter().map(|row| row.roll.con_np_yoy_roll),
    );
    push_f64_column(
        columns,
        "con_npcgrate_2y_roll",
        rows.iter().map(|row| row.roll.con_npcgrate_2y_roll),
    );
}

fn push_f64_column<I>(columns: &mut BTreeMap<String, ColumnData>, name: &str, values: I)
where
    I: Iterator<Item = Option<f64>>,
{
    columns.insert(
        name.to_string(),
        ColumnData::F64(values.map(|value| value.and_then(clean_value)).collect()),
    );
}

fn push_i32_column<I>(columns: &mut BTreeMap<String, ColumnData>, name: &str, values: I)
where
    I: Iterator<Item = Option<i32>>,
{
    columns.insert(name.to_string(), ColumnData::I32(values.collect()));
}

fn fiscal_years(trade_date: i32) -> [i32; 4] {
    let year = trade_date / 10_000;
    let month_day = trade_date % 10_000;
    let base = if month_day < 501 { year - 2 } else { year - 1 };
    [base, base + 1, base + 2, base + 3]
}

fn parse_annual_forecast_year(value: &str) -> Option<i32> {
    let value = value.trim();
    if value.len() != 6 || !value.ends_with("Q4") {
        return None;
    }
    value[..4].parse::<i32>().ok()
}

fn non_empty_string(value: &str) -> Option<&str> {
    let value = value.trim();
    (!value.is_empty()).then_some(value)
}

fn rating_strength(value: &str) -> Option<f64> {
    let value = value.trim().to_ascii_lowercase();
    match value.as_str() {
        "买入" | "强烈推荐" | "推荐" | "buy" | "strong buy" => Some(1.0),
        "增持" | "优于大市" | "跑赢行业" | "outperform" | "overweight" => Some(0.75),
        "中性" | "持有" | "无" | "neutral" | "hold" => Some(0.5),
        "减持" | "低于大市" | "underperform" | "underweight" => Some(0.25),
        "卖出" | "sell" => Some(0.0),
        _ => None,
    }
}

fn should_replace_forecast(
    current: Option<&ForecastObservation>,
    next: &ForecastObservation,
) -> bool {
    current
        .map(|current| {
            next.report_date > current.report_date
                || (next.report_date == current.report_date
                    && next.create_time > current.create_time)
        })
        .unwrap_or(true)
}

fn should_replace_rating(current: Option<&RatingObservation>, next: &RatingObservation) -> bool {
    current
        .map(|current| {
            next.report_date > current.report_date
                || (next.report_date == current.report_date
                    && next.create_time > current.create_time)
        })
        .unwrap_or(true)
}

fn should_replace_target(current: Option<&TargetObservation>, next: &TargetObservation) -> bool {
    current
        .map(|current| {
            next.report_date > current.report_date
                || (next.report_date == current.report_date
                    && next.create_time > current.create_time)
        })
        .unwrap_or(true)
}

fn effective_price(close: Option<f64>, pre_close: Option<f64>) -> Option<f64> {
    clean(close)
        .filter(|value| *value > 0.0)
        .or_else(|| clean(pre_close).filter(|value| *value > 0.0))
}

fn safe_div(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    let numerator = numerator.and_then(clean_value)?;
    let denominator = denominator.and_then(clean_value)?;
    (denominator.abs() > EPS).then_some(numerator / denominator)
}

fn yoy(current: Option<f64>, previous: Option<f64>) -> Option<f64> {
    let current = current.and_then(clean_value)?;
    let previous = previous.and_then(clean_value)?;
    (previous.abs() > EPS).then_some(100.0 * (current - previous) / previous.abs())
}

fn cagr_2y(current: Option<f64>, previous: Option<f64>) -> Option<f64> {
    let current = current.and_then(clean_value)?;
    let previous = previous.and_then(clean_value)?;
    if current < 0.0 || previous <= EPS {
        return None;
    }
    Some(100.0 * ((current / previous).sqrt() - 1.0))
}

fn weighted(left: Option<f64>, right: Option<f64>, left_weight: f64) -> Option<f64> {
    Some(
        left.and_then(clean_value)? * left_weight
            + right.and_then(clean_value)? * (1.0 - left_weight),
    )
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.and_then(clean_value)
}

fn clean_value(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

fn days_until_year_end(date: i32) -> i32 {
    let year = date / 10_000;
    days_between(date, year * 10_000 + 12_31).max(0)
}

fn days_between(start: i32, end: i32) -> i32 {
    days_from_civil_i32(end) - days_from_civil_i32(start)
}

fn add_days(date: i32, days_delta: i32) -> i32 {
    civil_from_days(days_from_civil_i32(date) + days_delta)
}

fn days_from_civil_i32(date: i32) -> i32 {
    let year = date / 10_000;
    let month = (date / 100) % 100;
    let day = date % 100;
    days_from_civil(year, month, day)
}

fn days_from_civil(year: i32, month: i32, day: i32) -> i32 {
    let y = year - i32::from(month <= 2);
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = month + if month > 2 { -3 } else { 9 };
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(days: i32) -> i32 {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096).div_euclid(365);
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2).div_euclid(153);
    let day = doy - (153 * mp + 2).div_euclid(5) + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    let year = y + i32::from(month <= 2);
    year * 10_000 + month * 100 + day
}

#[cfg(test)]
mod tests {
    use super::{
        cagr_2y, effective_price, fiscal_years, parse_annual_forecast_year, rating_strength, yoy,
    };

    #[test]
    fn analyst_consensus_fiscal_years_switch_on_may_first() {
        assert_eq!(fiscal_years(20260430), [2024, 2025, 2026, 2027]);
        assert_eq!(fiscal_years(20260501), [2025, 2026, 2027, 2028]);
    }

    #[test]
    fn analyst_consensus_parses_only_annual_q4_forecast_year() {
        assert_eq!(parse_annual_forecast_year("2026Q4"), Some(2026));
        assert_eq!(parse_annual_forecast_year("2026Q3"), None);
        assert_eq!(parse_annual_forecast_year(""), None);
    }

    #[test]
    fn analyst_consensus_rating_text_maps_to_strength() {
        assert_eq!(rating_strength("买入"), Some(1.0));
        assert_eq!(rating_strength("增持"), Some(0.75));
        assert_eq!(rating_strength("中性"), Some(0.5));
        assert_eq!(rating_strength("减持"), Some(0.25));
        assert_eq!(rating_strength("卖出"), Some(0.0));
    }

    #[test]
    fn analyst_consensus_formula_helpers_keep_document_boundaries() {
        assert_eq!(effective_price(Some(0.0), Some(9.0)), Some(9.0));
        assert_eq!(yoy(Some(120.0), Some(100.0)), Some(20.0));
        assert_eq!(yoy(Some(120.0), Some(0.0)), None);
        assert!((cagr_2y(Some(121.0), Some(100.0)).unwrap() - 10.0).abs() < 1e-10);
        assert_eq!(cagr_2y(Some(-1.0), Some(100.0)), None);
    }
}
