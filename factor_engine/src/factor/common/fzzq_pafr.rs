use std::any::Any;
use std::collections::{BTreeMap, HashMap, VecDeque};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, Lookback,
};
use crate::data::DataPool;

use crate::error::{err, Result};
use crate::factor::common::financial::{add_days, add_months, analyst_fiscal_years};
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_and_fill_missing};
use crate::factor::common::{
    ClassificationLevel, ClassificationMap, DailyPanel, FinancialPitReader, PanelColumn,
    ReportTypePreference,
};
use crate::factor::{Factor, FactorUpdatePolicy};
use crate::operators::cross_sectional::{cs_regression_residual};

const DAYS: i32 = 90;
const PAIR_LOOKBACK: usize = 270;
const WINDOW: usize = 252;
const MIN_HISTORY: usize = 120;
const MIN_EVENTS: usize = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Output {
    Raw,
    Zscore,
}
impl Output {
    fn id(self) -> &'static str {
        match self {
            Self::Raw => "pafr",
            Self::Zscore => "pafr_zscore",
        }
    }
}

pub struct Pafr(pub Output);
impl Factor for Pafr {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: self.0.id().into(), aliases: vec![self.0.id().to_ascii_uppercase()],
            name: match self.0 { Output::Raw => "Momentum-Purged Analyst Forecast Revision", Output::Zscore => "Momentum-Purged Analyst Forecast Revision Historical Z-score" }.into(),
            asset_class: AssetClass::Stock, frequency: Frequency::Daily, version: "0.2.0".into(),
            tags: ["deprecated", "FZZQ", "analyst", "fundamental", "pit", "forecast_revision", "momentum_residual", "daily", "neutralize", "size", "sector"].into_iter().map(str::to_string).collect(),
            description: format!("Daily PAFR approximation: match org/stock/absolute Q4 forecast year. Deduplicate date/org/author/title/year (last finite source row wins), then equally average reports for each institution/date/year. Current report's FY2 follows PIT annual disclosure with May-1 fallback. Compare strictly previous valid date in the same year, even if that year had another FY label then. New revisions must be in (T-90 calendar days,T]; the strictly previous prediction must be in (T minus twelve calendar months,T]. Revision=(current-previous)/abs(previous), zero base invalid, clipped +/-0.25. Adjusted-close interval return minus SAME-INTERVAL equal-weight cross-sectional mean, excludes BJ; endpoints last session <= report date, missing endpoints invalid. Daily pooled event OLS with intercept: revision~between-report excess return, then residual~report-to-T excess return; constant regressor reduces to demeaning. Stocks need >=3 complete institution-day revision events before both fits and aggregation. {} Final daily SW L1/SIZE neutralization, missing final raw gets zero residual with valid SIZE and a fitted industry; no extra winsorization; filled values never enter rolling history. No AFR output. Benchmark cache bounded to active intervals; z-score history cached by instrument, never recomputed using future data.",
                match self.0 { Output::Raw => "Equally average second-stage residuals by stock.", Output::Zscore => "Standardize that raw stock mean using previous 252 trading sessions EXCLUDING today, >=120 finite values, population standard deviation; zero deviation is null." }),
            dependencies: vec![
                DataRequest::new(DatasetId::StockAnalystReport, &["np", "report_date", "quarter", "org_name", "author_name", "report_title"]),
                DataRequest::financial_quarters(DatasetId::StockIncome, &[], 8),
                DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: vec![],
            // Cover twelve calendar months and the prior-session price anchor for every fit.
            lookback: Lookback { trading_days: PAIR_LOOKBACK + if self.0 == Output::Zscore { WINDOW } else { 0 } },
        }
    }
    fn requirements_for_context(&self, context: &FactorContext) -> Vec<DataRequest> {
        self.spec()
            .dependencies
            .into_iter()
            .map(|request| {
                let dates = if request.dataset == DatasetId::StockBarraDaily {
                    context.target_dates.clone()
                } else {
                    request.resolved_dates(context)
                };
                let request = request.with_explicit_dates(dates);
                if matches!(
                    request.dataset,
                    DatasetId::StockDailyPv
                        | DatasetId::StockAdjFactor
                        | DatasetId::StockBarraDaily
                ) {
                    request.with_panel_only()
                } else {
                    request
                }
            })
            .collect()
    }
    fn compute_provider_key(&self) -> String {
        "fzzq_pafr".into()
    }
    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventStateDailyFast
    }
    fn initial_compute_state(&self, _: &[String]) -> Box<dyn Any + Send> {
        Box::new(State::default())
    }
    fn compute(&self, _: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        compute(&[self.0.id().into()], data, &mut State::default())?
            .pop()
            .ok_or_else(|| err("missing PAFR output"))
    }
    fn compute_many(
        &self,
        ids: &[String],
        _: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<FactorSeries>> {
        compute(ids, data, &mut State::default())
    }
    fn compute_many_stateful(
        &self,
        ids: &[String],
        _: &FactorContext,
        data: &DataPool,
        state: &mut (dyn Any + Send),
    ) -> Result<Vec<FactorSeries>> {
        compute(
            ids,
            data,
            state
                .downcast_mut::<State>()
                .ok_or_else(|| err("PAFR state mismatch"))?,
        )
    }
}

#[derive(Clone, Debug)]
struct Event {
    stock: usize,
    previous: i32,
    date: i32,
    revision: f64,
}

struct RevisionWindow {
    date: i32,
    recent_lower: i32,
    pair_lower: i32,
}

impl RevisionWindow {
    fn new(date: i32) -> Self {
        Self {
            date,
            recent_lower: add_days(date, -DAYS),
            pair_lower: add_months(date, -12),
        }
    }

    fn contains(&self, event: &Event) -> bool {
        event.date > self.recent_lower
            && event.date <= self.date
            && event.previous > self.pair_lower
            && event.previous < event.date
    }
}

fn revision(current: f64, previous: f64) -> Option<f64> {
    if !current.is_finite() || !previous.is_finite() || previous == 0.0 {
        return None;
    }
    let value = (current - previous) / previous.abs();
    value.is_finite().then(|| value.clamp(-0.25, 0.25))
}

fn build_events<'a>(
    table: impl Into<crate::data::SourceRowsReader<'a>>,
    panel: &DailyPanel,
    income: &FinancialPitReader<'_>,
    lower: i32,
    upper: i32,
) -> Result<Vec<Event>> {
    let table = table.into();
    let codes = table.required_utf8("ts_code")?;
    let dates = table.required_i32_date_cast("report_date")?;
    let years = table.required_utf8("quarter")?;
    let orgs = table.required_utf8("org_name")?;
    let authors = table.required_utf8("author_name")?;
    let titles = table.required_utf8("report_title")?;
    let np = table.required_f64_cast("np")?;
    let stocks: HashMap<_, _> = panel
        .instruments()
        .iter()
        .enumerate()
        .filter(|(_, code)| !is_bj_stock(code))
        .map(|(i, code)| (code.as_str(), i))
        .collect();
    let mut unique = BTreeMap::new();
    for i in 0..table.len {
        let (Some(code), Some(date), Some(year), Some(org), Some(value)) = (
            codes[i].as_deref(),
            dates[i],
            years[i].as_deref(),
            orgs[i].as_deref(),
            np[i].filter(|v| v.is_finite()),
        ) else {
            continue;
        };
        if date <= lower || date > upper || org.trim().is_empty() {
            continue;
        }
        let Some(&stock) = stocks.get(code) else {
            continue;
        };
        let Some(year) = year
            .trim()
            .strip_suffix("Q4")
            .and_then(|v| v.parse::<i32>().ok())
        else {
            continue;
        };
        unique.insert(
            (
                stock,
                org.trim(),
                year,
                date,
                authors[i].as_deref(),
                titles[i].as_deref(),
            ),
            value,
        );
    }
    let mut daily: BTreeMap<(usize, &str, i32, i32), (f64, usize)> = BTreeMap::new();
    for ((stock, org, year, date, _, _), value) in unique {
        let entry = daily.entry((stock, org, year, date)).or_default();
        entry.0 += value;
        entry.1 += 1;
    }
    let mut previous = None;
    let mut events = Vec::new();
    let mut fiscal_cache = HashMap::new();
    for ((stock, org, year, date), (sum, count)) in daily {
        let value = sum / count as f64;
        if !value.is_finite() {
            continue;
        }
        let key = (stock, org, year);
        if let Some((prior_key, prior_date, prior_value)) = previous {
            if prior_key == key {
                let fy2 = *fiscal_cache.entry((stock, date)).or_insert_with(|| {
                    analyst_fiscal_years(date, &panel.instruments()[stock], income)[2]
                });
                if year == fy2 {
                    if let Some(revision) = revision(value, prior_value) {
                        events.push(Event {
                            stock,
                            previous: prior_date,
                            date,
                            revision,
                        });
                    }
                }
            }
        }
        previous = Some((key, date, value));
    }
    events.sort_by_key(|event| (event.date, event.stock, event.previous));
    Ok(events)
}

#[derive(Default)]
struct MarketCache {
    means: HashMap<(i32, i32), Option<f64>>,
}

fn anchor_day(panel: &DailyPanel, date: i32) -> Option<usize> {
    panel.dates().partition_point(|d| *d <= date).checked_sub(1)
}

fn interval_return(
    panel: &DailyPanel,
    prices: &[Option<f64>],
    stock: usize,
    start: usize,
    end: usize,
) -> Option<f64> {
    let n = panel.instruments().len();
    let a = start * n + stock;
    let b = end * n + stock;
    if !panel.is_present_offset(a) || !panel.is_present_offset(b) {
        return None;
    }
    let (p0, p1) = (prices[a]?, prices[b]?);
    if p0 <= 0.0 || p1 <= 0.0 {
        return None;
    }
    let value = p1 / p0 - 1.0;
    value.is_finite().then_some(value)
}

impl MarketCache {
    fn mean(
        &mut self,
        panel: &DailyPanel,
        prices: &[Option<f64>],
        eligible: &[usize],
        start: usize,
        end: usize,
    ) -> Option<f64> {
        *self
            .means
            .entry((panel.dates()[start], panel.dates()[end]))
            .or_insert_with(|| {
                let mut sum = 0.0;
                let mut count = 0;
                for &stock in eligible {
                    if let Some(value) = interval_return(panel, prices, stock, start, end) {
                        sum += value;
                        count += 1;
                    }
                }
                (count > 0)
                    .then(|| sum / count as f64)
                    .filter(|v| v.is_finite())
            })
    }
    fn excess(
        &mut self,
        panel: &DailyPanel,
        prices: &[Option<f64>],
        eligible: &[usize],
        stock: usize,
        start: usize,
        end: usize,
    ) -> Option<f64> {
        let own = interval_return(panel, prices, stock, start, end)?;
        let value = own - self.mean(panel, prices, eligible, start, end)?;
        value.is_finite().then_some(value)
    }
}

struct ActiveEvent {
    event: Event,
    report_day: usize,
    between: f64,
}

fn aggregate(stock_count: usize, observations: &[(usize, f64, f64, f64)]) -> Vec<Option<f64>> {
    let mut counts = vec![0; stock_count];
    for &(stock, _, _, _) in observations {
        counts[stock] += 1;
    }
    let selected: Vec<_> = observations
        .iter()
        .filter(|row| counts[row.0] >= MIN_EVENTS)
        .collect();
    let y: Vec<_> = selected.iter().map(|row| Some(row.1)).collect();
    let pre: Vec<_> = selected.iter().map(|row| Some(row.2)).collect();
    let post: Vec<_> = selected.iter().map(|row| Some(row.3)).collect();
    let first = cs_regression_residual(&y, &pre);
    let second = cs_regression_residual(&first, &post);
    let mut sums = vec![0.0; stock_count];
    counts.fill(0);
    for (row, value) in selected.into_iter().zip(second) {
        if let Some(value) = value.filter(|v| v.is_finite()) {
            sums[row.0] += value;
            counts[row.0] += 1;
        }
    }
    sums.into_iter()
        .zip(counts)
        .map(|(sum, count)| (count >= MIN_EVENTS).then(|| sum / count as f64))
        .collect()
}

#[derive(Default)]
struct RollingHistory {
    values: VecDeque<Option<f64>>,
    count: usize,
    mean: f64,
    m2: f64,
    steps: usize,
}
impl RollingHistory {
    fn zscore(&self, current: Option<f64>) -> Option<f64> {
        let current = current?;
        if self.count < MIN_HISTORY {
            return None;
        }
        // Deleting old observations can leave positive round-off for a constant window.
        let first = *self.values.iter().flatten().next()?;
        if self.values.iter().flatten().all(|value| *value == first) {
            return None;
        }
        let (mean, variance) =
            if self.m2 <= f64::EPSILON * self.mean.abs().max(1.0).powi(2) * self.count as f64 {
                let reference = *self.values.iter().flatten().next()?;
                let centered_mean = self
                    .values
                    .iter()
                    .flatten()
                    .map(|v| v - reference)
                    .sum::<f64>()
                    / self.count as f64;
                let variance = self
                    .values
                    .iter()
                    .flatten()
                    .map(|v| (v - reference - centered_mean).powi(2))
                    .sum::<f64>()
                    / self.count as f64;
                (reference + centered_mean, variance)
            } else {
                (self.mean, self.m2 / self.count as f64)
            };
        let std = variance.max(0.0).sqrt();
        (std > f64::EPSILON)
            .then(|| (current - mean) / std)
            .filter(|v| v.is_finite())
    }
    fn add_moment(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        self.m2 += delta * (value - self.mean);
    }
    fn push(&mut self, value: Option<f64>) {
        if self.values.len() == WINDOW {
            if let Some(old) = self.values.pop_front().flatten() {
                if self.count <= 1 {
                    self.count = 0;
                    self.mean = 0.0;
                    self.m2 = 0.0;
                } else {
                    let old_mean = self.mean;
                    self.count -= 1;
                    self.mean -= (old - old_mean) / self.count as f64;
                    self.m2 = (self.m2 - (old - old_mean) * (old - self.mean)).max(0.0);
                }
            }
        }
        let value = value.filter(|v| v.is_finite());
        self.values.push_back(value);
        if let Some(value) = value {
            self.add_moment(value);
        }
        self.steps += 1;
        // Periodic two-pass rebuild bounds deletion round-off over long production runs.
        if self.steps % WINDOW == 0 {
            self.count = self.values.iter().flatten().count();
            self.mean = if self.count == 0 {
                0.0
            } else {
                self.values.iter().flatten().sum::<f64>() / self.count as f64
            };
            self.m2 = self
                .values
                .iter()
                .flatten()
                .map(|v| (v - self.mean).powi(2))
                .sum();
        }
    }
}

#[derive(Default)]
struct State {
    last_date: Option<i32>,
    keep_history: bool,
    histories: HashMap<String, RollingHistory>,
    market: MarketCache,
}

fn adjusted_prices(data: &DataPool, panel: &DailyPanel) -> Result<PanelColumn> {
    let close = AlignedSource::new(data, panel, DatasetId::StockDailyPv, "close")?;
    let adj = AlignedSource::new(data, panel, DatasetId::StockAdjFactor, "adj_factor")?;
    let mut values = Vec::with_capacity(panel.shape_len());
    for &date in panel.dates() {
        let c = close.day(date);
        let a = adj.day(date);
        for stock in 0..panel.instruments().len() {
            values.push(
                close
                    .get(c, stock)
                    .zip(adj.get(a, stock))
                    .and_then(|(c, a)| {
                        let value = c * a;
                        (c > 0.0 && a > 0.0 && value.is_finite()).then_some(value)
                    }),
            );
        }
    }
    panel.column_from_values(values)
}

// Borrow cached source columns; align by keys without materializing extra full grids.
struct AlignedSource<'a> {
    panel: &'a DailyPanel,
    column: PanelColumn,
    stocks: Vec<Option<usize>>,
}
impl<'a> AlignedSource<'a> {
    fn new(
        data: &'a DataPool,
        target: &DailyPanel,
        dataset: DatasetId,
        name: &str,
    ) -> Result<Self> {
        let panel = data.daily_panel(dataset)?;
        let lookup: HashMap<_, _> = panel
            .instruments()
            .iter()
            .enumerate()
            .map(|(i, code)| (code.as_str(), i))
            .collect();
        let stocks = target
            .instruments()
            .iter()
            .map(|code| lookup.get(code.as_str()).copied())
            .collect();
        Ok(Self {
            panel,
            column: panel.column(name)?,
            stocks,
        })
    }
    fn day(&self, date: i32) -> Option<usize> {
        self.panel.dates().binary_search(&date).ok()
    }
    fn get(&self, day: Option<usize>, stock: usize) -> Option<f64> {
        self.column.values()[day? * self.panel.instruments().len() + self.stocks[stock]?]
    }
}

fn compute(ids: &[String], data: &DataPool, state: &mut State) -> Result<Vec<FactorSeries>> {
    let outputs: Vec<_> = [Output::Raw, Output::Zscore]
        .into_iter()
        .filter(|output| ids.iter().any(|id| id == output.id()))
        .collect();
    if outputs.is_empty() {
        return Ok(vec![]);
    }
    let panel = data.stock_universe_panel()?;
    let n = panel.instruments().len();
    let needs_z = outputs.contains(&Output::Zscore);
    let Some(first_target) = panel
        .dates()
        .iter()
        .position(|date| panel.is_target_date(*date))
    else {
        return outputs
            .into_iter()
            .map(|output| {
                Ok(panel
                    .column_from_values(vec![None; panel.shape_len()])?
                    .to_factor_series(Pafr(output).spec()))
            })
            .collect();
    };
    let last_target = panel
        .dates()
        .iter()
        .rposition(|date| panel.is_target_date(*date))
        .unwrap();
    let resume = state
        .last_date
        .filter(|date| *date < panel.dates()[first_target])
        .and_then(|date| panel.dates().binary_search(&date).ok())
        .filter(|_| !needs_z || state.keep_history);
    let start = if let Some(day) = resume {
        day + 1
    } else {
        *state = State::default();
        first_target.saturating_sub(if needs_z { WINDOW } else { 0 })
    };
    state.keep_history |= needs_z;
    let prices = adjusted_prices(data, panel)?;
    let size = AlignedSource::new(data, panel, DatasetId::StockBarraDaily, "SIZE")?;
    let sectors = ClassificationMap::from_table(
        data.daily(DatasetId::StockSwClassification)?,
        ClassificationLevel::Sector,
    )?;
    let income =
        data.financial_reader(DatasetId::StockIncome, ReportTypePreference::consolidated())?;
    let eligible: Vec<_> = panel
        .instruments()
        .iter()
        .enumerate()
        .filter(|(_, code)| !is_bj_stock(code))
        .map(|(i, _)| i)
        .collect();
    let lower = add_months(panel.dates()[start], -12);
    let events = build_events(
        data.analyst_report_reader()?,
        panel,
        &income,
        lower,
        panel.dates()[last_target],
    )?;
    let mut events = events.into_iter().peekable();
    let mut active = Vec::new();
    let mut result: Vec<_> = outputs
        .iter()
        .map(|output| FactorSeries {
            spec: Pafr(*output).spec(),
            values: Vec::new(),
        })
        .collect();
    for day in start..=last_target {
        let date = panel.dates()[day];
        let window = RevisionWindow::new(date);
        let cache_floor = add_days(window.pair_lower, -14);
        state.market.means.retain(|&(a, _), _| a >= cache_floor);
        while events.peek().is_some_and(|event| event.date <= date) {
            let event = events.next().unwrap();
            if !window.contains(&event) {
                continue;
            }
            let (Some(a), Some(b)) = (
                anchor_day(panel, event.previous),
                anchor_day(panel, event.date),
            ) else {
                continue;
            };
            if let Some(between) =
                state
                    .market
                    .excess(panel, prices.values(), &eligible, event.stock, a, b)
            {
                active.push(ActiveEvent {
                    event,
                    report_day: b,
                    between,
                });
            }
        }
        active.retain(|row| window.contains(&row.event));
        let observations: Vec<_> = active
            .iter()
            .filter_map(|row| {
                let post = state.market.excess(
                    panel,
                    prices.values(),
                    &eligible,
                    row.event.stock,
                    row.report_day,
                    day,
                )?;
                Some((row.event.stock, row.event.revision, row.between, post))
            })
            .collect();
        let raw = aggregate(n, &observations);
        let standardized: Vec<_> = if state.keep_history {
            panel
                .instruments()
                .iter()
                .enumerate()
                .map(|(i, code)| {
                    state
                        .histories
                        .entry(code.clone())
                        .or_default()
                        .zscore(raw[i])
                })
                .collect()
        } else {
            vec![]
        };
        if panel.is_target_date(date) {
            let size_day = size.day(date);
            let size_values: Vec<_> = (0..n).map(|i| size.get(size_day, i)).collect();
            let mut groups = sectors.groups_for(date, panel.instruments());
            for (i, code) in panel.instruments().iter().enumerate() {
                if is_bj_stock(code) || !panel.is_present_offset(day * n + i) {
                    groups[i] = None;
                }
            }
            for (j, output) in outputs.iter().enumerate() {
                let input = if *output == Output::Raw {
                    &raw
                } else {
                    &standardized
                };
                let residual =
                    neutralize_and_fill_missing(input, &size_values, &groups);
                for (i, code) in panel.instruments().iter().enumerate() {
                    if panel.is_present_offset(day * n + i) {
                        result[j].values.push(FactorValue {
                            key: FactorRowKey::Daily {
                                trade_date: date,
                                ts_code: code.clone(),
                            },
                            value: residual[i],
                        });
                    }
                }
            }
        }
        if state.keep_history {
            let by_code: HashMap<_, _> = panel
                .instruments()
                .iter()
                .enumerate()
                .map(|(i, code)| (code.as_str(), raw[i]))
                .collect();
            for (code, history) in &mut state.histories {
                history.push(by_code.get(code.as_str()).copied().flatten());
            }
        }
        state.last_date = Some(date);
    }
    Ok(result)
}
