use std::any::Any;
use std::collections::{BTreeMap, HashMap, VecDeque};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::{DataPool, Table};
use crate::error::{err, Result};
use crate::factor::common::financial::{add_days, analyst_fiscal_years};
use crate::factor::common::stock_daily_ops::is_bj_stock;
use crate::factor::common::{
    ClassificationLevel, ClassificationMap, DailyPanel, FinancialPitReader, PanelColumn,
    ReportTypePreference,
};
use crate::factor::{Factor, FactorUpdatePolicy};
use crate::operators::cross_sectional::{cs_neutralize_regression, cs_regression_residual};

const DAYS: i32 = 90;
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
            asset_class: AssetClass::Stock, frequency: Frequency::Daily, version: "0.1.0".into(),
            tags: ["deprecated", "FZZQ", "analyst", "fundamental", "pit", "forecast_revision", "momentum_residual", "daily", "neutralize", "size", "sector"].into_iter().map(str::to_string).collect(),
            description: format!("Daily PAFR approximation: match org/stock/absolute Q4 forecast year. Deduplicate date/org/author/title/year (last finite source row wins), then equally average reports for each institution/date/year. Current report's FY2 follows PIT annual disclosure with May-1 fallback. Compare strictly previous valid date in the same year, even if that year had another FY label then. Both dates must be in (T-90 calendar days,T]. Revision=(current-previous)/abs(previous), zero base invalid, clipped +/-0.25. Adjusted-close interval return minus SAME-INTERVAL equal-weight cross-sectional mean, excludes BJ; endpoints last session <= report date, missing endpoints invalid. Daily pooled event OLS with intercept: revision~between-report excess return, then residual~report-to-T excess return; constant regressor reduces to demeaning. Stocks need >=3 complete institution-day revision events before both fits and aggregation. {} Final daily SW L1/SIZE neutralization, no missing-value fill or extra winsorization. No AFR output. Benchmark cache bounded to active intervals; z-score history cached by instrument, never recomputed using future data.",
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
            // 90 trading sessions safely cover 90 calendar days plus the prior-session anchor.
            lookback: Lookback { trading_days: DAYS as usize + if self.0 == Output::Zscore { WINDOW } else { 0 } },
        }
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

fn revision(current: f64, previous: f64) -> Option<f64> {
    if !current.is_finite() || !previous.is_finite() || previous == 0.0 {
        return None;
    }
    let value = (current - previous) / previous.abs();
    value.is_finite().then(|| value.clamp(-0.25, 0.25))
}

fn build_events(
    table: &Table,
    panel: &DailyPanel,
    income: &FinancialPitReader<'_>,
    lower: i32,
    upper: i32,
) -> Result<Vec<Event>> {
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
    let close = panel.column_from_table(data.daily(DatasetId::StockDailyPv)?, "close")?;
    let adj = panel.column_from_table(data.daily(DatasetId::StockAdjFactor)?, "adj_factor")?;
    close.zip_binary(&adj, |close, adj| {
        let (close, adj) = (close?, adj?);
        let value = close * adj;
        (close > 0.0 && adj > 0.0 && value.is_finite()).then_some(value)
    })
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
    let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;
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
    let lower = add_days(panel.dates()[start], -DAYS);
    let events = build_events(
        data.daily(DatasetId::StockAnalystReport)?,
        panel,
        &income,
        lower,
        panel.dates()[last_target],
    )?;
    let mut events = events.into_iter().peekable();
    let mut active = Vec::new();
    let mut columns = vec![vec![None; panel.shape_len()]; outputs.len()];
    for day in start..=last_target {
        let date = panel.dates()[day];
        let lower = add_days(date, -DAYS);
        let cache_floor = add_days(lower, -14);
        state.market.means.retain(|&(a, _), _| a >= cache_floor);
        while events.peek().is_some_and(|event| event.date <= date) {
            let event = events.next().unwrap();
            if event.previous <= lower {
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
        active.retain(|row| row.event.previous > lower);
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
                let residual = cs_neutralize_regression(
                    input,
                    &[&size.values()[day * n..(day + 1) * n]],
                    Some(&groups),
                    None,
                );
                columns[j][day * n..(day + 1) * n].copy_from_slice(&residual);
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
    outputs
        .into_iter()
        .zip(columns)
        .map(|(output, values)| {
            Ok(panel
                .column_from_values(values)?
                .to_factor_series(Pafr(output).spec()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::FactorRowKey;
    use crate::data::ColumnData;

    fn strings(values: impl IntoIterator<Item = impl AsRef<str>>) -> ColumnData {
        ColumnData::Utf8(
            values
                .into_iter()
                .map(|v| Some(v.as_ref().to_string()))
                .collect(),
        )
    }
    const CODES: [&str; 8] = [
        "000001.SZ",
        "000002.SZ",
        "000003.SZ",
        "000004.SZ",
        "000005.SZ",
        "000006.SZ",
        "000007.SZ",
        "830001.BJ",
    ];

    fn dates() -> Vec<i32> {
        (0..650)
            .filter(|i| (i + 1) % 7 < 5)
            .map(|i| add_days(20240102, i))
            .take(450)
            .collect()
    }

    fn fixture(target_start: usize, target_end: usize, lookback: usize, reverse: bool) -> DataPool {
        let dates = dates();
        let context = FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: dates[target_start],
            end_date: dates[target_end],
            load_start_date: dates[target_start.saturating_sub(lookback)],
            load_dates: dates[target_start.saturating_sub(lookback)..=target_end].to_vec(),
            target_dates: dates[target_start..=target_end].to_vec(),
        };
        let basic = Table::new(BTreeMap::from([
            ("ts_code".into(), strings(CODES)),
            ("list_date".into(), ColumnData::I32(vec![Some(20200101); 8])),
            ("delist_date".into(), ColumnData::I32(vec![None; 8])),
        ]))
        .unwrap();
        let classification = Table::new(BTreeMap::from([
            ("ts_code".into(), strings(CODES)),
            ("in_date".into(), ColumnData::I32(vec![Some(20200101); 8])),
            ("out_date".into(), ColumnData::I32(vec![None; 8])),
            (
                "l1_code".into(),
                strings(["a", "a", "a", "a", "b", "b", "b", "a"]),
            ),
        ]))
        .unwrap();
        let row_codes = strings(dates.iter().flat_map(|_| CODES));
        let row_dates =
            ColumnData::I32(dates.iter().flat_map(|&date| vec![Some(date); 8]).collect());
        let prices = Table::new(BTreeMap::from([
            ("ts_code".into(), row_codes.clone()),
            ("trade_date".into(), row_dates.clone()),
            (
                "close".into(),
                ColumnData::F64(
                    (0..dates.len())
                        .flat_map(|d| {
                            (0..8).map(move |s| {
                                let d = d as f64;
                                let s = s as f64;
                                Some(
                                    100.0
                                        * (0.0008 * d * (s + 1.0)
                                            + 0.07 * (d * 0.19 + s).sin()
                                            + 0.03 * (d * 0.07 * (s + 1.0)).cos())
                                        .exp(),
                                )
                            })
                        })
                        .collect(),
                ),
            ),
        ]))
        .unwrap();
        let adj = Table::new(BTreeMap::from([
            ("ts_code".into(), row_codes.clone()),
            ("trade_date".into(), row_dates.clone()),
            (
                "adj_factor".into(),
                ColumnData::F64(vec![Some(1.0); 8 * dates.len()]),
            ),
        ]))
        .unwrap();
        let barra = Table::new(BTreeMap::from([
            ("ts_code".into(), row_codes),
            ("trade_date".into(), row_dates),
            (
                "SIZE".into(),
                ColumnData::F64(
                    dates
                        .iter()
                        .flat_map(|_| (0..8).map(|s| Some((s + 1) as f64)))
                        .collect(),
                ),
            ),
        ]))
        .unwrap();
        let income = Table::new(BTreeMap::from([
            ("ts_code".into(), strings([CODES, CODES].concat())),
            (
                "ann_date".into(),
                ColumnData::I32([vec![Some(20240315); 8], vec![Some(20250315); 8]].concat()),
            ),
            ("f_ann_date".into(), ColumnData::I32(vec![None; 16])),
            (
                "end_date".into(),
                ColumnData::I32([vec![Some(20231231); 8], vec![Some(20241231); 8]].concat()),
            ),
            ("report_type".into(), ColumnData::I64(vec![Some(1); 16])),
            ("update_flag".into(), ColumnData::I64(vec![Some(0); 16])),
        ]))
        .unwrap();
        let mut rc = Vec::new();
        let mut rd = Vec::new();
        let mut org = Vec::new();
        let mut quarter = Vec::new();
        let mut np = Vec::new();
        for (d, &date) in dates.iter().enumerate().filter(|(d, _)| d % 7 == 0) {
            for (s, &code) in CODES.iter().enumerate() {
                for o in 0..3 {
                    for year in 2024..=2027 {
                        rc.push(code);
                        rd.push(Some(date));
                        org.push(format!("org{o}"));
                        quarter.push(format!("{year}Q4"));
                        let x = d as f64;
                        let y = s as f64;
                        np.push(Some(
                            100.0
                                + 0.1 * x * (y + 1.0)
                                + 5.0 * (x * 0.3 + o as f64 * 0.9).sin()
                                + 3.0 * (x * 0.07 * (y + 1.0)).cos(),
                        ));
                    }
                }
            }
        }
        let report_count = rc.len();
        let reports = Table::new(BTreeMap::from([
            ("ts_code".into(), strings(rc)),
            ("report_date".into(), ColumnData::I32(rd)),
            ("org_name".into(), strings(org)),
            ("quarter".into(), strings(quarter)),
            ("np".into(), ColumnData::F64(np)),
            ("author_name".into(), strings(vec!["author"; report_count])),
            ("report_title".into(), strings(vec!["title"; report_count])),
        ]))
        .unwrap();
        let mut tables = HashMap::from([
            (DatasetId::StockBasic, basic),
            (DatasetId::StockSwClassification, classification),
            (DatasetId::StockDailyPv, prices),
            (DatasetId::StockAdjFactor, adj),
            (DatasetId::StockBarraDaily, barra),
            (DatasetId::StockIncome, income),
            (DatasetId::StockAnalystReport, reports),
        ]);
        if reverse {
            for table in tables.values_mut() {
                *table = table
                    .take(&(0..table.len).rev().collect::<Vec<_>>())
                    .unwrap();
            }
        }
        DataPool::from_daily_tables_for_test(tables, &context).unwrap()
    }

    fn value(series: &FactorSeries, date: i32, code: &str) -> Option<f64> {
        series
            .values
            .iter()
            .find_map(|row| match &row.key {
                FactorRowKey::Daily {
                    trade_date,
                    ts_code,
                } if *trade_date == date && ts_code == code => Some(row.value),
                _ => None,
            })
            .unwrap()
    }
    fn assert_close(a: Option<f64>, b: Option<f64>) {
        match (a, b) {
            (None, None) => (),
            (Some(a), Some(b)) => assert!((a - b).abs() < 1e-8, "{a} != {b}"),
            _ => panic!("{a:?} != {b:?}"),
        }
    }

    #[test]
    fn pafr_revision_and_shared_pit_fy2() {
        assert_eq!(revision(-90.0, -100.0), Some(0.1));
        assert_eq!(revision(200.0, 100.0), Some(0.25));
        assert_eq!(revision(-200.0, -100.0), Some(-0.25));
        assert_eq!(revision(1.0, 0.0), None);
        assert_eq!(revision(f64::NAN, 1.0), None);
        let pool = fixture(342, 343, 342, false);
        let income = pool
            .financial_reader(DatasetId::StockIncome, ReportTypePreference::consolidated())
            .unwrap();
        assert_eq!(analyst_fiscal_years(20240314, CODES[0], &income)[2], 2024);
        assert_eq!(analyst_fiscal_years(20240315, CODES[0], &income)[2], 2025);
        assert_eq!(analyst_fiscal_years(20240430, "missing", &income)[2], 2024);
        assert_eq!(analyst_fiscal_years(20240501, "missing", &income)[2], 2025);
    }

    #[test]
    fn pafr_same_day_average_pairing_and_absolute_year() {
        let pool = fixture(342, 343, 342, false);
        let panel = pool.stock_universe_panel().unwrap();
        let income = pool
            .financial_reader(DatasetId::StockIncome, ReportTypePreference::consolidated())
            .unwrap();
        let table = Table::new(BTreeMap::from([
            ("ts_code".into(), strings(vec![CODES[0]; 7])),
            (
                "report_date".into(),
                ColumnData::I32(
                    [
                        20240310, 20240401, 20240401, 20240401, 20240402, 20240402, 20240310,
                    ]
                    .map(Some)
                    .to_vec(),
                ),
            ),
            (
                "quarter".into(),
                strings([
                    "2025Q4", "2025Q4", "2025Q4", "2025Q4", "2024Q4", "2025Q4", "2024Q4",
                ]),
            ),
            (
                "org_name".into(),
                strings(["a", "a", "a", "a", "a", "b", "a"]),
            ),
            ("author_name".into(), strings(["x"; 7])),
            (
                "report_title".into(),
                strings(["old", "r1", "r1", "r2", "r3", "r4", "old"]),
            ),
            (
                "np".into(),
                ColumnData::F64(
                    [100.0, 110.0, 110.0, 130.0, 200.0, 300.0, 1.0]
                        .map(Some)
                        .to_vec(),
                ),
            ),
        ]))
        .unwrap();
        let events = build_events(&table, panel, &income, 20240301, 20240430).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!((events[0].previous, events[0].date), (20240310, 20240401));
        assert_close(Some(events[0].revision), Some(0.2));
        assert!(build_events(&table, panel, &income, 20240310, 20240430)
            .unwrap()
            .is_empty());
        assert!(build_events(&table, panel, &income, 20240301, 20240331)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn pafr_interval_mean_excludes_bj_and_is_not_compounded_daily_mean() {
        let pool = fixture(0, 2, 0, false);
        let panel = pool.stock_universe_panel().unwrap();
        let mut prices = vec![None; panel.shape_len()];
        for (stock, path) in [
            (0, [100.0, 110.0, 100.0]),
            (1, [100.0, 90.0, 100.0]),
            (7, [100.0, 200.0, 400.0]),
        ] {
            for (day, price) in path.into_iter().enumerate() {
                prices[day * 8 + stock] = Some(price);
            }
        }
        let eligible: Vec<_> = panel
            .instruments()
            .iter()
            .enumerate()
            .filter(|(_, c)| !is_bj_stock(c))
            .map(|(i, _)| i)
            .collect();
        let mut cache = MarketCache::default();
        assert_eq!(cache.mean(panel, &prices, &eligible, 0, 2), Some(0.0));
        assert_eq!(cache.excess(panel, &prices, &eligible, 0, 0, 2), Some(0.0));
        assert_eq!(cache.means.len(), 1);
        assert!(anchor_day(panel, 20230101).is_none());
        assert_eq!(anchor_day(panel, 20240106), Some(2));
    }

    #[test]
    fn pafr_two_stage_event_residuals_then_stock_aggregation() {
        let rows = [
            (0, 0.1, 0.2, 0.1),
            (0, 0.2, 0.1, 0.3),
            (0, -0.1, -0.1, 0.0),
            (1, 0.05, 0.3, -0.2),
            (1, 0.15, 0.2, 0.4),
            (1, 0.0, 0.1, -0.1),
            (2, 0.25, 2.0, 3.0),
        ];
        let raw = aggregate(3, &rows);
        let y: Vec<_> = rows[..6].iter().map(|r| Some(r.1)).collect();
        let pre: Vec<_> = rows[..6].iter().map(|r| Some(r.2)).collect();
        let post: Vec<_> = rows[..6].iter().map(|r| Some(r.3)).collect();
        let expected = cs_regression_residual(&cs_regression_residual(&y, &pre), &post);
        assert_close(
            raw[0],
            Some(expected[..3].iter().flatten().sum::<f64>() / 3.0),
        );
        assert_close(
            raw[1],
            Some(expected[3..].iter().flatten().sum::<f64>() / 3.0),
        );
        assert_eq!(raw[2], None);
        assert_ne!(
            raw,
            aggregate(
                3,
                &rows
                    .iter()
                    .map(|r| (r.0, r.1, r.2, r.3 * r.3))
                    .collect::<Vec<_>>()
            )
        );
    }

    #[test]
    fn pafr_zscore_excludes_current_population_min120_and_eviction() {
        let mut history = RollingHistory::default();
        for i in 0..119 {
            history.push(Some(i as f64));
        }
        assert_eq!(history.zscore(Some(999.0)), None);
        history.push(Some(119.0));
        assert_close(
            history.zscore(Some(120.0)),
            Some((120.0 - 59.5) / ((120_f64.powi(2) - 1.0) / 12.0).sqrt()),
        );
        for i in 120..700 {
            history.push(if i % 11 == 0 {
                None
            } else {
                Some((i as f64).sin())
            });
        }
        assert_eq!(history.values.len(), WINDOW);
        let values: Vec<_> = history.values.iter().flatten().copied().collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let std =
            (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt();
        assert_close(history.zscore(Some(2.0)), Some((2.0 - mean) / std));
        for _ in 0..WINDOW {
            history.push(Some(0.1));
        }
        assert_eq!(history.zscore(Some(1.0)), None);
    }

    #[test]
    fn pafr_daily_provider_subsets_warmup_batches_and_no_future_output() {
        let ids = ["pafr".into(), "pafr_zscore".into()];
        let dates = dates();
        let full = compute(&ids, &fixture(342, 449, 342, false), &mut State::default()).unwrap();
        assert!(full[1].values.iter().any(|v| v.value.is_some()));
        assert!(full[0].values.iter().all(|v| match v.key {
            FactorRowKey::Daily { trade_date, .. } => trade_date >= dates[342],
            _ => false,
        }));
        assert!(full[0]
            .values
            .iter()
            .filter(|v| match &v.key {
                FactorRowKey::Daily { ts_code, .. } => ts_code.ends_with(".BJ"),
                _ => false,
            })
            .all(|v| v.value.is_none()));
        let mut state = State::default();
        for (start, end) in [(342, 389), (390, 449)] {
            let result = compute(&ids, &fixture(start, end, 342, true), &mut state).unwrap();
            for (j, series) in result.iter().enumerate() {
                for &date in &dates[start..=end] {
                    for code in CODES {
                        assert_close(value(series, date, code), value(&full[j], date, code));
                    }
                }
            }
        }
        assert!(state.histories.values().all(|h| h.values.len() <= WINDOW));
        assert!(state.market.means.len() < 10000);
        let cold = compute(&ids, &fixture(449, 449, 342, true), &mut State::default()).unwrap();
        let raw_only = compute(
            &ids[..1],
            &fixture(449, 449, 90, false),
            &mut State::default(),
        )
        .unwrap();
        let z_only = compute(
            &ids[1..],
            &fixture(449, 449, 342, false),
            &mut State::default(),
        )
        .unwrap();
        assert_eq!(raw_only.len(), 1);
        assert_eq!(z_only.len(), 1);
        for code in CODES {
            for j in 0..2 {
                assert_close(
                    value(&cold[j], dates[449], code),
                    value(&full[j], dates[449], code),
                );
            }
            assert_close(
                value(&raw_only[0], dates[449], code),
                value(&full[0], dates[449], code),
            );
            assert_close(
                value(&z_only[0], dates[449], code),
                value(&full[1], dates[449], code),
            );
        }
        assert!(CODES[..7]
            .iter()
            .any(|code| value(&full[0], dates[448], code) != value(&full[0], dates[449], code)));
    }

    #[test]
    fn pafr_metadata_requested_dependencies() {
        for output in [Output::Raw, Output::Zscore] {
            let spec = Pafr(output).spec();
            for tag in ["FZZQ", "analyst", "fundamental"] {
                assert!(spec.tags.contains(&tag.into()));
            }
            assert!(!spec
                .dependencies
                .iter()
                .any(|r| r.dataset == DatasetId::IndexDaily));
            assert_eq!(Pafr(output).compute_provider_key(), "fzzq_pafr");
        }
        assert_eq!(Pafr(Output::Zscore).spec().lookback.trading_days, 342);
        assert_eq!(Pafr(Output::Raw).spec().lookback.trading_days, 90);
    }
}
