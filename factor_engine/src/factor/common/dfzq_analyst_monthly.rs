use std::any::Any;
use std::collections::{BTreeSet, HashMap};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, Lookback,
};
use crate::data::{DataPool, Table};
use crate::error::{err, Result};
use crate::factor::common::financial::{add_days, add_months, analyst_fiscal_years};
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_and_fill_missing};
use crate::factor::common::{
    ClassificationLevel, ClassificationMap, FinancialPitReader, ReportTypePreference,
};
use crate::factor::{Factor, FactorUpdatePolicy};
use crate::operators::cross_sectional::{cs_fill_mean_by_group, cs_neutralize_regression, cs_zscore};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Output {
    Fom,
    Roe,
    Ep,
    EpDiff,
    PegDiff,
}
const OUTPUTS: [Output; 5] = [
    Output::Fom,
    Output::Roe,
    Output::Ep,
    Output::EpDiff,
    Output::PegDiff,
];
impl Output {
    fn id(self) -> &'static str {
        match self {
            Self::Fom => "fom_123mean",
            Self::Roe => "rollingfyroe_resid_12mean",
            Self::Ep => "rollingepfy_12mean",
            Self::EpDiff => "rollingepfy_12mean_diff3",
            Self::PegDiff => "peg_diff6",
        }
    }
    fn lookback(self) -> usize {
        match self {
            Self::EpDiff => 110,
            Self::PegDiff => 190,
            Self::Roe => 95,
            _ => 32,
        }
    }
}

pub struct AnalystMonthly(pub Output);
impl Factor for AnalystMonthly {
    fn spec(&self) -> FactorSpec {
        let mut dependencies = vec![
            DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
            DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            DataRequest::financial_quarters(DatasetId::StockIncome, &["n_income_attr_p"], 16),
        ];
        if self.0 == Output::Fom {
            dependencies.push(DataRequest::new(
                DatasetId::StockAnalystReport,
                &[
                    "np",
                    "report_date",
                    "quarter",
                    "org_name",
                    "author_name",
                    "report_title",
                ],
            ));
        } else {
            let fields = if self.0 == Output::Roe {
                vec!["con_roe_fy0", "con_roe_fy1", "con_roe_fy2", "con_roe_fy3"]
            } else {
                vec!["con_np_fy0", "con_np_fy1", "con_np_fy2", "con_np_fy3"]
            };
            dependencies.push(DataRequest::new(DatasetId::StockConsensus, &fields));
            dependencies.push(DataRequest::new(
                DatasetId::StockDailyBasic,
                if self.0 == Output::Roe {
                    &["total_mv", "pb", "dv_ttm"]
                } else {
                    &["total_mv"]
                },
            ));
        }
        if self.0 == Output::Roe {
            dependencies.extend([
                DataRequest::financial_quarters(
                    DatasetId::StockBalanceSheet,
                    &["total_hldr_eqy_exc_min_int", "total_assets"],
                    16,
                ),
                DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
            ]);
        }
        FactorSpec {
            id: self.0.id().into(), aliases: vec![self.0.id().to_ascii_uppercase()], name: self.0.id().into(),
            asset_class: AssetClass::Stock, frequency: Frequency::Daily, version: "0.2.0".into(),
            tags: ["DFZQ", "analyst", "fundamental", "pit", "monthly_update", "daily", "neutralize", "size", "sector"].into_iter().map(str::to_string).collect(),
            description: format!("Month-end PIT analyst factor {:?}; hold final cross-section by stock until next month end. FY1=prior calendar year before Apr30, current calendar year from Apr30 inclusive; map consensus FY0..3 to absolute years using its own PIT/May1 anchor. No express/forecast announcements. FOM: 12 calendar months of annual report forecasts, N>=3 per year, actual annual parent profit/10000 preferred; average latest-day individual scores otherwise; sqrt(1+nonmissing mean over FY1..3). Rolling EP/ROE: (1-month/12)*FYk+month/12*FY(k+1); require each nonzero-weight input. Mean available FY1/FY2 rolls. ROE model: SW L1, ln(total_mv), 1/pb, parent profit TTM/average positive parent equity at q and q-4, asset YoY, ROE YoY difference, dv_ttm/100, adjusted RET60, min(ROE,0); ROE predictions divided by100. EP differences use raw month-end levels 3 months apart. PEG uses FY1 PE and decimal CAGR sqrt(FY2 NP/abs(actual FY0 NP))-1; zero denominators invalid, negative growth retained; output NEGATIVE 6-month raw PEG change. Before averaging FY signals, fill each annual FOM score, each rolled EP valuation, and each ROE residual separately with its SW L1 mean. Underlying predictions remain unfilled. All outputs: 1.5 IQR clipping, SW L1/SIZE neutralization, then cs_zscore; only the standalone PEG difference fills missing raw via zero residual for supported exposures, never post-fill composites. Excludes BJ. Requested outputs only; monthly state, no disk intermediates.", self.0),
            dependencies, intraday_raw_dependencies: vec![], lookback: Lookback { trading_days: self.0.lookback() },
        }
    }
    fn requirements_for_context(&self, context: &FactorContext) -> Vec<DataRequest> {
        // Every possible trading month-end is in the last calendar week. Avoid loading
        // daily consensus/basic/SIZE rows that cannot be used by this monthly model.
        let dates: Vec<_> = context
            .load_dates
            .iter()
            .copied()
            .filter(|d| d % 100 >= 22)
            .collect();
        self.spec()
            .dependencies
            .into_iter()
            .map(|r| {
                if matches!(
                    r.dataset,
                    DatasetId::StockConsensus
                        | DatasetId::StockDailyBasic
                        | DatasetId::StockBarraDaily
                ) {
                    r.with_explicit_dates(dates.clone())
                } else {
                    r
                }
            })
            .collect()
    }
    fn compute_provider_key(&self) -> String {
        "dfzq_analyst_monthly".into()
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
            .ok_or_else(|| err("missing monthly analyst output"))
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
                .downcast_mut()
                .ok_or_else(|| err("monthly analyst state mismatch"))?,
        )
    }
}

#[derive(Default)]
struct State {
    outputs: HashMap<Output, (i32, HashMap<String, Option<f64>>)>,
}
fn fy1(date: i32) -> i32 {
    date / 10000 - i32::from(date % 10000 < 430)
}
fn finite(x: f64) -> Option<f64> {
    x.is_finite().then_some(x)
}
fn mean(values: &[Option<f64>]) -> Option<f64> {
    let v: Vec<_> = values
        .iter()
        .flatten()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    if v.is_empty() {
        None
    } else {
        finite(v.iter().sum::<f64>() / v.len() as f64)
    }
}
fn blend(a: Option<f64>, b: Option<f64>, month: i32) -> Option<f64> {
    if month == 12 {
        return b;
    }
    finite(a? * (1.0 - month as f64 / 12.0) + b? * month as f64 / 12.0)
}
fn peg(mv: Option<f64>, p1: Option<f64>, p2: Option<f64>, base: Option<f64>) -> Option<f64> {
    let (mv, p1, p2, base) = (mv?, p1?, p2?, base?);
    if mv <= 0.0 || p1 == 0.0 || base == 0.0 || p2 < 0.0 {
        return None;
    }
    let growth = (p2 / base.abs()).sqrt() - 1.0;
    if growth == 0.0 {
        None
    } else {
        finite(mv / p1 / growth)
    }
}
fn clip_iqr(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut sorted: Vec<_> = values
        .iter()
        .flatten()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    if sorted.is_empty() {
        return vec![None; values.len()];
    }
    sorted.sort_by(f64::total_cmp);
    let quantile = |p: f64| {
        let x = p * (sorted.len() - 1) as f64;
        let i = x.floor() as usize;
        sorted[i] + (sorted[x.ceil() as usize] - sorted[i]) * (x - i as f64)
    };
    let (q1, q3) = (quantile(0.25), quantile(0.75));
    let (lo, hi) = (q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1));
    values
        .iter()
        .map(|v| v.filter(|v| v.is_finite()).map(|v| v.clamp(lo, hi)))
        .collect()
}
fn finish(
    raw: &[Option<f64>],
    size: &[Option<f64>],
    groups: &[Option<String>],
    fill_final: bool,
) -> Vec<Option<f64>> {
    let clipped = clip_iqr(raw);
    let residual = if fill_final {
        neutralize_and_fill_missing(&clipped, size, groups)
    } else {
        cs_neutralize_regression(&clipped, &[size], Some(groups), None)
    };
    cs_zscore(&residual)
}

// Store only requested month-end rows, not additional stock x day numeric panels.
struct Rows {
    values: HashMap<(i32, usize), Vec<Option<f64>>>,
}
impl Rows {
    fn new(
        table: &Table,
        columns: &[&str],
        dates: &BTreeSet<i32>,
        codes: &[String],
    ) -> Result<Self> {
        let stocks: HashMap<_, _> = codes
            .iter()
            .enumerate()
            .map(|(i, c)| (c.as_str(), i))
            .collect();
        let ts = table.required_utf8("ts_code")?;
        let ds = table.required_i32_date_cast("trade_date")?;
        let selected: Vec<_> = (0..table.len)
            .filter_map(|i| {
                let d = ds[i]?;
                if !dates.contains(&d) {
                    return None;
                }
                Some((i, (d, *stocks.get(ts[i].as_deref()?)?)))
            })
            .collect();
        let mut values: HashMap<_, _> = selected
            .iter()
            .map(|(_, k)| (*k, vec![None; columns.len()]))
            .collect();
        for (j, name) in columns.iter().enumerate() {
            let col = table.required_f64_cast(name)?;
            for &(i, k) in &selected {
                values.get_mut(&k).unwrap()[j] = col[i].filter(|v| v.is_finite());
            }
        }
        Ok(Self { values })
    }
    fn get(&self, date: i32, stock: usize, column: usize) -> Option<f64> {
        self.values
            .get(&(date, stock))?
            .get(column)
            .copied()
            .flatten()
    }
}

struct Reports<'a> {
    values: HashMap<(&'a str, i32), Vec<(i32, f64)>>,
}
impl<'a> Reports<'a> {
    fn new(table: impl Into<crate::data::SourceRowsReader<'a>>, lower: i32, upper: i32) -> Result<Self> {
        let table = table.into();
        let codes = table.required_utf8("ts_code")?;
        let dates = table.required_i32_date_cast("report_date")?;
        let years = table.required_utf8("quarter")?;
        let orgs = table.required_utf8("org_name")?;
        let authors = table.required_utf8("author_name")?;
        let titles = table.required_utf8("report_title")?;
        let np = table.required_f64_cast("np")?;
        let mut unique = HashMap::new();
        for i in 0..table.len {
            let (Some(c), Some(d), Some(y), Some(v)) = (
                codes[i].as_deref(),
                dates[i],
                years[i].as_deref(),
                np[i].filter(|v| v.is_finite()),
            ) else {
                continue;
            };
            if d <= lower || d > upper || is_bj_stock(c) {
                continue;
            }
            let Some(y) = y.strip_suffix("Q4").and_then(|s| s.parse::<i32>().ok()) else {
                continue;
            };
            unique.insert(
                (
                    c,
                    y,
                    d,
                    orgs[i].as_deref(),
                    authors[i].as_deref(),
                    titles[i].as_deref(),
                ),
                v,
            );
        }
        let mut values: HashMap<_, Vec<_>> = HashMap::new();
        for ((c, y, d, _, _, _), v) in unique {
            values.entry((c, y)).or_default().push((d, v));
        }
        for rows in values.values_mut() {
            rows.sort_unstable_by_key(|r| r.0);
        }
        Ok(Self { values })
    }
    fn score(&self, code: &str, year: i32, date: i32, actual: Option<Option<f64>>) -> Option<f64> {
        let rows = self.values.get(&(code, year))?;
        let start = rows.partition_point(|r| r.0 <= add_months(date, -12));
        let end = rows.partition_point(|r| r.0 <= date);
        fom_score(&rows[start..end], actual)
    }
}
fn fom_score(rows: &[(i32, f64)], actual: Option<Option<f64>>) -> Option<f64> {
    if rows.len() < 3 {
        return None;
    }
    let score = |base: f64| {
        rows.iter()
            .map(|(_, v)| {
                if *v < base {
                    1.0
                } else if *v > base {
                    -1.0
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            / rows.len() as f64
    };
    if let Some(value) = actual {
        return value.and_then(finite).map(score);
    }
    let latest = rows.last()?.0;
    mean(
        &rows
            .iter()
            .rev()
            .take_while(|r| r.0 == latest)
            .map(|r| Some(score(r.1)))
            .collect::<Vec<_>>(),
    )
}

fn roe_residual(
    y: &[Option<f64>],
    x: &[Vec<Option<f64>>],
    groups: &[Option<String>],
) -> Vec<Option<f64>> {
    let masked: Vec<_> = (0..y.len())
        .map(|i| {
            y[i].filter(|v| {
                v.is_finite()
                    && groups[i].is_some()
                    && x.iter().all(|c| c[i].is_some_and(f64::is_finite))
            })
        })
        .collect();
    let mut basis: Vec<Vec<Option<f64>>> = Vec::new();
    // Industry-demean and orthogonalize the eight controls on exactly the fit sample.
    // Redundant columns (e.g. no loss firms => the loss term is zero) do not invalidate OLS.
    for column in x {
        let column: Vec<_> = column
            .iter()
            .zip(&masked)
            .map(|(v, m)| if m.is_some() { *v } else { None })
            .collect();
        let mut v = cs_neutralize_regression(&column, &[], Some(groups), None);
        let norm = v.iter().flatten().map(|v| v * v).sum::<f64>().sqrt();
        if !norm.is_finite() || norm <= f64::EPSILON {
            continue;
        }
        for a in v.iter_mut().flatten() {
            *a /= norm;
        }
        for _ in 0..2 {
            for b in &basis {
                let dot: f64 = v
                    .iter()
                    .zip(b)
                    .filter_map(|(a, b)| a.zip(*b).map(|(a, b)| a * b))
                    .sum();
                for (a, b) in v.iter_mut().zip(b) {
                    if let (Some(a), Some(b)) = (a, b) {
                        *a -= dot * b;
                    }
                }
            }
        }
        let norm = v.iter().flatten().map(|v| v * v).sum::<f64>().sqrt();
        if norm <= 1e-10 {
            continue;
        }
        for a in v.iter_mut().flatten() {
            *a /= norm;
        }
        basis.push(v);
    }
    let refs: Vec<_> = basis.iter().map(Vec::as_slice).collect();
    cs_neutralize_regression(&masked, &refs, Some(groups), None)
}

fn historical_roe(
    income: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    code: &str,
    date: i32,
    end: i32,
) -> Option<f64> {
    let a = balance
        .record_for_end_date(code, date, end)?
        .column("total_hldr_eqy_exc_min_int")?;
    let b = balance
        .record_for_end_date(code, date, end - 10000)?
        .column("total_hldr_eqy_exc_min_int")?;
    if a <= 0.0 || b <= 0.0 {
        return None;
    }
    finite(income.ttm_sum_for_end_date(code, date, end, "n_income_attr_p")? / ((a + b) / 2.0))
}

fn compute(ids: &[String], data: &DataPool, state: &mut State) -> Result<Vec<FactorSeries>> {
    let outputs: Vec<_> = OUTPUTS
        .into_iter()
        .filter(|o| ids.iter().any(|id| id == o.id()))
        .collect();
    if outputs.is_empty() {
        return Ok(vec![]);
    }
    let panel = data.stock_universe_panel()?;
    let calendar = data.trading_calendar()?;
    let codes = panel.instruments();
    let n = codes.len();
    let mut anchors = BTreeSet::new();
    for &d in panel.dates().iter().filter(|d| panel.is_target_date(**d)) {
        if let Some(a) = calendar.month_end_on_or_before(d) {
            anchors.insert(a);
        }
    }
    let mut dates = anchors.clone();
    let lag_date = |d: i32, months: i32| {
        let first = d / 100 * 100 + 1;
        calendar.month_end_on_or_before(add_days(add_months(first, 1 - months), -1))
    };
    for &a in &anchors {
        if outputs.contains(&Output::EpDiff) {
            if let Some(d) = lag_date(a, 3) {
                dates.insert(d);
            }
        }
        if outputs.contains(&Output::PegDiff) {
            if let Some(d) = lag_date(a, 6) {
                dates.insert(d);
            }
        }
    }
    let size = Rows::new(
        data.daily(DatasetId::StockBarraDaily)?,
        &["SIZE"],
        &anchors,
        codes,
    )?;
    let sectors = ClassificationMap::from_table(
        data.daily(DatasetId::StockSwClassification)?,
        ClassificationLevel::Sector,
    )?;
    let annual =
        data.financial_reader(DatasetId::StockIncome, ReportTypePreference::consolidated())?;
    let has_values = outputs.iter().any(|o| *o != Output::Fom);
    let has_roe = outputs.contains(&Output::Roe);
    let has_np = outputs
        .iter()
        .any(|o| matches!(o, Output::Ep | Output::EpDiff | Output::PegDiff));
    let basic = if has_values {
        Some(Rows::new(
            data.daily(DatasetId::StockDailyBasic)?,
            if has_roe {
                &["total_mv", "pb", "dv_ttm"]
            } else {
                &["total_mv"]
            },
            &dates,
            codes,
        )?)
    } else {
        None
    };
    let np = if has_np {
        Some(Rows::new(
            data.daily(DatasetId::StockConsensus)?,
            &["con_np_fy0", "con_np_fy1", "con_np_fy2", "con_np_fy3"],
            &dates,
            codes,
        )?)
    } else {
        None
    };
    let predicted_roe = if has_roe {
        Some(Rows::new(
            data.daily(DatasetId::StockConsensus)?,
            &["con_roe_fy0", "con_roe_fy1", "con_roe_fy2", "con_roe_fy3"],
            &anchors,
            codes,
        )?)
    } else {
        None
    };
    let reports = if outputs.contains(&Output::Fom) && !anchors.is_empty() {
        Some(Reports::new(
            data.analyst_report_reader()?,
            add_months(*anchors.first().unwrap(), -12),
            *anchors.last().unwrap(),
        )?)
    } else {
        None
    };
    let single = if has_roe {
        Some(data.financial_reader(
            DatasetId::StockIncome,
            ReportTypePreference::income_single_quarter(),
        )?)
    } else {
        None
    };
    let balance = if has_roe {
        Some(data.financial_reader(
            DatasetId::StockBalanceSheet,
            ReportTypePreference::balance_sheet_consolidated(),
        )?)
    } else {
        None
    };
    let mut ret_dates = anchors.clone();
    let mut ret_start = HashMap::new();
    if has_roe {
        for &a in &anchors {
            if let Ok(idx) = panel.dates().binary_search(&a) {
                if idx >= 60 {
                    let d = panel.dates()[idx - 60];
                    ret_dates.insert(d);
                    ret_start.insert(a, d);
                }
            }
        }
    }
    let prices = if has_roe {
        Some((
            Rows::new(
                data.daily(DatasetId::StockDailyPv)?,
                &["close"],
                &ret_dates,
                codes,
            )?,
            Rows::new(
                data.daily(DatasetId::StockAdjFactor)?,
                &["adj_factor"],
                &ret_dates,
                codes,
            )?,
        ))
    } else {
        None
    };
    // Historical raw values are reconstructed at THEIR as-of dates, never at today's PIT date.
    let mut ep = HashMap::new();
    let mut pegs = HashMap::new();
    if let (Some(np), Some(basic)) = (&np, &basic) {
        for &d in &dates {
            let mut ep_parts = vec![vec![None; n]; 2];
            let mut pv = vec![None; n];
            let mut groups = sectors.groups_for(d, codes);
            for (i, c) in codes.iter().enumerate() {
                if is_bj_stock(c) {
                    groups[i] = None;
                    continue;
                }
                let target = fy1(d);
                let source = analyst_fiscal_years(d, c, &annual)[0];
                let values: [Option<f64>; 3] = std::array::from_fn(|k| {
                    let year = target + k as i32;
                    let idx = usize::try_from(year - source).ok()?;
                    np.get(d, i, idx)
                });
                let mv = basic.get(d, i, 0).filter(|v| *v > 0.0);
                if mv.is_none() { groups[i] = None; }
                for k in 0..2 {
                    ep_parts[k][i] = blend(values[k], values[k + 1], d / 100 % 100)
                        .and_then(|v| finite(v / mv?));
                }
                if outputs.contains(&Output::PegDiff) {
                    let value = |year| {
                        annual
                            .record_for_end_date(c, d, year * 10000 + 1231)
                            .map(|r| r.column("n_income_attr_p").map(|v| v / 10000.0))
                    };
                    pv[i] = peg(
                        mv,
                        value(target).unwrap_or(values[0]),
                        values[1],
                        value(target - 1).flatten(),
                    );
                }
            }
            let ep_parts: Vec<_> = ep_parts.iter()
                .map(|v| cs_fill_mean_by_group(v, &groups)).collect();
            ep.insert(d, (0..n).map(|i| mean(&[ep_parts[0][i], ep_parts[1][i]])).collect());
            pegs.insert(d, pv);
        }
    }
    let mut result: Vec<_> = outputs
        .iter()
        .map(|o| FactorSeries {
            spec: AnalystMonthly(*o).spec(),
            values: vec![],
        })
        .collect();
    for (day, &date) in panel.dates().iter().enumerate() {
        if !panel.is_target_date(date) {
            continue;
        }
        if let Some(anchor) = calendar.month_end_on_or_before(date) {
            let pending: Vec<_> = outputs
                .iter()
                .copied()
                .filter(|o| state.outputs.get(o).is_none_or(|v| v.0 != anchor))
                .collect();
            if !pending.is_empty() {
                let anchor_day = panel
                    .dates()
                    .binary_search(&anchor)
                    .map_err(|_| err("monthly analyst anchor missing from warmup"))?;
                let mut groups = sectors.groups_for(anchor, codes);
                for (i, c) in codes.iter().enumerate() {
                    if is_bj_stock(c) || !panel.is_present_offset(anchor_day * n + i) {
                        groups[i] = None;
                    }
                }
                let exposure: Vec<_> = (0..n).map(|i| size.get(anchor, i, 0)).collect();
                for output in pending {
                    let mut raw = vec![None; n];
                    match output {
                        Output::Fom => {
                            if let Some(reports) = &reports {
                                let mut parts = vec![vec![None; n]; 3];
                                for (i, c) in codes.iter().enumerate() {
                                    if groups[i].is_none() {
                                        continue;
                                    }
                                    for k in 0..3 {
                                        let y = fy1(anchor) + k as i32;
                                        let actual = annual
                                            .record_for_end_date(c, anchor, y * 10000 + 1231)
                                            .map(|r| {
                                                r.column("n_income_attr_p").map(|v| v / 10000.0)
                                            });
                                        parts[k][i] = reports.score(c, y, anchor, actual);
                                    }
                                }
                                let parts: Vec<_> = parts.iter()
                                    .map(|v| cs_fill_mean_by_group(v, &groups)).collect();
                                for i in 0..n {
                                    raw[i] = mean(&[parts[0][i], parts[1][i], parts[2][i]])
                                        .and_then(|v| finite((1.0 + v).max(0.0).sqrt()));
                                }
                            }
                        }
                        Output::Ep => raw = ep.get(&anchor).cloned().unwrap_or(raw),
                        Output::EpDiff | Output::PegDiff => {
                            let (cache, months, sign) = if output == Output::EpDiff {
                                (&ep, 3, 1.0)
                            } else {
                                (&pegs, 6, -1.0)
                            };
                            if let Some(past) = lag_date(anchor, months) {
                                if let (Some(a), Some(b)) = (cache.get(&anchor), cache.get(&past)) {
                                    for i in 0..n {
                                        raw[i] = a[i]
                                            .zip(b[i])
                                            .and_then(|(a, b)| finite(sign * (a - b)));
                                    }
                                }
                            }
                        }
                        Output::Roe => {
                            let basic = basic.as_ref().unwrap();
                            let pred = predicted_roe.as_ref().unwrap();
                            let inc = single.as_ref().unwrap();
                            let bal = balance.as_ref().unwrap();
                            let (price, adj) = prices.as_ref().unwrap();
                            let mut x = vec![vec![None; n]; 8];
                            let mut y = vec![vec![None; n]; 2];
                            for (i, c) in codes.iter().enumerate() {
                                if groups[i].is_none() {
                                    continue;
                                }
                                let source = analyst_fiscal_years(anchor, c, &annual)[0];
                                let v: [Option<f64>; 3] = std::array::from_fn(|k| {
                                    usize::try_from(fy1(anchor) + k as i32 - source)
                                        .ok()
                                        .and_then(|j| pred.get(anchor, i, j))
                                        .map(|v| v / 100.0)
                                });
                                y[0][i] = blend(v[0], v[1], anchor / 100 % 100);
                                y[1][i] = blend(v[1], v[2], anchor / 100 % 100);
                                x[0][i] = basic.get(anchor, i, 0).filter(|v| *v > 0.0).map(f64::ln);
                                x[1][i] = basic
                                    .get(anchor, i, 1)
                                    .filter(|v| *v != 0.0)
                                    .and_then(|v| finite(1.0 / v));
                                x[5][i] = basic.get(anchor, i, 2).map(|v| v / 100.0);
                                x[6][i] = (|| {
                                    let d = *ret_start.get(&anchor)?;
                                    let a = price.get(anchor, i, 0)? * adj.get(anchor, i, 0)?;
                                    let b = price.get(d, i, 0)? * adj.get(d, i, 0)?;
                                    if a <= 0.0 || b <= 0.0 {
                                        None
                                    } else {
                                        finite(a / b - 1.0)
                                    }
                                })();
                                if let Some(q) = inc.latest_quarter_end_date(c, anchor) {
                                    x[2][i] = historical_roe(inc, bal, c, anchor, q);
                                    x[3][i] = (|| {
                                        let a = bal
                                            .record_for_end_date(c, anchor, q)?
                                            .column("total_assets")?;
                                        let b = bal
                                            .record_for_end_date(c, anchor, q - 10000)?
                                            .column("total_assets")?;
                                        if b <= 0.0 {
                                            None
                                        } else {
                                            finite(a / b - 1.0)
                                        }
                                    })();
                                    x[4][i] = x[2][i]
                                        .zip(historical_roe(inc, bal, c, anchor, q - 10000))
                                        .and_then(|(a, b)| finite(a - b));
                                    x[7][i] = x[2][i].map(|v| v.min(0.0));
                                }
                            }
                            let a = cs_fill_mean_by_group(&roe_residual(&y[0], &x, &groups), &groups);
                            let b = cs_fill_mean_by_group(&roe_residual(&y[1], &x, &groups), &groups);
                            raw = (0..n).map(|i| mean(&[a[i], b[i]])).collect();
                        }
                    }
                    for i in 0..n {
                        if groups[i].is_none() {
                            raw[i] = None;
                        }
                    }
                    state.outputs.insert(
                        output,
                        (
                            anchor,
                            codes
                                .iter()
                                .cloned()
                                .zip(finish(&raw, &exposure, &groups, output == Output::PegDiff))
                                .collect(),
                        ),
                    );
                }
            }
        }
        for (j, o) in outputs.iter().enumerate() {
            for (i, c) in codes.iter().enumerate() {
                if !panel.is_present_offset(day * n + i) {
                    continue;
                }
                let value = if is_bj_stock(c) {
                    None
                } else {
                    state
                        .outputs
                        .get(o)
                        .filter(|v| Some(v.0) == calendar.month_end_on_or_before(date))
                        .and_then(|v| v.1.get(c).copied().flatten())
                };
                result[j].values.push(FactorValue {
                    key: FactorRowKey::Daily {
                        trade_date: date,
                        ts_code: c.clone(),
                    },
                    value,
                });
            }
        }
    }
    Ok(result)
}
