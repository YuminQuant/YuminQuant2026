use std::any::Any;
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::calendar::TradingCalendar;
use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, Lookback,
};
use crate::data::{DataPool, SourceRowsReader};
use crate::error::{err, Result};
use crate::factor::common::financial::{add_days, add_months, analyst_fiscal_years};
use crate::factor::common::{
    ClassificationLevel, ClassificationMap, DailyPanel, FinancialPitReader, PanelColumn,
    ReportTypePreference,
};
use crate::factor::{Factor, FactorUpdatePolicy};
use crate::operators::{cs_neutralize_regression, cs_zscore};

const EQUITY: &str = "total_hldr_eqy_exc_min_int";
const PROFIT: &str = "n_income_attr_p";
const FORECASTS: [&str; 4] = ["con_np_fy0", "con_np_fy1", "con_np_fy2", "con_np_fy3"];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Output {
    Coverage,
    RatingSize,
    RatingReversal,
    RatingValue,
    RatingRoe,
    Revision,
}
const OUTPUTS: [Output; 6] = [
    Output::Coverage,
    Output::RatingSize,
    Output::RatingReversal,
    Output::RatingValue,
    Output::RatingRoe,
    Output::Revision,
];

impl Output {
    pub fn id(self) -> &'static str {
        match self {
            Self::Coverage => "abnormal_analyst_coverage",
            Self::RatingSize => "improved_analyst_rating_size",
            Self::RatingReversal => "improved_analyst_rating_reversal",
            Self::RatingValue => "improved_analyst_rating_value",
            Self::RatingRoe => "improved_analyst_rating_roe",
            Self::Revision => "analyst_earnings_revision",
        }
    }
    fn rating_index(self) -> Option<usize> {
        match self {
            Self::RatingSize => Some(0),
            Self::RatingReversal => Some(1),
            Self::RatingValue => Some(2),
            Self::RatingRoe => Some(3),
            _ => None,
        }
    }
}

pub struct HtzqAnalyst(pub Output);
impl Factor for HtzqAnalyst {
    fn spec(&self) -> FactorSpec {
        let mut fields = vec!["report_date", "org_name", "author_name", "report_title"];
        if self.0.rating_index().is_some() {
            fields.push("rating");
        }
        if self.0 == Output::Revision {
            fields.extend(["np", "quarter"]);
        }
        let mut dependencies = vec![
            DataRequest::new(DatasetId::StockAnalystReport, &fields),
            DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
        ];
        if self.0 != Output::Coverage {
            dependencies.push(DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]));
        }
        match self.0 {
            Output::Coverage => dependencies.push(DataRequest::new(
                DatasetId::StockDailyBasic,
                &["total_mv", "turnover_rate_f"],
            )),
            Output::RatingSize => {
                dependencies.push(DataRequest::new(DatasetId::StockDailyBasic, &["total_mv"]))
            }
            Output::RatingValue => {
                dependencies.push(DataRequest::new(DatasetId::StockDailyBasic, &["pb"]))
            }
            _ => {}
        }
        if matches!(
            self.0,
            Output::Coverage | Output::RatingReversal | Output::Revision
        ) {
            dependencies.extend([
                DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
            ]);
        }
        if matches!(self.0, Output::RatingRoe | Output::Revision) {
            dependencies.push(DataRequest::financial_quarters(
                DatasetId::StockIncome,
                &[PROFIT],
                if self.0 == Output::RatingRoe { 24 } else { 12 },
            ));
        }
        if self.0 == Output::RatingRoe {
            dependencies.push(DataRequest::financial_quarters(
                DatasetId::StockBalanceSheet,
                &[EQUITY],
                24,
            ));
        }
        if self.0 == Output::Revision {
            dependencies.push(DataRequest::new(DatasetId::StockConsensus, &FORECASTS));
        }
        let mut tags = vec!["HTZQ", "analyst", "daily", "neutralize", "size", "sector"];
        tags.push(if self.0 == Output::Revision {
            "forecast_revision"
        } else {
            "monthly_update"
        });
        if matches!(self.0, Output::RatingRoe | Output::Revision) {
            tags.extend(["fundamental", "pit"]);
        }
        let detail = match self.0 {
            Output::Coverage => "Month-end three-calendar-month distinct report/author/institution counts (uncovered stocks zero), separately regress on three-month valid daily means of ln(total_mv), adjusted RET20 and turnover_rate_f plus SW L1. Zscore the three residuals, require all, equally average; no second neutralization. No institutional holdings regressor.",
            Output::Revision => "Daily three-calendar-month revisions of annual Q4 NP in ten-thousand yuan; forecast year=prior calendar year through Apr30 inclusive, current year from May1. Average deduplicated institution-day predictions and subtract strictly previous same-year prediction within twelve calendar months. Innovative revision must also exceed/fall below SAME absolute-year consensus on last trading day STRICTLY BEFORE report. Ratio denominators count all distinct finite-NP reports for target year, including unpaired reports; numerator counts corresponding reports of upward institution-day revisions. Mean amounts use institution-day events. Regress four raw metrics on adjusted three-calendar-month return, zscore residuals, require all four, equally average; then SW L1/Barra SIZE neutralize daily.",
            _ => "Month-end improved rating. Institution plus individual author identity, split coauthors; average-tie Spearman between report rating and report-date PIT anomaly over three calendar years, >=3 complete samples and nonconstant ranks. Select highest 50% (ceil, identity breaks ties), even if correlations negative. Average last-three-calendar-month ratings from reports with at least one selected author, each distinct report only once. Anomalies: -ln(total_mv), -adjusted RET20, 1/pb (nonzero), parent NP TTM/average parent equity at q and q-4 (both positive). Rating scores buy/recommend=7, increase/outperform=5, neutral/hold=3, reduce=2, sell=1, explicit unrated=0; unknown/null skipped. SW L1/Barra SIZE neutralize; hold FINAL cross-section daily.",
        };
        FactorSpec {
            id: self.0.id().into(), aliases: vec![self.0.id().to_ascii_uppercase()], name: self.0.id().into(),
            asset_class: AssetClass::Stock, frequency: Frequency::Daily, version: "0.1.0".into(),
            tags: tags.into_iter().map(str::to_string).collect(),
            description: format!("{detail} Common: legal stock-basic universe, excludes BJ, report key stock/date/org/author/title deduplicated across forecast-year rows; last valid duplicate wins. Windows (cutoff,date]. Requested outputs only. Monthly final snapshots keyed by stock, no disk intermediates, no winsorization/missing fills/extra final zscore."),
            dependencies, intraday_raw_dependencies: vec![],
            lookback: Lookback { trading_days: if self.0.rating_index().is_some() { 850 } else { 130 } },
        }
    }
    fn requirements_for_context(&self, context: &FactorContext) -> Vec<DataRequest> {
        let anchors: Vec<_> = context
            .load_dates
            .windows(2)
            .filter(|p| p[0] / 100 != p[1] / 100)
            .map(|p| p[0])
            .chain(context.load_dates.last().copied())
            .collect();
        self.spec()
            .dependencies
            .into_iter()
            .map(|r| {
                let r = if r.dataset == DatasetId::StockBarraDaily {
                    r.with_explicit_dates(if self.0 == Output::Revision {
                        context.target_dates.clone()
                    } else {
                        anchors.clone()
                    })
                } else {
                    let dates = r.resolved_dates(context);
                    r.with_explicit_dates(dates)
                };
                if matches!(
                    r.dataset,
                    DatasetId::StockDailyPv
                        | DatasetId::StockAdjFactor
                        | DatasetId::StockDailyBasic
                        | DatasetId::StockConsensus
                        | DatasetId::StockBarraDaily
                ) {
                    r.with_panel_only()
                } else {
                    r
                }
            })
            .collect()
    }
    fn compute_provider_key(&self) -> String {
        "htzq_analyst".into()
    }
    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventStateDailyFast
    }
    fn initial_compute_state(&self, _: &[String]) -> Box<dyn Any + Send> {
        Box::<State>::default()
    }
    fn compute(&self, _: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        compute(&[self.0.id().into()], data, &mut State::default())?
            .pop()
            .ok_or_else(|| err("HTZQ output missing"))
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
                .ok_or_else(|| err("HTZQ analyst state mismatch"))?,
        )
    }
}

#[derive(Default)]
struct State {
    monthly: HashMap<Output, (i32, HashMap<String, Option<f64>>)>,
    last_target: Option<i32>,
}

fn finite(v: f64) -> Option<f64> {
    v.is_finite().then_some(v)
}
fn text(v: Option<&str>) -> Option<&str> {
    v.map(str::trim).filter(|s| !s.is_empty())
}
fn fy1(d: i32) -> i32 {
    d / 10000 - i32::from(d % 10000 <= 430)
}
fn annual_year(v: &str) -> Option<i32> {
    v.trim().strip_suffix("Q4")?.parse().ok()
}

fn rating(v: &str) -> Option<f64> {
    match v.trim().to_ascii_lowercase().as_str() {
        "买入" | "买进" | "推荐" | "强烈推荐" | "强推" | "buy" | "strong buy" => {
            Some(7.0)
        }
        "增持" | "谨慎增持" | "谨慎推荐" | "跑赢行业" | "优于大市" | "强于大市" | "outperform"
        | "overweight" => Some(5.0),
        "中性" | "持有" | "neutral" | "hold" | "equal weight" | "market-perform" => Some(3.0),
        "减持" | "低于大市" | "reduce" | "underweight" | "underperform" => Some(2.0),
        "卖出" | "sell" => Some(1.0),
        "无" | "无评级" | "未评级" | "unrated" | "not rated" => Some(0.0),
        _ => None,
    }
}

struct Report {
    stock: usize,
    date: i32,
    org: Option<usize>,
    authors: Vec<usize>,
    rating: Option<f64>,
}
struct Forecast {
    stock: usize,
    org: usize,
    year: i32,
    date: i32,
    value: f64,
    count: usize,
}
struct Reports {
    reports: Vec<Report>,
    forecasts: Vec<Forecast>,
    author_keys: Vec<String>,
}

fn report_index(
    reader: SourceRowsReader<'_>,
    codes: &[String],
    lower: i32,
    forecast_lower: i32,
    upper: i32,
    needs_rating: bool,
    needs_revision: bool,
) -> Result<Reports> {
    let stocks: HashMap<_, _> = codes
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.ends_with(".BJ"))
        .map(|(i, c)| (c.as_str(), i))
        .collect();
    let ts = reader.required_utf8("ts_code")?;
    let dates = reader.required_i32_date_cast("report_date")?;
    let orgs = reader.required_utf8("org_name")?;
    let authors = reader.required_utf8("author_name")?;
    let titles = reader.required_utf8("report_title")?;
    let ratings = needs_rating
        .then(|| reader.required_utf8("rating"))
        .transpose()?;
    let years = needs_revision
        .then(|| reader.required_utf8("quarter"))
        .transpose()?;
    let np = needs_revision
        .then(|| reader.required_f64_cast("np"))
        .transpose()?;
    let mut publications = BTreeMap::new();
    let mut predictions = BTreeMap::new();
    for row in 0..reader.len {
        let (Some(stock), Some(date)) = (
            ts[row].as_deref().and_then(|c| stocks.get(c).copied()),
            dates[row],
        ) else {
            continue;
        };
        if date <= lower || date > upper {
            continue;
        }
        let key = (
            stock,
            date,
            text(orgs[row].as_deref()),
            text(authors[row].as_deref()),
            text(titles[row].as_deref()),
        );
        let score = ratings
            .as_ref()
            .and_then(|r| r[row].as_deref())
            .and_then(rating);
        let entry = publications.entry(key).or_insert(None);
        if score.is_some() {
            *entry = score;
        }
        if date <= forecast_lower {
            continue;
        }
        if let (Some(org), Some(year), Some(value)) = (
            key.2,
            years
                .as_ref()
                .and_then(|y| y[row].as_deref())
                .and_then(annual_year),
            np.as_ref().and_then(|v| v[row]).and_then(finite),
        ) {
            predictions.insert((stock, org, year, date, key.3, key.4), value);
        }
    }
    let mut org_ids = BTreeMap::new();
    let mut author_ids = BTreeMap::new();
    let mut reports = Vec::with_capacity(publications.len());
    for ((stock, date, org, names, _), rating) in publications {
        let org_id = org.map(|org| {
            let next = org_ids.len();
            *org_ids.entry(org).or_insert(next)
        });
        let mut ids = Vec::new();
        if let (Some(org), Some(names)) = (org, names) {
            for name in names
                .split([',', '，', ';', '；', '、'])
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                let next = author_ids.len();
                ids.push(*author_ids.entry((org, name)).or_insert(next));
            }
        }
        ids.sort_unstable();
        ids.dedup();
        reports.push(Report {
            stock,
            date,
            org: org_id,
            authors: ids,
            rating,
        });
    }
    reports.sort_unstable_by_key(|r| (r.date, r.stock));
    let mut daily: BTreeMap<_, (f64, usize)> = BTreeMap::new();
    for ((stock, org, year, date, _, _), value) in predictions {
        let entry = daily.entry((stock, org_ids[org], year, date)).or_default();
        entry.0 += value;
        entry.1 += 1;
    }
    let forecasts = daily
        .into_iter()
        .filter_map(|((stock, org, year, date), (sum, count))| {
            finite(sum / count as f64).map(|value| Forecast {
                stock,
                org,
                year,
                date,
                value,
                count,
            })
        })
        .collect();
    let mut author_keys = vec![String::new(); author_ids.len()];
    for ((org, author), id) in author_ids {
        author_keys[id] = format!("{org}\0{author}");
    }
    Ok(Reports {
        reports,
        forecasts,
        author_keys,
    })
}

// Borrow shared panel columns; never expand each input onto the long universe grid.
struct Columns<'a> {
    panel: &'a DailyPanel,
    stocks: Vec<Option<usize>>,
    columns: HashMap<&'static str, PanelColumn>,
}
impl<'a> Columns<'a> {
    fn new(
        data: &'a DataPool,
        dataset: DatasetId,
        names: &[&'static str],
        codes: &[String],
    ) -> Result<Self> {
        let panel = data.daily_panel(dataset)?;
        let lookup: HashMap<_, _> = panel
            .instruments()
            .iter()
            .enumerate()
            .map(|(i, c)| (c.as_str(), i))
            .collect();
        Ok(Self {
            panel,
            stocks: codes
                .iter()
                .map(|c| lookup.get(c.as_str()).copied())
                .collect(),
            columns: names
                .iter()
                .map(|&name| Ok((name, panel.column(name)?)))
                .collect::<Result<_>>()?,
        })
    }
    fn get(&self, date: i32, stock: usize, name: &str) -> Option<f64> {
        let day = self.panel.dates().binary_search(&date).ok()?;
        let offset = day * self.panel.instruments().len() + self.stocks[stock]?;
        self.panel.is_present_offset(offset).then_some(())?;
        self.columns.get(name)?.values()[offset].and_then(finite)
    }
}
struct Market<'a> {
    basic: Option<Columns<'a>>,
    close: Option<Columns<'a>>,
    adj: Option<Columns<'a>>,
    size: Option<Columns<'a>>,
    consensus: Option<Columns<'a>>,
}
impl Market<'_> {
    fn price(&self, date: i32, stock: usize) -> Option<f64> {
        let v = self.close.as_ref()?.get(date, stock, "close")?
            * self.adj.as_ref()?.get(date, stock, "adj_factor")?;
        finite(v).filter(|v| *v > 0.0)
    }
    fn ret(&self, first: i32, last: i32, stock: usize) -> Option<f64> {
        finite(self.price(last, stock)? / self.price(first, stock)? - 1.0)
    }
    fn ret20(&self, calendar: &TradingCalendar, date: i32, stock: usize) -> Option<f64> {
        let first = calendar.warmup_start(date, 20);
        (calendar.open_date_after(first, 20) == Some(date)).then_some(())?;
        self.ret(first, date, stock)
    }
    fn consensus(
        &self,
        annual: &FinancialPitReader<'_>,
        calendar: &TradingCalendar,
        code: &str,
        stock: usize,
        report_date: i32,
        year: i32,
    ) -> Option<f64> {
        let date = calendar.last_open_on_or_before(add_days(report_date, -1))?;
        let base = analyst_fiscal_years(date, code, annual)[0];
        let col = usize::try_from(year - base)
            .ok()
            .filter(|i| *i < FORECASTS.len())?;
        self.consensus.as_ref()?.get(date, stock, FORECASTS[col])
    }
}

fn roe(
    income: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    code: &str,
    date: i32,
) -> Option<f64> {
    let q = income.latest_quarter_end_date(code, date)?;
    let a = balance
        .record_for_end_date(code, date, q)?
        .column(EQUITY)
        .filter(|v| *v > 0.0)?;
    let b = balance
        .record_for_end_date(code, date, q - 10000)?
        .column(EQUITY)
        .filter(|v| *v > 0.0)?;
    finite(income.ttm_sum_for_end_date(code, date, q, PROFIT)? / (0.5 * a + 0.5 * b))
}

fn equal_zscore(parts: &[Vec<Option<f64>>]) -> Vec<Option<f64>> {
    let standardized: Vec<_> = parts.iter().map(|v| cs_zscore(v)).collect();
    (0..parts[0].len())
        .map(|i| {
            let sum = standardized
                .iter()
                .try_fold(0.0, |sum, v| Some(sum + v[i]?))?;
            finite(sum / parts.len() as f64)
        })
        .collect()
}

fn coverage(
    reports: &Reports,
    market: &Market<'_>,
    calendar: &TradingCalendar,
    panel: &DailyPanel,
    groups: &[Option<String>],
    anchor: i32,
) -> Vec<Option<f64>> {
    let n = panel.instruments().len();
    let lower = add_months(anchor, -3);
    let mut y = vec![vec![Some(0.0); n]; 3];
    let mut authors = vec![HashSet::new(); n];
    let mut orgs = vec![HashSet::new(); n];
    let first = reports.reports.partition_point(|r| r.date <= lower);
    let end = reports.reports.partition_point(|r| r.date <= anchor);
    for r in &reports.reports[first..end] {
        y[0][r.stock] = y[0][r.stock].map(|v| v + 1.0);
        authors[r.stock].extend(r.authors.iter().copied());
        if let Some(org) = r.org {
            orgs[r.stock].insert(org);
        }
    }
    for i in 0..n {
        y[1][i] = Some(authors[i].len() as f64);
        y[2][i] = Some(orgs[i].len() as f64);
    }
    let days = calendar.open_dates_between(lower, anchor);
    let mut x = vec![vec![None; n]; 3];
    let basic = market.basic.as_ref().unwrap();
    for i in 0..n {
        let mut sums = [0.0; 3];
        let mut counts = [0; 3];
        for &d in days.iter().filter(|d| **d > lower) {
            let entries = [
                basic
                    .get(d, i, "total_mv")
                    .filter(|v| *v > 0.0)
                    .and_then(|v| finite(v.ln())),
                market.ret20(calendar, d, i),
                basic.get(d, i, "turnover_rate_f"),
            ];
            for (j, v) in entries.into_iter().enumerate() {
                if let Some(v) = v {
                    sums[j] += v;
                    counts[j] += 1;
                }
            }
        }
        for j in 0..3 {
            if counts[j] > 0 {
                x[j][i] = finite(sums[j] / counts[j] as f64);
            }
        }
        if groups[i].is_none() {
            for v in &mut y {
                v[i] = None;
            }
        }
    }
    let xs: Vec<_> = x.iter().map(Vec::as_slice).collect();
    equal_zscore(
        &y.iter()
            .map(|v| cs_neutralize_regression(v, &xs, Some(groups), None))
            .collect::<Vec<_>>(),
    )
}

fn ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<_> = (0..values.len()).collect();
    order.sort_unstable_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut result = vec![0.0; values.len()];
    let mut first = 0;
    while first < order.len() {
        let mut end = first + 1;
        while end < order.len() && values[order[end]] == values[order[first]] {
            end += 1;
        }
        let rank = (first + end - 1) as f64 * 0.5;
        for &i in &order[first..end] {
            result[i] = rank;
        }
        first = end;
    }
    result
}
fn spearman(values: &[(f64, f64)]) -> Option<f64> {
    if values.len() < 3 {
        return None;
    }
    let x = ranks(&values.iter().map(|v| v.0).collect::<Vec<_>>());
    let y = ranks(&values.iter().map(|v| v.1).collect::<Vec<_>>());
    let mean = (values.len() - 1) as f64 * 0.5;
    let (mut xx, mut yy, mut xy) = (0.0, 0.0, 0.0);
    for (x, y) in x.into_iter().zip(y) {
        let a = x - mean;
        let b = y - mean;
        xx += a * a;
        yy += b * b;
        xy += a * b;
    }
    if xx == 0.0 || yy == 0.0 {
        None
    } else {
        finite(xy / (xx * yy).sqrt())
    }
}
struct RatingSample {
    date: i32,
    rating: f64,
    signals: [Option<f64>; 4],
}
struct RatingHistory {
    samples: Vec<RatingSample>,
    by_author: Vec<Vec<usize>>,
}
fn rating_history(
    reports: &Reports,
    outputs: &[Output],
    market: &Market<'_>,
    calendar: &TradingCalendar,
    codes: &[String],
    income: Option<&FinancialPitReader<'_>>,
    balance: Option<&FinancialPitReader<'_>>,
) -> RatingHistory {
    let mut by_author: Vec<Vec<_>> = (0..reports.author_keys.len()).map(|_| vec![]).collect();
    let mut samples = Vec::new();
    let mut cache = HashMap::new();
    for report in &reports.reports {
        let Some(score) = report.rating else {
            continue;
        };
        if report.authors.is_empty() {
            continue;
        }
        let Some(date) = calendar.last_open_on_or_before(report.date) else {
            continue;
        };
        let signals = *cache.entry((report.stock, date)).or_insert_with(|| {
            let mut signals = [None; 4];
            for &output in outputs {
                let Some(j) = output.rating_index() else {
                    continue;
                };
                signals[j] = match j {
                    0 => market
                        .basic
                        .as_ref()
                        .and_then(|v| v.get(date, report.stock, "total_mv"))
                        .filter(|v| *v > 0.0)
                        .and_then(|v| finite(-v.ln())),
                    1 => market.ret20(calendar, date, report.stock).map(|v| -v),
                    2 => market
                        .basic
                        .as_ref()
                        .and_then(|v| v.get(date, report.stock, "pb"))
                        .filter(|v| *v != 0.0)
                        .and_then(|v| finite(1.0 / v)),
                    _ => roe(
                        income.unwrap(),
                        balance.unwrap(),
                        &codes[report.stock],
                        date,
                    ),
                };
            }
            signals
        });
        let index = samples.len();
        samples.push(RatingSample {
            date: report.date,
            rating: score,
            signals,
        });
        for &author in &report.authors {
            by_author[author].push(index);
        }
    }
    RatingHistory { samples, by_author }
}
fn improved_rating(
    reports: &Reports,
    history: &RatingHistory,
    signal: usize,
    anchor: i32,
    n: usize,
) -> Vec<Option<f64>> {
    let lower = add_months(anchor, -36);
    let mut ability = Vec::new();
    for (author, indices) in history.by_author.iter().enumerate() {
        let first = indices.partition_point(|&i| history.samples[i].date <= lower);
        let end = indices.partition_point(|&i| history.samples[i].date <= anchor);
        let pairs: Vec<_> = indices[first..end]
            .iter()
            .map(|&i| &history.samples[i])
            .filter_map(|v| Some((v.rating, v.signals[signal]?)))
            .collect();
        if let Some(correlation) = spearman(&pairs) {
            ability.push((author, correlation));
        }
    }
    ability.sort_unstable_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then_with(|| reports.author_keys[a.0].cmp(&reports.author_keys[b.0]))
    });
    let selected: HashSet<_> = ability
        .iter()
        .take(ability.len().div_ceil(2))
        .map(|v| v.0)
        .collect();
    let mut sums = vec![0.0; n];
    let mut counts = vec![0; n];
    let lower = add_months(anchor, -3);
    let first = reports.reports.partition_point(|r| r.date <= lower);
    let end = reports.reports.partition_point(|r| r.date <= anchor);
    for r in &reports.reports[first..end] {
        if r.authors.iter().any(|a| selected.contains(a)) {
            if let Some(v) = r.rating {
                sums[r.stock] += v;
                counts[r.stock] += 1;
            }
        }
    }
    (0..n)
        .map(|i| {
            if counts[i] > 0 {
                finite(sums[i] / counts[i] as f64)
            } else {
                None
            }
        })
        .collect()
}

struct Revision {
    date: i32,
    previous: i32,
    delta: f64,
    innovative: bool,
    count: usize,
}
struct Revisions {
    events: BTreeMap<(usize, i32), Vec<Revision>>,
    counts: BTreeMap<(usize, i32), Vec<(i32, usize)>>,
}
fn revision_index(
    reports: &Reports,
    market: &Market<'_>,
    annual: &FinancialPitReader<'_>,
    calendar: &TradingCalendar,
    codes: &[String],
    recent_lower: i32,
) -> Revisions {
    let mut events: BTreeMap<_, Vec<_>> = BTreeMap::new();
    let mut counts: BTreeMap<_, BTreeMap<i32, usize>> = BTreeMap::new();
    let mut previous = None;
    let mut consensus_cache = HashMap::new();
    for f in &reports.forecasts {
        *counts
            .entry((f.stock, f.year))
            .or_default()
            .entry(f.date)
            .or_default() += f.count;
        let key = (f.stock, f.org, f.year);
        if let Some((old_key, old_date, old_value)) = previous {
            if key == old_key && f.date > recent_lower {
                if let Some(delta) = finite(f.value - old_value) {
                    let consensus = *consensus_cache
                        .entry((f.stock, f.date, f.year))
                        .or_insert_with(|| {
                            market.consensus(
                                annual,
                                calendar,
                                &codes[f.stock],
                                f.stock,
                                f.date,
                                f.year,
                            )
                        });
                    let innovative = consensus.is_some_and(|c| {
                        (delta > 0.0 && f.value > c) || (delta < 0.0 && f.value < c)
                    });
                    events.entry((f.stock, f.year)).or_default().push(Revision {
                        date: f.date,
                        previous: old_date,
                        delta,
                        innovative,
                        count: f.count,
                    });
                }
            }
        }
        previous = Some((key, f.date, f.value));
    }
    for events in events.values_mut() {
        events.sort_unstable_by_key(|e| e.date);
    }
    let counts = counts
        .into_iter()
        .map(|(key, counts)| {
            let mut total = 0;
            (
                key,
                counts
                    .into_iter()
                    .map(|(d, n)| {
                        total += n;
                        (d, total)
                    })
                    .collect(),
            )
        })
        .collect();
    Revisions { events, counts }
}
fn count_at(counts: &[(i32, usize)], date: i32) -> usize {
    counts
        .partition_point(|v| v.0 <= date)
        .checked_sub(1)
        .map_or(0, |i| counts[i].1)
}
fn earnings_revision(
    index: &Revisions,
    market: &Market<'_>,
    calendar: &TradingCalendar,
    panel: &DailyPanel,
    date: i32,
) -> Vec<Option<f64>> {
    let n = panel.instruments().len();
    let lower = add_months(date, -3);
    let old_lower = add_months(date, -12);
    let year = fy1(date);
    let mut parts = vec![vec![None; n]; 4];
    let mut momentum = vec![None; n];
    let start = calendar.last_open_on_or_before(lower);
    for i in 0..n {
        if panel.instruments()[i].ends_with(".BJ") {
            continue;
        }
        momentum[i] = start.and_then(|d| market.ret(d, date, i));
        if let Some(counts) = index.counts.get(&(i, year)) {
            let total = count_at(counts, date) - count_at(counts, lower);
            if total > 0 {
                parts[2][i] = Some(0.0);
                parts[3][i] = Some(0.0);
            }
            if let Some(events) = index.events.get(&(i, year)) {
                let first = events.partition_point(|e| e.date <= lower);
                let end = events.partition_point(|e| e.date <= date);
                let (mut sum, mut count, mut high_sum, mut high_count, mut up, mut high_up) =
                    (0.0, 0, 0.0, 0, 0, 0);
                for e in events[first..end].iter().filter(|e| e.previous > old_lower) {
                    sum += e.delta;
                    count += 1;
                    if e.delta > 0.0 {
                        up += e.count;
                    }
                    if e.innovative {
                        high_sum += e.delta;
                        high_count += 1;
                        if e.delta > 0.0 {
                            high_up += e.count;
                        }
                    }
                }
                if count > 0 {
                    parts[0][i] = finite(sum / count as f64);
                }
                if high_count > 0 {
                    parts[1][i] = finite(high_sum / high_count as f64);
                }
                if total > 0 {
                    parts[2][i] = finite(up as f64 / total as f64);
                    parts[3][i] = finite(high_up as f64 / total as f64);
                }
            }
        }
    }
    equal_zscore(
        &parts
            .iter()
            .map(|v| cs_neutralize_regression(v, &[&momentum], None, None))
            .collect::<Vec<_>>(),
    )
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
    let targets: Vec<_> = panel
        .dates()
        .iter()
        .copied()
        .filter(|d| panel.is_target_date(*d))
        .collect();
    let mut result: Vec<_> = outputs
        .iter()
        .map(|o| FactorSeries {
            spec: HtzqAnalyst(*o).spec(),
            values: Vec::with_capacity(targets.len() * n),
        })
        .collect();
    let Some(&first) = targets.first() else {
        return Ok(result);
    };
    let rating_needed = outputs.iter().any(|o| o.rating_index().is_some());
    let revision_needed = outputs.contains(&Output::Revision);
    let coverage_needed = outputs.contains(&Output::Coverage);
    let anchor = calendar.month_end_on_or_before(first).unwrap_or(first);
    let lower = add_months(
        anchor,
        if rating_needed {
            -36
        } else if revision_needed {
            -12
        } else {
            -3
        },
    );
    let reports = report_index(
        data.analyst_report_reader()?,
        codes,
        lower,
        add_months(first, -12),
        *targets.last().unwrap(),
        rating_needed,
        revision_needed,
    )?;
    let needs_basic = coverage_needed
        || outputs
            .iter()
            .any(|o| matches!(o, Output::RatingSize | Output::RatingValue));
    let needs_price =
        coverage_needed || revision_needed || outputs.contains(&Output::RatingReversal);
    let basic_fields: Vec<_> = [
        (
            "total_mv",
            coverage_needed || outputs.contains(&Output::RatingSize),
        ),
        ("turnover_rate_f", coverage_needed),
        ("pb", outputs.contains(&Output::RatingValue)),
    ]
    .into_iter()
    .filter_map(|(f, needed)| needed.then_some(f))
    .collect();
    let market = Market {
        basic: needs_basic
            .then(|| Columns::new(data, DatasetId::StockDailyBasic, &basic_fields, codes))
            .transpose()?,
        close: needs_price
            .then(|| Columns::new(data, DatasetId::StockDailyPv, &["close"], codes))
            .transpose()?,
        adj: needs_price
            .then(|| Columns::new(data, DatasetId::StockAdjFactor, &["adj_factor"], codes))
            .transpose()?,
        size: outputs
            .iter()
            .any(|o| *o != Output::Coverage)
            .then(|| Columns::new(data, DatasetId::StockBarraDaily, &["SIZE"], codes))
            .transpose()?,
        consensus: revision_needed
            .then(|| Columns::new(data, DatasetId::StockConsensus, &FORECASTS, codes))
            .transpose()?,
    };
    let sectors = ClassificationMap::from_table(
        data.daily(DatasetId::StockSwClassification)?,
        ClassificationLevel::Sector,
    )?;
    let needs_roe = outputs.contains(&Output::RatingRoe);
    let income = needs_roe
        .then(|| {
            data.financial_reader(
                DatasetId::StockIncome,
                ReportTypePreference::income_single_quarter(),
            )
        })
        .transpose()?;
    let balance = needs_roe
        .then(|| {
            data.financial_reader(
                DatasetId::StockBalanceSheet,
                ReportTypePreference::balance_sheet_consolidated(),
            )
        })
        .transpose()?;
    let annual = revision_needed
        .then(|| {
            data.financial_reader(DatasetId::StockIncome, ReportTypePreference::consolidated())
        })
        .transpose()?;
    let revisions = if revision_needed {
        Some(revision_index(
            &reports,
            &market,
            annual.as_ref().unwrap(),
            calendar,
            codes,
            add_months(first, -3),
        ))
    } else {
        None
    };
    let mut history = None;
    for &date in &targets {
        if state.last_target.is_some_and(|last| date < last) {
            return Err(err("HTZQ analyst state cannot move backwards"));
        }
        let day = panel.dates().binary_search(&date).unwrap();
        let anchor = calendar.month_end_on_or_before(date);
        let mut daily = None;
        if revision_needed {
            let raw =
                earnings_revision(revisions.as_ref().unwrap(), &market, calendar, panel, date);
            let (groups, size) = exposures(panel, &market, &sectors, date);
            daily = Some(cs_neutralize_regression(
                &raw,
                &[&size],
                Some(&groups),
                None,
            ));
        }
        for (j, &output) in outputs.iter().enumerate() {
            if output != Output::Revision {
                if let Some(anchor) = anchor {
                    if state.monthly.get(&output).is_none_or(|v| v.0 != anchor) {
                        let (groups, size) = exposures(panel, &market, &sectors, anchor);
                        let values = if output == Output::Coverage {
                            coverage(&reports, &market, calendar, panel, &groups, anchor)
                        } else {
                            let history = history.get_or_insert_with(|| {
                                rating_history(
                                    &reports,
                                    &outputs,
                                    &market,
                                    calendar,
                                    codes,
                                    income.as_ref(),
                                    balance.as_ref(),
                                )
                            });
                            let mut raw = improved_rating(
                                &reports,
                                history,
                                output.rating_index().unwrap(),
                                anchor,
                                n,
                            );
                            for i in 0..n {
                                if groups[i].is_none() {
                                    raw[i] = None;
                                }
                            }
                            cs_neutralize_regression(&raw, &[&size], Some(&groups), None)
                        };
                        state.monthly.insert(
                            output,
                            (anchor, codes.iter().cloned().zip(values).collect()),
                        );
                    }
                }
            }
            for (i, code) in codes.iter().enumerate() {
                if !panel.is_present_offset(day * n + i) {
                    continue;
                }
                let value = if code.ends_with(".BJ") {
                    None
                } else if output == Output::Revision {
                    daily.as_ref().unwrap()[i]
                } else {
                    state
                        .monthly
                        .get(&output)
                        .filter(|v| Some(v.0) == anchor)
                        .and_then(|v| v.1.get(code).copied().flatten())
                };
                result[j].values.push(FactorValue {
                    key: FactorRowKey::Daily {
                        trade_date: date,
                        ts_code: code.clone(),
                    },
                    value,
                });
            }
        }
        state.last_target = Some(date);
    }
    Ok(result)
}
fn exposures(
    panel: &DailyPanel,
    market: &Market<'_>,
    sectors: &ClassificationMap,
    date: i32,
) -> (Vec<Option<String>>, Vec<Option<f64>>) {
    let mut groups = sectors.groups_for(date, panel.instruments());
    let day = panel.dates().binary_search(&date).ok();
    let n = panel.instruments().len();
    let mut size = vec![None; n];
    for (i, c) in panel.instruments().iter().enumerate() {
        if c.ends_with(".BJ") || day.is_none_or(|d| !panel.is_present_offset(d * n + i)) {
            groups[i] = None;
        } else {
            size[i] = market.size.as_ref().and_then(|s| s.get(date, i, "SIZE"));
        }
    }
    (groups, size)
}
